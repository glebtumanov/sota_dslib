import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def sparsemax(x, dim=-1):
    """Реализация функции Sparsemax."""
    original_shape = x.shape

    # Перемещаем целевое измерение в конец для удобства
    if dim != -1:
        x = x.transpose(dim, -1)

    # Сглаживаем все измерения кроме последнего
    x_flat = x.reshape(-1, x.size(-1))
    dim = -1

    # Сортируем и вычисляем кумулятивную сумму
    sorted_x, _ = torch.sort(x_flat, dim=dim, descending=True)
    cumsum = torch.cumsum(sorted_x, dim=dim)

    # Создаем индексы и находим k (макс. индекс, удовлетворяющий условию)
    D = x_flat.size(dim)
    range_indices = torch.arange(1, D + 1, dtype=x.dtype, device=x.device).unsqueeze(0).expand_as(x_flat)
    condition = 1 + range_indices * sorted_x > cumsum
    k = torch.sum(condition.to(dtype=torch.int32), dim=dim, keepdim=True)

    # Вычисляем порог tau и результат
    tau = (cumsum.gather(dim, k - 1) - 1) / k
    result = torch.clamp(x_flat - tau, min=0)

    # Возвращаем к исходной форме
    if dim != -1:
        result = result.reshape(original_shape).transpose(dim, -1)
    else:
        result = result.reshape(original_shape)

    return result


class GhostBatchNorm(nn.Module):
    """Реализация Ghost Batch Normalization для TabNet."""
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, x):
        if self.training and x.size(0) > self.virtual_batch_size:
            # Разбиваем батч на виртуальные подбатчи
            chunks = x.chunk(max(1, x.size(0) // self.virtual_batch_size), 0)
            # Применяем BatchNorm к каждому подбатчу
            res = torch.cat([self.bn(x_) for x_ in chunks], dim=0)
        else:
            # В режиме оценки или для маленьких батчей используем обычную BatchNorm
            res = self.bn(x)
        return res


class GLUBlock(nn.Module):
    """Блок GLU (Gated Linear Unit) с Ghost Batch нормализацией и активацией."""
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.9):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.norm = GhostBatchNorm(output_dim * 2, virtual_batch_size, momentum)

    def forward(self, x):
        x = self.fc(x)

        # Приводим размерность для BatchNorm1d если необходимо
        shape = x.shape
        if len(shape) > 2:
            x = self.norm(x.view(-1, shape[-1])).view(shape)
        else:
            x = self.norm(x)

        return F.glu(x, dim=-1)


class FeatureTransformer(nn.Module):
    """Трансформер признаков с GLU блоками."""
    def __init__(self, input_dim, hidden_dim, n_glu_layers, dropout=0.1,
                 virtual_batch_size=128, momentum=0.9):
        super().__init__()
        layers = []
        for i in range(n_glu_layers):
            layers.append(GLUBlock(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum
            ))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return x


class AttentiveTransformer(nn.Module):
    """Модуль для вычисления маски внимания."""
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.9):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size, momentum)
        self.attentive = nn.Linear(output_dim, output_dim)

    def forward(self, x, prior_scales=None):
        shape = x.shape
        if len(shape) > 2:
            x = self.bn(self.fc(x).view(-1, shape[-1])).view(shape)
        else:
            x = self.bn(self.fc(x))

        x = F.relu(x)

        return sparsemax(self.attentive(x), dim=-1)


class TabNet(nn.Module):
    """Архитектура TabNet нейронной сети."""
    def __init__(self,
                 input_dim,
                 cat_idxs,
                 cat_dims,
                 cat_emb_dim=4,
                 n_steps=5,
                 n_d=64,  # Заменяем hidden_dim на n_d
                 n_a=64,  # Добавляем параметр n_a
                 decision_dim=64,
                 n_glu_layers=2,
                 dropout=0.1,
                 gamma=1.5,
                 lambda_sparse=0.0001,
                 virtual_batch_size=128,
                 momentum=0.9,
                 output_dim=1,
                 dynamic_emb_size=False,  # Параметр для динамического размера эмбеддингов
                 min_emb_dim=2,  # Минимальный размер эмбеддинга
                 max_emb_dim=16):  # Максимальный размер эмбеддинга
        super().__init__()

        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.n_d = n_d  # Сохраняем размерность для решающего слоя
        self.n_a = n_a  # Сохраняем размерность для блока внимания
        self.dynamic_emb_size = dynamic_emb_size
        self.min_emb_dim = min_emb_dim
        self.max_emb_dim = max_emb_dim

        # Определяем размеры эмбеддингов - динамически или фиксированным размером
        if self.dynamic_emb_size and len(cat_dims) > 0:
            # Для каждой категориальной переменной вычисляем размер эмбеддинга
            # исходя из формулы log2(размер_категории) + 1, но не меньше min_emb_dim и не больше max_emb_dim
            self.emb_dims = [min(max(int(math.ceil(np.log2(dim))) + 1, self.min_emb_dim), self.max_emb_dim)
                             for dim in cat_dims]
        else:
            # Используем фиксированный размер для всех категориальных переменных
            self.emb_dims = [cat_emb_dim] * len(cat_dims)

        # Эмбеддинги для категориальных признаков с динамическими размерами
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, self.emb_dims)]
        )

        # Размерность признаков после эмбеддингов (с учетом динамических размеров)
        self.post_embed_dim = input_dim - len(cat_idxs) + sum(self.emb_dims)

        self.feature_transformers = nn.ModuleList()
        self.attentive_transformers = nn.ModuleList()

        for step in range(n_steps):
            self.feature_transformers.append(
                FeatureTransformer(
                    self.post_embed_dim,
                    n_d + n_a,  # Общая размерность выхода n_d + n_a
                    n_glu_layers,
                    dropout,
                    virtual_batch_size,
                    momentum
                )
            )

            if step < n_steps - 1:
                self.attentive_transformers.append(
                    AttentiveTransformer(
                        n_a,  # Используем n_a для AttentiveTransformer
                        self.post_embed_dim,
                        virtual_batch_size,
                        momentum
                    )
                )

        self.decision_layer = nn.Linear(n_d, decision_dim)
        self.final_layer = nn.Linear(decision_dim, output_dim)

    def forward(self, x, return_masks=False):
        # Применяем эмбеддинги для категориальных признаков
        if len(self.cat_idxs) > 0:
            x_num = []
            x_cat = []
            cat_i = 0

            # Разделяем числовые и категориальные признаки
            for i in range(self.input_dim):
                if i in self.cat_idxs:
                    # Сохраняем категориальные признаки для эмбеддинга
                    x_cat.append(x[:, i].long())
                    cat_i += 1
                else:
                    # Сохраняем числовые признаки как есть
                    x_num.append(x[:, i].unsqueeze(1))

            # Применяем эмбеддинги к категориальным признакам
            embedded_cats = []
            for i, cat_values in enumerate(x_cat):
                # Добавляем безопасную индексацию - ограничиваем значения диапазоном эмбеддинга
                safe_indices = torch.clamp(cat_values, 0, self.cat_dims[i] - 1)
                embedded_cats.append(self.embeddings[i](safe_indices))

            # Объединяем все признаки
            if embedded_cats:
                x = torch.cat(x_num + embedded_cats, dim=1)
            else:
                x = torch.cat(x_num, dim=1)

        # Применяем последовательные шаги TabNet
        outputs = torch.zeros(x.size(0), self.final_layer.out_features).to(x.device)
        prior_scales = torch.ones_like(x).to(x.device)

        # Сохраняем маски для регуляризации разреженности
        masks = []

        for step in range(self.n_steps):
            # Применяем маску к входным данным
            masked_x = x * prior_scales

            # Преобразуем через feature transformer
            x_transformed = self.feature_transformers[step](masked_x)

            # Разделяем выход на d[i] и a[i]
            d_i = x_transformed[:, :self.n_d]  # Первые n_d элементов для decision
            a_i = x_transformed[:, self.n_d:]  # Оставшиеся n_a элементов для attention

            # Считаем выход текущего шага используя d_i
            step_output = F.relu(self.decision_layer(d_i))
            outputs += self.final_layer(step_output) / self.n_steps

            # Обновляем маску для следующего шага используя a_i
            if step < self.n_steps - 1:
                mask = self.attentive_transformers[step](a_i)
                masks.append(mask)

                # Обеспечиваем правильную размерность маски
                if mask.size(1) != prior_scales.size(1):
                    # Если размеры не совпадают, повторяем значения маски
                    mask = mask.repeat(1, prior_scales.size(1) // mask.size(1))
                    if mask.size(1) < prior_scales.size(1):
                        padding = torch.zeros(mask.size(0), prior_scales.size(1) - mask.size(1), device=mask.device)
                        mask = torch.cat([mask, padding], dim=1)
                    elif mask.size(1) > prior_scales.size(1):
                        mask = mask[:, :prior_scales.size(1)]

                # Обновляем prior_scales согласно формуле из статьи
                prior_scales = prior_scales * (self.gamma - mask)

        if return_masks:
            return outputs, masks
        return outputs

    def calculate_sparse_loss(self, masks):
        """Вычисляет регуляризацию разреженности для масок."""
        if self.lambda_sparse == 0 or not masks:
            return 0.0

        batch_size = masks[0].size(0)
        total_loss = 0.0

        for mask in masks:
            entropy = -torch.sum(mask * torch.log(mask + 1e-10)) / (batch_size * self.n_steps)
            total_loss += entropy

        return self.lambda_sparse * total_loss


def softmax(x, axis=None):
    """Вычисление softmax по указанной оси"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)