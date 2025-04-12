import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(x, dim=-1):
    """Реализация функции Sparsemax.

    Аргументы:
        x: входной тензор
        dim: размерность, по которой применяется sparsemax

    Возвращает:
        тензор с примененной функцией sparsemax
    """
    # Форма исходного тензора
    original_shape = x.shape

    # Перемещаем целевое измерение в конец для удобства
    if dim != -1:
        x = x.transpose(dim, -1)

    # Сглаживаем все измерения кроме последнего
    x_flat = x.reshape(-1, x.size(-1))
    dim = -1  # Теперь обрабатываем только последнее измерение

    # Сортируем входные данные в убывающем порядке
    sorted_x, _ = torch.sort(x_flat, dim=dim, descending=True)

    # Кумулятивная сумма
    cumsum = torch.cumsum(sorted_x, dim=dim)

    # Находим количество элементов
    D = x_flat.size(dim)

    # Создаем индексы
    range_indices = torch.arange(1, D + 1, dtype=x.dtype, device=x.device)
    range_indices = range_indices.unsqueeze(0).expand_as(x_flat)

    # Вычисляем условие
    condition = 1 + range_indices * sorted_x > cumsum

    # Находим k (макс. индекс, удовлетворяющий условию)
    k = torch.sum(condition.to(dtype=torch.int32), dim=dim, keepdim=True)

    # Выбираем порог tau
    tau = (cumsum.gather(dim, k - 1) - 1) / k

    # Вычисляем результат
    result = torch.clamp(x_flat - tau, min=0)

    # Возвращаем к исходной форме
    if dim != -1:
        result = result.reshape(original_shape).transpose(dim, -1)
    else:
        result = result.reshape(original_shape)

    return result


class GLUBlock(nn.Module):
    """Блок GLU (Gated Linear Unit) с нормализацией и активацией."""
    def __init__(self, input_dim, output_dim, norm_type='batch'):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.norm = nn.BatchNorm1d(output_dim * 2) if norm_type == 'batch' else nn.LayerNorm(output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return F.glu(x, dim=-1)


class FeatureTransformer(nn.Module):
    """Трансформер признаков с GLU блоками."""
    def __init__(self, input_dim, hidden_dim, n_glu_layers, norm_type='batch', dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_glu_layers):
            layers.append(GLUBlock(input_dim if i == 0 else hidden_dim,
                                   hidden_dim,
                                   norm_type=norm_type))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return x


class AttentiveTransformer(nn.Module):
    """Модуль для вычисления маски внимания."""
    def __init__(self, input_dim, output_dim, norm_type='batch'):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim) if norm_type == 'batch' else nn.LayerNorm(output_dim)
        self.attentive = nn.Linear(output_dim, output_dim)

    def forward(self, x, prior_scales):
        x = F.relu(self.bn(self.fc(x)))
        return sparsemax(self.attentive(x), dim=-1)


class TabNet(nn.Module):
    """Полная архитектура TabNet с поддержкой категориальных признаков."""
    def __init__(self,
                 input_dim,
                 cat_idxs,
                 cat_dims,
                 cat_emb_dim=4,  # Увеличиваем размерность эмбеддингов
                 n_steps=5,  # Увеличиваем количество шагов
                 hidden_dim=128,  # Увеличиваем размерность скрытого слоя
                 decision_dim=64,  # Увеличиваем размерность решающего слоя
                 n_glu_layers=2,
                 norm_type='batch',
                 dropout=0.1,
                 gamma=1.5,
                 output_dim=1):
        """
        :param input_dim: Количество признаков после эмбеддинга
        :param cat_idxs: Индексы категориальных признаков
        :param cat_dims: Размерность категорий
        :param cat_emb_dim: Размер эмбеддингов категориальных признаков
        """
        super().__init__()

        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim

        # Эмбеддинги для категориальных признаков
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_dim, cat_emb_dim) for cat_dim in cat_dims]
        )

        # Размерность признаков после эмбеддингов
        self.post_embed_dim = input_dim - len(cat_idxs) + len(cat_idxs) * cat_emb_dim

        self.feature_transformers = nn.ModuleList()
        self.attentive_transformers = nn.ModuleList()

        for step in range(n_steps):
            self.feature_transformers.append(
                FeatureTransformer(
                    self.post_embed_dim,
                    hidden_dim,
                    n_glu_layers,
                    norm_type,
                    dropout
                )
            )

            if step < n_steps - 1:
                self.attentive_transformers.append(
                    AttentiveTransformer(
                        hidden_dim,
                        self.post_embed_dim,
                        norm_type
                    )
                )

        self.decision_layer = nn.Linear(hidden_dim, decision_dim)
        self.final_layer = nn.Linear(decision_dim, output_dim)
        self.gamma = gamma
        self.n_steps = n_steps

    def forward(self, x):
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
                embedded_cats.append(self.embeddings[i](cat_values))

            # Объединяем все признаки
            if embedded_cats:
                x = torch.cat(x_num + embedded_cats, dim=1)
            else:
                x = torch.cat(x_num, dim=1)

        # Применяем последовательные шаги TabNet
        outputs = torch.zeros(x.size(0), 1).to(x.device)
        prior_scales = torch.ones_like(x).to(x.device)

        for step in range(self.n_steps):
            # Применяем маску к входным данным
            masked_x = x * prior_scales

            # Преобразуем через feature transformer
            x_transformed = self.feature_transformers[step](masked_x)

            # Считаем выход текущего шага
            step_output = F.relu(self.decision_layer(x_transformed))
            outputs += self.final_layer(step_output) / self.n_steps  # Нормализуем по количеству шагов

            # Обновляем маску для следующего шага
            if step < self.n_steps - 1:
                mask = self.attentive_transformers[step](x_transformed, prior_scales)
                # Обеспечиваем правильную размерность маски
                if mask.size(1) != prior_scales.size(1):
                    # Если размеры не совпадают, повторяем значения маски
                    mask = mask.repeat(1, prior_scales.size(1) // mask.size(1))
                    # Обрезаем или дополняем до нужной размерности
                    if mask.size(1) < prior_scales.size(1):
                        padding = torch.zeros(mask.size(0), prior_scales.size(1) - mask.size(1), device=mask.device)
                        mask = torch.cat([mask, padding], dim=1)
                    elif mask.size(1) > prior_scales.size(1):
                        mask = mask[:, :prior_scales.size(1)]

                # Обновляем масштабы для следующего шага
                prior_scales = prior_scales * (self.gamma - mask)

        # Не сжимаем последнюю размерность, чтобы соответствовать размерности целевых значений
        return outputs
