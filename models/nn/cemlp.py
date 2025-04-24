import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GhostBatchNorm(nn.Module):
    """Реализация Ghost Batch Normalization.

    Ghost Batch Normalization разбивает большой батч на виртуальные мини-батчи,
    что позволяет получить более стабильную статистику при нормализации.

    Параметры:
    -----------
    input_dim : int
        Размерность входных данных
    virtual_batch_size : int, default=128
        Размер виртуального батча
    momentum : float, default=0.9
        Параметр momentum для BatchNorm
    """
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) должно быть кратно n_heads ({n_heads})"
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(p_drop)
        self.n_heads = n_heads

    def _split(self, x):
        # (B, N, D) → (B, H, N, d_head)
        B, N, D = x.shape
        return x.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x):
        # (B, H, N, d_head) → (B, N, D)
        B, H, N, d = x.shape
        return x.transpose(1, 2).reshape(B, N, H * d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self._split, (q, k, v))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.drop(attn.softmax(-1))
        out = self._merge(attn @ v)
        return self.proj(out) + x            # residual

class NumericEmbedding(nn.Module):
    """Каждый числовой скаляр x → вектор x * w_i + b_i (w_i, b_i — обучаемые)."""
    def __init__(self, num_numeric: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_numeric, d_model))
        self.bias   = nn.Parameter(torch.zeros(num_numeric, d_model))

    def forward(self, x_num):                # x_num: (B, F_num)
        # (B, F_num, D)
        return x_num.unsqueeze(-1) * self.weight + self.bias


class CatEmbMLP(nn.Module):
    """Архитектура MLP с эмбеддингами для категориальных признаков.

    Эта нейронная сеть объединяет эмбеддинги категориальных признаков с числовыми
    и обрабатывает их через многослойный перцептрон с настраиваемой архитектурой.

    Параметры:
    -----------
    input_dim : int
        Общее количество входных признаков (и категориальных, и числовых)
    cat_idxs : list
        Индексы категориальных признаков в тензоре данных
    cat_dims : list
        Размерности (количество уникальных значений) каждой категориальной переменной
    cat_emb_dim : int, default=4
        Размерность эмбеддингов для категориальных признаков
    hidden_dims : list, default=[64, 32]
        Список размерностей скрытых слоев MLP
    output_dim : int, default=1
        Размерность выходного слоя
    activation : str, default='relu'
        Функция активации ('relu', 'leaky_relu', 'selu', 'elu', 'gelu', 'swish')
    dropout : float, default=0.1
        Вероятность дропаута для регуляризации
    attn_dropout : float, default=0.1
        Вероятность дропаута для attention
    normalization : str, default='batch'
        Тип нормализации ('batch', 'layer', 'ghost_batch')
    virtual_batch_size : int, default=128
        Размер виртуального батча для GhostBatchNorm
    momentum : float, default=0.9
        Параметр momentum для BatchNorm/GhostBatchNorm
    initialization : str, default='he_normal'
        Метод инициализации весов ('he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform',
        'uniform', 'normal', 'constant', 'ones', 'zeros')
    constant_value : float, default=0.001
        Значение для constant инициализации
    leaky_relu_negative_slope : float, default=0.1
        Отрицательный наклон для leaky_relu
    feature_dropout : float, default=0.0
        Dropout для входных признаков (feature-wise dropout)
    dynamic_emb_size : bool, default=False
        Использовать ли динамический размер эмбеддинга для каждого категориального признака
    min_emb_dim : int, default=2
        Минимальный размер эмбеддинга при dynamic_emb_size
    max_emb_dim : int, default=16
        Максимальный размер эмбеддинга при dynamic_emb_size
    use_self_attention : bool, default=False
        Использовать ли self-attention механизм для признаков
    num_attention_heads : int, default=4
        Количество голов внимания при use_self_attention=True
    d_model : int, default=80
        Размерность эмбеддингов для self-attention
    """
    def __init__(self,
                 input_dim,
                 cat_idxs,
                 cat_dims,
                 cat_emb_dim=4,
                 hidden_dims=[64, 32],
                 output_dim=1,
                 activation='relu',
                 dropout=0.1,
                 attn_dropout=0.1,
                 normalization='batch',
                 virtual_batch_size=128,
                 momentum=0.9,
                 initialization='he_normal',
                 constant_value=0.001,
                 leaky_relu_negative_slope=0.1,
                 feature_dropout=0.0,
                 dynamic_emb_size=False,
                 min_emb_dim=2,
                 max_emb_dim=16,
                 use_self_attention=False,
                 d_model=80,
                 num_attention_heads=4):
        super().__init__()
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.initialization = initialization
        self.constant_value = constant_value
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.feature_dropout = feature_dropout
        self.dynamic_emb_size = dynamic_emb_size
        self.min_emb_dim = min_emb_dim
        self.max_emb_dim = max_emb_dim
        self.use_self_attention = use_self_attention
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.attn_dropout = attn_dropout

        # Проверяем корректность параметра нормализации
        if normalization not in ['batch', 'layer', 'ghost_batch']:
            raise ValueError(f"Неподдерживаемый тип нормализации: {normalization}. "
                            "Допустимые значения: 'batch', 'layer', 'ghost_batch'")

        # Эмбеддинги для категориальных признаков
        if self.use_self_attention:
            # Для self‑attention представляем каждый признак в d_model‑мерном пространстве
            emb_dims = [self.d_model for _ in cat_dims]
            numeric_emb_dim = self.d_model
        elif self.dynamic_emb_size:
            emb_dims = [min(max(int(math.ceil(np.log2(dim))) + 1, self.min_emb_dim), self.max_emb_dim) for dim in cat_dims]
            numeric_emb_dim = self.cat_emb_dim
        else:
            emb_dims = [self.cat_emb_dim for _ in cat_dims]
            numeric_emb_dim = self.cat_emb_dim

        self.cat_emb = nn.ModuleList(
            [nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, emb_dims)]
        )
        # Число числовых признаков для последующих вычислений
        num_numeric = input_dim - len(cat_idxs)

        # Если используется self‑attention, размер вектора признака после него равен d_model,
        # иначе — размеру эмбеддинга числового/категориального признака.
        if self.use_self_attention:
            # (кол‑во признаков) * d_model
            self.post_embed_dim = (input_dim) * self.d_model
        else:
            # Числовые признаки → num_numeric * cat_emb_dim
            # Категориальные признаки → сумма их индивидуальных emb_dims
            self.post_embed_dim = num_numeric * self.cat_emb_dim + sum(emb_dims)

        self.num_emb = NumericEmbedding(num_numeric, numeric_emb_dim)

        # Feature-wise dropout
        self.input_dropout = nn.Dropout(self.feature_dropout) if self.feature_dropout > 0 else nn.Identity()

        # Self-attention для признаков
        if self.use_self_attention:
            self.self_attention = MultiHeadSelfAttention(
                d_model=self.d_model,
                n_heads=self.num_attention_heads,
                p_drop=self.attn_dropout
            )

        # PReLU поддержка: отдельный слой для каждого скрытого слоя, если activation == 'prelu'
        self.prelu_layers = None
        if self.activation == 'prelu':
            self.prelu_layers = nn.ModuleList([nn.PReLU() for _ in hidden_dims])

        # Строим MLP
        layers = []
        prev_dim = self.post_embed_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))

            # Добавляем соответствующий слой нормализации
            if self.normalization == 'batch':
                layers.append(nn.BatchNorm1d(dim, momentum=self.momentum))
            elif self.normalization == 'layer':
                layers.append(nn.LayerNorm(dim))
            elif self.normalization == 'ghost_batch':
                layers.append(GhostBatchNorm(dim, virtual_batch_size=self.virtual_batch_size, momentum=self.momentum))

            # PReLU как отдельный слой
            if self.activation == 'prelu':
                layers.append(self.prelu_layers[i])
            else:
                layers.append(self._get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_activation(self):
        """Возвращает функцию активации по названию."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(self.leaky_relu_negative_slope)
        elif self.activation == 'selu':
            return nn.SELU()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'swish':
            return nn.SiLU()  # Swish / SiLU
        elif self.activation == 'prelu':
            # PReLU добавляется как отдельный слой
            return nn.Identity()
        else:
            raise ValueError(f"Неизвестная функция активации: {self.activation}")

    def _initialize_weights(self):
        """Инициализирует веса сети."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initialization == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif self.initialization == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                elif self.initialization == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.initialization == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.initialization == 'uniform':
                    nn.init.uniform_(m.weight)
                elif self.initialization == 'normal':
                    nn.init.normal_(m.weight)
                elif self.initialization == 'constant':
                    nn.init.constant_(m.weight, self.constant_value)
                elif self.initialization == 'ones':
                    nn.init.ones_(m.weight)
                elif self.initialization == 'zeros':
                    nn.init.zeros_(m.weight)
                else:
                    raise ValueError(f"Неизвестный способ инициализации: {self.initialization}")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Применяем эмбеддинги для категориальных признаков
        if len(self.cat_idxs) > 0:
            x_num = []
            x_cat = []
            for i in range(self.input_dim):
                if i in self.cat_idxs:
                    x_cat.append(x[:, i].long())
                else:
                    x_num.append(x[:, i].unsqueeze(1))
        else:
            x_num = [x[:, i].unsqueeze(1) for i in range(self.input_dim)]
            x_cat = []

        # Объединяем числовые признаки
        x_num = torch.cat(x_num, dim=1) if x_num else torch.zeros(x.size(0), 0)

        # Преобразуем числовые и категориальные признаки
        num_vecs = self.num_emb(x_num)

        # Если есть категориальные признаки, обрабатываем их
        if x_cat:
            # Используем stack вместо cat для объединения одномерных тензоров
            x_cat_stacked = torch.stack(x_cat, dim=1)
            cat_vecs = torch.stack(
                [emb(x_cat_stacked[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1
            )
        else:
            filler_dim = self.d_model if self.use_self_attention else self.cat_emb_dim
            cat_vecs = torch.zeros(x.size(0), 0, filler_dim, device=x.device)

        # Применяем feature-wise dropout к векторам признаков
        num_vecs = self.input_dropout(num_vecs)
        cat_vecs = self.input_dropout(cat_vecs)

        # Объединяем все признаки
        feats = torch.cat([num_vecs, cat_vecs], dim=1) if cat_vecs.numel() else num_vecs

        # Применяем self-attention, если включено
        if self.use_self_attention:
            feats = self.self_attention(feats)

        # Готовим входные данные для MLP
        out = feats.flatten(1)

        # Передаем через слои MLP
        out = self.layers(out)

        return out


def softmax(x, axis=None):
    """Вычисление softmax по указанной оси"""
    # Вычитаем максимум для численной стабильности
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    # Нормализуем
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)