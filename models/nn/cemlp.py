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
    """Модуль многоголового self-attention для табличных данных.

    Параметры:
    -----------
    embed_dim : int
        Размерность входных данных/эмбеддингов
    num_heads : int
        Количество голов внимания
    dropout : float, default=0.0
        Вероятность дропаута для attention весов
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim {embed_dim} должно быть кратно num_heads {num_heads}"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim] = [bs, feature_dim, emb_dim]
        batch_size, seq_len, _ = x.shape

        # Проекции с разделением на головы
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Масштабируем скалярное произведение
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Получаем attention weights с помощью softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Применяем attention weights к value
        context = torch.matmul(attention_weights, v)

        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Финальная проекция
        output = self.out_proj(context)

        return output


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

        # Проверяем корректность параметра нормализации
        if normalization not in ['batch', 'layer', 'ghost_batch']:
            raise ValueError(f"Неподдерживаемый тип нормализации: {normalization}. "
                            "Допустимые значения: 'batch', 'layer', 'ghost_batch'")

        # Эмбеддинги для категориальных признаков
        # Если используется self-attention, устанавливаем размер эмбеддинга = 1
        if self.use_self_attention:
            emb_dims = [1 for _ in cat_dims]
        elif self.dynamic_emb_size:
            emb_dims = [min(max(int(math.ceil(np.log2(dim))) + 1, self.min_emb_dim), self.max_emb_dim) for dim in cat_dims]
        else:
            emb_dims = [self.cat_emb_dim for _ in cat_dims]

        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in zip(cat_dims, emb_dims)]
        )
        # Размерность признаков после эмбеддингов
        self.post_embed_dim = input_dim - len(cat_idxs) + sum(emb_dims)

        # Feature-wise dropout
        self.input_dropout = nn.Dropout(self.feature_dropout) if self.feature_dropout > 0 else nn.Identity()

        # Self-attention для признаков
        if self.use_self_attention:
            # Для self-attention, где каждый признак - элемент последовательности
            # Размерность эмбеддинга для внимания равна num_attention_heads
            self.self_attention = MultiHeadSelfAttention(
                embed_dim=self.num_attention_heads,
                num_heads=self.num_attention_heads,
                dropout=self.dropout
            )
            # Параметр для линейной проекции в методе forward
            self.attention_projection = nn.Parameter(
                torch.ones(self.num_attention_heads, 1)
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
            cat_i = 0
            for i in range(self.input_dim):
                if i in self.cat_idxs:
                    x_cat.append(x[:, i].long())
                    cat_i += 1
                else:
                    x_num.append(x[:, i].unsqueeze(1))
            embedded_cats = []
            for i, cat_values in enumerate(x_cat):
                embedded_cats.append(self.embeddings[i](cat_values))
            if embedded_cats:
                x = torch.cat(x_num + embedded_cats, dim=1)
            else:
                x = torch.cat(x_num, dim=1)

        # Применяем feature-wise dropout
        out = self.input_dropout(x)

        # Применяем self-attention, если включено
        if self.use_self_attention:
            # Преобразуем входные данные, чтобы каждый признак был элементом последовательности
            # [batch_size, feature_dim] -> [batch_size, feature_dim, 1]
            batch_size = out.size(0)
            feature_dim = out.size(1)

            # Добавляем размерность эмбеддинга для каждого признака
            attention_input = out.unsqueeze(2)  # [batch_size, feature_dim, 1]

            # Проецируем 1 -> embed_dim, кратное num_attention_heads, используя обучаемую проекцию
            attention_input = torch.nn.functional.linear(
                attention_input,
                self.attention_projection
            )  # [batch_size, feature_dim, num_attention_heads]

            # Применяем self-attention к последовательности признаков
            attention_output = self.self_attention(attention_input)  # [batch_size, feature_dim, num_attention_heads]

            # Обратная проекция в одномерное пространство
            out = attention_output.mean(dim=2)  # [batch_size, feature_dim]

        # Передаем через слои MLP
        for layer in self.layers:
            out = layer(out)

        return out


def softmax(x, axis=None):
    """Вычисление softmax по указанной оси"""
    # Вычитаем максимум для численной стабильности
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    # Нормализуем
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)