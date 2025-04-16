import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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
    batch_norm : bool, default=True
        Использовать ли batch normalization
    layer_norm : bool, default=False
        Использовать ли layer normalization
    initialization : str, default='he_normal'
        Метод инициализации весов ('he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform')
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
                 batch_norm=True,
                 layer_norm=False,
                 initialization='he_normal',
                 leaky_relu_negative_slope=0.1,
                 feature_dropout=0.0,
                 dynamic_emb_size=False,
                 min_emb_dim=2,
                 max_emb_dim=16):
        super().__init__()
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.initialization = initialization
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.feature_dropout = feature_dropout
        self.dynamic_emb_size = dynamic_emb_size
        self.min_emb_dim = min_emb_dim
        self.max_emb_dim = max_emb_dim

        # Эмбеддинги для категориальных признаков
        if self.dynamic_emb_size:
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

        # PReLU поддержка: отдельный слой для каждого скрытого слоя, если activation == 'prelu'
        self.prelu_layers = None
        if self.activation == 'prelu':
            self.prelu_layers = nn.ModuleList([nn.PReLU() for _ in hidden_dims])

        # Строим MLP
        layers = []
        prev_dim = self.post_embed_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            if layer_norm:
                layers.append(nn.LayerNorm(dim))
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
        out = self.input_dropout(x)
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