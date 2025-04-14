import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
                 initialization='he_normal'):
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

        # Эмбеддинги для категориальных признаков
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_dim, cat_emb_dim) for cat_dim in cat_dims]
        )

        # Размерность признаков после эмбеддингов
        self.post_embed_dim = input_dim - len(cat_idxs) + len(cat_idxs) * cat_emb_dim

        # Строим MLP
        layers = []
        prev_dim = self.post_embed_dim

        for dim in hidden_dims:
            # Линейный слой
            layers.append(nn.Linear(prev_dim, dim))

            # Нормализация
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            if layer_norm:
                layers.append(nn.LayerNorm(dim))

            # Активация
            layers.append(self._get_activation())

            # Дропаут
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        # Выходной слой
        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.Sequential(*layers)

        # Инициализация весов
        self._initialize_weights()

    def _get_activation(self):
        """Возвращает функцию активации по названию."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif self.activation == 'selu':
            return nn.SELU()
        elif self.activation == 'elu':
            return nn.ELU()
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'swish':
            return nn.SiLU()  # Swish / SiLU
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

        # Проходим через MLP
        x = self.layers(x)

        return x


def softmax(x, axis=None):
    """Вычисление softmax по указанной оси"""
    # Вычитаем максимум для численной стабильности
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    # Нормализуем
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)