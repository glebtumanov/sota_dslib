import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, KBinsDiscretizer


class CatEmbDataset(Dataset):
    """Датасет для моделей нейронных сетей с категориальными эмбеддингами.

    CatEmbDataset подготавливает табличные данные для моделей нейронных сетей,
    специализирующихся на обработке смешанных типов данных (числовые и категориальные
    признаки). Он автоматически обрабатывает следующие аспекты:

    1. Индексирование категориальных признаков для использования в эмбеддингах
    2. Масштабирование числовых признаков различными методами
    3. Обработка пропущенных значений и выбросов
    4. Преобразование всех данных в тензоры PyTorch

    Параметры:
    -----------
    df : pandas.DataFrame
        DataFrame с исходными данными. Должен содержать все необходимые признаки.
        Не должен содержать строки с полностью пропущенными значениями.

    features : list
        Список всех используемых признаков (и категориальных, и числовых),
        определяющий какие столбцы из df будут использоваться и в каком порядке.

    cat_features : list
        Список категориальных признаков. Все эти признаки должны присутствовать
        в списке features. Остальные признаки будут считаться числовыми.

    target_col : str, optional (default=None)
        Название столбца с целевой переменной, если она включена в df.
        Если указан, столбец будет исключен из признаков и сохранен как цель.

    index_col : str, optional (default=None)
        Название столбца с индексами, если он включен в df.
        Этот столбец будет исключен и не будет использоваться как признак.

    scale_numerical : bool, optional (default=True)
        Масштабировать ли числовые признаки. Рекомендуется всегда использовать
        масштабирование для улучшения сходимости нейронных сетей.

    scale_method : str, optional (default="standard")
        Метод масштабирования числовых признаков:
        - "standard": StandardScaler - нормализация с нулевым средним и единичной
                    дисперсией, лучше работает с нормально распределенными данными
        - "minmax": MinMaxScaler - масштабирование в диапазон [0, 1],
                   сохраняет распределение, но чувствителен к выбросам
        - "quantile": QuantileTransformer - преобразование к равномерному распределению,
                     устойчив к выбросам, но изменяет форму распределения
        - "binning": KBinsDiscretizer - дискретизация на n_bins бинов,
                   полезно при наличии нелинейных зависимостей

    n_bins : int, optional (default=10)
        Количество бинов для метода масштабирования "binning".
        Используется только когда scale_method="binning".

    scaler : object, optional (default=None)
        Предварительно обученный скейлер. Если указан, будет использован вместо
        создания нового. Должен соответствовать указанному scale_method.
        Обычно передается из ранее созданного CatEmbDataset.

    is_multiclass : bool, optional (default=False)
        Флаг, указывающий, используется ли датасет для многоклассовой классификации.
        Если True, целевой тензор будет одномерным (без unsqueeze), как требуется
        для CrossEntropyLoss в PyTorch.

    Атрибуты:
    ----------
    cat_idxs : list
        Индексы категориальных признаков в тензоре данных.

    data : torch.Tensor
        Тензор с преобразованными данными признаков.

    targets : torch.Tensor или None
        Тензор с целевыми значениями, если они были предоставлены.

    scaler : object или None
        Обученный скейлер для числовых признаков.

    Примечания:
    -----------
    1. Категориальные признаки:
       - В тренировочном наборе должны присутствовать все возможные значения
         категориальных признаков. Новые категории в тестовой выборке приведут к ошибкам.
       - Пустые значения (NaN) в категориальных признаках не обрабатываются
         автоматически и должны быть заполнены перед передачей в датасет.
       - Порядок признаков в features определяет их позиции в тензоре данных,
         изменение этого порядка приведет к неверной обработке категориальных признаков.

    2. Числовые признаки:
       - NaN и inf значения автоматически заменяются на 0.
       - Скейлер обучается один раз на тренировочных данных и должен применяться
         ко всем последующим данным через параметр scaler.
       - Для корректной работы, тренировочные данные должны хорошо представлять
         распределение значений, которые могут встретиться в тестовых данных.

    3. Общие требования:
       - Все признаки, указанные в features и cat_features, должны присутствовать в df.
       - Датасет не выполняет проверку типов данных, следует убедиться, что
         категориальные признаки имеют тип 'object' или 'category'.
       - Признаки должны быть предварительно обработаны (заполнение пропусков,
         преобразование типов и т.д.) до передачи в датасет.

    Примеры:
    --------
    >>> # Создание тренировочного датасета с обучением скейлера
    >>> features = ['numeric1', 'numeric2', 'category1', 'category2']
    >>> cat_features = ['category1', 'category2']
    >>> train_dataset = CatEmbDataset(
    ...     train_df,
    ...     features=features,
    ...     cat_features=cat_features,
    ...     target_col='target',
    ...     scale_numerical=True,
    ...     scale_method='standard'
    ... )
    >>>
    >>> # Создание тестового датасета с уже обученным скейлером
    >>> test_dataset = CatEmbDataset(
    ...     test_df,
    ...     features=features,
    ...     cat_features=cat_features,
    ...     scaler=train_dataset.scaler
    ... )
    """
    def __init__(self, df, features, cat_features, target_col=None, index_col=None,
                 scale_numerical=True, scale_method="standard", n_bins=10, scaler=None,
                 is_multiclass=False):
        """
        :param df: DataFrame с исходными данными
        :param features: Список признаков (категориальные + числовые)
        :param cat_features: Список категориальных признаков
        :param target_col: Целевая переменная (если есть)
        :param index_col: Индексная колонка (если есть)
        :param scale_numerical: Масштабировать ли числовые признаки
        :param scale_method: Метод масштабирования ("standard", "minmax", "quantile", "binning")
        :param n_bins: Количество бинов для KBinsDiscretizer (используется только при scale_method="binning")
        :param scaler: Предварительно обученный скейлер (если None, будет создан новый)
        :param is_multiclass: Используется ли датасет для многоклассовой классификации
        """
        self.features = features
        self.cat_features = cat_features
        self.target_col = target_col
        self.scale_numerical = scale_numerical
        self.scale_method = scale_method
        self.n_bins = n_bins
        self.is_multiclass = is_multiclass

        # Создаем копию данных для безопасности
        df_copy = df.copy()

        if index_col is not None and index_col in df_copy.columns:
            df_copy = df_copy.drop(columns=[index_col])

        if target_col and target_col in df_copy.columns:
            # Для многоклассовой классификации не применяем unsqueeze
            if self.is_multiclass:
                self.targets = torch.tensor(df_copy[target_col].values, dtype=torch.long)
            else:
                self.targets = torch.tensor(df_copy[target_col].values, dtype=torch.float32).unsqueeze(1)
            df_copy = df_copy.drop(columns=[target_col])
        else:
            self.targets = None

        # Сохраняем позиции категориальных признаков
        self.cat_idxs = [features.index(f) for f in cat_features if f in features]

        # Масштабируем числовые признаки, если включено
        self.scaler = scaler
        num_features = [f for f in features if f not in cat_features]

        if self.scale_numerical and num_features:
            if self.scaler is None:
                # Создаем и обучаем новый скейлер
                self.scaler = self._create_scaler()
                self.scaler.fit(df_copy[num_features])

            # Применяем скейлер
            if scale_method == "binning":
                # Для KBinsDiscretizer преобразуем данные и заменяем в DataFrame
                df_copy[num_features] = self.scaler.transform(df_copy[num_features])
            else:
                # Для остальных скейлеров
                df_copy[num_features] = self.scaler.transform(df_copy[num_features])
        else:
            # Если масштабирование отключено, просто заменяем nan и inf значения на 0
            if num_features:
                df_copy[num_features] = df_copy[num_features].fillna(0).replace([float('inf'), -float('inf')], 0)

        # Конвертируем данные в тензоры
        self.data = torch.tensor(df_copy[features].values, dtype=torch.float32)

    def _create_scaler(self):
        """Создает скейлер на основе выбранного метода масштабирования."""
        if self.scale_method == "standard":
            return StandardScaler()
        elif self.scale_method == "minmax":
            return MinMaxScaler()
        elif self.scale_method == "quantile":
            return QuantileTransformer(output_distribution='uniform', random_state=42)
        elif self.scale_method == "binning":
            return KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        else:
            raise ValueError(f"Неизвестный метод масштабирования: {self.scale_method}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]