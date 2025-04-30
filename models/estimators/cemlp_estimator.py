import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from tqdm.auto import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
import copy
warnings.filterwarnings('ignore')

# Импортируем CatEmbDataset из модуля dataset
from models.dataset import CatEmbDataset
# Импортируем PyTorch модель и вспомогательные функции
from models.nn.cemlp import CatEmbMLP, softmax


class CatEmbMLPEstimator(BaseEstimator):
    """MLP с эмбеддингами для категориальных признаков для табличных данных.

    Эта модель объединяет эмбеддинги категориальных переменных с числовыми признаками
    и обрабатывает их через многослойный перцептрон (MLP).

    Параметры
    ----------
    cat_emb_dim : int, default=4
        Размерность эмбеддингов для категориальных признаков.
        Рекомендуемый диапазон: [2-10]

    hidden_dims : list, default=[64, 32]
        Список размерностей скрытых слоев MLP.
        Например, [64, 32] создаст два скрытых слоя с 64 и 32 нейронами соответственно.

    activation : str, default='relu'
        Функция активации для скрытых слоев:
        - 'relu': ReLU активация
        - 'leaky_relu': Leaky ReLU с alpha=0.1
        - 'selu': SELU активация (self-normalizing)
        - 'elu': ELU активация
        - 'gelu': GELU активация (используется в BERT)
        - 'swish': Swish активация (SiLU в PyTorch)

    dropout : float, default=0.1
        Вероятность дропаута для регуляризации.
        Рекомендуемый диапазон: [0.0-0.5]

    feature_dropout : float, default=0.0
        Вероятность дропаута для регуляризации признаков.
        Рекомендуемый диапазон: [0.0-0.5]

    normalization : str, default='batch'
        Метод нормализации:
        - 'batch': Batch Normalization
        - 'layer': Layer Normalization
        - 'none': Без нормализации

    virtual_batch_size : int, default=128
        Размер виртуального батча для Batch Normalization.
        Используется только если normalization='batch'.

    momentum : float, default=0.9
        Параметр momentum для BatchNorm слоев.
        Рекомендуемый диапазон: [0.6-0.99]

    initialization : str, default='he_normal'
        Метод инициализации весов сети:
        - 'he_normal': He инициализация с нормальным распределением
        - 'he_uniform': He инициализация с равномерным распределением
        - 'xavier_normal': Xavier/Glorot инициализация с нормальным распределением
        - 'xavier_uniform': Xavier/Glorot инициализация с равномерным распределением

    batch_size : int, default=1024
        Размер батча для обучения.
        Рекомендуемый диапазон: [32-4096]

    epochs : int, default=50
        Количество эпох обучения.
        Рекомендуемый диапазон: [10-200]

    learning_rate : float, default=0.001
        Скорость обучения для оптимизатора Adam.
        Рекомендуемый диапазон: [0.0001-0.01]

    weight_decay : float, default=1e-5
        Коэффициент L2-регуляризации для оптимизатора.
        Рекомендуемый диапазон: [0-0.001]

    early_stopping_patience : int, default=5
        Количество эпох без улучшения до остановки обучения.

    scale_numerical : bool, default=True
        Масштабировать ли числовые признаки.

    scale_method : str, default="standard"
        Метод масштабирования числовых признаков:
        - "standard": StandardScaler (нормализация с нулевым средним и единичной дисперсией)
        - "minmax": MinMaxScaler (масштабирование в диапазон [0, 1])
        - "quantile": QuantileTransformer (преобразование к равномерному распределению)
        - "binning": KBinsDiscretizer (дискретизация на n_bins бинов)

    n_bins : int, default=10
        Количество бинов для метода масштабирования "binning".

    device : str или torch.device, default=None
        Устройство для обучения (cuda/cpu).
        Если None, используется CUDA при наличии.

    output_dim : int, default=1
        Размерность выходного слоя.

    verbose : bool, default=True
        Вывод прогресса обучения.

    num_workers : int, default=0
        Количество worker-процессов для DataLoader.
        0 означает однопроцессный режим.

    random_state : int, default=None
        Случайное состояние для воспроизводимости результатов.

    lr_scheduler_patience : int, default=10
        Пациентность для ReduceLROnPlateau

    lr_scheduler_factor : float, default=0.5
        Фактор для ReduceLROnPlateau

    dynamic_emb_size : bool, default=False
        Использовать ли динамическое определение размера эмбеддингов

    min_emb_dim : int, default=2
        Минимальный размер эмбеддингов

    max_emb_dim : int, default=16
        Максимальный размер эмбеддингов

    constant_value : float, default=0.001
        Константное значение для инициализации весов

    use_self_attention : bool, default=False
        Использовать ли self-attention в модели

    num_attention_heads : int, default=4
        Количество attention heads в модели

    cat_features : list, optional (default=None)
        Список имен категориальных признаков. Если None, будут определяться автоматически.

    Примечания
    ----------
    В модуле представлены три специализированных класса:
    - CatEmbMLPBinary: для бинарной классификации
    - CatEmbMLPMulticlass: для многоклассовой классификации
    - CatEmbMLPRegressor: для задач регрессии

    Примеры
    --------
    >>> from models.nn.cemlp import CatEmbMLPMulticlass
    >>> model = CatEmbMLPMulticlass(
    ...     hidden_dims=[128, 64],
    ...     dropout=0.2,
    ...     batch_norm=True,
    ...     activation='relu',
    ...     scale_numerical=True,
    ...     scale_method="standard",
    ...     cat_features=['cat_col1', 'cat_col2']
    ... )
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric='roc_auc')
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)
    """

    def __init__(self,
                 cat_emb_dim=4,
                 hidden_dims=[512, 256],
                 activation='swish',
                 dropout=0.2,
                 feature_dropout=0.0,
                 normalization='ghost_batch',
                 virtual_batch_size=128,
                 momentum=0.9,
                 initialization='xavier_uniform',
                 constant_value=0.001,
                 leaky_relu_negative_slope=0.1,
                 dynamic_emb_size=False,
                 min_emb_dim=2,
                 max_emb_dim=16,
                 batch_size=1024,
                 epochs=200,
                 learning_rate=0.001,
                 weight_decay=1e-5,
                 early_stopping_patience=15,
                 scale_numerical=True,
                 scale_method="standard",
                 n_bins=10,
                 device=None,
                 output_dim=1,
                 verbose=False,
                 num_workers=0,
                 random_state=None,
                 lr_scheduler_patience=5,
                 lr_scheduler_factor=0.7,
                 use_self_attention=False,
                 attn_dropout=0.1,
                 d_model=80,
                 num_attention_heads=2,
                 cat_features=None):
        self.cat_emb_dim = cat_emb_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.normalization = normalization
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.initialization = initialization
        self.constant_value = constant_value
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.dynamic_emb_size = dynamic_emb_size
        self.min_emb_dim = min_emb_dim
        self.max_emb_dim = max_emb_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.scale_numerical = scale_numerical
        self.scale_method = scale_method
        self.n_bins = n_bins
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim
        self.verbose = verbose
        self.num_workers = num_workers
        self.random_state = random_state
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.use_self_attention = use_self_attention
        self.attn_dropout = attn_dropout
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.cat_features = cat_features

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.model = None
        self.cat_idxs = []
        self.cat_dims = []
        self.features = None
        self.is_fitted_ = False
        self.scaler = None

    def _prepare_data(self, X, y=None, is_train=False, is_multiclass=False):
        """Подготовка данных для обучения/предсказания"""
        if isinstance(X, pd.DataFrame):
            # Если X - это DataFrame, извлекаем признаки
            if is_train:
                # В режиме обучения определяем признаки из X
                features = X.columns.tolist()

                # Определяем категориальные признаки
                if self.cat_features is None:
                    # Если категориальные признаки не указаны при инициализации, определяем по типу данных
                    cat_features = []
                    for col in features:
                        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                            cat_features.append(col)
                    self.cat_features = cat_features
                else:
                    # Используем указанные при инициализации категориальные признаки
                    cat_features = self.cat_features
            else:
                # При предсказании используем признаки из обучения
                features = self.features
                cat_features = self.cat_features if self.cat_features is not None else []

            # Создаем копию датафрейма для безопасности и гарантируем правильный порядок колонок
            X_processed = X[features].copy()

            # Индексы категориальных признаков
            cat_idxs = [features.index(f) for f in cat_features if f in features]

            # Размерности категориальных признаков и преобразование значений
            cat_dims = []
            for cat_feature in cat_features:
                if cat_feature in X_processed.columns:
                    # Если признак уже имеет тип category, получаем его коды
                    if X_processed[cat_feature].dtype.name == 'category':
                        # Получаем коды категорий, убеждаемся что они начинаются с 0
                        X_processed[cat_feature] = X_processed[cat_feature].cat.codes
                        # Заменяем -1 (для NaN) на 0, если такие есть
                        X_processed[cat_feature] = X_processed[cat_feature].fillna(0).astype(int)
                    else:
                        # Преобразуем в категориальный тип
                        X_processed[cat_feature] = X_processed[cat_feature].astype('category').cat.codes
                        # Заменяем -1 (для NaN) на 0, если такие есть
                        X_processed[cat_feature] = X_processed[cat_feature].fillna(0).astype(int)

                    # Определяем размерность (количество уникальных значений)
                    unique_values = X_processed[cat_feature].nunique()
                    cat_dims.append(int(unique_values))
                else:
                    # Если признак отсутствует, пропускаем его
                    if is_train:
                        print(f"Предупреждение: категориальный признак '{cat_feature}' не найден в данных")

            # Преобразуем DataFrame в CatEmbDataset
            if y is not None:
                # Объединяем X и y для создания датасета
                if isinstance(y, pd.Series):
                    y_name = y.name if y.name else 'target'
                    y_df = pd.DataFrame({y_name: y})
                else:
                    y_name = 'target'
                    y_df = pd.DataFrame({y_name: y})

                df = pd.concat([X_processed.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
                dataset = CatEmbDataset(
                    df, features, cat_features,
                    target_col=y_name,
                    scale_numerical=self.scale_numerical,
                    scale_method=self.scale_method,
                    n_bins=self.n_bins,
                    scaler=None if is_train else self.scaler,
                    is_multiclass=is_multiclass
                )

                # Если это тренировочный набор, сохраняем обученный скейлер
                if is_train:
                    self.scaler = dataset.scaler

            else:
                dataset = CatEmbDataset(
                    X_processed, features, cat_features,
                    scale_numerical=self.scale_numerical,
                    scale_method=self.scale_method,
                    n_bins=self.n_bins,
                    scaler=self.scaler,
                    is_multiclass=is_multiclass
                )

            if is_train:
                self.features = features
                self.cat_idxs = cat_idxs
                self.cat_dims = cat_dims

        else:
            # Если X - это numpy array, используем существующие атрибуты
            if self.features is None or self.cat_features is None:
                raise ValueError("При использовании numpy array необходимо предварительно обучить модель на DataFrame и задать cat_features при инициализации")

            # Создаем DataFrame из numpy array с правильными именами колонок
            X_df = pd.DataFrame(X, columns=self.features)

            # Преобразуем категориальные признаки
            for cat_feature in self.cat_features:
                if cat_feature in X_df.columns:
                    X_df[cat_feature] = X_df[cat_feature].astype('category').cat.codes
                    X_df[cat_feature] = X_df[cat_feature].fillna(0).astype(int)

            if y is not None:
                y_df = pd.DataFrame({'target': y})
                df = pd.concat([X_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
                dataset = CatEmbDataset(
                    df, self.features, self.cat_features,
                    target_col='target',
                    scale_numerical=self.scale_numerical,
                    scale_method=self.scale_method,
                    n_bins=self.n_bins,
                    scaler=None if is_train else self.scaler,
                    is_multiclass=is_multiclass
                )

                # Если это тренировочный набор, сохраняем обученный скейлер
                if is_train:
                    self.scaler = dataset.scaler
            else:
                dataset = CatEmbDataset(
                    X_df, self.features, self.cat_features,
                    scale_numerical=self.scale_numerical,
                    scale_method=self.scale_method,
                    n_bins=self.n_bins,
                    scaler=self.scaler,
                    is_multiclass=is_multiclass
                )

        return dataset

    def _init_model(self, input_dim):
        """Инициализация модели CatEmbMLP"""
        return CatEmbMLP(
            input_dim=input_dim,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            dropout=self.dropout,
            feature_dropout=self.feature_dropout,
            normalization=self.normalization,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            initialization=self.initialization,
            constant_value=self.constant_value,
            leaky_relu_negative_slope=self.leaky_relu_negative_slope,
            dynamic_emb_size=self.dynamic_emb_size,
            min_emb_dim=self.min_emb_dim,
            max_emb_dim=self.max_emb_dim,
            use_self_attention=self.use_self_attention,
            num_attention_heads=self.num_attention_heads
        )

    def _train_epoch(self, model, loader, optimizer, criterion, scheduler=None, pbar=True):
        """Обучение на одной эпохе"""
        model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []

        for x, y in tqdm(loader, desc="Training", leave=False, disable=not (self.verbose and pbar)):
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Градиентное клиппирование для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(y.cpu())

        # Собираем все предсказания и цели
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return total_loss / len(loader), all_outputs, all_targets

    def _validate_epoch(self, model, loader, criterion, pbar=True):
        """Валидация на одной эпохе"""
        model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for x, y in tqdm(loader, desc="Validation", leave=False, disable=not (self.verbose and pbar)):
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                loss = criterion(outputs, y)

                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(y.cpu())

        # Собираем все предсказания и цели
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return total_loss / len(loader), all_outputs, all_targets

    def _get_criterion(self):
        """Получение функции потерь (переопределяется в дочерних классах)"""
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _get_default_eval_metric(self):
        """Получение метрики по умолчанию для оценки (переопределяется в дочерних классах)"""
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _get_default_metric_mode(self):
        """Получение режима метрики по умолчанию (переопределяется в дочерних классах)"""
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _get_predictions(self, model, loader, pbar=True):
        """Получение предсказаний модели"""
        model.eval()
        all_outputs = []

        with torch.no_grad():
            for x in tqdm(loader, desc="Predicting", leave=False, disable=not (self.verbose and pbar)):
                if isinstance(x, tuple):
                    x = x[0]  # Если dataset возвращает tuple, берем только данные
                x = x.to(self.device)
                outputs = model(x)
                all_outputs.append(outputs.cpu().numpy())

        return np.concatenate(all_outputs)

    def _calculate_metric(self, y_true, y_pred, metric):
        """Вычисление указанной метрики"""
        if metric == 'roc_auc':
            # Для бинарной классификации
            if self.output_dim == 1:
                y_pred_proba = 1 / (1 + np.exp(-y_pred))
                return roc_auc_score(y_true, y_pred_proba)
            # Для многоклассовой (не поддерживается напрямую)
            else:
                raise ValueError("roc_auc не поддерживается для многоклассовой классификации без дополнительных параметров")

        elif metric == 'accuracy':
            # Для бинарной классификации
            if self.output_dim == 1:
                y_pred_class = (y_pred > 0).astype(int)
                return accuracy_score(y_true, y_pred_class)
            # Для многоклассовой
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
                return accuracy_score(y_true.squeeze(), y_pred_class)

        elif metric == 'mse':
            # Для регрессии
            return mean_squared_error(y_true, y_pred)

        elif metric == 'mae':
            # Для регрессии
            return mean_absolute_error(y_true, y_pred)

        elif metric == 'rmse':
            # Для регрессии
            return np.sqrt(mean_squared_error(y_true, y_pred))

        elif metric == 'r2':
            # Для регрессии
            return r2_score(y_true, y_pred)

        else:
            raise ValueError(f"Неподдерживаемая метрика: {metric}")

    def _evaluate_metrics(self, y_true, y_pred):
        """Оценка метрик (переопределяется в дочерних классах)"""
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _transform_predictions(self, raw_predictions):
        """Преобразование сырых предсказаний (переопределяется в дочерних классах)"""
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, pbar=True):
        """Обучение модели CEMLP

        Параметры:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Входные признаки размерности (n_samples, n_features)
        y : pandas.Series, numpy.ndarray или list
            Целевые значения размерности (n_samples,)
        eval_set : tuple, optional (default=None)
            Кортеж (X_val, y_val) с валидационными данными для мониторинга во время обучения
        eval_metric : str, optional (default=None)
            Метрика для мониторинга на валидации. Поддерживаемые значения:
            - 'roc_auc': AUC ROC (для бинарной классификации)
            - 'accuracy': точность (для классификации)
            - 'mse': среднеквадратичная ошибка (для регрессии)
            - 'mae': средняя абсолютная ошибка (для регрессии)
            - 'rmse': корень из среднеквадратичной ошибки (для регрессии)
            - 'r2': коэффициент детерминации (для регрессии)
            Если None, используется метрика по умолчанию для данного типа задачи
        mode : str, optional (default=None)
            Режим оптимизации метрики:
            - 'max': чем больше, тем лучше (для accuracy, auc, r2)
            - 'min': чем меньше, тем лучше (для loss, mse, rmse)
            Если None, определяется на основе метрики
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        self : объект
            Обученная модель
        """
        # Если метрика не указана, используем метрику по умолчанию для данного типа задачи
        if eval_metric is None:
            eval_metric = self._get_default_eval_metric()

        # Если режим не указан, определяем его на основе метрики
        if mode is None:
            mode = self._get_default_metric_mode()

        # Проверяем корректность режима
        if mode not in ['max', 'min']:
            raise ValueError("Параметр mode должен быть 'max' или 'min'")

        # Подготавливаем обучающие данные с флагом is_train=True для обучения скейлера
        train_dataset = self._prepare_data(X, y, is_train=True)

        # Подготавливаем валидационные данные, если они предоставлены
        val_dataset = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_dataset = self._prepare_data(X_val, y_val, is_train=False)

        # Создаем DataLoader для обучения с правильными настройками многопроцессности
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

        # Создаем DataLoader для валидации, если есть валидационные данные
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=False if self.num_workers == 0 else True,
                pin_memory=True
            )

        # Инициализируем модель
        if self.model is None:
            self.model = self._init_model(len(self.features))
            self.model.to(self.device)

        # Определяем оптимизатор, функцию потерь и scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        criterion = self._get_criterion()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=self.lr_scheduler_patience, factor=self.lr_scheduler_factor
        )

        # Сохраняем лучшую модель
        best_metric = float('inf') if mode == 'min' else float('-inf')
        no_improvement_epochs = 0
        best_model_state = None

        # Цикл обучения
        if self.verbose:
            print(f"Начинаем обучение на {self.device}...")

        for epoch in range(self.epochs):
            # Обучение
            train_loss, train_outputs, train_targets = self._train_epoch(
                self.model, train_loader, optimizer, criterion, scheduler=None, pbar=pbar
            )

            # Очищаем кэш CUDA для предотвращения утечек памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Освобождаем память
            gc.collect()

            # Вычисляем метрику на обучающем наборе
            train_metric = self._calculate_metric(train_targets, train_outputs, eval_metric)

            # Валидация, если предоставлены валидационные данные
            val_loss, val_metric = None, None
            if val_loader is not None:
                val_loss, val_outputs, val_targets = self._validate_epoch(self.model, val_loader, criterion, pbar=pbar)
                val_metric = self._calculate_metric(val_targets, val_outputs, eval_metric)

            # Выводим информацию о прогрессе
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, "
                      f"Train loss: {train_loss:.4f}, Train {eval_metric}: {train_metric:.4f}"
                      + (f", Val loss: {val_loss:.4f}, Val {eval_metric}: {val_metric:.4f}" if val_loader else ""))

            # Обновляем scheduler
            if val_loader is not None:
                # Если есть валидация, используем валидационные метрики
                current_metric = val_metric
                scheduler.step(val_loss)
            else:
                # Если нет валидации, используем обучающие метрики
                current_metric = train_metric
                scheduler.step(train_loss)

            # Проверяем улучшение метрик
            improved = (mode == 'max' and current_metric > best_metric) or (mode == 'min' and current_metric < best_metric)

            if improved:
                best_metric = current_metric
                no_improvement_epochs = 0
                # Сохраняем лучшую модель
                best_model_state = copy.deepcopy(self.model.state_dict())
                if self.verbose:
                    print(f"Сохраняем лучшую модель с метрикой {eval_metric}: {best_metric:.4f}")
            else:
                no_improvement_epochs += 1
                if self.verbose:
                    print(f"Нет улучшения в течение {no_improvement_epochs} эпох")

                if no_improvement_epochs >= self.early_stopping_patience:
                    if self.verbose:
                        print("Останавливаем обучение из-за отсутствия улучшений")
                    break

        # Закрываем DataLoader перед загрузкой новой модели
        del train_loader
        if val_loader is not None:
            del val_loader
        gc.collect()

        # Загружаем лучшую модель
        if best_model_state:
            # Убедимся, что тензоры на нужном устройстве
            for param_tensor in best_model_state:
                best_model_state[param_tensor] = best_model_state[param_tensor].to(self.device)

            self.model.load_state_dict(best_model_state)
            if self.verbose:
                print("Загружена лучшая модель")

        self.is_fitted_ = True
        return self

    def predict(self, X, pbar=True):
        """Предсказание целевых значений

        Параметры:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Входные признаки размерности (n_samples, n_features)
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_pred : numpy.ndarray
            Предсказанные значения размерности (n_samples,)
        """
        self._check_is_fitted()

        # Подготавливаем данные с обученным скейлером
        test_dataset = self._prepare_data(X, is_train=False)

        # Создаем DataLoader для предсказания
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

        # Получаем предсказания
        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)

        # Закрываем DataLoader
        del test_loader
        gc.collect()

        # Преобразуем предсказания в нужный формат
        predictions = self._transform_predictions(raw_predictions)

        return predictions

    def predict_proba(self, X, pbar=True):
        """Предсказание вероятностей классов

        Параметры:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Входные признаки размерности (n_samples, n_features)
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_proba : numpy.ndarray
            Вероятности классов размерности (n_samples, 2)
        """
        self._check_is_fitted()

        # Подготавливаем данные с обученным скейлером
        test_dataset = self._prepare_data(X, is_train=False)

        # Создаем DataLoader для предсказания
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

        # Получаем предсказания
        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)

        # Закрываем DataLoader
        del test_loader
        gc.collect()

        # Преобразуем логиты в вероятности
        proba_1 = 1 / (1 + np.exp(-raw_predictions)).squeeze()

        # Для scikit-learn API нужно возвращать вероятности для обоих классов
        proba_0 = 1 - proba_1

        return np.column_stack((proba_0, proba_1))

    def _check_is_fitted(self):
        """Проверка, что модель обучена"""
        if not self.is_fitted_:
            raise ValueError("Модель не обучена. Сначала выполните метод 'fit'.")


class CatEmbMLPBinary(CatEmbMLPEstimator):
    """MLP с эмбеддингами для задач бинарной классификации.

    Реализация MLP с эмбеддингами для категориальных признаков для задач бинарной классификации.
    Подробности о параметрах см. в документации базового класса CatEmbMLPEstimator.

    Дополнительные методы:
    ----------------------
    predict_proba(X) : возвращает вероятности классов
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_criterion(self):
        """Функция потерь для бинарной классификации"""
        return torch.nn.BCEWithLogitsLoss()

    def _get_default_eval_metric(self):
        """Метрика по умолчанию для бинарной классификации"""
        return 'roc_auc'

    def _get_default_metric_mode(self):
        """Режим метрики по умолчанию для бинарной классификации"""
        return 'max'

    def _evaluate_metrics(self, y_true, y_pred):
        """Оценка AUC для бинарной классификации"""
        # Преобразуем логиты в вероятности
        y_pred_proba = 1 / (1 + np.exp(-y_pred))
        return roc_auc_score(y_true, y_pred_proba)

    def _transform_predictions(self, raw_predictions):
        """Преобразование логитов в классы"""
        # Преобразуем логиты в вероятности
        probabilities = 1 / (1 + np.exp(-raw_predictions))
        # Преобразуем вероятности в классы 0/1
        return (probabilities > 0.5).astype(int).squeeze()

    def predict_proba(self, X, pbar=True):
        """Предсказание вероятностей классов

        Параметры:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Входные признаки размерности (n_samples, n_features)
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_proba : numpy.ndarray
            Вероятности классов размерности (n_samples, 2)
        """
        self._check_is_fitted()

        # Подготавливаем данные с обученным скейлером
        test_dataset = self._prepare_data(X, is_train=False)

        # Создаем DataLoader для предсказания
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

        # Получаем предсказания
        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)

        # Закрываем DataLoader
        del test_loader
        gc.collect()

        # Преобразуем предсказания в нужный формат
        predictions = self._transform_predictions(raw_predictions)

        return predictions


class CatEmbMLPMulticlass(CatEmbMLPEstimator):
    """MLP с эмбеддингами для задач многоклассовой классификации.

    Реализация MLP с эмбеддингами для категориальных признаков для задач многоклассовой классификации.
    Подробности о параметрах см. в документации базового класса CatEmbMLPEstimator.

    Дополнительные параметры:
    -------------------------
    n_classes : int, обязательный
        Количество классов для многоклассовой классификации

    Дополнительные методы:
    ----------------------
    predict_proba(X) : возвращает вероятности для всех классов
    """

    def __init__(self, n_classes=None, **kwargs):
        if n_classes is None:
            raise ValueError("Для многоклассовой классификации необходимо указать параметр 'n_classes'")
        self.n_classes = n_classes
        super().__init__(output_dim=n_classes, **kwargs)
        self.label_encoder = LabelEncoder()

    def _prepare_data(self, X, y=None, is_train=False):
        """Переопределенный метод для передачи is_multiclass=True"""
        return super()._prepare_data(X, y, is_train, is_multiclass=True)

    def _get_criterion(self):
        """Функция потерь для многоклассовой классификации"""
        return torch.nn.CrossEntropyLoss()

    def _get_default_eval_metric(self):
        """Метрика по умолчанию для многоклассовой классификации"""
        return 'accuracy'

    def _get_default_metric_mode(self):
        """Режим метрики по умолчанию для многоклассовой классификации"""
        return 'max'

    def _evaluate_metrics(self, y_true, y_pred):
        """Оценка точности для многоклассовой классификации"""
        # Преобразуем логиты в классы
        predicted_classes = np.argmax(y_pred, axis=1)
        true_classes = y_true.squeeze()
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def _transform_predictions(self, raw_predictions):
        """Преобразование логитов в классы"""
        # Находим класс с максимальной вероятностью
        predicted_classes = np.argmax(raw_predictions, axis=1)

        # Если была выполнена кодировка меток, декодируем их обратно
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.inverse_transform(predicted_classes)

        return predicted_classes

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, pbar=True):
        """Обучение модели с предварительной кодировкой меток классов"""
        # Кодируем метки классов
        encoded_y = self.label_encoder.fit_transform(y)

        # Подготавливаем валидационные данные, если они предоставлены
        if eval_set is not None:
            X_val, y_val = eval_set
            encoded_y_val = self.label_encoder.transform(y_val)
            eval_set = (X_val, encoded_y_val)

        # Обучаем модель с закодированными метками
        return super().fit(X, encoded_y, eval_set=eval_set, eval_metric=eval_metric, mode=mode, pbar=pbar)

    def predict_proba(self, X, pbar=True):
        """Предсказание вероятностей классов

        Параметры:
        -----------
        X : pandas.DataFrame или numpy.ndarray
            Входные признаки размерности (n_samples, n_features)
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_proba : numpy.ndarray
            Вероятности классов размерности (n_samples, n_classes)
        """
        self._check_is_fitted()

        # Подготавливаем данные с обученным скейлером
        test_dataset = self._prepare_data(X, is_train=False)

        # Создаем DataLoader для предсказания
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

        # Получаем предсказания
        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)

        # Закрываем DataLoader
        del test_loader
        gc.collect()

        # Применяем softmax для получения вероятностей
        return softmax(raw_predictions, axis=1)


class CatEmbMLPRegressor(CatEmbMLPEstimator):
    """MLP с эмбеддингами для задач регрессии.

    Реализация MLP с эмбеддингами для категориальных признаков для задач регрессии.
    Подробности о параметрах см. в документации базового класса CatEmbMLPEstimator.
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

    def _get_criterion(self):
        """Функция потерь для регрессии"""
        return torch.nn.MSELoss()

    def _get_default_eval_metric(self):
        """Метрика по умолчанию для регрессии"""
        return 'mse'

    def _get_default_metric_mode(self):
        """Режим метрики по умолчанию для регрессии"""
        return 'min'

    def _evaluate_metrics(self, y_true, y_pred):
        """Оценка MSE для регрессии"""
        return np.mean((y_true - y_pred) ** 2)

    def _transform_predictions(self, raw_predictions):
        """Преобразование выходов модели для регрессии"""
        # Для регрессии просто возвращаем предсказания
        return raw_predictions.squeeze()
