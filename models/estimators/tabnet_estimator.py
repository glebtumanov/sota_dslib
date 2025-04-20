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

# Импортируем TabNet и другие компоненты из nn.tabnet
from models.nn.tabnet import TabNet, softmax

# Добавляем функцию безопасного сигмоида
def sigmoid(x):
    """Безопасная реализация сигмоида, избегающая переполнения."""
    # Для положительных x используем стандартную формулу
    # Для отрицательных делаем преобразование для стабильности
    return np.where(x >= 0,
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))


class TabNetEstimator(BaseEstimator):
    """TabNet: Интерпретируемый алгоритм машинного обучения для табличных данных с механизмом внимания.

    TabNet - это нейросетевая архитектура, предложенная Google Research в статье
    "TabNet: Attentive Interpretable Tabular Learning" (https://arxiv.org/pdf/1908.07442v5).
    Она использует последовательные шаги обработки с механизмом внимания, который
    выбирает наиболее важные признаки на каждом шаге обучения.

    Основные преимущества TabNet:
    - Интерпретируемость: модель показывает, какие признаки важны для принятия решений
    - Эффективность: достигает высокой производительности на табличных данных
    - Гибкость: работает как с числовыми, так и с категориальными признаками
    - Встроенный отбор признаков: модель автоматически выбирает важные признаки

    Параметры
    ----------
    cat_emb_dim : int, default=6
        Размерность эмбеддингов для категориальных признаков.
        Рекомендуемый диапазон: [4-10]

    n_steps : int, default=4
        Количество шагов в архитектуре TabNet (количество слоев принятия решений).
        Рекомендуемый диапазон: [3-10]

    n_d : int, default=8
        Размерность выхода decision component (Nd).
        Рекомендуемый диапазон: [8-64]

    n_a : int, default=8
        Размерность выхода attention component (Na).
        Рекомендуемый диапазон: [8-64]

    decision_dim : int, default=8
        Размерность решающего слоя. Обычно меньше n_d.
        Рекомендуемый диапазон: [4-64]

    n_glu_layers : int, default=3
        Количество GLU (Gated Linear Unit) слоев.
        Рекомендуемый диапазон: [2-4]

    dropout : float, default=0.1
        Вероятность дропаута для регуляризации.
        Рекомендуемый диапазон: [0.1-0.9]

    gamma : float, default=1.5
        Коэффициент затухания для масок внимания.
        Рекомендуемый диапазон: [1.0-2.0]

    lambda_sparse : float, default=0.0001
        Коэффициент регуляризации разреженности.
        Рекомендуемый диапазон: [0-0.01]

    virtual_batch_size : int, default=128
        Размер виртуального батча для Ghost BatchNorm.
        Рекомендуемый диапазон: [128-4096]

    momentum : float, default=0.9
        Параметр momentum для BatchNorm.
        Рекомендуемый диапазон: [0.6-0.98]

    batch_size : int, default=1024
        Размер батча для обучения.
        Рекомендуемый диапазон: [256-32768]

    epochs : int, default=50
        Количество эпох обучения.
        Рекомендуемый диапазон: [20-100]

    learning_rate : float, default=0.005
        Скорость обучения для оптимизатора Adam.
        Рекомендуемый диапазон: [0.001-0.025]

    early_stopping_patience : int, default=5
        Количество эпох без улучшения до остановки обучения.

    weight_decay : float, default=1e-5
        Коэффициент L2-регуляризации для оптимизатора.

    reducelronplateau_patience : int, default=10
        Количество эпох без улучшения до снижения learning rate для ReduceLROnPlateau.
        Рекомендуемый диапазон: [5-15]

    reducelronplateau_factor : float, default=0.5
        Коэффициент снижения learning rate для ReduceLROnPlateau.
        Рекомендуемый диапазон: [0.1-0.9]

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

    Примечания
    ----------
    TabNet поддерживает как числовые, так и категориальные признаки. Категориальные
    признаки автоматически преобразуются в эмбеддинги заданной размерности.

    Модель работает только с данными в формате pandas.DataFrame.

    В модуле представлены три специализированных класса:
    - TabNetBinary: для бинарной классификации
    - TabNetMulticlass: для многоклассовой классификации
    - TabNetRegressor: для задач регрессии

    Примеры
    --------
    >>> from models.nn.tabnet import TabNetBinary
    >>> # Пример с автоматическим определением категориальных признаков
    >>> model = TabNetBinary(
    ...     n_d=32,
    ...     n_a=32,
    ...     n_steps=5,
    ...     dropout=0.3,
    ...     scale_numerical=True,
    ...     scale_method="standard"
    ... )
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric='roc_auc')
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)
    >>>
    >>> # Пример с явным указанием категориальных признаков
    >>> cat_features = ['cat_feature1', 'cat_feature2', 'cat_feature3']
    >>> model = TabNetBinary(
    ...     n_d=32,
    ...     n_a=32,
    ...     n_steps=5
    ... )
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
    >>> y_pred = model.predict(X_test, cat_features=cat_features)
    >>> y_proba = model.predict_proba(X_test, cat_features=cat_features)
    """

    def __init__(self,
                 cat_emb_dim=4,  # Размерность эмбеддингов для категориальных признаков
                 n_steps=5,  # Количество шагов в TabNet
                 n_d=64,  # Размерность выхода decision component (Nd)
                 n_a=64,  # Размерность выхода attention component (Na)
                 decision_dim=32,  # Размерность решающего слоя
                 n_glu_layers=2,  # Количество GLU слоев
                 dropout=0.1,  # Вероятность дропаута
                 gamma=1.5,  # Коэффициент затухания для масок внимания
                 lambda_sparse=0.00001,  # Коэффициент регуляризации разреженности
                 virtual_batch_size=256,  # Размер виртуального батча для Ghost BatchNorm
                 momentum=0.9,  # Параметр momentum для BatchNorm
                 batch_size=4096,  # Размер батча для обучения
                 epochs=100,  # Количество эпох обучения
                 learning_rate=0.02,  # Скорость обучения
                 early_stopping_patience=10,  # Количество эпох без улучшения до остановки
                 weight_decay=1e-5,  # Весовая регуляризация для оптимизатора
                 reducelronplateau_patience=5,  # Количество эпох без улучшения до снижения learning rate
                 reducelronplateau_factor=0.7,  # Коэффициент снижения learning rate [0.7-0.9]
                 scale_numerical=True,  # Масштабировать ли числовые признаки
                 scale_method="standard",  # Метод масштабирования ("standard", "minmax", "quantile", "binning")
                 n_bins=10,  # Количество бинов для binning
                 device=None,  # Устройство для обучения (cuda/cpu)
                 output_dim=1,  # Размерность выходного слоя
                 verbose=True,  # Вывод прогресса обучения
                 num_workers=0,  # Количество worker-процессов для DataLoader (0 - однопроцессный режим)
                 random_state=42):  # Случайное состояние для воспроизводимости

        self.cat_emb_dim = cat_emb_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.decision_dim = decision_dim
        self.n_glu_layers = n_glu_layers
        self.dropout = dropout
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.reducelronplateau_patience = reducelronplateau_patience
        self.reducelronplateau_factor = reducelronplateau_factor
        self.scale_numerical = scale_numerical
        self.scale_method = scale_method
        self.n_bins = n_bins
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim
        self.verbose = verbose
        self.num_workers = num_workers
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.model = None
        self.cat_idxs = []
        self.cat_dims = []
        self.features = None
        self.cat_features = None
        self.is_fitted_ = False
        self.scaler = None  # Будет содержать обученный скейлер

    def _prepare_data(self, X, y=None, is_train=False, is_multiclass=False, cat_features=None):
        # Извлекаем признаки
        features = X.columns.tolist()

        # Определяем категориальные признаки
        if cat_features is not None:
            # Используем указанные категориальные признаки
            cat_features = [f for f in cat_features if f in features]
        else:
            # Автоматически определяем категориальные признаки по типу данных
            cat_features = []
            for col in features:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    cat_features.append(col)

        # Индексы категориальных признаков
        cat_idxs = [features.index(f) for f in cat_features]

        # Размерности категориальных признаков
        cat_dims = []
        for cat_feature in cat_features:
            unique_values = X[cat_feature].nunique()
            cat_dims.append(int(unique_values))

        # Преобразуем DataFrame в CatEmbDataset
        if y is not None:
            # Объединяем X и y для создания датасета
            if isinstance(y, pd.Series):
                y_name = y.name if y.name else 'target'
                y_df = pd.DataFrame({y_name: y})
            else:
                y_name = 'target'
                y_df = pd.DataFrame({y_name: y})

            df = pd.concat([X.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
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
                X, features, cat_features,
                scale_numerical=self.scale_numerical,
                scale_method=self.scale_method,
                n_bins=self.n_bins,
                scaler=self.scaler,
                is_multiclass=is_multiclass
            )

        self.features = features
        self.cat_features = cat_features
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims

        return dataset

    def _init_model(self, input_dim):
        return TabNet(
            input_dim=input_dim,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_steps=self.n_steps,
            n_d=self.n_d,
            n_a=self.n_a,
            decision_dim=self.decision_dim,
            n_glu_layers=self.n_glu_layers,
            dropout=self.dropout,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            output_dim=self.output_dim
        )

    def _train_epoch(self, model, loader, optimizer, criterion, scheduler=None, pbar=True):
        model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []

        for x, y in tqdm(loader, desc="Training", leave=False, disable=not (self.verbose and pbar)):
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()

            # Получаем выходы и маски для регуляризации
            outputs, masks = model(x, return_masks=True)

            # Вычисляем основную функцию потерь
            main_loss = criterion(outputs, y)

            # Вычисляем регуляризацию разреженности
            sparse_loss = model.calculate_sparse_loss(masks)

            # Суммарная функция потерь
            loss = main_loss + sparse_loss

            loss.backward()

            # Добавляем градиентное клиппирование для стабильности
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
        model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for x, y in tqdm(loader, desc="Validation", leave=False, disable=not (self.verbose and pbar)):
                x, y = x.to(self.device), y.to(self.device)

                # Получаем только выходы без регуляризации разреженности при валидации
                outputs, _ = model(x, return_masks=True)
                loss = criterion(outputs, y)

                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(y.cpu())

        # Собираем все предсказания и цели
        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return total_loss / len(loader), all_outputs, all_targets

    def _get_predictions(self, model, loader, pbar=True):
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
        if metric == 'roc_auc':
            # Для бинарной классификации
            if self.output_dim == 1:
                y_pred_proba = sigmoid(y_pred)
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

    def _get_criterion(self):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _get_default_eval_metric(self):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _get_default_metric_mode(self):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _evaluate_metrics(self, y_true, y_pred):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _transform_predictions(self, raw_predictions):
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise ValueError("Модель не обучена. Сначала выполните метод 'fit'.")

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, cat_features=None, pbar=True):
        """Обучение модели TabNet

        Параметры:
        -----------
        X : pandas.DataFrame
            Входные признаки размерности (n_samples, n_features)
        y : pandas.Series или list
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
        cat_features : list, optional (default=None)
            Список имен категориальных признаков. Если указан, используются эти признаки.
            Если None, категориальные признаки определяются автоматически по типу данных.
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        self : объект
            Обученная модель
        """
        # Определяем метрику и режим оптимизации
        eval_metric = eval_metric or self._get_default_eval_metric()
        mode = mode or self._get_default_metric_mode()

        if mode not in ['max', 'min']:
            raise ValueError("Параметр mode должен быть 'max' или 'min'")

        # Подготавливаем данные
        train_dataset = self._prepare_data(X, y, is_train=True, cat_features=cat_features)

        val_dataset = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_dataset = self._prepare_data(X_val, y_val, is_train=False, cat_features=cat_features)

        # Создаем DataLoader для обучения
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

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

        # Инициализируем модель, если еще не инициализирована
        if self.model is None:
            self.model = self._init_model(len(self.features))
            self.model.to(self.device)

        # Настраиваем оптимизатор и функцию потерь
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        criterion = self._get_criterion()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=self.reducelronplateau_patience, factor=self.reducelronplateau_factor
        )

        # Переменные для отслеживания лучшей модели
        best_metric = float('inf') if mode == 'min' else float('-inf')
        no_improvement_epochs = 0
        best_model_state = None

        if self.verbose:
            print(f"Начинаем обучение на {self.device}...")

        # Цикл обучения
        for epoch in range(self.epochs):
            # Обучение
            train_loss, train_outputs, train_targets = self._train_epoch(
                self.model, train_loader, optimizer, criterion, scheduler=None, pbar=pbar
            )

            # Освобождаем память
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Вычисляем метрику на обучающем наборе
            train_metric = self._calculate_metric(train_targets, train_outputs, eval_metric)

            # Валидация
            val_loss, val_metric = None, None
            if val_loader is not None:
                val_loss, val_outputs, val_targets = self._validate_epoch(self.model, val_loader, criterion, pbar=pbar)
                val_metric = self._calculate_metric(val_targets, val_outputs, eval_metric)
                current_metric = val_metric
                scheduler.step(val_loss)
            else:
                current_metric = train_metric
                scheduler.step(train_loss)

            # Выводим информацию о прогрессе
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, "
                      f"Train loss: {train_loss:.4f}, Train {eval_metric}: {train_metric:.4f}"
                      + (f", Val loss: {val_loss:.4f}, Val {eval_metric}: {val_metric:.4f}" if val_loader else ""))

            # Проверяем улучшение метрик
            improved = (mode == 'max' and current_metric > best_metric) or (mode == 'min' and current_metric < best_metric)

            if improved:
                best_metric = current_metric
                no_improvement_epochs = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                if self.verbose:
                    print(f"Сохраняем лучшую модель с метрикой {eval_metric}: {best_metric:.4f}")
            else:
                no_improvement_epochs += 1
                if self.verbose and no_improvement_epochs >= self.early_stopping_patience:
                    print(f"Останавливаем обучение из-за отсутствия улучшений в течение {no_improvement_epochs} эпох")
                    break
                elif self.verbose:
                    print(f"Нет улучшения в течение {no_improvement_epochs} эпох")

        # Закрываем DataLoader перед загрузкой новой модели
        del train_loader
        if val_loader is not None:
            del val_loader
        gc.collect()

        # Загружаем лучшую модель
        if best_model_state:
            for param_tensor in best_model_state:
                best_model_state[param_tensor] = best_model_state[param_tensor].to(self.device)
            self.model.load_state_dict(best_model_state)
            if self.verbose:
                print("Загружена лучшая модель")

        self.is_fitted_ = True
        return self

    def predict(self, X, cat_features=None, pbar=True):
        """Предсказание целевых значений

        Параметры:
        -----------
        X : pandas.DataFrame
            Входные признаки размерности (n_samples, n_features)
        cat_features : list, optional (default=None)
            Список имен категориальных признаков. Если указан, используются эти признаки.
            Если None, используются категориальные признаки, определенные при обучении.
            Если ни один вариант не доступен, признаки определяются автоматически.
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_pred : numpy.ndarray
            Предсказанные значения размерности (n_samples,)
        """
        self._check_is_fitted()

        # Если cat_features не указан, используем сохраненный при обучении
        if cat_features is None and hasattr(self, 'cat_features'):
            cat_features = self.cat_features

        # Подготавливаем данные и создаем DataLoader
        test_dataset = self._prepare_data(X, is_train=False, cat_features=cat_features)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False if self.num_workers == 0 else True,
            pin_memory=True
        )

        # Получаем и преобразуем предсказания
        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)

        # Очищаем память
        del test_loader
        gc.collect()

        return self._transform_predictions(raw_predictions)


class TabNetBinary(TabNetEstimator):
    """TabNet для бинарной классификации.

    Реализация TabNet для задач бинарной классификации.
    Подробности о параметрах см. в документации базового класса TabNetEstimator.

    Дополнительные методы:
    ----------------------
    predict_proba(X) : возвращает вероятности классов
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

    def _get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def _get_default_eval_metric(self):
        return 'roc_auc'

    def _get_default_metric_mode(self):
        return 'max'

    def _evaluate_metrics(self, y_true, y_pred):
        # Преобразуем логиты в вероятности
        y_pred_proba = sigmoid(y_pred)
        return roc_auc_score(y_true, y_pred_proba)

    def _transform_predictions(self, raw_predictions):
        # Преобразуем логиты в вероятности
        probabilities = sigmoid(raw_predictions)
        # Преобразуем вероятности в классы 0/1
        return (probabilities > 0.5).astype(int).squeeze()

    def predict_proba(self, X, cat_features=None, pbar=True):
        """Предсказание вероятностей классов

        Параметры:
        -----------
        X : pandas.DataFrame
            Входные признаки размерности (n_samples, n_features)
        cat_features : list, optional (default=None)
            Список имен категориальных признаков. Если указан, используются эти признаки.
            Если None, используются категориальные признаки, определенные при обучении.
            Если ни один вариант не доступен, признаки определяются автоматически.
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_proba : numpy.ndarray
            Вероятности классов размерности (n_samples, 2)
        """
        self._check_is_fitted()

        # Если cat_features не указан, используем сохраненный при обучении
        if cat_features is None and hasattr(self, 'cat_features'):
            cat_features = self.cat_features

        # Подготавливаем данные и создаем DataLoader
        test_dataset = self._prepare_data(X, is_train=False, cat_features=cat_features)
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

        # Очищаем память
        del test_loader
        gc.collect()

        # Преобразуем логиты в вероятности, используя безопасную сигмоиду
        proba_1 = sigmoid(raw_predictions).squeeze()
        proba_0 = 1 - proba_1

        return np.column_stack((proba_0, proba_1))


class TabNetMulticlass(TabNetEstimator):
    """TabNet для многоклассовой классификации.

    Реализация TabNet для задач многоклассовой классификации.
    Подробности о параметрах см. в документации базового класса TabNetEstimator.

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

    def _prepare_data(self, X, y=None, is_train=False, cat_features=None):
        return super()._prepare_data(X, y, is_train, is_multiclass=True, cat_features=cat_features)

    def _get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def _get_default_eval_metric(self):
        return 'accuracy'

    def _get_default_metric_mode(self):
        return 'max'

    def _evaluate_metrics(self, y_true, y_pred):
        # Преобразуем логиты в классы
        predicted_classes = np.argmax(y_pred, axis=1)
        true_classes = y_true.squeeze()
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def _transform_predictions(self, raw_predictions):
        # Находим класс с максимальной вероятностью
        predicted_classes = np.argmax(raw_predictions, axis=1)

        # Если была выполнена кодировка меток, декодируем их обратно
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.inverse_transform(predicted_classes)

        return predicted_classes

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, cat_features=None, pbar=True):
        """Обучение модели с предварительной кодировкой меток классов"""
        # Кодируем метки классов
        encoded_y = self.label_encoder.fit_transform(y)

        # Подготавливаем валидационные данные, если они предоставлены
        if eval_set is not None:
            X_val, y_val = eval_set
            encoded_y_val = self.label_encoder.transform(y_val)
            eval_set = (X_val, encoded_y_val)

        # Обучаем модель с закодированными метками
        return super().fit(X, encoded_y, eval_set=eval_set, eval_metric=eval_metric, mode=mode, cat_features=cat_features, pbar=pbar)

    def predict_proba(self, X, cat_features=None, pbar=True):
        """Предсказание вероятностей классов

        Параметры:
        -----------
        X : pandas.DataFrame
            Входные признаки размерности (n_samples, n_features)
        cat_features : list, optional (default=None)
            Список имен категориальных признаков. Если указан, используются эти признаки.
            Если None, используются категориальные признаки, определенные при обучении.
            Если ни один вариант не доступен, признаки определяются автоматически.
        pbar : bool, optional (default=True)
            Отображать ли прогресс-бар при verbose=True

        Возвращает:
        -----------
        y_proba : numpy.ndarray
            Вероятности классов размерности (n_samples, n_classes)
        """
        self._check_is_fitted()

        # Если cat_features не указан, используем сохраненный при обучении
        if cat_features is None and hasattr(self, 'cat_features'):
            cat_features = self.cat_features

        # Подготавливаем данные и создаем DataLoader
        test_dataset = self._prepare_data(X, is_train=False, cat_features=cat_features)
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

        # Очищаем память
        del test_loader
        gc.collect()

        # Применяем softmax для получения вероятностей
        return softmax(raw_predictions, axis=1)

class TabNetRegressor(TabNetEstimator):
    """TabNet для регрессии.

    Реализация TabNet для задач регрессии.
    Подробности о параметрах см. в документации базового класса TabNetEstimator.
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

    def _get_criterion(self):
        return torch.nn.MSELoss()

    def _get_default_eval_metric(self):
        return 'mse'

    def _get_default_metric_mode(self):
        return 'min'

    def _evaluate_metrics(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _transform_predictions(self, raw_predictions):
        return raw_predictions.squeeze()