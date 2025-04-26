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

# Импортируем новую реализацию TabNet
from models.nn.tabnet import TabNet

# Добавляем функцию безопасного сигмоида
def sigmoid(x):
    """Безопасная реализация сигмоида, избегающая переполнения."""
    # Для положительных x используем стандартную формулу
    # Для отрицательных делаем преобразование для стабильности
    return np.where(x >= 0,
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))

def softmax(x, axis=None):
    """Вычисление softmax по указанной оси (используется для мультикласса)"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class TabNetEstimator(BaseEstimator):
    """Обновленный TabNetEstimator на основе новой архитектуры TabNet.

    Использует последовательные шаги обработки с механизмом внимания, который
    выбирает наиболее важные признаки на каждом шаге обучения.

    Параметры
    ----------
    d_model : int, default=8
        Размерность эмбеддингов для числовых и категориальных признаков.

    n_steps : int, default=3
        Количество шагов в архитектуре TabNet.
        Рекомендуемый диапазон: [3-10]

    decision_dim : int, default=64
        Общая размерность выхода FeatureTransformer на каждом шаге (Nd + Na).
        Должна быть четной.
        Рекомендуемый диапазон: [16-128]

    n_shared : int, default=2
        Количество общих GLU блоков в FeatureTransformer.
        Рекомендуемый диапазон: [1-4]

    n_independent : int, default=2
        Количество независимых GLU блоков в FeatureTransformer на каждом шаге.
        Рекомендуемый диапазон: [1-4]

    glu_dropout : float, default=0.0
        Вероятность дропаута в GLU блоках.
        Рекомендуемый диапазон: [0.0-0.5]

    dropout_emb : float, default=0.05
        Вероятность дропаута после слоя эмбеддингов.
        Рекомендуемый диапазон: [0.0-0.3]

    gamma : float, default=1.5
        Коэффициент затухания для масок внимания (prior relaxation).
        Рекомендуемый диапазон: [1.0-2.0]

    lambda_sparse : float, default=1e-4
        Коэффициент регуляризации разреженности (потеря энтропии масок).
        Рекомендуемый диапазон: [1e-6 - 1e-3]

    batch_size : int, default=1024
        Размер батча для обучения.

    epochs : int, default=100
        Максимальное количество эпох обучения.

    learning_rate : float, default=0.01
        Скорость обучения для оптимизатора Adam.

    early_stopping_patience : int, default=10
        Количество эпох без улучшения до остановки обучения.

    weight_decay : float, default=1e-5
        Коэффициент L2-регуляризации для оптимизатора.

    reducelronplateau_patience : int, default=5
        Количество эпох без улучшения до снижения learning rate.

    reducelronplateau_factor : float, default=0.7
        Коэффициент снижения learning rate.

    scale_numerical : bool, default=True
        Масштабировать ли числовые признаки.

    scale_method : str, default="standard"
        Метод масштабирования ("standard", "minmax", "quantile", "binning").

    n_bins : int, default=10
        Количество бинов для "binning".

    device : str или torch.device, default=None
        Устройство для обучения (cuda/cpu).

    output_dim : int, default=1
        Размерность выходного слоя (переопределяется в дочерних классах).

    verbose : bool, default=True
        Вывод прогресса обучения.

    num_workers : int, default=0
        Количество worker-процессов для DataLoader.

    random_state : int, default=42
        Случайное состояние для воспроизводимости.

    momentum : float, default=0.1
        Momentum для BatchNorm1d в AttentiveTransformer

    virtual_batch_size : int, default=128
        Размер виртуального батча для GhostBatchNorm в GLU

    Примечания
    ----------
    Модель работает только с данными в формате pandas.DataFrame.
    Категориальные признаки автоматически преобразуются в эмбеддинги.
    Представлены специализированные классы:
    - TabNetBinary: для бинарной классификации
    - TabNetMulticlass: для многоклассовой классификации
    - TabNetRegressor: для задач регрессии
    """

    def __init__(self,
                 d_model=8,             # Размерность эмбеддингов
                 n_steps=3,             # Количество шагов TabNet
                 decision_dim=64,       # Общая размерность выхода FeatureTransformer (Nd+Na)
                 n_shared=2,            # Кол-во общих GLU блоков
                 n_independent=2,       # Кол-во независимых GLU блоков на шаге
                 glu_dropout=0.0,       # Dropout в GLU
                 dropout_emb=0.05,      # Dropout после эмбеддингов
                 gamma=1.5,             # Коэффициент релаксации prior
                 lambda_sparse=1e-4,    # Коэффициент регуляризации разреженности
                 batch_size=1024,       # Размер батча
                 epochs=100,            # Количество эпох
                 learning_rate=0.01,    # Скорость обучения
                 early_stopping_patience=10, # Терпение для ранней остановки
                 weight_decay=1e-5,     # L2 регуляризация
                 reducelronplateau_patience=5, # Терпение для снижения LR
                 reducelronplateau_factor=0.7, # Фактор снижения LR
                 scale_numerical=True,  # Масштабировать числовые?
                 scale_method="standard", # Метод масштабирования
                 n_bins=10,             # Кол-во бинов для 'binning'
                 device=None,           # Устройство cuda/cpu
                 output_dim=1,          # Размерность выхода (задается подклассами)
                 momentum=0.1,          # Momentum для BatchNorm1d в AttentiveTransformer
                 verbose=True,          # Выводить прогресс?
                 num_workers=0,         # Кол-во воркеров DataLoader
                 random_state=42,
                 virtual_batch_size=128): # Размер виртуального батча для GhostBatchNorm в GLU

        self.d_model = d_model
        self.n_steps = n_steps
        self.decision_dim = decision_dim
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.glu_dropout = glu_dropout
        self.dropout_emb = dropout_emb
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
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
        self.momentum = momentum
        self.virtual_batch_size = virtual_batch_size

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.model = None
        self.cat_idxs = []
        self.cat_dims = []
        self.features = None
        self.num_continuous = 0 # Будет вычислено в _prepare_data
        self.cat_features = None
        self.is_fitted_ = False
        self.scaler = None

    def _prepare_data(self, X, y=None, is_train=False, cat_features=None, is_multiclass=False):
        features = X.columns.tolist()

        if cat_features is not None:
            cat_features = [f for f in cat_features if f in features]
        else:
            cat_features = []
            for col in features:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    cat_features.append(col)

        cat_idxs = [features.index(f) for f in cat_features]
        cat_dims = [int(X[cat_feature].nunique() + 1) for cat_feature in cat_features]
        num_continuous = len(features) - len(cat_features)

        if y is not None:
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

        # Сохраняем параметры для инициализации модели
        self.features = features
        self.cat_features = cat_features
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.num_continuous = num_continuous

        return dataset

    def _init_model(self):
        # Параметры для TabNetCore из __init__ эстиматора
        core_kw = {
            'n_steps': self.n_steps,
            'decision_dim': self.decision_dim,
            'n_shared': self.n_shared,
            'n_independent': self.n_independent,
            'glu_dropout': self.glu_dropout,
            'gamma': self.gamma
        }
        return TabNet(
            num_continuous=self.num_continuous,
            cat_dims=self.cat_dims,
            cat_idx=self.cat_idxs,
            d_model=self.d_model,
            output_dim=self.output_dim,
            dropout_emb=self.dropout_emb,
            att_momentum=self.momentum,
            virtual_batch_size=self.virtual_batch_size,
            **core_kw
        )

    def _train_epoch(self, model, loader, optimizer, criterion, scheduler=None, pbar=True):
        model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []

        for x, y in tqdm(loader, desc="Training", leave=False, disable=not (self.verbose and pbar)):
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()

            # Модель теперь возвращает (outputs, sparse_loss)
            outputs, sparse_loss = model(x)

            # Вычисляем основную функцию потерь
            main_loss = criterion(outputs, y)

            # Суммарная функция потерь с регуляризацией разреженности
            loss = main_loss + self.lambda_sparse * sparse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler: # Step per batch if using OneCycleLR or similar
                 pass # scheduler.step()

            total_loss += loss.item() # Используем суммарную потерю для отслеживания
            all_outputs.append(outputs.detach().cpu()) # Сохраняем только логиты
            all_targets.append(y.cpu())

        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return total_loss / len(loader), all_outputs, all_targets

    def _validate_epoch(self, model, loader, criterion, pbar=True):
        model.eval()
        total_loss = 0 # Отслеживаем только основную потерю для валидации
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for x, y in tqdm(loader, desc="Validation", leave=False, disable=not (self.verbose and pbar)):
                x, y = x.to(self.device), y.to(self.device)

                # Получаем выходы, игнорируем sparse_loss для валидации
                outputs, _ = model(x)
                loss = criterion(outputs, y)

                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(y.cpu())

        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return total_loss / len(loader), all_outputs, all_targets

    def _get_predictions(self, model, loader, pbar=True):
        model.eval()
        all_outputs = []

        with torch.no_grad():
            for x in tqdm(loader, desc="Predicting", leave=False, disable=not (self.verbose and pbar)):
                if isinstance(x, tuple):
                    x = x[0]
                x = x.to(self.device)
                # Получаем только логиты для предсказаний
                outputs, _ = model(x)
                all_outputs.append(outputs.cpu().numpy())

        return np.concatenate(all_outputs)

    def _calculate_metric(self, y_true, y_pred, metric):
        # Логика расчета метрик остается прежней
        if metric == 'roc_auc':
            if self.output_dim == 1:
                y_pred_proba = sigmoid(y_pred)
                return roc_auc_score(y_true, y_pred_proba)
            else:
                 # Для многоклассовой классификации используем вероятности softmax
                 # y_pred здесь уже содержит логиты
                 y_pred_proba = softmax(y_pred, axis=1)
                 try:
                     # OvR стратегия
                     return roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                 except ValueError as e:
                      print(f"Предупреждение при вычислении ROC AUC (multi-class): {e}")
                      # Если есть только один класс в y_true, roc_auc_score падает
                      return 0.0 # Возвращаем 0 или другое значение по умолчанию

        elif metric == 'accuracy':
            if self.output_dim == 1:
                y_pred_class = (y_pred > 0).astype(int) # Используем 0 как порог для логитов
                return accuracy_score(y_true, y_pred_class)
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
                # Убедимся, что y_true одномерный
                if y_true.ndim > 1 and y_true.shape[1] == 1:
                     y_true = y_true.squeeze()
                return accuracy_score(y_true, y_pred_class)

        elif metric == 'mse': return mean_squared_error(y_true, y_pred)
        elif metric == 'mae': return mean_absolute_error(y_true, y_pred)
        elif metric == 'rmse': return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'r2': return r2_score(y_true, y_pred)
        else: raise ValueError(f"Неподдерживаемая метрика: {metric}")

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
        """Обучение модели TabNet.

        Параметры:
        -----------
        X : pandas.DataFrame
            Входные признаки.
        y : pandas.Series или list/numpy.ndarray
            Целевые значения.
        eval_set : tuple, optional (default=None)
            Кортеж (X_val, y_val) для валидации.
        eval_metric : str, optional (default=None)
            Метрика для мониторинга.
        mode : str, optional (default=None)
            Режим оптимизации ('max' или 'min').
        cat_features : list, optional (default=None)
            Список имен категориальных признаков. Если None, определяются автоматически.
        pbar : bool, optional (default=True)
            Отображать прогресс-бар.

        Возвращает:
        -----------
        self : объект
            Обученная модель.
        """
        eval_metric = eval_metric or self._get_default_eval_metric()
        mode = mode or self._get_default_metric_mode()

        if mode not in ['max', 'min']:
            raise ValueError("Параметр mode должен быть 'max' или 'min'")

        # Подготовка данных (вычисляет self.num_continuous, self.cat_dims, self.cat_idxs)
        train_dataset = self._prepare_data(X, y, is_train=True, cat_features=cat_features)

        val_dataset = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_dataset = self._prepare_data(X_val, y_val, is_train=False, cat_features=self.cat_features)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0), pin_memory=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=(self.num_workers > 0), pin_memory=True
            )

        # Инициализируем модель здесь, после _prepare_data
        if self.model is None:
            self.model = self._init_model()
            self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = self._get_criterion()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min' if mode == 'min' else 'max', # Режим зависит от метрики потерь валидации
            patience=self.reducelronplateau_patience,
            factor=self.reducelronplateau_factor,
            verbose=self.verbose
        )

        best_metric_val = float('inf') if mode == 'min' else float('-inf')
        no_improvement_epochs = 0
        best_model_state = None

        if self.verbose:
            print(f"Начинаем обучение на {self.device}...")
            # print(f"Параметры модели: {self.model}")

        for epoch in range(self.epochs):
            train_loss, train_outputs, train_targets = self._train_epoch(
                self.model, train_loader, optimizer, criterion, pbar=pbar
            )
            train_metric = self._calculate_metric(train_targets, train_outputs, eval_metric)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            val_loss, val_metric = None, None
            current_metric_val = -1 # Значение по умолчанию, если нет валидации

            if val_loader is not None:
                val_loss, val_outputs, val_targets = self._validate_epoch(self.model, val_loader, criterion, pbar=pbar)
                val_metric = self._calculate_metric(val_targets, val_outputs, eval_metric)
                current_metric_val = val_metric # Используем метрику на валидации для early stopping
                scheduler.step(val_loss) # Снижаем LR по val_loss
            else:
                current_metric_val = train_metric # Используем метрику на трейне для early stopping
                scheduler.step(train_loss) # Снижаем LR по train_loss

            if self.verbose:
                log_msg = (f"Epoch {epoch + 1}/{self.epochs}, "
                           f"Train loss: {train_loss:.4f}, Train {eval_metric}: {train_metric:.4f}")
                if val_loader is not None:
                    log_msg += f", Val loss: {val_loss:.4f}, Val {eval_metric}: {val_metric:.4f}"
                print(log_msg)

            improved = (mode == 'max' and current_metric_val > best_metric_val) or \
                       (mode == 'min' and current_metric_val < best_metric_val)

            if improved:
                best_metric_val = current_metric_val
                no_improvement_epochs = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                if self.verbose:
                    print(f"---> Сохранена лучшая модель (Эпоха {epoch + 1}) с Val {eval_metric}: {best_metric_val:.4f}")
            else:
                no_improvement_epochs += 1
                if self.verbose:
                    print(f"Нет улучшения {no_improvement_epochs}/{self.early_stopping_patience} эпох.")
                if no_improvement_epochs >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Ранняя остановка на эпохе {epoch + 1}.")
                    break

        del train_loader
        if val_loader is not None:
            del val_loader
        gc.collect()

        if best_model_state:
            # Перемещаем state_dict на нужное устройство перед загрузкой
            # for key in best_model_state:
            #     best_model_state[key] = best_model_state[key].to(self.device)
            self.model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"Загружена лучшая модель с Val {eval_metric}: {best_metric_val:.4f}")
        elif self.verbose:
             print("Обучение завершено без сохранения лучшей модели (возможно, не было валидации или улучшений).")


        self.is_fitted_ = True
        return self

    def predict(self, X, cat_features=None, pbar=True):
        self._check_is_fitted()
        if cat_features is None:
            cat_features = self.cat_features

        test_dataset = self._prepare_data(X, is_train=False, cat_features=cat_features)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0), pin_memory=True
        )

        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)
        del test_loader
        gc.collect()
        return self._transform_predictions(raw_predictions)


# ==========================================================================
# Подклассы для конкретных задач
# ==========================================================================

class TabNetBinary(TabNetEstimator):
    """TabNet для бинарной классификации."""
    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

    def _get_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def _get_default_eval_metric(self):
        return 'roc_auc'

    def _get_default_metric_mode(self):
        return 'max'

    def _evaluate_metrics(self, y_true, y_pred):
        y_pred_proba = sigmoid(y_pred)
        return roc_auc_score(y_true, y_pred_proba)

    def _transform_predictions(self, raw_predictions):
        probabilities = sigmoid(raw_predictions)
        return (probabilities > 0.5).astype(int).squeeze()

    def predict_proba(self, X, cat_features=None, pbar=True):
        """Предсказание вероятностей классов [P(0), P(1)]."""
        self._check_is_fitted()
        if cat_features is None:
            cat_features = self.cat_features

        test_dataset = self._prepare_data(X, is_train=False, cat_features=cat_features, is_multiclass=False)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0), pin_memory=True
        )

        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)
        del test_loader
        gc.collect()

        proba_1 = sigmoid(raw_predictions).squeeze()
        # Убедимся, что proba_1 всегда является массивом
        if isinstance(proba_1, (float, int)):
             proba_1 = np.array([proba_1])
        elif proba_1.ndim == 0: # Если это скалярный тензор numpy
             proba_1 = proba_1.reshape(1)

        proba_0 = 1 - proba_1
        return np.column_stack((proba_0, proba_1))


class TabNetMulticlass(TabNetEstimator):
    """TabNet для многоклассовой классификации."""
    def __init__(self, n_classes=None, **kwargs):
        if n_classes is None:
            raise ValueError("Для многоклассовой классификации необходимо указать 'n_classes'")
        self.n_classes = n_classes
        super().__init__(output_dim=n_classes, **kwargs)
        self.label_encoder = LabelEncoder()

    def _prepare_data(self, X, y=None, is_train=False, cat_features=None, is_multiclass=False):
        # Используем is_multiclass=True для CatEmbDataset
        return super()._prepare_data(X, y, is_train=is_train, cat_features=cat_features, is_multiclass=True)

    def _get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def _get_default_eval_metric(self):
        return 'accuracy'

    def _get_default_metric_mode(self):
        return 'max'

    def _evaluate_metrics(self, y_true, y_pred):
        predicted_classes = np.argmax(y_pred, axis=1)
        true_classes = y_true.squeeze()
        return accuracy_score(true_classes, predicted_classes)

    def _transform_predictions(self, raw_predictions):
        predicted_classes = np.argmax(raw_predictions, axis=1)
        # Декодируем метки, если энкодер обучен
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
             # Проверка на случай, если fit не был вызван
             try:
                 return self.label_encoder.inverse_transform(predicted_classes)
             except ValueError:
                  # Если классы не найдены (например, predict вызван до fit)
                  return predicted_classes
        else:
             return predicted_classes

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, cat_features=None, pbar=True):
        """Обучение модели с кодировкой меток классов."""
        # Кодируем y перед передачей в базовый fit
        encoded_y = self.label_encoder.fit_transform(y)
        self.n_classes = len(self.label_encoder.classes_)
        self.output_dim = self.n_classes # Обновляем output_dim на случай, если n_classes изменился

        encoded_eval_set = None
        if eval_set is not None:
            X_val, y_val = eval_set
            # Используем transform, а не fit_transform для валидационных меток
            try:
                 encoded_y_val = self.label_encoder.transform(y_val)
                 encoded_eval_set = (X_val, encoded_y_val)
            except ValueError as e:
                 print(f"Предупреждение: Не удалось преобразовать метки валидации: {e}. Валидация будет пропущена.")
                 encoded_eval_set = None # Пропускаем валидацию, если метки не совпадают

        # Вызываем fit базового класса с закодированными метками
        return super().fit(X, encoded_y, eval_set=encoded_eval_set, eval_metric=eval_metric, mode=mode, cat_features=cat_features, pbar=pbar)

    def predict_proba(self, X, cat_features=None, pbar=True):
        """Предсказание вероятностей для всех классов."""
        self._check_is_fitted()
        if cat_features is None:
            cat_features = self.cat_features

        test_dataset = self._prepare_data(X, is_train=False, cat_features=cat_features)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0), pin_memory=True
        )

        raw_predictions = self._get_predictions(self.model, test_loader, pbar=pbar)
        del test_loader
        gc.collect()

        # Применяем softmax к логитам
        return softmax(raw_predictions, axis=1)


class TabNetRegressor(TabNetEstimator):
    """TabNet для регрессии."""
    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

    def _get_criterion(self):
        return torch.nn.MSELoss()

    def _get_default_eval_metric(self):
        return 'mae' # Часто MAE более интерпретируема, чем MSE/RMSE

    def _get_default_metric_mode(self):
        return 'min'

    def _evaluate_metrics(self, y_true, y_pred):
        # По умолчанию возвращаем MAE для оценки
        return mean_absolute_error(y_true, y_pred)

    def _transform_predictions(self, raw_predictions):
        # Выход модели - это уже предсказанные значения
        return raw_predictions.squeeze()