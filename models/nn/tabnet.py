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
warnings.filterwarnings('ignore')

# Импортируем CatEmbDataset из нового модуля
from models.nn.dataset import CatEmbDataset


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


class TabNetPytorch(nn.Module):
    """Архитектура TabNet нейронной сети."""
    def __init__(self,
                 input_dim,
                 cat_idxs,
                 cat_dims,
                 cat_emb_dim=4,
                 n_steps=5,
                 hidden_dim=128,
                 decision_dim=64,
                 n_glu_layers=2,
                 dropout=0.1,
                 gamma=1.5,
                 lambda_sparse=0.0001,
                 virtual_batch_size=128,
                 momentum=0.9,
                 output_dim=1):
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
                    dropout,
                    virtual_batch_size,
                    momentum
                )
            )

            if step < n_steps - 1:
                self.attentive_transformers.append(
                    AttentiveTransformer(
                        hidden_dim,
                        self.post_embed_dim,
                        virtual_batch_size,
                        momentum
                    )
                )

        self.decision_layer = nn.Linear(hidden_dim, decision_dim)
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
                embedded_cats.append(self.embeddings[i](cat_values))

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

            # Считаем выход текущего шага
            step_output = F.relu(self.decision_layer(x_transformed))
            outputs += self.final_layer(step_output) / self.n_steps

            # Обновляем маску для следующего шага
            if step < self.n_steps - 1:
                mask = self.attentive_transformers[step](x_transformed)
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


class TabNet(BaseEstimator):
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

    hidden_dim : int, default=16
        Размерность скрытого слоя.
        Рекомендуемый диапазон: [8-128]

    decision_dim : int, default=8
        Размерность решающего слоя. Обычно меньше hidden_dim.
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

    В модуле представлены три специализированных класса:
    - TabNetBinaryClassifier: для бинарной классификации
    - TabNetMulticlassClassifier: для многоклассовой классификации
    - TabNetRegressor: для задач регрессии

    Примеры
    --------
    >>> from models.nn.tabnet import TabNetBinaryClassifier
    >>> model = TabNetBinaryClassifier(
    ...     hidden_dim=32,
    ...     n_steps=5,
    ...     dropout=0.3,
    ...     scale_numerical=True,
    ...     scale_method="standard"
    ... )
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric='roc_auc')
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)
    """

    def __init__(self,
                 cat_emb_dim=6,  # Размерность эмбеддингов для категориальных признаков
                 n_steps=4,  # Количество шагов в TabNet
                 hidden_dim=16,  # Размерность скрытого слоя
                 decision_dim=8,  # Размерность решающего слоя
                 n_glu_layers=3,  # Количество GLU слоев
                 dropout=0.1,  # Вероятность дропаута
                 gamma=1.5,  # Коэффициент затухания для масок внимания
                 lambda_sparse=0.0001,  # Коэффициент регуляризации разреженности
                 virtual_batch_size=128,  # Размер виртуального батча для Ghost BatchNorm
                 momentum=0.9,  # Параметр momentum для BatchNorm
                 batch_size=1024,  # Размер батча для обучения
                 epochs=50,  # Количество эпох обучения
                 learning_rate=0.01,  # Скорость обучения
                 early_stopping_patience=5,  # Количество эпох без улучшения до остановки
                 weight_decay=1e-5,  # Весовая регуляризация для оптимизатора
                 scale_numerical=True,  # Масштабировать ли числовые признаки
                 scale_method="standard",  # Метод масштабирования ("standard", "minmax", "quantile", "binning")
                 n_bins=10,  # Количество бинов для binning
                 device=None,  # Устройство для обучения (cuda/cpu)
                 output_dim=1,  # Размерность выходного слоя
                 verbose=True,  # Вывод прогресса обучения
                 num_workers=0,  # Количество worker-процессов для DataLoader (0 - однопроцессный режим)
                 random_state=None):  # Случайное состояние для воспроизводимости

        self.cat_emb_dim = cat_emb_dim
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
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

    def _prepare_data(self, X, y=None, is_train=False, is_multiclass=False):
        """Подготовка данных для обучения/предсказания

        :param X: Входные признаки (DataFrame или ndarray)
        :param y: Целевые значения (если есть)
        :param is_train: Флаг обучающего набора (для обучения скейлера)
        :param is_multiclass: Флаг многоклассовой классификации
        :return: Датасет CatEmbDataset
        """

        if isinstance(X, pd.DataFrame):
            # Если X - это DataFrame, извлекаем признаки
            features = X.columns.tolist()

            # Определяем категориальные признаки
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
                    scaler=None if is_train else self.scaler,  # Используем None для обучения, иначе - готовый скейлер
                    is_multiclass=is_multiclass  # Передаем флаг multiclass
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
                    is_multiclass=is_multiclass  # Передаем флаг multiclass
                )

            self.features = features
            self.cat_features = cat_features
            self.cat_idxs = cat_idxs
            self.cat_dims = cat_dims

        else:
            # Если X - это numpy array, используем существующие атрибуты
            if self.features is None or self.cat_features is None:
                raise ValueError("При использовании numpy array необходимо предварительно обучить модель на DataFrame")

            # Создаем DataFrame из numpy array
            X_df = pd.DataFrame(X, columns=self.features)

            if y is not None:
                y_df = pd.DataFrame({'target': y})
                df = pd.concat([X_df, y_df], axis=1)
                dataset = CatEmbDataset(
                    df, self.features, self.cat_features,
                    target_col='target',
                    scale_numerical=self.scale_numerical,
                    scale_method=self.scale_method,
                    n_bins=self.n_bins,
                    scaler=None if is_train else self.scaler,
                    is_multiclass=is_multiclass  # Передаем флаг multiclass
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
                    is_multiclass=is_multiclass  # Передаем флаг multiclass
                )

        return dataset

    def _init_model(self, input_dim):
        """Инициализация модели TabNet"""
        return TabNetPytorch(
            input_dim=input_dim,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_steps=self.n_steps,
            hidden_dim=self.hidden_dim,
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
        """Обучение на одной эпохе"""
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
        """Валидация на одной эпохе"""
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
        """Обучение модели TabNet

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
            optimizer, 'min', patience=2, factor=0.5
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
                print(f"Epoch {epoch + 1}/{self.epochs}")
                print(f"Train loss: {train_loss:.4f}, Train {eval_metric}: {train_metric:.4f}")
                if val_loader is not None:
                    print(f"Val loss: {val_loss:.4f}, Val {eval_metric}: {val_metric:.4f}")

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
                best_model_state = self.model.state_dict().copy()
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

    def _check_is_fitted(self):
        """Проверка, что модель обучена"""
        if not self.is_fitted_:
            raise ValueError("Модель не обучена. Сначала выполните метод 'fit'.")


class TabNetBinaryClassifier(TabNet):
    """TabNet для бинарной классификации.

    Реализация TabNet для задач бинарной классификации.
    Подробности о параметрах см. в документации базового класса TabNet.

    Дополнительные методы:
    ----------------------
    predict_proba(X) : возвращает вероятности классов
    """

    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)

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

        # Преобразуем логиты в вероятности
        proba_1 = 1 / (1 + np.exp(-raw_predictions)).squeeze()

        # Для scikit-learn API нужно возвращать вероятности для обоих классов
        proba_0 = 1 - proba_1

        return np.column_stack((proba_0, proba_1))


class TabNetMulticlassClassifier(TabNet):
    """TabNet для многоклассовой классификации.

    Реализация TabNet для задач многоклассовой классификации.
    Подробности о параметрах см. в документации базового класса TabNet.

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

class TabNetRegressor(TabNet):
    """TabNet для регрессии.

    Реализация TabNet для задач регрессии.
    Подробности о параметрах см. в документации базового класса TabNet.
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


def softmax(x, axis=None):
    """Вычисление softmax по указанной оси"""
    # Вычитаем максимум для численной стабильности
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    # Нормализуем
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)