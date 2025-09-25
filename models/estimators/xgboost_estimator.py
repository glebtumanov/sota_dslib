from xgboost import XGBClassifier
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings

# Отключаем предупреждения
warnings.filterwarnings("ignore", category=UserWarning)

class XGBoostBinary:
    """
    Эстиматор XGBoost для бинарной классификации, совместимый с hp_tuning.
    """
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 50,
        eval_metric: str = 'auc',
        verbosity: int = 0,
        **kwargs
    ):
        """
        Инициализация модели XGBoost для бинарной классификации.

        Args:
            n_estimators: Максимальное число итераций (деревьев)
            learning_rate: Скорость обучения (eta)
            max_depth: Максимальная глубина дерева
            min_child_weight: Минимальный вес объектов в листе
            gamma: Минимальное уменьшение потерь для дальнейшего разделения
            subsample: Доля выборки данных для каждого дерева
            colsample_bytree: Доля признаков для каждого дерева
            colsample_bylevel: Доля признаков для каждого уровня дерева
            colsample_bynode: Доля признаков для каждого узла
            reg_alpha: L1 регуляризация
            reg_lambda: L2 регуляризация
            scale_pos_weight: Вес положительного класса
            random_state: Seed для генератора случайных чисел
            n_jobs: Число потоков (-1 = все доступные)
            early_stopping_rounds: Число итераций без улучшения для early stopping
            eval_metric: Метрика для оценки и early stopping
            verbosity: Уровень вывода информации (0 = без вывода)
            **kwargs: Дополнительные параметры для XGBClassifier
        """
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'early_stopping_rounds': early_stopping_rounds,
            'eval_metric': eval_metric,
            'verbosity': verbosity,
            'objective': 'binary:logistic'  # Для бинарной классификации
        }

        # Добавляем остальные параметры из kwargs
        self.params.update(kwargs)

        # Модель будет инициализирована в fit
        self.model = None
        self.feature_names = None

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, cat_features=None, pbar=True):
        """
        Обучение модели XGBoost.

        Args:
            X: Матрица признаков для обучения
            y: Целевая переменная
            eval_set: Кортеж (X_val, y_val) для валидации
            eval_metric: Метрика для оценки (будет использована вместо self.params['eval_metric'])
            mode: Режим оптимизации ('min' или 'max'), не используется для XGBoost
            cat_features: Индексы категориальных признаков (не используется напрямую в XGBoost)
            pbar: Отображать ли прогресс-бар (True/False)

        Returns:
            self: Обученная модель
        """
        # Сохраняем имена признаков, если X - это DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        # Устанавливаем параметр verbosity в зависимости от pbar
        self.params['verbosity'] = 1 if pbar else 0

        # Если передана другая метрика, используем её
        if eval_metric:
            self.params['eval_metric'] = eval_metric

        # Создаем копию параметров без early_stopping_rounds для инициализации модели
        # model_params = {k: v for k, v in self.params.items() if k != 'early_stopping_rounds'}
        model_params = self.params

        # Инициализируем модель
        self.model = XGBClassifier(**model_params)

        # Подготовка данных для валидации
        eval_set_data = None
        if eval_set:
            eval_set_data = [eval_set]

        # Получаем early_stopping_rounds
        # early_stopping = self.params.get('early_stopping_rounds', 0)

        # Обработка категориальных признаков для XGBoost
        # Для XGBoost необходимо преобразовать категориальные признаки в one-hot encoding
        # В данном случае мы предполагаем, что это уже сделано или будет обработано на этапе предобработки

        # Обучаем модель
        self.model.fit(
            X, y,
            eval_set=eval_set_data,
            # early_stopping_rounds=early_stopping,
            verbose=self.params['verbosity'] > 0
        )

        return self

    def predict(self, X, cat_features=None, pbar=None):
        """
        Предсказание вероятностей класса 1.

        Args:
            X: Матрица признаков для предсказания
            cat_features: Индексы категориальных признаков (не используется в XGBoost)
            pbar: Отображать ли прогресс-бар (не используется в predict)

        Returns:
            np.ndarray: Предсказанные вероятности класса 1
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit.")

        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X, cat_features=None, pbar=None):
        """
        Предсказание вероятностей классов.

        Args:
            X: Матрица признаков для предсказания
            cat_features: Индексы категориальных признаков (не используется в XGBoost)
            pbar: Отображать ли прогресс-бар (не используется в predict_proba)

        Returns:
            np.ndarray: Предсказанные вероятности всех классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit.")

        return self.model.predict_proba(X)
