from lightgbm import LGBMClassifier
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings

# Отключаем предупреждения от LightGBM
warnings.filterwarnings("ignore", category=UserWarning)

class LightGBMBinary:
    """
    Эстиматор LightGBM для бинарной классификации, совместимый с hp_tuning.
    """
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        min_child_weight: float = 0.001,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 50,
        metric: str = 'auc',
        verbose: int = -1,
        **kwargs
    ):
        """
        Инициализация модели LightGBM для бинарной классификации.

        Args:
            n_estimators: Максимальное число итераций (деревьев)
            learning_rate: Скорость обучения
            max_depth: Максимальная глубина дерева
            num_leaves: Максимальное число листьев
            min_child_samples: Минимальное число объектов в листе
            min_child_weight: Минимальный вес объектов в листе
            subsample: Доля выборки данных для каждого дерева
            subsample_freq: Частота подвыборки (0 = отключено)
            colsample_bytree: Доля признаков для каждого дерева
            reg_alpha: L1 регуляризация
            reg_lambda: L2 регуляризация
            random_state: Seed для генератора случайных чисел
            n_jobs: Число потоков (-1 = все доступные)
            early_stopping_rounds: Число итераций без улучшения для early stopping
            metric: Метрика для оценки и early stopping
            verbose: Уровень вывода информации (-1 = без вывода)
            **kwargs: Дополнительные параметры для LGBMClassifier
        """
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'early_stopping_rounds': early_stopping_rounds,
            'metric': metric,
            'verbose': verbose,
            'silent': True,  # Отключаем предупреждения от LightGBM
            'objective': 'binary'  # Для бинарной классификации
        }

        # Добавляем остальные параметры из kwargs
        self.params.update(kwargs)

        # Модель будет инициализирована в fit
        self.model = None
        self.feature_names = None

    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, cat_features=None, pbar=True):
        """
        Обучение модели LightGBM.

        Args:
            X: Матрица признаков для обучения
            y: Целевая переменная
            eval_set: Кортеж (X_val, y_val) для валидации
            eval_metric: Метрика для оценки (будет использована вместо self.params['metric'])
            mode: Режим оптимизации ('min' или 'max'), не используется для LightGBM
            cat_features: Индексы категориальных признаков
            pbar: Отображать ли прогресс-бар (True/False)

        Returns:
            self: Обученная модель
        """
        # Сохраняем имена признаков, если X - это DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        # Устанавливаем параметр verbose в зависимости от pbar
        # LightGBM использует verbose > 0 для вывода информации
        self.params['verbose'] = 1 if pbar else -1

        # Если передана другая метрика, используем её
        if eval_metric:
            self.params['metric'] = eval_metric

        # Создаем копию параметров без early_stopping_rounds для инициализации модели
        # model_params = {k: v for k, v in self.params.items() if k != 'early_stopping_rounds'}
        model_params = self.params

        # Инициализируем модель
        self.model = LGBMClassifier(**model_params)

        # Подготовка данных для валидации
        eval_set_data = None
        if eval_set:
            eval_set_data = [eval_set]

        # Получаем early_stopping_rounds
        # early_stopping = self.params.get('early_stopping_rounds', 0)

        # Обучаем модель
        self.model.fit(
            X, y,
            eval_set=eval_set_data,
            categorical_feature=cat_features,
            # early_stopping_rounds=early_stopping,
            # verbose=self.params['verbose'] > 0
        )

        return self

    def predict(self, X, cat_features=None, pbar=None):
        """
        Предсказание вероятностей класса 1.

        Args:
            X: Матрица признаков для предсказания
            cat_features: Индексы категориальных признаков (не используется в predict)
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
            cat_features: Индексы категориальных признаков (не используется в predict_proba)
            pbar: Отображать ли прогресс-бар (не используется в predict_proba)

        Returns:
            np.ndarray: Предсказанные вероятности всех классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit.")

        return self.model.predict_proba(X)
