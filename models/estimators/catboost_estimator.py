from catboost import CatBoostClassifier
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union


class CatBoostBinary:
    """
    Эстиматор CatBoost для бинарной классификации, совместимый с hp_tuning.
    """
    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        bootstrap_type: str = 'Bernoulli',
        subsample: Optional[float] = None,
        random_strength: float = 1.0,
        bagging_temperature: Optional[float] = None,
        grow_policy: str = 'SymmetricTree',
        min_data_in_leaf: int = 1,
        max_leaves: int = 31,
        early_stopping_rounds: int = 50,
        thread_count: int = -1,
        verbose: bool = False,
        eval_metric: str = 'AUC',
        **kwargs
    ):
        """
        Инициализация модели CatBoost для бинарной классификации.
        
        Args:
            iterations: Максимальное число итераций (деревьев)
            learning_rate: Скорость обучения
            depth: Глубина дерева
            l2_leaf_reg: Коэффициент L2 регуляризации
            random_seed: Seed для генератора случайных чисел
            bootstrap_type: Тип бутстрапа ('Bayesian', 'Bernoulli', 'MVS', 'Poisson')
            subsample: Коэффициент подвыборки для Bernoulli или Poisson
            random_strength: Сила случайности
            bagging_temperature: Температура бэггинга для Bayesian
            grow_policy: Политика роста дерева ('SymmetricTree', 'Depthwise', 'Lossguide')
            min_data_in_leaf: Минимальное число объектов в листе
            max_leaves: Максимальное число листьев
            early_stopping_rounds: Число итераций без улучшения для early stopping
            thread_count: Число потоков (-1 = все доступные)
            verbose: Выводить ли прогресс обучения
            eval_metric: Метрика для оценки и early stopping
            **kwargs: Дополнительные параметры для CatBoostClassifier
        """
        self.params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            'random_seed': random_seed,
            'bootstrap_type': bootstrap_type,
            'random_strength': random_strength,
            'grow_policy': grow_policy,
            'min_data_in_leaf': min_data_in_leaf,
            'max_leaves': max_leaves,
            'early_stopping_rounds': early_stopping_rounds,
            'thread_count': thread_count,
            'verbose': verbose,
            'eval_metric': eval_metric,
        }
        
        # Добавляем условные параметры
        if bootstrap_type in ['Bernoulli', 'Poisson'] and subsample is not None:
            self.params['subsample'] = subsample
        if bootstrap_type == 'Bayesian' and bagging_temperature is not None:
            self.params['bagging_temperature'] = bagging_temperature
            
        # Добавляем остальные параметры из kwargs
        self.params.update(kwargs)
        
        # Модель будет инициализирована в fit
        self.model = None
        self.feature_names = None
        
    def fit(self, X, y, eval_set=None, eval_metric=None, mode=None, cat_features=None, pbar=True):
        """
        Обучение модели CatBoost.
        
        Args:
            X: Матрица признаков для обучения
            y: Целевая переменная
            eval_set: Кортеж (X_val, y_val) для валидации
            eval_metric: Метрика для оценки (будет использована вместо self.params['eval_metric'])
            mode: Режим оптимизации ('min' или 'max'), не используется для CatBoost
            cat_features: Индексы категориальных признаков
            pbar: Отображать ли прогресс-бар (True/False)
            
        Returns:
            self: Обученная модель
        """
        # Сохраняем имена признаков, если X - это DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        # Устанавливаем параметр verbose в зависимости от pbar
        self.params['verbose'] = False if not pbar else self.params.get('verbose', False)
        
        # Если передана другая метрика, используем её
        if eval_metric:
            self.params['eval_metric'] = eval_metric
            
        # Инициализируем модель
        self.model = CatBoostClassifier(**self.params)
        
        # Подготовка данных для валидации
        eval_set_data = None
        if eval_set:
            eval_set_data = [eval_set]
        
        # Обучаем модель
        self.model.fit(
            X, y,
            eval_set=eval_set_data,
            cat_features=cat_features,
            verbose=self.params['verbose']
        )
        
        return self
    
    def predict(self, X, cat_features=None, pbar=None):
        """
        Предсказание вероятностей классов.
        
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