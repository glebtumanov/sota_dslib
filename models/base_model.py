# Базовый класс для всех моделей
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from metrics import Metrics, parse_metric_string

class BaseModel:
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None, verbose=True):
        """
        Инициализация базовой модели.

        Args:
            task (str): Тип задачи ('binary', 'multi', 'regression')
            hp (dict): Гиперпараметры модели или None для использования значений по умолчанию
            metrics (list): Список метрик для оценки модели
            calibrate (str): Метод калибровки ('betacal', 'isotonic') или None
            n_folds (int): Количество фолдов для кросс-валидации
            main_metric (str): Основная метрика для оптимизации
            verbose (bool): Выводить ли отладочную информацию
        """
        self.task = task
        self.hp = hp
        self.metrics_list = metrics or []
        self.models = []  # Список моделей (для кросс-валидации)
        self.calibrate = calibrate
        self.calibration_model = None  # Одна модель калибровки для всех фолдов
        self.n_folds = n_folds
        self.main_metric = main_metric
        self.verbose = verbose

    def train(self, train, test, target, features, cat_features=[]):
        params = self.get_hyperparameters()
        self.models = []

        train_methods = {
            'binary': self._train_fold_binary,
            'multi': self._train_fold_multi,
            'regression': self._train_fold_regression
        }

        train_method = train_methods[self.task]

        if self.n_folds > 1:
            if self.task == 'regression':
                kf = KFold(n_splits=self.n_folds, random_state=42, shuffle=True)
                split_indices = kf.split(train[features])
            else:
                kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
                split_indices = kf.split(train[features], train[target])

            for fold_idx, (train_idx, test_idx) in enumerate(split_indices):
                X_fold_train = train[features].iloc[train_idx]
                y_fold_train = train[target].iloc[train_idx]
                X_fold_test = train[features].iloc[test_idx]
                y_fold_test = train[target].iloc[test_idx]

                model = train_method(X_fold_train, y_fold_train, X_fold_test, y_fold_test,
                                     params, cat_features)
                self.evaluate(X_fold_test, y_fold_test, model, fold_idx=fold_idx)
                self.models.append(model)

        else:
            model = train_method(train[features], train[target], test[features], test[target],
                                 params, cat_features)
            self.models.append(model)

        print("Метрики на тестовых данных (holdout):")
        self.evaluate(test[features], test[target])

        if self.calibrate and self.task == 'binary':
            self._calibrate(test[features], test[target])

    def _predict(self, X, model):
        if self.task == 'binary':
            prediction = self._predict_fold_binary(model, X)
        elif self.task == 'multi':
            prediction = self._predict_fold_multi(model, X)
        elif self.task == 'regression':
            prediction = self._predict_fold_regression(model, X)
        else:
            raise ValueError(f"Неподдерживаемый тип задачи: {self.task}")

        return prediction

    def predict_cv(self, X):
        if not self.models:
            raise ValueError("Модель не обучена")

        # Получаем предсказания для каждой модели
        predictions = []
        for model in self.models:
            predictions.append(self._predict(X, model))

        # Усредняем предсказания по всем моделям
        if self.task == 'multi':
            # Для multi усредняем вероятности по классам
            result = np.zeros_like(predictions[0])
            for pred in predictions:
                result += pred
            result /= len(predictions)
        else:
            # Для binary и regression просто усредняем предсказания
            result = np.zeros(len(X))
            for pred in predictions:
                result += pred
            result /= len(predictions)

        # Применяем калибровку, если она используется
        if self.calibrate and self.calibration_model and self.task == 'binary':
            result = self.calibration_model.predict(result.reshape(-1, 1))

        return result

    def evaluate(self, X, y, model=None, fold_idx=None):
        # Если модель не указана, используем усредненные предсказания по всем моделям
        # Если модель указана, используем предсказания конкретной модели

        if model is None:
            y_pred = self.predict_cv(X)
        else:
            y_pred = self._predict(X, model)

        metrics_list = self.get_metrics()
        metrics_dict = {}

        for metric in metrics_list:
            value = metric.calculate(y, y_pred)
            metrics_dict[metric.metric_name_with_params] = value

        if model is not None and fold_idx is not None:
            print(f"Fold {fold_idx+1}: {self.main_metric}: {metrics_dict[self.main_metric]:.4f}")
        else:
            for metric_name, metric_value in metrics_dict.items():
                print(f"   {metric_name}: {metric_value:.4f}")

        return metrics_dict

    def get_hyperparameters(self):
        """
        Возвращает гиперпараметры модели в зависимости от типа задачи

        Returns:
            dict: Гиперпараметры модели
        """
        if self.hp is None:
            if self.task == 'binary':
                return self._get_default_hp_binary()
            elif self.task == 'multi':
                return self._get_default_hp_multi()
            elif self.task == 'regression':
                return self._get_default_hp_regression()
        else:
            if self.task == 'binary':
                self.hp.update(self._get_required_hp_binary())
                return self.hp
            elif self.task == 'multi':
                self.hp.update(self._get_required_hp_multi())
                return self.hp
            elif self.task == 'regression':
                self.hp.update(self._get_required_hp_regression())
                return self.hp

    def get_metrics(self):
        """
        Возвращает список объектов Metrics на основе списка строк метрик.

        Returns:
            List[Metrics]: Список экземпляров класса Metrics
        """
        from metrics import TASK_METRICS

        # Получаем список доступных метрик для данного типа задачи
        available_metrics = TASK_METRICS.get(self.task, set())

        metrics = []
        for metric_item in self.metrics_list:
            try:
                metrics.append(Metrics(metric_item))
            except ValueError as e:
                print(f"Warning: {e}")

        return metrics

    # Методы, которые должны быть реализованы в дочерних классах
    def _train_fold_binary(self, X_train, y_train, X_test, y_test, params, cat_features):
        raise NotImplementedError("Метод _train_fold_binary должен быть реализован в дочернем классе")

    def _train_fold_multi(self, X_train, y_train, X_test, y_test, params, cat_features):
        raise NotImplementedError("Метод _train_fold_multi должен быть реализован в дочернем классе")

    def _train_fold_regression(self, X_train, y_train, X_test, y_test, params, cat_features):
        raise NotImplementedError("Метод _train_fold_regression должен быть реализован в дочернем классе")

    def _predict_fold_binary(self, model, X):
        raise NotImplementedError("Метод _predict_binary_fold должен быть реализован в дочернем классе")

    def _predict_fold_multi(self, model, X):
        raise NotImplementedError("Метод _predict_multi_fold должен быть реализован в дочернем классе")

    def _predict_fold_regression(self, model, X):
        raise NotImplementedError("Метод _predict_regression_fold должен быть реализован в дочернем классе")

    # Специфичные для каждого типа задачи обязательные гиперпараметры
    def _get_required_hp_binary(self):
        return {}

    def _get_required_hp_multi(self):
        return {}

    def _get_required_hp_regression(self):
        return {}

    # Гиперпараметры по умолчанию, если в конфиге не указаны (use_custom_hyperparameters = false)
    def _get_default_hp_binary(self):
        return {}

    def _get_default_hp_multi(self):
        return {}

    def _get_default_hp_regression(self):
        return {}

    def save(self, path):
        """
        Сохраняет модель в файл

        Args:
            path: путь для сохранения модели
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _calibrate(self, X, y):
        """
        Калибрует вероятности модели с использованием указанного метода калибровки.
        Калибровка применяется к усредненным скорам со всех моделей.
        Поддерживаются методы 'betacal' и 'isotonic'.

        Args:
            X: Данные для калибровки
            y: Целевые значения
        """
        if self.task != 'binary':
            return

        print("Calibrating ...", end=" ")
        # Получаем усредненные некалиброванные вероятности
        # Используем реализацию от конкретного типа модели
        proba_sum = np.zeros(len(X))
        for model in self.models:
            proba_sum += self._predict_fold_binary(model, X)
        proba = proba_sum / len(self.models)

        # Выбираем и обучаем модель калибровки
        if self.calibrate == 'betacal':
            from betacal import BetaCalibration
            self.calibration_model = BetaCalibration()
        elif self.calibrate == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Неподдерживаемый метод калибровки: {self.calibrate}")

        self.calibration_model.fit(proba.reshape(-1, 1), y)
        print("done")
