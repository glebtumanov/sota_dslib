# Базовый класс для всех моделей
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from metrics import Metrics, parse_metric_string
import abc


class BaseModel(abc.ABC):
    """
    Абстрактный базовый класс для всех моделей.
    Определяет общий интерфейс и реализует общую логику.
    """
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        """
        Инициализация базовой модели.

        Args:
            task (str): Тип задачи ('binary', 'multiclass', 'regression')
            hp (dict): Гиперпараметры модели или None для использования значений по умолчанию
            metrics (list): Список метрик для оценки модели
            calibrate (str): Метод калибровки ('betacal', 'isotonic') или None
            n_folds (int): Количество фолдов для кросс-валидации
            main_metric (str): Основная метрика для оптимизации
            verbose (bool): Выводить ли отладочную информацию
            features (list): Список всех признаков
            cat_features (list): Список категориальных признаков
            target_name (str): Имя целевого признака
        """
        self.task = task
        self.hp = hp if hp is not None else {}
        self.metrics = metrics if metrics is not None else []
        self.calibrate = calibrate
        self.n_folds = n_folds
        self.main_metric = main_metric
        self.verbose = verbose
        self.features = features
        self.cat_features = cat_features if cat_features is not None else []
        self.target_name = target_name
        self.models = []
        self.oof_predictions = None
        self.test_predictions = None
        self.feature_importances = None

        # Определяем гиперпараметры (объединяем пользовательские и дефолтные)
        self._resolve_hyperparameters()

    def _resolve_hyperparameters(self):
        """Определяет финальный набор гиперпараметров."""
        default_hp = self._get_default_hp_for_task()
        required_hp = self._get_required_hp_for_task()

        # Объединяем: дефолтные < "обязательные" (специфичные для модели) < пользовательские
        self.hyperparameters = {**default_hp, **required_hp, **self.hp}

    def _prepare_estimator_kwargs(self, specific_kwargs=None):
        """
        Подготавливает словарь keyword-аргументов для конструктора эстиматора.

        Объединяет self.hyperparameters со specific_kwargs и добавляет
        self.cat_features, если 'cat_features' отсутствует.

        Args:
            specific_kwargs (dict, optional): Словарь специфичных для задачи
                                              аргументов (например, {'n_classes': N}).
                                              Defaults to None.

        Returns:
            dict: Финальный словарь аргументов для конструктора эстиматора.
        """
        estimator_kwargs = self.hyperparameters.copy()

        if specific_kwargs:
            estimator_kwargs.update(specific_kwargs)

        if 'cat_features' not in estimator_kwargs and self.cat_features:
            estimator_kwargs['cat_features'] = self.cat_features

        return estimator_kwargs

    def _get_default_hp_for_task(self):
        """Возвращает дефолтные гиперпараметры для текущей задачи."""
        if self.task == 'binary':
            return self._get_default_hp_binary()
        elif self.task == 'multiclass':
            return self._get_default_hp_multiclass()
        elif self.task == 'regression':
            return self._get_default_hp_regression()
        else:
            raise ValueError(f"Неизвестный тип задачи: {self.task}")

    def _get_required_hp_for_task(self):
        """Возвращает обязательные гиперпараметры для текущей задачи."""
        if self.task == 'binary':
            return self._get_required_hp_binary()
        elif self.task == 'multiclass':
            return self._get_required_hp_multiclass()
        elif self.task == 'regression':
            return self._get_required_hp_regression()
        else:
            raise ValueError(f"Неизвестный тип задачи: {self.task}")

    def train(self, train, test):
        self.models = []

        train_methods = {
            'binary': self._train_fold_binary,
            'multiclass': self._train_fold_multiclass,
            'regression': self._train_fold_regression
        }

        train_method = train_methods[self.task]

        if self.n_folds > 1:
            if self.task == 'regression':
                kf = KFold(n_splits=self.n_folds, random_state=42, shuffle=True)
                split_indices = kf.split(train[self.features])
            else:
                kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
                split_indices = kf.split(train[self.features], train[self.target_name])

            for fold_idx, (train_idx, test_idx) in enumerate(split_indices):
                X_fold_train = train[self.features].iloc[train_idx]
                y_fold_train = train[self.target_name].iloc[train_idx]
                X_fold_test = train[self.features].iloc[test_idx]
                y_fold_test = train[self.target_name].iloc[test_idx]

                model = train_method(X_fold_train,
                                     y_fold_train,
                                     X_fold_test,
                                     y_fold_test)
                self.evaluate(X_fold_test, y_fold_test, model, fold_idx=fold_idx)
                self.models.append(model)

        else:
            model = train_method(train[self.features],
                                 train[self.target_name],
                                 test[self.features],
                                 test[self.target_name])
            self.models.append(model)

        # print("Метрики на тестовых данных (holdout):")
        # self.evaluate(test[self.features], test[self.target_name])

        if self.calibrate and self.task == 'binary':
            self._calibrate(test[self.features], test[self.target_name])

    def _predict(self, X, model):
        if self.task == 'binary':
            prediction = self._predict_fold_binary(model, X)
        elif self.task == 'multiclass':
            prediction = self._predict_fold_multiclass(model, X)
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
        if self.task == 'multiclass':
            # Для multiclass усредняем вероятности по классам
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

    def get_metrics(self):
        """
        Возвращает список объектов Metrics на основе списка строк метрик.

        Returns:
            List[Metrics]: Список экземпляров класса Metrics
        """
        metrics = []
        for metric_item in self.metrics:
            try:
                metrics.append(Metrics(metric_item))
            except ValueError as e:
                print(f"Warning: {e}")

        return metrics

    # Методы, которые должны быть реализованы в дочерних классах
    @abc.abstractmethod
    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        pass

    @abc.abstractmethod
    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        pass

    @abc.abstractmethod
    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        pass

    @abc.abstractmethod
    def _predict_fold_binary(self, model, X):
        pass

    @abc.abstractmethod
    def _predict_fold_multiclass(self, model, X):
        pass

    @abc.abstractmethod
    def _predict_fold_regression(self, model, X):
        pass

    @abc.abstractmethod
    def _get_required_hp_binary(self):
        pass

    @abc.abstractmethod
    def _get_required_hp_multiclass(self):
        pass

    @abc.abstractmethod
    def _get_required_hp_regression(self):
        pass

    @abc.abstractmethod
    def _get_default_hp_binary(self):
        pass

    @abc.abstractmethod
    def _get_default_hp_multiclass(self):
        pass

    @abc.abstractmethod
    def _get_default_hp_regression(self):
        pass

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
