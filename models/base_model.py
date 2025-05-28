# Базовый класс для всех моделей
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from metrics import Metrics, parse_metric_string
import joblib
import json
import os
import torch # Для проверки isinstance(..., torch.device)
from typing import Optional, List, Union # Добавляем Optional, List и Union
import glob  # Добавляем для поиска файлов при загрузке
import shutil


class BaseModel:
    def __init__(self, task, hp=None, metrics: Optional[List[Union[str, Metrics]]] = None, calibrate=None, n_folds=1, main_metric=None,
                 verbose=True, features=[], cat_features=[], target_name=None, index_cols=None):
        """
        Инициализация базовой модели.

        Args:
            task (str): Тип задачи ('binary', 'multiclass', 'regression')
            hp (dict): Гиперпараметры модели или None для использования значений по умолчанию
            metrics (Optional[List[Union[str, Metrics]]]): Список метрик для оценки модели (строки или объекты Metrics).
                                                            Строки должны быть в формате "name;param1=value1;param2=value2".
            calibrate (str): Метод калибровки ('betacal', 'isotonic') или None
            n_folds (int): Количество фолдов для кросс-валидации
            main_metric (str): Основная метрика для оптимизации
            verbose (bool): Выводить ли отладочную информацию
            features (list): Список всех признаков
            cat_features (list): Список категориальных признаков
            target_name (str): Имя целевого признака
            index_cols (list): Список индексных колонок
        """
        self.task = task
        self.hyperparameters = self.get_hyperparameters(task, hp)

        # Метрики храним в формате:
        # {model_type: {metric_name: Metrics}}
        self.metrics_list: dict = {}

        # Сохраняем тип модели в виде строки (например, CatBoostModel -> catboost)
        self._model_type_name: str = self.__class__.__name__.replace('Model', '').lower()

        # verbose должен быть определён до вызова _set_metrics
        self.verbose = verbose

        # Инициализируем метрики для текущей модели
        self._set_metrics(metrics, self._model_type_name)

        self.models = []  # Список моделей (для кросс-валидации)
        self.calibrate = calibrate
        self.calibration_model = None  # Одна модель калибровки для всех фолдов
        self.n_folds = n_folds
        self.main_metric = main_metric
        self.features = features
        self.cat_features = cat_features
        self.target_name = target_name
        self.index_cols = index_cols if index_cols is not None else []

        # Словарь для хранения рассчитанных значений метрик
        # {model_type: {metric_name: value}}
        self.metrics_values: dict = {}

    def _extract_X_y(self, data, require_target=False):
        """
        Вспомогательный метод для выделения X (фичи) и y (таргет) из датафрейма по self.features и self.target_name.
        Если require_target=True, выбрасывает ошибку при отсутствии таргета.
        """
        X = data[self.features]
        if require_target:
            if self.target_name not in data.columns:
                raise ValueError(f"В датафрейме отсутствует колонка таргета '{self.target_name}'")
            y = data[self.target_name]
            return X, y
        else:
            y = data[self.target_name] if self.target_name in data.columns else None
            return X, y

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
                X_fold_train, y_fold_train = self._extract_X_y(train.iloc[train_idx], require_target=True)
                X_fold_test, y_fold_test = self._extract_X_y(train.iloc[test_idx], require_target=True)
                model = train_method(X_fold_train, y_fold_train, X_fold_test, y_fold_test)
                self.evaluate(train.iloc[test_idx], model=model, fold_idx=fold_idx)
                self.models.append(model)
        else:
            X_train, y_train = self._extract_X_y(train, require_target=True)
            X_test, y_test = self._extract_X_y(test, require_target=True)
            model = train_method(X_train, y_train, X_test, y_test)
            self.models.append(model)
        if self.calibrate and self.task == 'binary':
            self._calibrate(test)

    def _predict(self, data, model=None):
        X, _ = self._extract_X_y(data, require_target=False)
        if self.task == 'binary':
            prediction = self._predict_fold_binary(model, X)
        elif self.task == 'multiclass':
            prediction = self._predict_fold_multiclass(model, X)
        elif self.task == 'regression':
            prediction = self._predict_fold_regression(model, X)
        else:
            raise ValueError(f"Неподдерживаемый тип задачи: {self.task}")
        return prediction

    def predict_cv(self, data):
        if not self.models:
            raise ValueError("Модель не обучена")
        
        X, _ = self._extract_X_y(data, require_target=False)
        
        predictions = []
        
        for model in self.models:
            predictions.append(self._predict(data, model))
        
        if self.task == 'multiclass':
            result = np.zeros_like(predictions[0])
            for pred in predictions:
                result += pred
            result /= len(predictions)
        else:
            result = np.zeros(len(X))
            for pred in predictions:
                result += pred
            result /= len(predictions)
        
        if self.calibrate and self.calibration_model and self.task == 'binary':
            result = self.calibration_model.predict(result)
        return result
    
    def predict(self, data):
        return self.predict_cv(data)

    def evaluate(self, data, model=None, fold_idx=None, metrics=None):
        # Если в этот вызов передали новый набор метрик, переопределяем текущие метрики
        if metrics is not None:
            self._set_metrics(metrics, self._model_type_name)
            # Флаг нужен для логики save()
            self._metrics_passed_in_evaluate = isinstance(metrics, list)

        X, y = self._extract_X_y(data, require_target=True)
        if model is None:
            y_pred = self.predict_cv(data)
        else:
            y_pred = self._predict(data, model)
        metrics_objects = self.get_metrics(self._model_type_name)
        metrics_values = {}
        for metric in metrics_objects:
            value = metric.calculate(y, y_pred)
            metrics_values[metric.metric_name_with_params] = value
        self.metrics_values[self._model_type_name] = metrics_values
        if self.verbose:
            if model is not None and fold_idx is not None and self.main_metric in metrics_values:
                print(f"Fold {fold_idx+1}: {self.main_metric}: {metrics_values[self.main_metric]:.4f}")
            else:
                for metric_name, metric_value in metrics_values.items():
                    print(f"   {metric_name}: {metric_value:.4f}")
        return metrics_values

    def get_hyperparameters(self, task, hp=None):
        """
        Возвращает гиперпараметры модели в зависимости от типа задачи

        Returns:
            dict: Гиперпараметры модели
        """
        if hp is None:
            if task == 'binary':
                return self._get_default_hp_binary()
            elif task == 'multiclass':
                return self._get_default_hp_multiclass()
            elif task == 'regression':
                return self._get_default_hp_regression()
        else:
            if task == 'binary':
                hp.update(self._get_required_hp_binary())
                return hp
            elif task == 'multiclass':
                hp.update(self._get_required_hp_multiclass())
                return hp
            elif task == 'regression':
                hp.update(self._get_required_hp_regression())
                return hp

    def get_metrics(self, model_type: Optional[str] = None):
        """
        Возвращает список объектов Metrics для указанного типа модели.

        Args:
            model_type (str, optional): Тип модели. Если None – используется текущий.

        Returns:
            List[Metrics]: список объектов метрик
        """
        model_type = model_type or self._model_type_name
        return list(self.metrics_list.get(model_type, {}).values())

    # Методы, которые должны быть реализованы в дочерних классах
    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Метод _train_fold_binary должен быть реализован в дочернем классе")

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Метод _train_fold_multiclass должен быть реализован в дочернем классе")

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Метод _train_fold_regression должен быть реализован в дочернем классе")

    def _predict_fold_binary(self, model, X):
        raise NotImplementedError("Метод _predict_binary_fold должен быть реализован в дочернем классе")

    def _predict_fold_multiclass(self, model, X):
        raise NotImplementedError("Метод _predict_multiclass_fold должен быть реализован в дочернем классе")

    def _predict_fold_regression(self, model, X):
        raise NotImplementedError("Метод _predict_regression_fold должен быть реализован в дочернем классе")

    # Специфичные для каждого типа задачи обязательные гиперпараметры
    def _get_required_hp_binary(self):
        return {}

    def _get_required_hp_multiclass(self):
        return {}

    def _get_required_hp_regression(self):
        return {}

    # Гиперпараметры по умолчанию, если в конфиге не указаны (use_custom_hyperparameters = false)
    def _get_default_hp_binary(self):
        return {}

    def _get_default_hp_multiclass(self):
        return {}

    def _get_default_hp_regression(self):
        return {}

    def save(self, save_path: str, model_type_name: str, metrics_to_save: Optional[Union[dict, List[Union[str, Metrics]]]] = None):
        """
        Сохраняет модель и все связанные артефакты в указанную директорию.
        Все артефакты (общие и специфичные для модели) сохраняются в подпапку,
        имя которой соответствует model_type_name, внутри исходного save_path.

        Общие артефакты (метаданные, калибровочная модель) сохраняются этим методом.
        Специфичные файлы моделей (например, pickle-файлы обученных фолдов или state_dict)
        сохраняются через вызов save_model_files, который должен быть
        реализован в дочерних классах.

        Args:
            save_path (str): Полный путь к БАЗОВОЙ директории для сохранения артефактов (без типа модели).
            model_type_name (str): Имя типа модели (например, 'catboost'), используется для создания подпапки.
            metrics_to_save (Optional[Union[dict, List[Union[str, Metrics]]]]):
                Словарь с метриками для сохранения или список метрик (строк или объектов Metrics),
                которые нужно сохранить. Если передается список, значения будут взяты из
                `self.last_evaluation_metrics` (если доступны), иначе будет использовано 'N/A'.
                Если None, будут использованы `self.last_evaluation_metrics` (если доступны)
                или метрики по умолчанию с N/A значениями.
        """
        # Создаем конечную директорию для ВСЕХ артефактов этой модели, включая тип модели в пути
        final_model_path = os.path.join(save_path, model_type_name)

        # Очищаем директорию, если она существует
        if os.path.exists(final_model_path):
            shutil.rmtree(final_model_path)

        os.makedirs(final_model_path, exist_ok=True)

        # 1. Сохраняем специфичные для модели файлы (фолды и т.д.) в эту конечную директорию
        self.save_model_files(final_model_path) # Передаем только конечный путь

        # 2. Сохраняем калибровочную модель, если она есть, в эту же конечную директорию
        if self.calibration_model is not None:
            calibration_file = os.path.join(final_model_path, "calibration_model.pickle")
            joblib.dump(self.calibration_model, calibration_file)

        # 3. Сохраняем гиперпараметры в эту же конечную директорию
        hyperparams_file = os.path.join(final_model_path, "hyperparameters.json")
        hyperparams_to_save_processed = {}
        if self.hyperparameters:
            for key, value in self.hyperparameters.items():
                if isinstance(value, torch.device):
                    hyperparams_to_save_processed[key] = str(value)
                else:
                    hyperparams_to_save_processed[key] = value
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams_to_save_processed, f, indent=4, default=lambda o: '<not serializable>')

        # 4. Сохраняем метрики в эту же конечную директорию
        metrics_file = os.path.join(final_model_path, "metrics.json")

        final_metrics_to_write_to_json = {}

        # Если в evaluate уже был передан список метрик, а сюда вновь прилетел список – игнорируем его
        if isinstance(metrics_to_save, list) and getattr(self, '_metrics_passed_in_evaluate', False):
            print("Предупреждение: список метрик из save() проигнорирован, так как он уже был передан в evaluate().")
            metrics_to_save = None

        # Обновляем список метрик, если передали новые (и не проигнорировали выше)
        if metrics_to_save is not None:
            self._set_metrics(metrics_to_save, model_type_name)

        if isinstance(metrics_to_save, dict):
            # Пользователь сам передал готовый словарь
            final_metrics_to_write_to_json = metrics_to_save
        elif isinstance(metrics_to_save, list):
            # Передали список, значения берём из ранее рассчитанных метрик
            for m in metrics_to_save:
                metric_name = m.metric_name_with_params if isinstance(m, Metrics) else m
                value = self.metrics_values.get(model_type_name, {}).get(metric_name, 'N/A')
                final_metrics_to_write_to_json[metric_name] = value
        else:  # metrics_to_save is None
            if model_type_name in self.metrics_values:
                final_metrics_to_write_to_json = self.metrics_values[model_type_name]
            else:
                metric_names = list(self.metrics_list.get(model_type_name, {}).keys())
                final_metrics_to_write_to_json = {m: 'N/A' for m in metric_names}

        metrics_serializable = {k: (v.item() if isinstance(v, np.generic) else v)
                                for k, v in final_metrics_to_write_to_json.items()}
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)

        # 5. Сохраняем список всех признаков в эту же конечную директорию
        features_file = os.path.join(final_model_path, "features.txt")
        with open(features_file, 'w') as f:
            if self.features:
                f.write('\n'.join(self.features))
            else:
                f.write('')


        # 6. Сохраняем список категориальных признаков в эту же конечную директорию
        cat_features_file = os.path.join(final_model_path, "cat_features.txt")
        with open(cat_features_file, 'w') as f:
            if self.cat_features:
                f.write('\n'.join(self.cat_features))
            else:
                f.write('')


        # 7. Сохраняем информацию о таргете, индексе и задаче в эту же конечную директорию
        target_info_file = os.path.join(final_model_path, "target_index_info.txt")
        with open(target_info_file, 'w') as f:
            f.write(f"Target column: {self.target_name}\n")
            f.write(f"Index columns: {', '.join(self.index_cols)}\n")
            f.write(f"Task type: {self.task}\n")
            f.write(f"Main metric: {self.main_metric}\n")
            f.write(f"Model type: {model_type_name}\n")
            f.write(f"Calibrate type: {self.calibrate}\n")
            f.write(f"Number of folds: {self.n_folds}\n")

        print(f"Модель {model_type_name} и артефакты сохранены в директорию: {final_model_path}")

    def save_model_files(self, save_path: str):
        """
        Сохраняет специфичные для данной модели файлы (например, обученные фолды)
        в указанную директорию. Этот метод должен быть реализован в дочерних классах.

        Args:
            save_path (str): Путь к директории, куда сохранять файлы (этот путь уже включает тип модели).
        """
        raise NotImplementedError(
            "Метод save_model_files должен быть реализован в дочернем классе"
        )

    def _calibrate(self, data):
        if self.task != 'binary':
            return
        print("Calibrating ...", end=" ")

        X, y = self._extract_X_y(data, require_target=True)
        proba_sum = np.zeros(len(X))

        for model in self.models:
            proba_sum += self._predict_fold_binary(model, X)
        proba = proba_sum / len(self.models)
        
        if self.calibrate == 'betacal':
            from betacal import BetaCalibration
            self.calibration_model = BetaCalibration()
        elif self.calibrate == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Неподдерживаемый метод калибровки: {self.calibrate}")
        
        # BetaCalibration / IsotonicRegression ожидают 1-мерный массив вероятностей
        # Используем 1-D, чтобы гарантировать корректную сериализацию и воспроизводимость
        self.calibration_model.fit(proba, y)
        print("done")

    # -------------------------
    # Методы работы с метриками
    # -------------------------
    def _set_metrics(self, metrics: Optional[Union[List[Union[str, Metrics]], dict]], model_type: str):
        """Создаёт и сохраняет словарь объектов Metrics по типу модели.

        Args:
            metrics: Список строк / объектов Metrics, dict с метриками и значениями
                           или None – тогда метрики выбираются по умолчанию исходя из self.task.
            model_type: строковый идентификатор модели (catboost, lightgbm и т.д.)

        Returns:
            dict: {metric_name: Metrics} для заданного model_type
        """
        # Шаг 1. Определяем список имён метрик
        metric_names: List[str] = []

        if metrics is None:
            # Выбираем метрики по умолчанию в зависимости от задачи
            if self.task == 'binary':
                metric_names = ['roc_auc', 'f1', 'accuracy']
            elif self.task == 'multiclass':
                metric_names = ['accuracy', 'f1', 'log_loss']
            elif self.task == 'regression':
                metric_names = ['rmse', 'mae', 'r2']
        elif isinstance(metrics, dict):
            # Если передали словарь – берём ключи как названия метрик
            metric_names = list(metrics.keys())
        else:
            # metrics – список
            for item in metrics:
                if isinstance(item, Metrics):
                    metric_names.append(item.metric_name_with_params)
                elif isinstance(item, str):
                    metric_names.append(item)
                else:
                    print(f"Предупреждение: Неподдерживаемый тип элемента в метриках: {type(item)} – игнорируем")

        # Шаг 2. Создаём объекты Metrics
        metrics_objects: dict = {}
        for name in metric_names:
            try:
                metrics_objects[name] = Metrics(name)
            except ValueError as e:
                print(f"Warning: {e}")

        # Шаг 3. Сохраняем
        self.metrics_list[model_type] = metrics_objects
        return metrics_objects

    # ------------------------------------------------------------------
    # Загрузка модели из директории с артефактами
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model_class_by_type(model_type: str):
        """Возвращает класс модели по её строковому идентификатору."""
        mapping = {
            'catboost': ('models.catboost_model', 'CatBoostModel'),
            'lightgbm': ('models.lightgbm_model', 'LightGBMModel'),
            'xgboost': ('models.xgboost_model', 'XGBoostModel'),
            'random_forest': ('models.random_forest_model', 'RandomForestModel'),
            'lightautoml': ('models.lightautoml_model', 'LightAutoMLModel'),
            'tabnet': ('models.tabnet_model', 'TabNetModel'),
            'cemlp': ('models.cemlp_model', 'CEMLPModel'),
        }
        if model_type not in mapping:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        module_name, class_name = mapping[model_type]
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)

    def load_model_files(self, load_path: str):
        """По умолчанию загружает fold_*_model.pickle в список self.models.

        Дочерние классы могут переопределить метод при иной схеме хранения."""
        pattern = os.path.join(load_path, "fold_*_model.pickle")
        fold_files = sorted(glob.glob(pattern))
        return [joblib.load(fp) for fp in fold_files]

    @staticmethod
    def load(load_path: str, verbose: bool = False):
        """Статический метод для восстановления модели из директории.

        Args:
            load_path (str): Путь к директории, содержащей артефакты (та же, что передавалась в save()).
            verbose (bool): Выводить ли информацию в процессе загрузки.

        Returns:
            BaseModel: полностью восстановленная модель.
        """
        if not os.path.isdir(load_path):
            raise FileNotFoundError(f"Директория {load_path} не существует")

        # 1. Читаем файл с мета-информацией
        info_file = os.path.join(load_path, "target_index_info.txt")
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"target_index_info.txt не найден в {load_path}")

        meta = {}
        with open(info_file, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    meta[key.strip()] = value.strip()

        model_type = meta.get("Model type")
        task = meta.get("Task type")
        main_metric = meta.get("Main metric")
        target_name = meta.get("Target column")
        index_cols = [col.strip() for col in meta.get("Index columns", "").split(',') if col.strip()]
        calibrate_type = meta.get("Calibrate type", None)

        # 2. Читаем остальные артефакты
        features_file = os.path.join(load_path, "features.txt")
        if os.path.exists(features_file):
            with open(features_file, "r") as f:
                features = [l.strip() for l in f if l.strip()]
        else:
            features = []

        cat_features_file = os.path.join(load_path, "cat_features.txt")
        if os.path.exists(cat_features_file):
            with open(cat_features_file, "r") as f:
                cat_features = [l.strip() for l in f if l.strip()]
        else:
            cat_features = []

        hyperparams_file = os.path.join(load_path, "hyperparameters.json")
        hp = None
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, "r") as f:
                hp = json.load(f)

        metrics_file = os.path.join(load_path, "metrics.json")
        metrics_values = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_values = json.load(f)

        calibration_file = os.path.join(load_path, "calibration_model.pickle")
        calibration_model = joblib.load(calibration_file) if os.path.exists(calibration_file) else None

        # 3. Определяем количество фолдов
        n_folds = len([name for name in os.listdir(load_path) if name.startswith("fold_")])
        n_folds = max(1, n_folds)  # безопасность

        # 4. Создаём экземпляр соответствующего класса модели
        model_class = BaseModel._get_model_class_by_type(model_type)

        model_args = dict(
            task=task,
            hp=hp,
            metrics=list(metrics_values.keys()) if metrics_values else None,
            calibrate=calibrate_type,
            n_folds=n_folds,
            main_metric=main_metric,
            verbose=verbose,
            features=features,
            cat_features=cat_features,
            target_name=target_name,
            index_cols=index_cols,
        )

        model_instance = model_class(**model_args)

        # 5. Загружаем модели фолдов/пайплайны
        model_instance.models = model_instance.load_model_files(load_path)

        # 6. Калибровка (если есть)
        model_instance.calibration_model = calibration_model
        if calibration_model is not None and not calibrate_type:
            c_name = calibration_model.__class__.__name__.lower()
            if 'beta' in c_name:
                model_instance.calibrate = 'betacal'
            elif 'isotonic' in c_name:
                model_instance.calibrate = 'isotonic'
            else:
                model_instance.calibrate = True  # произвольное истинное значение

        # 7. Восстанавливаем сохранённые значения метрик
        if metrics_values:
            model_instance.metrics_values[model_instance._model_type_name] = metrics_values
            # Чтобы обеспечить наличие объектов Metrics внутри модели
            model_instance._set_metrics(list(metrics_values.keys()), model_instance._model_type_name)

        print(f"Модель {model_type} успешно загружена из {load_path}")

        return model_instance
