import os
import yaml
import pandas as pd
import numpy as np
import joblib
import json
import zipfile
import tempfile
import glob
import time
import torch
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from metrics import get_best_model_by_metric, METRIC_DIRECTIONS, MAXIMIZE, MINIMIZE
from tabulate import tabulate

class SOTAModels:
    def __init__(self, config_path):
        # Загружаем конфиг
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Сохраняем имя модели
        common_config = self.config.get('common', {})
        self.model_name = common_config.get('name')
        if not self.model_name:
            # Если имя не указано, используем имя файла конфига без расширения
            self.model_name = os.path.splitext(os.path.basename(config_path))[0]

        # Общие параметры
        self.task = common_config.get('task', 'binary')
        self.main_metric = common_config.get('main_metric', 'roc_auc')
        self.metrics_list = common_config.get('metrics', [])
        self.model_dir = common_config.get('model_dir', './models')
        self.skip_cols = common_config.get('skip_cols', [])
        self.selected_models = common_config.get('selected_models', ['catboost'])

        # Параметры разделения данных
        split_config = self.config.get('split_data', {})
        self.test_rate = split_config.get('test_rate', 0.2)
        self.validation_rate = split_config.get('validation_rate', 0.2)
        self.stratified_split = split_config.get('stratified_split', True)
        self.split_seed = split_config.get('split_seed', 42)

        # Параметры сэмплирования
        sampling_config = self.config.get('sampling', {})
        self.use_sampling = sampling_config.get('use_sampling', False)
        self.train_sample_size = sampling_config.get('train_sample_size', 100000)
        self.validation_sample_size = sampling_config.get('validation_sample_size', 100000)
        self.sample_seed = sampling_config.get('sample_seed', 42)
        self.balanced_sampling = sampling_config.get('balanced_sampling', False)
        self.positive_rate = sampling_config.get('positive_rate', 0.5)

        # Параметры обучения
        train_config = self.config.get('train', {})
        self.verbose = train_config.get('verbose', False)
        self.n_folds = train_config.get('n_folds', 1)

        # Информация о колонках
        columns_config = self.config.get('columns', {})
        self.target_col = columns_config.get('target_col', 'target')
        self.index_cols = columns_config.get('index_cols', ['id'])

        # Загрузка списков признаков
        self.selected_features = []
        all_features_file = columns_config.get('all_features_file', None)
        if all_features_file and os.path.exists(all_features_file):
            with open(all_features_file, 'r') as f:
                self.selected_features = [line.strip() for line in f.readlines()]

        self.category_columns = []
        category_features_file = columns_config.get('category_features_file', None)
        if category_features_file and os.path.exists(category_features_file):
            with open(category_features_file, 'r') as f:
                self.category_columns = [line.strip() for line in f.readlines()]

        # Загрузка данных
        data_config = self.config.get('data_source', {})
        self.train_path = data_config.get('train_path', None)
        self.valid_path = data_config.get('valid_path', None)

        if self.train_path:
            self.data = pd.read_parquet(self.train_path)

            # Если указан путь к валидационному набору, загружаем его
            if self.valid_path:
                self.valid_df = pd.read_parquet(self.valid_path)
            else:
                self.valid_df = None
        else:
            raise ValueError("Не указан путь к обучающим данным в конфигурации")

        # Удаляем столбцы, указанные для пропуска
        if self.skip_cols:
            self.data = self.data.drop(columns=self.skip_cols, errors='ignore')
            if self.valid_df is not None:
                self.valid_df = self.valid_df.drop(columns=self.skip_cols, errors='ignore')

        # Если не указан список признаков для использования, используем все столбцы кроме target_col и index_cols
        if not self.selected_features:
            self.selected_features = [col for col in self.data.columns
                                     if col not in [self.target_col] + self.index_cols]

        # Параметры калибровки
        calibration_config = self.config.get('calibration', {})
        self.use_calibration = calibration_config.get('use_calibration', False)
        self.calibration_type = calibration_config.get('calibration_type', 'isotonic')

        # Инициализация параметров моделей
        self.models_config = self.config.get('models', {})

        # Гиперпараметры и настройки моделей
        self.hp = {}
        self.model_n_folds = {}
        self.model_hyperparameters = {}

        # Заполняем параметры моделей
        for model_type in self.selected_models:
            model_config = self.models_config.get(model_type, {})

            self.use_custom_hyperparameters = model_config.get('use_custom_hyperparameters', False)

            if self.use_custom_hyperparameters:
                self.hp[model_type] = model_config.get('hyperparameters', {})

            # Количество фолдов для кросс-валидации
            self.model_n_folds[model_type] = model_config.get('n_folds', self.n_folds)

        # Инициализация контейнеров для хранения результатов
        self.trained_models = {}
        self.metrics_results = {}

    def _sampling(self):
        """
        Выполняет сэмплирование данных в соответствии с конфигурацией.
        """
        if not self.use_sampling:
            return

        print(f"Sampling data ...")

        # Сэмплирование обучающих данных
        if self.train_sample_size < len(self.train_df):
            if self.task == 'binary' and self.balanced_sampling:
                # Балансировочное сэмплирование для бинарной классификации
                pos_samples = self.train_df[self.train_df[self.target_col] == 1]
                neg_samples = self.train_df[self.train_df[self.target_col] == 0]

                pos_count = len(pos_samples)
                neg_count = len(neg_samples)

                # Рассчитываем долю негативных примеров, которую нужно взять
                neg_frac = pos_count * (1 / self.positive_rate - 1) / neg_count

                # Берем все позитивные примеры
                # Для негативных примеров применяем случайный отбор с вероятностью neg_frac
                neg_samples_selected = neg_samples.sample(frac=min(1.0, neg_frac), random_state=self.sample_seed)

                # Объединяем выборки
                self.train_df = pd.concat([pos_samples, neg_samples_selected]).sample(frac=1, random_state=self.sample_seed)

                # Если после балансировки размер больше требуемого, делаем обычное сэмплирование
                if len(self.train_df) > self.train_sample_size:
                    self.train_df = self.train_df.sample(n=self.train_sample_size, random_state=self.sample_seed)
            else:
                # Обычное случайное сэмплирование
                self.train_df = self.train_df.sample(n=self.train_sample_size, random_state=self.sample_seed)

        # Сэмплирование валидационных данных
        if self.valid_df is not None and self.validation_sample_size < len(self.valid_df):
            self.valid_df = self.valid_df.sample(n=self.validation_sample_size, random_state=self.sample_seed)

        print(f"Data sampled: train shape: {self.train_df.shape}, valid shape: {self.valid_df.shape}")

    def _split_data(self):
        print("Splitting data ...")
        stratify_col = self.data[self.target_col] if self.stratified_split and self.task in ['binary', 'multiclass'] else None

        # Если validation_rate не указан (None), делим только на train и test
        if self.validation_rate is None:
            if self.valid_path:
                 print("Предупреждение: valid_path указан, но validation_rate = None. valid_path будет проигнорирован, данные будут разделены на train/test.")

            self.train_df, self.test_df = train_test_split(
                self.data,
                test_size=self.test_rate,
                random_state=self.split_seed,
                stratify=stratify_col
            )
            self.valid_df = None # Явно указываем, что валидационного набора нет
            print(f"Data split (no validation set): train shape: {self.train_df.shape}, test shape: {self.test_df.shape}")

        # Если validation_rate указан (не None)
        else:
            # Если valid_path задан, используем его и делим data только на train/test
            if self.valid_path:
                self.train_df, self.test_df = train_test_split(
                    self.data,
                    test_size=self.test_rate,
                    random_state=self.split_seed,
                    stratify=stratify_col
                )
                # self.valid_df уже загружен в __init__
                print(f"Data split (using valid_path): train shape: {self.train_df.shape}, test shape: {self.test_df.shape}, valid shape: {self.valid_df.shape}")
            # Если valid_path не задан, делим data на train/valid/test
            else:
                # Сначала отделяем test
                train_valid_df, self.test_df = train_test_split(
                    self.data,
                    test_size=self.test_rate,
                    random_state=self.split_seed,
                    stratify=stratify_col
                )

                # Затем разделяем оставшуюся часть на train и valid
                # Пересчитываем стратификацию для оставшейся части
                stratify_train_valid = train_valid_df[self.target_col] if self.stratified_split and self.task in ['binary', 'multiclass'] else None
                # Рассчитываем долю для второго сплита
                relative_validation_rate = self.validation_rate / (1.0 - self.test_rate)

                self.train_df, self.valid_df = train_test_split(
                    train_valid_df,
                    test_size=relative_validation_rate,
                    random_state=self.split_seed,
                    stratify=stratify_train_valid
                )
                print(f"Data split (train/valid/test): train shape: {self.train_df.shape}, test shape: {self.test_df.shape}, valid shape: {self.valid_df.shape}")

    def _train_model(self, model_type):
        """Обучает модель используя интерфейс BaseModel"""
        # Определение параметра калибровки
        calibrate = self.calibration_type if (self.use_calibration and self.task == 'binary') else None

        # Формируем общий словарь аргументов для всех моделей
        model_args = {
            'task': self.task,
            'hp': self.hp.get(model_type, None),
            'metrics': self.metrics_list,
            'calibrate': calibrate,
            'n_folds': self.model_n_folds.get(model_type, self.n_folds),
            'main_metric': self.main_metric,
            'verbose': self.verbose,
            'features': self.selected_features,
            'cat_features': self.category_columns,
            'target_name': self.target_col
        }

        # Динамическая загрузка модулей моделей по мере необходимости
        if model_type == 'catboost':
            from models.catboost_model import CatBoostModel
            model_class = CatBoostModel
        elif model_type == 'lightgbm':
            from models.lightgbm_model import LightGBMModel
            model_class = LightGBMModel
        elif model_type == 'xgboost':
            from models.xgboost_model import XGBoostModel
            model_class = XGBoostModel
        elif model_type == 'random_forest':
            from models.random_forest_model import RandomForestModel
            model_class = RandomForestModel
        elif model_type == 'lightautoml':
            from models.lightautoml_model import LightAutoMLModel
            model_class = LightAutoMLModel
        elif model_type == 'tabnet':
            from models.tabnet_model import TabNetModel
            model_class = TabNetModel
        elif model_type == 'cemlp':
            from models.cemlp_model import CEMLPModel
            model_class = CEMLPModel
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        # Создаем экземпляр модели
        self.model = model_class(**model_args)

        # Обучение модели через интерфейс BaseModel
        self.model.train(
            train=self.train_df,
            test=self.test_df,
        )

        return self.model

    def train_models(self):
        """
        Последовательно обучает все модели из списка selected_models
        и сохраняет только лучшую модель по основной метрике
        """
        # Разделение данных перед обучением
        self._split_data()

        # Сэмплирование данных, если задано
        self._sampling()

        # Обучение всех выбранных моделей
        for model_type in self.selected_models:
            print('-' * 10)
            print(f"Обучение модели: {model_type} на {self.model_n_folds[model_type]} фолдах, тип задачи: {self.task}")

            # Замеряем время начала обучения
            start_time = time.time()

            # Обучение модели
            model = self._train_model(model_type)

            # Вычисляем затраченное время в секундах
            time_spent = time.time() - start_time

            # Сохранение модели в словаре
            self.trained_models[model_type] = model

            # Вычисление метрик на тестовых данных через интерфейс модели
            print(f"Метрики на тестовых данных:")
            metrics_result = model.evaluate(
                self.test_df[self.selected_features],
                self.test_df[self.target_col],
            )

            # Добавляем время обучения к метрикам
            metrics_result['time_spent'] = time_spent
            self.metrics_results[model_type] = metrics_result
            print()

        # Вывод таблицы с результатами
        self.print_metrics_table()

        # Определение лучшей модели по основной метрике
        best_model_type, best_metric_value = get_best_model_by_metric(self.metrics_results, self.main_metric)
        print(f"Лучшая модель по метрике {self.main_metric}: {best_model_type}")
        print(f"Значение метрики: {best_metric_value}")

        # Сохраняем только лучшую модель
        best_model = self.trained_models[best_model_type]
        self.save_model(best_model_type, best_model, best_metric_value)

        return self.trained_models, self.metrics_results

    def save_model(self, model_type, model, metric_value):
        """
        Сохраняет обученную модель и калибровочную модель (если есть) в zip-архив.
        Модели каждого фолда сохраняются соответствующим образом:
        - PyTorch модели: сохраняется state_dict (.pth) и остальная часть estimator (.pickle).
        - Остальные модели: сохраняется весь объект модели (.pickle).
        Также сохраняются дополнительные файлы с метаданными.

        Архив сохраняется в поддиректории с именем модели внутри model_dir.

        Args:
            model_type: Тип модели (например, 'catboost', 'cemlp')
            model: Экземпляр BaseModel (например, CEMLPModel)
            metric_value: Значение основной метрики
        """
        # Создаем директорию для модели, если не существует
        model_subdir = os.path.join(self.model_dir, self.model_name)
        os.makedirs(model_subdir, exist_ok=True)

        # Формируем название архива в формате ЗНАЧЕНИЕ_МЕТРИКИ_ТИП_МОДЕЛИ_ГГГГММДД_ЧЧММ.zip
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Округляем метрику для имени файла, заменяем точку на подчеркивание
        metric_str = f"{metric_value:.4f}".replace('.', '_')
        archive_name = f"{metric_str}_{model_type}_{current_timestamp}.zip"
        archive_path = os.path.join(model_subdir, archive_name)

        # Используем временную директорию для подготовки файлов
        with tempfile.TemporaryDirectory() as temp_dir:
            # Сохраняем каждую фолд-модель
            for i, fold_model in enumerate(model.models):
                # Проверяем, является ли модель PyTorch-based estimator
                is_pytorch_estimator = hasattr(fold_model, 'model') and isinstance(fold_model.model, torch.nn.Module)

                if is_pytorch_estimator:
                    # 1. Сохраняем state_dict PyTorch модели
                    state_dict_file = os.path.join(temp_dir, f"fold_{i}_statedict.pth")
                    torch.save(fold_model.model.state_dict(), state_dict_file)

                    # 2. Временно удаляем PyTorch модель из estimator для pickle
                    pytorch_model_backup = fold_model.model
                    fold_model.model = None

                    # 3. Сохраняем остальную часть estimator
                    estimator_file = os.path.join(temp_dir, f"fold_{i}_estimator.pickle")
                    joblib.dump(fold_model, estimator_file)

                    # 4. Восстанавливаем PyTorch модель в estimator
                    fold_model.model = pytorch_model_backup
                else:
                    # Сохраняем модель целиком (например, CatBoost, LGBM)
                    model_file = os.path.join(temp_dir, f"fold_{i}_model.pickle")
                    joblib.dump(fold_model, model_file)


            # Сохраняем калибровочную модель, если она есть (обычно sklearn)
            if model.calibration_model is not None:
                # Пока предполагаем, что калибровочная модель не PyTorch
                calibration_file = os.path.join(temp_dir, "calibration_model.pickle")
                joblib.dump(model.calibration_model, calibration_file)

            # Сохраняем гиперпараметры в JSON-файле
            hyperparams_file = os.path.join(temp_dir, "hyperparameters.json")
            # Используем .hyperparameters из BaseModel, который хранит использованные параметры
            hyperparams_to_save = model.hyperparameters if hasattr(model, 'hyperparameters') else {}
            # Преобразуем device в строку, если это объект torch.device
            if isinstance(hyperparams_to_save.get('device'), torch.device):
                 hyperparams_to_save['device'] = str(hyperparams_to_save['device'])
            with open(hyperparams_file, 'w') as f:
                # Сериализуем с обработкой несериализуемых типов (на всякий случай)
                json.dump(hyperparams_to_save, f, indent=4, default=lambda o: '<not serializable>')


            # Сохраняем метрики в JSON-файле
            metrics_file = os.path.join(temp_dir, "metrics.json")
            metrics_to_save = self.metrics_results.get(model_type, {})
            with open(metrics_file, 'w') as f:
                # Преобразуем numpy типы в стандартные python типы для JSON
                 metrics_serializable = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in metrics_to_save.items()}
                 json.dump(metrics_serializable, f, indent=4)


            # Сохраняем список всех признаков
            features_file = os.path.join(temp_dir, "features.txt")
            with open(features_file, 'w') as f:
                f.write('\n'.join(self.selected_features))

            # Сохраняем список категориальных признаков
            cat_features_file = os.path.join(temp_dir, "cat_features.txt")
            with open(cat_features_file, 'w') as f:
                f.write('\n'.join(self.category_columns))

            # Сохраняем информацию о таргете и индексе
            target_info_file = os.path.join(temp_dir, "target_index_info.txt")
            with open(target_info_file, 'w') as f:
                f.write(f"Target column: {self.target_col}\n")
                f.write(f"Index columns: {', '.join(self.index_cols)}\n")
                f.write(f"Task type: {self.task}\n")
                f.write(f"Main metric: {self.main_metric}\n")
                f.write(f"Model type: {model_type}\n") # Добавим тип модели для удобства

            # Создаем zip-архив
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Добавляем все файлы из временной директории
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Архивируем файл с относительным путем внутри архива
                        zipf.write(file_path, os.path.basename(file_path))

        if self.verbose:
            print(f"Лучшая модель {model_type} сохранена в {archive_path}")

    def print_metrics_table(self):
        """
        Выводит красивую таблицу с результатами всех моделей и метрик,
        отсортированную по главной метрике.
        """
        # Получаем все уникальные метрики из результатов
        all_metrics = set()
        for model_metrics in self.metrics_results.values():
            all_metrics.update(model_metrics.keys())

        # Удаляем time_spent из общего списка метрик, чтобы добавить его в конце
        if 'time_spent' in all_metrics:
            all_metrics.remove('time_spent')

        # Сортируем метрики: сначала главная метрика, затем остальные в алфавитном порядке
        sorted_metrics = sorted(all_metrics)
        if self.main_metric in sorted_metrics:
            sorted_metrics.remove(self.main_metric)
            sorted_metrics.insert(0, self.main_metric)

        # Добавляем time_spent в конец списка метрик
        sorted_metrics.append('time_spent')

        # Сортируем модели по главной метрике
        sorted_models = sorted(
            self.metrics_results.keys(),
            key=lambda m: self.metrics_results[m].get(self.main_metric, float('-inf')),
            reverse=METRIC_DIRECTIONS.get(self.main_metric.split(';')[0], MAXIMIZE) == MAXIMIZE
        )

        # Формируем данные для таблицы
        table_data = []
        for model in sorted_models:
            row = [model]
            for metric in sorted_metrics:
                value = self.metrics_results[model].get(metric, float('nan'))
                # Форматируем время в ЧЧ:ММ:СС
                if metric == 'time_spent' and isinstance(value, (int, float)):
                    time_delta = timedelta(seconds=int(value))
                    row.append(str(time_delta).split('.')[0])  # Формат HH:MM:SS
                elif isinstance(value, (int, float)):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            table_data.append(row)

        # Формируем заголовки
        headers = ["Модель"] + [metric if metric != 'time_spent' else 'Время обучения' for metric in sorted_metrics]

        # Выводим таблицу
        print("\nРезультаты моделей:")
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="right"))
        print()

if __name__ == '__main__':
    modelling = SOTAModels('my_config.yaml')
    modelling.train_models()

