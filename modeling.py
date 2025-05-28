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

        # Сначала всегда выделяем тестовую выборку
        self.train_df, self.test_df = train_test_split(
            self.data,
            test_size=self.test_rate,
            random_state=self.split_seed,
            stratify=stratify_col
        )

        # Определяем valid_df в зависимости от условий
        if self.valid_path:
            # Если valid_path указан, то в качестве valid_df используем загруженные в __init__ данные
            # self.valid_df уже загружен в __init__
            print(f"Data split (using valid_path): train shape: {self.train_df.shape}, test shape: {self.test_df.shape}, valid shape: {self.valid_df.shape}")
        elif self.validation_rate:
            # Если valid_path не указан, но validation_rate указан, 
            # выделяем valid_df из train_df (который получен после выделения test_df)
            stratify_train = self.train_df[self.target_col] if self.stratified_split and self.task in ['binary', 'multiclass'] else None
            
            self.train_df, self.valid_df = train_test_split(
                self.train_df,
                test_size=self.validation_rate,
                random_state=self.split_seed,
                stratify=stratify_train
            )
            print(f"Data split (train/valid/test): train shape: {self.train_df.shape}, test shape: {self.test_df.shape}, valid shape: {self.valid_df.shape}")
        else:
            # Если valid_path не указан и validation_rate не указан, то не используем valid_df
            self.valid_df = None
            print(f"Data split (no validation set): train shape: {self.train_df.shape}, test shape: {self.test_df.shape}")

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
            'target_name': self.target_col,
            'index_cols': self.index_cols
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
        и сохраняет все обученные модели
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

            # Вычисление метрик на тестовых или валидационных данных через интерфейс модели
            if self.valid_df is not None:
                print(f"Метрики на валидационных данных:")
                metrics_result = model.evaluate(self.valid_df)
            else:
                print(f"Метрики на тестовых данных:")
                metrics_result = model.evaluate(self.test_df)

            # Добавляем время обучения к метрикам
            metrics_result['time_spent'] = time_spent
            self.metrics_results[model_type] = metrics_result
            
            # Сохраняем модель сразу после обучения и оценки
            model.save(
                save_path=self.model_dir,
                metrics_to_save=self.metrics_results[model_type],
                model_type_name=model_type
            )

        # Вывод таблицы с результатами
        self.print_metrics_table()

        # Определение лучшей модели по основной метрике
        best_model_type, best_metric_value = get_best_model_by_metric(self.metrics_results, self.main_metric)
        print(f"Лучшая модель по метрике {self.main_metric}: {best_model_type}")
        print(f"Значение метрики: {best_metric_value}")

        return self.trained_models, self.metrics_results

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

