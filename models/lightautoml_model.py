# spark_sota_modeling/models/lightautoml_model.py
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from .base_model import BaseModel
import pandas as pd
import os
import joblib
from typing import Optional

class LightAutoMLModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None, 
                 index_cols=[]):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, 
                         verbose, features, cat_features, target_name, index_cols)
        self.num_features = [col for col in features if col not in cat_features]

    def _train_fold(self, task, X_train, y_train, X_test, y_test):
        X_train = pd.concat([X_train, y_train], axis=1)
        X_test = pd.concat([X_test, y_test], axis=1)
        X = pd.concat([X_train, X_test], axis=0)

        model = TabularAutoML(
            gpu_ids=None,
            task=task,
            **self.hyperparameters
        )

        roles = {
            "target": self.target_name,
            "numeric": self.num_features,
            "category": self.cat_features,
        }

        model.fit_predict(X, roles=roles, verbose=self.verbose)
        return model

    def _train_fold_binary(self, *args):
        task = Task("binary", metric="auc", greater_is_better=True)
        return self._train_fold(task, *args)

    def _train_fold_multiclass(self, *args):
        task = Task("multiclass", metric="crossentropy", greater_is_better=False)
        return self._train_fold(task, *args)

    def _train_fold_regression(self, *args):
        task = Task("reg", metric="mse", greater_is_better=False)
        return self._train_fold(task, *args)

    def _predict_fold_binary(self, model, X):
        return model.predict(X).data[:,0]

    def _predict_fold_multiclass(self, model, X):
        return model.predict(X).data

    def _predict_fold_regression(self, model, X):
        return model.predict(X).data[:,0]

    def save_model_files(self, save_path: str):
        """
        Сохраняет файлы модели LightAutoML (по одному для каждого фолда/пайплайна)
        в указанную директорию save_path (которая уже включает тип модели).
        Каждая модель сохраняется как отдельный pickle-файл.

        Args:
            save_path (str): Полный путь к директории для сохранения файлов модели.
        """
        for i, fold_model in enumerate(self.models):
            # Предполагаем, что fold_model это экземпляр TabularAutoML или подобный,
            # который можно сериализовать с помощью joblib/pickle.
            # Используем save_path напрямую
            model_file_path = os.path.join(save_path, f"fold_{i}_model.pickle")
            joblib.dump(fold_model, model_file_path)

    def save_all(self, save_path: str, metrics_to_save: Optional[dict] = None):
        """
        Сохраняет модель и все артефакты, используя настройки по умолчанию для model_type_name
        и опционально для metrics_to_save.

        Args:
            save_path (str): Полный путь к директории для сохранения артефактов.
            metrics_to_save (Optional[dict]): Словарь с метриками. Если None, BaseModel.save использует стандартные.
        """
        model_type_name = "lightautoml"
        super().save(
            save_path=save_path,
            model_type_name=model_type_name,
            metrics_to_save=metrics_to_save
        )

    def _get_default_hp(self):
        return {
            'memory_limit': 32,
            'timeout': 3600,
            'cpu_limit': 10,
            'reader_params': {'n_jobs': 4, 'cv': 2, 'memory_limit': 8},
            'debug': False,
        }

    def _get_default_hp_binary(self):
        return self._get_default_hp()

    def _get_default_hp_multiclass(self):
        return self._get_default_hp()

    def _get_default_hp_regression(self):
        return self._get_default_hp()
