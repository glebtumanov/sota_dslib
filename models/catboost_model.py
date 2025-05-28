# spark_sota_modeling/models/catboost_model.py
from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel
import os
import joblib
from typing import Optional


class CatBoostModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None, index_cols=[]):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, 
                         verbose, features, cat_features, target_name, index_cols)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = CatBoostClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            cat_features=self.cat_features,
            verbose=False
        )
        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        model = CatBoostClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            cat_features=self.cat_features,
            verbose=False
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = CatBoostRegressor(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            cat_features=self.cat_features,
            verbose=False
        )
        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X)[:, 1]

    def _predict_fold_multiclass(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

    def save_model_files(self, save_path: str):
        """
        Сохраняет файлы модели CatBoost (по одному для каждого фолда)
        в указанную директорию save_path (которая уже включает тип модели).
        Каждая модель сохраняется как отдельный pickle-файл.

        Args:
            save_path (str): Полный путь к директории для сохранения файлов модели.
        """
        for i, fold_model in enumerate(self.models):
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
        model_type_name = "catboost" # Имя типа модели для CatBoost
        super().save(
            save_path=save_path,
            model_type_name=model_type_name,
            metrics_to_save=metrics_to_save
        )

    def _get_required_hp_binary(self):
        return {'eval_metric': 'AUC', 'bootstrap_type': 'Bernoulli'}

    def _get_required_hp_multiclass(self):
        return {'eval_metric': 'MultiClass', 'bootstrap_type': 'MVS'}

    def _get_required_hp_regression(self):
        return {'eval_metric': 'RMSE', 'bootstrap_type': 'Poisson'}

    def _get_default_hp_binary(self):
        return {
            'eval_metric': 'AUC',
            'iterations': 1000,
            'early_stopping_rounds': 100,
            'thread_count': 4,
            'random_seed': 42,
            'verbose': False,
        }

    def _get_default_hp_multiclass(self):
        return {
            'eval_metric': 'MultiClass',
            'iterations': 1000,
            'early_stopping_rounds': 100,
            'thread_count': 4,
            'random_seed': 42,
            'verbose': False,
        }

    def _get_default_hp_regression(self):
        return {
            'eval_metric': 'RMSE',
            'iterations': 1000,
            'early_stopping_rounds': 100,
            'thread_count': 4,
            'random_seed': 42,
            'verbose': False,
        }