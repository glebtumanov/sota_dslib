from lightgbm import LGBMClassifier, LGBMRegressor
from .base_model import BaseModel
import warnings
import os
import joblib
from typing import Optional
warnings.filterwarnings("ignore", category=UserWarning)

class LightGBMModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None,
                 verbose=True, features=[], cat_features=[], target_name=None, index_cols=[]):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, 
                         verbose, features, cat_features, target_name, index_cols)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = LGBMClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=self.cat_features,
        )
        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        model = LGBMClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=self.cat_features,
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = LGBMRegressor(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=self.cat_features,
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
        Сохраняет файлы модели LightGBM (по одному для каждого фолда)
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
        model_type_name = "lightgbm" 
        super().save(
            save_path=save_path,
            model_type_name=model_type_name,
            metrics_to_save=metrics_to_save
        )

    def _get_required_hp_binary(self):
        return {'metric': 'auc'}

    def _get_required_hp_multiclass(self):
        return {'metric': 'multi_logloss'}

    def _get_required_hp_regression(self):
        return {'metric': 'rmse'}

    def _get_default_hp_binary(self):
        return {
            'metric': 'auc',
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'n_jobs': 4,
            'random_state': 42,
            'verbose': -1,
            'silent': True,
            'verbose': -1
        }

    def _get_default_hp_multiclass(self):
        return {
            'metric': 'multi_logloss',
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'n_jobs': 4,
            'verbose': -1,
            'silent': True,
            'verbose': -1
        }

    def _get_default_hp_regression(self):
        return {
            'metric': 'rmse',
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'n_jobs': 4,
            'random_state': 42,
            'verbose': -1,
            'silent': True,
            'verbose': -1
        }
