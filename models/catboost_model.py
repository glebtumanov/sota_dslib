# spark_sota_modeling/models/catboost_model.py
from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None, verbose=True):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose)

    def _train_fold_binary(self, X_train, y_train, X_val, y_val, params, cat_features, fold_idx=None):
        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            cat_features=cat_features,
            verbose=params.get('verbose', False)
        )
        return model

    def _train_fold_multi(self, X_train, y_train, X_val, y_val, params, cat_features, fold_idx=None):
        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            cat_features=cat_features,
            verbose=params.get('verbose', False)
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_val, y_val, params, cat_features, fold_idx=None):
        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            cat_features=cat_features,
            verbose=params.get('verbose', False)
        )
        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X)[:, 1]

    def _predict_fold_multi(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

    def _get_default_hp_binary(self):
        return {
            'eval_metric': 'AUC',
            'iterations': 1000,
            'early_stopping_rounds': 20,
            'thread_count': 4,
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'depth': 10,
            'min_data_in_leaf': 50,
            'l2_leaf_reg': 10.0,
            'border_count': 50,
            'rsm': 0.5,
            'subsample': 0.8,
            'allow_writing_files': False,
            'verbose': False,
            'use_best_model': False,
            'train_dir': '/tmp',
        }

    def _get_default_hp_multi(self):
        return {
            'eval_metric': 'MultiClass',
            'iterations': 1000,
            'early_stopping_rounds': 20,
            'thread_count': 4,
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'depth': 10,
            'min_data_in_leaf': 50,
            'l2_leaf_reg': 10.0,
            'border_count': 50,
            'rsm': 0.5,
            'subsample': 0.8,
            'allow_writing_files': False,
            'verbose': False,
            'use_best_model': False,
            'train_dir': '/tmp',
        }

    def _get_default_hp_regression(self):
        return {
            'eval_metric': 'RMSE',
            'iterations': 1000,
            'early_stopping_rounds': 20,
            'thread_count': 4,
            'random_seed': 42,
            'depth': 10,
            'min_data_in_leaf': 50,
            'l2_leaf_reg': 10.0,
            'border_count': 50,
            'rsm': 0.5,
            'subsample': 0.8,
            'allow_writing_files': False,
            'verbose': False,
            'use_best_model': False,
            'train_dir': '/tmp',
        }