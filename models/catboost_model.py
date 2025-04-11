# spark_sota_modeling/models/catboost_model.py
from catboost import CatBoostClassifier, CatBoostRegressor
from .base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)

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

    def _train_fold_multi(self, X_train, y_train, X_test, y_test):
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

    def _predict_fold_multi(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

    def _get_required_hp_binary(self):
        return {'eval_metric': 'AUC', 'bootstrap_type': 'Bernoulli'}

    def _get_required_hp_multi(self):
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

    def _get_default_hp_multi(self):
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