from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None, verbose=True):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test, params, cat_features):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model

    def _train_fold_multi(self, X_train, y_train, X_test, y_test, params, cat_features):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test, params, cat_features):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X)[:, 1]

    def _predict_fold_multi(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

    def _get_default_hp_binary(self):
        return {
            'max_samples': 0.5,
            'bootstrap': True,
            'verbose': 0,
            'n_jobs': -1,
            'random_state': 5,
            'n_estimators': 1000,
            'max_depth': 100,
        }

    def _get_default_hp_multi(self):
        return {
            'max_samples': 0.5,
            'bootstrap': True,
            'verbose': 0,
            'n_jobs': -1,
            'random_state': 5,
            'n_estimators': 1000,
            'max_depth': 100,
        }

    def _get_default_hp_regression(self):
        return {
            'max_samples': 0.5,
            'bootstrap': True,
            'verbose': 0,
            'n_jobs': -1,
            'random_state': 5,
            'n_estimators': 1000,
            'max_depth': 100,
        }
