from lightgbm import LGBMClassifier, LGBMRegressor
from .base_model import BaseModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class LightGBMModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None,
                 verbose=True, features=[], cat_features=[], target_name=None):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = LGBMClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=self.cat_features,
            verbose=False
        )
        return model

    def _train_fold_multi(self, X_train, y_train, X_test, y_test):
        model = LGBMClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=self.cat_features,
            verbose=False
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = LGBMRegressor(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=self.cat_features,
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
        return {'metric': 'auc'}

    def _get_required_hp_multi(self):
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

    def _get_default_hp_multi(self):
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
