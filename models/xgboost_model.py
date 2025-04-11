from xgboost import XGBClassifier, XGBRegressor
from .base_model import BaseModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class XGBoostModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None,
                 verbose=True, features=[], cat_features=[], target_name=None):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = XGBClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        model = XGBClassifier(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = XGBRegressor(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X)[:, 1]

    def _predict_fold_multiclass(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

    def _get_required_hp_binary(self):
        return {'eval_metric': 'auc'}

    def _get_required_hp_multiclass(self):
        return {'eval_metric': 'mlogloss'}

    def _get_required_hp_regression(self):
        return {'eval_metric': 'rmse'}

    def _get_default_hp_binary(self):
        return {
            'eval_metric': 'auc',
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'n_jobs': 4,
            'scale_pos_weight': 1,
            'random_state': 42,
            'verbosity': 0,
        }

    def _get_default_hp_multiclass(self):
        return {
            'eval_metric': 'mlogloss',
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'n_jobs': 4,
            'random_state': 42,
            'verbosity': 0,
        }

    def _get_default_hp_regression(self):
        return {
            'eval_metric': 'rmse',
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'n_jobs': 4,
            'random_state': 42,
            'verbosity': 0,
        }
