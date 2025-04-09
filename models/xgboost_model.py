from xgboost import XGBClassifier, XGBRegressor
from .base_model import BaseModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class XGBoostModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1, main_metric=None, verbose=True):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test, params, cat_features):
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=params.get('verbose', False)
        )
        return model

    def _train_fold_multi(self, X_train, y_train, X_test, y_test, params, cat_features):
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=params.get('verbose', False)
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test, params, cat_features):
        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
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
            'eval_metric': 'auc',
            'n_estimators': 1000,
            'early_stopping_rounds': 20,
            'n_jobs': 4,
            'scale_pos_weight': 1,
            'random_state': 42,
            'max_depth': 10,
            'min_child_weight': 50,
            'lambda': 10.0,
            'gamma': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'verbosity': 0,
        }

    def _get_default_hp_multi(self):
        return {
            'eval_metric': 'mlogloss',
            'n_estimators': 1000,
            'early_stopping_rounds': 20,
            'n_jobs': 4,
            'random_state': 42,
            'max_depth': 10,
            'min_child_weight': 50,
            'lambda': 10.0,
            'gamma': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'verbosity': 0,
        }

    def _get_default_hp_regression(self):
        return {
            'eval_metric': 'rmse',
            'n_estimators': 1000,
            'early_stopping_rounds': 20,
            'n_jobs': 4,
            'random_state': 42,
            'max_depth': 10,
            'min_child_weight': 50,
            'lambda': 10.0,
            'gamma': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'verbosity': 0,
        }
