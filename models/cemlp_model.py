from models.estimators.cemlp_estimator import CatEmbMLPBinary, CatEmbMLPMulticlass, CatEmbMLPRegressor
from .base_model import BaseModel

class CEMLPModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        estimator_kwargs = self._prepare_estimator_kwargs()
        model = CatEmbMLPBinary(**estimator_kwargs)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='roc_auc',
            mode='max'
        )
        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        n_classes = len(set(y_train))
        estimator_kwargs = self._prepare_estimator_kwargs({'n_classes': n_classes})
        model = CatEmbMLPMulticlass(**estimator_kwargs)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='accuracy',
            mode='max'
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        estimator_kwargs = self._prepare_estimator_kwargs()
        model = CatEmbMLPRegressor(**estimator_kwargs)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='mse',
            mode='min'
        )
        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X)[:, 1]

    def _predict_fold_multiclass(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

    def _get_required_hp_binary(self):
        return {}

    def _get_required_hp_multiclass(self):
        return {}

    def _get_required_hp_regression(self):
        return {}

    def _get_default_hp(self):
        return {
            'verbose': False,
        }

    def _get_default_hp_binary(self):
        return self._get_default_hp()

    def _get_default_hp_multiclass(self):
        hp = self._get_default_hp()
        return hp

    def _get_default_hp_regression(self):
        hp = self._get_default_hp()
        return hp