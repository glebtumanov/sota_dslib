from models.estimators.cemlp_estimator import CatEmbMLPBinary, CatEmbMLPMulticlass, CatEmbMLPRegressor
from .base_model import BaseModel

class CEMLPModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)
        self.cat_features = cat_features

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = CatEmbMLPBinary(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='roc_auc',
            mode='max',
            cat_features=self.cat_features
        )
        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        n_classes = len(set(y_train))
        hyperparameters = self.hyperparameters.copy()
        hyperparameters['n_classes'] = n_classes
        model = CatEmbMLPMulticlass(**hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='accuracy',
            mode='max',
            cat_features=self.cat_features
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = CatEmbMLPRegressor(**self.hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='mse',
            mode='min',
            cat_features=self.cat_features
        )
        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X, cat_features=self.cat_features)[:, 1]

    def _predict_fold_multiclass(self, model, X):
        return model.predict_proba(X, cat_features=self.cat_features)

    def _predict_fold_regression(self, model, X):
        return model.predict(X, cat_features=self.cat_features)

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