from models.estimators.cemlp_estimator import CatEmbMLPBinary, CatEmbMLPMulticlass, CatEmbMLPRegressor
from .base_model import BaseModel

class CEMLPModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = CatEmbMLPBinary(**self.hyperparameters)
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
        hyperparameters = self.hyperparameters.copy()
        hyperparameters['n_classes'] = n_classes
        model = CatEmbMLPMulticlass(**hyperparameters)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='accuracy',
            mode='max'
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = CatEmbMLPRegressor(**self.hyperparameters)
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
            'cat_emb_dim': 4,
            'hidden_dims': [64, 32],
            'activation': 'relu',
            'dropout': 0.1,
            'feature_dropout': 0.0,
            'batch_norm': True,
            'layer_norm': False,
            'initialization': 'he_normal',
            'leaky_relu_negative_slope': 0.1,
            'dynamic_emb_size': False,
            'min_emb_dim': 2,
            'max_emb_dim': 16,
            'batch_size': 1024,
            'epochs': 50,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'scale_numerical': True,
            'scale_method': 'standard',
            'n_bins': 10,
            'device': None,
            'output_dim': 1,
            'verbose': True,
            'num_workers': 0,
            'random_state': 42,
            'lr_scheduler_patience': 10,
            'lr_scheduler_factor': 0.5,
        }

    def _get_default_hp_binary(self):
        return self._get_default_hp()

    def _get_default_hp_multiclass(self):
        hp = self._get_default_hp()
        return hp

    def _get_default_hp_regression(self):
        hp = self._get_default_hp()
        return hp