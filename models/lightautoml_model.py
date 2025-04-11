# spark_sota_modeling/models/lightautoml_model.py
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from .base_model import BaseModel
import pandas as pd

class LightAutoMLModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)
        self.num_features = [col for col in features if col not in cat_features]

    def _train_fold(self, task, X_train, y_train, X_test, y_test):
        X_train = pd.concat([X_train, y_train], axis=1)
        X_test = pd.concat([X_test, y_test], axis=1)
        X = pd.concat([X_train, X_test], axis=0)

        model = TabularAutoML(
            gpu_ids=None,
            task=task,
            **self.hyperparameters
        )

        roles = {
            "target": self.target_name,
            "numeric": self.num_features,
            "category": self.cat_features,
        }

        model.fit_predict(X, roles=roles, verbose=0)
        return model

    def _train_fold_binary(self, *args):
        task = Task("binary", metric="auc", greater_is_better=True)
        return self._train_fold(task, *args)

    def _train_fold_multi(self, *args):
        task = Task("multiclass", metric="crossentropy", greater_is_better=False)
        return self._train_fold(task, *args)

    def _train_fold_regression(self, *args):
        task = Task("reg", metric="mse", greater_is_better=False)
        return self._train_fold(task, *args)

    def _predict_fold_binary(self, model, X):
        return model.predict(X).data[:,0]

    def _predict_fold_multi(self, model, X):
        return model.predict(X).data

    def _predict_fold_regression(self, model, X):
        return model.predict(X).data[:,0]

    def _get_default_hp(self):
        return {
            'memory_limit': 32,
            'timeout': 3600,
            'cpu_limit': 10,
            'reader_params': {'n_jobs': 4, 'cv': 2, 'memory_limit': 8},
            'debug': False,
        }

    def _get_default_hp_binary(self):
        return self._get_default_hp()

    def _get_default_hp_multi(self):
        return self._get_default_hp()

    def _get_default_hp_regression(self):
        return self._get_default_hp()
