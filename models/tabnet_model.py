from models.estimators.tabnet_estimator import TabNetBinary, TabNetMulticlass, TabNetRegressor
from .base_model import BaseModel


class TabNetModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)
        self.cat_features = cat_features

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = TabNetBinary(**self.hyperparameters)

        # Обучаем модель
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
        # Определяем количество классов из обучающих данных
        n_classes = len(y_train.unique())

        # Обновляем гиперпараметры, добавляя n_classes
        hyperparameters = self.hyperparameters.copy()
        hyperparameters['n_classes'] = n_classes

        model = TabNetMulticlass(**hyperparameters)

        # Обучаем модель
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test)
        )

        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = TabNetRegressor(**self.hyperparameters)

        # Обучаем модель
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='mae',
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
        # Возвращаем обязательные гиперпараметры для бинарной классификации
        return {}

    def _get_required_hp_multiclass(self):
        # Возвращаем обязательные гиперпараметры для мультиклассовой классификации
        return {}

    def _get_required_hp_regression(self):
        # Возвращаем обязательные гиперпараметры для регрессии
        return {}

    def _get_default_hp(self):
        # Общие гиперпараметры для всех типов задач
        return {
            'cat_emb_dim': 4,
            'n_steps': 5,
            'n_d': 64,
            'n_a': 64,
            'decision_dim': 32,
            'n_glu_layers': 2,
            'dropout': 0.1,
            'gamma': 1.5,
            'lambda_sparse': 0.0001,
            'virtual_batch_size': 256,
            'momentum': 0.9,
            'batch_size': 4096,
            'epochs': 100,
            'learning_rate': 0.02,
            'early_stopping_patience': 20,
            'weight_decay': 1e-4,
            'reducelronplateau_patience': 10,
            'reducelronplateau_factor': 0.8,
            'scale_numerical': True,
            'scale_method': 'standard',
            'n_bins': 10,
            'device': None,
            'output_dim': 1,
            'verbose': True,
            'num_workers': 0,
            'random_state': 42
        }

    def _get_default_hp_binary(self):
        return self._get_default_hp()

    def _get_default_hp_multiclass(self):
        hp = self._get_default_hp()
        # Дополнительные гиперпараметры для мультиклассовой классификации можно добавить здесь
        return hp

    def _get_default_hp_regression(self):
        hp = self._get_default_hp()
        # Дополнительные гиперпараметры для регрессии можно добавить здесь
        return hp