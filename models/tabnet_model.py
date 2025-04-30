from models.estimators.tabnet_estimator import TabNetBinary, TabNetMulticlass, TabNetRegressor
from .base_model import BaseModel


class TabNetModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)
        self.cat_features = cat_features

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = TabNetBinary(**self.hyperparameters, cat_features=self.cat_features)

        # Обучаем модель
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='roc_auc',
            mode='max',
        )

        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        # Определяем количество классов из обучающих данных
        n_classes = len(y_train.unique())

        # Обновляем гиперпараметры, добавляя n_classes
        hyperparameters = self.hyperparameters.copy()
        hyperparameters['n_classes'] = n_classes

        model = TabNetMulticlass(**hyperparameters, cat_features=self.cat_features)

        # Обучаем модель
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='accuracy',
            mode='max',
        )

        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = TabNetRegressor(**self.hyperparameters, cat_features=self.cat_features)

        # Обучаем модель
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='mae',
            mode='min',
        )

        return model

    def _predict_fold_binary(self, model, X):
        return model.predict_proba(X)[:, 1]

    def _predict_fold_multiclass(self, model, X):
        return model.predict_proba(X)

    def _predict_fold_regression(self, model, X):
        return model.predict(X)

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
        # Общие гиперпараметры для всех типов задач, обновленные для новой архитектуры
        return {
            'verbose': False,              # Выводить прогресс?
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