from models.nn.tabnet import TabNetBinaryClassifier, TabNetMulticlassClassifier, TabNetRegressor
from .base_model import BaseModel


class TabNetModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None):
        # Вызываем инициализатор базового класса
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, verbose, features, cat_features, target_name)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = TabNetBinaryClassifier(**self.hyperparameters)

        # Обучаем модель
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='roc_auc',
            mode='max'
        )

        return model

    def _train_fold_multiclass(self, X_train, y_train, X_test, y_test):
        # Определяем количество классов из обучающих данных
        n_classes = len(y_train.unique())

        # Обновляем гиперпараметры, добавляя n_classes
        hyperparameters = self.hyperparameters.copy()
        hyperparameters['n_classes'] = n_classes

        model = TabNetMulticlassClassifier(**hyperparameters)

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
            'cat_emb_dim': 4,  # Размерность эмбеддингов для категориальных признаков
            'n_steps': 4,  # Количество шагов в TabNet
            'hidden_dim': 16,  # Размерность скрытого слоя
            'decision_dim': 8,  # Размерность решающего слоя
            'n_glu_layers': 3,  # Количество GLU слоев
            'dropout': 0.1,  # Вероятность дропаута
            'gamma': 1.5,  # Коэффициент затухания для масок внимания
            'lambda_sparse': 0.0001,  # Коэффициент регуляризации разреженности
            'virtual_batch_size': 128,  # Размер виртуального батча для Ghost BatchNorm
            'momentum': 0.9,  # Параметр momentum для BatchNorm
            'batch_size': 1024,  # Размер батча для обучения
            'epochs': 50,  # Количество эпох обучения
            'learning_rate': 0.01,  # Скорость обучения
            'early_stopping_patience': 5,  # Количество эпох без улучшения до остановки
            'weight_decay': 1e-5,  # Весовая регуляризация для оптимизатора
            'scale_numerical': True,  # Масштабировать ли числовые признаки
            'scale_method': 'standard',  # Метод масштабирования
            'device': 'cuda',  # Устройство для обучения (cuda/cpu)
            'verbose': True,  # Вывод прогресса обучения
            'num_workers': 0,  # Количество worker-процессов для DataLoader
            'random_state': 42,  # Seed для воспроизводимости
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