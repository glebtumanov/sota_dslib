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
            eval_set=(X_test, y_test),
            eval_metric='accuracy',
            mode='max',
            cat_features=self.cat_features
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
        # Общие гиперпараметры для всех типов задач, обновленные для новой архитектуры
        return {
            'd_model': 16,                # Размерность эмбеддингов
            'n_steps': 4,                 # Количество шагов TabNet
            'decision_dim': 64,           # Общая размерность выхода FeatureTransformer (Nd+Na), должна быть четной
            'n_shared': 2,                # Кол-во общих GLU блоков
            'n_independent': 2,           # Кол-во независимых GLU блоков на шаге
            'dropout_glu': 0.1,           # Dropout в GLU блоках
            'dropout_emb': 0.1,           # Dropout после эмбеддингов
            'glu_norm': 'batch',          # Тип нормализации в GLU ('batch', 'layer', None)
            'gamma': 1.3,                 # Коэффициент релаксации prior (обычно 1.0-2.0)
            'lambda_sparse': 1e-4,        # Коэффициент регуляризации разреженности (важен для интерпретируемости)
            'batch_size': 2048,           # Размер батча
            'epochs': 150,                # Максимальное количество эпох
            'learning_rate': 0.01,        # Скорость обучения (может требовать подбора)
            'early_stopping_patience': 15,# Терпение для ранней остановки
            'weight_decay': 1e-5,         # L2 регуляризация
            'reducelronplateau_patience': 5, # Терпение для снижения LR
            'reducelronplateau_factor': 0.7, # Фактор снижения LR
            'scale_numerical': True,        # Масштабировать числовые?
            'scale_method': 'quantile',    # Метод масштабирования (quantile часто устойчивее)
            'n_bins': 10,                 # Кол-во бинов для 'binning' (если используется)
            'device': None,               # Устройство cuda/cpu (автоматическое определение)
            # 'output_dim': 1,            # Определяется задачей (binary=1, multiclass=n_classes, regression=1)
            'verbose': True,              # Выводить прогресс?
            'num_workers': 0,             # Кол-во воркеров DataLoader
            'random_state': 42            # Random state
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