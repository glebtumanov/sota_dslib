from models.estimators.cemlp_estimator import CatEmbMLPBinary, CatEmbMLPMulticlass, CatEmbMLPRegressor
from .base_model import BaseModel
import os
import joblib
import torch
from typing import Optional

class CEMLPModel(BaseModel):
    def __init__(self, task='binary', hp=None, metrics=None, calibrate=None, n_folds=1,
                 main_metric=None, verbose=True, features=[], cat_features=[], target_name=None, 
                 index_cols=[]):
        super().__init__(task, hp, metrics, calibrate, n_folds, main_metric, 
                         verbose, features, cat_features, target_name, index_cols)

    def _train_fold_binary(self, X_train, y_train, X_test, y_test):
        model = CatEmbMLPBinary(**self.hyperparameters, cat_features=self.cat_features)
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
        model = CatEmbMLPMulticlass(**hyperparameters, cat_features=self.cat_features)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            eval_metric='accuracy',
            mode='max'
        )
        return model

    def _train_fold_regression(self, X_train, y_train, X_test, y_test):
        model = CatEmbMLPRegressor(**self.hyperparameters, cat_features=self.cat_features)
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

    def save_model_files(self, save_path: str):
        """
        Сохраняет файлы модели CEMLP (по одному для каждого фолда)
        в указанную директорию save_path (которая уже включает тип модели).
        Для каждой модели сохраняется state_dict PyTorch и остальная часть estimator.

        Args:
            save_path (str): Полный путь к директории для сохранения файлов модели.
        """
        for i, fold_estimator in enumerate(self.models):
            # fold_estimator здесь это экземпляр CatEmbMLPBinary, CatEmbMLPMulticlass или CatEmbMLPRegressor
            # у которого есть атрибут .model (сама PyTorch модель)
            if hasattr(fold_estimator, 'model') and isinstance(fold_estimator.model, torch.nn.Module):
                # 1. Сохраняем state_dict PyTorch модели
                state_dict_file = os.path.join(save_path, f"fold_{i}_statedict.pth")
                torch.save(fold_estimator.model.state_dict(), state_dict_file)

                # 2. Временно удаляем PyTorch модель из estimator для pickle
                pytorch_model_backup = fold_estimator.model
                fold_estimator.model = None

                # 3. Сохраняем остальную часть estimator (без PyTorch модели)
                estimator_file = os.path.join(save_path, f"fold_{i}_estimator.pickle")
                joblib.dump(fold_estimator, estimator_file)

                # 4. Восстанавливаем PyTorch модель в estimator
                fold_estimator.model = pytorch_model_backup
            else:
                # Если это не ожидаемый PyTorch-based estimator, сохраняем как есть
                # Этого не должно происходить для CEMLPModel при правильной реализации
                print(f"Warning: Fold {i} for CEMLPModel is not a standard PyTorch estimator. Saving as is.")
                model_file_path = os.path.join(save_path, f"fold_{i}_model.pickle")
                joblib.dump(fold_estimator, model_file_path)

    def save_all(self, save_path: str, metrics_to_save: Optional[dict] = None):
        """
        Сохраняет модель и все артефакты, используя настройки по умолчанию для model_type_name
        и опционально для metrics_to_save.

        Args:
            save_path (str): Полный путь к директории для сохранения артефактов.
            metrics_to_save (Optional[dict]): Словарь с метриками. Если None, BaseModel.save использует стандартные.
        """
        model_type_name = "cemlp"
        super().save(
            save_path=save_path,
            model_type_name=model_type_name,
            metrics_to_save=metrics_to_save
        )

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

    # ---------------------------------------------------------------
    # Загрузка файлов модели CEMLP (state_dict + estimator)
    # ---------------------------------------------------------------

    def load_model_files(self, load_path: str):
        """Восстанавливает обученные эстиматоры CEMLP из директории.

        Схема хранения аналогична TabNetModel: пара файлов на фолд.
        Если найден только pickle – загружаем его напрямую.
        """
        restored_models = []
        idx = 0
        while True:
            est_file = os.path.join(load_path, f"fold_{idx}_estimator.pickle")
            state_file = os.path.join(load_path, f"fold_{idx}_statedict.pth")
            model_file = os.path.join(load_path, f"fold_{idx}_model.pickle")

            if os.path.exists(est_file) and os.path.exists(state_file):
                estimator = joblib.load(est_file)
                if estimator.model is None:
                    estimator.model = estimator._init_model()
                state_dict = torch.load(state_file, map_location=estimator.device)
                estimator.model.load_state_dict(state_dict)
                estimator.model.to(estimator.device)
                estimator.is_fitted_ = True
                restored_models.append(estimator)
            elif os.path.exists(model_file):
                restored_models.append(joblib.load(model_file))
            else:
                break
            idx += 1

        return restored_models