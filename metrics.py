import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, mean_squared_error, mean_absolute_error, r2_score,
    balanced_accuracy_score, log_loss, top_k_accuracy_score,
    mean_absolute_percentage_error, mean_squared_log_error
)

# Определяем доступные метрики для каждого типа задачи
BINARY_METRICS: Set[str] = {
    'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
    'precision@k', 'recall@k', 'f1@k', 'ap',  # ap - average_precision_score
    'balanced_accuracy', 'log_loss'
}

MULTI_METRICS: Set[str] = {
    'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
    'precision@k', 'recall@k', 'f1@k', 'ap',
    'balanced_accuracy', 'log_loss', 'top_k_accuracy'
}

REGRESSION_METRICS: Set[str] = {
    'mse', 'rmse', 'mae', 'r2',
    'mape', 'msle', 'rmsle'
}

# Словарь для быстрого доступа к метрикам по типу задачи
TASK_METRICS: Dict[str, Set[str]] = {
    'binary': BINARY_METRICS,
    'multi': MULTI_METRICS,
    'regression': REGRESSION_METRICS
}


def parse_metric_string(metric_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Парсит строку метрики с параметрами.

    Формат: "metric_name;param1=value1;param2=value2"

    Args:
        metric_str: Строка с названием метрики и параметрами

    Returns:
        tuple: (имя метрики, словарь параметров)
    """
    parts = metric_str.split(';')
    name = parts[0]
    params = {}

    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            # Попытка преобразования значения
            try:
                # Проверяем, является ли число
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                # Check for boolean strings
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                else:
                    params[key] = value # Keep as string if not number or boolean

    return name, params


class Metrics:
    """
    Класс для представления метрики с её параметрами и функцией расчёта.
    """
    def __init__(self, metric_name_with_params: str):
        """
        Инициализация метрики.

        Args:
            name: Имя метрики
            **params: Параметры метрики
        """
        self.metric_name_with_params = metric_name_with_params
        self.name, self.params = parse_metric_string(metric_name_with_params)

    # Всегда предполагаем что в calculate приходят y_pred сырые значения,
    # то есть вероятности классов или регрессионные значения или распределение вероятностей по классам
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Расчёт значения метрики.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения (вероятности классов или регрессионные значения)

        Returns:
            float: Значение метрики
        """
        # Преобразование данных для метрик, требующих классы
        y_pred_classes = None
        if self.name in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
            if y_pred.ndim > 1 and y_pred.shape[1] > 1: # Мультикласс
                y_pred_classes = np.argmax(y_pred, axis=1)
            elif y_pred.ndim == 1 or y_pred.shape[1] == 1: # Бинарный (или вероятности одного класса)
                 # Ensure y_pred is 1D array for thresholding
                _y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
                threshold = self.params.get('threshold', 0.5)
                y_pred_classes = (_y_pred >= threshold).astype(int)
            else:
                 y_pred_classes = y_pred # Already classes? Assume correct format


        # --- Метрики классификации (требующие классы) ---
        if self.name == 'accuracy':
            return accuracy_score(y_true, y_pred_classes, **self.params)
        elif self.name == 'f1':
            # 'average' parameter handled by self.params
            return f1_score(y_true, y_pred_classes, **self.params)
        elif self.name == 'precision':
             # 'average' parameter handled by self.params
            return precision_score(y_true, y_pred_classes, **self.params)
        elif self.name == 'recall':
             # 'average' parameter handled by self.params
            return recall_score(y_true, y_pred_classes, **self.params)
        elif self.name == 'balanced_accuracy':
            # 'adjusted' параметр специфичен для sklearn
            return balanced_accuracy_score(y_true, y_pred_classes, adjusted=self.params.get('adjusted', False))

        # --- Метрики классификации (требующие вероятности/скоры) ---
        elif self.name == 'roc_auc':
            if y_pred.ndim > 1 and y_pred.shape[1] > 1: # Мультикласс
                # 'ovo'/'ovr' handled by params if needed
                return roc_auc_score(y_true, y_pred, **self.params)
            else: # Бинарный
                 # Ensure y_pred is 1D array of scores/probabilities
                _y_pred_roc = y_pred.flatten() if y_pred.ndim > 1 else y_pred
                return roc_auc_score(y_true, _y_pred_roc, **self.params)
        elif self.name == 'ap': # average_precision_score
            # 'average' параметр handled by self.params
             # Binary case y_pred shape could be (n,) or (n, 1)
            _y_pred_for_ap = y_pred
            if y_pred.ndim > 1 and y_pred.shape[1] > 1: # Multiclass, y_true should be one-hot for micro/macro? No, sklearn handles label indicators
                 pass # Keep y_pred as is for multiclass
            elif y_pred.ndim > 1 and y_pred.shape[1] == 1: # Binary (n, 1) -> (n,)
                 _y_pred_for_ap = y_pred.flatten()
            # else: binary (n,), keep as is

            # Ensure y_true format matches expectation based on average parameter if needed
            # Sklearn's average_precision_score usually handles label indicator format for multiclass directly
            return average_precision_score(y_true, _y_pred_for_ap, **self.params)

        elif self.name == 'log_loss':
             # Requires probabilities. Sklearn handles binary/multiclass.
             # Ensure y_pred has shape (n_samples, n_classes) for multiclass
             # and (n_samples,) for binary (or (n_samples, 2) but only proba of positive class needed)
             _y_pred_for_logloss = y_pred
             is_binary = len(np.unique(y_true)) <= 2 # Basic check, might need refinement
             if is_binary and y_pred.ndim > 1 and y_pred.shape[1] == 2:
                 _y_pred_for_logloss = y_pred[:, 1] # Use proba of positive class
             elif is_binary and y_pred.ndim > 1 and y_pred.shape[1] == 1:
                 _y_pred_for_logloss = y_pred.flatten()

             # Clip preds for numerical stability - log_loss handles eps parameter
             # eps = np.finfo(y_pred.dtype).eps
             # _y_pred_for_logloss = np.clip(_y_pred_for_logloss, eps, 1 - eps)

             return log_loss(y_true, _y_pred_for_logloss, **self.params) # eps handled by default
        elif self.name == 'top_k_accuracy':
            # Requires probabilities y_pred shape (n_samples, n_classes)
            # Parameter k is integer, handled by self.params
            k = self.params.get('k')
            if k is None:
                raise ValueError("Parameter 'k' is required for top_k_accuracy")
            if not isinstance(k, int):
                 raise ValueError(f"Parameter 'k' must be an integer for top_k_accuracy, got {k}")
            if y_pred.ndim == 1:
                 raise ValueError("top_k_accuracy requires class probabilities (y_pred with shape (n_samples, n_classes))")
             # Ensure labels parameter is correctly passed if needed for specific classes
            labels = self.params.get('labels', None) # Or derive from y_true/y_pred if necessary
            if labels is None and y_pred.shape[1] > 1: # Only derive labels if multiclass probabilities are provided
                labels = np.arange(y_pred.shape[1]) # Assume classes 0 to n_classes-1

            return top_k_accuracy_score(y_true, y_pred, k=k, labels=labels, **{p:v for p,v in self.params.items() if p not in ['k', 'labels']})


        # --- Метрики @k (на топ K% выборки) ---
        elif self.name == 'precision@k':
            k_perc = self.params.get('k')
            if k_perc is None: raise ValueError("Parameter 'k' (percentage) is required for precision@k")
            return self._precision_at_k(y_true, y_pred, k=float(k_perc))
        elif self.name == 'recall@k':
            k_perc = self.params.get('k')
            if k_perc is None: raise ValueError("Parameter 'k' (percentage) is required for recall@k")
            return self._recall_at_k(y_true, y_pred, k=float(k_perc))
        elif self.name == 'f1@k':
            k_perc = self.params.get('k')
            if k_perc is None: raise ValueError("Parameter 'k' (percentage) is required for f1@k")
            return self._f1_at_k(y_true, y_pred, k=float(k_perc))


        # --- Метрики регрессии ---
        elif self.name == 'mse':
            return mean_squared_error(y_true, y_pred, **self.params)
        elif self.name == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred, **self.params))
        elif self.name == 'mae':
            return mean_absolute_error(y_true, y_pred, **self.params)
        elif self.name == 'r2':
            return r2_score(y_true, y_pred, **self.params)
        elif self.name == 'mape':
             # Handle potential zero values in y_true if necessary, sklearn does this
            return mean_absolute_percentage_error(y_true, y_pred, **self.params)
        elif self.name == 'msle':
            # Requires non-negative y_true and y_pred
            if np.any(y_true < 0) or np.any(y_pred < 0):
                 # Adjust negative predictions to 0 or raise error? Let's adjust for robustness.
                 print("Warning: Negative values detected for MSLE calculation. Clipping predictions to 0.")
                 y_pred = np.maximum(y_pred, 0)
                 if np.any(y_true < 0): # True values cannot be negative
                     raise ValueError("MSLE requires non-negative true values (y_true).")

            return mean_squared_log_error(y_true, y_pred, **self.params)
        elif self.name == 'rmsle':
            # Requires non-negative y_true and y_pred
            if np.any(y_true < 0) or np.any(y_pred < 0):
                 # Adjust negative predictions to 0 or raise error? Let's adjust.
                 print("Warning: Negative values detected for RMSLE calculation. Clipping predictions to 0.")
                 y_pred = np.maximum(y_pred, 0)
                 if np.any(y_true < 0): # True values cannot be negative
                     raise ValueError("RMSLE requires non-negative true values (y_true).")

            # Calculate MSLE first
            msle_val = mean_squared_log_error(y_true, y_pred, **self.params)
            return np.sqrt(msle_val)


        else:
            raise ValueError(f"Unknown metric: {self.name}")

    @staticmethod
    def _get_scores_for_k_metrics(y_scores: np.ndarray) -> np.ndarray:
        """Helper to extract relevant scores for @k metrics (binary/multiclass)."""
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            # Multiclass: Use probability of the positive class (assumed class 1)
            # Or handle differently based on specific needs?
            # For now, default to class 1 for recall/precision @ k concept.
            # This might need refinement depending on the exact use case for multiclass @k.
            if y_scores.shape[1] == 2: # Binary classifier outputting two columns
                return y_scores[:, 1]
            else:
                # What score to use for multiclass? This is ambiguous.
                # Defaulting to max probability across classes as the 'score'.
                # Or maybe it should be applied per class?
                # Sticking to max probability for now.
                # Consider adding a parameter to specify which class probability to use or how to aggregate.
                print("Warning: Using max probability as score for multiclass @k metrics. Interpretation might vary.")
                return np.max(y_scores, axis=1)
        elif y_scores.ndim > 1 and y_scores.shape[1] == 1:
             return y_scores.flatten() # Case like (n, 1)
        else:
            return y_scores # Assumed binary (n,) or already processed scores

    def _recall_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """
        Recall at k percent of samples.
        Assumes binary classification (0/1) or treats multiclass y_true as binary based on the positive class implicitly chosen by _get_scores_for_k_metrics.
        """
        assert 0.0 <= k <= 1.0, "k должно быть от 0 до 1"
        y_true = np.asarray(y_true)
        scores = self._get_scores_for_k_metrics(np.asarray(y_scores)) # Get relevant scores

        # Identify positive class label (assuming 1 for binary, needs care for multiclass)
        # Let's assume the goal is recall of class '1' if y_true is binary-like (0s and 1s)
        positive_label = 1
        if len(np.unique(y_true)) > 2:
             # How to handle multiclass recall@k? Calculate for a specific class?
             # For now, continue assuming we want recall for class '1' vs rest.
             print(f"Warning: Calculating recall@k assuming positive class is {positive_label} for potentially multiclass true labels.")

        num_positive_true = np.sum(y_true == positive_label)
        if num_positive_true == 0:
             # If there are no true positive samples, recall is undefined or 0/0.
             # Scikit-learn returns 0 in this case for precision/recall if no true samples exist.
            return 0.0

        top_n = int(len(scores) * k)
        if top_n == 0:
            return 0.0

        # Indices of top N scores
        top_indices = np.argsort(scores)[::-1][:top_n]

        # Number of true positives among top N predictions
        num_true_positives_at_k = np.sum(y_true[top_indices] == positive_label)

        return num_true_positives_at_k / num_positive_true


    def _precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """
        Precision at k percent of samples.
        Assumes binary classification (0/1) or treats multiclass y_true as binary based on the positive class implicitly chosen by _get_scores_for_k_metrics.
        """
        assert 0.0 <= k <= 1.0, "k должно быть от 0 до 1"
        y_true = np.asarray(y_true)
        scores = self._get_scores_for_k_metrics(np.asarray(y_scores)) # Get relevant scores

        top_n = int(len(scores) * k)
        if top_n == 0:
             # If we select 0 items, precision is arguably undefined or 0/0.
             # Scikit-learn returns 0 if TP+FP = 0.
            return 0.0

        # Indices of top N scores
        top_indices = np.argsort(scores)[::-1][:top_n]

        # Identify positive class label (assuming 1 for binary)
        positive_label = 1
        if len(np.unique(y_true)) > 2:
             print(f"Warning: Calculating precision@k assuming positive class is {positive_label} for potentially multiclass true labels.")


        # Number of true positives among top N predictions
        num_true_positives_at_k = np.sum(y_true[top_indices] == positive_label)

        return num_true_positives_at_k / top_n

    def _f1_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """F1 score at k percent of samples."""
        precision_k = self._precision_at_k(y_true, y_scores, k)
        recall_k = self._recall_at_k(y_true, y_scores, k)

        # F1 = 2 * (precision * recall) / (precision + recall)
        denominator = precision_k + recall_k
        if denominator == 0:
            return 0.0
        else:
            return 2 * (precision_k * recall_k) / denominator

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"Metric(name='{self.name}', params={{{params_str}}})"