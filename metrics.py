import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, mean_squared_error, mean_absolute_error, r2_score,
    balanced_accuracy_score, log_loss, top_k_accuracy_score,
    mean_absolute_percentage_error, mean_squared_log_error
)

# Направления оптимизации метрик
MAXIMIZE = 1  # Чем больше, тем лучше
MINIMIZE = -1  # Чем меньше, тем лучше

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

# Направления оптимизации для метрик (MAXIMIZE - чем больше, тем лучше; MINIMIZE - чем меньше, тем лучше)
METRIC_DIRECTIONS: Dict[str, int] = {
    # Метрики классификации (больше - лучше)
    'accuracy': MAXIMIZE,
    'f1': MAXIMIZE,
    'precision': MAXIMIZE,
    'recall': MAXIMIZE,
    'balanced_accuracy': MAXIMIZE,
    'roc_auc': MAXIMIZE,
    'ap': MAXIMIZE,
    'top_k_accuracy': MAXIMIZE,
    'precision@k': MAXIMIZE,
    'recall@k': MAXIMIZE,
    'f1@k': MAXIMIZE,

    # Метрики классификации (меньше - лучше)
    'log_loss': MINIMIZE,

    # Метрики регрессии (меньше - лучше)
    'mse': MINIMIZE,
    'rmse': MINIMIZE,
    'mae': MINIMIZE,
    'mape': MINIMIZE,
    'msle': MINIMIZE,
    'rmsle': MINIMIZE,

    # Метрики регрессии (больше - лучше)
    'r2': MAXIMIZE
}


def parse_metric_string(metric_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Парсит строку метрики с параметрами в формате: "metric_name;param1=value1;param2=value2"

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
            # Преобразование значения к соответствующему типу
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                else:
                    params[key] = value

    return name, params


class Metrics:
    """Класс для представления метрики с её параметрами и функцией расчёта."""

    def __init__(self, metric_name_with_params: str):
        """
        Инициализация метрики.

        Args:
            metric_name_with_params: Имя метрики с параметрами в формате "имя;параметр1=значение1;..."
        """
        self.metric_name_with_params = metric_name_with_params
        self.name, self.params = parse_metric_string(metric_name_with_params)

        # Словарь функций расчета метрик по их именам
        self._metric_functions = {
            # Метрики классификации (требующие классы)
            'accuracy': self._calc_accuracy,
            'f1': self._calc_f1,
            'precision': self._calc_precision,
            'recall': self._calc_recall,
            'balanced_accuracy': self._calc_balanced_accuracy,

            # Метрики классификации (требующие вероятности/скоры)
            'roc_auc': self._calc_roc_auc,
            'ap': self._calc_ap,
            'log_loss': self._calc_log_loss,
            'top_k_accuracy': self._calc_top_k_accuracy,

            # Метрики @k
            'precision@k': self._calc_precision_at_k,
            'recall@k': self._calc_recall_at_k,
            'f1@k': self._calc_f1_at_k,

            # Метрики регрессии
            'mse': self._calc_mse,
            'rmse': self._calc_rmse,
            'mae': self._calc_mae,
            'r2': self._calc_r2,
            'mape': self._calc_mape,
            'msle': self._calc_msle,
            'rmsle': self._calc_rmsle
        }

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Расчёт значения метрики.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения (вероятности классов или регрессионные значения)

        Returns:
            float: Значение метрики
        """
        metric_func = self._metric_functions.get(self.name)
        if not metric_func:
            raise ValueError(f"Неизвестная метрика: {self.name}")

        return metric_func(y_true, y_pred)

    def _prepare_classification_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для метрик классификации.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения/вероятности

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_true, y_pred_classes)
        """
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:  # Мультиклассовая классификация
            y_pred_classes = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 1 or y_pred.shape[1] == 1:  # Бинарная классификация
            _y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
            threshold = self.params.get('threshold', 0.5)
            y_pred_classes = (_y_pred >= threshold).astype(int)
        else:
            y_pred_classes = y_pred  # Предполагаем, что это уже классы

        return y_true, y_pred_classes

    def _prepare_regression_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для метрик регрессии.

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения

        Returns:
            Tuple[np.ndarray, np.ndarray]: (y_true, y_pred)
        """
        # Для регрессии обычно просто приводим к плоскому виду
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        return y_true, y_pred

    # Метрики классификации (требующие классы)
    def _calc_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred_classes = self._prepare_classification_data(y_true, y_pred)
        return accuracy_score(y_true, y_pred_classes, **self.params)

    def _calc_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred_classes = self._prepare_classification_data(y_true, y_pred)
        return f1_score(y_true, y_pred_classes, **self.params)

    def _calc_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred_classes = self._prepare_classification_data(y_true, y_pred)
        return precision_score(y_true, y_pred_classes, **self.params)

    def _calc_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred_classes = self._prepare_classification_data(y_true, y_pred)
        return recall_score(y_true, y_pred_classes, **self.params)

    def _calc_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred_classes = self._prepare_classification_data(y_true, y_pred)
        return balanced_accuracy_score(y_true, y_pred_classes, adjusted=self.params.get('adjusted', False))

    # Метрики классификации (требующие вероятности/скоры)
    def _calc_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:  # Мультиклассовая классификация
            return roc_auc_score(y_true, y_pred, **self.params)
        else:  # Бинарная классификация
            _y_pred_roc = y_pred.flatten() if y_pred.ndim > 1 else y_pred
            return roc_auc_score(y_true, _y_pred_roc, **self.params)

    def _calc_ap(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _y_pred_for_ap = y_pred
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            _y_pred_for_ap = y_pred.flatten()
        return average_precision_score(y_true, _y_pred_for_ap, **self.params)

    def _calc_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _y_pred_for_logloss = y_pred
        is_binary = len(np.unique(y_true)) <= 2

        if is_binary and y_pred.ndim > 1 and y_pred.shape[1] == 2:
            _y_pred_for_logloss = y_pred[:, 1]  # Используем вероятность положительного класса
        elif is_binary and y_pred.ndim > 1 and y_pred.shape[1] == 1:
            _y_pred_for_logloss = y_pred.flatten()

        return log_loss(y_true, _y_pred_for_logloss, **self.params)

    def _calc_top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        k = self.params.get('k')
        if k is None:
            raise ValueError("Параметр 'k' обязателен для top_k_accuracy")

        if not isinstance(k, int):
            raise ValueError(f"Параметр 'k' должен быть целым числом для top_k_accuracy, получено {k}")

        if y_pred.ndim == 1:
            raise ValueError("top_k_accuracy требует вероятностей классов (размерность y_pred: (n_samples, n_classes))")

        labels = self.params.get('labels')
        if labels is None and y_pred.shape[1] > 1:
            labels = np.arange(y_pred.shape[1])

        filtered_params = {p: v for p, v in self.params.items() if p not in ['k', 'labels']}
        return top_k_accuracy_score(y_true, y_pred, k=k, labels=labels, **filtered_params)

    # Метрики @k
    def _get_scores_for_k_metrics(self, y_scores: np.ndarray) -> np.ndarray:
        """Извлекает соответствующие скоры для метрик @k."""
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            if y_scores.shape[1] == 2:  # Бинарный классификатор с двумя столбцами
                return y_scores[:, 1]
            else:
                # Для мультиклассового случая используем максимальную вероятность
                return np.max(y_scores, axis=1)
        elif y_scores.ndim > 1 and y_scores.shape[1] == 1:
            return y_scores.flatten()
        else:
            return y_scores

    def _calc_precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        k_perc = self.params.get('k')
        if k_perc is None:
            raise ValueError("Параметр 'k' (процент) обязателен для precision@k")

        return self._precision_at_k(y_true, y_pred, k=float(k_perc))

    def _calc_recall_at_k(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        k_perc = self.params.get('k')
        if k_perc is None:
            raise ValueError("Параметр 'k' (процент) обязателен для recall@k")

        return self._recall_at_k(y_true, y_pred, k=float(k_perc))

    def _calc_f1_at_k(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        k_perc = self.params.get('k')
        if k_perc is None:
            raise ValueError("Параметр 'k' (процент) обязателен для f1@k")

        return self._f1_at_k(y_true, y_pred, k=float(k_perc))

    def _precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """
        Точность на k проценте выборки.
        """
        assert 0.0 <= k <= 1.0, "k должно быть от 0 до 1"
        y_true = np.asarray(y_true)
        scores = self._get_scores_for_k_metrics(np.asarray(y_scores))

        top_n = int(len(scores) * k)
        if top_n == 0:
            return 0.0

        top_indices = np.argsort(scores)[::-1][:top_n]
        positive_label = 1

        if len(np.unique(y_true)) > 2:
            print(f"Внимание: вычисление precision@k предполагает, что положительный класс - {positive_label}")

        num_true_positives_at_k = np.sum(y_true[top_indices] == positive_label)
        return num_true_positives_at_k / top_n

    def _recall_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """
        Полнота на k проценте выборки.
        """
        assert 0.0 <= k <= 1.0, "k должно быть от 0 до 1"
        y_true = np.asarray(y_true)
        scores = self._get_scores_for_k_metrics(np.asarray(y_scores))

        positive_label = 1
        if len(np.unique(y_true)) > 2:
            print(f"Внимание: вычисление recall@k предполагает, что положительный класс - {positive_label}")

        num_positive_true = np.sum(y_true == positive_label)
        if num_positive_true == 0:
            return 0.0

        top_n = int(len(scores) * k)
        if top_n == 0:
            return 0.0

        top_indices = np.argsort(scores)[::-1][:top_n]
        num_true_positives_at_k = np.sum(y_true[top_indices] == positive_label)

        return num_true_positives_at_k / num_positive_true

    def _f1_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """F1-мера на k проценте выборки."""
        precision_k = self._precision_at_k(y_true, y_scores, k)
        recall_k = self._recall_at_k(y_true, y_scores, k)

        denominator = precision_k + recall_k
        if denominator == 0:
            return 0.0
        else:
            return 2 * (precision_k * recall_k) / denominator

    # Метрики регрессии
    def _calc_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)
        return mean_squared_error(y_true, y_pred, **self.params)

    def _calc_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)
        return np.sqrt(mean_squared_error(y_true, y_pred, **self.params))

    def _calc_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)
        return mean_absolute_error(y_true, y_pred, **self.params)

    def _calc_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)
        return r2_score(y_true, y_pred, **self.params)

    def _calc_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)
        return mean_absolute_percentage_error(y_true, y_pred, **self.params)

    def _calc_msle(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)

        if np.any(y_true < 0) or np.any(y_pred < 0):
            y_pred = np.maximum(y_pred, 0)
            if np.any(y_true < 0):
                raise ValueError("MSLE требует неотрицательных истинных значений (y_true).")

        return mean_squared_log_error(y_true, y_pred, **self.params)

    def _calc_rmsle(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = self._prepare_regression_data(y_true, y_pred)

        if np.any(y_true < 0) or np.any(y_pred < 0):
            y_pred = np.maximum(y_pred, 0)
            if np.any(y_true < 0):
                raise ValueError("RMSLE требует неотрицательных истинных значений (y_true).")

        msle_val = mean_squared_log_error(y_true, y_pred, **self.params)
        return np.sqrt(msle_val)

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"Metric(name='{self.name}', params={{{params_str}}})"


def get_best_model_by_metric(metrics_results: Dict[str, Dict[str, float]], main_metric: str) -> Tuple[str, float]:
    """
    Определяет лучшую модель по указанной метрике с учетом направления оптимизации.
    Поддерживает метрики с параметрами в формате "metric_name;param1=value1;param2=value2"

    Args:
        metrics_results: Словарь вида {модель: {метрика: значение}}
        main_metric: Ключ метрики для сравнения (может содержать параметры)

    Returns:
        Tuple[str, float]: (имя лучшей модели, значение метрики)
    """
    if not metrics_results:
        raise ValueError("metrics_results не может быть пустым")

    # Парсим метрику и её параметры
    metric_name, _ = parse_metric_string(main_metric)

    # Получаем направление оптимизации метрики (по умолчанию MAXIMIZE)
    direction = METRIC_DIRECTIONS.get(metric_name, MAXIMIZE)

    # Находим лучшую модель
    if direction == MAXIMIZE:
        best_model = max(metrics_results, key=lambda m: metrics_results[m].get(main_metric, 0))
    else:  # MINIMIZE
        best_model = min(metrics_results, key=lambda m: metrics_results[m].get(main_metric, float('inf')))

    return best_model, metrics_results[best_model].get(main_metric, 0)