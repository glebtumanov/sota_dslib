import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
from sklearn.metrics import *

# Определяем доступные метрики для каждого типа задачи
BINARY_METRICS: Set[str] = {
    'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
    'precision@k', 'recall@k', 'ap'  # ap - average_precision_score
}

MULTI_METRICS: Set[str] = {
    'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
    'precision@k', 'recall@k', 'ap'
}

REGRESSION_METRICS: Set[str] = {
    'mse', 'rmse', 'mae', 'r2'
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
                params[key] = value

    return name, params


class Metrics:
    """
    Класс для представления метрики с её параметрами и функцией расчёта.
    """
    def __init__(self, name: str, **params):
        """
        Инициализация метрики.

        Args:
            name: Имя метрики
            **params: Параметры метрики
        """
        self.name = name
        self.params = params

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
        # Преобразование данных в зависимости от типа метрики
        if self.name in ['accuracy', 'f1', 'precision', 'recall']:
            # Для метрик, требующих классов, а не вероятностей
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                # Мультиклассовая классификация - берём argmax
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                # Бинарная классификация - округляем
                y_pred_classes = np.round(y_pred).astype(int)

            if self.name == 'accuracy':
                return accuracy_score(y_true, y_pred_classes, **self.params)
            elif self.name == 'f1':
                return f1_score(y_true, y_pred_classes, **self.params)
            elif self.name == 'precision':
                return precision_score(y_true, y_pred_classes, **self.params)
            elif self.name == 'recall':
                return recall_score(y_true, y_pred_classes, **self.params)

        elif self.name == 'roc_auc':
            # Для ROC AUC
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                # Мультиклассовая классификация
                return roc_auc_score(y_true, y_pred, multi_class='ovr', **self.params)
            else:
                # Бинарная классификация
                return roc_auc_score(y_true, y_pred, **self.params)

        elif self.name == 'ap':
            # average_precision_score
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                # Мультиклассовая классификация
                return average_precision_score(y_true, y_pred, **self.params)
            else:
                # Бинарная классификация
                return average_precision_score(y_true, y_pred, **self.params)

        elif self.name in ['mse', 'rmse', 'mae', 'r2']:
            # Для регрессионных метрик
            if self.name == 'mse':
                return mean_squared_error(y_true, y_pred, **self.params)
            elif self.name == 'rmse':
                return np.sqrt(mean_squared_error(y_true, y_pred, **self.params))
            elif self.name == 'mae':
                return mean_absolute_error(y_true, y_pred, **self.params)
            elif self.name == 'r2':
                return r2_score(y_true, y_pred, **self.params)

        elif self.name == 'precision@k':
            return self._precision_at_k(y_true, y_pred, self.params.get('k', 0.1))

        elif self.name == 'recall@k':
            return self._recall_at_k(y_true, y_pred, self.params.get('k', 0.1))

        else:
            raise ValueError(f"Unknown metric: {self.name}")

    @staticmethod
    def _recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """
        Recall at k percent of samples.

        Args:
            y_true: True labels
            y_scores: Predicted scores
            k: Top fraction of predictions to consider

        Returns:
            float: Recall at k
        """
        assert 0.0 <= k <= 1.0, "k должно быть от 0 до 1"
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # Обработка многомерных y_scores (для мультиклассовой классификации)
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            # Если нам интересен определенный класс, берем его вероятности
            # По умолчанию берем положительный класс (1)
            y_scores = y_scores[:, 1]

        top_n = int(len(y_scores) * k)
        if top_n == 0 or np.sum(y_true) == 0:
            return 0.0

        top_indices = np.argsort(y_scores)[::-1][:top_n]
        return np.sum(y_true[top_indices]) / np.sum(y_true)

    @staticmethod
    def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
        """
        Precision at k percent of samples.

        Args:
            y_true: True labels
            y_scores: Predicted scores
            k: Top fraction of predictions to consider

        Returns:
            float: Precision at k
        """
        assert 0.0 <= k <= 1.0, "k должно быть от 0 до 1"
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        # Обработка многомерных y_scores (для мультиклассовой классификации)
        if y_scores.ndim > 1 and y_scores.shape[1] > 1:
            # Если нам интересен определенный класс, берем его вероятности
            # По умолчанию берем положительный класс (1)
            y_scores = y_scores[:, 1]

        top_n = int(len(y_scores) * k)
        if top_n == 0:
            return 0.0

        top_indices = np.argsort(y_scores)[::-1][:top_n]
        return np.sum(y_true[top_indices]) / top_n

    def __repr__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"Metric(name='{self.name}', params={{{params_str}}})"