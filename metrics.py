import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
from sklearn.metrics import *
"""
Доступные метрики для различных типов задач:

Бинарная классификация (BINARY_METRICS):
- accuracy: точность классификации
  Параметры: нет
  Значение: доля правильных предсказаний (TP + TN) / (TP + TN + FP + FN)
  Источник: sklearn.metrics.accuracy_score

- f1: F1-score
  Параметры:
    - average: str, default='binary' ('binary', 'micro', 'macro', 'weighted')
    - pos_label: int, default=1
    - sample_weight: array-like, default=None
  Значение: гармоническое среднее precision и recall
  Источник: sklearn.metrics.f1_score

- precision: точность
  Параметры:
    - average: str, default='binary'
    - pos_label: int, default=1
    - sample_weight: array-like, default=None
  Значение: TP / (TP + FP)
  Источник: sklearn.metrics.precision_score

- recall: полнота
  Параметры:
    - average: str, default='binary'
    - pos_label: int, default=1
    - sample_weight: array-like, default=None
  Значение: TP / (TP + FN)
  Источник: sklearn.metrics.recall_score

- roc_auc: площадь под ROC-кривой
  Параметры:
    - average: str, default='macro'
    - sample_weight: array-like, default=None
    - max_fpr: float, default=None
  Значение: площадь под кривой ROC (0-1)
  Источник: sklearn.metrics.roc_auc_score

- precision@k: точность на k лучших предсказаниях
  Параметры:
    - k: float, default=0.1 (доля от общего количества)
  Значение: точность на top-k предсказаниях
  Источник: собственная реализация

- recall@k: полнота на k лучших предсказаниях
  Параметры:
    - k: float, default=0.1 (доля от общего количества)
  Значение: полнота на top-k предсказаниях
  Источник: собственная реализация

- ap: средняя точность (average precision)
  Параметры:
    - average: str, default='binary' ('binary', 'micro', 'macro', 'weighted')
    - pos_label: int, default=1
    - sample_weight: array-like, default=None
  Значение: средняя точность по всем порогам
  Источник: sklearn.metrics.average_precision_score

Многоклассовая классификация (MULTI_METRICS):
[Аналогичные метрики как в бинарной классификации, но с поддержкой multi-class]

Регрессия (REGRESSION_METRICS):
- mse: среднеквадратичная ошибка
  Параметры:
    - sample_weight: array-like, default=None
  Значение: среднее квадратов ошибок
  Источник: sklearn.metrics.mean_squared_error

- rmse: корень из среднеквадратичной ошибки
  Параметры:
    - sample_weight: array-like, default=None
  Значение: корень из MSE
  Источник: собственная реализация на основе MSE

- mae: средняя абсолютная ошибка
  Параметры:
    - sample_weight: array-like, default=None
  Значение: среднее абсолютных ошибок
  Источник: sklearn.metrics.mean_absolute_error

- r2: коэффициент детерминации
  Параметры:
    - sample_weight: array-like, default=None
    - multioutput: str, default='uniform_average'
  Значение: доля объясненной дисперсии (0-1)
  Источник: sklearn.metrics.r2_score

Примеры использования в конфиге:
metrics:
  - "f1;average=binary"
  - "precision@k;k=0.01"
  - "ap;average=micro"
  - "r2;multioutput=uniform_average"
  - "f1;average=weighted;sample_weight=balanced"
"""

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