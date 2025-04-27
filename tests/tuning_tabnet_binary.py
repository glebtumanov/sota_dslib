#! /usr/bin/env python3
import pandas as pd
import torch
import optuna # Нужно для trial.suggest_*
import optuna.logging
import sys
sys.path.append('../') # Добавляем корневую папку проекта в PYTHONPATH

# --- Импорты ---
# Импортируем класс модели
from models.estimators.tabnet_estimator import TabNetBinary # <-- Меняем на TabNetBinary
# Импортируем библиотеку для тюнинга
from hp_tuning import run_tuning

# --- Конфигурация Данных ---
TRAIN_DATA_PATH = '/www/dslib/spark_sota_modeling/dataset/home-credit-default-risk/train.parquet' # <-- Новый путь
# Опционально: файл со списком категориальных фичей (если None, не используется)
CAT_FEATURES_PATH = '/www/dslib/spark_sota_modeling/dataset/home-credit-default-risk/categorical_features.txt' # <-- Новый путь
TARGET_COL = 'target' # <-- Новая целевая колонка
INDEX_COL = 'sk_id_curr' # <-- Новая колонка ID

# --- Конфигурация Разделения Данных ---
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
STRATIFY_COL = TARGET_COL # <-- Указываем колонку для стратификации

# --- Конфигурация Optuna ---
STUDY_NAME = 'tabnet-binary-home-credit' # <-- Новое имя исследования
STORAGE_DIR = 'tuning' # Директория для SQLite и Excel
N_TRIALS = 30 # <-- Можно увеличить кол-во попыток для бинарной классификации
TIMEOUT_SECONDS = None # None - без ограничения времени
METRIC_TO_OPTIMIZE = 'roc_auc' # <-- Метрика для оптимизации
DIRECTION = 'maximize' # <-- Направление оптимизации
OPTUNA_LOG_LEVEL = optuna.logging.WARNING # Уровень логирования Optuna (WARNING или INFO)

# --- Конфигурация Модели ---
ESTIMATOR_CLASS = TabNetBinary # <-- Класс модели

# Параметры модели, которые НЕ будут подбираться Optuna, а будут фиксированы
# во время работы objective функции
# Можно оставить те же, что и для регрессии, или скорректировать при необходимости
STATIC_MODEL_PARAMS = { # <-- Переименовываем и используем как единый набор
    'epochs': 100,
    'early_stopping_patience': 10,
    'reducelronplateau_patience': 5,
    'reducelronplateau_factor': 0.3,
    'verbose': False, # <-- verbose для objective
    'random_state': SPLIT_RANDOM_STATE,
    'batch_size': 4096,
}

# Функция, определяющая пространство поиска гиперпараметров для Optuna
# Используем то же пространство, что и для регрессии, т.к. параметры схожи
def define_parameter_space(trial: optuna.Trial) -> dict:
    return {
        'd_model': trial.suggest_categorical('d_model', [8, 16, 24, 32]),
        'n_steps': trial.suggest_int('n_steps', 3, 7),
        'decision_dim': trial.suggest_categorical('decision_dim', [64, 96, 128]),
        'n_shared': trial.suggest_int('n_shared', 1, 4),
        'n_independent': trial.suggest_int('n_independent', 1, 4),
        'gamma': trial.suggest_float('gamma', 1.0, 2.5),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
        'glu_dropout': trial.suggest_float('glu_dropout', 0.0, 0.4),
        'dropout_emb': trial.suggest_float('dropout_emb', 0.0, 0.4),
        'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [128, 256, 512]),
        'att_momentum': trial.suggest_float('att_momentum', 0.01, 0.5),
        'glu_momentum': trial.suggest_float('glu_momentum', 0.01, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    }

# --- Конфигурация Финального Обучения ---
TRAIN_FINAL_MODEL_FLAG = True # Обучать ли модель с лучшими параметрами после подбора

# Параметры для переопределения при финальном обучении
# Если оставить None, будут использованы значения из STATIC_MODEL_PARAMS
FINAL_EPOCHS = 200
FINAL_EARLY_STOPPING = 20
FINAL_VERBOSE = True


# --- Запуск Тюнинга ---
if __name__ == "__main__":
    run_tuning(
        # Данные
        train_data_path=TRAIN_DATA_PATH,
        cat_features_path=CAT_FEATURES_PATH,
        target_col=TARGET_COL,
        index_col=INDEX_COL,
        stratify_col=STRATIFY_COL, # <-- Передаем колонку для стратификации
        # Разделение
        test_size=TEST_SIZE,
        split_random_state=SPLIT_RANDOM_STATE,
        # Optuna
        study_name=STUDY_NAME,
        storage_dir=STORAGE_DIR,
        n_trials=N_TRIALS,
        timeout_seconds=TIMEOUT_SECONDS,
        metric_to_optimize=METRIC_TO_OPTIMIZE,
        direction=DIRECTION,
        optuna_log_level=OPTUNA_LOG_LEVEL,
        # Модель и параметры
        estimator_class=ESTIMATOR_CLASS,
        param_space_func=define_parameter_space,
        static_params_objective=STATIC_MODEL_PARAMS, # <-- Передаем единый набор
        # Финальное обучение
        train_final_flag=TRAIN_FINAL_MODEL_FLAG,
        final_epochs=FINAL_EPOCHS,
        final_early_stopping=FINAL_EARLY_STOPPING,
        final_verbose=FINAL_VERBOSE
    )