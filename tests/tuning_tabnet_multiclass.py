#! /usr/bin/env python3
import pandas as pd
import torch
import optuna
import optuna.logging
import sys
sys.path.append('../')

from models.estimators.tabnet_estimator import TabNetMulticlass
from hp_tuning import run_tuning

TRAIN_DATA_PATH = '/www/dslib/spark_sota_modeling/dataset/forest-cover-type/train.parquet'
CAT_FEATURES_PATH = '/www/dslib/spark_sota_modeling/dataset/forest-cover-type/categorical_features.txt'
TARGET_COL = 'cover_type'
INDEX_COL = None

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
STRATIFY_COL = TARGET_COL

STUDY_NAME = 'tabnet-multiclass-forest-cover'
STORAGE_DIR = 'tuning'
N_TRIALS = 30
TIMEOUT_SECONDS = None
METRIC_TO_OPTIMIZE = 'accuracy'
DIRECTION = 'maximize'
OPTUNA_LOG_LEVEL = optuna.logging.WARNING
ROUNDED_OUTPUT = False

ESTIMATOR_CLASS = TabNetMulticlass

STATIC_MODEL_PARAMS = {
    'epochs': 150,
    'early_stopping_patience': 15,
    'reducelronplateau_patience': 5,
    'reducelronplateau_factor': 0.3,
    'verbose': False,
    'random_state': SPLIT_RANDOM_STATE,
    'batch_size': 16384,
    'n_classes': 7,
}

def define_parameter_space(trial: optuna.Trial) -> dict:
    return {
        'd_model': trial.suggest_categorical('d_model', [4, 8, 16, 32]),
        'n_steps': trial.suggest_int('n_steps', 3, 8),
        'decision_dim': trial.suggest_categorical('decision_dim', [64, 96, 128, 192]),
        'n_shared': trial.suggest_int('n_shared', 1, 4),
        'n_independent': trial.suggest_int('n_independent', 1, 4),
        'gamma': trial.suggest_float('gamma', 1.0, 2.5),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
        'dropout_glu': trial.suggest_float('dropout_glu', 0.001, 0.3, log=True),
        'dropout_emb': trial.suggest_float('dropout_emb', 0.001, 0.3, log=True),
        'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [128, 256, 512]),
        'momentum_att': trial.suggest_float('momentum_att', 0.01, 0.5),
        'momentum_glu': trial.suggest_float('momentum_glu', 0.01, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    }


if __name__ == "__main__":
    run_tuning(
        # Данные
        train_data_path=TRAIN_DATA_PATH,
        cat_features_path=CAT_FEATURES_PATH,
        target_col=TARGET_COL,
        index_col=INDEX_COL,
        stratify_col=STRATIFY_COL,
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
        static_params_objective=STATIC_MODEL_PARAMS,
        rounded_output=ROUNDED_OUTPUT
    )