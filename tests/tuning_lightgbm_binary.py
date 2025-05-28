#! /usr/bin/env python3
import pandas as pd
import torch
import optuna
import optuna.logging
import sys
sys.path.append('../')
import getpass
user_name = getpass.getuser()

from models.estimators.lightgbm_estimator import LightGBMBinary
from hp_tuning import run_tuning

TRAIN_DATA_PATH = f'/home/{user_name}/data/shadow_smz_2504/train_prep.parquet'
CAT_FEATURES_PATH = f'/home/{user_name}/data/shadow_smz_2504/final_categorical_list.txt'
FEATURES_PATH = f'/home/{user_name}/data/shadow_smz_2504/final_feature_list.txt'
TARGET_COL = 'target'
INDEX_COL = 'epk_id'
STUDY_NAME = 'lightgbm-shadow-smz'

TEST_SIZE = 0.4
SPLIT_RANDOM_STATE = 42
STRATIFY_COL = TARGET_COL

STORAGE_DIR = 'tuning'
N_TRIALS = 30
TIMEOUT_SECONDS = None
METRIC_TO_OPTIMIZE = 'auc'
DIRECTION = 'maximize'
OPTUNA_LOG_LEVEL = optuna.logging.WARNING
ROUNDED_OUTPUT = False

ESTIMATOR_CLASS = LightGBMBinary

STATIC_MODEL_PARAMS = {
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'n_jobs': 4,
    'random_state': SPLIT_RANDOM_STATE,
    'verbose': -1,
    'metric': 'auc',
    'silent': True
}

def define_parameter_space(trial: optuna.Trial) -> dict:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 7, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # Добавляем ограничение: num_leaves не должен быть слишком большим для заданной max_depth
    # Теоретический максимум: 2^max_depth - 1
    params['num_leaves'] = min(params['num_leaves'], 2**params['max_depth'] - 1)

    return params

if __name__ == "__main__":
    run_tuning(
        # Данные
        train_data_path=TRAIN_DATA_PATH,
        cat_features_path=CAT_FEATURES_PATH,
        features_path=FEATURES_PATH,
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
