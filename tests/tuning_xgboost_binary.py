#! /usr/bin/env python3
import pandas as pd
import torch
import optuna
import optuna.logging
import sys
sys.path.append('../')
import getpass
user_name = getpass.getuser()

from models.estimators.xgboost_estimator import XGBoostBinary
from hp_tuning import run_tuning

TRAIN_DATA_PATH = f'/home/{user_name}/data/shadow_smz_2504/train_prep.parquet'
CAT_FEATURES_PATH = f'/home/{user_name}/data/shadow_smz_2504/final_categorical_list.txt'
FEATURES_PATH = f'/home/{user_name}/data/shadow_smz_2504/final_feature_list.txt'
TARGET_COL = 'target'
INDEX_COL = 'epk_id'
STUDY_NAME = 'xgboost-shadow-smz'

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

ESTIMATOR_CLASS = XGBoostBinary

STATIC_MODEL_PARAMS = {
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'n_jobs': 4,
    'random_state': SPLIT_RANDOM_STATE,
    'verbosity': 0,
    'eval_metric': 'auc'
}

def define_parameter_space(trial: optuna.Trial) -> dict:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0)
    }

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
