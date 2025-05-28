#! /usr/bin/env python3
import pandas as pd
import torch
import optuna
import optuna.logging
import sys
sys.path.append('../')
import getpass
user_name = getpass.getuser()

from models.estimators.catboost_estimator import CatBoostBinary
from hp_tuning import run_tuning

TRAIN_DATA_PATH = f'/home/{user_name}/data/shadow_smz_2504/train_prep.parquet'
CAT_FEATURES_PATH = f'/home/{user_name}/data/shadow_smz_2504/final_categorical_list.txt'
FEATURES_PATH = f'/home/{user_name}/data/shadow_smz_2504/final_feature_list.txt'
TARGET_COL = 'target'
INDEX_COL = 'epk_id'
STUDY_NAME = 'catboost-shadow-smz'

TEST_SIZE = 0.4
SPLIT_RANDOM_STATE = 42
STRATIFY_COL = TARGET_COL

STORAGE_DIR = 'tuning'
N_TRIALS = 30
TIMEOUT_SECONDS = None
METRIC_TO_OPTIMIZE = 'AUC'
DIRECTION = 'maximize'
OPTUNA_LOG_LEVEL = optuna.logging.WARNING
ROUNDED_OUTPUT = False

ESTIMATOR_CLASS = CatBoostBinary

STATIC_MODEL_PARAMS = {
    'iterations': 1000,
    'early_stopping_rounds': 50,
    'thread_count': 4,
    'random_seed': SPLIT_RANDOM_STATE,
    'verbose': False,
    'eval_metric': 'AUC',
}

def define_parameter_space(trial: optuna.Trial) -> dict:
    # Определяем стратегию бутстрапа
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'Bayesian'])

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bootstrap_type': bootstrap_type,
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    }

    # Параметры зависимые от bootstrap_type
    if bootstrap_type == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.01, 10.0, log=True)
    else:  # Bernoulli
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

    # Параметры для Lossguide
    if params['grow_policy'] == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 10, 128)

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
