#!/usr/bin/env python3
# ссылка на скачивание датасета: https://www.kaggle.com/competitions/home-credit-default-risk/data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import gc
import warnings
import time # Added for timing memory optimization
import re
warnings.filterwarnings('ignore')


# Define data directories
DATA_DIR = './home-credit-default-risk'
DST_DIR = './home-credit-default-risk'

def normalize_column_name(col_name):
    """Normalize column name: lowercase and replace non-alphanumeric with underscores"""
    return re.sub(r'[^a-z0-9]', '_', str(col_name).lower())

def read_data(file_path):
    """Read a CSV file and display its shape"""
    df = pd.read_csv(file_path)
    # Convert column names to lowercase and replace all non-alphanumeric characters with underscores
    df.columns = [normalize_column_name(col) for col in df.columns]
    print(f"Read {os.path.basename(file_path)}, shape: {df.shape}")
    return df

def handle_missing_values(df, is_train=True, train_df=None):
    """Fill missing values in numerical columns with median"""
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().sum() > 0:
            if is_train:
                median_val = df[col].median()
            else:
                # Use train median for test data
                median_val = train_df[col].median()

            df[col] = df[col].fillna(median_val)
    return df

def encode_categorical_features(train_df, test_df):
    """Encode categorical features using LabelEncoder (fit only on train data)"""
    categorical_features = []

    # Identify object columns in the original train_df BEFORE any merge/aggregation
    original_object_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    for col in original_object_cols:
        if col in train_df.columns and col in test_df.columns: # Ensure column exists after merges
            categorical_features.append(normalize_column_name(col))

            # Fill missing values consistently
            train_df[col] = train_df[col].fillna('Unknown').astype(str)
            test_df[col] = test_df[col].fillna('Unknown').astype(str)

            # Fit encoder on combined unique values of train and test
            le = LabelEncoder()
            # Combine unique values, ensuring 'Unknown' is present
            unique_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
            le.fit(unique_values)

            # Transform both train and test
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
        else:
            print(f"Warning: Categorical column '{col}' not found in both train and test df after merge, skipping encoding.")


    # Handle remaining object columns possibly created during aggregation (should be rare if handled well)
    remaining_object_cols = [col for col in train_df.select_dtypes(include=['object']).columns if col not in original_object_cols]
    if remaining_object_cols:
         print(f"Warning: Found unexpected object columns after merge/aggregation: {remaining_object_cols}. Attempting LabelEncoding.")
         for col in remaining_object_cols:
             if col in train_df.columns and col in test_df.columns:
                 categorical_features.append(normalize_column_name(col))
                 train_df[col] = train_df[col].fillna('Unknown').astype(str)
                 test_df[col] = test_df[col].fillna('Unknown').astype(str)
                 le = LabelEncoder()
                 unique_values = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
                 le.fit(unique_values)
                 train_df[col] = le.transform(train_df[col])
                 test_df[col] = le.transform(test_df[col])


    return train_df, test_df, categorical_features

def one_hot_encode_categorical(df, column):
    """One-hot encode a categorical column and return aggregate functions for it"""
    dummies = pd.get_dummies(df[column], prefix=column, dummy_na=True)
    # Normalize column names
    dummies.columns = [normalize_column_name(col) for col in dummies.columns]
    df = pd.concat([df, dummies], axis=1)

    # Create aggregation dict for dummies
    agg_dict = {}
    for dummy in dummies.columns:
        agg_dict[dummy] = ['mean']

    return df, agg_dict

def aggregate_bureau_data(bureau, bureau_balance):
    """Aggregate bureau and bureau_balance data by client ID"""
    print("Aggregating bureau data...")
    start_time = time.time()

    # Aggregate bureau_balance by bureau ID
    bb_aggregations = {
        'months_balance': ['min', 'max', 'mean', 'size']
    }

    # Handle STATUS column specially with a custom function
    def status_to_numeric(x):
        mapping = {'C': 0, 'X': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        return np.mean([mapping.get(str(v), 0) for v in x])

    # Aggregate bureau_balance
    bb_agg = bureau_balance.groupby('sk_id_bureau').agg({**bb_aggregations, 'status': status_to_numeric})
    # Normalize column names
    bb_agg.columns = pd.Index([normalize_column_name(f'bb_{e[0]}_{e[1].lower()}' if e[1] != '<lambda>' else f'bb_{e[0]}_mean')
                              for e in bb_agg.columns.tolist()])

    # Merge with bureau data
    bureau = bureau.join(bb_agg, how='left', on='sk_id_bureau')

    # Prepare numerical aggregations
    num_aggregations = {
        'days_credit': ['min', 'max', 'mean'],
        'days_credit_enddate': ['min', 'max', 'mean'],
        'days_credit_update': ['mean'],
        'credit_day_overdue': ['max', 'mean'],
        'amt_credit_max_overdue': ['mean'],
        'amt_credit_sum': ['max', 'mean', 'sum'],
        'amt_credit_sum_debt': ['max', 'mean', 'sum'],
        'amt_credit_sum_overdue': ['mean'],
        'amt_credit_sum_limit': ['mean', 'sum'],
        'amt_annuity': ['max', 'mean'],
        'cnt_credit_prolong': ['sum']
    }

    # Add bureau balance aggregated features
    for col in bb_agg.columns:
        num_aggregations[col] = ['mean']

    # One-hot encode categorical columns
    cat_cols = bureau.select_dtypes(include=['object']).columns
    categorical_aggs = {}

    for col in cat_cols:
        bureau, cat_agg = one_hot_encode_categorical(bureau, col)
        categorical_aggs.update(cat_agg)

    # Combine all aggregations
    bureau_agg = {**num_aggregations, **categorical_aggs}

    # Perform aggregation
    bureau_agg_df = bureau.groupby('sk_id_curr').agg({**num_aggregations, **categorical_aggs})

    # Rename columns
    bureau_agg_df.columns = pd.Index([normalize_column_name(f'bureau_{e[0]}_{e[1].lower()}')
                                    for e in bureau_agg_df.columns.tolist()])

    # Add counts
    bureau_agg_df[normalize_column_name('bureau_count')] = bureau.groupby('sk_id_curr').size()

    # Calculate active loans ratio
    active_col = [col for col in bureau.columns if 'credit_active_active' in col.lower()]
    if active_col:
        active_bureau = bureau[active_col[0]].groupby(bureau['sk_id_curr']).sum()
        bureau_agg_df[normalize_column_name('bureau_active_ratio')] = active_bureau / bureau_agg_df[normalize_column_name('bureau_count')].replace(0, 1)

    print(f"Bureau data aggregated in {time.time() - start_time:.2f} seconds.")
    return bureau_agg_df.fillna(0)

def aggregate_prev_applications(prev):
    """Aggregate previous application data by client ID"""
    print("Aggregating previous application data...")
    start_time = time.time()

    # Replace 365243 with NaN for date calculations if needed, do this early
    prev['days_first_drawing'].replace(365243, np.nan, inplace=True)
    prev['days_first_due'].replace(365243, np.nan, inplace=True)
    prev['days_last_due_1st_version'].replace(365243, np.nan, inplace=True)
    prev['days_last_due'].replace(365243, np.nan, inplace=True)
    prev['days_termination'].replace(365243, np.nan, inplace=True)

    # Add new features based on days
    prev[normalize_column_name('days_decision_diff')] = prev['days_decision'] - prev['days_first_due'] # Example
    # Add more relevant features here based on domain knowledge or exploration

    # Numerical aggregations
    num_aggregations = {
        'amt_annuity': ['min', 'max', 'mean'],
        'amt_application': ['min', 'max', 'mean'],
        'amt_credit': ['min', 'max', 'mean'],
        'amt_down_payment': ['min', 'max', 'mean'],
        'amt_goods_price': ['min', 'max', 'mean'],
        'hour_appr_process_start': ['min', 'max', 'mean'],
        'days_decision': ['min', 'max', 'mean'],
        'cnt_payment': ['mean', 'sum'],
        normalize_column_name('days_decision_diff'): ['mean', 'max', 'min'] # Aggregate new features
        # Add more aggregations for new features
    }

    # One-hot encode categorical columns
    cat_cols = prev.select_dtypes(include=['object']).columns
    categorical_aggs = {}

    for col in cat_cols:
        prev, cat_agg = one_hot_encode_categorical(prev, col)
        categorical_aggs.update(cat_agg)

    # Combine all aggregations and perform aggregation
    prev_agg_df = prev.groupby('sk_id_curr').agg({**num_aggregations, **categorical_aggs})

    # Rename columns
    prev_agg_df.columns = pd.Index([normalize_column_name(f'prev_{e[0]}_{e[1].lower()}')
                                  for e in prev_agg_df.columns.tolist()])

    # Add counts
    prev_agg_df[normalize_column_name('prev_count')] = prev.groupby('sk_id_curr').size()

    # Calculate approval ratio using dummy variable
    approved_col = [col for col in prev.columns if 'name_contract_status_approved' in col.lower()]
    if approved_col:
        approved_count = prev.groupby('sk_id_curr')[approved_col[0]].sum()
        prev_agg_df[normalize_column_name('prev_approved_ratio')] = approved_count / prev_agg_df[normalize_column_name('prev_count')].replace(0, 1)

    print(f"Previous application data aggregated in {time.time() - start_time:.2f} seconds.")
    return prev_agg_df.fillna(0)

def aggregate_pos_cash(pos):
    """Aggregate POS_CASH_balance data by client ID"""
    print("Aggregating POS CASH balance data...")
    start_time = time.time()

    # Numerical aggregations
    num_aggregations = {
        'months_balance': ['min', 'max', 'mean', 'size'],
        'sk_dpd': ['max', 'mean', 'sum'],
        'sk_dpd_def': ['max', 'mean', 'sum']
    }

    # One-hot encode categorical columns
    cat_cols = pos.select_dtypes(include=['object']).columns
    categorical_aggs = {}

    for col in cat_cols:
        pos, cat_agg = one_hot_encode_categorical(pos, col)
        categorical_aggs.update(cat_agg)

    # Combine all aggregations and perform aggregation
    pos_agg_df = pos.groupby('sk_id_curr').agg({**num_aggregations, **categorical_aggs})

    # Rename columns
    pos_agg_df.columns = pd.Index([normalize_column_name(f'pos_{e[0]}_{e[1].lower()}')
                                 for e in pos_agg_df.columns.tolist()])

    # Add counts
    pos_agg_df[normalize_column_name('pos_count')] = pos.groupby('sk_id_curr').size()

    print(f"POS CASH balance data aggregated in {time.time() - start_time:.2f} seconds.")
    return pos_agg_df.fillna(0)

def aggregate_installments(ins):
    """Aggregate installments_payments data by client ID"""
    print("Aggregating installments payments data...")
    start_time = time.time()

    # Create additional features
    ins[normalize_column_name('payment_perc')] = ins['amt_payment'] / ins['amt_instalment'].replace(0, 1)
    ins[normalize_column_name('payment_diff')] = ins['amt_instalment'] - ins['amt_payment']
    ins[normalize_column_name('dpd')] = ins['days_entry_payment'] - ins['days_instalment']
    ins[normalize_column_name('dbd')] = ins['days_instalment'] - ins['days_entry_payment']
    ins[normalize_column_name('dpd')] = ins[normalize_column_name('dpd')].apply(lambda x: max(x, 0))
    ins[normalize_column_name('dbd')] = ins[normalize_column_name('dbd')].apply(lambda x: max(x, 0))

    # All columns in installments are numerical
    ins_agg = {
        'num_instalment_version': ['nunique'],
        normalize_column_name('dpd'): ['max', 'mean', 'sum'],
        normalize_column_name('dbd'): ['max', 'mean', 'sum'],
        normalize_column_name('payment_perc'): ['max', 'mean', 'sum', 'var'],
        normalize_column_name('payment_diff'): ['max', 'mean', 'sum', 'var'],
        'amt_instalment': ['max', 'mean', 'sum'],
        'amt_payment': ['min', 'max', 'mean', 'sum'],
        'days_entry_payment': ['max', 'mean', 'sum'],
    }

    ins_agg_df = ins.groupby('sk_id_curr').agg(ins_agg)
    ins_agg_df.columns = pd.Index([normalize_column_name(f'ins_{e[0]}_{e[1].lower()}')
                                 for e in ins_agg_df.columns.tolist()])

    # Add counts
    ins_agg_df[normalize_column_name('ins_count')] = ins.groupby('sk_id_curr').size()

    print(f"Installments payments data aggregated in {time.time() - start_time:.2f} seconds.")
    return ins_agg_df.fillna(0)

def aggregate_credit_card(cc):
    """Aggregate credit_card_balance data by client ID"""
    print("Aggregating credit card balance data...")
    start_time = time.time()

    # Numerical aggregations
    num_aggregations = {
        'months_balance': ['min', 'max', 'mean', 'size'],
        'amt_balance': ['max', 'mean', 'sum'],
        'amt_credit_limit_actual': ['max', 'mean'],
        'amt_drawings_atm_current': ['max', 'mean', 'sum'],
        'amt_drawings_current': ['max', 'mean', 'sum'],
        'amt_drawings_other_current': ['max', 'mean', 'sum'],
        'amt_drawings_pos_current': ['max', 'mean', 'sum'],
        'amt_inst_min_regularity': ['max', 'mean'],
        'amt_payment_current': ['max', 'mean', 'sum'],
        'amt_payment_total_current': ['max', 'mean', 'sum'],
        'amt_receivable_principal': ['max', 'mean', 'sum'],
        'cnt_drawings_atm_current': ['max', 'mean', 'sum'],
        'cnt_drawings_current': ['max', 'mean', 'sum'],
        'sk_dpd': ['max', 'mean'],
        'sk_dpd_def': ['max', 'mean'],
    }

    # One-hot encode categorical columns
    cat_cols = cc.select_dtypes(include=['object']).columns
    categorical_aggs = {}

    for col in cat_cols:
        cc, cat_agg = one_hot_encode_categorical(cc, col)
        categorical_aggs.update(cat_agg)

    # Combine all aggregations and perform aggregation
    cc_agg_df = cc.groupby('sk_id_curr').agg({**num_aggregations, **categorical_aggs})

    # Rename columns
    cc_agg_df.columns = pd.Index([normalize_column_name(f'cc_{e[0]}_{e[1].lower()}')
                                for e in cc_agg_df.columns.tolist()])

    # Add counts
    cc_agg_df[normalize_column_name('cc_count')] = cc.groupby('sk_id_curr').size()

    # Calculate utilization
    cc[normalize_column_name('utilization')] = cc['amt_balance'] / cc['amt_credit_limit_actual'].replace(0, 1)
    cc_agg_df[normalize_column_name('cc_utilization_mean')] = cc.groupby('sk_id_curr')[normalize_column_name('utilization')].mean()
    cc_agg_df[normalize_column_name('cc_utilization_max')] = cc.groupby('sk_id_curr')[normalize_column_name('utilization')].max()

    print(f"Credit card balance data aggregated in {time.time() - start_time:.2f} seconds.")
    return cc_agg_df.fillna(0)

def get_optimal_types(df):
    """
    Analyzes DataFrame columns and determines the most memory-efficient dtype.
    """
    print("Determining optimal data types...")
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem_usg:.2f} MB")
    optimal_types = {}

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "object" or isinstance(col_type, pd.CategoricalDtype):
             # Keep object/category as is for now, handled by LabelEncoder primarily
             # Consider converting low-cardinality objects to 'category' if LabelEncoder wasn't used
            optimal_types[col] = df[col].dtype
        elif pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min is pd.NA or c_max is pd.NA: # Skip if min/max calc failed (e.g., all NA)
                 optimal_types[col] = df[col].dtype
                 continue
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                optimal_types[col] = np.int8
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                optimal_types[col] = np.int16
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                optimal_types[col] = np.int32
            else:
                optimal_types[col] = np.int64
        elif pd.api.types.is_float_dtype(col_type):
            # Downcast floats to float32 if possible
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min is pd.NA or c_max is pd.NA: # Skip if min/max calc failed
                 optimal_types[col] = df[col].dtype
                 continue
            # Check if float can be represented as integer potentially (after fillna)
            # Note: This check might be less useful after fillna with median
            # if np.all(np.mod(df[col].dropna(), 1) == 0):
            #     # Try integer downcasting first
            #     if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
            #          optimal_types[col] = np.int8
            #     elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
            #          optimal_types[col] = np.int16
            #     elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
            #          optimal_types[col] = np.int32
            #     else:
            #          optimal_types[col] = np.int64
            # else:
                 # Use float32 if it doesn't lose precision significantly for the range
                 # np.finfo(np.float32).max is large enough for most financial data
            optimal_types[col] = np.float32 # Default to float32 for floats
        elif pd.api.types.is_datetime64_any_dtype(col_type):
             optimal_types[col] = df[col].dtype # Keep datetime types
        else:
            # Keep other types (like boolean) as they are
            optimal_types[col] = df[col].dtype

    return optimal_types

def apply_types(df, type_map):
    """
    Applies the specified data types to the DataFrame columns.
    """
    print("Applying optimal data types...")
    start_time = time.time()
    for col, dtype in type_map.items():
        if col in df.columns:
            try:
                # Handle potential overflow/conversion errors gracefully
                if pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_integer_dtype(df[col].dtype):
                     # Attempt conversion to float first if converting float->int to avoid errors with NaN/Inf
                     if pd.api.types.is_float_dtype(df[col].dtype):
                         # Use Int64 (nullable int) if target is int and source is float with NAs
                          if df[col].isnull().any():
                              if dtype == np.int8: df[col] = df[col].astype(pd.Int8Dtype())
                              elif dtype == np.int16: df[col] = df[col].astype(pd.Int16Dtype())
                              elif dtype == np.int32: df[col] = df[col].astype(pd.Int32Dtype())
                              else: df[col] = df[col].astype(pd.Int64Dtype())
                          else:
                              df[col] = df[col].astype(np.float64).astype(dtype) # Go via float64 for safety
                     else:
                         df[col] = df[col].astype(dtype) # Direct conversion otherwise
                elif pd.api.types.is_float_dtype(dtype) and df[col].dtype != dtype:
                     df[col] = df[col].astype(dtype)
                elif df[col].dtype != dtype: # Apply other types if different
                     df[col] = df[col].astype(dtype)

            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to {dtype}. Error: {e}. Keeping original type {df[col].dtype}.")
        else:
             print(f"Warning: Column '{col}' from type map not found in DataFrame.")

    end_mem_usg = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem_usg:.2f} MB")
    print(f"Type application finished in {time.time() - start_time:.2f} seconds.")
    return df

def handle_infinite_values(df, default_value=0):
    """
    Заменяет бесконечные значения (inf) на NaN, а затем на указанное значение по умолчанию.
    """
    # Заменяем положительную и отрицательную бесконечность на NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Заполняем NaN значением по умолчанию
    df = df.fillna(default_value)

    return df

def main():
    """Main function to process and prepare the dataset"""
    print("Starting Home Credit Default Risk dataset preparation...")
    main_start_time = time.time()

    # Read main tables
    train_df = read_data(os.path.join(DATA_DIR, 'application_train.csv'))
    test_df = read_data(os.path.join(DATA_DIR, 'application_test.csv'))

    # Read supplementary tables
    bureau = read_data(os.path.join(DATA_DIR, 'bureau.csv'))
    bureau_balance = read_data(os.path.join(DATA_DIR, 'bureau_balance.csv'))
    prev_app = read_data(os.path.join(DATA_DIR, 'previous_application.csv'))
    pos_cash = read_data(os.path.join(DATA_DIR, 'POS_CASH_balance.csv'))
    installments = read_data(os.path.join(DATA_DIR, 'installments_payments.csv'))
    credit_card = read_data(os.path.join(DATA_DIR, 'credit_card_balance.csv'))

    # Aggregate supplementary tables
    bureau_agg = aggregate_bureau_data(bureau, bureau_balance)
    prev_agg = aggregate_prev_applications(prev_app)
    pos_cash_agg = aggregate_pos_cash(pos_cash)
    installments_agg = aggregate_installments(installments)
    credit_card_agg = aggregate_credit_card(credit_card)

    # Обработка бесконечных значений
    bureau_agg = handle_infinite_values(bureau_agg)
    prev_agg = handle_infinite_values(prev_agg)
    pos_cash_agg = handle_infinite_values(pos_cash_agg)
    installments_agg = handle_infinite_values(installments_agg)
    credit_card_agg = handle_infinite_values(credit_card_agg)

    # Free up memory
    del bureau, bureau_balance, prev_app, pos_cash, installments, credit_card
    gc.collect()

    # Merge aggregated data with main tables
    print("Merging aggregated data with main tables...")

    # For train data
    train_df = train_df.merge(bureau_agg, on='sk_id_curr', how='left')
    train_df = train_df.merge(prev_agg, on='sk_id_curr', how='left')
    train_df = train_df.merge(pos_cash_agg, on='sk_id_curr', how='left')
    train_df = train_df.merge(installments_agg, on='sk_id_curr', how='left')
    train_df = train_df.merge(credit_card_agg, on='sk_id_curr', how='left')

    # For test data
    test_df = test_df.merge(bureau_agg, on='sk_id_curr', how='left')
    test_df = test_df.merge(prev_agg, on='sk_id_curr', how='left')
    test_df = test_df.merge(pos_cash_agg, on='sk_id_curr', how='left')
    test_df = test_df.merge(installments_agg, on='sk_id_curr', how='left')
    test_df = test_df.merge(credit_card_agg, on='sk_id_curr', how='left')

    # Free up memory
    del bureau_agg, prev_agg, pos_cash_agg, installments_agg, credit_card_agg
    gc.collect()

    # Handle missing values BEFORE optimizing types
    print("\nHandling missing values...")
    start_time = time.time()
    train_df = handle_missing_values(train_df)
    # Ensure test_df uses train_df's medians correctly
    test_df_cols = test_df.columns # Store original columns before potential adds
    train_for_test_impute = train_df[[col for col in train_df.select_dtypes(include=['number']).columns if col in test_df_cols]]
    test_df = handle_missing_values(test_df, is_train=False, train_df=train_for_test_impute)
    del train_for_test_impute # free memory
    gc.collect()
    print(f"Missing values handled in {time.time() - start_time:.2f} seconds.")


    # Encode categorical features BEFORE optimizing types
    # LabelEncoder produces int types, which will be optimized later
    print("\nEncoding categorical features...")
    start_time = time.time()
    train_df, test_df, categorical_features = encode_categorical_features(train_df, test_df)
    gc.collect()
    print(f"Categorical features encoded in {time.time() - start_time:.2f} seconds.")

    # Optimize memory usage by downcasting types AFTER imputation and encoding
    print("\nOptimizing memory usage...")
    optimal_types = get_optimal_types(train_df)
    train_df = apply_types(train_df, optimal_types)
    # Apply the SAME types derived from train_df to test_df for consistency
    test_df = apply_types(test_df, optimal_types)
    gc.collect()

    # Обработка бесконечных значений в итоговых датасетах
    train_df = handle_infinite_values(train_df)
    test_df = handle_infinite_values(test_df)


    # Get the list of features (excluding TARGET and ID)
    # Ensure features list is derived from the final train_df columns and all names are normalized
    features = [normalize_column_name(col) for col in train_df.columns if col not in ['target', 'sk_id_curr']]

    # Align columns: Ensure test_df has the same columns as train_df (except 'target') in the same order
    test_features = [normalize_column_name(col) for col in test_df.columns if col not in ['sk_id_curr']]
    missing_in_test = set(features) - set(test_features)
    missing_in_train = set(test_features) - set(features)

    if missing_in_test:
        print(f"\nWarning: Columns missing in test_df found: {missing_in_test}. Adding them with 0 fill.")
        for c in missing_in_test:
            # Use the dtype from the optimal_types map if available, else float32
            dtype_to_use = optimal_types.get(c, np.float32)
            test_df[c] = pd.Series([0] * len(test_df), index=test_df.index).astype(dtype_to_use)

    if missing_in_train:
         print(f"\nWarning: Columns missing in train_df (but in test) found: {missing_in_train}. This is unusual.")
         # Decide how to handle this, maybe drop from test? For now, just report.
         # test_df = test_df.drop(columns=list(missing_in_train))


    # Reorder test_df columns to match train_df feature order
    test_df = test_df[['sk_id_curr'] + features] # Ensure sk_id_curr is first, then features in order


    # Save prepared data
    print("\nSaving prepared data...")
    start_time = time.time()
    output_train_path = os.path.join(DST_DIR, 'train.parquet')
    output_test_path = os.path.join(DST_DIR, 'test.parquet')
    output_features_path = os.path.join(DST_DIR, 'features.txt')
    output_cat_features_path = os.path.join(DST_DIR, 'categorical_features.txt')

    # Ensure destination directory exists
    os.makedirs(DST_DIR, exist_ok=True)

    train_df.to_parquet(output_train_path)
    test_df.to_parquet(output_test_path)

    # Save feature lists
    with open(output_features_path, 'w') as f:
        f.write('\n'.join(features))

    with open(output_cat_features_path, 'w') as f:
        f.write('\n'.join(categorical_features))

    print(f"Saved train dataset to: {output_train_path}")
    print(f"Saved test dataset to: {output_test_path}")
    print(f"Saved features list to: {output_features_path}")
    print(f"Saved categorical features list to: {output_cat_features_path}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Total features: {len(features)}, Categorical features: {len(categorical_features)}")

    # Print target and auxiliary columns
    print(f"\nTarget column: 'target'")
    print(f"Auxiliary columns: 'sk_id_curr' (ID column)")

    print(f"\nTotal script execution time: {time.time() - main_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()