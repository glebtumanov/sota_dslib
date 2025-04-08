#!/usr/bin/env python3
# ссылка на скачивание датасета: https://www.kaggle.com/competitions/home-credit-default-risk/data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import gc
import warnings
warnings.filterwarnings('ignore')


# Define data directories
DATA_DIR = './home-credit-default-risk'
DST_DIR = './home-credit-default-risk'

def read_data(file_path):
    """Read a CSV file and display its shape"""
    df = pd.read_csv(file_path)
    # Convert column names to lowercase
    df.columns = [col.lower() for col in df.columns]
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

    for col in train_df.select_dtypes(include=['object']).columns:
        categorical_features.append(col)

        # Fill missing values
        train_df[col] = train_df[col].fillna('Unknown')
        test_df[col] = test_df[col].fillna('Unknown')

        # Fit encoder on train data only
        le = LabelEncoder()
        le.fit(train_df[col])

        # Transform both train and test
        train_df[col] = le.transform(train_df[col])

        # Handle unseen categories in test set
        test_df[col] = test_df[col].map(lambda x: x if x in le.classes_ else 'Unknown')
        test_df[col] = le.transform(test_df[col])

    return train_df, test_df, categorical_features

def one_hot_encode_categorical(df, column):
    """One-hot encode a categorical column and return aggregate functions for it"""
    dummies = pd.get_dummies(df[column], prefix=column, dummy_na=True)
    # Convert column names to lowercase
    dummies.columns = [col.lower() for col in dummies.columns]
    df = pd.concat([df, dummies], axis=1)

    # Create aggregation dict for dummies
    agg_dict = {}
    for dummy in dummies.columns:
        agg_dict[dummy] = ['mean']

    return df, agg_dict

def aggregate_bureau_data(bureau, bureau_balance):
    """Aggregate bureau and bureau_balance data by client ID"""
    print("Aggregating bureau data...")

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
    bb_agg.columns = pd.Index([f'bb_{e[0]}_{e[1].lower()}' if e[1] != '<lambda>' else f'bb_{e[0]}_mean'
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
    bureau_agg_df.columns = pd.Index([f'bureau_{e[0]}_{e[1].lower()}' for e in bureau_agg_df.columns.tolist()])

    # Add counts
    bureau_agg_df['bureau_count'] = bureau.groupby('sk_id_curr').size()

    # Calculate active loans ratio
    active_col = [col for col in bureau.columns if 'credit_active_active' in col.lower()]
    if active_col:
        active_bureau = bureau[active_col[0]].groupby(bureau['sk_id_curr']).sum()
        bureau_agg_df['bureau_active_ratio'] = active_bureau / bureau_agg_df['bureau_count']

    return bureau_agg_df.fillna(0)

def aggregate_prev_applications(prev):
    """Aggregate previous application data by client ID"""
    print("Aggregating previous application data...")

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
    prev_agg_df.columns = pd.Index([f'prev_{e[0]}_{e[1].lower()}' for e in prev_agg_df.columns.tolist()])

    # Add counts
    prev_agg_df['prev_count'] = prev.groupby('sk_id_curr').size()

    # Calculate approval ratio using dummy variable
    approved_col = [col for col in prev.columns if 'name_contract_status_approved' in col.lower()]
    if approved_col:
        approved_count = prev.groupby('sk_id_curr')[approved_col[0]].sum()
        prev_agg_df['prev_approved_ratio'] = approved_count / prev_agg_df['prev_count']

    return prev_agg_df.fillna(0)

def aggregate_pos_cash(pos):
    """Aggregate POS_CASH_balance data by client ID"""
    print("Aggregating POS CASH balance data...")

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
    pos_agg_df.columns = pd.Index([f'pos_{e[0]}_{e[1].lower()}' for e in pos_agg_df.columns.tolist()])

    # Add counts
    pos_agg_df['pos_count'] = pos.groupby('sk_id_curr').size()

    return pos_agg_df.fillna(0)

def aggregate_installments(ins):
    """Aggregate installments_payments data by client ID"""
    print("Aggregating installments payments data...")

    # Create additional features
    ins['payment_perc'] = ins['amt_payment'] / ins['amt_instalment']
    ins['payment_diff'] = ins['amt_instalment'] - ins['amt_payment']
    ins['dpd'] = ins['days_entry_payment'] - ins['days_instalment']
    ins['dbd'] = ins['days_instalment'] - ins['days_entry_payment']
    ins['dpd'] = ins['dpd'].apply(lambda x: max(x, 0))
    ins['dbd'] = ins['dbd'].apply(lambda x: max(x, 0))

    # All columns in installments are numerical
    ins_agg = {
        'num_instalment_version': ['nunique'],
        'dpd': ['max', 'mean', 'sum'],
        'dbd': ['max', 'mean', 'sum'],
        'payment_perc': ['max', 'mean', 'sum', 'var'],
        'payment_diff': ['max', 'mean', 'sum', 'var'],
        'amt_instalment': ['max', 'mean', 'sum'],
        'amt_payment': ['min', 'max', 'mean', 'sum'],
        'days_entry_payment': ['max', 'mean', 'sum'],
    }

    ins_agg_df = ins.groupby('sk_id_curr').agg(ins_agg)
    ins_agg_df.columns = pd.Index([f'ins_{e[0]}_{e[1].lower()}' for e in ins_agg_df.columns.tolist()])

    # Add counts
    ins_agg_df['ins_count'] = ins.groupby('sk_id_curr').size()

    return ins_agg_df.fillna(0)

def aggregate_credit_card(cc):
    """Aggregate credit_card_balance data by client ID"""
    print("Aggregating credit card balance data...")

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
    cc_agg_df.columns = pd.Index([f'cc_{e[0]}_{e[1].lower()}' for e in cc_agg_df.columns.tolist()])

    # Add counts
    cc_agg_df['cc_count'] = cc.groupby('sk_id_curr').size()

    # Calculate utilization
    cc['utilization'] = cc['amt_balance'] / cc['amt_credit_limit_actual']
    cc_agg_df['cc_utilization_mean'] = cc.groupby('sk_id_curr')['utilization'].mean()
    cc_agg_df['cc_utilization_max'] = cc.groupby('sk_id_curr')['utilization'].max()

    return cc_agg_df.fillna(0)

def main():
    """Main function to process and prepare the dataset"""
    print("Starting Home Credit Default Risk dataset preparation...")

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

    # Handle missing values
    print("Handling missing values...")
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df, is_train=False, train_df=train_df)

    # Encode categorical features
    print("Encoding categorical features...")
    train_df, test_df, categorical_features = encode_categorical_features(train_df, test_df)

    # Get the list of features (excluding TARGET and ID)
    features = [col for col in train_df.columns if col not in ['target', 'sk_id_curr']]

    # Save prepared data
    print("Saving prepared data...")
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

    print(f"Data preparation completed successfully!")
    print(f"Saved train dataset to: {output_train_path}")
    print(f"Saved test dataset to: {output_test_path}")
    print(f"Saved features list to: {output_features_path}")
    print(f"Saved categorical features list to: {output_cat_features_path}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Total features: {len(features)}, Categorical features: {len(categorical_features)}")

    # Print target and auxiliary columns
    print(f"\nTarget column: 'target'")
    print(f"Auxiliary columns: 'sk_id_curr' (ID column)")

if __name__ == "__main__":
    main()