import pandas as pd
import os
import joblib
from generate_features import generate_time_lags, assign_datetime, onehot_encode_pd, generate_cyclical_features, add_holiday_col
from model_inputs_preparation import get_scaled_values, get_loaders

# Constants
N_LAGS = 5  # Number of time lags
CSV_PATH_1 = 'table_1.csv'  # Path to table_1.csv
CSV_PATH_2 = 'table_2.csv'  # Path to table_2.csv
CUTOFF_DATE_1 = 56 # Cutoff date for table_1
CUTOFF_DATE_2 = 57  # Cutoff date for table_2

# Directory setup for saving
forecasting_dir = 'forecasting'
loaders_dir = os.path.join(forecasting_dir, 'loaders')
scalers_dir = os.path.join(forecasting_dir, 'scalers')

os.makedirs(loaders_dir, exist_ok=True)
os.makedirs(scalers_dir, exist_ok=True)

def save_object(obj, filepath):
    with open(filepath, 'wb') as file:
        joblib.dump(obj, file)


def preprocess_data(csv_path, cutoff_date, target_column):
    # Load the dataset
    df_train = pd.read_csv(csv_path, index_col='period_begin').head(cutoff_date)
    df_train.index = pd.to_datetime(df_train.index)

    # Isolate the target column for prediction
    df_train = df_train[[target_column]]

    # Apply preprocessing steps to training data
    # df_train = generate_time_lags(df_train, N_LAGS)
    df_train = assign_datetime(df_train)
    df_train = onehot_encode_pd(df_train, ['month', 'day_of_week', 'week_of_year'])
    df_train['month'] = df_train.index.month
    df_train = generate_cyclical_features(df_train, 'month', 12, 1)
    df_train = add_holiday_col(df_train)

    # Handle missing values in training data
    df_train.fillna(method='ffill', inplace=True)

    # Separate features (X) and target (y)
    X = df_train.drop(columns=[target_column])
    y = df_train[[target_column]]

    # Scale the features and target
    X_train_arr, X_test_arr, y_train_arr, y_test_arr, scaler = get_scaled_values(X, X, y, y)

    # Prepare DataLoader
    train_loader = get_loaders(X_train_arr, X_test_arr, y_train_arr, y_test_arr, batch_size=1)

    return train_loader, scaler


def preprocess_and_save_all_columns(csv_path, cutoff_date, table_name):
    df = pd.read_csv(csv_path, index_col='period_begin')
    for column in df.columns:
        if column != 'period_begin':
            train_loader, scaler = preprocess_data(csv_path, cutoff_date, column)

            # Save only the train DataLoader
            train_loader_path = os.path.join(loaders_dir, f'{table_name}_{column}_train_loader.joblib')
            scaler_path = os.path.join(scalers_dir, f'{table_name}_{column}_scaler.joblib')
    
            save_object(train_loader[0], train_loader_path)  # Save only the train DataLoader
            save_object(scaler, scaler_path)


# Preprocess and save data for all columns in table_1.csv and table_2.csv
preprocess_and_save_all_columns(CSV_PATH_1, CUTOFF_DATE_1, 'table_1')
preprocess_and_save_all_columns(CSV_PATH_2, CUTOFF_DATE_2, 'table_2')
