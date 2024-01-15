import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import joblib
from generate_features import generate_time_lags, assign_datetime, onehot_encode_pd, generate_cyclical_features, add_holiday_col
from model_inputs_preparation import get_scaler

# Constants
Housing_Directory = 'housing_subset'  # Path to the housing directory
Models_Directory = 'forecasting/models'  # Path to the saved models directory
N_LAGS = 5  # Number of time lags used in preprocessing
Table_1_Types = ['Mall', 'Office', 'Industrial']
CUTOFF_DATE_1 = 56 # Cutoff date for table_1
CUTOFF_DATE_2 = 57  # Cutoff date for table_2
device="cuda" if torch.cuda.is_available() else "cpu"

# Function to load a model
def load_model(model_path):
    model_checkpoint = torch.load(model_path)
    model = model_checkpoint['model']
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

# Function to prepare data for prediction
def prepare_data_for_prediction(df, target_column):
    df = df[[target_column]].copy()
    

    # Apply preprocessing steps
    df = assign_datetime(df)
    df = onehot_encode_pd(df, ['month', 'day_of_week', 'week_of_year'])
    df['month'] = df.index.month
    df = generate_cyclical_features(df, 'month', 12, 1)
    df = add_holiday_col(df)
    
    df_train = df.head(cutoff)
    df_test = df.iloc[cutoff:len(df)]

    # Handle missing values
    df_train.fillna(method='ffill', inplace=True)

    # Separate features (X) and target (y)
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[[target_column]]

    # Separate features (X) and target (y)
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[[target_column]]

    # Scale the features
    X_train_arr, X_test_arr, y_train_arr, y_test_arr, scaler = get_scaled_values(X_train, X_test, y_train, y_test)

    # Create DataLoader
    _, test_loader = get_loaders(X_train_arr, X_test_arr, y_train_arr, y_test_arr, batch_size=1)

    return test_loader, scaler

# Function to predict using a model
def predict_with_model(model, loader):
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.view([1, -1, len(x[0])]).to(device)
            self.model.eval()
            yhat = model(x).cpu().data.numpy()
            predictions.append(yhat.item())
    return predictions
    
# Function to predict and fill data
def predict_and_fill_data(csv_path, table_name, cutoff, column):
    df = pd.read_csv(csv_path, index_col='period_begin')
    df.index = pd.to_datetime(df.index)

    model_path = os.path.join(Models_Directory, f'{table_name}_{column}_model.pth')
    model = load_model(model_path).to(device)

    data_loader, scaler = prepare_data_for_prediction(df, column)
    predictions = predict_with_model(model, data_loader)

    # Inverse transform predictions
    predictions = scaler.inverse_transform([[p] for p in predictions])

    # Save the updated dataframe
    df.to_csv(csv_path)

# Iterate over the housing folder
for state in os.listdir(Housing_Directory):
    state_path = os.path.join(Housing_Directory, state)
    if os.path.isdir(state_path):
        for city in os.listdir(state_path):
            city_path = os.path.join(state_path, city)
            if os.path.isdir(city_path):
                for property_type in os.listdir(city_path):
                    property_type_path = os.path.join(city_path, property_type)
                    if os.path.isdir(property_type_path):
                        dataset_path = os.path.join(property_type_path, 'dataset.csv')
                        if os.path.isfile(dataset_path):
                            # Determine if it's Table 1 or Table 2 type
                            table_type, cutoff = 'table_1', CUTOFF_DATE_1 if property_type in Table_1_Types else 'table_2', CUTOFF_DATE_2
                            df = pd.read_csv(dataset_path)
                            for column in df.columns:
                                if column != 'period_begin':
                                    predict_and_fill_data(dataset_path, table_type, cutoff, column)


