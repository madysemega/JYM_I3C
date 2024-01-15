import torch
import torch.multiprocessing as mp
import pandas as pd
import os
import json
import random
from torch.utils.data import DataLoader
from models import RNNModel, Optimization
from model_inputs_preparation import get_scaled_values, get_loaders, get_scaler
from generate_features import generate_time_lags, assign_datetime, onehot_encode_pd, generate_cyclical_features, add_holiday_col
from tqdm import tqdm 
import warnings
from multiprocessing import Manager, Lock

# Set start method for multiprocessing
mp.set_start_method('spawn', force=True)

# Constants
N_EPOCHS = 400
HIDDEN_DIM = 64
LAYER_DIM = 4
OUTPUT_DIM = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_GPUS = torch.cuda.device_count()  # Get the number of available GPUs
Housing_Directory = 'housing'
CUTOFF_DATE_1 = 56
CUTOFF_DATE_2 = 57

# Function to load DataLoader
def load_data_loader(loader_path):
    with open(loader_path, 'rb') as file:
        return joblib.load(file)

# Function to save a model and its optimizer state
def save_model(model, optimizer, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)


# Function to train the model
def train_model(loader, input_dim, output_dim, epochs, gpu_id):
    # Set the device to the specific GPU
    device = torch.device(f'cuda:{gpu_id}')

    # Initialize the model and move it to the specified GPU
    model = RNNModel(input_dim, hidden_dim=100, layer_dim=4, output_dim=output_dim, dropout_prob=0).to(device)

    # Initialize loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)

    # Optimization object
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # Train the model
    predictions, values = opt.train(loader, batch_size=1, n_epochs=epochs, n_features=input_dim, device=device)

    return model 


# Function to prepare data for prediction
def prepare_data_for_prediction(df, cutoff, column_names):
    loaders = {}
    scalers = {}
    for target_column in column_names:
        df_current = df.copy()
        df_current = df_current[[target_column]]
        
    
        # Apply preprocessing steps
        df_current = assign_datetime(df_current)
        df_current = onehot_encode_pd(df_current, ['month', 'day_of_week', 'week_of_year'])
        df_current['month'] = df_current.index.month
        df_current = generate_cyclical_features(df_current, 'month', 12, 1)
        df_current = add_holiday_col(df_current)
        
        df_train = df_current.head(cutoff)
        df_test = df_current.iloc[cutoff:len(df)]
    
        # Handle missing values
        df_train.fillna(df_train.mean(), inplace=True)
    
        # Separate features (X) and target (y)
        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[[target_column]]
    
        # Separate features (X) and target (y)
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[[target_column]]
    
        # Scale the features
        X_train_arr, X_test_arr, y_train_arr, y_test_arr, scaler = get_scaled_values(X_train, X_test, y_train, y_test)
    
        # Create DataLoader
        train_loader, test_loader = get_loaders(X_train_arr, X_test_arr, y_train_arr, y_test_arr, batch_size=1)
    
        loaders[target_column] = train_loader, test_loader
        scalers[target_column] = scaler
    return loaders, scalers

# Function to predict using a model
def predict_with_model(model, loader, device):
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.view([1, -1, len(x[0])]).to(device)
            model.eval()
            yhat = model(x).cpu().data.numpy()
            predictions.append(yhat.item())
    return predictions


# Function to predict and fill data
def predict_and_fill_data(model, data_loader, scaler, column, cutoff, property_type_path, gpu_id):
    # Set the device to the specific GPU
    device = torch.device(f'cuda:{gpu_id}')
    predictions = predict_with_model(model, data_loader, device)

    # Inverse transform predictions
    predictions = scaler.inverse_transform([[p] for p in predictions])

    # Create a new DataFrame for the predictions
    predictions_df = pd.DataFrame(predictions, columns=[f'predictions_{column}'])

    # Save the predictions to a CSV file
    csv_path = f'{property_type_path}/{column}.csv'
    predictions_df.to_csv(csv_path, index=False)

    
def start_training_process(loaders, scaler, gpu_id, column, cutoff, property_type_path):
    model = train_model(loaders[0], input_dim=len(loaders[0].dataset.tensors[0][0]), output_dim=OUTPUT_DIM, 
                        epochs=N_EPOCHS, gpu_id=gpu_id)
    predict_and_fill_data(model, loaders[1], scaler, column, cutoff, property_type_path, gpu_id)

def property_type_loop(dataset_path, cutoff, property_type_path, shared_gpus, lock, to_predict_columns):
    processes = []
    if os.path.isfile(dataset_path):
        df = pd.read_csv(dataset_path, index_col='period_begin')
        df.index = pd.to_datetime(df.index)
        loaders, scalers = prepare_data_for_prediction(df, cutoff, to_predict_columns)
        for column_name in to_predict_columns:
            with lock:  # Ensure only one process accesses the list at a time
                gpu_id = shared_gpus[0]  # Get the next available GPU ID
                shared_gpus.pop(0)  # Remove this GPU ID from the list
                shared_gpus.append(gpu_id)
            p = mp.Process(target=start_training_process, args=(loaders[column_name], scalers[column_name], gpu_id, column_name, cutoff, property_type_path))
            p.start()
            processes.append(p)

def main():

    # Load to_predict.json
    with open('to_predict.json', 'r') as file:
        to_predict = json.load(file)

    with Manager() as manager:
        # Create a shared list and a lock
        shared_gpus = manager.list(range(NUM_GPUS))
        lock = Lock()

        # Process items in batches of batch_size
        batch_size = 64
        to_predict_items = list(to_predict.items())
        for batch_start in tqdm(range(0, len(to_predict_items), batch_size), desc='Batch Progress'):
            batch_end = min(batch_start + batch_size, len(to_predict_items))
            current_batch = to_predict_items[batch_start:batch_end]
            processes = []
    
            for property_type_path, (columns, type) in current_batch:
                dataset_path = os.path.join(property_type_path, 'dataset.csv')
                # Remove .csv extension from column names
                column_names = [col.replace('.csv', '') for col in columns]
                cutoff = CUTOFF_DATE_1 if type == 'Commercial' else CUTOFF_DATE_2
                # Assign GPU in a round-robin fashion
                p = mp.Process(target=property_type_loop, args=(dataset_path, cutoff, property_type_path, shared_gpus, lock, column_names))
                p.start()
                processes.append(p)
            
            # Wait for the processes in the current batch to complete
            for p in processes:
                p.join()

if __name__ == "__main__":
    main()
