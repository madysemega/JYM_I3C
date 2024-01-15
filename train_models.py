import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from models import RNNModel, Optimization
import joblib
import os
import pandas as pd
from generate_features import generate_time_lags, assign_datetime, onehot_encode_pd, generate_cyclical_features, add_holiday_col
from model_inputs_preparation import get_scaled_values, get_loaders, get_scaler
import random
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
CUTOFF_DATE_1 = 56 # Cutoff date for table_1
CUTOFF_DATE_2 = 57  # Cutoff date for table_2
Commercial = ['Mall', 'Industrial', 'Office']
commercial_targets = ['occupancy', 'rank_desirability', 'asset_value_momentum', 'remote_work_share']
housing_targets = ['asset_value_momentum', 'remote_work_share']


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

def property_type_loop(dataset_path, cutoff, property_type, property_type_path, gpu_id):
    processes = []
    if os.path.isfile(dataset_path):
        df = pd.read_csv(dataset_path, index_col='period_begin')
        df.index = pd.to_datetime(df.index)
        column_names = commercial_targets if property_type in Commercial else housing_targets
        loaders, scalers = prepare_data_for_prediction(df, cutoff, column_names)
        for i, column_name in enumerate(column_names):
            p = mp.Process(target=start_training_process, args=(loaders[column_name], scalers[column_name], gpu_id, column_name, cutoff, property_type_path))
            p.start()
            processes.append(p)

def city_loop(city_path, gpu_id):
    processes = []
    if os.path.isdir(city_path):
        for i, property_type in enumerate(os.listdir(city_path)):
            property_type_path = os.path.join(city_path, property_type)
            dataset_path = os.path.join(property_type_path, 'dataset.csv')
            cutoff = CUTOFF_DATE_1 if property_type in Commercial else CUTOFF_DATE_2
            p = mp.Process(target=property_type_loop, args=(dataset_path, cutoff, property_type, property_type_path, gpu_id))
            p.start()
            processes.append(p)

           
if __name__ == "__main__":
    for state in os.listdir(Housing_Directory):
        state_path = os.path.join(Housing_Directory, state)
        processes = []
        if os.path.isdir(state_path):
            for i, city in enumerate(os.listdir(state_path)):
                city_path = os.path.join(state_path, city)
                gpu_id = random.randint(0, NUM_GPUS - 1)  # Distribute processes across GPUs
                p = mp.Process(target=city_loop, args=(city_path, gpu_id))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                
                        
                

                        