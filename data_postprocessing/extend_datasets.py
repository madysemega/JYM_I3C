import os
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import MonthEnd

# Constants
CUTOFF_DATE_1 = pd.Timestamp('2023-09-30')  # Start date for commercial predictions
CUTOFF_DATE_2 = pd.Timestamp('2023-10-31')  # Start date for residential predictions
Commercial = ['Mall', 'Industrial', 'Office']
commercial_targets = ['occupancy', 'rank_desirability', 'asset_value_momentum', 'remote_work_share']
housing_targets = ['asset_value_momentum', 'remote_work_share']

# Directory containing the housing data
Housing_Directory = 'housing'

# Function to update dataset with predictions
def update_dataset(property_type_path, cutoff, targets):
    dataset_path = os.path.join(property_type_path, 'dataset.csv')
    df = pd.read_csv(dataset_path, index_col='period_begin', parse_dates=True)

    for target_name in targets:
        prediction_path = os.path.join(property_type_path, target_name + '.csv')
        predictions = pd.read_csv(prediction_path, header=None, squeeze=True).drop(0)

        # Create a date range for predictions starting from cutoff
        prediction_dates = pd.date_range(start=cutoff + pd.DateOffset(1), periods=len(predictions), freq='M') - MonthEnd(1)

        # Ensure predictions only till December 2025
        prediction_dates = prediction_dates[prediction_dates <= pd.Timestamp('2025-12-31')]
        predictions = predictions[:len(prediction_dates)]

        # Align predictions with their dates
        predictions.index = prediction_dates
        df.loc[prediction_dates, target_name] = predictions

    forecasted_dataset_path = os.path.join(property_type_path, 'forecasted_dataset.csv')
    df.to_csv(forecasted_dataset_path, index=True)

# Iterate through the housing directory
for state in tqdm(os.listdir(Housing_Directory), desc='States'):
    state_path = os.path.join(Housing_Directory, state)
    if os.path.isdir(state_path):
        for city in os.listdir(state_path):
            city_path = os.path.join(state_path, city)
            if os.path.isdir(city_path):
                for property_type in os.listdir(city_path):
                    property_type_path = os.path.join(city_path, property_type)
                    cutoff = CUTOFF_DATE_1 if property_type in Commercial else CUTOFF_DATE_2
                    current_targets = commercial_targets if property_type in Commercial else housing_targets
                    update_dataset(property_type_path, cutoff, current_targets)