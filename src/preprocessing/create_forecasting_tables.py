import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extend_period_begin_dates(df, end_date):
    """
    Extend the period_begin column to the last day of each month up to a specified end date.
    """
    start_date = df['period_begin'].max()
    if start_date.month == 12:
        start_date = datetime(start_date.year + 1, 1, 1)
    else:
        start_date = datetime(start_date.year, start_date.month + 1, 1)

    extended_dates = pd.date_range(start=start_date, end=end_date, freq='M').to_pydatetime().tolist()
    extended_df = pd.DataFrame(extended_dates, columns=['period_begin'])
    return pd.concat([df, extended_df], ignore_index=True)


def create_forecast_files(directory):
    """
    Create to_forecast.csv files for each property type in each city and state.
    """
    end_date = datetime(2025, 12, 31)
    commercial_types = ['Mall', 'Industrial', 'Office']

    for state in os.listdir(directory):
        state_path = os.path.join(directory, state)
        if os.path.isdir(state_path):
            for city in os.listdir(state_path):
                city_path = os.path.join(state_path, city)
                if os.path.isdir(city_path):
                    for property_type in os.listdir(city_path):
                        property_path = os.path.join(city_path, property_type)
                        if os.path.isdir(property_path):
                            data_file = os.path.join(property_path, 'selected_data.csv')
                            if os.path.isfile(data_file):
                                df = pd.read_csv(data_file, parse_dates=['period_begin'])

                                # Check if commercial or residential and select appropriate columns
                                if property_type in commercial_types:
                                    columns = ['period_begin', 'asset_value_momentum', 'occupancy', 'rank_desirability', 'remote_work_share']
                                else:
                                    columns = ['period_begin', 'asset_value_momentum', 'remote_work_share']

                                # Filter columns and extend period_begin dates
                                df = df[columns]
                                df = extend_period_begin_dates(df, end_date)

                                # Save to to_forecast.csv
                                forecast_file = os.path.join(property_path, 'to_forecast.csv')
                                df.to_csv(forecast_file, index=False)
                                print(f"Created forecast file: {forecast_file}")

# Define the directory and call the function
directory = 'filtered_housing'  # Replace with the actual path
create_forecast_files(directory)
