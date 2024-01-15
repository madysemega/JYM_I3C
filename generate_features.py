import pandas as pd
import numpy as np
from datetime import date
import holidays

US_HOLIDAYS = holidays.US()

def generate_time_lags(df, n_lags):
    df_lagged = [df]  # Start with a list containing the original dataframe

    for n in range(1, n_lags + 1):
        # Create lagged data for each lag and add it to the list
        lagged_df = df.shift(n)
        lagged_df = lagged_df.rename(columns=lambda x: f'{x}_lag{n}')
        df_lagged.append(lagged_df)

    # Concatenate all dataframes in the list along the columns
    df_n = pd.concat(df_lagged, axis=1)

    # Drop the first n_lags rows which contain NaN values
    df_n = df_n.iloc[n_lags:]
    
    return df_n

def assign_datetime(df):
    return (
    df
    .assign(month=lambda x: x.index.month)
    .assign(year=lambda x: x.index.year)
    .assign(day_of_week=lambda x: x.index.dayofweek)
    .assign(week_of_year=lambda x: x.index.isocalendar().week)
    )
    
def onehot_encode_pd(df, cols):
    dummies_list = []
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        dummies_list.append(dummies)
    
    all_dummies = pd.concat(dummies_list, axis=1)
    return pd.concat([df, all_dummies], axis=1).drop(columns=cols)

def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs).drop(columns=[col_name])



def is_holiday(date):
    date = date.replace(hour = 0)
    return 1 if (date in US_HOLIDAYS) else 0

def add_holiday_col(df):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))