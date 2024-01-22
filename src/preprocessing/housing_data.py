import os
import pandas as pd
from tqdm import tqdm 

class HousingData:
    def __init__(self, directory):
        self.directory = directory
        # Automatically determine the columns for residential and commercial property types
        self.commercial_columns = self._get_columns_for_property_type(is_commercial=True)
        self.residential_columns = self._get_columns_for_property_type(is_commercial=False)

    def _get_columns_for_property_type(self, is_commercial):
        file_path = None  # Initialize file_path to None
        for state in self._get_subdirectories(self.directory):
            for city in self._get_subdirectories(os.path.join(self.directory, state)):
                for property_type in self._get_subdirectories(os.path.join(self.directory, state, city)):
                    if is_commercial and property_type in ['Mall', 'Industrial', 'Office']:
                        file_path = os.path.join(self.directory, state, city, property_type, 'forecasted_dataset.csv')
                    elif not is_commercial and property_type not in ['Mall', 'Industrial', 'Office', 'All Residential']:
                        file_path = os.path.join(self.directory, state, city, property_type, 'forecasted_dataset.csv')
                    if file_path and os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        return df.columns.tolist()
        return []
    
    def calculate_asset_value_momentum(self, df):
        """ Calculate the asset value momentum for each property type in each city """
        def calculate_momentum(group):
            group = group.sort_values('period_begin')
            group['asset_value_momentum'] = group['median_ppsf'].pct_change()
            group['asset_value_momentum'].iloc[0] = 0
            return group

        return df.groupby(['City', 'Property_Type']).apply(calculate_momentum).reset_index(drop=True)

    def filter_data_based_on_momentum(self, df, mean_zero_momentum_by_property):
        """ Filter out property types in each city based on zero momentum percentage """
        def calculate_zero_momentum_percentage(group):
            zero_count = (group['asset_value_momentum'] == 0).sum()
            total_count = len(group)
            return (zero_count / total_count) * 100

        # Calculate zero momentum percentage for each city and property type
        zero_momentum_percentage = df.groupby(['City', 'Property_Type']).apply(calculate_zero_momentum_percentage).reset_index(name='Zero Momentum Percentage')

        # Filter out rows based on the condition
        filtered_rows = zero_momentum_percentage.apply(
            lambda row: row['Zero Momentum Percentage'] <= mean_zero_momentum_by_property[row['Property_Type']], axis=1
        )
        filtered_cities_property_types = zero_momentum_percentage[filtered_rows]

        # Merge the filtered data back with the original dataframe
        return pd.merge(df, filtered_cities_property_types[['City', 'Property_Type']], on=['City', 'Property_Type'], how='inner')


    def load_residential_data(self):
        """
        Load and process residential data:
        - Exclude commercial property types.
        - Handle missing values.
        - Calculate and filter based on asset value momentum.
        """
        # Load residential data, excluding commercial property types
        print("Loading residential data...")
        residential_data = self.load_data(is_commercial=False)
        initial_row_count = len(residential_data)
        print(f"Initial residential data rows: {initial_row_count}")

        # Exclude certain property types and count the number of rows removed
        residential_data = residential_data[~residential_data['Property_Type'].isin(['Mall', 'Office', 'Industrial'])]
        post_exclude_row_count = len(residential_data)
        print(f"Rows after excluding certain property types: {post_exclude_row_count}")
        print(f"Rows removed during exclusion: {initial_row_count - post_exclude_row_count}")

        # Calculate missing value percentages and filter cities
        print("Calculating missing value percentages and filtering cities...")
        missing_percentage_ppsf = residential_data.groupby('City')['median_ppsf'].apply(
            lambda x: (x.isnull().sum() / len(x)) * 100
        ).reset_index(name='Percentage Missing median_ppsf')
        filtered_cities = missing_percentage_ppsf[missing_percentage_ppsf['Percentage Missing median_ppsf'] < 25]['City']
        pre_filter_row_count = len(residential_data)
        residential_data = residential_data[residential_data['City'].isin(filtered_cities)]
        post_filter_row_count = len(residential_data)
        print(f"Rows after filtering cities: {post_filter_row_count}")
        print(f"Rows removed during city filter: {pre_filter_row_count - post_filter_row_count}")

        # Fill missing values and calculate asset value momentum
        print("Filling missing values and calculating asset value momentum...")
        for city in tqdm(filtered_cities, desc='Filling missing values per city and property type'):
            for property_type in residential_data['Property_Type'].unique():
                city_type_data = residential_data[(residential_data['City'] == city) & (residential_data['Property_Type'] == property_type)]
                fill_value = city_type_data['median_ppsf'].mean()
                residential_data.loc[(residential_data['City'] == city) & (residential_data['Property_Type'] == property_type), 'median_ppsf'] = city_type_data['median_ppsf'].fillna(fill_value)
        residential_data = self.calculate_asset_value_momentum(residential_data)

        # Filter data based on zero momentum percentage
        print("Filtering data based on zero momentum percentage...")
        mean_zero_momentum_by_property = residential_data.groupby('Property_Type')['asset_value_momentum'].apply(
            lambda x: (x == 0).mean() * 100
        )
        filtered_selected_data = self.filter_data_based_on_momentum(residential_data, mean_zero_momentum_by_property)
        final_row_count = len(filtered_selected_data)
        print(f"Final rows after all processing: {final_row_count}")
        print(f"Total rows removed during processing: {initial_row_count - final_row_count}")

        return filtered_selected_data

    def remove_unwanted_columns(self, df):
        unwanted_columns = [
            'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',
            'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'Unnamed: 0.1.1.1.1.1'
        ]
        df = df.drop(columns=unwanted_columns, errors='ignore')
        return df

    def load_data(self, state=None, city=None, property_type=None, is_commercial=False):
        dataframes = []
        commercial_types = ['Mall', 'Industrial', 'Office']
        
        for st in self._get_subdirectories(self.directory) if state is None else [state]:
            for ct in self._get_subdirectories(os.path.join(self.directory, st)) if city is None else [city]:
                for pt in self._get_subdirectories(os.path.join(self.directory, st, ct)) if property_type is None else [property_type]:
                    if pt == 'All Residential':
                        continue
                    
                    # Corrected typo: Check if the property type is commercial when is_commercial is True
                    if is_commercial and pt not in commercial_types:
                        continue

                    file_path = os.path.join(self.directory, st, ct, pt, 'forecasted_dataset.csv')
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, parse_dates=['period_begin'])
                        df = self.remove_unwanted_columns(df)
                        df = self._filter_by_columns(df, is_commercial)
                        df = self._filter_by_date(df, is_commercial)
                        df['State'] = st
                        df['City'] = ct
                        df['Property_Type'] = pt
                        dataframes.append(df)
        
        all_data = pd.concat(dataframes, ignore_index=True)
        return all_data

    def _filter_by_columns(self, df, is_commercial):
        columns = self.commercial_columns if is_commercial else self.residential_columns
        common_columns = [col for col in columns if col in df.columns]
        return df[common_columns]


    def _filter_by_date(self, df, is_commercial):
        cutoff_date = '2023-09-30' if is_commercial else '2023-10-31'
        return df[df['period_begin'] < pd.to_datetime(cutoff_date)]

    def _get_subdirectories(self, path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def aggregate_data(self, group_by, is_commercial=False):
        dataframes = []
        for state in self._get_subdirectories(self.directory):
            for city in self._get_subdirectories(os.path.join(self.directory, state)):
                for property_type in self._get_subdirectories(os.path.join(self.directory, state, city)):
                    if property_type == 'All Residential' and not is_commercial:
                        continue
                    file_path = os.path.join(self.directory, state, city, property_type, 'forecasted_dataset.csv')
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, parse_dates=['period_begin'])
                        df = self.remove_unwanted_columns(df)
                        df = self._filter_by_columns(df, is_commercial)
                        df = self._filter_by_date(df, is_commercial)
                        df['State'] = state
                        df['City'] = city
                        df['Property_Type'] = property_type
                        dataframes.append(df)
        all_data = pd.concat(dataframes, ignore_index=True)
        return all_data.groupby(group_by).mean().reset_index()

    def get_states(self, data):
        return data['State'].unique().tolist()

    def get_cities(self, data):
        return data['City'].unique().tolist()

    def get_property_types(self, data):
        return data['Property_Type'].unique().tolist()
    
    def save_filtered_data(self, filtered_selected_data):
        """
        Save the filtered data into separate CSV files for each state, city, and property type.
        """
        for state in tqdm(filtered_selected_data['State'].unique(), desc='Processing states'):
            for city in filtered_selected_data[filtered_selected_data['State'] == state]['City'].unique():
                for property_type in filtered_selected_data[(filtered_selected_data['State'] == state) & (filtered_selected_data['City'] == city)]['Property_Type'].unique():
                    # Filter data for the specific state, city, and property type
                    subset = filtered_selected_data[(filtered_selected_data['State'] == state) & (filtered_selected_data['City'] == city) & (filtered_selected_data['Property_Type'] == property_type)]
                    
                    # Create directory path
                    directory_path = os.path.join(self.directory, state, city, property_type)
                    os.makedirs(directory_path, exist_ok=True)

                    # Save to CSV
                    file_path = os.path.join(directory_path, 'selected_data.csv')
                    subset.to_csv(file_path, index=False)
