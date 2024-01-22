import os
import json
from tqdm import tqdm

Housing_Directory = 'filtered_housing'
Commercial = ['Mall', 'Industrial', 'Office']
commercial_targets = ['occupancy', 'rank_desirability', 'asset_value_momentum', 'remote_work_share']
housing_targets = ['asset_value_momentum', 'remote_work_share']

to_predict = {}
total_predictions = 0

# Iterate through states
for state in tqdm(os.listdir(Housing_Directory), desc='States'):
    state_path = os.path.join(Housing_Directory, state)
    
    # Check if it's a directory
    if os.path.isdir(state_path):
        
        # Iterate through cities
        for city in os.listdir(state_path):
            city_path = os.path.join(state_path, city)
            if city == 'Jersey City':
                commercial_targets = ['occupancy', 'rank_desirability', 'asset_value_momentum']
            else:
                commercial_targets = ['occupancy', 'rank_desirability', 'asset_value_momentum', 'remote_work_share']
            
            # Check if it's a directory
            if os.path.isdir(city_path):
                
                # Iterate through property types
                for property_type in os.listdir(city_path):
                    property_type_path = os.path.join(city_path, property_type)
                    
                    # Check if it's a directory
                    if os.path.isdir(property_type_path):
                        current_targets = commercial_targets if property_type in Commercial else housing_targets
                        type = 'Commercial' if property_type in Commercial else 'Housing'
                        
                        # Check for missing target files
                        missing_targets = [target + '.csv' for target in current_targets if not os.path.exists(os.path.join(property_type_path, target + '.csv'))]
                        if missing_targets:
                            to_predict[property_type_path] = (missing_targets, type)
                            total_predictions += len(missing_targets)

# Print the number of predictions to make
print(f"Total predictions to make: {total_predictions}")

# Write the to_predict dictionary to a JSON file
with open('to_predict.json', 'w') as file:
    json.dump(to_predict, file, indent=4)
