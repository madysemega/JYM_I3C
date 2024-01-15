import os
import shutil

def move_and_delete_condo_coop_files(housing_directory):
    for state in os.listdir(housing_directory):
        state_path = os.path.join(housing_directory, state)
        if os.path.isdir(state_path):
            for city in os.listdir(state_path):
                city_path = os.path.join(state_path, city)
                if os.path.isdir(city_path):
                    for property_type in os.listdir(city_path):
                        if property_type.startswith('Condo'):
                            condo_path = os.path.join(city_path, property_type)
                            co_op_path = os.path.join(condo_path, 'Co-op')

                            if os.path.isdir(co_op_path):
                                dataset_file = os.path.join(co_op_path, 'dataset.csv')
                                if os.path.isfile(dataset_file):
                                    # Move dataset.csv from Co-op to Condo folder
                                    shutil.move(dataset_file, condo_path)

                                # Delete the Co-op folder
                                shutil.rmtree(co_op_path)
                                print(f"Processed Co-op folder in {condo_path}")

if __name__ == "__main__":
    housing_directory = 'housing'  # Modify as per your directory structure
    move_and_delete_condo_coop_files(housing_directory)
