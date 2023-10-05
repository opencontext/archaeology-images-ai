import os
import json

# Folder
path_to_folder = 'training'

# Json
json_file_path = 'simple.json'

# get list of files
folder_files = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]

# Load JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

new_data = [item for item in data if item['filename'] in folder_files]

# Save JSON
with open(json_file_path, 'w') as f:
    json.dump(new_data, f, indent=4)