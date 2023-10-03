import os
import json

# Load the JSON file
with open('step1-output.json') as file:
    data = json.load(file)

# Get a list of all files in the 'training' directory
file_list = os.listdir('training')

# Initialize the new array
training_set = []

# Go through each JSON object and check if the 'image_path' exists in the
# training directory
for item in data:
    if 'image_path' in item and item['image_path']:
        image_name = item['image_path'].replace('training/', '')  # Remove 'training/' from path
        if image_name in file_list:
            training_set.append(item)

# Write the new array to a JSON file
with open('training_set.json', 'w') as file:
    json.dump(training_set, file)