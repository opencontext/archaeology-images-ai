import os
import pandas as pd
import requests
import time
from requests.exceptions import Timeout

# Load the csv file
data = pd.read_csv('csv_data/artifact_images_w_descriptions.csv')

# Filter the dataset to content with image_file__uri
image_data = data[data['image_file__uri'].notna()]

# Reset index before split to ensure unique indices
image_data = image_data.reset_index(drop=True)

# Randomly select 2% for training, rest for testing
train_data = image_data.sample(frac=0.10, random_state=42)
test_data = image_data.drop(train_data.index)

# Save metadata as csv
train_data.to_csv('training_metadata.csv', index=False)
test_data.to_csv('testing_metadata.csv', index=False)

# Create training folder if it doesn't exist
os.makedirs('training10', exist_ok=True)



url_errors = []
# Download images and save into training folder
# this handles the repeated 'default.jpg' image name
for _, row in train_data.iterrows():
    url = row['image_file__uri']
    # Get the extension for the image file from the URL
    extension = url.split('.')[-1]
    # Get the UUID for the media file
    media_uuid = row['media__uri'].split('/')[-1]

    # Make a unique file_name from the UUID of the Open Context media resource. This has
    # the advantage of making sure that the image files can be easily looked up on
    # Open Context itself.
    file_name = f'{media_uuid}.{extension}'
    file_path = os.path.join('training10', file_name)

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except (requests.exceptions.RequestException, Timeout):
        print(f'An error occurred while fetching: {url}')
        url_errors.append(url)
        continue

    with open(file_path, 'wb') as img_file:
        img_file.write(response.content)

    # Wait for .1 second before the next request to reduce the load on server
    time.sleep(0.1)
