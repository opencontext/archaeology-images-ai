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
train_data = image_data.sample(frac=0.02, random_state=42)
test_data = image_data.drop(train_data.index)

# Save metadata as csv
train_data.to_csv('training_metadata.csv', index=False)
test_data.to_csv('testing_metadata.csv', index=False)

# Create training folder if it doesn't exist
os.makedirs('training', exist_ok=True)

# Download images and save into training folder
for _, row in train_data.iterrows():
    url = row['image_file__uri']
    
    if url.endswith(".jpg"):
        try: 
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            # Extract the name from the URL
            try:
                file_name = url.split("/iiif/")[1].split("/full/")[0] + '.jpg'
                file_path = os.path.join('training', file_name)
                
            except IndexError:
                print(f"Skipping URL due to parsing problem: {url}")
                continue

            with open(file_path, 'wb') as img_file:
                img_file.write(response.content)
        
        except (requests.exceptions.RequestException, Timeout):
            print('An error occurred while fetching: ', url)
            continue

        # Wait for .1 second before the next request to reduce the load on server
        time.sleep(.1)