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
# this handles the repeated 'default.jpg' image name
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

# Download images and save into training folder
# this bit grabbed all the other filenames that were unique
# and whose filepaths were not predictable, and so the code
# above missed.
#
#for _, row in train_data.iterrows():
#    url = row['image_file__uri']
#    if url.endswith(".jpg"):
#        try:
#            response = requests.get(url, timeout=5)
#            response.raise_for_status()
#        except (requests.exceptions.RequestException, Timeout):
#            print('An error occurred while fetching: ', url)
#            continue
#
#        file_path = os.path.join('training', os.path.basename(url))
#
#        with open(file_path, 'wb') as img_file:
#            img_file.write(response.content)
#
#        # Sleep for 1 second before fetching the next image to reduce load on server
#        time.sleep(1)
#
## I ran this whole thing twice and then merged the results because
## I'm just not smart enough to handle everything once.