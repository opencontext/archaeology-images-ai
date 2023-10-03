import json
from PIL import Image
import random

# Load JSON file
with open('simple.json') as json_file:  #the flattened json file
    data = json.load(json_file)

# Define modification types and their corresponding PIL methods
modifications = {
    'rotate': lambda img: img.rotate(random.randint(10, 359)),
#    'rotate_90': lambda img: img.rotate(90),
#    'rotate_180': lambda img: img.rotate(180),
    'reflect': lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    'flip': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)
}

new_data = []
# Process each file
for item in data:
    filename = item['filename']
    captions = item['captions']
    original_img = Image.open(f'{filename}')

    for mod_name, mod_func in modifications.items():
        # Create modified image
        modified_img = mod_func(original_img)

        # Resize image
        #resized_img = modified_img.resize((300, 300)) #not preserving aspect ratio, but when we do, great big black bands!
        #resized_orig = original_img.resize((300, 300))

        # Create new filename
        new_filename = f'{filename.split(".")[0]}_{mod_name}.jpg'

        # Save modified image
        resized_img.save(f'{new_filename}')
        #resized_orig.save(filename)

        # Add new data to the json data
        new_item = {'filename': new_filename, 'captions': captions}
        new_data.append(new_item)

data.extend(new_data)

# Save the new JSON data
with open('augmented-data.json', 'w') as json_file:
    json.dump(data, json_file)