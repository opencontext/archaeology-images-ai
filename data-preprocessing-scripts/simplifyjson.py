import json

# Load your json file
with open('training_set.json') as file:
    data = json.load(file)

new_data = []
for d in data:
    # Rename "image_path" to "filename", if it exists.
    if 'image_path' in d:
        d['filename'] = d.pop('image_path')

    # Initialize "captions" string
    captions = ""

    for key in list(d.keys()):
        # Delete 'image_file__uri', 'media__uri' keys and keys with None values
        if key in ['image_file__uri', 'media__uri'] or d[key] is None:
            del d[key]
        # If key is 'image_path' rename it to 'filename'
        elif key == 'image_path':
            d['filename'] = d.pop('image_path')
        # Check for non-empty values and if not 'none' and not the 'filename'
        elif key != 'filename' and  str(d[key]).strip().lower() != 'none':
            captions += str(d[key]) + " "
            del d[key]

    # Trim trailing whitespace and add "captions" to the dict
    d['captions'] = captions.strip()

    new_data.append(d)

# Save the modified data back to a new json file
with open('simple.json', 'w') as file:
    json.dump(new_data, file, indent=4)
