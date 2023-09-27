import json

# Open and load the JSON file
with open('json_data/artifact_images_w_descriptions.json') as file:
    data = json.load(file)

# Loop over the items in JSON data
for item in data:
    # Check the existence of "image_file__uri" key and that its value is not None
    if "image_file__uri" in item and item["image_file__uri"]:
        url = item["image_file__uri"]
        file_name = ''

        # If URL ends with 'default.jpg', extract file name from the /iiif/ part
        if url.endswith('default.jpg'):
            try:
                file_name = url.split("/iiif/")[1].split("/full/")[0] + '.jpg'
            except IndexError:
                print(f"Skipping URL due to parsing problem: {url}")
                continue
        else:
            # Normal case, just extract the filename
            file_name = url.split("/")[-1]
        
        # Create the path (assuming filename has been properly created)
        item["image_path"] = f"training/{file_name}"
    else:
        item["image_path"] = None

# Write the updated data to a new JSON file
with open('output.json', 'w') as file:
    json.dump(data, file)