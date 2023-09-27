## background

See https://medium.com/aimonks/a-guide-to-fine-tuning-clip-models-with-custom-data-6c7c0d1416fb 

and

https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset?select=train_data.json

## things to look into

- how many images are sufficient - what is a representative sample size?
- if data augmentation is used on the images (creating copies with different orientations, flipped horizontal, vertical, etc) how to make sure the augmented images are automatically tied to the correct descriptive metadata?
- how much juice is this gonna take: currently using a mac mini m1 2020 with 16 gb.

## sept 27

- there are 72363 images in the dataset. What's a representative sample size?

some quick googling...

```
N population size		72363
e margin of error		0.025
	n = N/(1+N*e^2)	
n  	1565.388099	
n = 1566		
```

...or 2 percent. That will certainly make things a bit easier on my machine. Let's start with that.

- [x] write a quick scraper to obtain images. See [download.py](download.py)
- [x] make sure filenames are present in the json; if not, add them. see [add_file_paths_to_json.py](add_file_paths_to_json.py). Should've done this first.
- [x] create a json file with just the info for the images downloaded. See [create_training_json.py](create_training_json.py)
- [ ] retrain clip using the guide & code from above
- [ ] make this retrained clip model available to [llm-clip](https://simonwillison.net/2023/Sep/12/llm-clip-and-chat/)

### problems and thoughts 
 - ~~some files end in filenames that will make it easy more or less to associate the image with the metadata. But there are numerous images that end with 'default.jpg'. So need to rename these consistently so that I can make the connection with the metadata~~
 - ~~need to modify the json so that there is this kind of thing: `"image_path":string"images/train/0.jpeg"`~~
 - I wonder if including the findspot data, decimal degrees, might be a useful bit of info to encode in the text model? 
 - error in retrain_clip.py:

 ```
 retrain_clip.py", line 57, in <module>
    img_path = image_path + item['image_path'].split('/')[-1]
                            ~~~~^^^^^^^^^^^^^^
TypeError: list indices must be integers or slices, not str
```
Eventually figured out what I was doing wrong; loading data incorrectly and setting my own captions wrong. It's now training and I don't know how long it will take, but progress!
