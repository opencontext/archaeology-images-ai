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
- [x] retrain clip using the guide & code from above
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
Eventually figured out what I was doing wrong; loading data incorrectly, batch size too large, and setting my own captions wrong. 

~~It's now training and I don't know how long it will take, but progress!~~

Model trains, saves. I think I have m1 gpu properly used; training with cpu took about 5 minutes per epoch, but now it's running at about 3 minutes per epoch. Loss values are jumping all over the place which is probably a function of not enough training data.

## gpt the great leveller

I (SG) am not the world's best coder. I'm pretty crap, truth be told, and to fully explore and understand the code that Shashank Vats wrote I asked gpt4 to explain different elements of it to me when I needed to, and followed that up with very careful tweaking to see if what I thought I should be doing would actually work. I ran into a bit of trouble adding more than just caption/description data (we've got so much archaeological info!) and I used a combination of combing stack overflow and passing error messages to GPT4 to solve. What it came up with is in lines 47 to 57 of retrain_clip.py, which is pretty close to the original. It's explanation:

```
This double appending is creating pairs of images and their corresponding texts (captions and time ranges) for training. 

Here's the idea: each image is associated with two pieces of textual information -- a caption and a time range. For this reason, each image is appended twice to the `list_image_path` -- once for its caption and once for its time range. The corresponding pieces of text (the caption and the time range) are appended to `list_txt`. 

So, for each image, you will have two (image, text) pairs. For instance:

- Pair 1: (Image1, Caption1)
- Pair 2: (Image1, TimeRange1)

This can be useful when training models like CLIP, where each image paired with the corresponding text can represent a training sample. Both pieces of text — the caption and the time range — are considered vital contextual information for the image, and having them both in training data should help improve the model performance.

Remember, in this case your labels (if any) need also to be duplicated or have the same structure for continuity.
```

