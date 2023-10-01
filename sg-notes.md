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
- [x] make this retrained clip model available to [llm-clip](https://simonwillison.net/2023/Sep/12/llm-clip-and-chat/)

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

## sept 29

- [x] simplifyjson.py to make a json file with just filename and all available data as lines in a caption
- [x] which simplifies a bit of retrain_clip.py. And speeds it up to about 1.5 minutes per epoch.
- [x] figure out how to get llm-clip to use this model

I have saved the model, but I'm not sure if I'm saving everything that needs to be saved.

I find where clip stores its model information using `dirname "$(llm logs path)"` (see [LLM documentation](https://llm.datasette.io/en/stable/other-models.html))

I create a new .yaml file, with model_id, aliases, and the best I can see write now for specifying the location is 'api_base' which is likely wrong, but I point it to the model anyway. I put that in the io.datasette.llm folder in '~library/application support'.

Then, in llm-clip.py which is in my environment packages under llm-clip, I change line 13 model_id to point to archae_ai. 

Then the moment of truth:

`llm embed-multi photos --files testing/ '*.jpg' --binary -m archae_ai`

This creates `embeddings.db` in the io.datasette.llm folder. 

`llm similar photos -c 'coarseware'`

Which returns from my three testing images:
```
{"id": "opencontext-24-19820081e.jpg", "score": 0.2408098776503917, "content": null, "metadata": null}
{"id": "opencontext-24-19820008illi.jpg", "score": 0.2381943557648078, "content": null, "metadata": null}
{"id": "opencontext-24-20070123b.jpg", "score": 0.23373448574984274, "content": null, "metadata": null}
```
Top result therefore is **Image 1**:

![https://iiif.archivelab.org/iiif/opencontext-24-19820081ejpg/full/675,/0/default.jpg](https://iiif.archivelab.org/iiif/opencontext-24-19820081ejpg/full/675,/0/default.jpg)

...which open context tells me is a fragment of a terracotta figurine.

If I search for 'figurine', this image is tops again, but with a score of 0.27. For contrast, if I say 'Donald Trump', who admittedly is very orange/terracotta coloured, the same image is tops again but at 0.21. ('Terracotta': same image again, but at 0.25)

If I look for 'line drawing', I get this image at 0.25- **Image 2**:

![https://iiif.archivelab.org/iiif/opencontext-24-19820008illjpg/full/675,/0/default.jpg](https://iiif.archivelab.org/iiif/opencontext-24-19820008illjpg/full/675,/0/default.jpg)

If I look for 'A sketch of a terracotta object', I get the same image again, but at the highest score (so far, in my testing) of 0.29.

If I look for 'bucchero', **Image 3**:

![https://iiif.archivelab.org/iiif/opencontext-24-20070123bjpg/full/675,/0/default.jpg](https://iiif.archivelab.org/iiif/opencontext-24-20070123bjpg/full/675,/0/default.jpg)

it returns the terracotta fragment of figurine at 0.24 and the actual bucchero at 0.23. (But 'black pottery' returns the piece of bucchero at 0.25, terracotta at 0.239. It also scores the line drawing at 0.239 too).

So... I *think* I've successfully fine tuned the model. But to be sure, I'll need to do the test again, but generate the embeddings from the standard CLIP model.

So... I need to test on a much broader scale.


### Archae_AI versus Vanilla Clip:

|phrase|top image|correct?|archae_ai score|vanilla clip score|
|------|---------|--------|---------------|------------------|
|coarseware|1|yes|0.2408098776503917|0.2408098776503917|

...at which point I stop and say, oh crap, it was _always_ just using CLIP. So my understanding of what I needed to change up was wrong.

Back to the drawing board. Well, maybe part of the problem is I should upload the model to my huggingface space so that I can load/call/use it properly instead of rolling about trying to do what can probably be done but I don't know how.

## Sept 30

By jove, I think I've got it...

- [x] This [notebook](https://colab.research.google.com/drive/1v2tnk5dcWfZr7Gg4mBP89HSjAvrBtdMo) will retrain the model; I will tidy it up.
- [x] Download the pytorch_model.bin
- [x] Create a folder on your machine that matches the structure of this: https://huggingface.co/sentence-transformers/clip-ViT-B-32/tree/main and put the pytorch_model.bin file inside the `0_CLIPModel` folder. You _need_ all those .json files. And since you're not otherwise futzing with the basic CLIP-ness, it should be ok.
- [x] find the llm-clip.py file in your environment. Change

```
if self._model is None:
   self._model = SentenceTransformer('clip-ViT-B-32')
```
to point to your new model, like so: 

```
    def embed_batch(self, items):
        # Embeds a mix of text strings and binary images
        if self._model is None:
            self._model = SentenceTransformer('/Users/shawngraham/Documents/code-experiments/llm-commandline/archaeology-images-ai/retrained-model')
 ```

Then, at the command line,
```bash
$ llm embed-multi photos --files testing/ '*.jpg' --binary -m clip
```

will create new embeddings from your retrained model!!!

...and how do I know it's using the retrained model? The results that it returns are completely different:

```
$ llm similar photos -c 'coarseware'                              
{"id": "opencontext-24-19820008illi.jpg", "score": 0.10605615481948473, "content": null, "metadata": null}
{"id": "opencontext-24-20070123b.jpg", "score": 0.10564941635017805, "content": null, "metadata": null}
{"id": "opencontext-24-19820081e.jpg", "score": 0.10531343752578372, "content": null, "metadata": null}
```

now... they're not _right_; 19820081e is the piece of coarseware or terracotta. My finetuned model I think didn't have enough data, and if you look at the loss values, well, it looks like it got stuck.

But progress!

...further investigation - the loss values when the model trains plateau quite quickly. So I'm probably not using enough data here. GPT has sensible suggestions:

```
If the loss plateaus very quickly during the training process, it may indicate a few potential issues with the training data or model:

1. **Insufficient Data**: You may not have enough training data. Deep learning models typically need a lot of data to learn effectively. If your dataset is small, your model may not have enough examples to learn from, causing it to plateau quickly.

2. **Lack of Diversity in Data**: If your data does not have enough variety, the model might not learn effectively. It's important that your training data has adequate representation of the different classes or features you're trying to predict.

3. **Complexity of the Model**: The model may not be complex enough to learn the underlying patterns in the data. This is usually referred to as underfitting. You could try increasing model complexity by adding more layers or nodes.

4. **Learning Rate**: If the learning rate is too high, the loss may quickly decrease and then oscillate or plateau because the steps in the gradient descent are too large and it is unable to find the minimum.

5. **Saturated features**: Features used do not carry enough information to make predictions more accurate. This could happen if you've used a lot of irrelevant features or if there's a lot of redundant data.

6. **Data Preprocessing**: There may be issues with how the data was preprocessed — outliers, incorrect labels, data leaks, normalization issues, etc.

In each of these cases, it would be beneficial to investigate your data and model to identify why the model performance is plateauing, possibly incorporating techniques like data augmentation, changing learning rate, regularization, adding dropout, etc. to overcome this issue.
```

### to do

- [ ] augment image data by translating, reflecting, rotating images. 
- [ ] tie the metadata json to those images
- [ ] use a whole shittonne more data
- [ ] maybe pay for compute time to retrain


## gpt the great leveller

I (SG) am not the world's best coder. I'm pretty crap, truth be told, and to fully explore and understand the code that Shashank Vats wrote I asked gpt4 to explain different elements of it to me when I needed to, and followed that up with very careful tweaking to see if what I thought I should be doing would actually work. I ran into a bit of trouble adding more than just caption/description data (we've got so much archaeological info!) and I used a combination of combing stack overflow and passing error messages to GPT4 to solve. What it came up with is in lines 47 to 57 of retrain_clip.py, which is pretty close to the original. Its explanation:

```
This double appending is creating pairs of images and their corresponding texts (captions and time ranges) for training. 

Here's the idea: each image is associated with two pieces of textual information -- a caption and a time range. For this reason, each image is appended twice to the `list_image_path` -- once for its caption and once for its time range. The corresponding pieces of text (the caption and the time range) are appended to `list_txt`. 

So, for each image, you will have two (image, text) pairs. For instance:

- Pair 1: (Image1, Caption1)
- Pair 2: (Image1, TimeRange1)

This can be useful when training models like CLIP, where each image paired with the corresponding text can represent a training sample. Both pieces of text — the caption and the time range — are considered vital contextual information for the image, and having them both in training data should help improve the model performance.

Remember, in this case your labels (if any) need also to be duplicated or have the same structure for continuity.
```

