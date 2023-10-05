So, starting with your csv folder and the data in artifact_images_w_descriptions.csv, run:

1. download.py' to grab 10% at random, 7k ish.
2. then I create the json file with the relevant info using 'add_file_paths_to_json.py'
3. then 'create_training_json.py'. This two-step was necessary because initially I did things in two steps, but maybe I can merge 'em together.
4. then simplifyjson.json to create the metadata json for the retraining step (filenames, captions)

You end up with a folder called ‘training’ and a json file called simple.json, looks like this:

[
    {
        "filename": "training/opencontext-1-photo-dt-28.jpg",
        "captions": "archaeology artifact Coin Asia Turkey Domuztepe 6500 BCE to 5500 BCE coins (money) <copper and copper alloy> Artifact Name: Coin \n Object Type: Other objects \n Material: Cooper/Bronze \n Disposition: Marash Museum study collection;  Marash Mus. study"
    },
    {
        "filename": "training/opencontext-1-photo1-dt-33.jpg",
        "captions": "archaeology artifact Object Asia Turkey Domuztepe 6500 BCE to 5500 BCE beads (pierced objects) serpentine (mineral) Artifact Name: Bead \n Material: Serpentine/Serpetinit \n Disposition: Marash Mus. study"
    },
    {
        "filename": "training/opencontext-1-draw-dt-97.jpg",
        "captions": "archaeology artifact Object Asia Turkey Domuztepe 6500 BCE to 5500 BCE sling bullet ceramic (material) Artifact Name: Sling ball \n Material: Fired clay \n Disposition: Depot"
    }
]

Style of thing. Move ‘training’ and ‘simple.json’ into the retraining-notebook folder.

Then, in the retraining-notebook, start the finetune-clip.ipnyb. Make sure that my-datasets.py is in there too, since the code imports it.

Once you’ve run that notebook, each epoch writes a folder. Grab the last one, since it’ll be the most trained version.

You now need to go and get this https://huggingface.co/sentence-transformers/clip-ViT-B-32/tree/main and copy it to your machine. On my machine, I called it ‘retrained-model’. Move the pytorch_model.bin from the model_epoch_x folder in the retraining-notebook, and put it inside the 0_CLIPModel folder (overwrite the existing pytroch_model.bin if it’s there). Drop the config.json from the the model_epoch_x folder in the retraining-notebook inside the 0_CLIPModel too.

I’m assuming you’ve got llm-clip installed somewhere. Go find llm-clip.py and change

```
if self._model is None:
   self._model = SentenceTransformer('clip-ViT-B-32')
```
to point to your new model, like so: 

```
    def embed_batch(self, items):
        # Embeds a mix of text strings and binary images
        if self._model is None:
            self._model = SentenceTransformer('/path/to/your/retrained-model')
```

Now you’re ready to roll! Stuff a bunch of images into a testing folder.

At command line: `$ llm embed-multi photos --files testing/ '*.jpg' --binary -m clip`

Then search: `$ llm similar photos -c 'a fragment of a terracotta statue'`
