SG: I admit, I borked something completely with the other approach. So I went back to trawling through github to see if someone else had done this.

And of course, they had: https://github.com/damian0815/finetune-clip-huggingface based on: https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text

But I've learned alot about reshaping json...

If you're looking at the hot mess this repo is, the notebook in here is the one you want.

oh, and 'testing.ipynb' has some neat functions in it to query llm-clip and retrieve images, metadata, and link back to opencontext. It also has some instructions on where to put the retrained model and get it working with llm-clip.
