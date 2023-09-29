#from https://github.com/shashnkvats/Indofashionclip/blob/main/indofashion_clip.py
#with modifications
import json
from PIL import Image
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip


json_path = 'simple.json'
image_path = 'training'

with open(json_path, 'r') as f:
    input_data = json.load(f)

# Load the CLIP model and processor
#device = "cuda:0" if torch.cuda.is_available() else "cpu" 
device = torch.device("mps")# if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load('ViT-B/32', jit=False)

# Define a custom dataset
class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        self.list_txt = list_txt

    def __len__(self):
        return len(self.list_txt)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        try:
            image = preprocess(Image.open(self.image_path[idx]))
        except Exception as e:
            print(f"Unable to open image at {self.image_path[idx]} due to error : {e}")
            return None
        title = clip.tokenize([self.list_txt[idx]]).squeeze(0)
        return image, title

# use your own data
# Make sure each image path has one text
list_image_path = []
list_txt = []
for item in input_data:
  if 'filename' in item and 'captions' in item:
    img_path = os.path.join('training', item['filename'].split('/')[-1])
    caption = item['captions'][:40]
    # appending path to image then the corresponding caption
    list_image_path.append(img_path)
    list_txt.append(caption)
    

dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=50, shuffle=True) #Define your own dataloader

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

num_epochs = 30
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total = len(train_dataloader))
    for batch in pbar:
        if batch is None:
            continue
        optimizer.zero_grad()
        images,texts = batch 
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        total_loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

#save model after training https://github.com/openai/CLIP/issues/83
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"model_checkpoint/archae-ai.pt") #just change to your preferred folder/filename
