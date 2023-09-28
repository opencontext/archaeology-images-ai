import torch
import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-B/32', jit=False)
model.to(device)

# Load the state dict previously saved
model.load_state_dict(torch.load('retrained_clip_model.ckpt'))