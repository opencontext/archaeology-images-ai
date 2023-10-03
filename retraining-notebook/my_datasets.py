from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.transforms import functional as F
import torch

class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        self.image_path = list_image_path
        self.list_txt = list_txt
        #self.transform = Compose([ToTensor()])
        self.transform = Compose([Resize((300, 300)), ToTensor()])
        #self.transform = Compose([lambda img: F.resize(img, (300, 300)), ToTensor()])
        
    def __len__(self):
        return len(self.list_txt)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_path[idx]).convert("RGB")
            image = self.transform(image)  # Apply the transforms
        except Exception as e:
            print(f"Unable to open image at {self.image_path[idx]} due to error : {e}")
            image = torch.zeros(3, 300, 300)
        return image, self.list_txt[idx] if self.list_txt[idx] else [""]