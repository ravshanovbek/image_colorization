import torch
import os
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

class MyDataLoader(Dataset):
    def __init__(self, img_x_path, img_y_path, transform=None):
        self.img_x = img_x_path
        self.img_y = img_y_path
        self.transform = transform
        self.file_list = os.listdir(self.img_x)  # List of filenames in the directory

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        single = self.file_list[idx]
        image_x = read_image(os.path.join(self.img_x, single), ImageReadMode.RGB)
        image_y = read_image(os.path.join(self.img_y, single), ImageReadMode.RGB)

        if self.transform:
            image_x = self.transform(image_x)
            image_y = self.transform(image_y)
        return image_x, image_y