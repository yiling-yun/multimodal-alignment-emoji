import os
import torch
from PIL import Image
from typing import Callable
from torch.utils.data import Dataset
import pandas as pd


class Emoji(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 inference=False):
        self.data_dir = data_path
        self.transforms = transform
        all_data = pd.read_csv(os.path.join(self.data_dir, '{}.csv'.format(split)))
        self.id = all_data['id']
        self.imgs = all_data['images']
        self.names = all_data['name']
        self.inference = inference

    def __len__(self):
        return len(self.imgs)

    def get_image_by_id(self, images, id):
        img = images[id]
        img = Image.open(os.path.join(self.data_dir, img)).convert('RGBA')
        img = pure_pil_alpha_to_color_v2(img)
        return img

    def __getitem__(self, idx):
        images = self.imgs[idx].strip().split(' ')
        if not self.inference:
            img_id = torch.randint(len(images), (1,))
            img = self.get_image_by_id(images, img_id)

            if self.transforms is not None:
                img = self.transforms(img)
            return img, 0.0
        else:
            img_list = []
            label = self.names[idx]
            companies = []
            for img_id in range(len(images)):
                img = self.get_image_by_id(images, img_id)
                if self.transforms is not None:
                    img = self.transforms(img)
                img_list.append(img)
                companies.append(images[img_id].split('/')[-2])
            return torch.stack(img_list), label, companies



def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


