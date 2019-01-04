from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random

'''
    Dataset for the lensless images.
    The csv file must contain the relative path to the images
    The root dir must be the top level directory of where all of the images.

    Bare transform is a list of image transformations. These transformations will be taken as is, if and only if
    extra_transform is None. If extra_transform is not none then bare_transform MUST contain a None entry where the
    extra transforms will be placed. 

    The extra transforms are a list of more image transformations. These transformations are randomly chosen. The
    random selection either contains one of the transformations or all of them.
'''
class LenslessDataset(Dataset):
    def __init__(self, csv_file, root_dir, bare_transform = None, extra_transform = None):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.bare_transform = bare_transform
        self.extra_transform = extra_transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])

        image = Image.open(img_name)
        label = self.csv_data.iloc[idx, 1]

        transform = self.generate_transforms(self.bare_transform, self.extra_transform)

        transform = transforms.Compose(transform)

        if transform:
            image = transform(image)

        image = image.type(torch.FloatTensor)

        return image, label

    def generate_transforms(self, bare_transform, extra_transforms):
        data_transform = []

        if extra_transforms is not None:
            random_transform = []
            for transform in extra_transforms:
                random_transform.append(transform)
            random_transform.append(random_transform[:])
            chosen = self.random_choice(random_transform)
            
            for t in bare_transform:
                if t is None:
                    if isinstance(chosen, list):
                        data_transform += chosen
                    else:
                        data_transform.append(chosen)
                else:
                    data_transform.append(t)
        else:
            data_transform += bare_transform
                
        return data_transform


    def random_choice(self, seq):
        choice = list(seq)
        choice = random.choice(choice)
        
        return choice



