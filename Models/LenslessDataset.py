from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class LenslessDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform= None):
		self.csv_data = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.csv_data)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])

		image = Image.open(img_name)
		label = self.csv_data.iloc[idx, 1]

		if self.transform is not None:
			image = self.transform(image)


		return image, label
