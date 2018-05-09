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
		self.img_arr = np.asarray(self.csv_data.iloc[:, 0])
		self.label_arr = np.asarray(self.csv_data.iloc[:, 1])
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.csv_data)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.img_arr[idx])

		image = Image.open(img_name)
		label = self.label_arr[idx]

		if self.transform is not None:
			image = self.transform(image)

		sample = {'image' : image, 'label': label}

		return sample
