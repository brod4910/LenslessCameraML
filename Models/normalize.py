# %load normalize.py
import os
import sys
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import numpy as np
from PIL import Image
import pandas as pd
'''
Fits the training to a Scaler. Scaler can be Standard or MinMax.
When used as a Pytorch Transform, the call funciton transforms the image
and returns the transformed image and label.
'''
class Scaler(object):
    def __init__(self, root_dir, train_csv, resize, scaler= "std"):
        if scaler == "minmax":
            self.online_scaler = MinMaxScaler()
        elif scaler == "std":
            self.online_scaler = StandardScaler()
        elif: scaler == "robust":
            self.online_scaler = RobustScaler()

        self.root_dir = root_dir
        self.train_csv = train_csv
        self.resize = resize
        # fit training data
        self.fit_data(self.train_csv, self.root_dir, self.online_scaler)
    
    def __call__(self, image, label):
        x,y = image.shape
        img = image.reshape(1, x*y)
        img = self.online_scaler.transform(img)
        img = img.reshape(x,y)
        img = np.uint8(img*255)
        img = np.clip(img, 0, 255)

        return img, label

    def fit_data(self, csv_path, root_dir, online_scaler):
        print("======> fitting training data")
        csv_data = pd.read_csv(self.train_csv)
        for row in csv_data.itertuples():
            img_name = os.path.join(self.root_dir, row[1])
            image = Image.open(img_name)
            image = image.resize((self.resize, self.resize))
            image = np.asarray(image, dtype= np.float)
            x, y = image.shape
            image = image.reshape(1, x*y)
            self.online_scaler.partial_fit(image)

    # function modifies original data parameter to save memory
    def save_images(self, path, data, label):
        print("======> saving data")
        data = data.reshape((nsamples, 224, 224))
        for im, img_name, label in zip(data, label):
            if not os.path.exists(path + str(label)):
                os.makedirs(path + str(label))
            im = Image.fromarray(np.uint8(im*255)) # multiply to 255 to save image so the image can be used by ToTensor() in pytorch
            im.save(os.path.join(path, label, img_name))
