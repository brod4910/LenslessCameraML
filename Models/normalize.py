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
        elif scaler == "robust":
            self.online_scaler = RobustScaler()

        self.root_dir = root_dir
        self.train_csv = train_csv
        self.resize = resize
        # fit training data
        self.fit_data(self.train_csv, self.root_dir, self.online_scaler)
    
    def __call__(self, image):
        img = np.asarray(image, dtype=np.float)
        x, y = img.shape
        img = img.reshape(1, x * y)
        img = self.online_scaler.transform(img)
        img = img.reshape(x, y, 1)
        img = np.uint8(img*255)
        img = np.clip(img, 0, 255)

        return img

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

class RunningStats:

    def __init__(self, resize= (240,240)):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.resize = resize

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1
        
        image = Image.open(x)
        image = image.resize(self.resize)
        image = np.asarray(image)
        image = image.reshape(1, self.resize[0] * self.resize[1])
#         print(image)

        if self.n == 1:
            self.old_m = self.new_m = np.mean(image)
            self.old_s = 0
        else:
            self.new_m = self.old_m + (np.mean(image) - self.old_m) / self.n
            self.new_s = self.old_s + (np.std(image) - self.old_m) * (np.std(image) - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())

class CastTensor(object):
    def __init__(self, dtype):
        self.type = dtype

    def __call__(self, image):
        img = image.type(self.type)

        return img

'''
Shifts the image by shift. Shift here IS the shifting array.
The shifting matrix has dimensions 2x3 Ex: [[1, 0, s], [0, 1, t]], where s and t are the shifting constants
and the array is an np.float32
'''
class TranslateImage(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, img):
        translated_img = np.array(img, dtype= np.float)
        rows, cols, chan  = translated_img.shape
        translated_img = cv2.warpAffine(translated_img, self.shift, (cols, rows))
        translated_img = translated_img.reshape((cols, rows, chan))

        return translated_img

# class Defocus(network, batch_size, device, test_csv, root_dir):

'''
Adds bias noise to the input image.
The bias is added when the object is called
'''
class BiasNoise(object):
    def __init__(self, bias_noise):
        self.bias_noise = bias_noise

    def __call__(self, img):
        noisy_img = np.array(img, dtype= np.float) + self.bias_noise
        rows, cols, chan = noisy_img.shape
        noisy_img = noisy_img.reshape((cols, rows, chan))
        noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise

        return noisy_img_clipped

'''
Adds Gaussian Noise to the image with mean and std.
The bias is added when the object is called
'''
class GaussianNoise(object):
    def __init__(self, std, mean=0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noisy_img = np.array(img, dtype= np.float)
        rows, cols, chan = noisy_img.shape
        noisy_img = noisy_img + np.random.normal(self.mean, self.std, noisy_img.shape)
        noisy_img = noisy_img.reshape((cols, rows, chan))
        noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise

        return noisy_img_clipped

class MaxNormalization(object):
    def __init__(self, max_val):
        self.max = max_val

    def __call__(self, img):
        norm_img = np.array(img, dtype= np.float) * self.max
        rows, cols = norm_img.shape
        norm_img = norm_img.reshape((cols, rows, 1))

        return norm_img
