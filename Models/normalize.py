import os
import sys
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import KFold
np.set_printoptions(threshold=np.inf)
import pandas as pd

def CreateArgsParser():
    parser =  argparse.ArgumentParser(description='Normalize data in a csv')

    parser.add_argument('--root-dir', required= True,  
                    help='root directory where enclosing image files are located')
    parser.add_argument('--test-csv', required= True, 
                    help='path to the location of the test csv')
    parser.add_argument('--train-csv', required= True, 
                    help='path to the location of the train csv')
    parser.add_argument('--path-to-save', required= True, 
                    help='path to the location where the Normalized images will be saved')
    parser.add_argument('--scaler', default= "minmax", 
                    help='Scaler to normalize data. Options: Min Max Scaler (default), Standard Scaler (std)')

    return parser

def main():
    args = CreateArgsParser().parse_args()

    Scale_data(args.root_dir, args.train_csv, args.test_csv, args.path_to_save, args.scaler)

def Scale_data(root_dir, train_csv, test_csv, path_to_save, scaler):
    train_path = path_to_save + '/TRAIN'
    test_path = path_to_save + '/TEST'

    print("======> Creating folder to save to, if needed")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    print("======> Creating train folder to save to, if needed")
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    print("======> Creating test folder to save to, if needed")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if scaler == "minmax":
        offline_scaler = MinMaxScaler()
    elif scaler == "std":
        offline_scaler = StandardScaler()

    # extract train images from the csv
    train_data, train_labels = extract_images(train_csv)

    print("======> fitting training data")
    offline_scaler.fit(train_data) # fit to training set

    print("======> transforming training data")
    train_data = offline_scaler.transform(train_data)

    # save images to disk
    save_array(train_path, train_data, train_labels)

    del train_data, train_label

    # extract test images from csv
    test_data, test_labels = extract_images(test_csv)

    print("======> transforming testing data")
    test_data = offline_scaler.transform(test_data)

    save_array(test_path, test_data, test_labels)

def extract_images(csv_path):
    print("======> extracting images from csv")
    csv_data = pd.read_csv(csv_path)
    data = []
    labels = []
    for row in csv_data.itertuples():
        img_name = os.path.join(root_dir, row[1])
        image = Image.open(img_name)
        image = image.resize((224,224))
        image = np.asarray(image)
        data.append(image)
        labels.append([row[1], row[2]])

    data = np.array(data, dtype= np.float)
    nsamples, x, y = data.shape 
    data = data.reshape(nsamples, x*y) # reshape the images to input into scaler function

    return data, labels

# function modifies original data parameter to save memory
def save_array(path, data, label):
    print("======> saving data")
    data = data.reshape((nsamples, 224, 224))
    for im, img_name, label in zip(data, label):
        if not os.path.exists(path + str(label)):
            os.makedirs(path + str(label))
        im = Image.fromarray(np.uint8(im*255)) # multiply to 255 to save image so the image can be used by ToTensor() in pytorch
        im.save(os.path.join(path, label, img_name))

if __name__ == '__main__':
    main()