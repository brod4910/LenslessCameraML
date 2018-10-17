import os
import sys
import argparse
from sklearn.preprocessing import MinMaxScaler
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

    return parser

def main():
    args = CreateArgsParser().parse_args()

    MinMaxScale(args.root_dir, args.train_csv, args.test_csv, args.path_to_save)


def MinMaxScale(root_dir, train_csv, test_csv, path_to_save):
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

    offline_scaler = MinMaxScaler()
    train_data = []
    test_data = []

    print("======> reading train csv")
    csv_data = pd.read_csv(train_csv)

    print("======> extracting images from csv")
    for row in csv_data.itertuples():
        img_name = os.path.join(root_dir, row[1])
        image = Image.open(img_name)
        image = image.resize((224,224))
        image = np.asarray(image)
        train_data.append([image, row[1], row[2]])
    
    print("======> fitting training data")
    offline_scaler.fit(train_data[:,0]) # fit to training set

    print("======> transforming training data")
    train_data[:,0] = offline_scaler.transform(train_data[:,0])

    print("======> saving training data")
    for im in train_data:
        if not os.path.exists(train_path + str(im[2])):
            os.makedirs(train_path + str(im[2]))
        im[0] = Image.fromarray(np.uint8(im[0]*255)) # multiply to 255 to save image so the image can be used by ToTensor() in pytorch
        im[0].save(os.path.join(train_path, im[2], im[1]))

    del train_data

    print("======> readining test csv")
    csv_data = pd.read_csv(test_csv)

    print("======> extracting images from csv")
    for row in csv_data.itertuples():
        img_name = os.path.join(root_dir, row[1])
        image = Image.open(img_name)
        image = image.resize((224,224))
        image = np.asarray(image)
        test_data.append([image, row[1], row[2]])

    print("======> transforming test data")
    test_data[:,0] = offline_scaler.transform(test_data[:,0])

    print("======> saving test data")
    for im in train_data:
        if not os.path.exists(test_path + str(im[2])):
            os.makedirs(test_path + str(im[2]))
        im[0] = Image.fromarray(np.uint8(im[0]*255)) # multiply to 255 to save image so the image can be used by ToTensor() in pytorch
        im[0].save(os.path.join(test_path, im[2], im[1]))



if __name__ == '__main__':
    main()