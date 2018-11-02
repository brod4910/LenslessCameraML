import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import make_model
import models
import cv2
import LenslessDataset
from normalize import CastTensor
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
# from train import test_epoch

def CreateArgsParser():
    parser =  argparse.ArgumentParser(description='Evaluate Pretrained Model')

    parser.add_argument('--model', default= None, required= True,
                    help='file to load checkpoint from')
    parser.add_argument('--root-dir', required= True,  
                    help='root directory where enclosing image files are located')
    parser.add_argument('--test-csv', required= True, 
                    help='path to the location of the test csv')
    parser.add_argument('--gaussian', default= None, type= int,
                    help='Adds gaussian noise with std given by user')
    parser.add_argument('--shift', default= None, type= int, 
                    help='Shifts the image by the int given by user')
    parser.add_argument('--bias', default= None, type= int,
                    help='Adds bias noise to the image by the int given by user')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--resize', required= True, type=int, default=None, 
                    help='dimensions of both height and width to be resized')

    return parser

def main():
    args = CreateArgsParser().parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda is True:
        cudnn.benchmark = True

    if args.model is not None:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)

            network = make_model.Model(make_model.make_layers(models.feature_layers[checkpoint['f_layers']]), 
                    make_model.make_classifier_layers(models.classifier_layers[checkpoint['c_layers']]), checkpoint= True)

            network.load_state_dict(checkpoint['state_dict'])
            network.eval()
            network = network.to(device)

            batch_size = checkpoint['batch_size']
            resize = checkpoint['resize']
            print("=> loaded model '{}'"
                  .format(args.model))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
            raise AssertionError("Failed to load Model")

    if torch.cuda.device_count() > 1:
        print("===> Number of GPU's available: %d" % torch.cuda.device_count())
        network = nn.DataParallel(network)

    print("\nBatch Size: %d" % (batch_size))

    if args.shift is not None:
        data_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        Shift(np.float32([[1, 0, args.shift], [0, 1, 0]])),
        transforms.ToTensor(),
        CastTensor('torch.FloatTensor'),
        transforms.Normalize([40414.038877341736], [35951.78672059086])
        ])
    elif args.gaussian is not None:
        data_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        GaussianNoise(args.gaussian),
        transforms.ToTensor(),
        CastTensor('torch.FloatTensor'),
        transforms.Normalize([40414.038877341736], [35951.78672059086])
        ])
    elif args.bias is not None:
        data_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        BiasNoise(args.bias),
        transforms.ToTensor(),
        CastTensor('torch.FloatTensor'),
        transforms.Normalize([40414.038877341736], [35951.78672059086])
        ])

    # # load the test dataset
    # test_dataset = LenslessDataset.LenslessDataset(
    # csv_file= args.test_csv,
    # root_dir= args.root_dir,
    # transform= data_transform
    # )

    # test_loader = torch.utils.data.DataLoader(
    # test_dataset,
    # batch_size= batch_size,
    # shuffle= True,
    # num_workers= 2,
    # pin_memory= True
    # )
    
if __name__ == '__main__':
    main()
