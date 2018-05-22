import os
import sys
import argparse
import torch
import torch.nn as nn
import make_model
import models
import wide_models
import make_wide_model
from train import train

def CreateArgsParser():
    parser = argparse.ArgumentParser(description='Lensless Camera Pytorch')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--resize', type=int, default=None, 
                    help='dimensions of both height and width to be resized')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
    parser.add_argument('--weight-decay', type=int, default=None, metavar='N',
                    help='L2 decay (default: None)')
    # parser.add_argument('--lr-scheduler', action='store_true')
    parser.add_argument('--plateau', default= 'loss', 
                    help= 'Measurement to plateau on. Either loss or accuracy')
    parser.add_argument('--architecture', default= 'deep', 
                    help= 'Model architecture to use. Options: deep and wide. (default: deep)')
    parser.add_argument('--optimizer', default= 'SGD', 
                    help= 'Type of optimizer to use. Options: SGD, AdaG, AdaD, Adam, RMS')
    parser.add_argument('--root-dir', required= True,  
                    help='root directory where enclosing image files are located')
    parser.add_argument('--train-csv', required= True, 
                    help='path to the location of the training csv')
    parser.add_argument('--test-csv', required= True, 
                    help='path to the location of the test csv')

    return parser

def main():
    args = CreateArgsParser().parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.architecture == 'deep':
        network = make_model.Model(make_model.make_layers(models.feature_layers['2']), make_model.make_classifier_layers(models.classifier_layers['2']))
    elif args.architecture == 'wide':
        network = make_wide_model.Wide_Model(make_wide_model.make_wide_layers(wide_models.feature_layers['1']), make_wide_model.make_classifier_layers(wide_models.classifier_layers['1']))

    if torch.cuda.device_count() > 1:
        print("===> Number of GPU's available: %d" % torch.cuda.device_count())
        network = nn.DataParallel(network)

    network = network.to(device)

    train(args, network, device)


if __name__ == '__main__':
    main()