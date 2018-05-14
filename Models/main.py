import os
import sys
import argparse
import torch
import torch.nn as nn
import model
from train import train

feature_layers = {
    '1': [['C', 1, 128, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 128, 256, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 256, 512, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], 
    ['M', (2,2), 2, 0]],

    # acheived a 92% accuracy on digits 0-9
    '2': [['C', 1, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 256, (3,3), 1, 1], ['C', 256, 512, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (3,3), 1, 1], 
    ['M', (2,2), 2, 0]],

    # 91% accuracy on digits 0-9
    '3': [['C', 1, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (5,5), 1, 2], 
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (5,5), 1, 2],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (5,5), 1, 2],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 512, (5,5), 1, 2],
    ['M', (2,2), 2, 0]],

    # 90% accuracy on digits 0-9
    '4': [['C', 1, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (5,5), 1, 2], ['C', 256, 512, (3,3), 1, 1], 
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (5,5), 1, 2], ['C', 256, 512, (3,3), 1, 1],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (5,5), 1, 2], ['C', 256, 512, (3,3), 1, 1],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1], ['C', 256, 128, (3,3), 1, 1], ['C', 128, 256, (5,5), 1, 2], ['C', 256, 512, (3,3), 1, 1],
    ['M', (2,2), 2, 0]],

    '5': [['C', 1, 64, (3,3), 1, 1], ['C', 64, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 128, (1,1), 1, 0], ['C', 128, 256, (3,3), 2, 0],
    ['C', 256, 64, (3,3), 1, 1], ['C', 64, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 128, (1,1), 1, 0], ['M', (2,2), 2, 0],
    ['C', 128, 64, (3,3), 1, 1], ['C', 64, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 128, (1,1), 1, 0], ['C', 128, 256, (3,3), 2, 0],
    ['C', 256, 64, (3,3), 1, 1], ['C', 64, 128, (3,3), 1, 1], ['C', 128, 256, (3,3), 1, 1], ['C', 256, 128, (1,1), 1, 0], ['M', (2,2), 2, 0]
    ]

}

classifier_layers = {
    '1': [['L', 256 * 40 * 30, 1024], ['D'], ['FC', 1024, 5]],
    '2': [['L', 512 * 10 * 10, 1024], ['D'], ['L', 1024, 2048], ['D'], ['FC', 2048, 10]],
    '3': [['L', 512 * 7 * 7, 2048], ['D'], ['L', 2048, 4096], ['D'], ['FC', 4096, 10]],
    '4': [['L', 512 * 7 * 7, 1024], ['D'], ['L', 1024, 2048], ['D'], ['FC', 2048, 10]],
    '5': [['L', 128 * 6 * 6, 1024], ['D'], ['L', 1024, 2048], ['D'], ['FC', 2048, 10]]
}

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
    parser.add_argument('--lr-scheduler', action='store_true')
    parser.add_argument('--root-dir', required= True, help='root directory where enclosing image files are located')
    parser.add_argument('--train-csv', required= True, help='path to the location of the training csv')
    parser.add_argument('--test-csv', required= True, help='path to the location of the test csv')

    return parser

def main():
    args = CreateArgsParser().parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    network = model.Model(model.make_layers(feature_layers['4']), model.make_classifier_layers(classifier_layers['4']))

    if torch.cuda.device_count() > 1:
        print("===> Number of GPU's available: %d" % torch.cuda.device_count())
        network = nn.DataParallel(network)

    network.to(device)

    train(args, network, device)


if __name__ == '__main__':
    main()