import os
import sys
import torch
import torch.nn as nn
import argparse
import make_model
import models
from train import train

def CreateArgsParser():
    parser = argparse.ArgumentParser(description='Lensless Camera Pytorch')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=.01, metavar='LR',
                    help='learning rate to start search (default: .01)')
    parser.add_argument('--momentum', type=float, default= .9, metavar='M',
                    help='SGD momentum. Do a search ranging from value to +.02 and -.02 (default: None)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--resize', type=int, default=None, 
                    help='dimensions of both height and width to be resized')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
    parser.add_argument('--optimizer', required= True, 
                    help= 'Type of optimizer to use. Select one optimzer. Options: SGD, AdaG, AdaD, Adam, RMS')
    parser.add_argument('--plateau', default= None, 
                    help= 'Measurement to plateau on. Options: loss, accuracy, None (for no LR Scheduling) (default: None)')
    parser.add_argument('--weight-decay', type=int, default=.1, metavar='N',
                    help='L2 decay starting value. (default: .1)')
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

    hyperparams = []

    if args.optimizer == 'SGD':
        # set the 3 learning rate, weight decay and momentum values for the nets
        lrs = [args.lr * 10]
        weight_decays = [args.weight_decay * 10]
        momentums = [args.momentum + .01]

        for i in range(1, 3):
            lrs.append(lrs[i-1] * .1)
            weight_decays.append(weight_decays[i-1] * .1)
            momentums.append(momentums[i-1] - .01)

        for i in range(len(lrs)):
            for j in range(len(weight_decays)):
                for k in range(len(momentums)):
                    hyperparams.append([lrs[i], weight_decays[j], momentums[k]])

    # print(hyperparams, len(hyperparams))

    for i in range(len(hyperparams)):
        if args.optimizer == 'SGD':
            args.lr = hyperparams[i][0]
            args.weight_decay = hyperparams[i][1]
            args.momentum = hyperparams[i][2]

        print("Creating net with lr: %.4f , weight decay: %.3f, momentum: %.2f" % (args.lr, args.weight_decay, args.momentum))

        network = make_model.Model(make_model.make_layers(models.feature_layers['2']), make_model.make_classifier_layers(models.classifier_layers['2']))

        if torch.cuda.device_count() > 1:
            print("===> Number of GPU's available: %d" % torch.cuda.device_count())
            network = nn.DataParallel(network)

        network = network.to(device)

        train(args, network, device)
        del network
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

