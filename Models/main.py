import os
import sys
import argparse
import torch
import torch.nn as nn
import model
from train import train

feature_layers = {
	'1': [['C', 1, 128, (3,3), 2, 1], ['M', (2,2), 2, 0], ['C', 128, 256, (3,3), 2, 0], ['M', (2,2), 2, 0], ['C', 256, 512, (3,3), 2, 1], ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 2, 1], ['M', (2,2), 2, 0]]
}

classifier_layers = {
	'1': [['L', 256 * 40 * 30, 1024], ['D'], ['FC', 1024, 2]]
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
	parser.add_argument('--root-dir', required= True, help='root directory where enclosing image files are located')
	parser.add_argument('--train-csv', required= True, help='path to the location of the training csv')
	parser.add_argument('--test-csv', required= True, help='path to the location of the test csv')

	return parser

def main():
	args = CreateArgsParser().parse_args()

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	network = model.Model(model.make_layers(feature_layers['1']), model.make_classifier_layers(classifier_layers['1'])).to(device)

	train(args, network, device)


if __name__ == '__main__':
	main()