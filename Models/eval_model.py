import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import make_model
import models

def CreateArgsParser():
	parser =  argparse.ArgumentParser(description='Evaluate Pretrained Model')

	parser.add_argument('--model', default= None, required= True 
                    help='file to load checkpoint from')
    parser.add_argument('--root-dir', required= True,  
                    help='root directory where enclosing image files are located')
    parser.add_argument('--test-csv', required= True, 
                    help='path to the location of the test csv')

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
            batch_size = checkpoint['batch_size']
            # args.start_epoch = checkpoint['epoch'] + 1
            # best_prec1 = checkpoint['best_prec1']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded model '{}'"
                  .format(args.model))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))

        del checkpoint

    if torch.cuda.device_count() > 1:
        print("===> Number of GPU's available: %d" % torch.cuda.device_count())
        network = nn.DataParallel(network)

    network = network.to(device)

    print("\nBatch Size: %d" % (args.batch_size))




def shift_data(network, batch_size, device, test_csv, root_dir):
	# look into translate

def defocus_data(network, batch_size, device, test_csv, root_dir):


def add_bias_noise(network, batch_size, device, test_csv, root_dir):


def add_guassian_noise(network, batch_size, device, test_csv, root_dir):

	

