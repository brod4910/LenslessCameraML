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
from normalize import Scaler
from train import test_epoch

def CreateArgsParser():
	parser =  argparse.ArgumentParser(description='Evaluate Pretrained Model')

	parser.add_argument('--model', default= None, required= True 
                    help='file to load checkpoint from')
    parser.add_argument('--root-dir', required= True,  
                    help='root directory where enclosing image files are located')
    parser.add_argument('--test-csv', required= True, 
                    help='path to the location of the test csv')
    parser.add_argument('--train-csv', required= True, 
                    help='path to the location of the training csv')
    parser.add_argument('--guassian', default= None, type= int,
                    help='Adds gaussian noise with std given by user')
    parser.add_argument('--shift', default= None, type= int, 
                    help='Shifts the image by the int given by user')
	parser.add_argument('--bias', default= None, type= int,
                    help='Adds bias noise to the image by the int given by user')
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
            resize = checkpoint['resize']
            # args.start_epoch = checkpoint['epoch'] + 1
            # best_prec1 = checkpoint['best_prec1']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded model '{}'"
                  .format(args.model))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
            raise AssertionError("Failed to load Model")

        del checkpoint

    if torch.cuda.device_count() > 1:
        print("===> Number of GPU's available: %d" % torch.cuda.device_count())
        network = nn.DataParallel(network)

    network = network.to(device)

    print("\nBatch Size: %d" % (batch_size))

    if args.shift is not None:
	    data_transform = transforms.Compose([
	    transforms.Resize((resize, resize)),
	    # Scaler(args.root_dir, args.train_csv, resize),
	    Shift(np.float32([[1, 0, args.shift], [0, 1, 0]])),
	    transforms.ToTensor()          
	    ])
	elif args.guassian is not None:
	    data_transform = transforms.Compose([
	    transforms.Resize((resize, resize)),
	    # Scaler(args.root_dir, args.train_csv, resize),
	    GuassianNoise(args.gaussian),
	    transforms.ToTensor()          
	    ])
	elif args.bias is not None:
	    data_transform = transforms.Compose([
	    transforms.Resize((resize, resize)),
	    # Scaler(args.root_dir, args.train_csv, resize),
	    Bias(args.bias),
	    transforms.ToTensor()          
	    ])

    # load the test dataset
    test_dataset = LenslessDataset.LenslessDataset(
    csv_file= args.test_csv,
    root_dir= args.root_dir,
    transform= data_transform
    )

    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size= batch_size,
    shuffle= True,
    num_workers= 2,
    pin_memory= True
    )

    test_lost, accuracy = test_epoch(network, test_loader, device)

'''
Shifts the image by shift. Shift here IS the shifting array.
The shifting matrix has dimensions 2x3 Ex: [[1, 0, s], [0, 1, t]], where s and t are the shifting constants
and the array is an np.float32
'''
class Shift(object):
	def __init__(self, shift):
		self.shift = shift

	def __call__(self, img):
		im = np.array(img, dtype= np.float)
		rows, cols = im.shape
		shifted_img = cv2.wrapAffine(im, self.shift, (cols, rows))

		return shifted_img
	
	# look into translate

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
		noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise

		return noisy_img_clipped

'''
Adds Guassian Noise to the image with mean and std.
The bias is added when the object is called
'''
class GuassianNoise(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, img):
		noisy_img = np.array(img, dtype= np.float)
		noisy_img = img + np.random.normal(mean, std, img.shape)
		noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise

		return noisy_img_clipped

	

