import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import sys
import LenslessDataset

def train(args, model, device):

	if args.resize is not None:
		data_transform = transforms.Compose([
			transforms.Resize(args.resize),
			transforms.ToTensor()
			])
	else:
		data_transform = transforms.Compose([
			transforms.ToTensor()
			])
	# create both training and testing datasets
	train_dataset = LenslessDataset.LenslessDataset(
		csv_file= args.train_csv,
		root_dir= args.root_dir,
		transform= data_transform
		)

	test_dataset = LenslessDataset.LenslessDataset(
		csv_file= args.test_csv,
		root_dir= args.root_dir,
		transform= data_transform
		)
	# use the torch dataloader class to enumerate over the data during training
	train_loader = torch.utils.data.DataLoader(
		train_dataset, 
		batch_size= args.batch_size, 
		shuffle= True, 
		num_workers= args.num_processes
		)

	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size= args.batch_size,
		shuffle= True,
		num_workers = args.num_processes
		)

	optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= args.momentum)
	criterion = torch.nn.CrossEntropyLoss().cuda() if device == "cuda:0" else torch.nn.CrossEntropyLoss()

	for epoch in range(1, args.epoch + 1):
		train(epoch, args, model, optimizer, criterion, train_loader)


def train_epoch(epoch, args, model, optimizer, criterion, train_loader):
