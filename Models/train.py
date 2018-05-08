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
		train_epoch(epoch, args, model, optimizer, criterion, train_loader, device)
		test_epoch(model, test_loader, device)



def train_epoch(epoch, args, model, optimizer, criterion, train_loader, device):
	model.train()
	correct = 0

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()

		output = model(data)
		loss = criterion(output, target)

		loss.backward()
		optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_epoch(model, test_loader, device):
	model.eval()
	test_loss = 0
	correct = 0

	with torch.no_grad():
		for data, target in train_loader:
			data, target = data.to(device), target.to(device)

			output = model(data)
			loss = F.cross_entropy(output, target, size_average=False)
			test_loss += loss.item()
			pred = output.data.max(1)[1]
			correct += pred.eq(target.data).sum()

    test_loss /= len(train_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
