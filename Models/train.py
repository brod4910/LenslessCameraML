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
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomHorizontalFlip(),
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

    optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= args.momentum, nesterov= True)
    criterion = torch.nn.CrossEntropyLoss().cuda() if device == "cuda" else torch.nn.CrossEntropyLoss()

    if args.lr_scheduler is True:
        print('Using LR scheduler on loss Plateau')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', verbose= True, patience= 6)

    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, optimizer, criterion, train_loader, device)
        test_loss = test_epoch(model, test_loader, device)

        if args.lr_scheduler is True:
            scheduler.step(test_loss)


def train_epoch(epoch, args, model, optimizer, criterion, train_loader, device):
    model.train()
    correct = 0

    for batch_idx, data in enumerate(train_loader):
        input_data, target = data['image'].type(torch.cuda.FloatTensor).to(device), data['label'].to(device)
        optimizer.zero_grad()

        output = model(input_data)
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
        for batch_idx, data in enumerate(test_loader):
            input_data, target = data['image'].type(torch.cuda.FloatTensor).to(device), data['label'].to(device)

            output = model(input_data)
            loss = F.cross_entropy(output, target, size_average=False)
            test_loss += loss.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss
