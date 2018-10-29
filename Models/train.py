import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import sys
import shutil
import LenslessDataset
from normalize import Scaler

def train(args, model, device, checkpoint):

    # Data transformations
    if args.hflip is True and args.vflip is False:
        data_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
    elif args.vflip is True and args.hflip is False:
        data_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])
    elif args.vflip is True and args.hflip is True:
        data_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            # Scaler(args.root_dir, args.train_csv, args.resize, scaler="minmax"),
            transforms.ToTensor()            
            ])

    print("\nImages resized to %d x %d" % (args.resize, args.resize))

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
        num_workers= args.num_processes,
        pin_memory= True
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size= args.batch_size,
        shuffle= True,
        num_workers = args.num_processes,
        pin_memory= True
        )

    # set the optimizer depending on choice
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum= args.momentum, dampening=0, weight_decay= 0 if args.weight_decay is None else args.weight_decay, nesterov= False)
    elif args.optimizer == 'AdaG':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif args.optimizer == 'AdaD':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif args.optimizer == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay= 0 if args.weight_decay is None else args.weight_decay, momentum=args.momentum, centered=False)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("\nUsing optimizer: %s" % (args.optimizer))

    if args.loss_fn == 'CELoss':
        # set the Loss function as CrossEntropy
        criterion = torch.nn.CrossEntropyLoss().cuda() if device == "cuda" else torch.nn.CrossEntropyLoss()
    elif args.loss_fn == 'MMLoss':
        criterion = torch.nn.MultiMarginLoss().cuda() if device == "cuda" else torch.nn.MultiMarginLoss()

    # either take the minimum loss then reduce LR or take max of accuracy then reduce LR
    if args.plateau == 'loss':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', verbose= True, patience= 6)
    elif args.plateau == 'accuracy':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'max', verbose= True, patience= 6)

    print("\nReducing learning rate on %s plateau\n" % (args.plateau))

    best_prec1 = 0 if checkpoint is None else checkpoint['best_prec1']
    is_best = False

    del checkpoint

    # train and validate the model accordingly
    total_time = time.clock()
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_epoch(epoch, args, model, optimizer, criterion, train_loader, device)
        test_loss, accuracy = test_epoch(model, test_loader, device)

        if args.plateau == 'loss':
            scheduler.step(test_loss)
        elif args.plateau == 'accuracy':
            scheduler.step(accuracy)

        if accuracy > best_prec1:
            best_prec1 = accuracy
            is_best = True

        # save the model every epoch
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'time': time.clock() - total_time,
            'c_layers' : args.c_layers,
            'f_layers' : args.f_layers,
            'batch_size' : args.batch_size,
            'resize' : args.resize
        }, is_best)

        is_best = False

def train_epoch(epoch, args, model, optimizer, criterion, train_loader, device, accumulation_steps= 16):
    model.train()

    total_train_loss = 0
    batch_loss = 0

    optimizer.zero_grad()                                   # Reset gradients tensors

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)

        output = model(inputs)                     # Forward pass

        loss = criterion(output, targets) / accumulation_steps      # Compute loss function

        loss.backward()
        
        batch_loss += (loss.item()/accumulation_steps)
        total_train_loss += loss.item()

        if (batch_idx + 1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()

            if (batch_idx + 1) % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), batch_loss))
            # report the train metrics depending on the log interval

            batch_loss = 0 

        del inputs, targets, loss, output

    total_train_loss /= len(train_loader.dataset)
    print('\nAveraged loss for training epoch: {:.4f}'.format(total_train_loss))

def test_epoch(model, test_loader, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    correct = 0

    # validate the model over the test set and record no gradient history
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):

            input, target = input.to(device), target.to(device)

            output = model(input)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            del input, target, output

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

    return test_loss, accuracy

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def kFold(inputs, targets):
        kfold = KFold(5, True, 11)
        idxs = []

        for train, test in enumerate(kfold.split(inputs, targets)):
            idxs.append((train, test))

        return idxs
