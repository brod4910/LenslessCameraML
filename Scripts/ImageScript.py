import shutil
import os, os.path
import sys
import argparse
import random


def CreateArgParser():
    # Training settings
    parser = argparse.ArgumentParser(description='Image Mover')

    parser.add_argument('--source', required= True)

    parser.add_argument('--dest_train', required= True)
    parser.add_argument('--dest_test', required= True)

    return parser

def MoveRandomImages():
    args = CreateArgParser().parse_args()

    path, dirs, files = os.walk(args.source).__next__()
    print(len(files))
    file_num = len(files)

    valid_files = {x: x for x in range(file_num)}
    train = {}

    print(valid_files)

    while True:
        rand_num = random.randrange(file_num)

        if str(rand_num) in train == False:
            train[str(rand_num)] = rand_num
            del valid_files[str(rand_num)]

        if len(valid_files) == 2200:
            break

    print(len(train))
    print(len(valid_files))
    print(train)

def MoveImages():
    args = CreateArgParser().parse_args()

    path, dirs, files = os.walk(args.source).__next__()
    print(len(files))
    num_files = len(files)

    fs = args.source
    fdtrain = args.dest_train
    fdtest = args.dest_test

    for i in range(5400):
        file_name = str(i) + '.png'
        file_source =  fs + file_name

        if i < 3200:
            file_train = fdtrain + file_name
            shutil.move(file_source, file_train)
        else:
            file_test = fdtest + file_name
            shutil.move(file_source, file_test) 

if __name__ == '__main__':
    MoveImages()

