#!/usr/bin/python

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

MEAN_COEF = [0.485, 0.456, 0.406]
DIV_COEF = [0.229, 0.224, 0.225]

# functions to show an image
def imshow(img):
    npimg = img.numpy() * np.array(DIV_COEF).reshape([3,1,1]) \
          + np.array(MEAN_COEF).reshape([3,1,1])            # Un-normalize
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ImageNet Loader Test')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')

    args = parser.parse_args()

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=MEAN_COEF,
                                     std=DIV_COEF)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # Sample training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    plt.show()

    # Sample validation images
    dataiter = iter(val_loader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    plt.show()
