"""
Mixup code adapted from https://github.com/hongyi-zhang/mixup, the official
github repo from [Zhang et al. 2017].
"""
import torch
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def mixup_data(x, y, alpha, device):
    # Pick one sample in the batch and use that to mixup with the whole batch
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    # This mixture assumes that the batch is shuffled
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    # a lambda function that takes in loss function and dummy y-hat to generate mixup loss
    return lambda lossfun, pred: lam * lossfun(pred, y_a) + (1 - lam) * lossfun(pred, y_b)


if __name__ == '__main__':
    # Visualize data mixture
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
    img1, label1 = dataset1[5]
    img1 = np.asarray(img1)
    img2, label2 = dataset1[1]
    img2 = np.asarray(img2)
    print(img1)
    input()
    print(label1, label2)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    mixup = (0.5*img2+0.5*img1)/255
    print(mixup.shape)
    plt.imshow(mixup)
    plt.show()

    # alpha = 0.2
    # for i in range(20):
    #     print(np.random.beta(alpha, alpha))