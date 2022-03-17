"""
Mixup code adapted from https://github.com/hongyi-zhang/mixup, the official
github repo from [Zhang et al. 2017].
"""
import torch
import numpy as np

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
    alpha = 0.2
    for i in range(20):
        print(np.random.beta(alpha, alpha))