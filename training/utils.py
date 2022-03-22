import torch
import matplotlib.pyplot as plt
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(train_loss_list, test_loss_list):
    plt.plot(range(len(train_loss_list)),train_loss_list,'-',linewidth=3,label='Train error')
    plt.plot(range(len(test_loss_list)), test_loss_list, '-',linewidth=3,label='Test error')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def analyze_checkpoint(path):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    train_loss_list = loss[0]
    test_loss_list = loss[1]
    plot_losses(train_loss_list, test_loss_list)
    accuracy_list = loss[2]
    print(np.amax(accuracy_list))


if __name__=='__main__':
    path = "../checkpoints/Mixup/lr1e-3/e300_b64_lr0.001.pt"
    analyze_checkpoint(path)