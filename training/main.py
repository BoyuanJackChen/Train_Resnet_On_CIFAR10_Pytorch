# Reference: We modified the train/test framework from
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from torch.optim.lr_scheduler import StepLR

from preprocessing import mixup_data, mixup_criterion
from project1_model import project1_model
from wide_resnet import Wide_ResNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--checkpoint', type=int, default=20)
parser.add_argument('--load_checkpoint', type=str, default="../checkpoints/Mixup/lr5e-4/e140_b64_lr0.0005.pt")
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--train_batch', type=int, default=64, help="batch size for training dataset")
parser.add_argument('--test_batch', type=int, default=64)
parser.add_argument('--alpha', type=float, default=0.2)   # For beta distribution in mixup augmentation
FLAGS = parser.parse_args()

def train(model, device, train_loader, optimizer, epoch, alpha, mixup=True, verbose=False):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha, device)
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
        else:  # For comparison, you can set mixup to False.
            outputs = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if verbose and batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss = train_loss / len(train_loader)
    print(f"Avg train loss for epoch {epoch}: ", train_loss)
    return train_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            # No need for mixup in test set
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct/len(test_loader.dataset)
    print(f"Test set average loss: {test_loss}, Accuracy: {accuracy} \n")
    return test_loss, accuracy


def main(args):
    # Use gpu if available
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Running on device: {torch.cuda.get_device_name(0)}")
    # Parameters
    train_kwargs = {'batch_size': args.train_batch}
    test_kwargs = {'batch_size': args.test_batch}
    PATH = "../checkpoints/Mixup/run_5"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalization parameters from https://github.com/kuangliu/pytorch-cifar/issues/19
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                                transform=transform_train)  # 50k
    dataset2 = datasets.CIFAR10('../data', train=False,
                                transform=transform_test)   # 10k
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    # model = efficientnet.to(device)
    # model = Wide_ResNet(28, 2, 0.3, 10).to(device)
    model = project1_model().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    print(f"Model has {count_parameters(model)} parameters")

    # Initialize loss list
    train_loss_list = np.array([])
    test_loss_list = np.array([])
    accuracy_list = np.array([])
    start = time.time()
    start_epoch = 1

    # Load Checkpoint
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        train_loss_list = loss[0]
        test_loss_list = loss[1]
        accuracy_list = loss[2]
        start_epoch = checkpoint['epoch']+1

    # Training
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"starting epoch {epoch}")
        train_loss = train(model, device, train_loader, optimizer, epoch, args.alpha, mixup=True)
        test_loss, accuracy = test(model, device, test_loader)
        test_loss_list = np.append(test_loss_list, test_loss)
        train_loss_list = np.append(train_loss_list, train_loss)
        accuracy_list = np.append(accuracy_list, accuracy)
        if epoch % args.checkpoint == 0:
            loss_list = np.concatenate((np.array([train_loss_list]), np.array([test_loss_list])), axis=0)
            loss_list = np.concatenate((loss_list, np.array([accuracy_list])), axis=0)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
            }, PATH+f"/e{epoch}_b{args.train_batch}_lr{args.lr}.pt")
            print(f"Checkpoint saved at epoch {epoch}")
    duration = time.time()-start
    print(f"Training done. Took {format_time(duration)}")
    # plot_losses(train_loss_list, test_loss_list)
    print(f"Model, accuracy and loss history saved to {PATH}")


if __name__ == '__main__':
    main(FLAGS)
