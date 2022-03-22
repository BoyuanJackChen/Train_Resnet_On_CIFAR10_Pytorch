from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from project1_model import project1_model
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--load_checkpoint', type=str, default="checkpoints/lr1e-3/e300_b64_lr0.001.pt")
FLAGS = parser.parse_args()

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
    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 64}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalization parameters from https://github.com/kuangliu/pytorch-cifar/issues/19
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    dataset2 = datasets.CIFAR10('../data', train=False,
                                transform=transform_test)   # 10k
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Initialize model
    model = project1_model().to(device)
    print(f"Model has {count_parameters(model)} parameters")

    # Load checkpoint
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    train_loss_list = loss[0]
    test_loss_list = loss[1]
    accuracy_list = loss[2]

    test_loss, accuracy = test(model, device, test_loader)
    plot_losses(train_loss_list, test_loss_list)



if __name__ == '__main__':
    main(FLAGS)