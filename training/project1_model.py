# resnet backbone reference: https://github.com/NERSC/pytorch-examples/blob/master/models/resnet_cifar10.py

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
in_planes: number of input channels
planes: number of output channels
"""
class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Bi = 2 residual blocks in each residual layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                # Ki = 1x1 kernel size for skip layer
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)    # Concatenate in the third dimension
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Create N residual layers
        # Bigger stride makes your dimension smaller, since your hidden layers are getting deeper
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # Ci = 64 channels in this layer
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)   # [stride, 1, 1, ...]. Only the first layer can have non-1 stride.
        # You can only use non-1 stride once, because otherwise you can't concatenate your identity skip connection
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
class ResNet_Small(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Small, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Create N residual layers
        # Bigger stride makes your dimension smaller, since your hidden layers are getting deeper
        self.layer1 = self._make_layer(block, int(64*1.3), num_blocks[0], stride=1)   # Ci = 64 channels in this layer
        self.layer2 = self._make_layer(block, int(128*1.3), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*1.3), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(256*1.3), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)   # [stride, 1, 1, ...]. Only the first layer can have non-1 stride.
        # You can only use non-1 stride once, because otherwise you can't concatenate your identity skip connection
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out    
    

def project1_model():
    #return ResNet(BasicBlock, [2, 2, 2, 2])
    return  ResNet_Small(BasicBlock, [2, 2, 2])
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)