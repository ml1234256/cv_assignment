import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 62 * 62, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        #print('1.', x.shape)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        #print('2.', x.shape)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

       # print('3.', x.shape)
        x = x.view(-1, 6 * 62 * 62)
        #print('4.', x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        #print('5.', x.shape)

        x = F.dropout(x, training=self.training)

        x_classes = self.fc2(x)
        x_classes = self.softmax1(x_classes)

        return x_classes

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == "__main__":
    import torch
    import numpy as np
    input = torch.Tensor(np.zeros((1,3, 224, 224)))
    model = ResNet34()
    print(model)
    output = model(input)
    print(output.shape)
