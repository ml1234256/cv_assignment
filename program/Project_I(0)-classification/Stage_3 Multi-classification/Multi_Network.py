# It's empty. Surprise!
# Please complete this by yourself.
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

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

        self.fc21 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1)
        self.fc22 = nn.Linear(150, 3)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 62 * 62)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        out1 = self.fc21(x)
        out1 = self.softmax1(out1)
        out2 = self.fc22(x)
        out2 = self.softmax2(out2)

        return {'classes': out1, 'species': out2}

class MultiClassificationNetWork(nn.Module):
    def __init__(self):
        super(MultiClassificationNetWork, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Sequential()
        self.class_classifier = nn.Sequential(nn.Linear(512, 2), nn.Softmax(dim=1))
        self.species_classifier = nn.Sequential(nn.Linear(512, 3), nn.Softmax(dim=1))
        
    def forward(self,x):
        out = self.model(x)
        out1 = self.class_classifier(out)
        out2 = self.species_classifier(out)
        return {'classes': out1, 'species': out2}


if __name__ == '__main__':
    import numpy as np
    model = MultiClassificationNetWork()
    print(model)
    input = torch.Tensor(np.zeros((1, 3, 224, 224)))
    out = model(input)
    print(out['classes'].shape, out['species'].shape)
