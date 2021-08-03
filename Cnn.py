import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

class Cnn(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        x = torch.randn(size, size, 3).view(1, size, size, 3)
        self.to_linear = None
        self.im_size = size
        self.convs(x)
        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 306)
        self.fc3 = nn.Linear(306, 153)
        self.fc4 = nn.Linear(153, 2)


    def convs(self, x):

        x_reshaped = []
        if len(x.shape) == 4:
            for i in range(len(x)):
                x_reshaped.append(self.rgb2gray(x[i]))
                x_reshaped[i] = x_reshaped[i].view(1, self.im_size, self.im_size)
        else:
            x_reshaped.append(self.rgb2gray(x))
            x_reshaped[0] = x_reshaped[0].view(1, self.im_size, self.im_size)
        x_reshaped = torch.cat(x_reshaped)
        x_reshaped = x_reshaped.view(-1, 1, self.im_size, self.im_size)
        x_reshaped = F.max_pool2d(F.relu(self.conv1(x_reshaped)), (2, 2))
        x_reshaped = F.max_pool2d(F.relu(self.conv2(x_reshaped)), (2, 2))
        x_reshaped = F.max_pool2d(F.relu(self.conv3(x_reshaped)), (2, 2))
        self.to_linear = x_reshaped[0].shape[0] * x_reshaped[0].shape[1] * x_reshaped[0].shape[2]
        return x_reshaped

    def forward(self, x):

        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)

    def rgb2gray(self, t):
        if t.shape[2] == 3:
            t = t[:, :, 0] * 0.2989 + t[:, :, 1] * 0.5870 + t[:, :, 2] * 0.1140
        else:
            t = t[0, :, :] * 0.2989 + t[1, :, :] * 0.5870 + t[2, :, :] * 0.1140
        t.view(1, self.im_size, self.im_size)
        return t