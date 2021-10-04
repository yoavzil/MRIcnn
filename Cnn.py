import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

class Cnn(nn.Module):
    def __init__(self, size):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 5) #Input size = 1, output size = 32, filter size = 5x5 pxl
        self.conv2 = nn.Conv2d(32, 64, 5) #Input size = 32, output size = 64, filter size = 5x5 pxl
        self.conv3 = nn.Conv2d(64, 128, 5) #Input size = 64, output size = 128, filter size = 5x5 pxl
        x = torch.randn(size, size, 3).view(1, size, size, 3)
        self.to_linear = None
        self.im_size = size
        self.convs(x) #Passing a random param to get the input size for fc1.
        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 306)
        self.fc3 = nn.Linear(306, 153)
        self.fc4 = nn.Linear(153, 2)

    #This function passing input x through the 3 conv layers to a relu func and and 3 pooling layers with 2x2 pxl filter and 2 straids.
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
    #This function getting input x of size 100X100 pixles and passing it through the model and returning the output as a tensor of 2X1.
    def forward(self, x):

        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)
    #This function getting a tensor t in RGB form and convert it to a gray scale tensor according to the formula of matlab.
    def rgb2gray(self, t):
        if t.shape[2] == 3:
            t = t[:, :, 0] * 0.2989 + t[:, :, 1] * 0.5870 + t[:, :, 2] * 0.1140
        else:
            t = t[0, :, :] * 0.2989 + t[1, :, :] * 0.5870 + t[2, :, :] * 0.1140
        t.view(1, self.im_size, self.im_size)
        return t
