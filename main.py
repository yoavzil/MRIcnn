from Brains_Dataset import Brains
from Cnn import Cnn
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torch.utils.data import Subset, DataLoader
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    data = Brains('brain_train')#Prepering the data for the cnn model.
    data.build_data()           #
    data.shuffle_data()         # 
    data.balance_data()         #

    data.train_test_split()

    train_x = data.train_data
    train_y = data.train_lables

    test_x = data.test_data
    test_y = data.test_lables

    cnn = Cnn(data.size) #Init cnn model.

    for i in range(10):
        optimizer = optim.Adam(cnn.parameters(), lr=0.00001) #Parameters for the training.
        loss_func = nn.MSELoss()                             #
        BATCH_SIZE = 8                                       #
        EPOCHS = 20                                          #

        #training
        for e in range(EPOCHS):
            for batch in range(0, len(train_x), BATCH_SIZE):
                batch_train = train_x[batch:batch + BATCH_SIZE]
                batch_train_l = train_y[batch:batch + BATCH_SIZE]
                cnn.zero_grad()
                out = cnn.forward(batch_train)
                loss = loss_func(out, batch_train_l) #Getting the loss of the model
                loss.backward() #Activating back propogation with that loss
                optimizer.step()
        cnn.eval()

            #testing
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(test_x)):
                real_out = torch.argmax(test_y[i])
                cnn_out = cnn.forward(test_x[i])
                pred_out = torch.argmax(cnn_out)
                if pred_out == real_out:
                    correct += 1
                total += 1
        print("Accuracy", correct / total)

