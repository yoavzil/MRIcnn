import torch
import cv2 as cv
import numpy as np
import os

class Brains:
    #Constructor of the data handler class. gets the path to the data as a parameter.
    def __init__(self, path):
        self.labels = {'NO': [0, 1], 'YES': [1, 0]}
        self.data = []
        self.path = path
        self.size = 100
        self.tumor_count = 0
        self.healthy_count = 0
        self.train_test_div = 0.2
        self.train_data = None
        self.train_lables = None
        self.test_data = None
        self.test_lables = None
        
    #This function construct the data from the file.
    #Each image is read as a gray scale image and than resize to the specified size.
    #The amount of images of each label is keep tracking for balance.
    def build_data(self):
        for subdir1, dirs1, files in os.walk(self.path):
            for dir1 in dirs1:
                for subdir2, dirs2, files in os.walk(os.path.join(subdir1, dir1)):
                    for dir2 in dirs2:
                        for subdir, dirs, files in os.walk(os.path.join(subdir2, dir2)):
                            for file in files:
                                p = os.path.join(subdir2, dir2, file)
                                img = cv.imread(os.path.join(subdir, file), 1)
                                img = cv.resize(img, (self.size, self.size))
                                self.data.append([np.array(img), self.labels[dir2]])
                                if dir2 == 'NO':
                                    self.healthy_count += 1
                                else:
                                    self.tumor_count += 1
    #This function shuffles the data.
    def shuffle_data(self):
        np.random.shuffle(self.data)
    #This function balance the amount of images of each lable.
    def balance_data(self):
        diff = self.tumor_count - self.healthy_count
        i = 0
        tDiff = diff
        if diff > 0: #Amount of tumor images is larger than healthy images so we need to reduce the tumor images.
            while tDiff > 0:

                if self.data[i][1][0] == 1:
                    del(self.data[i])
                    self.tumor_count -= 1
                    tDiff -= 1
                i += 1
        if diff < 0: #Amount of healthy images is larger than tumor images so we need to reduce the healthy images.
            diff *= -1
            while tDiff > 0:
                if self.data[i][1][0] == 0:
                    del(self.data[i])
                    self.healthy_count -= 1
                    tDiff -= 1
                i += 1
    #This function splits the data to a train set and a test set according to a certain decimal point.
    def train_test_split(self):
        X = torch.Tensor([i[0] for i in self.data])#.view(-1, self.size, self.size)
        X = X / 255.0 #Scaling the pxls values to be between 0 and 1.
        Y = torch.Tensor([i[1] for i in self.data])
        val_pct = 0.2
        val_size = int(len(X) * val_pct)
        self.train_data = X[:-val_size]
        self.train_lables = Y[:-val_size]
        self.test_data = X[-val_size:]
        self.test_lables = Y[-val_size:]



