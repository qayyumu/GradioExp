from matplotlib import pyplot as plt 
import torch
from torchvision.transforms import ToTensor
from operator import mul
from functools import reduce
import numpy as np


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='valid')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='valid')
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


    
class toTensor:
    def __init__(self, data) -> None:
        self.data= data

    def __call__(self, image):
        image = ToTensor(image)
        return image.float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    y = list(map(int,y))
    one_h = np.eye(num_classes, dtype='uint8')[y]
    return torch.Tensor(one_h)

class ToCategorical:
    def __init__(self, max_classes=10) -> None:
        self.max_classes = max_classes

    def __call__(self, lable):
        lable = int(lable)
        categorical = torch.zeros(self.max_classes)
        if lable > self.max_classes:
            raise f'Provided {lable} maps out of class range as max classes are {self.max_classes}'
        categorical[lable] = 1
        return categorical

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class Flatten():
    def __init__(self,target_input=(28,28)) -> None:
        self.target_input = target_input
    def __call__(self, img):        
        return img.reshape(-1, reduce(mul,self.target_input)).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
def show_image(image, label):
    print(label)
    plt.imshow(image.numpy(), cmap='gray')