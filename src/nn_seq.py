# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # self.conv1 = Conv2d(3,32,5,padding="same")
        # self.maxpool1=MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding="same")
        # self.maxpool2=MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding="same")
        # self.maxpool3=MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1=Sequential(
            Conv2d(3,32,5,padding="same"),
            MaxPool2d(2),
            Conv2d(32,32,5,padding="same"),
            MaxPool2d(2),
            Conv2d(32,64,5,padding="same"),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self,x):
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        x=self.model1(x)

        return x

tudui = Tudui()
print(tudui)

input = torch.ones((64, 3, 32, 32)) 
output = tudui(input)
print(output.shape)

writer = SummaryWriter("logs/nn_seq")
writer.add_graph(tudui,input)
writer.close()