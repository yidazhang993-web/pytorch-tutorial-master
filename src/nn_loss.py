# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torch
from torch.nn import L1Loss
from torch import nn

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

input = input.reshape(1,1,1,3)# (batch_size,channel,height,width)
target = target.reshape(1,1,1,3)

loss = L1Loss(reduction='sum')
result = loss(input,target)
print(result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(input,target)
print(result_mse)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=x.reshape(1,3)
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)