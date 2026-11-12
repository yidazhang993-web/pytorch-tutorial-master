import torch
from torch import nn

class Tudui(nn.Module):# 定义一个简单的神经网络层
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

tuidui = Tudui()
x = torch.tensor(1.0)# 输入一个标量
output = tuidui(x)
print(output)
