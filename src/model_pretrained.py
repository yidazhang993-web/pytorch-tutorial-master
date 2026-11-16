# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆
import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./data_image_net", split='train', download=True,
#                                             transform=torchvision.transforms.ToTensor())


vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights='DEFAULT')

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./P10.data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))# 在最后增加一个全连接层
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)# 替换最后一个全连接层
print(vgg16_false)


