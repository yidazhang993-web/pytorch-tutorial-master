# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：我是土堆

import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from model import *# 导入模型类，model.py中定义的Tudui类

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="./P10.data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./P10.data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型,已经在model.py中定义

tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()# 交叉熵损失函数

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)# 随机梯度下降优化器

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs/logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始

    tudui.train()# 切换到训练模式

    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:# 每训练100次, 打印一次损失值
            print(f"训练次数：{total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始

    tudui.eval()# 切换到测试模式

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():# 测试时, 不计算梯度，不会进行反向传播，也不会更新模型参数，不调优
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()# 计算正确预测的样本数
            total_accuracy = total_accuracy + accuracy

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui.state_dict(), f"tudui_{i}.pth")# 保存模型参数
    print("模型已保存")

writer.close()
