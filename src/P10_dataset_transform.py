import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./P10.dataset", train=True, transform=dataset_transform, download=True)#下载训练数据集
test_set = torchvision.datasets.CIFAR10(root="./P10.dataset", train=False, transform=dataset_transform, download=True)#下载测试数据集

# print(train_set[0])
# print(train_set.classes)

# img, target = train_set[0]
# print(img)
# print(target)
# print(train_set.classes[target])
# torchvision.transforms.ToPILImage()(img).show()

writer = SummaryWriter("logs/P10_dataset_transforms")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("train_set", img, i)

writer.close()