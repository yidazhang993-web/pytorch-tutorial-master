from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/transforms")
img = Image.open("dataset/train/bees/465133211_80e0c27f60.jpg")
print(img)

#ToTensor使用
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor", img_tensor)

#Normalize使用
print(img_tensor[0][0][0])
trans_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])#归一化,将图片像素值归一化到[-1,1]
img_normalize = trans_normalize(img_tensor)
print(img_normalize[0][0][0])
writer.add_image("Normalize", img_normalize,2)

#Resize使用
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image("Resize", img_resize,0)
print(img_resize.shape)

#Compose使用
trans_resize2 = transforms.Resize((512))
trans_compose = transforms.Compose([trans_resize2, trans_tensor])#前面的输出作为后面输入的输入
img_resize2 = trans_compose(img)
writer.add_image("Compose", img_resize2,1)


writer.close()
