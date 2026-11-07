from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs")

# 只收集真正的图片文件，避免目录里的 .gif/.txt/.DS_Store 等干扰
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

class MyData(Dataset):

    def __init__(self, root_dir, class_dir, transform):
        # 数据集是 ImageFolder 结构：root_dir/ants/*.jpg，标签就是文件夹名
        self.root_dir = root_dir
        self.class_dir = class_dir
        self.image_path = os.path.join(self.root_dir, self.class_dir)
        self.image_list = sorted(
            name for name in os.listdir(self.image_path)
            if name.lower().endswith(IMG_EXTENSIONS)
        )
        self.label = class_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_item_path).convert('RGB')

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': self.label}
        return sample

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # 用脚本所在位置定位数据集，避免受运行时工作目录影响
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'train')
    ants_dataset = MyData(root_dir, 'ants', transform)
    bees_dataset = MyData(root_dir, 'bees', transform)
    train_dataset = ants_dataset + bees_dataset

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    writer.add_image('error', train_dataset[119]['img'])
    writer.close()
    # for i, j in enumerate(dataloader):
    #     # imgs, labels = j
    #     print(type(j))
    #     print(i, j['img'].shape)
    #     # writer.add_image("train_data_b2", make_grid(j['img']), i)
    #
    # writer.close()



