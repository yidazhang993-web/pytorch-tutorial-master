from PIL import Image

# 把这里的路径换成你电脑上图片的路径
img_path = r"hymenoptera_data\train\ants\0013035.jpg"

# 为了能直接看到图片，你可以在下面加一句打开图片的代码
img = Image.open(img_path)
img.show() 