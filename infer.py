import paddle
from datasets import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from models import *
import cv2

image_size = (608, 608)


# 获得预测结果
def get_mask(img):
    t = paddle.vision.transforms.Compose([
        paddle.vision.transforms.Transpose((2, 0, 1)),  # HWC -> CHW
        paddle.vision.transforms.Normalize(mean=0., std=255.)
    ])
    img1 = paddle.to_tensor([t(img)])  # NCHW
    pred = np.array(model(img1)).squeeze(0)  # CHW
    pred = pred.argmax(axis=0)
    return pred


# 画图
def draw(img_list, name=None):
    plt.figure(figsize=(15, 15))
    for i in range(len(img_list)):
        plt.subplot(1, len(img_list), i + 1)
        plt.imshow(img_list[i])
        if name:
            plt.title(name[i])
    plt.legend()
    plt.show()


# 读取模型
num_classes = 2
model_name = 'PSPNet40'
model = PSPnet(num_classes=num_classes)
# 模型加载
load_model(model=model, model_name=model_name)
model.eval()
# 预测
tsy_img = cv2.imread('./test/6.png')
tsy_img = cv2.resize(tsy_img, image_size)
tsy_mask = get_mask(tsy_img)
draw(
    [cv2.cvtColor(tsy_img, cv2.COLOR_BGR2RGB), tsy_mask]
)
