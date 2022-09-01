import numpy as np
from models import *
import cv2
from utils import *


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


# 读取模型
num_classes = 2
model = PSPnet(num_classes=num_classes)
# 模型加载
model_name = 'PSPNet50'
load_model(model=model, model_name=model_name)
model.eval()
# 预测
image = './test/1.png'
image_size = (608, 608)
tsy_img = cv2.imread(image)
tsy_img = cv2.resize(tsy_img, image_size)
tsy_mask = get_mask(tsy_img)

visualize(image, tsy_img, tsy_mask, is_contrast=True, save_dir='./result')
