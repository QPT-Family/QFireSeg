import os
import numpy as np
from tqdm import tqdm

data = []
train_data = []
test_data = []
img_path = 'E:/public_datasets/Fire-Segmentation/JPEGImages'
mask_path = 'E:/public_datasets/Fire-Segmentation/Annotations'


for item in os.listdir(img_path):
    data.append([
        os.path.join(img_path, item),  # 原图路径
        os.path.join(mask_path, item)  # 分割图路径
    ])


# 打乱数据集
np.random.shuffle(data)
print(len(data) // 10)
# 划分训练集和验证集
train_data = data[len(data) // 10:]
val_data = data[:len(data) // 10]


# 将路径写入.txt文件
def write_path(data, path):
    with open(path, 'w') as f:
        for item in tqdm(data):
            f.write(item[0] + ' ' + item[1] + '\n')  # 原图路径 分割图路径


write_path(train_data, '../train.txt')
write_path(val_data, '../val.txt')
