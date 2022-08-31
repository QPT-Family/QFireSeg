import os
import numpy as np
from paddle.vision import transforms as T
import cv2
import paddle
from models import *
from datasets import *
from Metric import *
from utils import *


image_size = (608, 608)
batch_size = 2
num_classes = 2
train_transform = T.Compose([
    # T.RandomHorizontalFlip(0.5),
    # T.RandomVerticalFlip(0.5),
    # T.RandomRotation(45),
    T.ColorJitter(0.4, 0.4, 0.4, 0.4),
    T.Transpose((2, 0, 1)),
    T.Normalize(mean=0., std=255.)
])
eval_transform = T.Compose([
    T.Transpose((2, 0, 1)),
    T.Normalize(mean=0., std=255.)
])
test_transform = T.Compose([
    T.Transpose((2, 0, 1)),
    T.Normalize(mean=0., std=255.)
])

# 数据集
train_dataset = MyDataset(mode='train', transform=train_transform, img_size=image_size)
val_dataset = MyDataset(mode='val', transform=eval_transform, img_size=image_size)


# dataloader
train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=batch_size, places=paddle.CUDAPlace(0),
                                    shuffle=True, drop_last=True, num_workers=4)
val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=batch_size, places=paddle.CUDAPlace(0),
                                  shuffle=True, drop_last=True, num_workers=4)


# 实例化，网络三选一，默认PSPnet
model = PSPnet(num_classes)  # PSPNet
# model = UNet(num_classes)  # U-Net
# model = Deeplabv3(num_classes) #Deeplabv3
# 优化器
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
# 损失函数
loss = paddle.nn.CrossEntropyLoss(axis=1)
# 分割效果指标
miou = IOUMetric(num_classes=num_classes)
model_name = 'PSPnet'
# 模型加载
load_model(model=model, model_name=model_name)
epochs = 50
train_loss_list = []
train_miou_list = []
val_loss_list = []
val_miou_list = []
print('Start Training...')

for epoch in range(1, epochs + 1):
    print('Epoch/Epochs:{}/{}'.format(epoch, epochs))

    print('Train...')
    train_loss = 0
    train_miou = 0
    model.train()
    for batch_id, (img, label) in enumerate(train_loader):
        optimizer.clear_grad()
        pred = model(img)
        step_loss = loss(pred, label)
        train_loss += step_loss.numpy()[0]
        # 计算miou, pred: num_loss * NCHW -> NHW
        mask = np.argmax(pred.numpy(), axis=1)
        step_miou = 0
        for i in range(mask.shape[0]):
            # print(mask[i].shape, label.shape)
            step_miou += miou.evaluate(mask[i], label.numpy()[i])
        step_miou /= mask.shape[0]
        train_miou += step_miou
        step_loss.backward()
        optimizer.step()
        # 打印信息
        if (batch_id + 1) % 50 == 0:
            print('Epoch/Epochs:{}/{} Batch/Batchs:{}/{} Step Loss:{} Step Miou:{}'.format(epoch, epochs, batch_id + 1,
                                                                                           len(train_loader), \
                                                                                           step_loss.numpy(),
                                                                                           step_miou))

    print('Train Loss:{} Train Miou:{}'.format(train_loss / len(train_loader), train_miou / len(train_loader)))
    train_loss_list.append(train_loss / len(train_loader))
    train_miou_list.append(train_miou / len(train_loader))

    if epoch % 5 == 0:
        print('Val...')
        val_loss = 0
        val_miou = 0
        model.eval()
        for batch_id, (img, label) in enumerate(val_loader):
            pred = model(img)
            step_loss = loss(pred, label)
            val_loss += step_loss.numpy()[0]
            # 计算miou, pred: num_loss * NCHW -> NHW
            mask = np.argmax(pred.numpy(), axis=1)
            step_miou = 0
            for i in range(mask.shape[0]):
                # print(mask[i].shape, label.shape)
                step_miou += miou.evaluate(mask[i], label.numpy()[i])
            step_miou /= mask.shape[0]
            val_miou += step_miou

        print('Val Loss:{} Val Miou:{}'.format(val_loss / len(val_loader), val_miou / len(val_loader)))
        val_loss_list.append(val_loss / len(val_loader))
        val_miou_list.append(val_miou / len(val_loader))

        save_model(model, model_name + str(epoch))

print('Train Over...')



