"""
paddlepaddle-gpu==2.2.1
time:2021.07.16 9:00
author:CP
backbone：PSPnet
"""
import paddle
import paddle.nn as nn


class PSPModule(nn.Layer):
    """
    num_channels：输入通道数为C
    num_filters ：输出通道数为C/4
    bin_size_list=[1,2,3,6]
    get1:
        nn.LayerList创建一个空列表的层
        .append拼接“层”的列表
    get2:
        paddle.nn.AdaptiveMaxPool2D输出固定尺寸的image_size[H,W]
        paddle.nn.functional.interpolate卷积操作后，还原图片尺寸大小
        paddle.concat [H,W]同尺寸的图片，合并通道[C]
    """

    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        num_filters = num_channels // len(bin_size_list)  # C/4
        self.features = nn.LayerList()  # 一个层的空列表
        for i in range(len(bin_size_list)):
            self.features.append(
                paddle.nn.Sequential(
                    paddle.nn.AdaptiveMaxPool2D(output_size=bin_size_list[i]),
                    paddle.nn.Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=1),
                    paddle.nn.BatchNorm2D(num_features=num_filters)
                )
            )

    def forward(self, inputs):
        out = [inputs]  # list
        for idx, layerlist in enumerate(self.features):
            x = layerlist(inputs)
            x = paddle.nn.functional.interpolate(x=x, size=inputs.shape[2::], mode='bilinear', align_corners=True)
            out.append(x)
        out = paddle.concat(x=out, axis=1)  # NCHW
        return out


from paddle.vision.models import resnet50


class PSPnet(nn.Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPnet, self).__init__()
        """
        https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/vision/models/resnet.py
        重复利用resnet50网络模型：
            1.初始化函数关键词——backbone
            2.神经网络模型实例化
            3.源代码查找层的变量名
        """
        # resnet50 3->2048
        res = resnet50()
        self.layer0 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4  # 输出通道为2048

        # pspmodule 2048->4096
        num_channels = 2048
        self.pspmodule = PSPModule(num_channels, [1, 2, 3, 6])
        num_channels *= 2

        # cls 4096->num_classes
        self.classifier = nn.Sequential(
            nn.Conv2D(num_channels, 512, 3, 1, 1),
            nn.BatchNorm(512, act='relu'),
            nn.Dropout(),
            nn.Conv2D(512, num_classes, 1))
        # aux:1024->256->num_classes
        # 单独分离出一层来计算函数损失

    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pspmodule(x)
        x = self.classifier(x)
        x = paddle.nn.functional.interpolate(x=x, size=inputs.shape[2::], mode='bilinear', align_corners=True)
        return x


# paddle.summary(PSPnet(), (1, 3, 600, 600))