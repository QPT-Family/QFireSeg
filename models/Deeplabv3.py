"""
paddlepaddle-gpu==2.2.1
time:2021.07.20 9:00
author:CP
Deeplabv3
"""
import paddle
import paddle.nn as nn


class ASPPPooling(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(ASPPPooling, self).__init__()
        self.adaptive_pool = nn.AdaptiveMaxPool2D(output_size=3)
        self.features = nn.Sequential(
            nn.Conv2D(num_channels, num_filters, 1),
            nn.BatchNorm(num_filters, act="relu")
        )

    def forward(self, inputs):
        n1, c1, h1, w1 = inputs.shape
        x = self.adaptive_pool(inputs)
        x = self.features(x)
        x = nn.functional.interpolate(x, (h1, w1), align_corners=False)
        return x


class ASPPConv(nn.Layer):
    def __init__(self, num_channels, num_filters, dilations):
        super(ASPPConv, self).__init__()
        self.asppconv = nn.Sequential(
            nn.Conv2D(num_channels, num_filters, 3, padding=dilations, dilation=dilations),
            nn.BatchNorm(num_filters, act="relu")
        )

    def forward(self, inputs):
        x = self.asppconv(inputs)
        return x


# ASPP模块最大的特点是使用了空洞卷积来增大感受野
class ASPPModule(nn.Layer):
    def __init__(self, num_channels, num_filters, rates):
        super(ASPPModule, self).__init__()
        self.features = nn.LayerList()
        # 层一
        self.features.append(nn.Sequential(
            nn.Conv2D(num_channels, num_filters, 1),
            nn.BatchNorm(num_filters, act="relu")
        )
        )
        # 层二
        for r in rates:
            self.features.append(ASPPConv(num_channels, num_filters, r))
        # 层三
        self.features.append(ASPPPooling(num_channels, num_filters))
        # 层四
        self.project = nn.Sequential(
            nn.Conv2D(num_filters * (2 + len(rates)), num_filters, 1),  # TODO
            nn.BatchNorm(num_filters, act="relu")
        )

    def forward(self, inputs):
        out = []
        for op in self.features:
            out.append(op(inputs))
        x = paddle.concat(x=out, axis=1)
        x = self.project(x)
        return x


class DeeplabHead(nn.Layer):
    def __init__(self, num_channels, num_classes):
        super(DeeplabHead, self).__init__()
        self.head = nn.Sequential(
            ASPPModule(num_channels, 256, [12, 24, 36]),
            nn.Conv2D(256, 256, 3, padding=1),
            nn.BatchNorm(256, act="relu"),
            nn.Conv2D(256, num_classes, 1)
        )

    def forward(self, inputs):
        x = self.head(inputs)
        return x


from paddle.vision.models import resnet50
from paddle.vision.models.resnet import BottleneckBlock


class Deeplabv3(nn.Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(Deeplabv3, self).__init__()
        # resnet50 3->2048
        # resnet50 四层layers = [3 4 6 3]
        # 调用resnet.py模块，空洞卷积[2 4 8 16]
        res = resnet50()
        res.inplanes = 64  # 初始化输入层
        self.layer0 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        self.layer1 = res._make_layer(BottleneckBlock, 64, 3)
        self.layer2 = res._make_layer(BottleneckBlock, 128, 4)
        self.layer3 = res._make_layer(BottleneckBlock, 256, 6)
        self.layer4 = res._make_layer(BottleneckBlock, 512, 3, stride=2, dilate=2)  # dilation=2
        self.layer5 = res._make_layer(BottleneckBlock, 512, 3, stride=2, dilate=4)  # dilation=4
        self.layer6 = res._make_layer(BottleneckBlock, 512, 3, stride=2, dilate=8)  # dilation=8
        self.layer7 = res._make_layer(BottleneckBlock, 512, 3, stride=2, dilate=16)  # dilation=16
        feature_dim = 2048  # 输出层通道2048
        self.deeplabhead = DeeplabHead(feature_dim, num_classes)

    def forward(self, inputs):
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.deeplabhead(x)  # ASPP模块进行分类
        # 恢复原图尺寸
        x = paddle.nn.functional.interpolate(x=x, size=inputs.shape[2::], mode='bilinear', align_corners=True)
        return x


# paddle.summary(Deeplabv3(), (1, 3, 600, 600))