import paddle
from core.predict import predict
from models import PSPnet

num_classes = 2
model = PSPnet(num_classes=num_classes)
# 模型加载
model_name = 'PSPNet180'

transforms = paddle.vision.transforms.Compose([
    paddle.vision.transforms.Transpose((2, 0, 1)),  # HWC -> CHW
    paddle.vision.transforms.Normalize(mean=0., std=255.)
])

image_list = ['./test/1.png',
              './test/2.png',
              './test/3.png',
              './test/4.png',
              './test/5.png',
              './test/6.png']

predict(model,
        model_name,
        transforms,
        image_list,
        image_dir=None,
        is_contrast=True,
        save_dir='output'
        )
