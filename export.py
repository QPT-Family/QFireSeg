import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.static import InputSpec
from paddle.jit import to_static
from paddle.vision.transforms import ToTensor
from models import *
from utils import load_model

if __name__ == '__main__':
    # paddle version

    # build network
    num_classes = 2
    model = PSPnet(num_classes=num_classes)
    # load format model
    model_name = 'PSPNet50'
    load_model(model=model, model_name=model_name)

    # save inferencing format model
    net = to_static(model,
                    input_spec=[InputSpec(shape=[-1, 3, 608, 608], name='x')])
    paddle.jit.save(net, 'inference_model/PSPnet')
