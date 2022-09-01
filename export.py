import paddle
from paddle.static import InputSpec
from paddle.jit import to_static
from models import *
from utils import load_model

if __name__ == '__main__':
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

