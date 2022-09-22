import argparse
from paddle.inference import Config
from paddle.inference import create_predictor
import cv2
import numpy as np
from utils import visualize


def preprocess(img):
    mean = [0., 0., 0.]
    std = [255., 255., 255.]

    # bgr-> rgb && hwc->chw
    img = img.astype('float32').transpose((2, 0, 1))
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img[np.newaxis, :]


def postprocess(results):
    result = results[0].squeeze(0)
    pred = result.argmax(axis=0)
    return pred


def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return postprocess(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./inference_model/PSPnet.pdmodel",
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./inference_model/PSPnet.pdiparams",
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    image_name = './test/6.png'
    img = cv2.imread(image_name)
    image_size = (608, 608)
    img = cv2.resize(img, image_size)
    pre_img = preprocess(img)
    result = run(pred, [pre_img])
    visualize(image_name, img, result, is_contrast=True, save_dir='./result')
