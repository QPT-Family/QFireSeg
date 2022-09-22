
import numpy as np
import cv2
import paddle
import paddle.nn.functional as F


def inference(model,
              im):
    """
    Inference for image.
    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Image shape informating changed process. Default: None.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    return model(im)
