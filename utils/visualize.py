from utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re


def visualize(im_path, result, save_dir=None, is_contrast=True):
    """
    Convert predict result to color image, and save added image.

    Args:
        im_path(str): The path of origin image.
        result (np.ndarray): The predict result of image.
        is_contrast (bool): Whether to compare the original image and mask. Default: True.
        save_dir (str): The directory for saving visual image. Default: None.
    Returns:
        None
    """
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(im_path)
    name = data[0]
    name = name + '.jpg'
    if is_contrast:
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_list = [image, result]
    else:
        img_list = result
    plt.figure(figsize=(15, 15))
    for i in range(len(img_list)):
        plt.subplot(1, len(img_list), i + 1)
        plt.imshow(img_list[i])
        if name:
            plt.title(name)
    plt.savefig(os.path.join(save_dir, name))
    plt.legend()
