from utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re


def visualize(image_name=None, image=None, result=None, is_contrast=True, save_dir=None):
    """
    Convert predict result to color image, and save added image.

    Args:
        image_name(str): The path of origin image. Default: None.
        image (np.ndarray): The image data. Default: None.
        result (np.ndarray): The predict result of image. Default: None.
        is_contrast (bool): Whether to compare the original image and mask. Default: None.
        save_dir (str): The directory for saving visual image. Default: None.
    Returns:
        None
    """
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(image_name)
    name = data[0]
    name = name + '.jpg'
    if is_contrast:
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
