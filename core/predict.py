import cv2
from utils import *
from core import infer


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def preprocess(im_path, transforms):
    img = cv2.imread(im_path)
    image_size = (608, 608)
    img = cv2.resize(img, image_size)
    img = paddle.to_tensor([transforms(img)])  # NCHW
    return img

def predict(model,
            model_name,
            transforms,
            image_list,
            image_dir=None,
            is_contrast=True,
            save_dir='output'):
    load_model(model=model, model_name=model_name)
    model.eval()
    img_lists = image_list

    with paddle.no_grad():
        for i, im_path in enumerate(img_lists):
            im = preprocess(im_path, transforms)
            pred = infer.inference(
                model,
                im)

            pred = pred.squeeze(0)
            pred = pred.argmax(axis=0)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            visualize(im_path, pred, is_contrast=is_contrast, save_dir='./result')
            # added_image_path = os.path.join(added_saved_dir, im_file)
            # mkdir(added_image_path)
            # cv2.imwrite(added_image_path, added_image)
