import os
import json
import numpy as np
import torch
from PIL import Image
from lib.utils import get_mask
from lib.augment import augment_img


def get_dataset(path, is_test):
    """Returns images with annotations (or only images)

    If the data is for training or validation, returns annotations and images.
    Otherwise, returns only images
    :param path: str
        path to dataset
    :param is_test: bool
        flag which defines whether data is for test or not
    :return:
    """
    if not is_test:
        images = os.listdir(f"{path}/images")
        annotations = json.load(open(f"{path}/coco_annotations.json", "r"))
        return images, annotations
    else:
        images = os.listdir(f"{path}")
        return images


def get_batch(indices, annotations, path, mode, augment=0):
    """Loads batch of data

    This method loads images located in 'path' and masks for these images,
    converts it to pytorch tensors and normalizes so that every pixel value lies in [0; 1].

    :param indices: list
        list that contains id's of images which will be loaded
    :param annotations: dict
        annotations to the data
    :param path: str
        path to the data folder
    :param mode: str
        can be only 'train', 'val', or 'test' -- according to the data
    :param augment: int
        Number describing how many new images should be generated from one image
    :return: torch.tensor, torch.tensor
        Image and mask tensors. The shape of X is (N, C, W, H) , the shape of y is (N, W, H)
    where N is a batch size, C is a number of channels, W is a width, H is a height of an image
    """

    if (mode != 'train') & (mode != 'val'):
        raise ValueError('Wrong \'mode\' parameter. Should be one of: \'test\', \'train\', \'val\'')

    X = []
    y = []
    for index in indices:
        img_id = index

        if mode == 'train':
            img_path = f"{path}/images/{img_id:08}.jpg"
        elif mode == 'val':
            img_path = f"{path}/images/{img_id:08}.png"
        else:
            img_path = f"{path}/{img_id:04}.JPG"

        img = np.array(Image.open(img_path))
        mask = get_mask(img_id, annotations)

        X.append(img)
        y.append(mask)

        for i in range(augment):
            aug_img, aug_mask = augment_img(img, mask)
            X.append(aug_img)
            y.append(aug_mask)

    X = torch.tensor(X).float().permute(0, 3, 1, 2)
    y = torch.tensor(y).float()

    X /= 255
    y /= 255

    return X, y