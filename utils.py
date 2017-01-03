# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.
import os

import numpy as np
import scipy
import scipy.misc
from os.path import exists

def load_image(image_path, img_size=None):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = scipy.misc.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_size is not None):
        img = scipy.misc.imresize(img, img_size)

    img = img.astype("float32")
    return img


def save_image(img, path):
    scipy.misc.imsave(path, np.clip(img, 0, 255).astype(np.uint8))

def get_files(img_dir):
    files = list_files(img_dir)
    return map(lambda x: os.path.join(img_dir,x), files)

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files
