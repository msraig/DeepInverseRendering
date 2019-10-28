#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import glob
from ops import *
import tensorflow as tf

"""
load svbrdf
file extension:
    .npyimage.npy: float32 numpy array
    .png: uint8 image.
"""
def load_svbrdf(name, scale_size, data_format):
    if data_format == 'npy':
        tmp_path, tmp_name = os.path.split(name)
        tmp_name = os.path.splitext(tmp_name)[0]
        new_name = os.path.join(tmp_path, '%s.npyimage.npy'%tmp_name)
        # if not exist, use .png instead
        if not os.path.exists(new_name):
            print("[Warning] npy image %s not exist, load png instead. %s" % (new_name,name))
            img = cv2.imread(name)
            if img is None:
                print(name)
            img = img.astype(np.float32) / 255.0
            img = img[:,:,::-1]
        else:
            img = np.load(new_name)
    else:
        img = cv2.imread(name)
        img = img.astype(np.float32) / 255.0
        img = img[:,:,::-1]


    h,w, _ = img.shape
    nb_imgs = w // h
    image_width = w // nb_imgs

    images = []
    for i in range(4):
        tmp = img[:, i * image_width: (i+1) * image_width, :]
        if i == 1:
            tmp = tmp ** 2.2        # diffuse: srgb => linear
        images.append(tmp)  

    new_images = []
    for img in images:
        img = img * 2.0 - 1.0
        if h != scale_size:
            if h > scale_size:
                print("[info] AREA resize image from %d to %d" % (h, scale_size))
                img = cv2.resize(img, (scale_size, scale_size), interpolation=cv2.INTER_AREA)
            else:
                print("[info] LINEAR resize image from %d to %d" % (h, scale_size))
                img = cv2.resize(img, (scale_size, scale_size))
        new_images.append(img)
    images = new_images

    # 4 x [size, size, 3] => [size, size, 12]
    res =  np.concatenate(images, axis = -1)

    # [1,size,size,12]
    res = res[np.newaxis,...]
    return res


def load_input_img(name, data_format):
    if data_format == 'npy':
        tmp_path, tmp_name = os.path.split(name)
        tmp_name = os.path.splitext(tmp_name)[0]
        new_name = os.path.join(tmp_path, '%s.npyimage.npy'%tmp_name)
        if os.path.exists(new_name):
            img = np.load(new_name)
            img = img[...,::-1]
        else:
            print("[Warning] npy image %s not exist, load png instead. %s" % (new_name,name))
            img = cv2.imread(name)
            img = img.astype(np.float32) / 255.0
            img = img[...,::-1]
    else:
        img = cv2.imread(name)
        img = img.astype(np.float32) / 255.0
        img = img[...,::-1]
    return img

def load_files(path):
    files =  glob.glob(os.path.join(path, "*.png"))
    files.extend(glob.glob(os.path.join(path, "*.jpg")))
    return files

