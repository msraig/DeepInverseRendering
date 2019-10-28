#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import os
from ops import deprocess, toLDR

def get_var_list (name_list):
    layer_vars = []
    for name in name_list:
        layer_vars.append([var for var in tf.all_variables() if name in var.name])
    layer_vars = [var for layer in layer_vars for var in layer]
    return layer_vars

def transform(img):
    img = deprocess(img)
    n,d,r,s = np.split(img, 4, axis = 1)
    d = toLDR(d)
    img = np.concatenate([n,d,r,s], axis = 1)
    return img

def save_imgs(imgs, name_prefix, output_folder):
    count = 0
    for img in imgs:
        name = name_prefix + ("_%d.png" % count)
        cv2.imwrite(os.path.join(output_folder,name), img[:,:,::-1] * 255)
        count += 1
        