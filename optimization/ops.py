#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import tensorflow as tf


def tensor_norm(tensor):
    t = tf.identity(tensor)
    Length = tf.sqrt(tf.reduce_sum(tf.square(t), axis = -1, keepdims =True))
    return tf.div(t, Length + 1e-12)

def tensor_dot(a,b):
    return tf.reduce_sum(a * b, axis = -1) [..., tf.newaxis]

def numpy_norm(arr):
    length = np.sqrt(np.sum(arr * arr, axis = -1, keepdims=True))
    return arr / (length + 1e-12)
    
def numpy_dot(a,b):
    return np.sum(a * b, axis=-1)[...,np.newaxis]

# image utils.
def preprocess(img):
    # [0,1] => [-1,1]
    return img * 2.0 - 1.0

def deprocess(img):
    # [-1,1] => [0,1]
    return (img + 1.0) / 2.0

def toLDR(img):
    return img ** (1.0/2.2)

def log_tensor_norm(tensor):
    return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))

def reconstruct_output(outputs, order='DSRN'):
    with tf.variable_scope("reconstruct_output"):
        if order == 'NDRS':
            partial_normal = outputs[:,:,:, 0:2]
            diffuse = outputs[:,:,:, 2:5]
            roughness = outputs[:,:,:, 5:6]
            specular = outputs[:,:,:, 6:9]
        elif order == 'DSRN':
            partial_normal = outputs[:,:,:, 7:9]
            diffuse = outputs[:,:,:, 0:3]
            roughness = outputs[:,:,:, 6:7]
            specular = outputs[:,:,:, 3:6]

        normal_shape = tf.shape(partial_normal)
        normal_z = tf.ones([normal_shape[0],normal_shape[1],normal_shape[2],1], tf.float32)
        normal = tensor_norm(tf.concat([partial_normal, normal_z], axis= -1))

        outputs_final = tf.concat([normal, diffuse, roughness, roughness, roughness, specular], axis=-1)
        return outputs_final
