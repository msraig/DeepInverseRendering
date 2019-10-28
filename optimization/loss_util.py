#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def L1Loss(inputs, targets, weight = 1.0):
    diff = tf.abs(inputs - targets)
    return tf.reduce_mean(diff) * weight

def L1LogLoss(inputs, targets, weight = 1.0):
    return tf.reduce_mean(tf.abs(tf.log(inputs + 0.01) - tf.log(targets + 0.01))) * weight

def L2Loss(inputs, targets,  weight = 1.0):
    return tf.reduce_mean(tf.squared_difference(inputs, targets)) * weight
