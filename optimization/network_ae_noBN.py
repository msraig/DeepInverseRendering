#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

code_shape = (1, 8, 8, 512)
encoder_var_names = ['encoder_1', 'encoder_2', 'encoder_3', 'encoder_4', 'encoder_5']
decoder_var_names = ['decoder_1', 'decoder_2', 'decoder_3', 'decoder_4', 'decoder_5']

def deconv2d(batch_input, out_channels, name="deconv"):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2),
            padding="same", kernel_initializer=initializer)

def conv2d(batch_input, out_channels, stride, name="conv"):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2),
            padding="same", kernel_initializer=initializer)
            
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def encoder(inputs):
    with tf.variable_scope("Encoder"):
        layers = []
        ngf = 64
        #encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv2d(inputs, ngf, stride=2)
            layers.append(output)

        layer_specs = [
            ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ]

        for encoder_layer, out_channels in enumerate(layer_specs):
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]

                convolved = conv2d(rectified, out_channels, stride=2)
                outputs = convolved

                layers.append(outputs)
        return layers[-1]


# inputs: [batch, 8, 8, 512]
def decoder(inputs, output_channels):
    with tf.variable_scope("Decoder"):
        layers =[inputs]

        ngf = 64
        layer_specs = [
            ngf * 8,  # decoder_5: [batch, 8, 8, ngf * 8] => [batch, 16, 16, ngf * 8]
            ngf * 4,  # decoder_4: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 4]
            ngf * 2,  # decoder_3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
            ngf * 1   # decoder_2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        ]

        for decoder_layer, out_channels in enumerate(layer_specs):
            with tf.variable_scope("decoder_%d" % (len(layer_specs) + 1 - decoder_layer)):

                rectified = lrelu(layers[-1], 0.2)

                output = deconv2d(rectified, out_channels)
                layers.append(output)


        with tf.variable_scope("decoder_1"):
            rectified = lrelu(layers[-1], 0.2)
            output = deconv2d(rectified, output_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]

