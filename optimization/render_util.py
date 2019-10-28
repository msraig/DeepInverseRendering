#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from ops import deprocess, tensor_dot, tensor_norm, numpy_dot, numpy_norm
import math
import numpy as np
import random
import os


#diffuse, specular, roughness, normal,
def render(inputs, l, v, roughness_factor=0.0):
    INV_PI = 1.0 / math.pi
    EPS = 1e-12
    def GGX(NoH, roughness):
        with tf.name_scope("ggx"):
            alpha = roughness  * roughness
            tmp = alpha / tf.maximum(1e-8,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
            return tmp * tmp * INV_PI

    def SmithG(NoV, NoL, roughness):
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k ) + k)

        with tf.name_scope("smith_g"):
            k = tf.maximum(1e-8, roughness * roughness * 0.5)
            return _G1(NoL,k) * _G1(NoV, k)

    def Fresnel(F0, VoH):
        with tf.name_scope("fresnel"):
            coeff = VoH * (-5.55473 * VoH - 6.98316)
            return F0 + (1.0 - F0) * tf.pow(2.0, coeff)


    with tf.name_scope("render_disney"):
        normal, diffuse, roughness, specular = tf.split(inputs, 4, axis = -1)

        v = tf.identity(v, name= 'v')
        l = tf.identity(l, name = 'l')
        n = tf.identity(normal, name='n')
        s = tf.identity(specular, name='s')
        d = tf.identity(diffuse, name='d')
        r = tf.identity(roughness, name='r')

        r = deprocess(r)
        d = deprocess(d)
        s = deprocess(s)


        n = tensor_norm(n)
        h = tensor_norm((l+v) * 0.5 )
        h = tf.identity(h, name='h')

        NoH = tensor_dot(n,h)
        NoV = tensor_dot(n,v)
        NoL = tensor_dot(n,l)
        VoH = tensor_dot(v,h)

        NoH = tf.maximum(NoH, 1e-8)
        NoV = tf.maximum(NoV, 1e-8)
        NoL = tf.maximum(NoL, 1e-8)
        VoH = tf.maximum(VoH, 1e-8)

        NoH = tf.identity(NoH)
        NoV = tf.identity(NoV)
        NoL = tf.identity(NoL)
        VoH = tf.identity(VoH)

        f_d = d * INV_PI

        D = GGX(NoH,r)
        G = SmithG(NoV, NoL, r)
        F = Fresnel(s, VoH)
        f_s = D * G * F / (4.0 * NoL * NoV + EPS)

        res =  (f_d + f_s) * NoL * math.pi

        return res
        
#diffuse, specular, roughness, normal,
def render_np(inputs, l, v, roughness_factor=0.0):
    INV_PI = 1.0 / math.pi
    EPS = 1e-12
    def GGX(NoH, roughness):
        alpha = roughness  * roughness
        tmp = alpha / np.maximum(1e-8,  (NoH * NoH * (alpha * alpha - 1.0) + 1.0  ) )
        return tmp * tmp * INV_PI

    def SmithG(NoV, NoL, roughness):
        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k ) + k)

        k = np.maximum(1e-8, roughness * roughness * 0.5)
        return _G1(NoL,k) * _G1(NoV, k)

    def Fresnel(F0, VoH):
        coeff = VoH * (-5.55473 * VoH - 6.98316)
        return F0 + (1.0 - F0) * np.power(2.0, coeff)

    n,d,r,s = np.split(inputs, 4, axis = -1)

    r = deprocess(r)
    d = deprocess(d)
    s = deprocess(s)

    n = numpy_norm(n)
    h = numpy_norm((l+v) * 0.5 )

    NoH = numpy_dot(n,h)
    NoV = numpy_dot(n,v)
    NoL = numpy_dot(n,l)
    VoH = numpy_dot(v,h)

    NoH = np.maximum(NoH, 1e-8)
    NoV = np.maximum(NoV, 1e-8)
    NoL = np.maximum(NoL, 1e-8)
    VoH = np.maximum(VoH, 1e-8)

    f_d = d * INV_PI

    D = GGX(NoH,r)
    G = SmithG(NoV, NoL, r)
    F = Fresnel(s, VoH)
    f_s = D * G * F / (4.0 * NoL * NoV + EPS)

    res =  (f_d + f_s) * NoL * math.pi

    return res



def log_view_light_dirs(light_camera_pos, N, Ns, Nd, output_folder,d_name='view_light.txt'):
    def _join(lst):
        if isinstance(lst, list):
            return ",".join([str(i) for i in lst])
        elif isinstance(lst, np.ndarray):
            return ",".join([str(i) for i in lst.tolist()])

    # inira
    name = os.path.join(output_folder, d_name)
    if N == -1:
        with open(name, 'w+') as f:
            f.write("inira Ns:%d, Nd%d\n" %(Ns, Nd))
            for item in light_camera_pos:
                light_pos, camera_pos = item

                f.write(_join(light_pos) + "\t" + _join(camera_pos) + '\n')

    # predefined views:
    else:
        with open(name,'w+') as f:
            f.write("predefined N: %d\n" % N)
            for item in light_camera_pos:
                light_pos, camera_pos = item
                f.write(_join(light_pos) + "\t" + _join(camera_pos) + '\n')


def get_wlvs_np(scale_size, total_num = 10):
    def generate(camera_pos_world):
        light_pos_world = camera_pos_world
        x_range = np.linspace(-1,1,scale_size)
        y_range = np.linspace(-1,1,scale_size)
        x_mat, y_mat = np.meshgrid(x_range, y_range)
        pos = np.stack([x_mat, -y_mat, np.zeros(x_mat.shape)],axis=-1)

        view_dir_world = numpy_norm(camera_pos_world - pos)
        light_dir_world = numpy_norm(light_pos_world - pos)

        light_dir_world = light_dir_world.astype(np.float32)
        view_dir_world = view_dir_world.astype(np.float32)
        return view_dir_world, light_dir_world

    def random_pos():
        x = random.uniform(-1.2,1.2)
        y = random.uniform(-1.2,1.2)
        #z = random.uniform(2.0, 4.0)
        z = 2.146
        return [x,y,z]

    def record(camera_pos):
        Wlvs.append(generate(camera_pos))
        camera_light_pos.append([camera_pos, camera_pos])

    Wlvs = []
    camera_light_pos = []

    record([0,0,2.146])

    current_count = len(Wlvs)

    if total_num > current_count:
        for i in range(total_num - current_count):
            pos = random_pos()
            record(pos)

    return Wlvs[:total_num], camera_light_pos[:total_num]


def recover_wlvs_np(filename, scale_size):
    def generate(camera_pos_world):
        light_pos_world = camera_pos_world
        x_range = np.linspace(-1,1,scale_size)
        y_range = np.linspace(-1,1,scale_size)
        x_mat, y_mat = np.meshgrid(x_range, y_range)
        pos = np.stack([x_mat, -y_mat, np.zeros(x_mat.shape)],axis=-1)

        view_dir_world = numpy_norm(camera_pos_world - pos)
        light_dir_world = numpy_norm(light_pos_world - pos)

        light_dir_world = light_dir_world.astype(np.float32)
        view_dir_world = view_dir_world.astype(np.float32)
        return view_dir_world, light_dir_world

    def parse_view_light(name):
        with open(name,'r') as f:
            lines = f.readlines()
            wlvs = []
            for line in lines[1:]:
                line = line[:-1]

                l,v = line.split()
                camera_pos = [float(i) for i in v.split(',')]
                light_pos = [float(i) for i in l.split(',')]

                item = (light_pos, camera_pos)
                wlvs.append(item)
            return wlvs
    def record(camera_pos):
        Wlvs.append(generate(camera_pos))
        camera_light_pos.append([camera_pos, camera_pos])

    lv_pos = parse_view_light(filename)

    Wlvs = []
    camera_light_pos = []
    for lv in lv_pos:
        light_pos, camera_pos = lv
        record(camera_pos)
    return Wlvs, camera_light_pos
