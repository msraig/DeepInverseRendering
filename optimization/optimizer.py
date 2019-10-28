#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy
import cv2
from ops import *
from data_util import *
from render_util import *
from loss_util import L1Loss, L1LogLoss
from util import *
import importlib
import collections
import argparse
import time
import math
import os
import numpy as np
import random
import glob

AeOptModel = collections.namedtuple("AeOptModel", "outputs, latent_code, data_loss, loss, grads_and_vars, train, others")
RefineOptModel = collections.namedtuple("RefineOptModel", "outputs, loss, grads_and_vars, train, others")


class Optimizer(object):
    def __init__(self, code_shape, encoder_var_name, decoder_var_name, encoder_net, decoder_net, args):
        self.code_shape = code_shape
       
        self.encoder_var_name = encoder_var_name
        self.decoder_var_name = decoder_var_name

        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        self.scale_size = args.scale_size

        self.N = args.N

        self.checkpoint = args.checkpoint
        self.lr = args.lr
        self.beta1 = args.beta1
        self.max_steps = args.max_steps
        self.refine_max_steps = args.refine_max_steps
        self.progress_freq = args.progress_freq
        self.refine_progress_freq = args.refine_progress_freq
        self.save_freq = args.save_freq
        self.refine_save_freq = args.refine_save_freq

        self.data_format = args.data_format

        self.initial_mu = args.initial_mu
        self.initial_std = args.initial_std

        self.data_loss_type = args.data_loss_type

        self.init_method = args.init_method
        self.initDir = args.initDir
        self.input_type = args.input_type

        self.wlv_type = args.wlv_type
        self.dataDir = args.dataDir
        self.LDR = args.LDR

        self.wlvDir = args.wlvDir
        self.exposure_scale = args.exposure_scale
        self.refine_init = args.refine_init
      

        self.svbrdf_ae_opt = None
        self.svbrdf_ae_opt_image_path = None
        self.Wlvs = None
        self.input_samples = None
        self.svbrdf_sample = None
        self.input_as_svbrdf = None
        self.random_wlv = None
       

    def initial_from_svbrdf(self, inpt):
        g1 = tf.Graph()
        
        with g1.as_default() as g:
            inpt = tf.convert_to_tensor(inpt)
            tf_code = self.encoder(inpt)

            encoder_var_list = get_var_list(self.encoder_var_name)
            saver = tf.train.Saver(var_list = encoder_var_list)

            with tf.Session(graph=g) as sess:
                checkpoint = tf.train.latest_checkpoint(self.checkpoint)
                saver.restore(sess, checkpoint)
                code = sess.run(tf_code)
                return code

    def encoder(self, inputs):
        return self.encoder_net(inputs)

    def decoder(self, inputs):
        output =  self.decoder_net(inputs, 9)
        output = reconstruct_output(output, 'DSRN')
        return output

    def optimize_ae_model(self, input_imgs, Wlvs, initial_code):

        x = tf.get_variable(name='x', dtype=tf.float32,initializer = initial_code, trainable=True)

        D = self.decoder(x)
        #alpha = self.encoder(D)
        input_imgs = tf.convert_to_tensor(input_imgs, dtype=tf.float32)

        render_imgs = []
        if self.data_loss_type != 'test':
            for wlv in Wlvs:
                render_imgs.append(render(D, wlv[1], wlv[0],0.0))
        render_imgs = tf.convert_to_tensor(render_imgs, dtype=tf.float32)

        with tf.name_scope("loss"):
            with tf.name_scope("data_loss"):
                if self.data_loss_type != 'test':
                    input_imgs = input_imgs[:,0,:,:,:]#tf.squeeze(input_imgs)
                    render_imgs = render_imgs[:,0,:,:,:]#tf.squeeze(render_imgs)
                    
                    if self.LDR:
                        input_imgs = tf.clip_by_value(input_imgs, 0.0, 1.0)
                        render_imgs = tf.clip_by_value(render_imgs, 0.0, 1.0)

                        input_imgs = input_imgs ** (1.0/2.2)
                        input_imgs = tf.cast(input_imgs * 255, tf.uint8)
                        input_imgs = tf.cast(input_imgs, tf.float32) / 255.0
                        input_imgs = input_imgs ** (2.2)

                if self.data_loss_type == 'l1':
                    data_loss = L1Loss(input_imgs, render_imgs)
                elif self.data_loss_type == 'l1log':
                    data_loss = L1LogLoss(input_imgs, render_imgs)
                elif self.data_loss_type == 'test':
                    data_loss = L1Loss(D, self.svbrdf_sample) / 2.0
                
            loss = data_loss
        # train
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        learning_rate = self.lr

        with tf.name_scope("optimize_train"):
            train_vars = [x]
            optim = tf.train.AdamOptimizer(learning_rate, self.beta1)
            grads_and_vars = optim.compute_gradients(loss, var_list = train_vars)
            train = optim.apply_gradients(grads_and_vars)


        if self.data_loss_type == 'test':
            others = None
        else:
            others = [input_imgs, render_imgs]
        # training_op, outputs, grads_and_vars, loss
        return AeOptModel(
            outputs = D,
            latent_code = x,
            data_loss = data_loss,
            loss = loss,
            grads_and_vars = grads_and_vars,
            train = tf.group(incr_global_step, train),
            others = others
        )
    
    def optimize_refine_model(self, input_imgs, Wlvs, initial_code):
        x = tf.get_variable(name='x', dtype=tf.float32,initializer = initial_code, trainable=True)
        
        n,d,r,s = tf.split(x,4,axis=-1)
        r0 = r[...,0:1]
        new_r = tf.concat([r0,r0,r0], axis=-1)
        
        new_x = tf.concat([n,d,new_r,s], axis=-1)
        D = tf.clip_by_value(new_x, -1.0, 1.0)

        input_imgs = tf.convert_to_tensor(input_imgs, dtype=tf.float32)

        render_imgs = []
        if self.data_loss_type != 'test':
            for wlv in Wlvs:
                render_imgs.append(render(D, wlv[1], wlv[0],0.0))
        render_imgs = tf.convert_to_tensor(render_imgs, dtype=tf.float32)


        with tf.name_scope("loss"):
            with tf.name_scope("data_loss"):
                if self.data_loss_type != 'test':
                    input_imgs = input_imgs[:,0,:,:,:]#tf.squeeze(input_imgs)
                    render_imgs = render_imgs[:,0,:,:,:]#tf.squeeze(render_imgs)

                    if self.LDR:
                        input_imgs = tf.clip_by_value(input_imgs, 0.0, 1.0)
                        render_imgs = tf.clip_by_value(render_imgs, 0.0, 1.0)

                        input_imgs = input_imgs ** (1.0/2.2)
                        input_imgs = tf.cast(input_imgs * 255, tf.uint8)
                        input_imgs = tf.cast(input_imgs, tf.float32) / 255.0
                        input_imgs = input_imgs ** (2.2)

                if self.data_loss_type == 'l1':
                    data_loss = L1Loss(input_imgs, render_imgs)
                elif self.data_loss_type == 'l1log':
                    data_loss = L1LogLoss(input_imgs, render_imgs)
                elif self.data_loss_type == 'test':
                    data_loss = L1Loss(D, self.svbrdf_sample) / 2.0

            loss = data_loss
        
        # train
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)
        learning_rate = self.lr

        with tf.name_scope("optimize_train"):
            train_vars = [x]
            optim = tf.train.AdamOptimizer(learning_rate, self.beta1)
            grads_and_vars = optim.compute_gradients(loss, var_list = train_vars)
            train = optim.apply_gradients(grads_and_vars)


        if self.data_loss_type == 'test':
            others = None
        else:
            others = [input_imgs, render_imgs]
        # training_op, outputs, grads_and_vars, loss
        return RefineOptModel(
            outputs = D,
            loss = loss,
            grads_and_vars = grads_and_vars,
            train = tf.group(incr_global_step, train),
            others = others
        )

    
    def get_wlvs(self, basename, output_folder):
        # Wlvs
        if self.data_loss_type != 'test':
            if self.random_wlv:
                Wlvs, light_camera_pos = get_wlvs_np(self.scale_size, self.N)
            else:
                filename = os.path.join(self.wlvDir, '%s.txt'% basename)
                Wlvs, light_camera_pos = recover_wlvs_np(filename, self.scale_size)
            log_view_light_dirs(light_camera_pos, self.N, -1, -1, output_folder)
        else:
            Wlvs = None
        return Wlvs
    
    def get_initial_code(self, basename):
        # Initial
        if self.init_method == 'rand':
            # random latent code of AE as init.
            code = np.random.normal(size=self.code_shape,loc=self.initial_mu,scale=self.initial_std)
            code = code.astype(np.float32)
            code_initial_value = code
        elif self.init_method == 'code':
            # specific latent code of AE as init.
            code_initial_value = np.load(os.path.join(self.initDir, '%s.npy' % basename))
            code_initial_value = code_initial_value[np.newaxis,...]
        elif self.init_method == 'svbrdf':
            # specific SVBRDF as init.
            initial_sample = load_svbrdf(os.path.join(self.initDir, '%s.png' % basename), self.scale_size, self.data_format)
            code_initial_value = self.initial_from_svbrdf(initial_sample)
        return code_initial_value

    def get_input_images(self, basename, Wlvs):
        if self.input_as_svbrdf:
            svbrdf_name = os.path.join(self.dataDir, '%s.png' % basename)
            svbrdf_sample = load_svbrdf(svbrdf_name, self.scale_size, self.data_format)
            #self.svbrdf_sample = svbrdf_sample

            input_samples = []
            if self.data_loss_type != 'test':
                for wlv in Wlvs:
                    input_samples.append(render_np(svbrdf_sample, wlv[1], wlv[0]) * self.exposure_scale)
            #input_samples = tf.convert_to_tensor(input_samples, dtype=tf.float32)
            input_samples = np.array(input_samples)
            return input_samples, svbrdf_sample
        else:
            input_samples = []

            for idx in range(self.N):
                name = os.path.join(self.dataDir,'%s_%d.png'%(basename, idx))
                img = load_input_img(name, self.data_format)
                print("load %s_%d.png" % (basename, idx))
                img = img ** (2.2)
                img = img[np.newaxis,...]
                input_samples.append(img)
                
            input_samples = np.array(input_samples)
            #input_samples = tf.convert_to_tensor(input_samples, dtype=tf.float32)
            return input_samples, None
    
    def convert_results(self, output):

        convert_output = output
        convert_output = tf.split(convert_output, 4, axis = -1)
        convert_output = tf.concat(convert_output, axis = 1)
        
        if self.input_as_svbrdf:
            convert_input = self.svbrdf_sample[0]
            convert_input = tf.split(convert_input, 4, axis = -1)
            convert_input = tf.concat(convert_input, axis = 1)
            GT_loss = tf.reduce_mean(tf.abs(convert_input - convert_output)) / 2.0

        else:
            convert_input = tf.ones_like(convert_output) * -1
            GT_loss = tf.constant(-1,dtype=tf.float32)

        return convert_input, convert_output, GT_loss

    def logging(self, step, max_steps, start, results, buffer, batch_size = 1):
        train_step = (results["global_step"])
        rate = (step + 1) * batch_size / (time.time() - start)
        remaining = (max_steps - step) * batch_size / rate
        print("> progress  [step %d] total_step %d  image/sec %0.1f  remaining %dm" % (train_step, results["global_step"], rate, remaining / 60))
        print("  data_loss: %f, GT_loss: %f" % (results["data_loss"] ,  results["gt_loss"]))
        buffer.append(
            "step %d data_loss %f,  gt_loss %f\n" %(train_step, results["data_loss"] ,results["gt_loss"]))

    def dump_outputs(self, res, output_folder, first_save, step):
        if self.data_loss_type != 'test':
            render_samples_res = toLDR(res["render_img"])
            input_samples_res = toLDR(res["input_img"])
        outputs = transform(res["outputs"])
        inputs = transform(res["inputs"])

        if first_save == False:
            if self.data_loss_type != 'test':
                save_imgs(input_samples_res, "input", output_folder)
            cv2.imwrite(os.path.join(output_folder, "image.png"), inputs[:,:,::-1] * 255)

        if self.data_loss_type != 'test':
            save_imgs(render_samples_res, "render_%d" % step, output_folder)
            save_imgs(render_samples_res, "render", output_folder)

        cv2.imwrite(os.path.join(output_folder, "output_%d.png" % step), outputs[:,:,::-1] * 255)

    def init(self, input_basename, output_folder):
        if self.input_type == 'svbrdf':
            self.input_as_svbrdf = True
        elif self.input_type == 'image':
            self.input_as_svbrdf = False

        if self.wlv_type == 'load':
            self.random_wlv = False
        elif self.wlv_type == 'random':
            self.random_wlv = True

        self.Wlvs = self.get_wlvs(input_basename, output_folder)
        self.input_samples, self.svbrdf_sample = self.get_input_images(input_basename, self.Wlvs)


    def ae_opt(self, input_basename, output_folder):
        self.init(input_basename, output_folder)

        code_initial_value = self.get_initial_code(input_basename)
        model = self.optimize_ae_model(self.input_samples, self.Wlvs, code_initial_value)
        
        x = model.latent_code

       
        convert_input, convert_output, GT_loss = self.convert_results(model.outputs[0])
        
            
        with tf.name_scope("encode_images"):
            if self.data_loss_type != 'test':
                display_fetches = {
                    "input_img": model.others[0],
                    "render_img": model.others[1],
                    "inputs": convert_input,
                    "outputs": convert_output,
                    "code": x
                }
            else:
                display_fetches = {
                    "inputs" : convert_input,
                    "outputs" : convert_output,
                    "code": x
                }

        var_list = get_var_list(self.encoder_var_name + self.decoder_var_name)
        if len(var_list) != 0:
            saver = tf.train.Saver(var_list=var_list)
        else:
            saver = None
        

        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(logdir=None,init_op = init_op , save_summaries_secs=0, saver=None, summary_writer=None)

        logging_text = []
        first_save = False
        batch_size = 1

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            tf.reset_default_graph()
            
            if self.checkpoint is not None and saver is not None:
                print("loading model from checkpoint " + self.checkpoint)
                checkpoint = tf.train.latest_checkpoint(self.checkpoint)
                saver.restore(sess, checkpoint)

           
            max_steps = self.max_steps
            start = time.time()


            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            for step in range(max_steps):
                fetches = {
                    "global_step": sv.global_step,
                    "data_loss": model.data_loss,
                    "gt_loss": GT_loss
                }
               
                results = sess.run(fetches, options = None, run_metadata = None)

                if should(self.progress_freq) or step == 0:
                    self.logging(step, max_steps, start, results, logging_text, batch_size)

                if should(self.save_freq) or step == 0:# or results['gt_loss'] < 0.006:
                    res = sess.run(display_fetches)

                    self.dump_outputs(res, output_folder, first_save, step)

                    code = res["code"]
                    np.save(os.path.join(output_folder, "z_%d.npy" % step), code)
                    first_save = True

                if sv.should_stop():
                    print("sv stopped.")
                    break

                # end event
                if step == max_steps - 1:
                    self.svbrdf_ae_opt = res["outputs"]
                    n,d,r,s = np.split(self.svbrdf_ae_opt, 4, axis = 1)
                    self.svbrdf_ae_opt = np.concatenate([n,d,r,s], axis = -1)
                    self.svbrdf_ae_opt = self.svbrdf_ae_opt[np.newaxis, ...]

                    self.svbrdf_ae_opt_image_path = os.path.join(output_folder, "output_%d.png" % step)
                    np.save(os.path.join(output_folder, 'output_%d.npy' % step), self.svbrdf_ae_opt)

                    with open(os.path.join(output_folder, "log.txt"),'w+') as f:
                        for line in logging_text:
                            f.write(line)
                    
                    print("Ae opt done.")

                    break

                sess.run(model.train)



    def refine_opt(self, input_basename, output_folder, ae_output_folder=None):

        if self.svbrdf_ae_opt is None:
            if self.refine_init == 'npy':
                code_initial_value = np.load(os.path.join(ae_output_folder, 'output_%d.npy' % (self.max_steps -1) ))
            elif self.refine_init == 'image':
                code_initial_value = load_svbrdf(os.path.join(ae_output_folder, 'output_%d.png' % (self.max_steps - 1)), self.scale_size, self.data_format)
        else:   
            if self.refine_init == 'npy':
                code_initial_value = self.svbrdf_ae_opt
            elif self.refine_init == 'image':
                code_initial_value = load_svbrdf(self.svbrdf_ae_opt_image_path, self.scale_size, self.data_format)
        
        if self.Wlvs is None or self.input_samples is None:
            self.init(input_basename, output_folder)
            
        # get model
        model = self.optimize_refine_model(self.input_samples, self.Wlvs, code_initial_value)

        convert_input, convert_output, GT_loss = self.convert_results(model.outputs[0])

        with tf.name_scope("encode_images"):
            if self.data_loss_type != 'test':
                display_fetches = {
                    "input_img": model.others[0],
                    "render_img": model.others[1],
                    "inputs": convert_input,
                    "outputs": convert_output,
                }
            else:
                display_fetches = {
                    "inputs" : convert_input,
                    "outputs" : convert_output
                }


        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(logdir=None,init_op = init_op , save_summaries_secs=0, saver=None, summary_writer=None)

        logging_text = []
        first_save = False
        batch_size = 1

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True


        with sv.managed_session(config=config) as sess:
            tf.reset_default_graph()

            max_steps = self.refine_max_steps
            start = time.time()

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)


            for step in range(max_steps):
                fetches = {
                    "global_step": sv.global_step,
                    "data_loss": model.loss,
                    "gt_loss": GT_loss
                }

                results = sess.run(fetches, options = None, run_metadata = None)
                
                if should(self.refine_progress_freq) or step == 0:
                    self.logging(step,max_steps, start, results, logging_text, batch_size)

                if should(self.refine_save_freq) or step == 0:
                    res = sess.run(display_fetches)
                    self.dump_outputs(res, output_folder, first_save, step)
                    first_save = True


                if sv.should_stop():
                    print("sv stopped.")
                    break

                # end event
                if step == max_steps - 1:# or results['gt_loss'] < 0.006:
                    #print("mean, std: %f, %f" % (np.mean(code), np.std(code)))
                    with open(os.path.join(output_folder, "log.txt"),'w+') as f:
                        for line in logging_text:
                            f.write(line)
                    print("Refine opt done.")
                    break

                sess.run(model.train)
