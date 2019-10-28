#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import importlib
from optimizer import Optimizer
from data_util import load_files
import os
import numpy as np
import random

parser = argparse.ArgumentParser()

parser.add_argument("--dataDir", "-d", required=True)
parser.add_argument("--logDir", "-l", required=True)
parser.add_argument("--initDir", type=str, required=True)
parser.add_argument("--init_method", type=str, required=True, choices=['rand','code','svbrdf'])
parser.add_argument("--network", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--N", type=int,required=True)

parser.add_argument("--wlvDir", type=str, default=None)
parser.add_argument("--exposure_scale", type=float, default=1.0)
parser.add_argument("--input_type", type=str, default='svbrdf', choices=['svbrdf','image'])
parser.add_argument("--wlv_type", type=str, default='random', choices=['load','random'])
parser.add_argument("--LDR", dest = "LDR", action = "store_true")
parser.add_argument("--not-LDR", dest = "LDR", action = "store_false")
parser.set_defaults(LDR=False)
parser.add_argument("--seed", type=int, default=20181120)
parser.add_argument("--refine_init", type=str, default='npy', choices=['npy','image'])


parser.add_argument("--scale_size", type = int,default=256)
parser.add_argument("--max_steps", type = int, default = 4000)
parser.add_argument("--refine_max_steps", type = int, default = 200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--progress_freq", type = int, default = 500)
parser.add_argument("--save_freq", type = int, default = 500)
parser.add_argument("--refine_progress_freq", type = int, default = 50)
parser.add_argument("--refine_save_freq", type = int, default = 50)
parser.add_argument("--initial_mu", type=float, default=0.0)
parser.add_argument("--initial_std", type=float, default=1.0)
parser.add_argument("--data_format", type=str, default='npy')
parser.add_argument("--data_loss_type", type=str, choices=['l1' ,'l1log','test'], default='l1log')

args,unknown = parser.parse_known_args()

def set_seed():
    if args.seed == -1:
        args.seed = random.randint(0, 2 ** 31 - 1)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def main():
    set_seed()
    print(args)
    print(unknown)

    network = importlib.import_module(args.network)

    optimizer = Optimizer(network.code_shape,
                    network.encoder_var_names,
                    network.decoder_var_names,
                    network.encoder,
                    network.decoder, 
                    args)

    BaseNames = []
    with open(os.path.join(args.dataDir, 'files.txt'),'r') as f:
        for line in f:
            BaseNames.append(line.strip())
    
    for f in BaseNames:
        set_seed()
        print('[INFO]' + f)
       
        output_folder = os.path.join(args.logDir, f)
        refine_output_folder = os.path.join(args.logDir, f + '_refine')

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not os.path.exists(refine_output_folder):
            os.mkdir(refine_output_folder)
        
        if os.path.exists(os.path.join(output_folder, 'log.txt')):
            print('Already done ae opt. ignore: %s' % output_folder)
            if os.path.exists(os.path.join(refine_output_folder, 'log.txt')):
                print('Already done refine opt. ignore: %s' % output_folder)
            else:
                optimizer.refine_opt(f, refine_output_folder, output_folder)
        else:
            optimizer.ae_opt(f, output_folder)
            optimizer.refine_opt(f, refine_output_folder, output_folder)

if __name__ == "__main__":
    if not os.path.exists(args.logDir):
        os.makedirs(args.logDir)
    main()
