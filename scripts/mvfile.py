#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--src_path',type=str,required=True)
parser.add_argument('--dst_path',type=str,required=True)
parser.add_argument('--Ns',type=str,required=True)
parser.add_argument('--idx',type=int,required=True)

args, unknown = parser.parse_known_args()

Ns = [int(i) for i in args.Ns.split(',')]


for idx in Ns:
    dst_path = '%s/N%d' % (args.dst_path,idx)

    if not os.path.exists(dst_path):
        print('[INFO]makedir: %s' % dst_path)
        os.makedirs(dst_path)

    src_path = '%s/render_N%d' % (args.src_path,idx)

    folders = os.listdir(src_path)

    for folder in folders:
        full_path = os.path.join(src_path, folder)
        f = os.path.join(full_path,'output_%d.png' % args.idx)
        shutil.copyfile(f, os.path.join(dst_path, '%s.png' % folder))
