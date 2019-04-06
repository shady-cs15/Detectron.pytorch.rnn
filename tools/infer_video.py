from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import cv2
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN_with_RNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default=None)
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)
    parser.add_argument('--save_memory', help='save l2 norm of memories', action='store_true')
    parser.add_argument('--memory_dir', help='directory to save memories', default=None)
    parser.add_argument('--skip_frames', help='frames to skip', default=0, type=int)

    args = parser.parse_args()

    return args


def main():
    """main function"""
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if not cfg.RNN.RNN_ON:
        print('RNN mode must be on')
        raise Exception
    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    root_dir = os.path.dirname(args.image_dir.strip('/'))
    if args.memory_dir is None and args.save_memory:
        args.memory_dir = os.path.join(root_dir, 'memory')
        print('memories will be saved to', args.memory_dir)
    if args.output_dir is None:
        args.output_dir = os.path.join(root_dir, 'dets-rnn')
        print('outputs will be saved to', args.output_dir)
    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'imnet_vid':
        dataset = datasets.get_imnet_vid_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN_with_RNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()
    if args.image_dir:
        imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images
    num_images = len(imglist)

    # clear output and memory dir
    shutil.rmtree(args.output_dir)
    shutil.rmtree(args.memory_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_memory and not os.path.exists(args.memory_dir):
        os.makedirs(args.memory_dir)

    memory = None
    imglist = sorted(imglist)
    for i in xrange(0, num_images, args.skip_frames+1):
        print('img', imglist[i])
        im = cv2.imread(imglist[i])
        assert im is not None

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps, blob_conv_with_mem = im_detect_all(maskRCNN, im, timers=timers, memory=memory)
        blob_conv = blob_conv_with_mem['blob_conv']
        memory = blob_conv_with_mem['memory']

        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
        
        if args.save_memory:
            memory_norm = torch.norm(memory[0], dim=1).cpu().numpy()
            plt.imshow(memory_norm[0], cmap=plt.get_cmap('viridis'))
            file_id = os.path.join(args.memory_dir, imglist[i].split('/')[-1].split('.')[0]+'.png')
            plt.savefig(file_id)
        

if __name__ == '__main__':
    main()
