# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""PASCAL VOC dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import os
import shutil
import uuid

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR
from utils.io import save_object

logger = logging.getLogger(__name__)


def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir,
    use_salt=True,
    cleanup=True,
    use_matlab=False
):
    filename = _write_imnet_vid_results_file(json_dataset, all_boxes)
    
    ## we will only do matlab eval
    _do_matlab_eval(json_dataset, output_dir)
    
    if cleanup:
        shutil.copy(filename, output_dir)
        os.remove(filename)
    return None

def _write_imnet_vid_results_file(json_dataset, all_boxes):
    # read val.txt from devkit
    devkit_path = DATASETS[json_dataset.name][DEVKIT_DIR]
    val_txt_file = os.path.join(devkit_path, 'ImageSets/VID_val_frames.txt')
    with open(val_txt_file) as f:
        line_splits = [(_get_frame_name_from_path(line.strip().split()[0]),\
                                                line.strip().split()[1])\
                                                for line in f.readlines()]
    
    # check order of names in roidb and val_txt
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        im_name_in_roidb = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert im_name_in_roidb == line_splits[i][0] and \
                i+1 == int(line_splits[i][1]), 'order of names is messed up!'
        
    # create text file for all boxes
    file_name = os.path.join(devkit_path, 'data/all_boxes.txt')
    all_boxes_file = open(file_name, 'w')
    logger.info('saving boxes to {}'.format(file_name))

    # write dets to file
    for im_ind, im_info in enumerate(line_splits):
        frame_name, frame_id = im_info
        for cls_ind, cls in enumerate(json_dataset.classes):
            if cls == '__background__':
                continue
            dets = all_boxes[cls_ind][im_ind]
            if type(dets) == list:
                assert len(dets) == 0, \
                    'dets should be numpy.ndarray or empty list'
                continue
            for k in range(dets.shape[0]):
                det_formatted = '{:s} {:d} -1 {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.\
                                format(frame_id, cls_ind, dets[k, -1],\
                                   dets[k, 0], dets[k, 1],\
                                   dets[k, 2], dets[k, 3])
                all_boxes_file.write(det_formatted)

    all_boxes_file.close()
    logger.info('boxes saved ...')
    return file_name


def _get_frame_name_from_path(frame_name_path):
    frame_name_path = frame_name_path.split('/')
    frame_name = ''
    for split in frame_name_path:
        frame_name+= (split+'_')
    return frame_name[:-1]


def _do_matlab_eval(json_dataset, output_dir):
    import subprocess
    logger.info('-----------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code.')
    logger.info('-----------------------------------------------------') 

    # set up paths
    devkit_path = DATASETS[json_dataset.name][DEVKIT_DIR]
    pred_file = os.path.join(devkit_path, 'data/all_boxes.txt')
    meta_file = os.path.join(devkit_path, 'data/meta_vid.mat')
    eval_file = os.path.join(devkit_path, 'ImageSets/VID_val_frames.txt')
    cache_file = '../data/gt_cache.mat'
    gnth_dir = os.path.join(devkit_path, 'data/gt')
    blacklist_file = ''

    # gnth dir sanity check
    video_dirs = os.listdir(gnth_dir+'/val')
    total_xmls = 0
    for vid_dir in video_dirs:
        vid_dir = os.path.join(gnth_dir+'/val', vid_dir)
        if os.path.isdir(vid_dir):
            xmls = os.listdir(vid_dir)
            for xml in xmls:
                if xml.endswith('.xml'):
                    total_xmls +=1
    assert total_xmls == 176126, '176126 files should be present, # files found: '\
                                                                 + str(total_xmls)

    # run matlab command
    path = os.path.join(
        cfg.ROOT_DIR, 'data', 'imnet_vid', 'devkit', 'evaluation')
    cmd = 'cd {} && '.format(path)
    cmd += 'matlab -nodisplay -nodesktop '
    cmd += '-r "dbstop if error; '
    cmd += 'eval_vid_detection(\'{:s}\', \'{:s}\', \'{:s}\', \'{:s}\', \'{:s}\', \'{:s}\'); quit;"'\
            .format(pred_file, gnth_dir, meta_file, eval_file, blacklist_file, cache_file)
    logger.info('Running:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)
