from __future__ import absolute_import
import torch
import numpy as np
from ._ext import assemble
import pdb

def assemble_gpu(cur_prev_aff, feat, output, masked_cpa, pad):
    assemble.gpu_assemble(cur_prev_aff, feat, output, masked_cpa, pad)
