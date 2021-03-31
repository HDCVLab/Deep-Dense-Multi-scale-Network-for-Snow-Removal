"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: test.py
about: test on Dataset
date: 07/03/20
"""

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DDMSNet
from utils import validation, validationSnow100K
from val_data import ValData, Snow100KValData
import cv2
import dill
import numpy as np
from VNL_depth.lib.utils.net_tools import load_ckpt
from VNL_depth.lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from VNL_depth.lib.models.metric_depth_model import MetricDepthModel
from VNL_depth.lib.core.config import merge_cfg_from_file, print_configs
from VNL_depth.lib.models.image_transfer import bins_to_depth
from collections import OrderedDict

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from torch.backends import cudnn
import semantic_seg.network
from semantic_seg.datasets import cityscapes, kitti
# We only need BN layer
from semantic_seg.config import infer_cfg

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=8, type=int)
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\nnum_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\n'
      .format(val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss))

# --- Set category-specific hyper-parameters  --- #

val_data_dir = './Snow100K-Dataset/test/'


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define depth network --- #
depth_extract_net = MetricDepthModel()
depth_extract_net = depth_extract_net.to(device)
depth_extract_net = nn.DataParallel(depth_extract_net)

# --- load depth model --- #
try:
    ckpt_path = './kitti_eigen.pth'
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, pickle_module=dill)
    state_dict = ckpt['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  # add `module.`
        new_state_dict[name] = v
    depth_extract_net.load_state_dict(new_state_dict)
    #depth_extract_net.load_state_dict(ckpt['model_state_dict'])
    print('--- depth net weight loaded ---')
except:
    print('--- no depth weight loaded ---')
# --- frozen all params of depth network --- #
for param in depth_extract_net.parameters():
    param.requires_grad = False
depth_extract_net.eval()


infer_cfg(train_mode=False)
arch = 'semantic_seg.network.deepv3.DeepWV3Plus'
dataset_cls = kitti
semantic_extract_net = semantic_seg.network.get_net(arch, dataset_cls, criterion=None)
#semantic_extract_net = nn.DataParallel(semantic_extract_net, device_ids=device_ids)

# --- Load semantic model --- #
try:
    ckpt_path = './kitti_best.pth'
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, pickle_module=dill)
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        #name = 'module.' + k  # add `module.`
        name = k[7:]
        new_state_dict[name] = v
    semantic_extract_net.load_state_dict(new_state_dict)
    #semantic_extract_net.load_state_dict(ckpt['state_dict'])
    print('--- semantic net weight loaded ---')
except:
    print('--- no semantic weight loaded ---')
# --- frozen all params of depth network --- #
for param in semantic_extract_net.parameters():
    param.requires_grad = False
semantic_extract_net.eval()

# --- Define the network --- #
net = DDMSNet(depth_extract_model=depth_extract_net, semantic_extract_model=semantic_extract_net, height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)

net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
# try:
    #ckpt_path = './checkpoint_semantic/semantic_prune2_99'
ckpt_path = './ckpt_snow100k/snow100k_28_5000'
ckpt = torch.load(ckpt_path)
net.load_state_dict(ckpt['net'])
#optimizer.load_state_dict(ckpt['opt'])
epoch = ckpt['epoch'] + 1
print('--- backbone weight loaded ---')
# except:
#     print('--- no weight loaded')

val_data_loader = DataLoader(ValData(val_data_dir, dataset_name='snow100k'), batch_size=1, shuffle=False, num_workers=24, drop_last=True)

# --- Use the evaluation model in testing --- #


net.eval()
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation(net, val_data_loader, device, save_tag=False)
end_time = time.time() - start_time
subsets = ['smallSnow/', 'mediumSnow/', 'largeSnow/']
for i, subset in enumerate(subsets): 
  print('val_psnr in {0}: {1:.2f}, val_ssim in {0}: {2:.4f}'.format(subset, val_psnr[i], val_ssim[i]))
print('validation time is {0:.4f}'.format(end_time))
