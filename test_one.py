"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: test_one.py
about: test single image 
date: 07/03/20
"""

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData, Snow100KValData
from model import GridDehazeNet, MultiScaleGridModel, DesnowModelDepth, DesnowModelSemantic, DesnowModelMulti, GridDehazeNetSingle, ImageMultiScaleNet, DDMSNet
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
plt.switch_backend('agg')

import cv2
import dill
import numpy as np
from VNL_depth.lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from VNL_depth.lib.models.metric_depth_model import MetricDepthModel
from VNL_depth.lib.core.config import merge_cfg_from_file, print_configs
from VNL_depth.lib.models.image_transfer import bins_to_depth
from collections import OrderedDict
from torchvision.transforms import ToPILImage

import os
os.environ['CUDA_VISIBLE_DEVICES']= '2, 3'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from torch.backends import cudnn
import semantic_seg.network
from semantic_seg.datasets import cityscapes, kitti
# We only need BN layer
from semantic_seg.config import infer_cfg

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=0.0002, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=8, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-levels', help='Set multi-scale levels of the network', default=3, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
# category = "args.category"

# print('--- Hyper-parameters for training ---')
# print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
#       'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
#       train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #

train_data_dir = './'
val_data_dir = './Cityscape-Dataset/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define depth network --- #
depth_extract_net = MetricDepthModel()
depth_extract_net = depth_extract_net.to(device)
depth_extract_net = nn.DataParallel(depth_extract_net, device_ids=device_ids)
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
    print('--- depth net weight loaded ---')
except:
    print('--- no depth weight loaded ---')

# --- frozen all params of depth network --- #
for param in depth_extract_net.parameters():
    param.requires_grad = False


# --- Define semantic network --- #
infer_cfg(train_mode=False)
arch = 'semantic_seg.network.deepv3.DeepWV3Plus'
dataset_cls = kitti
semantic_extract_net = semantic_seg.network.get_net(arch, dataset_cls, criterion=None)
semantic_extract_net = semantic_extract_net.to(device)

# --- Load semantic model --- #
ckpt_path = './kitti_best.pth'
ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, pickle_module=dill)
state_dict = ckpt['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
semantic_extract_net.load_state_dict(new_state_dict)
print('--- semantic net weight loaded ---')

# --- frozen all params of depth network --- #
for param in semantic_extract_net.parameters():
    param.requires_grad = False
semantic_extract_net.eval()

# --- Define the backbone network --- #
print("DDMSNet:")
net = DDMSNet(depth_extract_model=depth_extract_net, semantic_extract_model=semantic_extract_net, height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
# --- Load the network weight --- #
try:
    ckpt_path = './cityscapes_DDMSNet'
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['net'])
    epoch = ckpt['epoch'] + 1
    print('--- backbone weight loaded ---')
except:
    epoch = 0
    print('--- no weight loaded')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

net.eval()

# --- Test a image --- #
raw_img_dir = './img/raw'
desnow_img_dir = './img/desnow'
filename = os.path.join(raw_img_dir, 'test_pic.png')
save_path = os.path.join(desnow_img_dir, 'desnow_{}'.format("test_pic.png"))
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img_raw = Image.open(filename)
O = transform(img_raw)
O = O[None, :, : ,:]
print(O.size())
desnow = net(O)

print(save_path)
save_img = desnow[0].detach().cpu().numpy()
save_img = np.transpose(save_img, (1, 2, 0))
save_img = save_img[:, :, ::-1]
cv2.imwrite(save_path, save_img * 255)

