import os
import sys
o_path = os.getcwd()
sys.path.append(o_path) 
import cv2
import torch
import dill
import numpy as np
from lib.utils.net_tools import load_ckpt
from lib.utils.logging import setup_logging
import torchvision.transforms as transforms
#from tools.parse_arg_test import TestOptions
from lib.models.metric_depth_model import MetricDepthModel
from lib.core.config import cfg, merge_cfg_from_file
from lib.models.image_transfer import bins_to_depth

logger = setup_logging(__name__)


def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img


if __name__ == '__main__':

    cfg_file = 'lib/configs/resnext101_32x4d_kitti_class'
    merge_cfg_from_file(cfg_file)

    # load model
    model = MetricDepthModel()
    model.eval()

    # load checkpoint
    ckpt_path = '../kitti_eigen.pth'
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, pickle_module=dill)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.cuda()
    model = torch.nn.DataParallel(model)

    #@path = os.path.join(cfg.ROOT_DIR, './test_imgs') # the dir of imgs

    with torch.no_grad():
        img = cv2.imread('/home/omnisky/Desnow/GridDehazeNet-master/VNL_depth/test_imgs/kitti-13.png')
        img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
        img_torch = scale_torch(img_resize, 255)
        img_torch = img_torch[None, :, :, :].cuda()

        
        _, pred_depth_softmax= model.module.depth_model(img_torch)
        #print(pred_depth_softmax.size())
        # we only need pred_depth
        pred_depth = bins_to_depth(pred_depth_softmax)


        # -------------For visualization----------------#
        #print(pred_depth.size())
        pred_depth = pred_depth.cpu().numpy().squeeze()
        #print(pred_depth.shape)
        pred_depth_scale = (pred_depth / pred_depth.max() * 65536).astype(np.uint16)  # scale 60000 for visualization
        #print(pred_depth_scale)
        cv2.imwrite('-raw.png', pred_depth_scale)