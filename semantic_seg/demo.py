import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

import network
from optimizer import restore_snapshot
from datasets import cityscapes, kitti
# We only need BN layer
from config import infer_cfg, cfg

infer_cfg(train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

img_dir = './test_imgs/'
# get net
arch = 'network.deepv3.DeepWV3Plus'
dataset_cls = kitti
net = network.get_net(arch, dataset_cls, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
ckpt_path = './kitti_best.pth'
ckpt = torch.load(ckpt_path)
net.load_state_dict(ckpt['state_dict'])
#net, _ = restore_snapshot(net, optimizer=None, snapshot=ckpt_path, restore_optimizer_bool=False)
net.eval()
print('Net restored.')

# get data
demo_img_path = os.path.join(img_dir, 'kitti-13.png')
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
img = Image.open(demo_img_path).convert('RGB')
img_tensor = img_transform(img)

# predict
with torch.no_grad():
    img = img_tensor.unsqueeze(0).cuda()
    pred = net(img)
    print('Inference done.')

pred = pred.cpu().numpy().squeeze()
pred = np.argmax(pred, axis=0)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

save_dir = 'test_imgs/'
colorized = dataset_cls.colorize_mask(pred)
#print(colorized.shape)
colorized.save(os.path.join(save_dir, 'color_mask.png'))
print('Results saved.')

# label_out = np.zeros_like(pred)
# for label_id, train_id in args.dataset_cls.id_to_trainid.items():
#     label_out[np.where(pred == train_id)] = label_id
#     cv2.imwrite(os.path.join(args.save_dir, 'pred_mask.png'), label_out)
# print('Results saved.')
