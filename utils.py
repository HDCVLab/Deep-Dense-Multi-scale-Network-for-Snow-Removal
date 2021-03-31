"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: model.py
about: model for DDMSNet
date: 03/07/20
"""

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize
import os 
import cv2
def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list
  

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def validation(net, val_data_loader, device, save_tag=False):
    """
    :param net: DDMSNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    
    # transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # f = './desnow_pic/snow100K_realsnow_图8/4/crossing_03017.jpg'
    # save_path = './desnow_pic/snow100K_realsnow_图8/4/ours.png'
    # file_list = os.listdir('./new_1')
    with torch.no_grad():   
        # for f in file_list:
        #     if f[0] == 'o':
        #         continue
        #     save_path = './new_1/ours/' + f
        #     print(f)
        #     img_raw = Image.open('./new_1/' + f)
        #     O = transform(img_raw)
        #     O = O[None, :, : ,:]
        #     print(O.size())
        #     desnow = net(O)
        #     prefix = f.split('.')[0]
            
        #     print(save_path)
        #     save_img = desnow[0].cpu().numpy()
        #     save_img = np.transpose(save_img, (1, 2, 0))
        #     save_img = save_img[:, :, ::-1]
        #     # save_img = np.clip(save_img, 0, 1)
        #     # save_img = save_img * 255
        #     cv2.imwrite(save_path, save_img * 255)

        psnr_ = []
        ssim_ = []
        print(val_data_loader)    
        for batch_id, val_data in enumerate(val_data_loader):
            # print(np.array(val_data).shape)
            with torch.no_grad():
                haze, gt, image_name = val_data
                for i in range(3):
                    haze[i] = haze[i].to(device)
                    gt[i] = gt[i].to(device)
                dehaze = [net(haze_img) for haze_img in haze]
                
                haze_pic = haze[2][0].cpu().numpy()
                haze_pic = np.transpose(haze_pic, (1, 2, 0))
                haze_pic = haze_pic[:, :, ::-1]

                dehaze_pic = dehaze[2][0].cpu().numpy()
                dehaze_pic = np.transpose(dehaze_pic, (1, 2, 0))
                dehaze_pic = dehaze_pic[:, :, ::-1]

                cv2.imwrite(os.path.join('pic', 'dehaze{}.png'.format(batch_id)), dehaze_pic*255)
                cv2.imwrite(os.path.join('pic', 'haze{}.png'.format(batch_id)), haze_pic*255)

            psnr_list = []
            ssim_list = []
            for i in range(3):
                # --- Calculate the average PSNR --- #
                psnr_list.extend(to_psnr(dehaze[i], gt[i]))
                # --- Calculate the average SSIM --- #
                ssim_list.extend(to_ssim_skimage(dehaze[i], gt[i]))
            psnr_.append(psnr_list)
            ssim_.append(ssim_list)
            if(batch_id % 1000 == 0):
                print("processed %d images" % batch_id)
            
        
        ret_psnr = []
        ret_ssim = []
        for i in range(3):
            avr_psnr = []
            avr_ssim = []
            for psnr_val in psnr_:
                avr_psnr.append(psnr_val[i])
            for ssim_val in ssim_:
                avr_ssim.append(ssim_val[i])
            ret_psnr.append(sum(avr_psnr) / len(avr_psnr))
            ret_ssim.append(sum(avr_ssim) / len(avr_ssim))

        return ret_psnr, ret_ssim

def validationSnow100K(net, val_data_loader, device, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
     
    psnr_ = []
    ssim_ = []
    print(val_data_loader)    
    for batch_id, val_data in enumerate(val_data_loader):
        # print(np.array(val_data).shape)
        with torch.no_grad():
            haze, gt, image_name = val_data
            #print(len(haze))
            #print(image_name)
            b, _, _, _ = list(haze[0].size())
            # small_snow = haze[0]
            # medium_snow = haze[1]
            # large_snow = haze[2]
            # for i in range(b):
            #     small_snow.append(haze[i][0])
            #     medium_snow.append(haze[i][1])
            #     large_snow.append(haze[i][2])
            #haze = torch.Tensor([small_snow, medium_snow, large_snow]).cuda()
            
            for i in range(3):
                haze[i] = haze[i].to(device)
                gt[i] = gt[i].to(device)
            dehaze = [net(haze_img) for haze_img in haze]

            haze_pic = haze[0][0].cpu().numpy()
            haze_pic = np.transpose(haze_pic, (1, 2, 0))
            haze_pic = haze_pic[:, :, ::-1]

            dehaze_pic = dehaze[0][0].cpu().numpy()
            dehaze_pic = np.transpose(dehaze_pic, (1, 2, 0))
            dehaze_pic = dehaze_pic[:, :, ::-1]

            cv2.imwrite(os.path.join('pic', 'dehaze{}.png'.format(batch_id)), dehaze_pic*255)
            cv2.imwrite(os.path.join('pic', 'haze{}.png'.format(batch_id)), dehaze_pic*255)

        psnr_list = []
        ssim_list = []
        for i in range(3):
            # --- Calculate the average PSNR --- #
            psnr_list.extend(to_psnr(dehaze[i], gt[i]))
            # --- Calculate the average SSIM --- #
            ssim_list.extend(to_ssim_skimage(dehaze[i], gt[i]))
        psnr_.append(psnr_list)
        ssim_.append(ssim_list)
        if(batch_id % 1000 == 0):
          print("processed %d images" % batch_id)

    
    ret_psnr = []
    ret_ssim = []
    for i in range(3):
      avr_psnr = []
      avr_ssim = []
      for psnr_val in psnr_:
        avr_psnr.append(psnr_val[i])
      for ssim_val in ssim_:
        avr_ssim.append(ssim_val[i])
      ret_psnr.append(sum(avr_psnr) / len(avr_psnr))
      ret_ssim.append(sum(avr_ssim) / len(avr_ssim))

    return ret_psnr, ret_ssim

def validationSingle(net, val_data_loader, device, category, save_tag=False):
    print(val_data_loader)    
    for batch_id, val_data in enumerate(val_data_loader):
        # print(np.array(val_data).shape)
        with torch.no_grad():
            haze, gt = val_data
            b, _, _, _ = list(haze.size())
        
            dehaze = net(haze)
            haze = haze.to(device)
            gt = gt.to(device)
            #dehaze_pic = dehaze[0][0].cpu().numpy()
            #dehaze_pic = np.transpose(dehaze_pic, (1, 2, 0))
            #dehaze_pic = dehaze_pic[:, :, ::-1]

        psnr_list = []
        ssim_list = []
        #for i in range(3):
        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))
        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))

        if(batch_id % 1000 == 0):
          print("processed %d images" % batch_id)
        
        ret_psnr = sum(psnr_list) / len(psnr_list)
        ret_ssim = sum(ssim_list) / len(ssim_list)


    return ret_psnr, ret_ssim

def save_image(dehaze, image_name, category):
    for i in range(3):
        dehaze_images = torch.split(dehaze[i], 1, dim=0)
        batch_num = len(dehaze_images)

        for ind in range(batch_num):
            utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))

subsets = ['Snow100K-S', 'Snow100K-M', 'Snow100K-L']
def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr))
    for i, subset in enumerate(subsets):
        print('Val_PSNR in {0}: {1:.2f}, Val_SSIM in {0}: {2:.4f}', subset, val_psnr[i], val_ssim[i])

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        ''' 
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)
        '''
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr), file=f)
        for i, subset in enumerate(subsets):
             print('Val_PSNR in {0}: {1:.2f}, Val_SSIM in {0}: {2:.4f}', subset, val_psnr[i], val_ssim[i], file=f)

def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20 if category == 'indoor' else 2

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
