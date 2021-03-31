"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: model.py
about: model for DDMSNet
date: 03/07/20
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import os

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir, dataset_name='kitti'):
        super().__init__()
        self.subsets = ['smallSnow/', 'mediumSnow/', 'largeSnow/']
        self.dataset_name = dataset_name
        val_list_gt_path = []
        val_list_syn_path = []
        if self.dataset_name == 'snow100k':
            for subset in self.subsets:
                val_list_gt_path.append(os.listdir(val_data_dir  + subset + 'gt/'))
                val_list_syn_path.append(os.listdir(val_data_dir + subset + 'synthetic/'))
        else:
            for subset in self.subsets:
                val_list_gt_path.append(os.listdir(val_data_dir  + 'gt/'))
                val_list_syn_path.append(os.listdir(val_data_dir + subset))
        self.haze_names = val_list_syn_path
        self.gt_names = val_list_gt_path
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = [self.haze_names[i][index] for i in range(3)]
        gt_name = [self.gt_names[i][index] for i in range(3)]

        haze_img = []
        gt_img = []
        if self.dataset_name == 'snow100k':
            for i, subset in enumerate(self.subsets):
                haze_img.append(Image.open(self.val_data_dir + subset + 'synthetic/' + haze_name[i]))
                gt_img.append(Image.open(self.val_data_dir + subset + 'gt/' + gt_name[i]))
        else:
            for i, subset in enumerate(self.subsets):
                haze_img.append(Image.open(self.val_data_dir + subset + haze_name[i]))
                gt_img.append(Image.open(self.val_data_dir + 'gt/' + gt_name[i]))
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = []
        gt = []
        for i in range(3):
            haze.append(transform_haze(haze_img[i]))
            gt.append(transform_gt(gt_img[i]))

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names[0])


class Snow100KValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        self.subsets = ['smallSnow/', 'mediumSnow/', 'largeSnow/']
        val_list_gt_path = []
        val_list_syn_path = []
        for subset in self.subsets:
            val_list_gt_path.append(sorted(os.listdir(val_data_dir + subset + 'gt/')))
            val_list_syn_path.append(sorted(os.listdir(val_data_dir + subset + 'synthetic')))
       
        self.haze_names = val_list_syn_path
        self.gt_names = val_list_gt_path
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = [self.haze_names[i][index] for i in range(3)]
        gt_name = [self.gt_names[i][index] for i in range(3)]

        haze_img = []
        gt_img = []
        for i, subset in enumerate(self.subsets):
            haze_img.append(Image.open(self.val_data_dir + subset + haze_name[i]))
            gt_img.append(Image.open(self.val_data_dir + 'gt/' + gt_name[i]))

        # --- Transform to tensor --- #
        
        transform_haze = Compose([ ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = []
        gt = []
        for i in range(3):
            haze.append(transform_haze(haze_img[i]))
            gt.append(transform_gt(gt_img[i]))
            #print(haze[i].size())
            #print(gt[i].size())
        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names[0])

'''
class Snow100KSmallValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        #self.subsets = ['Snow100K-S/', 'Snow100K-M/', 'Snow100K-L/']
        #for subset in self.subsets:
        val_list_gt_path = os.listdir(val_data_dir + 'Snow100K-S/' + 'gt/')[0:20]
        val_list_syn_path = os.listdir(val_data_dir + 'Snow100K-S/' + 'synthetic/')[0:20]
       
        self.haze_names = val_list_syn_path
        self.gt_names = val_list_gt_path
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index] 

        #for i, subset in enumerate(self.subsets):
        haze_img = Image.open(self.val_data_dir + 'Snow100K-S/' + 'synthetic/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'Snow100K-S/' + 'gt/' + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])


        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)
        #print(haze.size())
        #print(gt.size())
        #print()
        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
'''