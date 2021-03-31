"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: model.py
about: train dataset for DDMSNet
date: 03/07/20
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        train_list = train_data_dir + 'snow100k.txt'
        with open(train_list) as f:
            contents = f.readlines()
            #snow_names = [i.strip() for i in contents]
            #gt_names = [i.split('  ')[0] for i in snow_names]
            snow_names = [i.split('  ')[0].strip() for i in contents]
            gt_names = [i.split('  ')[1].strip() for i in contents]

        self.snow_names = snow_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        snow_name = self.snow_names[index]
        gt_name = self.gt_names[index]

        #snow_img = Image.open(self.train_data_dir + 'hazy/' + snow_name)
        snow_img = Image.open(self.train_data_dir + snow_name)

        try:
           # gt_img = Image.open(self.train_data_dir + 'clear/' + gt_name + '.jpg')
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
           # gt_img = Image.open(self.train_data_dir + 'clear/' + gt_name + '.png').convert('RGB')
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = snow_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        snow_crop_img = snow_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_snow = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        snow = transform_snow(snow_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(snow.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return snow, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.snow_names)

