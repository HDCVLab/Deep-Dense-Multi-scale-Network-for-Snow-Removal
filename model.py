"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: model.py
about: model for DDMSNet
date: 03/07/20
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from residual_dense_block import RDB
from torchvision.transforms import Compose, ToTensor, Resize
from depth_attention_model import GroupConv, AttentionBlock
from VNL_depth.lib.models.image_transfer import bins_to_depth
from semantic_seg.datasets import cityscapes, kitti
from torchvision.transforms import ToPILImage
from PIL import Image
import cv2

# --- Downsampling block in Backbone  --- #
class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

# --- Upsampling block in Backbone  --- #
class UpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return out

# --- Main model  --- #
class GridDehazeNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNet, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(2 * in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)
        # x[0][1], x[0][2]
        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])
        
        # x[1][0], x[2][0]
        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        # x[1][1], x[1][2]
        # x[2][1], x[2][2]
        for i in range(1, self.height):
            for j in range(1, self.width // 2):
                channel_num = int(2**(i-1)*self.stride*self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, j)](x_index[i-1][j])
        
        # x[2][3]
        x_index[i][j+1] = self.rdb_module['{}_{}'.format(i, j)](x_index[i][j])
        k = j

        # x[2][4], x[2][5]
        for j in range(self.width // 2 + 1, self.width):
            x_index[i][j] = self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1])

        # x[1][3], x[0][3]
        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][k+1] = self.coefficient[i, k+1, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, k)](x_index[i][k]) + \
                              self.coefficient[i, k+1, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, k+1)](x_index[i+1][k+1], x_index[i][k].size())
        # x[0][4], x[0][5]
        # x[1][4], x[1][5]
        for i in range(self.height - 2, -1, -1):
            for j in range(self.width // 2 + 1, self.width):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][j] = self.coefficient[i, j, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, j-1)](x_index[i][j-1]) + \
                                self.coefficient[i, j, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, j)](x_index[i+1][j], x_index[i][j-1].size())

        out = self.rdb_out(x_index[i][j])
        out = F.relu(self.conv_out(out))

        return out

class GridDehazeNetPrune1(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNetPrune1, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        self.conv_in = nn.Conv2d(2 * in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)

        # 15 rdbs
        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        # 2 downsamples
        _in_channels = depth_rate
        for i in range(height - 1):
            self.downsample_module.update({'{}_{}'.format(i, 0): DownSample(_in_channels)})
            _in_channels *= stride

        # 2 upsamples
        for i in range(height - 2, -1, -1):
            self.upsample_module.update({'{}_{}'.format(i, 0): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x):
        inp = self.conv_in(x)

        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)

        # x[0][1], x[0][2], x[0][3], x[0][4]
        for j in range(1, self.width - 1):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])

        # x[1][0]
        x_index[1][0] = self.downsample_module['0_0'](x_index[0][0])
        
        # x[1][1], x[1][2], x[1][3], x[1][4]
        for j in range(1, self.width - 1):
            x_index[1][j] = self.rdb_module['{}_{}'.format(1, j - 1)](x_index[1][j-1])
        
        # x[2][0]
        x_index[2][0] = self.downsample_module['1_0'](x_index[1][0])

        # x[2][1], x[2][2], x[2][3], x[2][4]
        for j in range(1, self.width - 1):
            x_index[2][j] = self.rdb_module['{}_{}'.format(2, j - 1)](x_index[2][j-1])
        
        # x[2][5]
        x_index[2][5] = self.rdb_module['2_4'](x_index[2][4])
        
        # x[1][5]
        x_index[1][5] = self.coefficient[1, 5, 0, :32][None, :, None, None] * self.rdb_module['1_4'](x_index[1][4]) + \
                        self.coefficient[1, 5, 1, :32][None, :, None, None] * self.upsample_module['1_0'](x_index[2][5], x_index[1][4].size()) 

        # x[0][5]
        x_index[0][5] = self.coefficient[0, 5, 0, :16][None, :, None, None] * self.rdb_module['0_4'](x_index[0][4]) + \
                        self.coefficient[0, 5, 1, :16][None, :, None, None] * self.upsample_module['0_0'](x_index[1][5], x_index[0][4].size())
       
        out = self.rdb_out(x_index[0][5])
        out = F.relu(self.conv_out(out))

        return out

class GridDehazeNetPrune2(nn.Module):
    def __init__(self, need_semantic=False, multiscale=False, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(GridDehazeNetPrune2, self).__init__()
        self.need_semantic = need_semantic
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.stride = stride
        self.depth_rate = depth_rate
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((height, width, 2, depth_rate*stride**(height-1)))), requires_grad=attention)
        if multiscale == True:
            self.conv_in = nn.Conv2d(2 * in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        else:
            self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)
        if need_semantic:
            #print('semantic features are transformed to 120 channels')
            self.transform_feature_map_to_120 =  nn.Conv2d(depth_rate, 120, kernel_size=3, padding=(kernel_size - 1) // 2)
            self.transform_feature_map_to_depth_rate = nn.Conv2d(120, depth_rate, kernel_size=3, padding=(kernel_size - 1) // 2)
        rdb_in_channels = depth_rate
        for i in range(height):
            for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(i, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})
            rdb_in_channels *= stride

        _in_channels = depth_rate
        for i in range(height - 1):
            for j in range(width // 2):
                self.downsample_module.update({'{}_{}'.format(i, j): DownSample(_in_channels)})
            _in_channels *= stride

        for i in range(height - 2, -1, -1):
            for j in range(width // 2, width):
                self.upsample_module.update({'{}_{}'.format(i, j): UpSample(_in_channels)})
            _in_channels //= stride

    def forward(self, x):
        # if args.semantic set, x = [concat_input, semantic_attention_weight]
        if self.need_semantic:
            semantic_attention_weight = x[1]
            inp = self.conv_in(x[0])
        else:
            inp = self.conv_in(x)
            
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        i, j = 0, 0

        x_index[0][0] = self.rdb_in(inp)
        #print('index[0][0]' + str(x_index[0][0].size()))
        # Add semantic information
        if self.need_semantic:
            net_out = self.transform_feature_map_to_120(x_index[0][0])
            net_out = self.groupAttentionMul(net_out, semantic_attention_weight, group=30)
            net_out = self.transform_feature_map_to_depth_rate(net_out)
            x_index[0][0] = x_index[0][0] + net_out
        # x[0][1], x[0][2]
        for j in range(1, self.width // 2):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])
        
        # x[1][0], x[2][0]
        for i in range(1, self.height):
            x_index[i][0] = self.downsample_module['{}_{}'.format(i-1, 0)](x_index[i-1][0])

        # x[1][1], x[2][1]
        for i in range(1, self.height):
            x_index[i][1] = self.rdb_module['{}_{}'.format(i, 0)](x_index[i][0])
    
        # x[1][2], x[2][2]
        for i in range(1, self.height):
            channel_num = int(2**(i-1)*self.stride*self.depth_rate)
            x_index[i][2] = self.coefficient[i, 2, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, 1)](x_index[i][1]) + \
                            self.coefficient[i, 2, 1, :channel_num][None, :, None, None] * self.downsample_module['{}_{}'.format(i-1, 2)](x_index[i-1][2])
        
        # x[2][3], x[2][4], x[2][5]
        for j in range(self.width // 2, self.width):
            x_index[2][j] = self.rdb_module['{}_{}'.format(2, j-1)](x_index[2][j-1])

        # x[1][3], x[0][3]
        for i in range(self.height - 2, -1, -1):
            channel_num = int(2 ** (i-1) * self.stride * self.depth_rate)
            x_index[i][3] = self.coefficient[i, 3, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, 2)](x_index[i][2]) + \
                            self.coefficient[i, 3, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, 3)](x_index[i+1][3], x_index[i][2].size())
        # x[0][4], x[1][4]
        for i in range(0, self.height - 1):
            x_index[i][4] = self.rdb_module['{}_{}'.format(i, 3)](x_index[i][3])

        # x[1][5], x[0][5]
        
        for i in range(self.height - 2, -1, -1):
                channel_num = int(2 ** (i - 1) * self.stride * self.depth_rate)
                x_index[i][5] = self.coefficient[i, 5, 0, :channel_num][None, :, None, None] * self.rdb_module['{}_{}'.format(i, 4)](x_index[i][4]) + \
                                self.coefficient[i, 5, 1, :channel_num][None, :, None, None] * self.upsample_module['{}_{}'.format(i, 5)](x_index[i+1][5], x_index[i][4].size())

        out = self.rdb_out(x_index[0][5])
        out = F.relu(self.conv_out(out))
        return out
    
    def groupAttentionMul(self, net_out, attention, group=8):
        _, net_c, _, _ = net_out.size()
        _, att_c, _, _ = attention.size()
        out_list = torch.split(net_out, net_c // group, dim=1)
        attention_list = torch.split(attention, att_c // group, dim=1)
        # print(np.array(out_list).shape)
        # print(np.array(attention_list).shape)
        #print(out_list)
        #print(attention_list)
        out = []
        for i in range(len(out_list)):
            out.append(out_list[i] * attention_list[i])
        out = torch.cat(out, dim=1)
        return out

class MultiScaleGridModel(nn.Module):
    def __init__(self, need_depth=False, need_semantic=False, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(MultiScaleGridModel, self).__init__()
        self.GDN = GridDehazeNetPrune2(need_semantic=need_semantic, multiscale=True, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True)
        self.conv_out = nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.levels = 3
        self.scale = 0.5
        self.need_semantic = need_semantic
        self.need_depth = need_depth
    def forward(self, x):
        # if adding semantic information, x = [coarse_img, semantic_attention]
        if not self.need_semantic:
            batch, c, h, w = list(x.size())
            inp_pred = x
            x_ori = x
        else:
            batch, c, h, w = list(x[0].size())
            inp_pred = x[0]
            x_ori = x[0]
        x_unwrap = []
        for i in range(self.levels):
            scale = self.scale ** (self.levels - i - 1)
            hi = int(round(h * scale))
            wi = int(round(w * scale))

            inp_haze = F.interpolate(x_ori, size=[hi, wi])
            inp_pred = F.interpolate(inp_pred, size=[hi, wi]).detach()
            if self.need_semantic:
                att = F.interpolate(x[1], size=[hi, wi])
            inp_all = torch.cat((inp_haze, inp_pred), 1)

            # att is semantic_attention
            if self.need_semantic:
                inp_pred = self.GDN([inp_all, att])
                # save_img = ToPILImage()(inp_pred[0].cpu())
                # save_img.save(os.path.join('pic', 'scale_{}_img.png'.format(i)))
            else:
                inp_pred = self.GDN(inp_all)

            if i == self.levels - 1 and self.need_depth == True:
                inp_pred = self.conv_out(inp_pred)
            x_unwrap.append(inp_pred)
        return x_unwrap
       
class DesnowModelDepth(nn.Module):
    def __init__(self, depth_extract_model=None, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(DesnowModelDepth, self).__init__()
        self.coarse_model = GridDehazeNetPrune2(multiscale=False, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.depth_extract_model = depth_extract_model
        self.fine_model = MultiScaleGridModel(need_depth=True, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.depth_attention_model = AttentionBlock(init_channels=4, inc_rate=2)
        self.group_conv = GroupConv(in_channels=32, group=8, kernel_size=3)
    
    def forward(self, x):
        coarse_out = self.coarse_model(x)
        _, depth_map = self.depth_extract_model.module.depth_model(coarse_out)
        depth_map = bins_to_depth(depth_map)
        depth_attention_weight = self.depth_attention_model(depth_map)
        save_img = ToPILImage()(depth_attention_weight[0].cpu())
        save_img.save(os.path.join('pic', 'depth_attention_weight.png'))
        fine_out = self.fine_model(coarse_out)[-1]
        depth_attention_map = self.groupAttentionMul(fine_out, depth_attention_weight)
        residual_map = self.group_conv(depth_attention_map)
        #print(residual_map.size())
        save_img = ToPILImage()(fine_out[0].cpu())
        save_img.save(os.path.join('pic', 'fine_out.png'))
        # print(residual_map.size())
        # print(coarse_out.size())
        # print(depth_map.requires_grad)
        # print(coarse_out.requires_grad)
        out = residual_map + coarse_out
        return out

    def groupAttentionMul(self, net_out, attention, group=8):
        _, net_c, _, _ = net_out.size()
        _, att_c, _, _ = attention.size()
        out_list = torch.split(net_out, net_c // group, dim=1)
        attention_list = torch.split(attention, att_c // group, dim=1)
        # print(np.array(out_list).shape)
        # print(np.array(attention_list).shape)
        # print(out_list)
        # print(attention_list)
        out = []
        for i in range(len(out_list)):
            out.append(out_list[i] * attention_list[i])
        out = torch.cat(out, dim=1)
        return out

class DesnowModelSemantic(nn.Module):
    def __init__(self, semantic_extract_model=None, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(DesnowModelSemantic, self).__init__()
        self.coarse_model = GridDehazeNetPrune2(need_semantic=False, multiscale=False, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.semantic_extract_model = semantic_extract_model
        self.fine_model = MultiScaleGridModel(need_semantic=True, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.attention_model = AttentionBlock(init_channels=15, inc_rate=2, need_semantic=True)
        self.group_conv = GroupConv(in_channels=32, group=8, kernel_size=3)
        self.to_tensor = ToTensor()

    def forward(self, x):
        coarse_out = self.coarse_model(x)
        save_img = ToPILImage()(coarse_out[0].cpu())
        save_img.save(os.path.join('pic', 'coarse_out_img.png'))
        #print('coarse_out:' + str(coarse_out.size()))
        semantic_map = self.semantic_extract_model(coarse_out)
        semantic_map = semantic_map.detach().cpu().numpy()
        
        b, c, h, w, = semantic_map.shape
        new_map = []
        for i in range(b):
            new_map.append(np.argmax(semantic_map[i], axis=0))
        new_map = np.array(new_map)
        new_color = []
        for i in range(b):
            new_color.append(self.to_tensor(kitti.colorize_mask(new_map)[i]))
        new_color = torch.stack(new_color, dim=0).cuda()
        #print(new_color.shape)
        semantic_attention_weight = self.attention_model(new_color)
        # save_img = ToPILImage()(semantic_attention_weight[0].cpu())
        # save_img.save(os.path.join('pic', 'semantic_attention_weight.png'))
        fine_out = self.fine_model([coarse_out, semantic_attention_weight])
        return fine_out

class DesnowModelMulti(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(DesnowModelMulti, self).__init__()
        self.coarse_model = GridDehazeNetPrune2(need_semantic=False, multiscale=False, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.fine_model = MultiScaleGridModel(height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)


    def forward(self, x):
        coarse_out = self.coarse_model(x)
        #save_img = ToPILImage()(coarse_out[0].cpu())
        #save_img.save(os.path.join('pic', 'coarse_out_img.png'))
        #print('coarse_out:' + str(coarse_out.size()))
        #semantic_map = self.semantic_extract_model(coarse_out)
        # semantic_map = semantic_map.detach().cpu().numpy()
        
        # b, c, h, w, = semantic_map.shape
        # new_map = []
        # for i in range(b):
        #     new_map.append(np.argmax(semantic_map[i], axis=0))
        # new_map = np.array(new_map)
        # new_color = []
        # for i in range(b):
        #     new_map = kitti.colorize_mask(new_map)[i]
        #     new_color.append(self.to_tensor(new_map))
        # new_color = torch.stack(new_color, dim=0).cuda()
        # print(new_color.shape)
        #semantic_attention_weight = self.attention_model(semantic_map)
        # save_img = ToPILImage()(semantic_attention_weight[0].cpu())
        # save_img.save(os.path.join('pic', 'semantic_attention_weight.png'))
        fine_out = self.fine_model(coarse_out)
        return fine_out

class GridDehazeNetSingle(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, height=1, width=6, num_dense_layer=4, growth_rate=16):
        super(GridDehazeNetSingle, self).__init__()
        self.rdb_module = nn.ModuleDict()
        self.upsample_module = nn.ModuleDict()
        self.downsample_module = nn.ModuleDict()
        self.height = height
        self.width = width
        self.depth_rate = depth_rate
        self.conv_in = nn.Conv2d(2 * in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.rdb_in = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb_out = RDB(depth_rate, num_dense_layer, growth_rate)
        rdb_in_channels = depth_rate
        for j in range(width - 1):
                self.rdb_module.update({'{}_{}'.format(0, j): RDB(rdb_in_channels, num_dense_layer, growth_rate)})

    def forward(self, x):
        inp = self.conv_in(x)
        x_index = [[0 for _ in range(self.width)] for _ in range(self.height)]
        x_index[0][0] = self.rdb_in(inp)

        # x[0][1] ~ x[0][5]
        for j in range(1, self.width):
            x_index[0][j] = self.rdb_module['{}_{}'.format(0, j-1)](x_index[0][j-1])
        
        out = self.rdb_out(x_index[0][5])
        out = F.relu(self.conv_out(out))
        return out

class ImageMultiScaleNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16):
        super(ImageMultiScaleNet, self).__init__()
        self.GDN = GridDehazeNetSingle(height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.levels = 3
        self.scale = 0.5
    def forward(self, x):
        batch, c, h, w = list(x.size())
        inp_pred = x
        x_ori = x
        
        x_unwrap = []
        for i in range(self.levels):
            scale = self.scale ** (self.levels - i - 1)
            hi = int(round(h * scale))
            wi = int(round(w * scale))

            inp_haze = F.interpolate(x_ori, size=[hi, wi])
            inp_pred = F.interpolate(inp_pred, size=[hi, wi]).detach()
            inp_all = torch.cat((inp_haze, inp_pred), 1)
            inp_pred = self.GDN(inp_all)
            x_unwrap.append(inp_pred)

        return x_unwrap

class DDMSNet(nn.Module):
    def __init__(self, semantic_extract_model=None, depth_extract_model=None, in_channels=3, depth_rate=16, kernel_size=3, stride=2, height=3, width=6, num_dense_layer=4, growth_rate=16, attention=True):
        super(DDMSNet, self).__init__()
        self.coarse_model = GridDehazeNetPrune2(multiscale=False, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.depth_extract_model = depth_extract_model
        self.semantic_extract_model = semantic_extract_model
        self.fine_model = MultiScaleGridModel(need_semantic=True, need_depth=True, height=height, width=width, num_dense_layer=num_dense_layer, growth_rate=growth_rate, attention=attention)
        self.depth_attention_block = AttentionBlock(init_channels=4, inc_rate=2)
        self.semantic_attention_block = AttentionBlock(init_channels=15, inc_rate=2, need_semantic=True)
        self.group_conv = GroupConv(in_channels=32, group=8, kernel_size=3)
        self.to_tensor = ToTensor()

    def forward(self, x):
        coarse_out = self.coarse_model(x)
        #save coarse_out
        save_img = coarse_out[0].detach().cpu().numpy()
        save_img = np.transpose(save_img, (1, 2, 0))[:, :, ::-1]
        #cv2.imwrite(os.path.join('pic', 'train_coarse.png'), save_img*255)
        depth_attention_weight = self.depth_extract(coarse_out)
        
        semantic_attention_weight = self.semantic_extract(coarse_out)
        fine_out = self.fine_model([coarse_out, semantic_attention_weight])[-1]
        depth_aware_feature = self.groupAttentionMul(fine_out, depth_attention_weight)
        residual_map = self.group_conv(depth_aware_feature)
        # save residual_map
        # save_img = residual_map[0].detach().cpu().numpy()
        # save_img = np.transpose(save_img, (1, 2, 0))[:, :, ::-1]
        #cv2.imwrite(os.path.join('pic', 'train_residual_map.png'), save_img*255)
        out = residual_map + coarse_out
        return out

    def depth_extract(self, coarse_out):
        _, depth_map = self.depth_extract_model.module.depth_model(coarse_out)
        depth_map = bins_to_depth(depth_map)
        depth_attention_weight = self.depth_attention_block(depth_map)
        return depth_attention_weight

    def semantic_extract(self, coarse_out):
        semantic_map = self.semantic_extract_model(coarse_out)
        semantic_map = semantic_map.detach().cpu().numpy()

        b, c, h, w, = semantic_map.shape
        new_map = []
        for i in range(b):
            new_map.append(np.argmax(semantic_map[i], axis=0))
        new_map = np.array(new_map)
        new_color = []
        for i in range(b):
            new_color.append(self.to_tensor(kitti.colorize_mask(new_map)[i]))
        new_color = torch.stack(new_color, dim=0).cuda()
        #print(new_color.shape)
        semantic_attention_weight = self.semantic_attention_block(new_color)
        return semantic_attention_weight

    def groupAttentionMul(self, net_out, attention, group=8):
        _, net_c, _, _ = net_out.size()
        _, att_c, _, _ = attention.size()
        out_list = torch.split(net_out, net_c // group, dim=1)
        attention_list = torch.split(attention, att_c // group, dim=1)
        out = []
        for i in range(len(out_list)):
            out.append(out_list[i] * attention_list[i])
        out = torch.cat(out, dim=1)
        return out