"""
paper: Deep Dense Multi-scale Network for Snow Removal Using Semantic and Geometric Priors
file: model.py
about: model for DDMSNet
author: Rongqing Li
date: 03/07/20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, init_channels, inc_rate, need_semantic=False):
        super(AttentionBlock, self).__init__()
        self.inc_rate = inc_rate
        self.kernel_size = 3
        self.in_channels = init_channels
        self.out_channels = init_channels
        self.dilation_rate = 2
        self.phase = 3
        self.module = nn.ModuleDict()
        # Layer 0, take depth map of 1 channel as input
        
        self.module.update({'0_0':
                                nn.Conv2d(1, self.out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)})
        self.module.update({'0_1':
                            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)})
        self.out_channels *= inc_rate
        self.module.update({'1_0':
                            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)})
        self.in_channels *= inc_rate
        self.module.update({'1_1':
                            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)})
        self.out_channels *= inc_rate
        self.module.update({'2_0':
                            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)})
        self.in_channels *= inc_rate
        self.module.update({'2_1':
                            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)})
        self.module.update({'2_2':
                            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=1)})
    
    def forward(self, x):
        out = x
        for i in range(self.phase):
            out = self.module['{}_0'.format(i)](out)
            out = F.relu(out)
            out = self.module['{}_1'.format(i)](out)
            out = F.relu(out)
            if i == self.phase - 1:
                out = self.module['{}_2'.format(i)](out)
        b, c, h, w = list(out.size())
        out = torch.reshape(out, (b, c, h * w))
        softmax = nn.Softmax(dim=2)
        out = softmax(out)
        out = torch.reshape(out, (b, c, h, w))
        return out

class GroupConv(nn.Module):
    def __init__(self, in_channels=32, group=8, kernel_size=3):
        super(GroupConv, self).__init__()
        self.in_channels = 32
        self.out_channels = 3 * group
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, groups=group)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels , kernel_size=kernel_size, padding=(self.kernel_size - 1) // 2, groups=group)
        self.conv3 = nn.Conv2d(self.out_channels, 3, kernel_size=1)
       
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        residual_map = out
        return residual_map

if __name__ == '__main__':
    x = torch.rand(2, 3, 5, 3)
    print(x)
    x = torch.reshape(x, (2, 3, 3 * 5))
    softmax = nn.Softmax(dim=2)
    x = softmax(x)
    x = torch.reshape(x, (2, 3, 5, 3))
    print(x)