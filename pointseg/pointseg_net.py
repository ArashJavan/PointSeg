import os
import datetime
import copy
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Fire, FireDeconv, SELayer, ASPP


class BaseNet(nn.Module):
    """
    Basenet for all modules
    """

    def __init__(self):
        super(BaseNet, self).__init__()

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})
        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))
        return next(iter(devices))

    def get_modules(self):
        return [self]


class PSEncoder(BaseNet):
    def __init__(self, cfg):
        super(PSEncoder, self).__init__()
        bn_d = 0.1
        self.bypass = cfg['bypass']
        self.input_shape = cfg['input-shape']

        h, w, c = self.input_shape

        ### Ecnoder part
        self.conv1a = nn.Sequential(nn.Conv2d(c, 64, kernel_size=3, stride=(1, 2), padding=1),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        self.conv1b = nn.Sequential(nn.Conv2d(c, 64, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        # First block
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire2 = Fire(64, 16, 64, 64, bn=True, bn_d=bn_d)
        self.fire3 = Fire(128, 16, 64, 64, bn=True, bn_d=bn_d)
        self.se1 = SELayer(128, reduction=2)

        # second block
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire4 = Fire(128, 32, 128, 128, bn=True, bn_d=bn_d)
        self.fire5 = Fire(256, 32, 128, 128, bn=True, bn_d=bn_d)
        self.se2 = SELayer(256, reduction=2)

        # third block
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire6 = Fire(256, 48, 192, 192, bn=True, bn_d=bn_d)
        self.fire7 = Fire(384, 48, 192, 192, bn=True, bn_d=bn_d)
        self.fire8 = Fire(384, 64, 256, 256, bn=True, bn_d=bn_d)
        self.fire9 = Fire(512, 64, 256, 256, bn=True, bn_d=bn_d)
        self.se3 = SELayer(512, reduction=2)

        self.aspp = ASPP(512, [6, 9, 12])

    def forward(self, x):
        x_1a = self.conv1a(x)  # (H, W/2)
        x_1b = self.conv1b(x)

        ### Encoder forward
        # first fire block
        x_p1 = self.pool1(x_1a)
        x_f2 = self.fire2(x_p1)
        x_f3 = self.fire3(x_f2)
        x_se1 = self.se1(x_f3)
        if self.bypass:
            x_se1 += x_f2

        # second fire block
        x_p2 = self.pool2(x_se1)
        x_f4 = self.fire4(x_p2)
        x_f5 = self.fire5(x_f4)
        x_se2 = self.se2(x_f5)
        if self.bypass:
            x_se2 += x_f4

        # third fire block
        x_p3 = self.pool3(x_se2)
        x_f6 = self.fire6(x_p3)
        x_f7 = self.fire7(x_f6)
        if self.bypass:
            x_f7 += x_f6
        x_f8 = self.fire8(x_f7)
        x_f9 = self.fire9(x_f8)
        x_se3 = self.se3(x_f9)
        if self.bypass:
            x_se3 += x_f8

        # EL forward
        x_el = self.aspp(x_se3)
        return x_1a, x_1b, x_se1, x_se2, x_se3, x_el


class PSDecoder(BaseNet):
    def __init__(self, cfg):
        super(PSDecoder, self).__init__()
        bn_d = 0.1
        num_classes = len(cfg['classes'])
        self.p = cfg['dropout']

        self.fdeconv_el = FireDeconv(128, 32, 128, 128, bn=True, bn_d=bn_d)

        self.fdeconv_1 = FireDeconv(512, 64, 128, 128, bn=True, bn_d=bn_d)
        self.fdeconv_2 = FireDeconv(512, 64, 64, 64, bn=True, bn_d=bn_d)
        self.fdeconv_3 = FireDeconv(128, 16, 32, 32, bn=True, bn_d=bn_d)
        self.fdeconv_4 = FireDeconv(64, 16, 32, 32, bn=True, bn_d=bn_d)

        self.drop = nn.Dropout2d(p=self.p)
        self.conv2 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_1a, x_1b, x_se1, x_se2, x_se3, x_el = x
        x_el = self.fdeconv_el(x_el)

        ### Decoder forward
        x_fd1 = self.fdeconv_1(x_se3)  # (H, W/8)
        x_fd1_fused = torch.add(x_fd1, x_se2)
        x_fd1_fused = torch.cat((x_fd1_fused, x_el), dim=1)

        x_fd2 = self.fdeconv_2(x_fd1_fused)  # (H, W/4)
        x_fd2_fused = torch.add(x_fd2, x_se1)

        x_fd3 = self.fdeconv_3(x_fd2_fused)  # (H, W/2)
        x_fd3_fused = torch.add(x_fd3, x_1a)

        x_fd4 = self.fdeconv_4(x_fd3_fused)  # (H, W/2)
        x_fd4_fused = torch.add(x_fd4, x_1b)

        x_d = self.drop(x_fd4_fused)
        x = self.conv2(x_d)
        return x


class PointSegNet(BaseNet):
    def __init__(self, cfg):
        super(PointSegNet, self).__init__()

        ### Ecnoder part
        self.feat_encoder = PSEncoder(cfg)

        ### Decoder part
        self.feat_decoder = PSDecoder(cfg)

    def forward(self, x):
        x = self.feat_encoder(x)
        x = self.feat_decoder(x)
        return x

    def get_modules(self):
        return [self.feat_encoder, self.feat_decoder]



