import os
import datetime
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .modules import Conv2dBlk, Fire, FireDeconv, SELayer, ASPP


class PointSegNet(nn.Module):
    def __init__(self, cfg, bypass=False):
        super(PointSegNet, self).__init__()
        self.bypass = bypass


        num_classes = len(cfg['classes'])
        self.input_shape = cfg['input-shape']
        h, w, c = self.input_shape

        # Ecnoder part
        self.conv1a = Conv2dBlk(c, out_channels=64, kernel_size=3, stride=(1, 2), padding=1, bn=False)
        self.conv1b = Conv2dBlk(c, out_channels=64, kernel_size=1, stride=(1, 1), bn=False)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.fire2 = Fire(64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.sr1 = SELayer(128, ratio=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.sr2 = SELayer(256, ratio=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)
        self.sr3 = SELayer(512, ratio=2)
        #self.aspp = ASPP(512, [6, 9, 12])

        # Decoder part
        #self.fdeconv_el = FireDeconv(128, 32, 128, 128)
        self.fdeconv_1 = FireDeconv(512, 64, 128, 128)
        self.fdeconv_2 = FireDeconv(256, 64, 64, 64)
        self.fdeconv_3 = FireDeconv(128, 16, 32, 32)
        self.fdeconv_4 = FireDeconv(64, 16, 32, 32)
        self.drop = nn.Dropout2d(p=0.25)
        self.conv2 = Conv2dBlk(64, num_classes, 3, (1, 1), 1, activation=None, bn=True)

    def forward(self, x):
        x_in = copy.deepcopy(x)
        x_1a = self.conv1a(x) # (H, W/2)
        x_1b = self.conv1b(x)
        x = self.pool1(x_1a)  # (H, W/4)

        ### Encoder forward
        # first fire block
        x_f2 = self.fire2(x)
        x_f3 = self.fire3(x_f2)
        if self.bypass:
            x = torch.add(x_f2, x_f3)
        else:
            x = x_f3

        x_sr1 = self.sr1(x)
        x_p2 = self.pool2(x_sr1) # (H, W/8)

        # second fire block
        x_f4 = self.fire4(x_p2)
        x_f5 = self.fire5(x_f4)
        if self.bypass:
            x = torch.add(x_f4, x_f5)
        else:
            x = x_f5

        x_sr2 = self.sr2(x)
        x_p3 = self.pool3(x_sr2) # (H, W/16)

        x = self.fire6(x_p3)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)

        ### Decoder forward
        x_sr3 = self.sr3(x)
        #x_el = self.aspp(x_sr3)
        #x_fd_el = self.fdeconv_el(x_el)  # (H, W/8)

        x_fd1 = self.fdeconv_1(x_sr3)  # (H, W/8)

        x_fused = torch.add(x_fd1, x_sr2)
        #x_fused = torch.cat((x_fused, x_fd_el), dim=1)

        x_fd2 = self.fdeconv_2(x_fused)  # (H, W/4)
        x_fused = torch.add(x_fd2, x_sr1)

        x_fd3 = self.fdeconv_3(x_fused)  # (H, W/2)
        x_fused = torch.add(x_fd3, x_1a)

        x_fd4 = self.fdeconv_4(x_fused)  # (H, W/2)
        x_fused = torch.add(x_fd4, x_1b)

        x_d = self.drop(x_fused)
        x = self.conv2(x_d)

        plot_module(x_in[0, -1].detach().cpu().numpy(),
                    x_sr3[0, -1].detach().cpu().numpy(),
                    x_fd1[0, -1].detach().cpu().numpy(),
                    x_fd2[0, -1].detach().cpu().numpy(),
                    x_fd3[0, -1].detach().cpu().numpy(),
                    x_fd4[0, -1].detach().cpu().numpy(),
                    x_d[0, -1].detach().cpu().numpy(),
                    x[0].detach().cpu().numpy())
        return x

    @property
    def name(self):
        return self.__class__.__name__.lower()

index = 0
dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
plt.ioff()
def plot_module(x1, x2, x3, x4, x5, x6, x7, x8):
    global index

    img_path = Path("{}/images".format(content_dir))
    img_path.mkdir(parents=True, exist_ok=True)

    # output
    x8 = np.argmax(x8, axis=0)
    x8c = np.zeros((x8.shape[0], x8.shape[1], 3))
    colors = [[0., 0., 0.],
              [0.9, 0., 0.],
              [0., 0.9, 0.],
              [0., 0., 0.9]]
    for j in range(4):
        x8c[x8 == j, :] = colors[j]

    X = [x1, x2, x3, x4, x5, x6, x7, x8c]


    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    axes = ax.flatten()
    for i in range(len(X)):
        x = X[i]
        min_ = x.min()
        max_ = x.max()
        if i != (len(X) - 1):
            x = (x - min_) / (max_ - min_ + 1e-6)
        axes[i].imshow(x)
        axes[i].set_title("{}, {:.5f}-{:.5f}".format(str(i), min_, max_))
    im_path = "{}/{}_{}.png".format(str(img_path), index, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    fig.savefig(im_path, bbox_inches='tight')
    index += 1

