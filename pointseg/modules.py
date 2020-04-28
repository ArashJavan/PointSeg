import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=128):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.25))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Conv2d(nn.Module):
    """A 2D Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation='relu', bn=True, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.activation = None
        if activation:
            if activation.lower() == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation.lower() == 'elu':
                alpha = kwargs.get('alpha', 1.)
                self.activation = nn.ELU(alpha, alpha, inplace=True)
            elif activation.lower() == 'leakrelu':
                negative_slope = kwargs.get('negative_slope', 0.01)
                self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)

        if self.bn:
            x = self.bn(x)

        if self.activation:
            x = self.activation(x)
        return x


class Deconv2d( nn.Module ):
    """ A 2D Deconvolution Block"""
    def __init__( self, inputs, outputs, kernel_size, stride, padding=0, activation='relu', **kwargs):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel_size, stride=stride, padding=padding)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'elu':
            alpha = kwargs.get('alpha', 1.)
            self.activation = nn.ELU(alpha, alpha, inplace=True)
        elif activation.lower() == 'leakrelu':
            negative_slope = kwargs.get('negative_slope', 0.01)
            self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.deconv(x)
        if self.activation:
            x = self.activation(x)
        return x


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.squeeze = Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.squeeze(x)
        x_1x1 = self.expand1x1(x)
        x_3x3 = self.expand3x3(x)
        out = torch.cat([x_1x1, x_3x3], 1)
        return out


class FireDeconv(nn.Module):
    """Fire deconvolution layer constructor.
    Args:
      inputs: input channels
      squeeze_planes: number of 1x1 filters in squeeze layer.
      expand1x1_planes: number of 1x1 filters in expand layer.
      expand3x3_planes: number of 3x3 filters in expand layer.
      factors: spatial upsampling factors.[1,2]
    Returns:
      fire deconv layer operation.
    """

    def __init__(self, in_channels,  squeeze_planes, expand1x1_planes, expand3x3_planes, factors=[1, 2], padding=(0, 1)):
        super(FireDeconv, self).__init__()
        ksize_h = factors[0] * 2 - factors[0] % 2
        ksize_w = factors[1] * 2 - factors[1] % 2

        self.squeeze = Conv2d(in_channels, squeeze_planes, 1, (1, 1), 0)
        self.squeeze_deconv = Deconv2d(squeeze_planes, squeeze_planes, (ksize_h, ksize_w),
                                                 (factors[0], factors[1]), padding)

        self.expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, 1, (1, 1), 0)
        self.expand3x3 = Conv2d(squeeze_planes, expand3x3_planes, 3, (1, 1), 1)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_deconv(x)
        x_1x1 = self.expand1x1(x)
        x_3x3 = self.expand3x3(x)

        out = torch.cat([x_1x1, x_3x3], 1)
        return out


class SELayer(nn.Module):
    """Squeeze and Excitation layer from SEnet
    """
    def __init__(self, in_features, ratio=16):
        super(SELayer, self).__init__()

        h_features = int(in_features / ratio)
        self.fc1 = nn.Linear(in_features, h_features)
        self.fc2 = nn.Linear(h_features, in_features)

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1).squeeze()
        s = self.fc1(z)
        s = torch.relu(s)
        s = self.fc2(s)
        s = torch.sigmoid(s)
        s = s.view(-1, z.size()[-1], 1,1)
        x_scaled = x * s
        return x_scaled
