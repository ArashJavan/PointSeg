# PointSeg
This repository is the __pytorch__ implementation of [PointSeg](https://arxiv.org/abs/1807.06288)

```
@article{Wang2018PointSegRS,
  title={PointSeg: Real-Time Semantic Segmentation Based on 3D LiDAR Point Cloud},
  author={Yuan Wang and Tianyue Shi and Peng Yun and Lei Tai and Ming Liu},
  journal={ArXiv},
  year={2018},
  volume={abs/1807.06288}
}
```

## Dependencies
- Pytorch 1.4 >

## Dataset
The dataset used for training pointsge is the same as [squeezeseg](https://github.com/xuanyuzhou98/SqueezeSegV2), which can be downloaded from 
[here](https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz?dl=0).

## Usage 
Example:

> ~ cd PointSeg
>
> ~ python --csv-path ImageSet/csv/ --data-path /path/to/Datasets/lidar_2d/ -c ./config.yaml -b 32 --lr 0.001

## Netwrok Architecture


```
-------------------------------------------------------------------------
      Layer (type)          Output Shape         Param #     Tr. Param #
=========================================================================
          Conv2d-1      [1, 64, 64, 256]           2,944           2,944
     BatchNorm2d-2      [1, 64, 64, 256]             128             128
            ReLU-3      [1, 64, 64, 256]               0               0
          Conv2d-4      [1, 64, 64, 512]             384             384
     BatchNorm2d-5      [1, 64, 64, 512]             128             128
            ReLU-6      [1, 64, 64, 512]               0               0
       MaxPool2d-7      [1, 64, 64, 128]               0               0
            Fire-8     [1, 128, 64, 128]          11,696          11,696
            Fire-9     [1, 128, 64, 128]          12,720          12,720
        SELayer-10     [1, 128, 64, 128]          16,384          16,384
      MaxPool2d-11      [1, 128, 64, 64]               0               0
           Fire-12      [1, 256, 64, 64]          45,920          45,920
           Fire-13      [1, 256, 64, 64]          50,016          50,016
        SELayer-14      [1, 256, 64, 64]          65,536          65,536
      MaxPool2d-15      [1, 256, 64, 32]               0               0
           Fire-16      [1, 384, 64, 32]         105,744         105,744
           Fire-17      [1, 384, 64, 32]         111,888         111,888
           Fire-18      [1, 512, 64, 32]         190,144         190,144
           Fire-19      [1, 512, 64, 32]         198,336         198,336
        SELayer-20      [1, 512, 64, 32]         262,144         262,144
           ASPP-21      [1, 128, 64, 32]       1,984,000       1,984,000
     FireDeconv-22      [1, 256, 64, 64]          50,048          50,048
     FireDeconv-23      [1, 256, 64, 64]         132,096         132,096
     FireDeconv-24     [1, 128, 64, 128]          90,752          90,752
     FireDeconv-25      [1, 64, 64, 256]           8,448           8,448
     FireDeconv-26      [1, 64, 64, 512]           7,424           7,424
      Dropout2d-27      [1, 64, 64, 512]               0               0
         Conv2d-28       [1, 4, 64, 512]           2,308           2,308
=========================================================================
Total params: 3,349,188
Trainable params: 3,349,188
Non-trainable params: 0
```