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
          Conv2d-2      [1, 64, 64, 512]             384             384
       MaxPool2d-3      [1, 64, 64, 128]               0               0
            Fire-4     [1, 128, 64, 128]          11,408          11,408
            Fire-5     [1, 128, 64, 128]          12,432          12,432
         SELayer-6     [1, 128, 64, 128]          16,576          16,576
       MaxPool2d-7      [1, 128, 64, 64]               0               0
            Fire-8      [1, 256, 64, 64]          45,344          45,344
            Fire-9      [1, 256, 64, 64]          49,440          49,440
        SELayer-10      [1, 256, 64, 64]          65,920          65,920
      MaxPool2d-11      [1, 256, 64, 32]               0               0
           Fire-12      [1, 384, 64, 32]         104,880         104,880
           Fire-13      [1, 384, 64, 32]         111,024         111,024
           Fire-14      [1, 512, 64, 32]         188,992         188,992
           Fire-15      [1, 512, 64, 32]         197,184         197,184
        SELayer-16      [1, 512, 64, 32]         262,912         262,912
           ASPP-17      [1, 128, 64, 32]       1,984,000       1,984,000
     FireDeconv-18      [1, 256, 64, 64]          49,472          49,472
     FireDeconv-19      [1, 256, 64, 64]         131,456         131,456
     FireDeconv-20     [1, 128, 64, 128]          90,368          90,368
     FireDeconv-21      [1, 64, 64, 256]           8,288           8,288
     FireDeconv-22      [1, 64, 64, 512]           7,264           7,264
      Dropout2d-23      [1, 64, 64, 512]               0               0
         Conv2d-24       [1, 4, 64, 512]           2,316           2,316
=========================================================================
Total params: 3,342,604
Trainable params: 3,342,604
Non-trainable params: 0
```