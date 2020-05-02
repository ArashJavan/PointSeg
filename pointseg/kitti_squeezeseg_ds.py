""" Kitti Dataset for SqueezeSeg """
import os

import numpy as np
import torch

from torch.utils.data import Dataset


class KittiSqueezeSegDS(Dataset):
    def __init__(self, cfg, root_path, csv_path, transform=None, mode='train'):
        csv_root = csv_path
        root_dir = root_path

        self.data_augmentation = cfg['data-augmentation']
        self.random_flipping = cfg['random-flipping']
        self.classes = cfg['classes']
        self.class_weights = cfg['class-weights']
        self.num_classes = len(self.classes)

        csv_file = os.path.join(csv_root, '{}.csv'.format(mode))
        self.lidar_2d_csv = np.genfromtxt(csv_file, skip_header=1, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lidar_2d_csv)

    def __getitem__(self, idx):

        lidar_name = os.path.join(self.root_dir, self.lidar_2d_csv[idx])
        lidar_data = np.load(lidar_name).astype(np.float32)

        if self.data_augmentation:
            if self.random_flipping:
                if np.random.rand() > 0.5:
                    # flip y
                    lidar_data = lidar_data[:, ::-1, :]
                    lidar_data[:, :, 1] *= -1

        lidar_mask = (lidar_data[:, :, 4] > 0) * 1

        lidar_label = lidar_data[:, :, 5]
        weights = np.zeros(lidar_label.shape)
        for l in range(self.num_classes):
            weights[lidar_label == l] = self.class_weights[int(l)]

        # x, y, z, intensity, range, label
        lidar_all_channels = np.dstack((lidar_data, lidar_mask, weights))
        if self.transform:
            lidar_all_channels = self.transform(lidar_all_channels).float()

        lidar_inputs = lidar_all_channels[:5]
        lidar_label = lidar_all_channels[5].long()
        lidar_mask = lidar_all_channels[6]
        lidar_wights = lidar_all_channels[7]

        xyz = np.ascontiguousarray(lidar_data[:, :, :3])

        return (lidar_inputs,
                lidar_mask,
                lidar_label,
                lidar_wights,
                xyz)
