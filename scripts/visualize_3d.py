import os
import sys
import argparse
import glob
import yaml
import time
import shutil
import datetime
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import open3d as o3d

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)


def visualize_seg(label_map):
    cmap = np.array([[0., 0., 0.],
                     [0., 0., 0.9],
                     [0., 0.9, 0.],
                     [0.9, 0., 0.]])
    n_cls = 4
    out = np.zeros([label_map.shape[0], label_map.shape[1], 3])

    for l in range(1, n_cls):
        out[label_map == l, :] = cmap[l]
    return out


def custom_draw_geometry_load_option(pcd, i, path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_renderer()
    vis.poll_events()
    vis.capture_screen_image("{}/pcd-{}.png".format(path, i))
    vis.destroy_window()


def main(args):
    files = None
    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = glob.glob("{}/*.npy".format(args.path))

    if files is None:
        print("Error: no file found!")

    for i, file in enumerate(files):
        dir_path = os.path.dirname(file)

        frame = np.load(file)
        preds = frame[:, :, -1]
        labels_gt = frame[:, :, -2]

        labels = preds.reshape(-1)
        colros = np.zeros((len(labels), 3))
        colros[labels == 0] = np.array([0.5, 0.5, 0.5])
        colros[labels == 1] = np.array([1., 0., 0.])
        colros[labels == 2] = np.array([0., 1., 0.])
        colros[labels == 3] = np.array([0., 0., 1.])

        alpha = 0.5
        im_range = np.linalg.norm(frame[:, :, :3], axis=2)
        im_range = (im_range - im_range.min()) / (im_range.max() - im_range.min())

        img_pred = visualize_seg(preds)
        img_pred = alpha * im_range[..., None] + (1-alpha) * img_pred

        img_gt = visualize_seg(labels_gt)
        img_gt = alpha * im_range[..., None] + (1 - alpha) * img_gt

        plt.subplot(311)
        plt.imshow(im_range, cmap='gray')
        plt.title('Range Image')
        plt.subplot(312)
        plt.imshow(img_pred)
        plt.title('predicted labels')
        plt.subplot(313)
        plt.imshow(img_gt)
        plt.title('ground-truth')
        plt.savefig("{}/{}.png".format(dir_path, i))

        xyz = frame[:, :, :3].reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colros)

        custom_draw_geometry_load_option(pcd, i, dir_path)

        #o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch PointSeg Training')

    parser.add_argument('-p', '--path',  required=True, type=str, help='path to files or a file to visualize', metavar='PATH')

    args = parser.parse_args()
    main(args)

