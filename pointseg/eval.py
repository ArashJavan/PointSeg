import os
import sys
import argparse
import yaml
import time
import shutil
import datetime
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
sys.path.append(dname)
sys.path.append(content_dir)

from pointseg.pointseg_net import PointSegNet
from pointseg.kitti_squeezeseg_ds import KittiSqueezeSegDS
from pointseg.utils import img_normalize, visualize_seg, cal_eval_metrics
from pointseg.train import VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH


def main(args):
    main_worker(args)


def main_worker(args):
    global writer, cfg

    if args.device == "gpu":
        print("Use GPU for training")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cls_weights = torch.tensor(cfg['class-weights'])
    mean = cfg['mean']
    std = cfg['std']

    model = PointSegNet(cfg)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(weight=cls_weights).to(args.device)

    # load checkpoint
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))

        checkpoint = torch.load(args.model, map_location=args.device)

        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        return

    cudnn.benchmark = True

    trans = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean, std)))

    dataset = KittiSqueezeSegDS(cfg, args.data_path, args.csv_path, transform=trans, mode=args.ds_type)
    dataset.data_augmentation = False
    dataset.random_flipping = False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             num_workers=args.workers, shuffle=False)

    writer = SummaryWriter()

    print("Starting PointSeg v{}.{}.{}".format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH))
    print("Starting test")
    print("batch-size: {}, num-workers: {}".format(args.batch_size, args.workers))
    test(dataloader, model, criterion, args)

        # evaluate on validatio

def test(dataloader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Test:")

    # empty the cache for training
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    n_classes = len(cfg['classes'])
    total_tp = torch.zeros(n_classes)
    total_fp = torch.zeros(n_classes)
    total_fn = torch.zeros(n_classes)

    if args.save:
        save_dir = "{}/test-preds/".format(content_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # switch to train mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(dataloader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, mask, labels, weights, xyz = data
            inputs, mask, labels, weights = \
                    inputs.to(args.device), mask.to(args.device), labels.to(args.device), weights.to(args.device)

            outputs = model(inputs)
            loss = criterion(torch.log(outputs.clamp(min=1e-8)), labels) # criterion(outputs, labels, mask, weights)

            # measure accuracy and record loss
            _, preds = torch.max(outputs.data, 1)


            n_classes = outputs.size(1)
            tp, fp, fn = cal_eval_metrics(labels, preds, n_classes)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Record loss
            losses.update(loss.item(), inputs.size(0))
            writer.add_scalar("Test/Loss", losses.avg, i)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

                writer.add_scalar("Test/Loss", losses.avg, i)

                # Tensoorboard Save Input Image and Visualized Segmentation
                writer.add_image('Input/Image/', (img_normalize(inputs[0, 3, :, :])).cpu(), i)
                writer.add_image('Predict/Image/', visualize_seg(preds, cfg)[0], i)
                writer.add_image('Target/Image/', visualize_seg(labels, cfg)[0], i)

            # add the predictions to input channels and save them
            if args.save:
                for j in range(args.batch_size):
                    in_channel = xyz[j].detach().cpu().numpy()
                    preds_j = preds[j].detach().cpu().numpy()
                    labels_gt = labels[j].detach().cpu().numpy()
                    frame = np.dstack((in_channel, labels_gt[..., None], preds_j[..., None]))
                    np.save("{}/{}.npy".format(save_dir, i*args.batch_size+j), frame)

    iou = total_tp / (total_tp + total_fn + total_fp + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    print("Averege over whole validation set:")
    print("IOU-car: {:.3}, Precision-car: {:.3}, Recall-car: {:.3}".format(iou[1], precision[1], recall[1]))
    print("IOU-ped: {:.3}, Precision-ped: {:.3}, Recall-ped: {:.3}".format(iou[2], precision[2], recall[2]))
    print("IOU-cyc: {:.3}, Precision-cyc: {:.3}, Recall-cyc: {:.3}".format(iou[3], precision[3], recall[3]))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


best_acc = 0
args = None
writer = None
cfg = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch PointSeg Training')

    parser.add_argument('--csv-path',  required=True, type=str, help='path to csv file', metavar='PATH')
    parser.add_argument('--data-path', required=True, type=str, help='path to lidar data', metavar='PATH')
    parser.add_argument('-c', '--config', default='../config.yaml', type=str, help='path to config file')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-m', '--model', required=True, type=str, metavar='PATH',
                        help='path to the model checkpoint (default: none)')
    parser.add_argument('--ds-type', default='train', type=str, metavar='train, test, val',
                        help='Type of dataset for test (default: train)')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')
    parser.add_argument('--version', action='version', version='{}.{}.{}'.format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH))
    parser.add_argument('-s', '--save', default=True, dest='save', action='store_true',
                        help='add label predictions to lidar input and save them for reconstruction.')

    args = parser.parse_args()
    main(args)
