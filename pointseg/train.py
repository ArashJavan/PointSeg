import os
import sys
import argparse
import random
import warnings
import yaml
import time
import shutil
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim
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

#from pytorch_model_summary import summary

VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION_PATCH = 0


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args)


def main_worker(args):
    global best_acc, writer, cfg

    if args.device == "gpu":
        print("Use GPU for training")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dname = os.path.abspath(os.path.dirname(__file__))
    content_dir = os.path.abspath("{}/..".format(dname))

    input_shape = cfg['input-shape']
    cls_weights = torch.tensor(cfg['class-weights'])
    mean = cfg['mean']
    std = cfg['std']

    model = PointSegNet(cfg)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(weight=cls_weights).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location=args.device)

            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    trans = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean, std)))

    train_dataset = KittiSqueezeSegDS(cfg, args.data_path, args.csv_path, transform=trans, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.workers, shuffle=True)

    val_dataset = KittiSqueezeSegDS(cfg, args.data_path, args.csv_path, transform=trans, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                   num_workers=args.workers, shuffle=False)

    writer = SummaryWriter()
    #input = torch.rand((1, input_shape[2], input_shape[0], input_shape[1])).to(args.device)
    #print(summary(model, input))
    #writer.add_graph(model, input)

    print("Starting PointSeg v{}.{}.{}".format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH))

    if args.evaluate:
        validate(val_dataloader, model, criterion, 0, args, cfg)
        return

    print("Starting trainig")
    print("lr: {}, batch-size: {}, num-workers: {}".format(args.lr, args.batch_size, args.workers))

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_dataloader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc = validate(val_dataloader, model, criterion, epoch, args, cfg)

        # evaluate on validation set
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        chk_path = Path(os.path.join(content_dir, "checkpoints"))
        chk_path.mkdir(parents=True, exist_ok=True)
        fname = "{}/checkpoint_{}_{}.tar".format(str(chk_path), epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, fname)

        lr_scheduler.step()

        # recore learning rate changes
        writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], epoch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # empty the cache for training
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, mask, labels, weights, _ = data
        inputs = inputs[:, 3:, :, :].contiguous()

        inputs, mask, labels, weights = \
                inputs.to(args.device), mask.to(args.device), labels.to(args.device), weights.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs.clamp(min=1e-8), labels) # criterion(outputs, labels, mask, weights)


        _, preds = torch.max(outputs.detach().data, 1)

        # Record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            step_val = epoch * len(train_loader) + i
            writer.add_scalar("Train/Loss", losses.avg, step_val)

            # Tensoorboard Save Input Image and Visualized Segmentation
            writer.add_image('Input/Image/', (img_normalize(inputs[0, 0, :, :])).cpu(), step_val)
            writer.add_image('Predict/Image/', visualize_seg(preds, cfg)[0], step_val)
            writer.add_image('Target/Image/', visualize_seg(labels, cfg)[0], step_val)


def validate(val_loader, model, criterion, epoch, args, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    n_classes = len(cfg['classes'])
    total_tp = torch.zeros(n_classes)
    total_fp = torch.zeros(n_classes)
    total_fn = torch.zeros(n_classes)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader, 1):
            inputs, mask, labels, weights, _ = data
            inputs, mask, labels, weights = \
                inputs.to(args.device), mask.to(args.device), labels.to(args.device), weights.to(args.device)

            # compute output
            outputs = model(inputs)
            loss = criterion(torch.log(outputs.clamp(min=1e-8)), labels)

            # measure accuracy and record loss
            _, predicted = torch.max(outputs.data, 1)

            n_classes = outputs.size(1)
            tp, fp, fn = cal_eval_metrics(labels, predicted, n_classes)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            losses.update(loss.item(), inputs.size(0))

            # Record loss
            step_val = epoch * len(val_loader) + i
            writer.add_scalar("Val/Loss", losses.avg, step_val)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    iou = total_tp / (total_tp + total_fn + total_fp + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    print("Averege over whole validation set:")
    print("IOU-object: {:.3}, Precision-object: {:.3}, Recall-object: {:.3}".format(iou[1], precision[1], recall[1]))

    return iou.mean()


def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.tar'.format(os.path.dirname(filename)))


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

    parser.add_argument('--csv-path', required=True, type=str, help='path to csv file', metavar='PATH')
    parser.add_argument('--data-path', required=True, type=str, help='path to lidar data', metavar='PATH')
    parser.add_argument('-c', '--config', default='../config.yaml', type=str, help='path to config file')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--version', action='version', version='{}.{}.{}'.format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH))

    args = parser.parse_args()
    main(args)
