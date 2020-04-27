import os
import argparse
import random
import warnings
import yaml
import time
import shutil
import datetime

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from pytorch_model_summary import summary

from pointseg.pointseg_net import PointSegNet
from pointseg.kitti_squeezeseg_ds import KittiSqueezeSegDS
from pointseg.loss import WeightedCrossEntropy, evaluate
from pointseg.utils import img_normalize, visualize_seg


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
    mean = cfg['mean']
    std = cfg['std']

    model = PointSegNet(cfg, bypass=False)
    model.to(args.device)

    inputs = torch.randn((1, input_shape[2], input_shape[0], input_shape[1])).to(args.device)
    print(summary(model, inputs))

    criterion = WeightedCrossEntropy(cfg)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location=args.device)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.to(args.device)
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

    if args.evaluate:
        validate(val_dataloader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_dataloader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc = validate(val_dataloader, model, criterion, args)

        # evaluate on validation set
        is_best = acc > best_acc1
        best_acc1 = max(acc, best_acc1)

        fname = os.path.join(content_dir, "../{}_{}.tar".format(epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, mask, labels, weights = data
        inputs, mask, labels, weights = \
                inputs.to(args.device), mask.to(args.device), labels.to(args.device), weights.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, labels, mask, weights)

        _, preds = torch.max(outputs.detach().data, 1)

        # Record loss
        losses.update(loss.item(), inputs.size(0))
        writer.add_scalar("Train/Loss", loss.detach().cpu() / inputs.size(0), i * epoch)

        # compute gradient and step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            # TensoorboardX Save Input Image and Visualized Segmentation
            writer.add_image('Input/Image/', (img_normalize(inputs[0, 3, :, :])).cpu(), i * epoch)

            writer.add_image('Predict/Image/', visualize_seg(preds, cfg)[0], i * epoch)

            writer.add_image('Target/Image/', visualize_seg(labels, cfg)[0], i * epoch)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    iou_loss = AverageMeter('iou', ':.4e')
    precision = AverageMeter('prec', ':.4e')
    recall = AverageMeter('recall', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, iou_loss, precision, recall],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, mask, labels, weights = data
            inputs, mask, labels, weights = \
                inputs.to(args.device), mask.to(args.device), labels.to(args.device), weights.to(args.device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels, mask, weights)

            # measure accuracy and record loss
            _, predicted = torch.max(outputs.data, 1)

            n_classes = inputs.size(1)
            tp, fp, fn = evaluate(labels, predicted, n_classes)

            iou = tp / (tp + fn + fp + 1e-12)
            prec = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)

            losses.update(loss.item(), inputs.size(0))
            iou.update(iou, inputs.size(0))
            prec.update(prec, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    return iou.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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



best_acc1 = 0
args = None
writer = None
cfg = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch PointSeg Training')

    parser.add_argument('--csv-path',  type=str, help='path to csv file', metavar='PATH')
    parser.add_argument('--data-path', type=str, help='path to lidar data', metavar='PATH')
    parser.add_argument('-c', '--config', default='../config.yaml', type=str, help='path to config file')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
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
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--device', default='cpu', type=str, metavar='DEVICE',
                        help='Device to use [cpu, cuda].')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    main(args)


