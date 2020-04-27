import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def evaluate(label, pred, n_class):
    """Evaluation script to compute pixel level IoU.
    Args:
        label: N-d array of shape [batch, W, H], where each element is a class index.
        pred: N-d array of shape [batch, W, H], the each element is the predicted class index.
        n_class: number of classes
        epsilon: a small value to prevent division by 0
    Returns:
        IoU: array of lengh n_class, where each element is the average IoU for this class.
        tps: same shape as IoU, where each element is the number of TP for each class.
        fps: same shape as IoU, where each element is the number of FP for each class.
        fns: same shape as IoU, where each element is the number of FN for each class.
    """

    assert label.shape == pred.shape, \
        'label and pred shape mismatch: {} vs {}'.format(label.shape, pred.shape)

    label = label.cpu().numpy()
    pred = pred.cpu().numpy()

    tp = np.zeros(n_class)
    fn = np.zeros(n_class)
    fp = np.zeros(n_class)

    for cls_id in range(n_class):
        tp_cls = np.sum(pred[label == cls_id] == cls_id)
        fp_cls = np.sum(label[pred == cls_id] != cls_id)
        fn_cls = np.sum(pred[label == cls_id] != cls_id)

        tp[cls_id] = tp_cls
        fp[cls_id] = fp_cls
        fn[cls_id] = fn_cls

    return tp, fp, fn



class WeightedCrossEntropy(nn.Module):
    def __init__(self, cfg):
        super(WeightedCrossEntropy, self).__init__()
        self.cls_loss_coef = cfg['class-loss-coef']
        self.num_cls = len(cfg['classes'])

    def forward(self, outputs, targets, lidar_mask, loss_weight):
        loss = F.cross_entropy(outputs.view(-1, self.num_cls), targets.view(-1,))
        loss = lidar_mask.view(-1,) * loss
        loss = loss_weight.view(-1,) * loss
        loss = torch.sum(loss) / torch.sum(lidar_mask)
        return loss * self.cls_loss_coef