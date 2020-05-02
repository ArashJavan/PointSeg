import numpy as np

import torch


def img_normalize(x):
    return ((x - torch.min(x)) / (torch.max(x) - torch.min(x))).view(
                1, x.size()[0], x.size()[1]
            )

def visualize_seg(label_map, cfg, one_hot=False):
    cmap = np.array(cfg['class-cmap'])
    n_cls = len(cfg['classes'])

    if one_hot:
        label_map = torch.argmax(label_map, axis=-1)

    out = torch.zeros([label_map.size(0), label_map.size(1), label_map.size(2), 3])

    for l in range(1, n_cls):
        out[label_map == l, :] = torch.from_numpy(cmap[l]).type(torch.float32)

    out = out.permute(0, 3, 1, 2)
    return out


def cal_eval_metrics(label, pred, n_class):
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

    #label = label.cpu().numpy()
    #pred = pred.cpu().numpy()

    tp = torch.zeros(n_class) # np.zeros(n_class)
    fn = torch.zeros(n_class)
    fp = torch.zeros(n_class)

    for cls_id in range(n_class):
        tp_cls = torch.sum(pred[label == cls_id] == cls_id)
        fp_cls = torch.sum(label[pred == cls_id] != cls_id)
        fn_cls = torch.sum(pred[label == cls_id] != cls_id)

        tp[cls_id] = tp_cls
        fp[cls_id] = fp_cls
        fn[cls_id] = fn_cls

    return tp, fp, fn
