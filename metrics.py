import torch


def _confusion(pred, target):
    # pred, target: H, W binary
    pred = pred.view(-1)
    target = target.view(-1)
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    return tp, fp, tn, fn


def binary_metrics(pred, target):
    tp, fp, tn, fn = _confusion(pred, target)
    iou_pos = tp / max(tp + fp + fn, 1)
    iou_neg = tn / max(tn + fp + fn, 1)
    miou = (iou_pos + iou_neg) / 2
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "miou": miou,
        "precision": precision,
        "recall": recall,
    }
