import torch
import numpy as np
import scipy.stats


def spearman_rank_correlation(pred, target):
    assert pred.dim() == 2 and target.dim() == 2
    assert pred.size(1) == target.size(1)

    pred = pred.detach().cpu().numpy()
    target = target.cpu().numpy()

    corr_sum = 0
    cnt = 0
    for i in range(len(pred)):
        corr = scipy.stats.spearmanr(pred[i], target[i])[0]
        if not np.isnan(corr):
            corr_sum += corr
            cnt += 1
    if cnt > 0:
        return corr_sum / cnt
    return 0


def top_k_recall(pred, target, k=20):
    assert pred.dim() == 2 and target.dim() == 2
    assert pred.size(1) == target.size(1)

    pred = pred.detach().cpu().numpy()
    target = target.cpu().numpy()

    recall_sum = 0
    cnt = 0
    for i in range(len(pred)):
        thresh = sorted(list(pred[i]), reverse=True)[k - 1]
        pred_idx = [idx for idx, val in enumerate(pred[i]) if val >= thresh]
        thresh = sorted(list(target[i]), reverse=True)[k - 1]
        target_idx = [idx for idx, val in enumerate(target[i]) if val >= thresh]
        tp = len(set(pred_idx).intersection(set(target_idx)))
        recall = tp / min(len(pred_idx), len(target_idx))
        if not np.isnan(recall):
            recall_sum += recall
            cnt += 1
    if cnt > 0:
        return recall_sum / cnt
    return 0