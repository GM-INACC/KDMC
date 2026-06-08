import copy
import numpy as np
import pandas as pd


class MetricsDAG(object):

    def __init__(self, B_est, B_true):
        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)

    @staticmethod
    def _count_accuracy(B_est, B_true, decimal_num=4):
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0

        if (B_est == -1).any():
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')

        d = B_true.shape[0]

        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.unique(np.concatenate([cond, cond_reversed]))

        true_pos_dir = np.intersect1d(pred, cond)
        false_pos_dir = np.setdiff1d(pred, cond_skeleton)

        false_pos_und = np.setdiff1d(pred_und, cond_skeleton)
        false_pos = np.unique(np.concatenate([false_pos_dir, false_pos_und]))

        extra = np.setdiff1d(pred, cond)
        reverse = np.intersect1d(extra, cond_reversed)

        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)

        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos_dir)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        nhd = float(shd) / max(float(d * (d - 1)), 1.0)

        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {
            'fdr': fdr, 'tpr': tpr, 'fpr': fpr,
            'shd': shd, 'nhd': nhd, 'nnz': pred_size,
            'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore
        }
        for k in mt:
            mt[k] = round(mt[k], decimal_num)

        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        num_true = W_true.sum(axis=1).sum()
        assert num_true != 0
        num_tp = (W_p + W_true).applymap(lambda elem: 1 if elem == 2 else 0).sum(axis=1).sum()
        num_fn_r = (W_p - W_true).applymap(lambda elem: 1 if elem == 1 else 0).sum(axis=1).sum()
        score = np.max((num_tp - num_fn_r, 0)) / num_true
        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        assert (W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
        TP = (W_p + W_true).applymap(lambda elem: 1 if elem == 2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()

        precision = TP / max(TP_FP, 1)
        recall = TP / max(TP_FN, 1)

        denom = (recall + precision)
        F1 = 0.0
        if denom > 0:
            F1 = 2.0 * (recall * precision) / denom

        return precision, recall, F1


def output_result(causal_graph, true_graph, _time=0, var=0):
    met = MetricsDAG(causal_graph, true_graph)
    m = dict(met.metrics)

    result = (
        f"type:KDMC-{var},"
        f"time:{_time},"
        f"fdr:{m.get('fdr')},"
        f"tpr:{m.get('tpr')},"
        f"shd:{m.get('shd')},"
        f"nhd:{m.get('nhd')},"
        f"F1:{m.get('F1')},"
        f"precision:{m.get('precision')},"
        f"recall:{m.get('recall')}\n"
    )
    print(*(result.split(',')), sep='\n')
    print("---------------------------")

    m["type"] = f"KDMC-{var}"
    m["time"] = float(_time)
    return m
