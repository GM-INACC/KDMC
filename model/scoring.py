import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve
from sklearn.preprocessing import PolynomialFeatures
import torch


class GPR_mine:
    def __init__(self, optimize=False):
        self.is_fit = False
        self.params = {"l": 1, "sigma_f": 1}
        self.optimize = optimize
        self.alpha = 1e-10

    def fit(self, y, median, p_eu):
        self.train_y = np.asarray(y)
        K = self.kernel(median, p_eu)
        np.fill_diagonal(K, 1)
        self.K_trans = K.copy()
        K[np.diag_indices_from(K)] += self.alpha
        self.L_ = cholesky(K, lower=True)
        self.alpha_ = cho_solve((self.L_, True), self.train_y)
        self.is_fit = True

    def predict(self):
        if not self.is_fit:
            return None
        y_mean = self.K_trans.dot(self.alpha_)
        return y_mean

    def kernel(self, median, p_eu):
        p_eu_nor = p_eu / max(median, 1e-12)
        K = np.exp(-0.5 * p_eu_nor)
        K = squareform(K)
        return K


class BICScorer(object):


    def __init__(self, config, inputdata):
        self.maxlen = int(config.num_variables)
        self.alpha = float(config.alpha)
        self.med_w = float(getattr(config, "med_w", 1.0))
        self.med_w_flag = bool(getattr(config, "med_w_flag", False))

        raw = np.asarray(inputdata, dtype=np.float32)
        self.raw_inputdata = raw
        self.raw_observed = np.isfinite(raw)

        col_means = np.nanmean(raw, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
        self.inputdata = np.where(np.isnan(raw), col_means[None, :], raw).astype(np.float32)
        self.n_rows = int(self.inputdata.shape[0])
        self.bic_penalty = np.log(max(float(self.n_rows), 2.0)) / max(float(self.n_rows), 1.0)
        self.score_type = config.score_type
        self.reg_type = config.reg_type
        self.complete_case_reward = bool(getattr(config, "complete_case_reward", False))
        assert self.score_type in ("BIC", "BIC_different_var")
        assert self.reg_type in ("LR", "QR", "GPR")

        self.poly = PolynomialFeatures()

        self.d = {}
        self.d_RSS = [{} for _ in range(self.maxlen)]
        self.d_RSS_rows = [{} for _ in range(self.maxlen)]

    def cal_rewards(self, graphs, positions=None, errors=None):
        rewards_batches = []
        if positions is None:
            positions = [None] * len(graphs)
        for g, pos in zip(graphs, positions):
            rewards_batches.append(self.calculate_reward_single_graph(g, position=pos))
        return rewards_batches

    def _lr_residual(self, X, y):
        Xd = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
        theta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        y_pred = Xd @ theta
        return (y - y_pred)

    def _qr_residual(self, X, y):
        X_poly = self.poly.fit_transform(X)[:, 1:]
        return self._lr_residual(X_poly, y)

    def _gpr_residual(self, X, y):
        if X.shape[0] < 2:
            return (y - np.mean(y))
        p_eu = pdist(X, "sqeuclidean")
        med_w = np.median(p_eu) if self.med_w_flag else self.med_w
        K = np.exp(-0.5 * p_eu / max(med_w, 1e-12))
        K = squareform(K)
        np.fill_diagonal(K, 1)
        gpr = GPR_mine()
        gpr.fit(y, med_w, p_eu)
        pred = gpr.predict()
        if pred is None:
            return (y - np.mean(y))
        return (y - pred)

    def calculate_reward_single_graph(self, graph_batch, position=None):
        if isinstance(graph_batch, torch.Tensor):
            graph_batch = graph_batch.detach().cpu().numpy()
        graph_batch = np.asarray(graph_batch, dtype=np.float32)
        graph_key = (graph_batch > 0.5).astype(np.int8, copy=False).tobytes()

        if graph_key in self.d:
            return self.d[graph_key]

        RSS_ls = []
        node_scores = []
        for i in range(self.maxlen):
            RSSi, n_eff, k = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)
            if self.complete_case_reward:
                n_eff = max(float(n_eff), 1.0)
                bic_i = np.log((float(RSSi) / n_eff) + 1e-8)
                bic_i += float(k) * self._bic_penalty_for_n(n_eff)
                node_scores.append(bic_i)

        if self.score_type == "BIC":
            if self.complete_case_reward:
                bic = float(np.mean(node_scores))
            else:
                edge_count = float((graph_batch > 0.5).sum())
                n = max(float(self.n_rows), 1.0)
                bic = np.log((float(np.sum(RSS_ls)) / n) + 1e-8)
                bic += edge_count * self.bic_penalty / max(float(self.maxlen), 1.0)
        else:
            if self.complete_case_reward:
                bic = float(np.sum(node_scores))
            else:
                n = max(float(self.n_rows), 1.0)
                bic = float(np.sum(np.log((np.asarray(RSS_ls, dtype=np.float32) / n) + 1e-8)))

        reward = -float(bic)
        self.d[graph_key] = reward
        return reward

    def _bic_penalty_for_n(self, n_eff):
        return np.log(max(float(n_eff), 2.0)) / max(float(n_eff), 1.0)

    def _valid_rows_for(self, target, parents):
        parents = [int(p) for p in parents]
        cols = [int(target)] + parents
        return np.all(self.raw_observed[:, cols], axis=1)

    def _rss_from_arrays(self, y, X=None):
        y = np.asarray(y, dtype=np.float32)
        if y.shape[0] == 0:
            return 0.0
        if X is None or X.shape[1] == 0:
            y_err = y - np.mean(y)
        else:
            X = np.asarray(X, dtype=np.float32)
            if self.reg_type == "LR":
                y_err = self._lr_residual(X, y)
            elif self.reg_type == "QR":
                y_err = self._qr_residual(X, y)
            else:
                y_err = self._gpr_residual(X, y)
        return float(np.sum((y_err ** 2)))

    def cal_RSSi_for_rows(self, i, parents, row_mask=None):
        parents = [int(p) for p in parents]
        if row_mask is None:
            row_mask = self._valid_rows_for(i, parents)
        else:
            row_mask = np.asarray(row_mask, dtype=bool)
            if parents:
                row_mask = row_mask & self._valid_rows_for(i, parents)
            else:
                row_mask = row_mask & self.raw_observed[:, int(i)]

        k = int(len(parents))
        n_eff = int(np.sum(row_mask))
        min_rows = max(k + 2, 3)
        if n_eff < min_rows:
            # Avoid rewarding parent sets that only look good because complete-case
            # filtering left too few samples to estimate them.
            target_rows = self.raw_observed[:, int(i)]
            y = self.raw_inputdata[target_rows, int(i)]
            RSSi = self._rss_from_arrays(y, None)
            return RSSi, int(np.sum(target_rows)), 0

        y = self.raw_inputdata[row_mask, int(i)]
        if k == 0:
            RSSi = self._rss_from_arrays(y, None)
        else:
            X = self.raw_inputdata[row_mask][:, parents]
            RSSi = self._rss_from_arrays(y, X)
        return RSSi, n_eff, k

    def local_bic(self, i, parents, row_mask=None):
        if not self.complete_case_reward:
            col = np.zeros((self.maxlen,), dtype=np.float32)
            col[[int(p) for p in parents]] = 1.0
            graph_col = np.zeros((self.maxlen, self.maxlen), dtype=np.float32)
            graph_col[:, int(i)] = col
            RSSi, n_eff, k = self.cal_RSSi(i, graph_col)
            n_eff_f = max(float(n_eff), 1.0)
            bic_i = np.log((float(RSSi) / n_eff_f) + 1e-8)
            bic_i += float(k) * self.bic_penalty / max(float(self.maxlen), 1.0)
            return float(bic_i), int(n_eff), int(k)

        RSSi, n_eff, k = self.cal_RSSi_for_rows(i, parents, row_mask=row_mask)
        n_eff_f = max(float(n_eff), 1.0)
        bic_i = np.log((float(RSSi) / n_eff_f) + 1e-8)
        bic_i += float(k) * self._bic_penalty_for_n(n_eff_f) / max(float(self.maxlen), 1.0)
        return float(bic_i), int(n_eff), int(k)

    def local_gain(self, i, current_parents, candidate_parent):
        candidate_parent = int(candidate_parent)
        current_parents = [int(p) for p in current_parents]
        candidate_parents = current_parents + [candidate_parent]
        row_mask = self._valid_rows_for(i, candidate_parents)
        old_bic, n_eff, _ = self.local_bic(i, current_parents, row_mask=row_mask)
        new_bic, n_eff, _ = self.local_bic(i, candidate_parents, row_mask=row_mask)
        coverage = float(n_eff) / max(float(self.n_rows), 1.0)
        return float(old_bic - new_bic), coverage

    def cal_RSSi(self, i, graph_batch):
        col = graph_batch[:, i]
        key = str(col.astype(np.int8).tolist())
        if key in self.d_RSS[i]:
            return self.d_RSS[i][key]

        parents = np.where(col > 0.5)[0]
        if self.complete_case_reward:
            RSSi, n_eff, k = self.cal_RSSi_for_rows(i, parents)
        else:
            k = int(len(parents))
            n_eff = self.n_rows
            y = self.inputdata[:, i]
            if k == 0:
                RSSi = self._rss_from_arrays(y, None)
            else:
                RSSi = self._rss_from_arrays(y, self.inputdata[:, parents])
        out = (RSSi, n_eff, k)
        self.d_RSS[i][key] = out
        return out

    def update_all_scores(self):
        score_items = list(self.d.items())
        return sorted(score_items, key=lambda x: x[1], reverse=True)
