# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = float(dropout)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.alpha = float(alpha)
        self.concat = bool(concat)

        self.W = nn.Linear(self.in_features, self.out_features, bias=True)
        self.a = nn.Parameter(torch.randn(2 * self.out_features + 2, 1))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, h, missing_rate, adj, prior_confidence=None, lambda_prior=0.0):
        Wh = self.W(h)
        e = self._attention_logits(Wh, missing_rate)

        if prior_confidence is not None and float(lambda_prior) > 0.0:
            e = e + float(lambda_prior) * prior_confidence.unsqueeze(0)

        adj_expand = adj.unsqueeze(0) if adj.dim() == 2 else adj
        zero_vec = torch.full_like(e, -1e15)
        attention = torch.where(adj_expand > 0, e, zero_vec)
        attention = torch.nan_to_num(attention, nan=-1e15, posinf=-1e15, neginf=-1e15)

        att = F.softmax(attention, dim=2)
        att = self.dropout_layer(att)
        Wh = torch.nan_to_num(Wh, nan=0.0, posinf=0.0, neginf=0.0)
        h_prime = torch.matmul(att, Wh)
        return F.elu(h_prime) if self.concat else h_prime

    def _attention_logits(self, Wh, missing_rate):
        batch_size, num_nodes, hidden_dim = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        Wh_j = Wh.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)

        m_i = missing_rate.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, 1)
        m_j = missing_rate.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, 1)

        att_input = torch.cat([Wh_i, Wh_j, m_i, m_j], dim=-1)
        logits = self.leakyrelu(torch.matmul(att_input, self.a).squeeze(-1))
        return torch.nan_to_num(logits, nan=-1e15, posinf=-1e15, neginf=-1e15)


class GraphAttentionEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nblocks, nvars):
        super().__init__()
        self.dropout = float(dropout)
        self.nblocks = int(nblocks)
        self.nfeat = int(nfeat)
        self.nhid = int(nhid)
        self.alpha = float(alpha)
        self.nheads = int(nheads)
        self.nvars = int(nvars)

        if self.nhid % self.nheads != 0:
            raise ValueError(
                f"nhid({self.nhid}) must be divisible by nheads({self.nheads})."
            )

        self.missing_embedding = nn.Parameter(torch.randn(1, 1, self.nfeat))
        head_dim = int(self.nhid / self.nheads)

        self.attentions = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GATLayer(
                            in_features=(self.nfeat if block_idx == 0 else self.nhid),
                            out_features=head_dim,
                            dropout=self.dropout,
                            alpha=self.alpha,
                            concat=True,
                        )
                        for _ in range(self.nheads)
                    ]
                )
                for block_idx in range(self.nblocks)
            ]
        )

        self.init_normalization = nn.BatchNorm1d(self.nvars)
        self.normalizations = nn.ModuleList(
            [nn.BatchNorm1d(self.nvars) for _ in range(self.nblocks)]
        )

    def forward(self, x, mask, adj, prior_confidence=None, lambda_prior=0.0):
        x_tilde = (1 - mask) * x + mask * self.missing_embedding
        enc = self.init_normalization(x_tilde)
        missing_rate = torch.mean(mask, dim=-1, keepdim=True)

        for block_idx in range(self.nblocks):
            heads = [
                att(
                    enc,
                    missing_rate,
                    adj,
                    prior_confidence=prior_confidence,
                    lambda_prior=lambda_prior,
                )
                for att in self.attentions[block_idx]
            ]
            enc = torch.cat(heads, dim=2)
            enc = self.normalizations[block_idx](enc)
        return enc


class OrderDecoder(nn.Module):
    def __init__(self, batch_size, in_features, out_features):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.batch_size = int(batch_size)
        self.W = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        self.flatten = nn.Flatten()

    def forward(self, enc):
        batch_size, num_nodes, _ = enc.shape
        if num_nodes != self.out_features:
            raise ValueError("Decoder output size must match the number of variables.")

        mask = torch.ones((batch_size, num_nodes), device=enc.device, dtype=enc.dtype)

        flat_enc = self.flatten(enc)
        pos, log_prob, error, entropy = self._sample_position(flat_enc, mask)
        mask = mask - F.one_hot(pos.squeeze(1), num_classes=num_nodes).to(mask.dtype)

        positions = pos
        log_probs = log_prob
        errors = error
        entropies = entropy

        for _ in range(num_nodes - 1):
            enc = enc * mask.unsqueeze(2)
            flat_enc = self.flatten(enc)
            next_pos, next_log_prob, next_error, next_entropy = self._sample_position(
                flat_enc, mask
            )
            mask = mask - F.one_hot(next_pos.squeeze(1), num_classes=num_nodes).to(mask.dtype)
            positions = torch.cat([next_pos, positions], dim=1)
            log_probs = log_probs + next_log_prob
            errors += next_error
            entropies = entropies + next_entropy

        return positions, log_probs, errors, entropies

    def _sample_position(self, flat_enc, mask):
        logits = self.W(flat_enc)
        logits = torch.nan_to_num(logits, nan=-1e15, posinf=-1e15, neginf=-1e15)
        logits = torch.where(mask > 0, logits, torch.full_like(logits, -1e15))

        with torch.no_grad():
            argsort_logits = torch.argsort(logits, descending=True).cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            selected = argsort_logits[:, 0]
            error = np.array([mask_np[idx, node] == 0 for idx, node in enumerate(selected)], dtype=np.int64)

        log_p = torch.log_softmax(logits, dim=1)
        log_p = torch.nan_to_num(log_p, nan=-1e15, posinf=-1e15, neginf=-1e15)
        probs = torch.exp(log_p).clamp_min(1e-12)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

        idx = torch.multinomial(probs, 1).squeeze(1)
        log_prob = log_p.gather(1, idx.unsqueeze(1)).squeeze(1)
        return idx.unsqueeze(1), log_prob, error, entropy


class StructureActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            int(config.n_samples),
            int(config.n_samples),
            float(config.dropout),
            float(config.alpha),
            int(config.nheads),
            int(config.nblocks),
            int(config.num_variables),
        )
        self.decoder = OrderDecoder(
            batch_size=int(config.batch_size),
            in_features=int(config.num_variables) * int(config.n_samples),
            out_features=int(config.num_variables),
        )

    def forward(self, X_full, adj, G_llm=None, lambda_llm=0.0, i=0):
        del i
        mask_full = torch.isnan(X_full).float()
        X_full = torch.nan_to_num(X_full, nan=0.0)

        enc = self.encoder(
            x=X_full,
            mask=mask_full,
            adj=adj,
            prior_confidence=G_llm,
            lambda_prior=lambda_llm,
        )
        positions, log_probs, errors, entropies = self.decoder(enc)
        return enc, positions, log_probs, errors, entropies


class ValueCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W1 = nn.Linear(config.num_variables, config.n_samples)
        self.W2 = nn.Linear(config.n_samples, 1)
        self.relu = nn.ReLU()

    def forward(self, enc):
        mean_enc = torch.mean(enc, dim=-1)
        value = self.relu(self.W1(mean_enc))
        value = self.W2(value)
        return value.squeeze()
