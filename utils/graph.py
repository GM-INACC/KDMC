# -*- coding: utf-8 -*-


from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def is_acyclic(adj: np.ndarray) -> bool:

    n = int(adj.shape[0])
    indeg = adj.sum(axis=0).astype(np.int64)
    queue = [i for i in range(n) if indeg[i] == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in np.where(adj[u] != 0)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(int(v))
    return visited == n


def _find_one_cycle_edges(adj: np.ndarray) -> List[Tuple[int, int]]:

    n = int(adj.shape[0])
    state = np.zeros(n, dtype=np.int8)   # 0 unvisited, 1 visiting, 2 done
    parent = np.full(n, -1, dtype=np.int64)

    def dfs(u: int):
        state[u] = 1
        for v in np.where(adj[u] != 0)[0]:
            v = int(v)
            if state[v] == 0:
                parent[v] = u
                res = dfs(v)
                if res is not None:
                    return res
            elif state[v] == 1:
                # back edge u->v
                path = [u]
                cur = u
                while cur != v and parent[cur] != -1:
                    cur = int(parent[cur])
                    path.append(cur)
                if cur != v:
                    return [(u, v)]
                path = path[::-1]  # v ... u
                cycle_nodes = path + [v]
                edges = []
                for i in range(len(cycle_nodes) - 1):
                    edges.append((int(cycle_nodes[i]), int(cycle_nodes[i + 1])))
                return edges
        state[u] = 2
        return None

    for s in range(n):
        if state[s] == 0:
            res = dfs(int(s))
            if res is not None:
                return res
    return []


def make_acyclic_by_confidence(
    adj: np.ndarray,
    weight: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
    max_iter: int = 100000,
) -> np.ndarray:

    out = (adj != 0).astype(np.int32).copy()
    n = int(out.shape[0])
    np.fill_diagonal(out, 0)

    if weight is None:
        weight = np.zeros((n, n), dtype=np.float32)
    if keep_mask is None:
        keep_mask = np.zeros((n, n), dtype=np.int32)

    it = 0
    while (not is_acyclic(out)) and it < max_iter:
        it += 1
        cycle_edges = _find_one_cycle_edges(out)
        if not cycle_edges:
            break
        removable = [(u, v) for (u, v) in cycle_edges if keep_mask[u, v] == 0]
        if not removable:
            break
        u, v = min(removable, key=lambda e: float(weight[e[0], e[1]]))
        out[u, v] = 0

    return out


@torch.no_grad()
def build_prior_masks(confidence: torch.Tensor, tau: float):

    device = confidence.device
    conf = confidence.detach().cpu().numpy().astype(np.float32)
    N = conf.shape[0]

    hard = (confidence >= tau).float()
    soft = ((confidence > 0.0) & (confidence < tau)).float()
    free = (confidence == 0.0).float()

    eye = torch.eye(N, device=device)
    hard = hard * (1.0 - eye)
    soft = soft * (1.0 - eye)
    free = free * (1.0 - eye)

    hard_np = hard.detach().cpu().numpy().astype(np.int32)
    soft_np = soft.detach().cpu().numpy().astype(np.int32)

    if not is_acyclic(hard_np):
        hard_tmp = hard_np.copy()
        while not is_acyclic(hard_tmp):
            cycle_edges = _find_one_cycle_edges(hard_tmp)
            if not cycle_edges:
                break
            u, v = min(cycle_edges, key=lambda e: float(conf[e[0], e[1]]))
            hard_tmp[u, v] = 0
            if conf[u, v] > 0:
                soft_np[u, v] = 1
        hard_np = hard_tmp

    hard = torch.from_numpy(hard_np).float().to(device)
    soft = torch.from_numpy(soft_np).float().to(device)

    summary = {
        "hard": int(hard.sum().item()),
        "soft": int(soft.sum().item()),
        "free": int(free.sum().item()),
        "tau": float(tau),
    }

    return {
        "hard": hard,
        "soft": soft,
        "free": free,
        "summary": summary
    }


def enforce_on_graphs(graphs: torch.Tensor, masks: Dict[str, torch.Tensor]) -> torch.Tensor:

    hard = masks["hard"].to(graphs.device)
    block = masks.get("block", None)
    if block is not None:
        block = block.to(graphs.device)

    if graphs.dim() == 2:
        g = graphs.clone()
        if block is not None:
            g = g * (1.0 - block)
        eye = torch.eye(g.shape[0], device=g.device)
        g = g * (1.0 - eye)
        g = torch.maximum(g, hard)
        return g

    g = graphs.clone()
    if block is not None:
        g = g * (1.0 - block.unsqueeze(0))
    eye = torch.eye(g.shape[-1], device=g.device).unsqueeze(0)
    g = g * (1.0 - eye)
    g = torch.maximum(g, hard.unsqueeze(0))
    return g


import torch
import numpy as np



def ordering_matrix_pruning(bs, positions, _adj) -> torch.Tensor:
    if torch.is_tensor(positions):
        positions_np = positions.detach().cpu().numpy()
    else:
        positions_np = np.asarray(positions)

    if torch.is_tensor(_adj):
        adj_np = _adj.detach().cpu().numpy()
    else:
        adj_np = np.asarray(_adj)

    mats = []
    for j in range(bs):
        m = from_order_to_graph(positions_np[j])
        mats.append(m * adj_np)
    return torch.from_numpy(np.array(mats)).float()


def from_order_to_graph(true_position) -> np.ndarray:
    d = len(true_position)
    zero_matrix = np.zeros([d, d], dtype=np.float32)
    for n in range(d - 1):
        row_index = int(true_position[n])
        col_index = true_position[n + 1:]
        zero_matrix[row_index, col_index] = 1.0
    return zero_matrix
