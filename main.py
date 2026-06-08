# -*- coding: utf-8 -*-


import os
import time
import json
import random
import pandas as pd
import numpy as np

from model.structure_learner import KDMCModel
from prompt_generate.exact_graph import extract_graph
from utils.data import DataLoader
from utils.config import get_parser, Config
from utils.metrics import output_result
from utils.graph import build_prior_masks, make_acyclic_by_confidence

import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def stable_topological_order(adj: np.ndarray, fallback_order: list[int]) -> np.ndarray:
    adj = (np.asarray(adj, dtype=np.float32) > 0.5).astype(np.int32)
    n = int(adj.shape[0])
    fallback = [int(x) for x in fallback_order if 0 <= int(x) < n]
    seen = set(fallback)
    fallback += [idx for idx in range(n) if idx not in seen]
    rank = {node: pos for pos, node in enumerate(fallback)}
    indeg = adj.sum(axis=0).astype(np.int64)
    available = [node for node in fallback if indeg[node] == 0]
    order = []
    used = set()

    while available:
        available.sort(key=lambda node: rank[node])
        node = int(available.pop(0))
        if node in used:
            continue
        used.add(node)
        order.append(node)
        for child in np.where(adj[node] > 0.5)[0]:
            child = int(child)
            indeg[child] -= 1
            if indeg[child] == 0 and child not in used:
                available.append(child)

    if len(order) != n:
        return np.asarray(fallback, dtype=np.int32)
    return np.asarray(order, dtype=np.int32)


def candidate_static_orders(prior_graph: np.ndarray, confidence_graph: np.ndarray, X_full: np.ndarray) -> list[np.ndarray]:
    n = int(np.asarray(X_full).shape[1])
    natural = np.arange(n, dtype=np.int32)
    prior = np.zeros((n, n), dtype=np.float32) if prior_graph is None else np.asarray(prior_graph, dtype=np.float32)
    confidence = np.zeros((n, n), dtype=np.float32) if confidence_graph is None else np.asarray(confidence_graph, dtype=np.float32)
    prior = make_acyclic_by_confidence(prior, weight=confidence, keep_mask=None).astype(np.float32)

    missing_ratio = np.isnan(np.asarray(X_full, dtype=np.float32)).mean(axis=0)
    source_score = confidence.sum(axis=1) - confidence.sum(axis=0)
    seeds = [
        stable_topological_order(prior, natural.tolist()),
        stable_topological_order(prior, np.argsort(missing_ratio).tolist()),
        stable_topological_order(prior, np.argsort(-missing_ratio).tolist()),
        stable_topological_order(prior, np.argsort(-source_score).tolist()),
    ]

    unique = []
    seen_orders = set()
    for order in seeds:
        key = tuple(int(x) for x in order.tolist())
        if key not in seen_orders:
            unique.append(order.astype(np.int32))
            seen_orders.add(key)
    return unique


def finalize_static_graph(graph: np.ndarray, hard_mask: np.ndarray | None, confidence_graph: np.ndarray | None) -> np.ndarray:
    out = (np.asarray(graph, dtype=np.float32) > 0.5).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    keep = None
    if hard_mask is not None:
        keep = (np.asarray(hard_mask, dtype=np.float32) > 0.5).astype(np.int32)
        out = np.maximum(out, keep.astype(np.float32))
    weight = None if confidence_graph is None else np.asarray(confidence_graph, dtype=np.float32)
    return make_acyclic_by_confidence(out, weight=weight, keep_mask=keep).astype(np.float32)


def run_greedy_search(kdmc_model: KDMCModel, config, X_full, hard_mask_np, soft_mask_np, free_mask_np, confidence_graph):
    n = int(config.num_variables)
    has_prior_edges = bool(
        ((hard_mask_np is not None) and np.any(np.asarray(hard_mask_np) > 0.5))
        or ((soft_mask_np is not None) and np.any(np.asarray(soft_mask_np) > 0.5))
    )
    max_parents = int(config.max_parents)
    if max_parents <= 0:
        max_parents = 6 if has_prior_edges else 4

    observed_missing_rate = float(np.isnan(np.asarray(X_full, dtype=np.float32)).mean())
    delta_bic_thr = float(config.delta_bic_thr)
    if delta_bic_thr < 0.0:
        delta_bic_thr = 0.0 if (has_prior_edges or (observed_missing_rate > 0.0 and n >= 15)) else 0.001
    delta_bic_thr_soft = float(config.delta_bic_thr_soft)
    lambda_free = float(config.lambda_free)
    if lambda_free <= 0.0:
        lambda_free = 0.04 * (1.0 + observed_missing_rate) if has_prior_edges else 0.0

    hard_t = torch.from_numpy(hard_mask_np.astype("float32")).to(kdmc_model.device) if hard_mask_np is not None else None
    soft_t = torch.from_numpy(soft_mask_np.astype("float32")).to(kdmc_model.device) if soft_mask_np is not None else None
    free_t = torch.from_numpy(free_mask_np.astype("float32")).to(kdmc_model.device) if free_mask_np is not None else None
    confidence_np = confidence_graph.astype(np.float32) if confidence_graph is not None else None
    auto_free_edges_per_node = 1 if has_prior_edges else 0

    best_graph = None
    best_score = -np.inf
    orders = candidate_static_orders(kdmc_model.initinal_graph, confidence_graph, X_full)
    for order in orders:
        built = kdmc_model._graph_from_order(
            order=order,
            max_parents=max_parents,
            delta_bic_thr=delta_bic_thr,
            delta_bic_thr_soft=delta_bic_thr_soft,
            hard_mask=hard_t,
            soft_mask=soft_t,
            free_mask=free_t,
            confidence_graph_np=confidence_np,
            prior_conf_gain=float(config.prior_conf_gain),
            coverage_gamma=float(config.coverage_gamma),
            prior_policy=str(config.prior_policy),
            max_new_edges_per_node=int(config.max_new_edges_per_node),
            has_prior_edges=has_prior_edges,
            auto_free_edges_per_node=auto_free_edges_per_node,
            free_edge_penalty=lambda_free if has_prior_edges else 0.0,
            anchor_soft_prior=bool(config.anchor_soft_prior),
        )
        graph = finalize_static_graph(built["graph"], hard_mask_np, confidence_np)
        score = float(kdmc_model.scorer.calculate_reward_single_graph(graph))
        if score > best_score:
            best_score = score
            best_graph = graph

    if best_graph is None:
        best_graph = finalize_static_graph(kdmc_model.initinal_graph, hard_mask_np, confidence_np)
    print(f"[greedy search] candidate_orders={len(orders)} best_reward={best_score:.4f}")
    return best_graph.astype(np.float32)

def build_strategy_prior_masks(confidence_graph: np.ndarray, missing_rate: float, strategy: str) -> dict:
    """Build hard/soft/free masks for prior-injection strategy experiments."""
    strategy = str(strategy or "adaptive_hmp")
    conf_np = np.asarray(confidence_graph, dtype=np.float32)
    n = int(conf_np.shape[0])
    eye_np = np.eye(n, dtype=np.float32)
    positive_np = ((conf_np > 0.0).astype(np.float32) * (1.0 - eye_np)).astype(np.float32)

    if strategy == "adaptive_hmp":
        tau_prior = 1.0 - float(missing_rate)
        masks = build_prior_masks(torch.from_numpy(conf_np.astype("float32")), tau=tau_prior)
        masks["summary"]["strategy"] = strategy
        return masks

    if strategy == "fixed_hmp":
        tau_prior = 0.5
        masks = build_prior_masks(torch.from_numpy(conf_np.astype("float32")), tau=tau_prior)
        masks["summary"]["strategy"] = strategy
        return masks

    if strategy == "all_soft":
        hard = torch.zeros((n, n), dtype=torch.float32)
        soft = torch.from_numpy(positive_np).float()
        free = torch.from_numpy(((1.0 - positive_np) * (1.0 - eye_np)).astype(np.float32)).float()
        return {
            "hard": hard,
            "soft": soft,
            "free": free,
            "summary": {
                "strategy": strategy,
                "tau": None,
                "hard": 0,
                "soft": int(soft.sum().item()),
                "free": int(free.sum().item()),
                "prior_edges": int(positive_np.sum()),
            },
        }

    if strategy == "all_hard":
        hard_np = make_acyclic_by_confidence(positive_np, weight=conf_np, keep_mask=None).astype(np.float32)
        soft_np = np.zeros((n, n), dtype=np.float32)
        free_np = ((1.0 - hard_np) * (1.0 - eye_np)).astype(np.float32)
        hard = torch.from_numpy(hard_np).float()
        soft = torch.from_numpy(soft_np).float()
        free = torch.from_numpy(free_np).float()
        return {
            "hard": hard,
            "soft": soft,
            "free": free,
            "summary": {
                "strategy": strategy,
                "tau": 0.0,
                "hard": int(hard.sum().item()),
                "soft": 0,
                "free": int(free.sum().item()),
                "prior_edges": int(positive_np.sum()),
                "cycle_dropped_prior_edges": int(positive_np.sum() - hard_np.sum()),
            },
        }

    raise ValueError(f"Unknown prior_strategy: {strategy}")

def main():
    config = get_parser()
    set_random_seed(int(getattr(config, "seed", 42)))

    if not config.datapath:
        raise ValueError("main.py requires --datapath.")
    if not config.labelpath:
        raise ValueError("main.py requires --labelpath for evaluation.")

    data = DataLoader(
        datasetpath=config.datapath,
        labelpath=config.labelpath,
        sorted=False,
        is_syn=config.is_synthetic
    )
    X_full = data.X_full

    if not config.prior_path:
        raise ValueError("main.py requires --prior_path, usually generated by run_all.py.")

    initinal_graph, confidence_graph = extract_graph(
        csv_file_path=config.datapath,
        result_txt_file_path=config.prior_path
    )
    if getattr(config, "empty_initial_prior_graph", False):
        initinal_graph = np.zeros_like(initinal_graph, dtype=np.float32)

    llm_confidence_lambda = float(config.confidence_weight)

    print("Initial graph (from prior):")
    print(initinal_graph)
    print("Confidence graph (from prior):")
    print(confidence_graph)
    print("lambda_llm =", llm_confidence_lambda)

    missing_rate = getattr(config, "missing_rate", 0.0)
    prior_strategy = getattr(config, "prior_strategy", "adaptive_hmp")
    if prior_strategy == "all_soft":
        # All-soft is a bias-only baseline: prior edges must not be anchored as fixed parents.
        config.anchor_soft_prior = False
    prior_masks = build_strategy_prior_masks(
        confidence_graph=confidence_graph,
        missing_rate=float(missing_rate),
        strategy=prior_strategy,
    )

    print("[prior strategy]", prior_strategy)
    print("[mask summary]", prior_masks["summary"])

    hard_mask_np = prior_masks["hard"].cpu().numpy()
    soft_mask_np = prior_masks["soft"].cpu().numpy()
    free_mask_np = prior_masks["free"].cpu().numpy()

    must_exist_edges_adj = hard_mask_np

    config.num_variables = data.num_variables
    config.true_dag = data.true_dag

    begin_time = time.time()
    if config.out_dir:
        out_dir = config.out_dir
    else:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(begin_time))
        out_dir = os.path.join(config.result_root, time_str)
    os.makedirs(out_dir, exist_ok=True)
    config.out_dir = out_dir
    with open(os.path.join(out_dir, "prior_strategy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(prior_masks["summary"], f, ensure_ascii=False, indent=2)


    kdmc_model = KDMCModel(
        actor_config=Config(config, "actor", must_exist_edges_adj=must_exist_edges_adj),
        critic_config=Config(config, "critic"),
        reward_config=Config(config, "reward"),
        record_config=Config(config, "record"),
        device=config.device,
        X_full=X_full,
        initinal_graph=initinal_graph,
        confidence_graph=confidence_graph,
        lambda_llm=llm_confidence_lambda,
        hard_mask=hard_mask_np,
        soft_mask=soft_mask_np,
        free_mask=free_mask_np,
    )

    print("\n----- before training -----")
    before_metrics = output_result(initinal_graph, config.true_dag, var=config.num_variables)
    with open(os.path.join(out_dir, "metrics_before.json"), "w", encoding="utf-8") as f:
        json.dump(before_metrics, f, ensure_ascii=False, indent=2)

    if getattr(config, "search_mode", "rl") == "greedy":
        print("\n----- greedy search (w/o R) -----")
        causal_graph = run_greedy_search(
            kdmc_model=kdmc_model,
            config=config,
            X_full=X_full,
            hard_mask_np=hard_mask_np,
            soft_mask_np=soft_mask_np,
            free_mask_np=free_mask_np,
            confidence_graph=confidence_graph,
        )
    else:
        print("\n----- training -----")
        causal_graph = kdmc_model.trainer(
            X_full=X_full,
            trainer_config=Config(config, "trainer")
        )
    end_time = time.time()

    print("\n----- after training -----")
    after_metrics = output_result(
        causal_graph,
        config.true_dag,
        _time=end_time - begin_time,
        var=config.num_variables
    )
    with open(os.path.join(out_dir, "metrics_after.json"), "w", encoding="utf-8") as f:
        json.dump(after_metrics, f, ensure_ascii=False, indent=2)

    node_list = data.node_list

    causal_graph_df = pd.DataFrame(causal_graph, columns=node_list, index=node_list)
    causal_graph_df.to_csv(os.path.join(out_dir, "causal_graph.csv"))

    edges = []
    for i in range(config.num_variables):
        for j in range(config.num_variables):
            if causal_graph[i][j] == 1:
                edges.append((node_list[i], node_list[j]))
    causal_graph_list = pd.DataFrame(edges, columns=["from", "to"])
    causal_graph_list.to_csv(os.path.join(out_dir, "causal_graph_list.csv"), index=False)

    print(f"\nFinal causal graph saved to: {out_dir}")


if __name__ == "__main__":
    main()
