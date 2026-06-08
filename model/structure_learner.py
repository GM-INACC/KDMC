# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.networks import StructureActor, ValueCritic

from model.scoring import BICScorer
from model.training import Record
from utils.graph import enforce_on_graphs, make_acyclic_by_confidence, is_acyclic


class KDMCModel(nn.Module):
    def __init__(
            self,
            actor_config,
            critic_config,
            reward_config,
            record_config,
            device,
            X_full=None,
            initinal_graph=None,
            confidence_graph=None,
            lambda_llm=0.0,
            hard_mask=None,
            soft_mask=None,
            free_mask=None,
    ):
        super(KDMCModel, self).__init__()
        self.device = device

        self.actor = StructureActor(actor_config).to(device)
        self.critic = ValueCritic(critic_config).to(device)

        self.scorer = BICScorer(reward_config, inputdata=X_full)
        self.record = Record(record_config)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_config.critic_lr)

        self.base_line = reward_config.base_line
        self.base_line_rate = reward_config.base_line_rate

        # LLM prior
        self.initinal_graph = initinal_graph
        self.confidence_graph = confidence_graph
        self.lambda_llm = float(lambda_llm)

        # masks (np)
        self.hard_mask = hard_mask
        self.soft_mask = soft_mask
        self.free_mask = free_mask

        # keep for final enforce
        self._masks_torch = None
        if (hard_mask is not None) and (soft_mask is not None) and (free_mask is not None):
            self._masks_torch = {
                "hard": torch.from_numpy(hard_mask).float().to(device),
                "soft": torch.from_numpy(soft_mask).float().to(device),
                "free": torch.from_numpy(free_mask).float().to(device),
            }

    def trainer(self, X_full, trainer_config):

        bs = int(trainer_config.batch_size)
        epochs = int(trainer_config.epoch)
        seq_len = max(1, int(getattr(trainer_config, "n_samples", 1)))

        max_parents = int(getattr(trainer_config, "max_parents", 2))
        delta_bic_thr = float(getattr(trainer_config, "delta_bic_thr", -1.0))
        delta_bic_thr_soft = float(getattr(trainer_config, "delta_bic_thr_soft", 0.0))
        entropy_coef = float(getattr(trainer_config, "entropy_coef", 0.001))
        grad_clip_norm = float(getattr(trainer_config, "grad_clip_norm", 5.0))
        prior_policy = str(getattr(trainer_config, "prior_policy", "augment"))
        max_new_edges_per_node = int(getattr(trainer_config, "max_new_edges_per_node", 0))
        max_global_new_edges = int(getattr(trainer_config, "max_global_new_edges", -1))
        prior_conf_gain = float(getattr(trainer_config, "prior_conf_gain", 0.5))
        coverage_gamma = float(getattr(trainer_config, "coverage_gamma", 0.5))

        lambda_free = float(getattr(trainer_config, "lambda_free", 0.0))
        lambda_soft = float(getattr(trainer_config, "lambda_soft", 0.0))
        lambda_edit = float(getattr(trainer_config, "lambda_edit", -1.0))
        lambda_density = float(getattr(trainer_config, "lambda_density", -1.0))
        target_edges = int(getattr(trainer_config, "target_edges", 0))
        accept_margin = float(getattr(trainer_config, "accept_margin", 0.0))
        anchor_soft_prior = bool(getattr(trainer_config, "anchor_soft_prior", True))
        order_refine_steps = int(getattr(trainer_config, "order_refine_steps", 1))
        observed_missing_rate = float(np.isnan(np.asarray(X_full, dtype=np.float32)).mean())

        # numpy -> torch (for attention bias)
        G_llm = None
        if self.confidence_graph is not None:
            G_llm = torch.from_numpy(self.confidence_graph.astype("float32")).to(self.device)

        # masks
        hard_mask = None
        soft_mask = None
        free_mask = None
        confidence_graph_np = None
        if self._masks_torch is not None:
            hard_mask = self._masks_torch["hard"]
            soft_mask = self._masks_torch["soft"]
            free_mask = self._masks_torch["free"]
            confidence_graph_np = self.confidence_graph.astype(np.float32) if self.confidence_graph is not None else None
        has_prior_edges = bool(
            (hard_mask is not None and torch.sum(hard_mask).item() > 0)
            or (soft_mask is not None and torch.sum(soft_mask).item() > 0)
        )
        prior_edge_count = 0
        hard_edge_count = 0
        if has_prior_edges:
            hard_edge_count = int(torch.sum(hard_mask).item()) if hard_mask is not None else 0
            soft_edge_count = int(torch.sum(soft_mask).item()) if soft_mask is not None else 0
            prior_edge_count = hard_edge_count + soft_edge_count
        hard_prior_ratio = float(hard_edge_count) / max(float(prior_edge_count), 1.0)
        high_quality_prior = bool(
            has_prior_edges
            and prior_edge_count <= (2 * int(trainer_config.num_variables))
            and hard_prior_ratio >= 0.65
        )
        N = int(trainer_config.num_variables)
        auto_free_edges_per_node = 1 if has_prior_edges else 0
        if max_global_new_edges < 0:
            if not has_prior_edges:
                max_global_new_edges = 0
            elif high_quality_prior:
                if N <= 12:
                    max_global_new_edges = max(4, prior_edge_count // 2)
                else:
                    max_global_new_edges = max(4, min(6, N // 2))
            else:
                max_global_new_edges = N
        if max_parents <= 0:
            max_parents = 6 if has_prior_edges else 4

        max_possible_edges = max(1, N * (N - 1) // 2)
        if delta_bic_thr < 0.0:
            if has_prior_edges:
                delta_bic_thr = 0.0
            elif observed_missing_rate > 0.0 and N >= 15:
                delta_bic_thr = 0.0
            else:
                delta_bic_thr = 0.001
        if target_edges <= 0:
            if has_prior_edges:
                target_edges = min(max_possible_edges, prior_edge_count + max(2, N // 2))
            else:
                target_edges = min(max_possible_edges, max(N, 3 * N))
        if lambda_free <= 0.0:
            lambda_free = 0.04 * (1.0 + observed_missing_rate) if has_prior_edges else 0.0
        if lambda_soft <= 0.0:
            lambda_soft = 0.01 if has_prior_edges else 0.0
        if lambda_edit < 0.0:
            lambda_edit = 0.04 if high_quality_prior else (0.02 if has_prior_edges else 0.0)
        if lambda_density < 0.0:
            lambda_density = 0.01 if has_prior_edges else 0.0

        print(
            f"[trainer config] max_parents={max_parents} "
            f"prior_policy={prior_policy} max_new_edges_per_node={max_new_edges_per_node} "
            f"has_prior_edges={has_prior_edges} high_quality_prior={high_quality_prior} "
            f"auto_free_edges_per_node={auto_free_edges_per_node} "
            f"delta_bic_thr={delta_bic_thr:.4f} target_edges={target_edges} lambda_free={lambda_free:.4f} "
            f"lambda_edit={lambda_edit:.4f} lambda_density={lambda_density:.4f} "
            f"anchor_soft_prior={anchor_soft_prior} max_global_new_edges={max_global_new_edges}"
        )

        prior_graph_np = None
        prior_conf_np = None
        if self.initinal_graph is not None:
            prior_graph_np = (np.asarray(self.initinal_graph, dtype=np.float32) > 0.5).astype(np.float32)
        if self.confidence_graph is not None:
            prior_conf_np = np.asarray(self.confidence_graph, dtype=np.float32)

        def utility_graph_np(graph_np: np.ndarray) -> float:
            graph_np = np.asarray(graph_np, dtype=np.float32)
            reward_value = float(self.scorer.calculate_reward_single_graph(graph_np))

            if (lambda_free > 0.0) and (self.free_mask is not None):
                reward_value -= float(lambda_free) * float((graph_np * self.free_mask).sum())
            if (lambda_soft > 0.0) and (self.soft_mask is not None):
                if prior_conf_np is not None:
                    reward_value += float(lambda_soft) * float((graph_np * self.soft_mask * prior_conf_np).sum())
                else:
                    reward_value += float(lambda_soft) * float((graph_np * self.soft_mask).sum())
            if (lambda_edit > 0.0) and (prior_graph_np is not None) and has_prior_edges:
                added = graph_np * (1.0 - prior_graph_np)
                removed = prior_graph_np * (1.0 - graph_np)
                if prior_conf_np is not None:
                    removed = removed * (1.0 + prior_conf_np)
                reward_value -= float(lambda_edit) * float(added.sum() + removed.sum())
            if lambda_density > 0.0:
                edge_count = float((graph_np > 0.5).sum())
                over = max(0.0, edge_count - float(target_edges))
                reward_value -= float(lambda_density) * over * over

            return reward_value

        def finalize_graph_np(graph_np: np.ndarray) -> np.ndarray:
            out = (np.asarray(graph_np) > 0.5).astype(np.float32)
            np.fill_diagonal(out, 0.0)

            if self._masks_torch is not None:
                out_t = torch.from_numpy(out).float().to(self.device)
                out = enforce_on_graphs(out_t, self._masks_torch).detach().cpu().numpy().astype(np.float32)

            keep_mask = self.hard_mask.astype(np.int32) if self.hard_mask is not None else None
            weight = self.confidence_graph.astype(np.float32) if self.confidence_graph is not None else None
            out = make_acyclic_by_confidence(
                adj=out,
                weight=weight,
                keep_mask=keep_mask,
            ).astype(np.float32)
            return out

        def score_graph_np(graph_np: np.ndarray) -> float:
            return utility_graph_np(graph_np)

        best_graph_np = None
        best_reward_value = -np.inf
        best_order_np = None
        init_graph_np = None
        init_reward_value = -np.inf

        if self.initinal_graph is not None:
            init_graph_np = finalize_graph_np(self.initinal_graph)
            best_graph_np = init_graph_np.copy()
            init_reward_value = score_graph_np(init_graph_np)
            best_reward_value = init_reward_value

        last_reward = None
        last_graphs_final_np = None
        for ep in range(epochs):
            X_batch = self.get_batch_data(X_full, bs, seq_len).to(self.device)

            # Actor forward: uses LLM bias in attention
            enc, positions, log_softmaxs, errors, entropies = self.actor(
                X_batch,
                adj=torch.ones((trainer_config.num_variables, trainer_config.num_variables), device=self.device),
                G_llm=G_llm,
                lambda_llm=self.lambda_llm
            )

            # ordering -> graphs (and prune parents)
            graphs_np, se_num = self._positions_to_graphs(
                positions=positions,
                max_parents=max_parents,
                delta_bic_thr=delta_bic_thr,
                delta_bic_thr_soft=delta_bic_thr_soft,
                hard_mask=hard_mask,
                soft_mask=soft_mask,
                free_mask=free_mask,
                confidence_graph_np=confidence_graph_np,
                prior_conf_gain=prior_conf_gain,
                coverage_gamma=coverage_gamma,
                prior_policy=prior_policy,
                max_new_edges_per_node=max_new_edges_per_node,
                has_prior_edges=has_prior_edges,
                auto_free_edges_per_node=auto_free_edges_per_node,
                free_edge_penalty=lambda_free if has_prior_edges else 0.0,
                anchor_soft_prior=anchor_soft_prior,
            )

            graphs_final_np = np.stack([finalize_graph_np(graph_np) for graph_np in graphs_np], axis=0).astype(np.float32)

            # Reward uses the same prior-aware utility as final model selection.
            reward = np.asarray([utility_graph_np(graph_np) for graph_np in graphs_final_np], dtype=np.float32)
            # critic baseline
            enc_detached = enc.detach()
            v = self.critic(enc_detached).reshape(-1)
            reward_t_raw = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            reward_mean = float(reward_t_raw.mean().item())
            if (ep == 0) and (not np.isfinite(self.base_line) or self.base_line < 0):
                self.base_line = reward_mean
            else:
                self.base_line = (
                    float(self.base_line_rate) * float(self.base_line)
                    + (1.0 - float(self.base_line_rate)) * reward_mean
                )
            baseline_t = torch.full_like(reward_t_raw, float(self.base_line))
            critic_target = reward_t_raw - baseline_t

            adv_raw = critic_target - v.detach()
            adv = (adv_raw - adv_raw.mean()) / adv_raw.std(unbiased=False).clamp_min(1e-6)

            # ===== actor loss =====
            logp = log_softmaxs.reshape(-1) if torch.is_tensor(log_softmaxs) else log_softmaxs
            entropy = entropies.reshape(-1) if torch.is_tensor(entropies) else entropies
            actor_loss = -torch.mean(adv * logp)
            if entropy_coef > 0.0:
                actor_loss = actor_loss - (float(entropy_coef) * torch.mean(entropy))

            # ===== critic loss =====
            critic_loss = F.smooth_l1_loss(v, critic_target)

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip_norm)
            self.critic_optim.step()

            # record
            self.record.update_reward(reward, ep=ep, se_num=se_num)
            if self.record.true_dag is not None:
                # pick best graph in batch
                best_idx = int(np.argmax(reward))
                self.record.update_met(graphs_final_np[best_idx], ep)

            batch_best_idx = int(np.argmax(reward))
            batch_best_reward = float(reward[batch_best_idx])
            if batch_best_reward > best_reward_value:
                best_reward_value = batch_best_reward
                best_graph_np = graphs_final_np[batch_best_idx].copy()
                best_order_np = positions.detach().cpu().numpy().astype(np.int32)[batch_best_idx].copy()
            last_reward = reward
            last_graphs_final_np = graphs_final_np

            if (ep + 1) % 200 == 0:
                print(f"[ep {ep+1}] actor_loss={actor_loss.item():.4f} critic_loss={critic_loss.item():.4f} "
                      f"reward_mean={float(np.mean(reward)):.4f} reward_max={float(np.max(reward)):.4f} "
                      f"entropy={float(torch.mean(entropy).item()):.4f} baseline={float(self.base_line):.4f}")

        # ====== finalize: choose best ever graph (from last batch best) and enforce hard ======
        if best_graph_np is None:
            if (last_reward is None) or (last_graphs_final_np is None):
                raise RuntimeError("Training finished without producing any candidate graph.")
            best_idx = int(np.argmax(last_reward))
            best_graph_np = last_graphs_final_np[best_idx].copy()
        if (best_order_np is not None) and (order_refine_steps > 0):
            best_order_np, refined_graph_np, refined_reward = self._refine_order_by_utility(
                order=best_order_np,
                current_graph=best_graph_np,
                current_score=best_reward_value,
                score_fn=score_graph_np,
                finalize_fn=finalize_graph_np,
                max_parents=max_parents,
                delta_bic_thr=delta_bic_thr,
                delta_bic_thr_soft=delta_bic_thr_soft,
                hard_mask=hard_mask,
                soft_mask=soft_mask,
                free_mask=free_mask,
                confidence_graph_np=confidence_graph_np,
                prior_conf_gain=prior_conf_gain,
                coverage_gamma=coverage_gamma,
                prior_policy=prior_policy,
                max_new_edges_per_node=max_new_edges_per_node,
                has_prior_edges=has_prior_edges,
                auto_free_edges_per_node=auto_free_edges_per_node,
                free_edge_penalty=lambda_free if has_prior_edges else 0.0,
                anchor_soft_prior=anchor_soft_prior,
                steps=order_refine_steps,
            )
            if refined_reward > best_reward_value:
                best_reward_value = refined_reward
                best_graph_np = refined_graph_np
                print(f"[order refine] improved utility to {best_reward_value:.4f}")

        if has_prior_edges and (max_global_new_edges > 0):
            sweep_starts = [(best_graph_np, best_reward_value)]
            if init_graph_np is not None:
                sweep_starts.append((init_graph_np, init_reward_value))
            best_sweep_graph = None
            best_sweep_reward = -np.inf
            best_sweep_added = 0
            for start_graph, start_score in sweep_starts:
                swept_graph_np, swept_reward, added_count = self._support_sweep_by_utility(
                    current_graph=start_graph,
                    current_score=start_score,
                    score_fn=score_graph_np,
                    max_additions=max_global_new_edges,
                    free_edge_penalty=lambda_free,
                    coverage_gamma=coverage_gamma,
                    min_gain=delta_bic_thr,
                )
                if added_count > 0 and swept_reward > best_sweep_reward:
                    best_sweep_reward = swept_reward
                    best_sweep_graph = swept_graph_np
                    best_sweep_added = added_count
            support_added = best_sweep_graph is not None
            if support_added:
                best_reward_value = best_sweep_reward
                best_graph_np = best_sweep_graph
                print(
                    f"[support sweep] added={best_sweep_added} "
                    f"utility={best_reward_value:.4f}"
                )
        else:
            support_added = False

        if has_prior_edges and (init_graph_np is not None):
            final_reward_value = score_graph_np(best_graph_np)
            if (not support_added) and final_reward_value < (init_reward_value + accept_margin):
                print(
                    f"[accept gate] keep prior graph: final_utility={final_reward_value:.4f}, "
                    f"prior_utility={init_reward_value:.4f}, margin={accept_margin:.4f}"
                )
                best_graph_np = init_graph_np.copy()
        return best_graph_np.astype(np.float32)

    # -------------------------
    # scheme1 core: ordering -> full DAG -> parent selection
    # -------------------------
    def _refine_order_by_utility(
        self,
        order: np.ndarray,
        current_graph: np.ndarray,
        current_score: float,
        score_fn,
        finalize_fn,
        max_parents: int,
        delta_bic_thr: float,
        delta_bic_thr_soft: float,
        hard_mask: torch.Tensor = None,
        soft_mask: torch.Tensor = None,
        free_mask: torch.Tensor = None,
        confidence_graph_np: np.ndarray = None,
        prior_conf_gain: float = 0.5,
        coverage_gamma: float = 0.0,
        prior_policy: str = "augment",
        max_new_edges_per_node: int = 0,
        has_prior_edges: bool = False,
        auto_free_edges_per_node: int = 1,
        free_edge_penalty: float = 0.0,
        anchor_soft_prior: bool = True,
        steps: int = 1,
    ):
        best_order = np.asarray(order, dtype=np.int32).copy()
        best_graph = np.asarray(current_graph, dtype=np.float32).copy()
        best_score = float(current_score)

        for _ in range(max(0, int(steps))):
            improved = False
            for idx in range(len(best_order) - 1):
                cand_order = best_order.copy()
                cand_order[idx], cand_order[idx + 1] = cand_order[idx + 1], cand_order[idx]
                cand = self._graph_from_order(
                    order=cand_order,
                    max_parents=max_parents,
                    delta_bic_thr=delta_bic_thr,
                    delta_bic_thr_soft=delta_bic_thr_soft,
                    hard_mask=hard_mask,
                    soft_mask=soft_mask,
                    free_mask=free_mask,
                    confidence_graph_np=confidence_graph_np,
                    prior_conf_gain=prior_conf_gain,
                    coverage_gamma=coverage_gamma,
                    prior_policy=prior_policy,
                    max_new_edges_per_node=max_new_edges_per_node,
                    has_prior_edges=has_prior_edges,
                    auto_free_edges_per_node=auto_free_edges_per_node,
                    free_edge_penalty=free_edge_penalty,
                    anchor_soft_prior=anchor_soft_prior,
                )
                cand_graph = finalize_fn(cand["graph"])
                cand_score = float(score_fn(cand_graph))
                if cand_score > best_score + 1e-9:
                    best_order = cand_order
                    best_graph = cand_graph
                    best_score = cand_score
                    improved = True
            if not improved:
                break

        return best_order, best_graph, best_score

    def _support_sweep_by_utility(
        self,
        current_graph: np.ndarray,
        current_score: float,
        score_fn,
        max_additions: int,
        free_edge_penalty: float = 0.0,
        coverage_gamma: float = 0.0,
        min_gain: float = 0.0,
    ):
        graph = (np.asarray(current_graph, dtype=np.float32) > 0.5).astype(np.float32)
        best_score = float(current_score)
        n = int(graph.shape[0])
        added = 0

        for _ in range(max(0, int(max_additions))):
            best_edge = None
            best_rank = -1e18
            for j in range(n):
                parents = [int(p) for p in np.where(graph[:, j] > 0.5)[0]]
                for i in range(n):
                    if i == j or graph[i, j] > 0.5 or graph[j, i] > 0.5:
                        continue
                    if self.free_mask is not None and self.free_mask[i, j] <= 0.5:
                        continue
                    if hasattr(self.scorer, "local_gain"):
                        gain, coverage = self.scorer.local_gain(j, parents, i)
                    else:
                        continue
                    rank = float(gain) * (float(coverage) ** float(coverage_gamma))
                    rank -= float(free_edge_penalty)
                    if rank <= min_gain:
                        continue
                    cand = graph.copy()
                    cand[i, j] = 1.0
                    if not is_acyclic(cand):
                        continue
                    if rank > best_rank:
                        best_rank = rank
                        best_edge = (i, j)

            if best_edge is None:
                break

            cand = graph.copy()
            cand[best_edge[0], best_edge[1]] = 1.0
            cand_score = float(score_fn(cand))
            graph = cand
            best_score = cand_score
            added += 1

        return graph, best_score, added

    def _positions_to_graphs(
        self,
        positions: torch.Tensor,
        max_parents: int,
        delta_bic_thr: float,
        delta_bic_thr_soft: float,
        hard_mask: torch.Tensor = None,
        soft_mask: torch.Tensor = None,
        free_mask: torch.Tensor = None,
        confidence_graph_np: np.ndarray = None,
        prior_conf_gain: float = 0.5,
        coverage_gamma: float = 0.5,
        prior_policy: str = "augment",
        max_new_edges_per_node: int = 0,
        has_prior_edges: bool = False,
        auto_free_edges_per_node: int = 2,
        free_edge_penalty: float = 0.0,
        anchor_soft_prior: bool = True,
    ):
        """
        For each sampled ordering:
          1) construct full DAG: edges from earlier -> later (upper-tri by order)
          2) for each node, select at most max_parents parents from its predecessors by greedy gain
             - classify candidate edges by free/soft/hard (mask)
             - apply separate gain threshold (thr_free/thr_soft)
             - (enhanced) for soft/hard: rank = gain + prior_conf_gain * conf(i->j)
        Returns:
          graphs_np: (B,N,N) 0/1
          se_num: number of selected soft/hard edges (for logging)
        """
        pos_np = positions.detach().cpu().numpy().astype(np.int32)
        B, N = pos_np.shape
        graphs = []
        se_num_total = 0

        for b in range(B):
            full = self._graph_from_order(
                order=pos_np[b],
                max_parents=max_parents,
                delta_bic_thr=delta_bic_thr,
                delta_bic_thr_soft=delta_bic_thr_soft,
                hard_mask=hard_mask,
                soft_mask=soft_mask,
                free_mask=free_mask,
                confidence_graph_np=confidence_graph_np,
                prior_conf_gain=prior_conf_gain,
                coverage_gamma=coverage_gamma,
                prior_policy=prior_policy,
                max_new_edges_per_node=max_new_edges_per_node,
                has_prior_edges=has_prior_edges,
                auto_free_edges_per_node=auto_free_edges_per_node,
                free_edge_penalty=free_edge_penalty,
                anchor_soft_prior=anchor_soft_prior,
            )
            graphs.append(full["graph"])
            se_num_total += int(full["se_num"])

        return np.stack(graphs, axis=0).astype(np.float32), se_num_total

    def _graph_from_order(
        self,
        order: np.ndarray,
        max_parents: int,
        delta_bic_thr: float,
        delta_bic_thr_soft: float,
        hard_mask: torch.Tensor = None,
        soft_mask: torch.Tensor = None,
        free_mask: torch.Tensor = None,
        confidence_graph_np: np.ndarray = None,
        prior_conf_gain: float = 0.5,
        coverage_gamma: float = 0.5,
        prior_policy: str = "augment",
        max_new_edges_per_node: int = 0,
        has_prior_edges: bool = False,
        auto_free_edges_per_node: int = 2,
        free_edge_penalty: float = 0.0,
        anchor_soft_prior: bool = True,
    ):

        hard_np = hard_mask.detach().cpu().numpy() if hard_mask is not None else None
        if hard_np is not None:
            order = self._project_order_by_hard_prior(order, hard_np)

        N = int(order.shape[0])
        pos = order.tolist()
        # predecessor set for each node
        idx_in_order = {node: k for k, node in enumerate(pos)}

        soft_np = soft_mask.detach().cpu().numpy() if soft_mask is not None else None
        free_np = free_mask.detach().cpu().numpy() if free_mask is not None else None

        # init empty graph
        g = np.zeros((N, N), dtype=np.int32)
        se_num = 0

        # helper: compute node-local score (smaller is better) on available rows
        def node_local_stats(j: int, parents: list[int]) -> tuple[float, float]:
            if hasattr(self.scorer, "local_bic"):
                bic_j, n_eff, _ = self.scorer.local_bic(j, parents)
                coverage = float(n_eff) / max(float(getattr(self.scorer, "n_rows", n_eff)), 1.0)
                return float(bic_j), float(coverage)

            col = np.zeros((N,), dtype=np.float32)
            col[parents] = 1.0
            graph_col = np.zeros((N, N), dtype=np.float32)
            graph_col[:, j] = col
            RSSi, n_eff, k = self.scorer.cal_RSSi(j, graph_col)
            ni = max(float(n_eff), 1.0)
            eps = 1e-8
            bic_j = np.log((float(RSSi) / ni) + eps)
            bic_j += float(k) * float(getattr(self.scorer, "bic_penalty", np.log(ni + eps) / ni))
            coverage = float(n_eff) / max(float(getattr(self.scorer, "n_rows", n_eff)), 1.0)
            return float(bic_j), float(coverage)

        # For each node j in order, choose parents from its predecessors
        for j in pos:
            order_preds = pos[:idx_in_order[j]]
            chosen = []

            if prior_policy == "augment":
                hard_preds = [i for i in range(N) if (hard_np is not None) and (hard_np[i, j] > 0.5)]
            else:
                hard_preds = [i for i in order_preds if (hard_np is not None) and (hard_np[i, j] > 0.5)]

            soft_all_preds = [
                i for i in range(N)
                if i not in hard_preds and (soft_np is not None) and (soft_np[i, j] > 0.5)
            ]
            soft_preds = [i for i in order_preds if i in soft_all_preds]

            if hard_preds:
                chosen.extend(hard_preds)
                se_num += len(hard_preds)
            if prior_policy == "augment" and anchor_soft_prior and soft_all_preds:
                for i in soft_all_preds:
                    if i not in chosen:
                        chosen.append(i)
                se_num += len(soft_all_preds)
            bic0, _ = node_local_stats(j, chosen)

            def greedy_select(
                candidates: list[int],
                thr: float,
                use_prior_rank: bool,
                max_additions: int | None = None,
                max_total_size: int | None = None,
                edge_penalty: float = 0.0,
            ) -> None:
                nonlocal chosen, bic0, se_num
                added = 0
                while True:
                    size_limit = (N - 1) if max_total_size is None else max_total_size
                    if (size_limit is not None) and (len(chosen) >= size_limit):
                        break
                    if (max_additions is not None) and (added >= max_additions):
                        break
                    best = None
                    best_rank = -1e18

                    for i in candidates:
                        if i in chosen:
                            continue

                        if hasattr(self.scorer, "local_gain"):
                            gain, coverage = self.scorer.local_gain(j, chosen, i)
                        else:
                            cand = chosen + [i]
                            bic1, coverage = node_local_stats(j, cand)
                            gain = bic0 - bic1
                        gain_eff = gain * (coverage ** coverage_gamma)
                        gain_eff = gain_eff - float(edge_penalty)
                        if use_prior_rank and (confidence_graph_np is not None):
                            rank = gain_eff
                            if gain_eff <= thr:
                                continue
                            rank += float(prior_conf_gain) * float(confidence_graph_np[i, j])
                        else:
                            rank = gain_eff
                            if gain_eff <= thr:
                                continue

                        if rank > best_rank:
                            best_rank = rank
                            best = i

                    if best is None:
                        break

                    chosen.append(best)
                    added += 1
                    bic0, _ = node_local_stats(j, chosen)
                    if use_prior_rank:
                        se_num += 1

            free_preds = []
            free_source_pool = order_preds
            for i in free_source_pool:
                if i == j:
                    continue
                if i in chosen or i in hard_preds or i in soft_all_preds:
                    continue
                reverse_is_hard = (hard_np is not None) and (hard_np[j, i] > 0.5)
                reverse_is_soft = (soft_np is not None) and (soft_np[j, i] > 0.5)
                if reverse_is_hard:
                    continue
                if reverse_is_soft and (prior_policy != "augment" or anchor_soft_prior):
                    continue
                if free_np is None or free_np[i, j] > 0.5:
                    free_preds.append(i)

            if prior_policy == "augment":
                parent_cap = max(len(chosen), max_parents)
                if not anchor_soft_prior:
                    greedy_select(
                        soft_preds,
                        delta_bic_thr_soft,
                        use_prior_rank=True,
                        max_total_size=parent_cap,
                    )
                if max_new_edges_per_node > 0:
                    free_additions = int(max_new_edges_per_node)
                elif has_prior_edges:
                    free_additions = min(int(auto_free_edges_per_node), max(0, parent_cap - len(chosen)))
                else:
                    free_additions = max(0, parent_cap - len(chosen))
                if free_additions > 0:
                    greedy_select(
                        free_preds,
                        delta_bic_thr,
                        use_prior_rank=False,
                        max_additions=free_additions,
                        max_total_size=parent_cap,
                        edge_penalty=free_edge_penalty,
                    )
            else:
                greedy_select(soft_preds, delta_bic_thr_soft, use_prior_rank=True, max_total_size=max_parents)
                greedy_select(
                    free_preds,
                    delta_bic_thr,
                    use_prior_rank=False,
                    max_total_size=max_parents,
                    edge_penalty=free_edge_penalty,
                )

            for i in chosen:
                g[i, j] = 1

        return {"graph": g, "se_num": se_num}

    def _project_order_by_hard_prior(self, order: np.ndarray, hard_np: np.ndarray) -> np.ndarray:
        if (hard_np is None) or (not np.any(hard_np > 0.5)):
            return np.asarray(order, dtype=np.int32)

        n_nodes = int(hard_np.shape[0])
        raw_order = np.asarray(order, dtype=np.int32).ravel().tolist()
        normalized_order = []
        seen = set()
        for node in raw_order:
            node = int(node)
            if 0 <= node < n_nodes and node not in seen:
                normalized_order.append(node)
                seen.add(node)
        for node in range(n_nodes):
            if node not in seen:
                normalized_order.append(node)
                seen.add(node)

        rank = {int(node): idx for idx, node in enumerate(normalized_order)}
        indeg = (hard_np > 0.5).sum(axis=0).astype(np.int64)
        available = [int(node) for node in normalized_order if indeg[int(node)] == 0]
        projected = []
        used = np.zeros(n_nodes, dtype=np.int8)

        while available:
            best_pos = min(range(len(available)), key=lambda idx: rank[available[idx]])
            u = int(available.pop(best_pos))
            if used[u]:
                continue
            used[u] = 1
            projected.append(u)
            for v in np.where(hard_np[u] > 0.5)[0]:
                indeg[v] -= 1
                if indeg[v] == 0 and not used[int(v)]:
                    available.append(int(v))

        if len(projected) != n_nodes:
            return np.asarray(normalized_order, dtype=np.int32)
        return np.asarray(projected, dtype=np.int32)

    # -------------------------
    # data utils
    # -------------------------
    def get_batch_data(self, X_full, bs, seq_len):
        X = np.asarray(X_full, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected X_full to be 2D, got shape {X.shape!r}")

        n, d = X.shape
        seq_len = max(1, int(seq_len))
        replace = n < seq_len

        batch = np.zeros((bs, d, seq_len), dtype=np.float32)
        for b in range(bs):
            idx = np.random.choice(n, size=seq_len, replace=replace)
            idx.sort()
            batch[b] = X[idx].T
        return torch.as_tensor(batch, dtype=torch.float32)
