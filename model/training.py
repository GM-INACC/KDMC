# -*- coding: utf-8 -*-
import os
import time
import torch
from utils.metrics import MetricsDAG
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    from aim import Run, Image
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False


class Record:
    def __init__(self, config):
        self.max_rewards_per_batch = []
        self.mean_rewards_per_batch = []

        self.mets = {
            'fdr': [],
            'tpr': [],
            'shd': [],
            'F1': [],
            'precision': [],
            'recall': []
        }
        self.max_rewards = []
        self.max_reward = float('-inf')

        self.record_aim = getattr(config, 'record_aim', False) and AIM_AVAILABLE
        self.true_dag = getattr(config, 'true_dag', None)

        out_dir = getattr(config, "out_dir", None)
        if out_dir is None and hasattr(config, "all_config"):
            out_dir = getattr(config.all_config, "out_dir", None)

        if out_dir is None:
            root = getattr(config, "result_root", None)
            if root is None and hasattr(config, "all_config"):
                root = getattr(config.all_config, "result_root", "results")
            if root is None:
                root = "results"
            ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            out_dir = os.path.join(root, f"record_{ts}")

        self.out_dir = out_dir
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception:
            self.out_dir = None

        if self.record_aim:
            self.run = Run(experiment="KDMC")
            self.run["hparams"] = {
                k: v for k, v in config.all_config.__dict__.items()
                if k not in ("device", "true_dag")
            }

    def update_batch(self, max_reward: float, mean_reward: float):
        self.max_rewards_per_batch.append(max_reward)
        self.mean_rewards_per_batch.append(mean_reward)

    def update_global(self, new_max: float):
        if new_max > self.max_reward:
            self.max_reward = new_max
        self.max_rewards.append(self.max_reward)

    def update_reward(self, reward, ep=0, se_num=0):
        if isinstance(reward, torch.Tensor):
            max_reward = reward.max().item()
            mean_reward = reward.mean().item()
        else:
            reward = np.asarray(reward, dtype=np.float32)
            max_reward = float(np.max(reward))
            mean_reward = float(np.mean(reward))

        if self.record_aim:
            self.run.track({
                "max_reward": max_reward,
                "mean_reward": mean_reward,
                "Strong_Edge": se_num,
            }, epoch=ep)

        self.update_batch(max_reward, mean_reward)
        self.update_global(max_reward)

    def update_met(self, graph, ep):
        if isinstance(graph, torch.Tensor):
            graph = graph.detach().cpu().numpy()
        assert isinstance(graph, np.ndarray), "Graphs should be numpy arrays"

        met = MetricsDAG(graph, self.true_dag)
        for k in ['fdr', 'tpr', 'shd', 'F1', 'precision', 'recall']:
            self.mets[k].append(met.metrics[k])
            if self.record_aim:
                self.run.track({k: met.metrics[k]}, epoch=ep)

    def update_img_graph(self, graph: np.ndarray, ep):
        img = graph.astype(np.uint8)
        img = np.where(img == 1, 0, 255)
        if self.record_aim:
            self.run.track(Image(img), name='ordering graph', step=ep)
