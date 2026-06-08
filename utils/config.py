# -*- coding: utf-8 -*-
import argparse
import torch

DEFAULT_LLM_TEMPERATURE = 0.5
DEFAULT_LLM_TIMEOUT = 300
DEFAULT_LLM_REQUEST_RETRIES = 4
DEFAULT_LLM_REQUEST_RETRY_WAIT = 3.0
DEFAULT_LLM_REQUEST_INTERVAL = 0.0
DEFAULT_TRIPLE_CHUNK_SIZE = 80


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Configuration")

    # Data and labels
    parser.add_argument("--datapath", type=str, default=None, help="Path to the input CSV file.")
    parser.add_argument(
        "--labelpath",
        type=str,
        default=None,
        help="Path to the ground-truth DAG CSV file. Required for evaluation in main.py/run_all.py.",
    )

    # Orchestrator
    parser.add_argument("--iterations", type=int, default=5, help="Number of outer iterations.")
    parser.add_argument(
        "--ablation_variant",
        type=str,
        default="full",
        choices=["full", "no_k", "no_r", "no_f", "r_only", "kr_full", "no_ms", "no_mr", "no_tq"],
        help=(
            "Ablation variant. Module-level variants: full, no_k, no_r, no_f, or r_only. "
            "Knowledge-reasoning variants: kr_full, no_ms, no_mr, or no_tq."
        ),
    )
    parser.add_argument(
        "--template",
        type=str,
        default="ASI",
        choices=["ASI", "CRAC", "SAN", "ASIA", "Child"],
        help="Prompt template name.",
    )

    # LLM config
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="LLM model name. Pass it explicitly in the run command.",
    )
    parser.add_argument("--llm_temperature", type=float, default=DEFAULT_LLM_TEMPERATURE, help="LLM temperature.")
    parser.add_argument("--llm_timeout", type=int, default=DEFAULT_LLM_TIMEOUT, help="LLM timeout in seconds.")
    parser.add_argument(
        "--llm_request_retries",
        type=int,
        default=DEFAULT_LLM_REQUEST_RETRIES,
        help="Retry count for transient LLM connection failures.",
    )
    parser.add_argument(
        "--llm_request_retry_wait",
        type=float,
        default=DEFAULT_LLM_REQUEST_RETRY_WAIT,
        help="Base wait seconds between LLM retries.",
    )
    parser.add_argument(
        "--llm_request_interval",
        type=float,
        default=DEFAULT_LLM_REQUEST_INTERVAL,
        help="Sleep seconds after each successful LLM prompt chunk.",
    )
    parser.add_argument(
        "--triple_chunk_size",
        type=int,
        default=DEFAULT_TRIPLE_CHUNK_SIZE,
        help="Number of triples sent to the LLM per request when building priors.",
    )

    # Training / RL
    parser.add_argument("--actor_lr", type=float, default=0.0011, help="Actor learning rate.")
    parser.add_argument("--add_error", action="store_true", default=False, help="Enable error injection.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for GAT.")
    parser.add_argument("--base_line", type=float, default=-1, help="Initial reward baseline.")
    parser.add_argument("--base_line_rate", type=float, default=0.99, help="Baseline update rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--critic_lr", type=float, default=0.0041, help="Critic learning rate.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in GAT.")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument(
        "--reg_type",
        type=str,
        default="LR",
        choices=["LR", "QR", "GPR"],
        help="Regression type for BIC scoring.",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default="BIC",
        choices=["BIC", "BIC_different_var"],
        help="Reward score type.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=48,
        help="Expected sequence length for each variable input.",
    )
    parser.add_argument("--nblocks", type=int, default=4, help="Number of GAT blocks.")
    parser.add_argument("--nheads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--record_aim", action="store_true", default=False, help="Record metrics with Aim.")
    parser.add_argument("--sem_type", type=str, default=None, help="SEM type for synthetic data.")
    parser.add_argument(
        "--search_mode",
        type=str,
        default="rl",
        choices=["rl", "greedy"],
        help="Structure search backend. `greedy` is used for the w/o R ablation.",
    )

    # Prior usage
    parser.add_argument(
        "--confidence_weight",
        type=float,
        default=0.7,
        help="Weight used to inject LLM confidence into training.",
    )
    parser.add_argument(
        "--prior_path",
        type=str,
        default=None,
        help="Path to the prior graph text file used for training, typically llm_result.txt.",
    )
    parser.add_argument(
        "--prior_strategy",
        type=str,
        default="adaptive_hmp",
        choices=["adaptive_hmp", "fixed_hmp", "all_soft", "all_hard"],
        help=(
            "Prior injection strategy for hierarchical mixed prior experiments. "
            "adaptive_hmp uses tau=1-missing_rate; fixed_hmp uses tau=0.5; "
            "all_soft treats every prior edge as soft; all_hard treats DAG-projected prior edges as hard."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Explicit output directory. If omitted, a timestamped directory is used.",
    )

    # Missingness / masks
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.2,
        help="Overall missing rate used to compute tau = 1 - missing_rate.",
    )

    # Parent selection
    parser.add_argument(
        "--max_parents",
        type=int,
        default=0,
        help="Maximum number of parents per node. Use 0 for auto: 4 for empty-prior RL-only, 6 when soft/hard priors exist.",
    )
    parser.add_argument(
        "--delta_bic_thr",
        type=float,
        default=-1.0,
        help="Gain threshold for free edges during parent selection. Use a negative value for auto.",
    )
    parser.add_argument(
        "--delta_bic_thr_soft",
        type=float,
        default=0.001,
        help="Gain threshold for soft/hard prior edges during parent selection.",
    )
    parser.add_argument(
        "--prior_conf_gain",
        type=float,
        default=0.5,
        help="Confidence bonus used to rank soft/hard prior edges.",
    )
    parser.add_argument(
        "--prior_policy",
        type=str,
        default="augment",
        choices=["augment", "rewrite"],
        help="How RL uses prior edges: `augment` keeps prior edges and only adds a few new ones; `rewrite` reselects parents from scratch.",
    )
    parser.add_argument(
        "--max_new_edges_per_node",
        type=int,
        default=0,
        help="Maximum number of free-edge additions per node in `augment` mode. The default `0` enables auto mode: 1 for sparse high-hard-ratio priors, 2 for ordinary priors, full parent search for empty-prior RL-only runs.",
    )
    parser.add_argument(
        "--max_global_new_edges",
        type=int,
        default=-1,
        help="Optional cap for the extra global support sweep after RL parent selection. Use -1 for auto in prior runs, 0 to disable.",
    )
    parser.add_argument(
        "--lambda_free",
        type=float,
        default=0.0,
        help="Optional reward penalty on free edges.",
    )
    parser.add_argument(
        "--lambda_soft",
        type=float,
        default=0.0,
        help="Optional reward bonus on soft prior edges.",
    )
    parser.add_argument(
        "--lambda_edit",
        type=float,
        default=-1.0,
        help="Penalty for deviating from prior edges. Use a negative value for the built-in auto setting.",
    )
    parser.add_argument(
        "--lambda_density",
        type=float,
        default=-1.0,
        help="Quadratic penalty for graphs denser than the target edge count. Use a negative value for auto.",
    )
    parser.add_argument(
        "--target_edges",
        type=int,
        default=0,
        help="Target edge count for the density penalty. Use 0 for auto.",
    )
    parser.add_argument(
        "--accept_margin",
        type=float,
        default=0.0,
        help="In prior runs, keep the prior graph unless the learned graph beats prior utility by this margin.",
    )
    parser.add_argument(
        "--empty_initial_prior_graph",
        action="store_true",
        help="Use the prior confidence/masks but start RL from an empty initial graph.",
    )
    parser.add_argument(
        "--anchor_soft_prior",
        action="store_true",
        default=True,
        help="In augment mode, seed parent sets with all LLM prior edges instead of only hard edges.",
    )
    parser.add_argument(
        "--no_anchor_soft_prior",
        dest="anchor_soft_prior",
        action="store_false",
        help="In augment mode, allow soft prior edges to be re-selected from data instead of anchored.",
    )
    parser.add_argument(
        "--coverage_gamma",
        type=float,
        default=0.0,
        help="Exponent used to down-weight gains supported by few complete cases.",
    )
    parser.add_argument(
        "--order_refine_steps",
        type=int,
        default=0,
        help="Number of greedy adjacent-swap refinement passes applied to the best sampled ordering.",
    )
    parser.add_argument("--entropy_coef", type=float, default=0.005, help="Entropy bonus coefficient for actor exploration.")
    parser.add_argument("--grad_clip_norm", type=float, default=2.0, help="Gradient clipping norm for actor/critic updates.")

    # Paths
    parser.add_argument("--result_root", type=str, default="results", help="Root directory for outputs.")
    parser.add_argument("--prompt_dir", type=str, default="prompt_generate", help="Prompt directory.")

    # Misc
    parser.add_argument("--debugger", action="store_true", default=False, help="Enable debugger mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for RL training.")

    config = parser.parse_args()
    config.is_synthetic = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", config.device)
    return config


class Config:
    """Config slices consumed by the KDMC submodules."""

    def __init__(self, cfg: argparse.Namespace, name: str, must_exist_edges_adj=None) -> None:
        self.must_exist_edges_adj = must_exist_edges_adj

        if name == "actor":
            self.actor_lr = cfg.actor_lr
            self.alpha = cfg.alpha
            self.batch_size = cfg.batch_size
            self.dropout = cfg.dropout
            self.n_samples = cfg.n_samples
            self.nblocks = cfg.nblocks
            self.nheads = cfg.nheads
            self.num_variables = cfg.num_variables

        elif name == "critic":
            self.critic_lr = cfg.critic_lr
            self.num_variables = cfg.num_variables
            self.n_samples = cfg.n_samples

        elif name == "reward":
            self.alpha = cfg.alpha
            self.base_line = cfg.base_line
            self.base_line_rate = cfg.base_line_rate
            self.n_samples = cfg.n_samples
            self.num_variables = cfg.num_variables
            self.score_type = cfg.score_type
            self.reg_type = cfg.reg_type
            self.med_w = 1.0
            self.med_w_flag = False

        elif name == "trainer":
            self.batch_size = cfg.batch_size
            self.epoch = cfg.epoch
            self.num_variables = cfg.num_variables
            self.n_samples = cfg.n_samples
            self.seed = getattr(cfg, "seed", 42)
            self.missing_rate = float(getattr(cfg, "missing_rate", 0.0))

            self.mask_hard = getattr(cfg, "mask_hard", None)
            self.mask_soft = getattr(cfg, "mask_soft", None)
            self.mask_free = getattr(cfg, "mask_free", None)
            self.mask_forbid_reverse = getattr(cfg, "mask_forbid_reverse", None)

            self.max_parents = int(getattr(cfg, "max_parents", 0))
            self.delta_bic_thr = float(getattr(cfg, "delta_bic_thr", -1.0))
            self.delta_bic_thr_soft = float(getattr(cfg, "delta_bic_thr_soft", 0.001))
            self.prior_conf_gain = float(getattr(cfg, "prior_conf_gain", 0.5))
            self.prior_policy = str(getattr(cfg, "prior_policy", "augment"))
            self.max_new_edges_per_node = int(getattr(cfg, "max_new_edges_per_node", 0))
            self.max_global_new_edges = int(getattr(cfg, "max_global_new_edges", -1))

            self.lambda_free = float(getattr(cfg, "lambda_free", 0.0))
            self.lambda_soft = float(getattr(cfg, "lambda_soft", 0.0))
            self.lambda_edit = float(getattr(cfg, "lambda_edit", -1.0))
            self.lambda_density = float(getattr(cfg, "lambda_density", -1.0))
            self.target_edges = int(getattr(cfg, "target_edges", 0))
            self.accept_margin = float(getattr(cfg, "accept_margin", 0.0))
            self.anchor_soft_prior = bool(getattr(cfg, "anchor_soft_prior", True))
            self.coverage_gamma = float(getattr(cfg, "coverage_gamma", 0.0))
            self.order_refine_steps = int(getattr(cfg, "order_refine_steps", 1))
            self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.005))
            self.grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 5.0))

        elif name == "record":
            self.record_aim = cfg.record_aim
            self.true_dag = cfg.true_dag
            self.all_config = cfg
            self.out_dir = getattr(cfg, "out_dir", None)
            self.result_root = getattr(cfg, "result_root", "results")

        else:
            raise ValueError(f"Unknown Config section: {name}")
