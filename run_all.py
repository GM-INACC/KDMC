# -*- coding: utf-8 -*-
import csv
import datetime
import json
import os
import re
import subprocess
import sys

import pandas as pd

from prompt_generate.memory import build_memory
from prompt_generate.miss_report.miss_calculate import run_miss_calculate
from prompt_generate.miss_report.miss_reason import run_miss_reason
from utils.config import get_parser


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def base_of_csv(path):
    return re.sub(r"[^0-9A-Za-z_\-\.]+", "_", os.path.splitext(os.path.basename(path))[0])



def write_empty_prior(csv_path: str, out_dir: str, name: str = "zero_prior.txt") -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("")
    return out_path


def write_prior_from_causal_graph(graph_csv_path: str, out_dir: str, confidence: float = 0.1) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "structural_feedback_prior.txt")
    edges = []
    with open(graph_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return write_empty_prior(graph_csv_path, out_dir, name="structural_feedback_prior.txt")
    headers = rows[0][1:]
    for row in rows[1:]:
        if not row:
            continue
        src = row[0]
        for dst, value in zip(headers, row[1:]):
            try:
                active = float(value) > 0.5
            except ValueError:
                active = False
            if active and src and dst and src != dst:
                edges.append((src, dst))
    with open(out_path, "w", encoding="utf-8") as f:
        for src, dst in edges:
            f.write(f"({src},{dst},{confidence:.4f})\n")
    return out_path


def ablation_controls(variant: str) -> dict:
    variant = (variant or "full").lower()
    knowledge_variants = {"kr_full", "no_ms", "no_mr", "no_tq"}
    uses_llm = variant in {"full", "no_r", "no_f"} or variant in knowledge_variants
    controls = {
        "variant": variant,
        "uses_llm_knowledge": uses_llm,
        "uses_rl_search": variant != "no_r",
        "uses_diff_feedback": variant in {"full", "no_r"} or variant in knowledge_variants,
        "uses_structural_feedback": variant == "no_k",
        "search_mode": "greedy" if variant == "no_r" else "rl",
        "query_mode": "pairwise" if variant == "no_tq" else "triple",
        "missing_context_mode": {
            "no_ms": "coarse_report",
            "no_mr": "stats_only",
        }.get(variant, "full_report"),
        "is_knowledge_reasoning_ablation": variant in knowledge_variants,
    }
    return controls
def now_stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def run_checked(cmd):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, check=True, env=env)


def parse_cell_from_csv_name(csv_path: str) -> tuple[str | None, float | None]:
    base = base_of_csv(csv_path)
    match = re.search(r"_([A-Z]+)_([0-9]+(?:\.[0-9]+)?)$", base)
    if not match:
        return None, None
    return match.group(1), float(match.group(2))


def write_coarse_missing_report(csv_path: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    df = pd.read_csv(csv_path)
    observed_ratio = float(df.isna().to_numpy().mean())
    mechanism, target_rate = parse_cell_from_csv_name(csv_path)
    report = {
        "introduction": (
            "Coarse missingness report for the w/o MS ablation. Fine-grained variable-level "
            "missing rates, co-missing patterns, and missingness correlations are intentionally omitted."
        ),
        "ablation_context": "no_ms",
        "missing_mechanism": {
            "declared_mechanism": mechanism or "unknown",
            "target_missing_rate": target_rate,
            "observed_overall_missing_rate": round(observed_ratio, 4),
        },
        "variable_missing_ratio": {
            "description": "Omitted in the w/o MS ablation.",
            "data": {},
        },
        "high_missing_variables": {
            "description": "Omitted in the w/o MS ablation.",
            "data": [],
        },
        "missing_mechanisms_analysis": {
            "global": {
                "possible_related_variables": [],
                "description": (
                    "Only the declared missing mechanism and the overall missing rate are provided. "
                    "Do not infer variable-specific missingness risks from unavailable statistics."
                ),
            }
        },
        "missing_patterns": {
            "description": "Omitted in the w/o MS ablation.",
            "data": {"counts": {}},
        },
        "spatio_temporal_features": {},
        "bias_risk": {
            "selection_bias": "Assess conservatively because fine-grained missingness statistics are hidden.",
            "information_bias": "Assess conservatively because variable-level missingness evidence is hidden.",
        },
    }
    out_path = os.path.join(out_dir, f"{base_of_csv(csv_path)}_miss_report.json")
    write_json(out_path, report)
    return out_path


def top_items(mapping: dict, limit: int = 12, key_abs: bool = False) -> dict:
    def sort_key(item):
        value = item[1]
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        return abs(value) if key_abs else value

    return dict(sorted((mapping or {}).items(), key=sort_key, reverse=True)[:limit])


def compact_missing_statistics(stats: dict, top_k: int = 12, per_variable_top_k: int = 5) -> dict:
    ratios = (stats.get("variable_missing_ratio") or {}).get("data", {}) or {}
    high = (stats.get("high_missing_variables") or {}).get("data", []) or []
    sample_stats = (stats.get("sample_missing_stats") or {}).get("data", {}) or {}

    patterns = ((stats.get("missing_patterns") or {}).get("data", {}) or {})
    pattern_counts = patterns.get("counts", {}) or {}
    pattern_props = patterns.get("proportions", {}) or {}
    top_pattern_counts = top_items(pattern_counts, top_k)
    top_pattern_props = {key: pattern_props.get(key) for key in top_pattern_counts}

    conditional = ((stats.get("conditional_missing_ratios") or {}).get("data", {}) or {})
    compact_conditional = {}
    for var, related in conditional.items():
        filtered = {k: v for k, v in (related or {}).items() if k != var}
        compact_conditional[var] = top_items(filtered, per_variable_top_k)

    correlations = ((stats.get("missing_value_correlations") or {}).get("data", {}) or {})
    compact_corr = {}
    for var, related in correlations.items():
        filtered = {k: v for k, v in (related or {}).items() if k != var}
        compact_corr[var] = top_items(filtered, per_variable_top_k, key_abs=True)

    return {
        "introduction": (
            "Compact numeric missingness statistics for the w/o MR ablation. "
            "The natural-language missingness report and LLM missing-mechanism reasoning are disabled. "
            "Large full matrices are intentionally compressed to top entries to control API cost."
        ),
        "ablation_context": "no_mr_compact_stats_only",
        "variable_missing_ratio": {
            "description": "Variable-level missing ratio in percent.",
            "data": ratios,
        },
        "high_missing_variables": {
            "description": "Variables with missing ratio above the original high-missing threshold.",
            "data": high,
        },
        "sample_missing_stats": {
            "description": "Summary of number of missing variables per sample.",
            "data": sample_stats,
        },
        "missing_patterns": {
            "description": f"Top {top_k} missingness patterns by count.",
            "data": {
                "counts": top_pattern_counts,
                "proportions": top_pattern_props,
            },
        },
        "conditional_missing_ratios_top": {
            "description": f"For each variable A, top {per_variable_top_k} values of P(B missing | A missing), in percent.",
            "data": compact_conditional,
        },
        "missing_value_correlations_top": {
            "description": f"For each missingness indicator, top {per_variable_top_k} absolute point-biserial correlations with observed numeric variables.",
            "data": compact_corr,
        },
    }


def write_stats_only_missing_context(csv_path: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    calc_path = run_miss_calculate(csv_path, out_dir)
    with open(calc_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    compact = compact_missing_statistics(stats)
    out_path = os.path.join(out_dir, f"{base_of_csv(csv_path)}_miss_report.json")
    write_json(out_path, compact)
    return out_path


def prepare_missing_context(cfg, csv_path: str, out_dir: str, mode: str) -> str:
    if mode == "coarse_report":
        return write_coarse_missing_report(csv_path, out_dir)
    if mode == "stats_only":
        return write_stats_only_missing_context(csv_path, out_dir)
    return compute_and_reason(cfg, csv_path, out_dir)


def compute_and_reason(cfg, csv_path: str, out_dir: str) -> str:
    ensure_dir(out_dir)

    calc_path = run_miss_calculate(csv_path, out_dir)
    reason_path = run_miss_reason(
        stats_path=calc_path,
        output_dir=out_dir,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout,
        template=cfg.template,
        request_retries=cfg.llm_request_retries,
        request_retry_wait=cfg.llm_request_retry_wait,
    )

    base = base_of_csv(csv_path)
    std_reason = os.path.join(out_dir, f"{base}_miss_reason.json")
    if os.path.abspath(reason_path) != os.path.abspath(std_reason):
        os.replace(reason_path, std_reason)

    with open(calc_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(std_reason, "r", encoding="utf-8") as f:
        reason = json.load(f)

    counts = (stats.get("missing_patterns", {}) or {}).get("data", {}).get("counts", {}) or {}
    top10 = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10])
    report = {
        "introduction": "This report summarizes missingness statistics and LLM-based mechanism analysis.",
        "variable_missing_ratio": stats.get("variable_missing_ratio", {}),
        "high_missing_variables": stats.get("high_missing_variables", []),
        "missing_mechanisms_analysis": (
            reason.get("missing_mechanisms_analysis") or reason.get("missing_mechanisms") or {}
        ),
        "missing_patterns": {
            "description": (stats.get("missing_patterns", {}) or {}).get("description", ""),
            "data": {"counts": top10},
        },
        "spatio_temporal_features": reason.get("spatio_temporal_features", {}),
        "bias_risk": reason.get("bias_risk", {}),
    }

    out_path = os.path.join(out_dir, f"{base}_miss_report.json")
    write_json(out_path, report)
    return out_path


def run_prompt_py(cfg, csv_path: str, miss_report_path: str, iter_dir: str, memory_text: str | None):
    ensure_dir(iter_dir)

    if not cfg.llm_model:
        raise ValueError("run_all.py requires --llm_model. Example: --llm_model gpt-5.4")

    memory_file = ""
    if memory_text:
        memory_file = os.path.join(iter_dir, "memory_prompt_prev.txt")
        with open(memory_file, "w", encoding="utf-8") as f:
            f.write(memory_text)

    cmd = [
        sys.executable,
        os.path.join("prompt_generate", "prompt.py"),
        "--datapath",
        csv_path,
        "--miss_report",
        miss_report_path,
        "--out_dir",
        iter_dir,
        "--template",
        cfg.template,
        "--model",
        cfg.llm_model,
        "--temperature",
        str(cfg.llm_temperature),
        "--timeout",
        str(cfg.llm_timeout),
        "--triple_chunk_size",
        str(cfg.triple_chunk_size),
        "--request_retries",
        str(cfg.llm_request_retries),
        "--request_retry_wait",
        str(cfg.llm_request_retry_wait),
        "--request_interval",
        str(cfg.llm_request_interval),
        "--query_mode",
        str(getattr(cfg, "query_mode", "triple")),
    ]
    if memory_file:
        cmd += ["--memory_file", memory_file]

    print("[prompt] launch:", " ".join(cmd))
    run_checked(cmd)

    p_all = os.path.join(iter_dir, "llm_result_all.txt")
    p_dag = os.path.join(iter_dir, "llm_result.txt")
    return p_all, p_dag


def run_training(cfg, csv_path: str, label_path: str | None, prior_path: str, iter_dir: str):
    cmd = [
        sys.executable,
        "main.py",
        "--datapath",
        csv_path,
        "--prior_path",
        prior_path,
        "--out_dir",
        iter_dir,
        "--confidence_weight",
        str(cfg.confidence_weight),
        "--prior_strategy",
        str(getattr(cfg, "prior_strategy", "adaptive_hmp")),
        "--missing_rate",
        str(cfg.missing_rate),
        "--epoch",
        str(cfg.epoch),
        "--batch_size",
        str(cfg.batch_size),
        "--actor_lr",
        str(cfg.actor_lr),
        "--critic_lr",
        str(cfg.critic_lr),
        "--alpha",
        str(cfg.alpha),
        "--dropout",
        str(cfg.dropout),
        "--base_line",
        str(cfg.base_line),
        "--base_line_rate",
        str(cfg.base_line_rate),
        "--reg_type",
        str(cfg.reg_type),
        "--score_type",
        str(cfg.score_type),
        "--n_samples",
        str(cfg.n_samples),
        "--nblocks",
        str(cfg.nblocks),
        "--nheads",
        str(cfg.nheads),
        "--max_parents",
        str(cfg.max_parents),
        "--delta_bic_thr",
        str(cfg.delta_bic_thr),
        "--delta_bic_thr_soft",
        str(cfg.delta_bic_thr_soft),
        "--prior_conf_gain",
        str(cfg.prior_conf_gain),
        "--prior_policy",
        str(cfg.prior_policy),
        "--max_new_edges_per_node",
        str(cfg.max_new_edges_per_node),
        "--max_global_new_edges",
        str(cfg.max_global_new_edges),
        "--lambda_free",
        str(cfg.lambda_free),
        "--lambda_soft",
        str(cfg.lambda_soft),
        "--lambda_edit",
        str(cfg.lambda_edit),
        "--lambda_density",
        str(cfg.lambda_density),
        "--target_edges",
        str(cfg.target_edges),
        "--accept_margin",
        str(cfg.accept_margin),
        "--coverage_gamma",
        str(cfg.coverage_gamma),
        "--order_refine_steps",
        str(cfg.order_refine_steps),
        "--entropy_coef",
        str(cfg.entropy_coef),
        "--grad_clip_norm",
        str(cfg.grad_clip_norm),
        "--search_mode",
        str(getattr(cfg, "search_mode", "rl")),
        "--seed",
        str(cfg.seed),
    ]
    if getattr(cfg, "anchor_soft_prior", True):
        cmd.append("--anchor_soft_prior")
    else:
        cmd.append("--no_anchor_soft_prior")
    if label_path:
        cmd += ["--labelpath", label_path]
    if getattr(cfg, "record_aim", False):
        cmd.append("--record_aim")

    print("[train] launch:", " ".join(cmd))
    run_checked(cmd)


def main():
    cfg = get_parser()
    controls = ablation_controls(getattr(cfg, "ablation_variant", "full"))
    cfg.search_mode = controls["search_mode"]
    cfg.query_mode = controls["query_mode"]
    cfg.missing_context_mode = controls["missing_context_mode"]

    if not cfg.datapath:
        raise ValueError("run_all.py requires --datapath.")
    if not cfg.labelpath:
        raise ValueError("run_all.py requires --labelpath for evaluation outputs.")
    if controls["uses_llm_knowledge"] and not cfg.llm_model:
        raise ValueError("This ablation variant requires --llm_model. Example: --llm_model gpt-5.4")

    run_name = f"{base_of_csv(cfg.datapath)}-{controls['variant']}-{now_stamp()}"
    run_root = os.path.join(cfg.result_root, run_name)
    ensure_dir(run_root)
    print("[run] output dir:", run_root)
    print("[ablation]", json.dumps(controls, ensure_ascii=False))

    metadata = {
        "ablation_variant": controls["variant"],
        "controls": controls,
        "datapath": cfg.datapath,
        "labelpath": cfg.labelpath,
        "template": cfg.template,
        "llm_model": cfg.llm_model,
        "iterations": cfg.iterations,
        "query_mode": controls["query_mode"],
        "missing_context_mode": controls["missing_context_mode"],
        "selection_rule": "choose the best metrics_after.json by F1 within each seed/run",
    }
    write_json(os.path.join(run_root, "ablation_metadata.json"), metadata)

    miss_report_path = None
    if controls["uses_llm_knowledge"]:
        miss_report_path = prepare_missing_context(
            cfg, cfg.datapath, run_root, controls["missing_context_mode"]
        )
        print("[miss_report]", miss_report_path)
    else:
        print("[miss_report] skipped: knowledge reasoning disabled for this ablation.")

    original_anchor_soft_prior = bool(getattr(cfg, "anchor_soft_prior", True))
    if controls["variant"] == "no_k":
        cfg.anchor_soft_prior = False

    memory_text = None
    previous_graph_path = None
    for it in range(1, cfg.iterations + 1):
        iter_dir = os.path.join(run_root, f"iter_{it}")
        ensure_dir(iter_dir)
        print(f"\n==== iteration {it} ({controls['variant']}) ====")

        if controls["uses_llm_knowledge"]:
            memory_for_prompt = memory_text if controls["uses_diff_feedback"] else None
            _, p_dag = run_prompt_py(cfg, cfg.datapath, miss_report_path, iter_dir, memory_for_prompt)
        elif controls["uses_structural_feedback"] and previous_graph_path:
            p_dag = write_prior_from_causal_graph(previous_graph_path, iter_dir, confidence=0.1)
            print("[prior] structural feedback prior:", p_dag)
        else:
            p_dag = write_empty_prior(cfg.datapath, iter_dir)
            print("[prior] zero prior:", p_dag)

        run_training(cfg, cfg.datapath, cfg.labelpath, p_dag, iter_dir)

        causal_graph_path = os.path.join(iter_dir, "causal_graph.csv")
        if controls["uses_diff_feedback"]:
            diff_report_path, memory_path = build_memory(cfg.datapath, iter_dir, p_dag)
            memory_text = read_text(memory_path)
            print("[diff ]", diff_report_path)
            print("[memo ]", memory_path)
        elif controls["uses_structural_feedback"]:
            diff_report_path, memory_path = build_memory(cfg.datapath, iter_dir, p_dag)
            previous_graph_path = causal_graph_path
            print("[diff ]", diff_report_path)
            print("[memo ] structural feedback only; not sent to LLM:", memory_path)
        else:
            previous_graph_path = causal_graph_path
            memory_text = None
            print("[diff ] skipped: feedback disabled for this ablation.")

    cfg.anchor_soft_prior = original_anchor_soft_prior
    print("\nPipeline finished:", run_root)


if __name__ == "__main__":
    main()
