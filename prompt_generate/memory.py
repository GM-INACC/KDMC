# -*- coding: utf-8 -*-
import argparse
import os
import re

import pandas as pd

try:
    from .exact_graph import extract_graph
except ImportError:
    from exact_graph import extract_graph


USAGE_NOTE = (
    "The Final DAG is obtained after one round of RL-based graph optimization. "
    "Compared with the LLM-derived prior graph, it may introduce added and removed edges. "
    "These edge changes summarize data-driven structural evidence and are provided only as "
    "review cues for the next LLM reasoning round. They should be considered together with "
    "the missingness report and domain metadata, rather than enforced as mandatory updates "
    "to the next prior graph."
)


def infer_round(iter_dir: str) -> str:
    """Infer round number from an iteration directory such as iter_1."""
    base = os.path.basename(os.path.normpath(iter_dir))
    match = re.search(r"(?:iter|round)[_-]?(\d+)", base, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)", base)
    return match.group(1) if match else "unknown"


def edge_to_text(edge: tuple[str, str]) -> str:
    return f"{edge[0]}->{edge[1]}"


def format_edge_list(edges: list[tuple[str, str]]) -> str:
    if not edges:
        return "None"
    return ", ".join(edge_to_text(edge) for edge in edges)


def format_edge_block(edges: list[tuple[str, str]]) -> list[str]:
    if not edges:
        return ["{}"]
    lines = ["{"]
    for idx, edge in enumerate(edges):
        suffix = "," if idx < len(edges) - 1 else ""
        lines.append(f'  "{edge_to_text(edge)}"{suffix}')
    lines.append("}")
    return lines


def to_binary_value(value) -> int:
    try:
        return 1 if float(value) > 0.5 else 0
    except (TypeError, ValueError):
        return 0


def format_adjacency_matrix(title: str, node_names: list[str], adj) -> list[str]:
    names = [str(name) for name in node_names]
    width = max(3, max((len(name) for name in names), default=1))
    label_width = width
    lines = [f"{title}:"]
    header = " " * (label_width + 2) + " ".join(f"{name:>{width}}" for name in names)
    lines.append(header.rstrip())
    for row_idx, name in enumerate(names):
        values = []
        for col_idx in range(len(names)):
            values.append(f"{to_binary_value(adj[row_idx][col_idx]):>{width}}")
        lines.append(f"{name:>{label_width}}  " + " ".join(values))
    return lines


def build_report_lines(
    round_id: str,
    node_names: list[str],
    prior_adj,
    final_adj,
    added_edges: list[tuple[str, str]],
    removed_edges: list[tuple[str, str]],
) -> list[str]:
    lines = [
        "Differential Report",
        "=" * 80,
        f"Round: {round_id}",
        "",
    ]
    lines.extend(format_adjacency_matrix("Prior graph", node_names, prior_adj))
    lines.append("")
    lines.extend(format_adjacency_matrix("Final graph", node_names, final_adj))
    lines.extend(
        [
            "",
            f"Added edges ({len(added_edges)}):",
            *format_edge_block(added_edges),
            "",
            f"Removed edges ({len(removed_edges)}):",
            *format_edge_block(removed_edges),
            "",
            "Usage Note:",
            USAGE_NOTE,
        ]
    )
    return lines


def build_memory(csv_path: str, iter_dir: str, llm_dag_path: str | None = None):
    """
    Build iteration feedback from:
    - the exact prior graph used at the start of this iteration
    - the final learned graph in `causal_graph.csv`

    The report includes only graph-structure information so that label-based
    evaluation metrics do not leak into the next LLM iteration.
    """
    if llm_dag_path is None:
        llm_dag_path = os.path.join(iter_dir, "llm_result.txt")

    prior_adj, _ = extract_graph(csv_path, llm_dag_path)

    causal_graph_path = os.path.join(iter_dir, "causal_graph.csv")
    causal_graph_df = pd.read_csv(causal_graph_path, index_col=0)
    final_adj = causal_graph_df.values
    node_names = causal_graph_df.columns.astype(str).tolist()

    added_edges = []
    removed_edges = []
    num_nodes = prior_adj.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            prior_value = to_binary_value(prior_adj[i, j])
            final_value = to_binary_value(final_adj[i, j])
            if prior_value == 0 and final_value == 1:
                added_edges.append((node_names[i], node_names[j]))
            elif prior_value == 1 and final_value == 0:
                removed_edges.append((node_names[i], node_names[j]))

    round_id = infer_round(iter_dir)
    lines = build_report_lines(
        round_id=round_id,
        node_names=node_names,
        prior_adj=prior_adj,
        final_adj=final_adj,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )

    diff_path = os.path.join(iter_dir, "diff_report.txt")
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    memory_lines = [
        "Use the following differential report as structural feedback for the next prior generation.",
        "Do not treat the listed edits as mandatory constraints; use them as review cues.",
        "",
        *lines,
    ]
    memory_path = os.path.join(iter_dir, "memory_prompt.txt")
    with open(memory_path, "w", encoding="utf-8") as f:
        f.write("\n".join(memory_lines) + "\n")

    return diff_path, memory_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--iter_dir", required=True, help="Iteration directory containing metrics and graphs.")
    parser.add_argument(
        "--llm_dag",
        default=None,
        help="Optional path to the exact prior graph file. Defaults to iter_dir/llm_result.txt.",
    )
    args = parser.parse_args()

    diff_path, memory_path = build_memory(args.csv, args.iter_dir, args.llm_dag)
    print("Diff report:", diff_path)
    print("Memory prompt:", memory_path)


if __name__ == "__main__":
    main()
