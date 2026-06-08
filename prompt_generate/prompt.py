# prompt_generate/prompt.py
# -*- coding: utf-8 -*-
"""
Generate LLM priors from prompt templates and save:
  - prompt_full.txt
  - prompt_chunk_XXX.txt
  - llm_raw.txt
  - llm_raw_chunk_XXX.txt
  - llm_results_by_triple.json
  - llm_result_all.txt
  - llm_result.txt
  - llm_confidence_matrix.csv
  - llm_adj_matrix_dag.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

try:
    from cd_prompt import ObjTaskASI, ObjTaskASIA, ObjTaskCRAC, ObjTaskChild, ObjTaskSAN
except ModuleNotFoundError:
    from prompt_generate.cd_prompt import ObjTaskASI, ObjTaskASIA, ObjTaskCRAC, ObjTaskChild, ObjTaskSAN

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import (
    DEFAULT_LLM_REQUEST_RETRIES,
    DEFAULT_LLM_REQUEST_RETRY_WAIT,
    DEFAULT_LLM_REQUEST_INTERVAL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_TRIPLE_CHUNK_SIZE,
)

TEMPLATE_CLASSES = {
    "ASI": ObjTaskASI,
    "CRAC": ObjTaskCRAC,
    "SAN": ObjTaskSAN,
    "ASIA": ObjTaskASIA,
    "Child": ObjTaskChild,
}
SYSTEM_PROMPT = (
    "You are a careful causal discovery assistant. "
    "Return only valid JSON that strictly follows the user's schema."
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_openai_base_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return raw_url

    url = raw_url.rstrip("/")
    if re.search(r"/v\d+(?:[A-Za-z0-9._-]*)?$", url):
        return url
    return f"{url}/v1"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", required=True)
    ap.add_argument("--miss_report", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--template", default="ASI", choices=list(TEMPLATE_CLASSES))
    ap.add_argument("--model", required=True, help="LLM model name.")
    ap.add_argument("--temperature", "--llm_temperature", dest="temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    ap.add_argument("--timeout", "--llm_timeout", dest="timeout", type=int, default=DEFAULT_LLM_TIMEOUT)
    ap.add_argument("--memory_file", default=None)
    ap.add_argument(
        "--triple_chunk_size",
        type=int,
        default=DEFAULT_TRIPLE_CHUNK_SIZE,
        help="Number of triples sent to the LLM per request.",
    )
    ap.add_argument(
        "--request_retries",
        "--llm_request_retries",
        dest="request_retries",
        type=int,
        default=DEFAULT_LLM_REQUEST_RETRIES,
        help="Retry count for transient LLM connection failures.",
    )
    ap.add_argument(
        "--request_retry_wait",
        "--llm_request_retry_wait",
        dest="request_retry_wait",
        type=float,
        default=DEFAULT_LLM_REQUEST_RETRY_WAIT,
        help="Base wait seconds between LLM retries.",
    )
    ap.add_argument(
        "--request_interval",
        "--llm_request_interval",
        dest="request_interval",
        type=float,
        default=DEFAULT_LLM_REQUEST_INTERVAL,
        help="Sleep seconds after each successful LLM prompt chunk.",
    )
    ap.add_argument(
        "--query_mode",
        choices=["triple", "pairwise"],
        default="triple",
        help="Use triple queries for the full method or pairwise queries for the w/o TQ ablation.",
    )
    ap.add_argument(
        "--top_k_triplet_aux",
        default="all",
        help="For triple mode, use all triples or select top-k auxiliary variables per variable pair.",
    )
    return ap.parse_args()


def edge_entropy(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [value / total for value in counts.values() if value > 0]
    return -sum(p * np.log2(p) for p in probs)


def save_edges_with_conf(edges, confidences, path):
    with open(path, "w", encoding="utf-8") as f:
        for u, v in edges:
            conf = confidences.get((u, v), 0.0)
            f.write(f"({u},{v},{conf:.4f})\n")


def extract_json_object(reply_text: str):
    text = reply_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    first_start = text.find("{")
    if first_start == -1:
        raise RuntimeError("LLM reply does not contain a JSON object.")

    decoder = json.JSONDecoder()
    last_error = None
    for start in (idx for idx, char in enumerate(text) if char == "{"):
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    try:
        return json.loads(text[first_start:])
    except json.JSONDecodeError as exc:
        last_error = last_error or exc
        raise RuntimeError(f"Failed to parse LLM JSON reply: {last_error}") from exc


def extract_message_text(response):
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unsupported response payload: {response!r}")

    choices = getattr(response, "choices", None)
    if choices:
        message = choices[0].message
        return message.content or ""

    if hasattr(response, "output_text"):
        return response.output_text or ""

    raise RuntimeError(f"Unsupported response type: {type(response)!r}")


def extract_usage(response) -> dict:
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)

    if usage is None:
        return {}

    def usage_get(name):
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    return {
        "prompt_tokens": usage_get("prompt_tokens"),
        "completion_tokens": usage_get("completion_tokens"),
        "total_tokens": usage_get("total_tokens"),
    }


def make_triple_key(triple) -> str:
    return "|".join(triple)


def iter_chunks(seq, size: int):
    if size <= 0:
        raise ValueError("--triple_chunk_size must be a positive integer.")
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def vector_for_association(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(0.2 * len(series))):
        return numeric.to_numpy(dtype=float)
    codes, _ = pd.factorize(series, sort=True)
    out = codes.astype(float)
    out[codes < 0] = np.nan
    return out


def abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return 0.0
    aa = a[mask].astype(float)
    bb = b[mask].astype(float)
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return 0.0
    val = float(np.corrcoef(aa, bb)[0, 1])
    return abs(val) if np.isfinite(val) else 0.0


def parse_top_k_triplet_aux(value: str, n_nodes: int) -> int | None:
    text = str(value or "all").strip().lower()
    if text in {"all", "none", "full"}:
        return None
    k = int(text)
    if k <= 0:
        raise ValueError("--top_k_triplet_aux must be 'all' or a positive integer.")
    return min(k, max(0, n_nodes - 2))


def select_topk_triples(datapath: str, headers_long: list[str], short_headers: list[str], top_k_value: str):
    k = parse_top_k_triplet_aux(top_k_value, len(headers_long))
    all_triples = list(itertools.combinations(short_headers, 3))
    if k is None or len(headers_long) <= 2:
        return all_triples, {
            "top_k_triplet_aux": "all",
            "all_triples": len(all_triples),
            "selected_triples": len(all_triples),
            "selection_method": "all_combinations",
        }

    df = pd.read_csv(datapath)
    data_vectors = [vector_for_association(df[col]) for col in headers_long]
    miss_vectors = [df[col].isna().astype(float).to_numpy() for col in headers_long]
    n = len(headers_long)
    assoc = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            data_score = abs_corr(data_vectors[i], data_vectors[j])
            miss_score = abs_corr(miss_vectors[i], miss_vectors[j])
            score = data_score + miss_score
            assoc[i, j] = score
            assoc[j, i] = score

    selected_index_triples = set()
    for i in range(n):
        for j in range(i + 1, n):
            candidates = []
            for c in range(n):
                if c == i or c == j:
                    continue
                score = max(assoc[i, c], assoc[j, c]) + 0.25 * min(assoc[i, c], assoc[j, c])
                candidates.append((float(score), c))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _score, c in candidates[:k]:
                selected_index_triples.add(tuple(sorted((i, j, c))))

    ordered = sorted(selected_index_triples)
    triples = [tuple(short_headers[idx] for idx in triple) for triple in ordered]
    return triples, {
        "top_k_triplet_aux": k,
        "all_triples": len(all_triples),
        "selected_triples": len(triples),
        "selection_method": "pairwise_topk_aux_by_data_and_missingness_correlation",
    }


def sanitize_edges(edges, triple):
    if edges is None:
        return []
    if not isinstance(edges, list):
        raise RuntimeError(f"edges must be a list, got {type(edges)!r}")

    allowed = set(triple)
    normalized = []
    seen = set()

    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise RuntimeError(f"Each edge must be a 2-item list, got {edge!r}")
        u, v = edge
        if not isinstance(u, str) or not isinstance(v, str):
            raise RuntimeError(f"Edge endpoints must be strings, got {edge!r}")
        if u not in allowed or v not in allowed:
            raise RuntimeError(f"Edge {edge!r} uses nodes outside the triple {triple!r}")
        if u == v:
            continue
        pair = (u, v)
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append([u, v])

    edge_set = {(u, v) for u, v in normalized}
    conflicted = {
        frozenset((u, v))
        for u, v in edge_set
        if u != v and (v, u) in edge_set
    }

    cleaned = []
    for u, v in normalized:
        if frozenset((u, v)) in conflicted:
            continue
        cleaned.append([u, v])
    return cleaned


def coerce_batch_results(payload, expected_items, item_field: str = "triple"):
    expected_keys = [make_triple_key(item) for item in expected_items]

    if not isinstance(payload, dict):
        raise RuntimeError("LLM reply JSON must be an object.")

    if "results" in payload:
        results = payload["results"]
        if not isinstance(results, list):
            raise RuntimeError("The `results` field must be a list.")
    else:
        # Backward-compatible fallback: treat the top-level object as {key: edges}.
        results = [
            {item_field: list(item), "key": key, "edges": payload.get(key, [])}
            for item, key in zip(expected_items, expected_keys)
        ]

    if len(results) != len(expected_items):
        raise RuntimeError(
            f"LLM returned {len(results)} results, but {len(expected_items)} query items were expected."
        )

    normalized = OrderedDict()
    fallback_field = "pair" if item_field == "triple" else "triple"
    for result, item, expected_key in zip(results, expected_items, expected_keys):
        if not isinstance(result, dict):
            raise RuntimeError(f"Each result item must be an object, got {type(result)!r}")

        expected_item = list(item)
        item_value = result.get(item_field, result.get(fallback_field, expected_item))
        if item_value != expected_item:
            raise RuntimeError(
                f"Result {item_field} mismatch. Expected {expected_item!r}, got {item_value!r}."
            )

        key_value = result.get("key", expected_key)
        if key_value != expected_key:
            raise RuntimeError(
                f"Result key mismatch. Expected {expected_key!r}, got {key_value!r}."
            )

        edges_value = sanitize_edges(result.get("edges", []), expected_item)
        normalized[expected_key] = edges_value

    return normalized


def build_counts_from_triples(triple_relations_short, triples_short):
    counts_short = {}
    for triple in triples_short:
        key = make_triple_key(triple)
        nodes = list(triple)
        edge_set = {tuple(edge) for edge in triple_relations_short.get(key, [])}
        for u, v in itertools.permutations(nodes, 2):
            counts_short.setdefault((u, v), {"yes": 0, "no": 0})
            if (u, v) in edge_set:
                counts_short[(u, v)]["yes"] += 1
            else:
                counts_short[(u, v)]["no"] += 1
    return counts_short


def create_completion(client: OpenAI, args, prompt_text: str):
    last_exc = None
    for attempt in range(1, max(1, args.request_retries) + 1):
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=args.temperature,
                timeout=args.timeout,
            )
            usage = extract_usage(response)
            usage["attempt"] = attempt
            usage["request_elapsed_sec"] = round(time.time() - started, 3)
            reply_text = extract_message_text(response).strip()
            if not reply_text:
                raise RuntimeError("LLM returned an empty reply.")
            return reply_text, usage
        except Exception as exc:
            last_exc = exc
            if attempt >= max(1, args.request_retries):
                break
            wait_s = args.request_retry_wait * attempt
            print(
                f"[prompt] LLM request failed on attempt {attempt}/{args.request_retries}: {exc}. "
                f"Retrying in {wait_s:.1f}s."
            )
            time.sleep(wait_s)
    raise last_exc


def build_prompt_text(base_prompt: str, query_batch, miss_report_text: str, memory_text: str) -> str:
    query_json_short = json.dumps([list(t) for t in query_batch], ensure_ascii=False)
    return (
        base_prompt.replace("{{TRIPLES_JSON_USING_SHORTNAMES}}", query_json_short)
        .replace("{{PAIRS_JSON_USING_SHORTNAMES}}", query_json_short)
        .replace("{{MISSING_REPORT_JSON}}", miss_report_text)
        .replace("{{DIFF_FEEDBACK_TEXT}}", memory_text or "")
    )



def build_pairwise_prompt_template(obj) -> str:
    short_map_json = json.dumps(obj.get_shortname_map(), ensure_ascii=False, indent=2)
    return f"""
Role:
You are a cautious causal discovery assistant. This is the pairwise-query
variant used for ablation. For each input pair, judge whether there is a
reliable direct directed edge inside the pair.

Dataset:
{obj.DATASET_NAME}

Output contract:
Return only one JSON object with exactly one top-level key, "results".
Each result must follow this schema:

{{
  "results": [
    {{
      "pair": ["X", "Y"],
      "key": "X|Y",
      "edges": [["X", "Y"]]
    }}
  ]
}}

Strict rules:
- The number and order of results must match the input pairs.
- Each "key" is the pair joined by "|".
- Each edge is [source, target].
- Use only short names appearing in the current pair.
- Do not output self-loops, duplicate edges, or both directions for one pair.
- Return an empty edge list when evidence is insufficient.

Variable short-name dictionary:
{short_map_json}

Current pair batch:
{{{{PAIRS_JSON_USING_SHORTNAMES}}}}

Domain knowledge:
{obj.generate_instructor_scene()}

Missingness report:
{{{{MISSING_REPORT_JSON}}}}

Differential feedback:
{{{{DIFF_FEEDBACK_TEXT}}}}
"""

def fetch_batch_relations(
    client: OpenAI,
    args,
    base_prompt: str,
    miss_report_text: str,
    memory_text: str,
    triple_batch,
    batch_tag: str,
    combined_prompt_parts: list[str],
    combined_raw_parts: list[str],
    usage_rows: list[dict],
):
    prompt_text = build_prompt_text(base_prompt, triple_batch, miss_report_text, memory_text)
    prompt_path = os.path.join(args.out_dir, f"prompt_chunk_{batch_tag}.txt")
    save_text(prompt_path, prompt_text)
    combined_prompt_parts.append(f"===== chunk {batch_tag} =====\n{prompt_text}\n")

    def retry_smaller(exc: Exception):
        if len(triple_batch) <= 1:
            raise RuntimeError(
                f"{exc} Chunk {batch_tag} could not be recovered even after shrinking to a single triple."
            ) from exc

        mid = max(1, len(triple_batch) // 2)
        left_batch = triple_batch[:mid]
        right_batch = triple_batch[mid:]
        print(
            f"[prompt] chunk {batch_tag} failed ({exc}); "
            f"retrying with smaller batches: {len(left_batch)} + {len(right_batch)}."
        )

        normalized = OrderedDict()
        normalized.update(
            fetch_batch_relations(
                client,
                args,
                base_prompt,
                miss_report_text,
                memory_text,
                left_batch,
                f"{batch_tag}a",
                combined_prompt_parts,
                combined_raw_parts,
                usage_rows,
            )
        )
        normalized.update(
            fetch_batch_relations(
                client,
                args,
                base_prompt,
                miss_report_text,
                memory_text,
                right_batch,
                f"{batch_tag}b",
                combined_prompt_parts,
                combined_raw_parts,
                usage_rows,
            )
        )
        return normalized

    try:
        reply, usage = create_completion(client, args, prompt_text)
    except Exception as exc:
        return retry_smaller(exc)

    raw_path = os.path.join(args.out_dir, f"llm_raw_chunk_{batch_tag}.txt")
    save_text(raw_path, reply)
    combined_raw_parts.append(f"===== chunk {batch_tag} =====\n{reply}\n")
    usage_rows.append(
        {
            "batch": batch_tag,
            "model": args.model,
            "triples": len(triple_batch),
            "prompt_chars": len(prompt_text),
            "reply_chars": len(reply),
            **usage,
        }
    )

    try:
        payload = extract_json_object(reply)
        item_field = "pair" if args.query_mode == "pairwise" else "triple"
        return coerce_batch_results(payload, triple_batch, item_field=item_field)
    except RuntimeError as exc:
        return retry_smaller(exc)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=normalize_openai_base_url(os.getenv("OPENAI_BASE_URL")),
    )

    headers_long = pd.read_csv(args.datapath, nrows=0).columns.tolist()

    obj = TEMPLATE_CLASSES[args.template]()
    if args.query_mode == "pairwise":
        base_prompt = build_pairwise_prompt_template(obj)
    else:
        base_prompt = obj.generate_prompt_qa(add_cot=False)["prompt"]
    short2long = obj.get_shortname_map()
    long2short = {v: k for k, v in short2long.items()}

    short_headers = []
    for long_name in headers_long:
        short_name = long2short.get(long_name)
        if short_name is None:
            short_name = re.sub(r"[^A-Za-z0-9]", "", long_name.upper())[:5] or long_name
        short_headers.append(short_name)

    if len(set(short_headers)) != len(short_headers):
        raise RuntimeError(f"Short names are not unique: {short_headers!r}")

    item_size = 2 if args.query_mode == "pairwise" else 3
    item_field = "pair" if args.query_mode == "pairwise" else "triple"
    if args.query_mode == "pairwise":
        triples_short = list(itertools.combinations(short_headers, item_size))
        query_selection_summary = {
            "query_mode": "pairwise",
            "top_k_triplet_aux": None,
            "all_triples": None,
            "selected_triples": len(triples_short),
            "selection_method": "all_pairs",
        }
    else:
        triples_short, query_selection_summary = select_topk_triples(
            args.datapath,
            headers_long,
            short_headers,
            args.top_k_triplet_aux,
        )
    print(f"[query] selected {len(triples_short)} {item_field}s: {query_selection_summary}")

    with open(args.miss_report, "r", encoding="utf-8") as f:
        miss_report_text = f.read().strip()

    memory_text = ""
    if args.memory_file and os.path.exists(args.memory_file):
        with open(args.memory_file, "r", encoding="utf-8") as f:
            memory_text = f.read().strip()

    triple_relations_short = OrderedDict()
    combined_prompt_parts = []
    combined_raw_parts = []
    usage_rows = []
    run_started = time.time()

    for batch_idx, triple_batch in enumerate(iter_chunks(triples_short, args.triple_chunk_size), start=1):
        normalized_batch = fetch_batch_relations(
            client,
            args,
            base_prompt,
            miss_report_text,
            memory_text,
            triple_batch,
            f"{batch_idx:03d}",
            combined_prompt_parts,
            combined_raw_parts,
            usage_rows,
        )
        triple_relations_short.update(normalized_batch)
        if args.request_interval > 0:
            time.sleep(float(args.request_interval))

    save_text(os.path.join(args.out_dir, "prompt_full.txt"), "\n".join(combined_prompt_parts))
    save_text(os.path.join(args.out_dir, "llm_raw.txt"), "\n".join(combined_raw_parts))
    token_keys = ["prompt_tokens", "completion_tokens", "total_tokens"]
    usage_summary = {
        "model": args.model,
        "temperature": args.temperature,
        "query_mode": args.query_mode,
        "query_size": item_size,
        "triple_chunk_size": args.triple_chunk_size,
        "query_count": len(triples_short),
        **query_selection_summary,
        "successful_requests": len(usage_rows),
        "run_elapsed_sec": round(time.time() - run_started, 3),
    }
    for key in token_keys:
        values = [row.get(key) for row in usage_rows if row.get(key) is not None]
        usage_summary[key] = int(sum(values)) if values else None
    usage_summary["request_elapsed_sec"] = round(
        sum(float(row.get("request_elapsed_sec") or 0.0) for row in usage_rows),
        3,
    )
    save_json(
        os.path.join(args.out_dir, "llm_usage.json"),
        {"summary": usage_summary, "requests": usage_rows},
    )

    save_json(os.path.join(args.out_dir, "query_selection_summary.json"), query_selection_summary)

    results_payload = {
        "query_mode": args.query_mode,
        "query_selection": query_selection_summary,
        "results": [
            {item_field: list(item), "key": make_triple_key(item), "edges": triple_relations_short[make_triple_key(item)]}
            for item in triples_short
        ]
    }
    save_json(os.path.join(args.out_dir, "llm_results_by_triple.json"), results_payload)

    counts_short = build_counts_from_triples(triple_relations_short, triples_short)

    def to_long(short_name):
        return short2long.get(short_name, short_name)

    counts_long = {}
    for (short_u, short_v), count in counts_short.items():
        long_u, long_v = to_long(short_u), to_long(short_v)
        counts_long[(long_u, long_v)] = dict(count)

    raw_edges_long = [(u, v) for (u, v), c in counts_long.items() if c["yes"] > 0]
    conf_all_long = {
        (u, v): c["yes"] / max(1, c["yes"] + c["no"])
        for (u, v), c in counts_long.items()
        if c["yes"] > 0
    }
    p_all = os.path.join(args.out_dir, "llm_result_all.txt")
    save_edges_with_conf(raw_edges_long, conf_all_long, p_all)
    print(f"[prior] saved raw prior edges to {p_all} ({len(raw_edges_long)} edges)")

    graph = nx.DiGraph()
    for (u, v), c in counts_long.items():
        if c["yes"] > c["no"]:
            graph.add_edge(u, v)

    entropy = {edge: edge_entropy(counts_long[edge]) for edge in graph.edges()}
    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
            removable = max(cycle, key=lambda e: entropy.get((e[0], e[1]), 0.0))
            graph.remove_edge(removable[0], removable[1])
        except nx.NetworkXNoCycle:
            break

    dag_edges_long = list(graph.edges())
    conf_dag_long = {}
    for u, v in dag_edges_long:
        count = counts_long[(u, v)]
        conf_dag_long[(u, v)] = count["yes"] / max(1, count["yes"] + count["no"])

    p_dag = os.path.join(args.out_dir, "llm_result.txt")
    save_edges_with_conf(dag_edges_long, conf_dag_long, p_dag)
    print(f"[prior] saved acyclic prior edges to {p_dag}")

    idx = {name: i for i, name in enumerate(headers_long)}
    n = len(headers_long)

    conf_mat = np.zeros((n, n), dtype=float)
    for (u, v), c in counts_long.items():
        if u in idx and v in idx:
            total = c["yes"] + c["no"]
            conf_mat[idx[u], idx[v]] = (c["yes"] / total) if total > 0 else 0.0
    pd.DataFrame(conf_mat, index=headers_long, columns=headers_long).to_csv(
        os.path.join(args.out_dir, "llm_confidence_matrix.csv")
    )

    adj_dag = np.zeros((n, n), dtype=int)
    for u, v in dag_edges_long:
        if u in idx and v in idx:
            adj_dag[idx[u], idx[v]] = 1
    pd.DataFrame(adj_dag, index=headers_long, columns=headers_long).to_csv(
        os.path.join(args.out_dir, "llm_adj_matrix_dag.csv")
    )


if __name__ == "__main__":
    main()
