#!/usr/bin/env python3
"""
Infer missingness mechanisms from precomputed missingness statistics via an LLM.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from prompt_generate.cd_prompt import ObjTaskASI, ObjTaskASIA, ObjTaskCRAC, ObjTaskChild, ObjTaskSAN
from utils.config import (
    DEFAULT_LLM_REQUEST_RETRIES,
    DEFAULT_LLM_REQUEST_RETRY_WAIT,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
)

load_dotenv()

TEMPLATE_CLASSES = {
    "ASI": ObjTaskASI,
    "CRAC": ObjTaskCRAC,
    "SAN": ObjTaskSAN,
    "ASIA": ObjTaskASIA,
    "Child": ObjTaskChild,
}


def extract_json_object(reply_text: str):
    text = reply_text.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if match:
        text = match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("LLM reply does not contain a JSON object.")

    return json.loads(text[start : end + 1])


def normalize_openai_base_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return raw_url

    url = raw_url.rstrip("/")
    if re.search(r"/v\d+(?:[A-Za-z0-9._-]*)?$", url):
        return url
    return f"{url}/v1"


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


def build_compact_stats_summary(stats: dict, top_k_patterns: int = 12) -> dict:
    variable_missing_ratio = (stats.get("variable_missing_ratio") or {}).get("data", {}) or {}
    high_missing_variables = (stats.get("high_missing_variables") or {}).get("data", []) or []
    sample_missing_stats = (stats.get("sample_missing_stats") or {}).get("data", {}) or {}
    pattern_counts = (((stats.get("missing_patterns") or {}).get("data") or {}).get("counts") or {})
    top_patterns = dict(
        sorted(pattern_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k_patterns))]
    )

    return {
        "variable_missing_ratio": variable_missing_ratio,
        "high_missing_variables": high_missing_variables,
        "sample_missing_stats": sample_missing_stats,
        "top_missing_patterns": top_patterns,
    }



def fallback_missing_reason(stats: dict) -> dict:
    compact = build_compact_stats_summary(stats)
    ratios = compact.get("variable_missing_ratio", {}) or {}
    high = compact.get("high_missing_variables", []) or []
    variables = high or [name for name, value in ratios.items() if float(value or 0.0) > 0.0]
    analysis = {}
    for name in variables:
        analysis[str(name)] = {
            "possible_related_variables": [],
            "description": "Fallback report: LLM missingness reasoning was unavailable; use the numeric missingness statistics conservatively."
        }
    return {
        "missing_mechanisms_analysis": analysis,
        "spatio_temporal_features": {
            "temporal": "Fallback report generated from missingness statistics only.",
            "spatial": "Fallback report generated from missingness statistics only."
        },
        "bias_risk": {
            "selection_bias": "Potential bias should be assessed from variable-level and pattern-level missingness statistics.",
            "information_bias": "Potential information loss should be considered for variables with high missing ratios."
        }
    }
def run_miss_reason(
    stats_path: str,
    output_dir: str,
    model: str,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    template: str = "ASI",
    request_retries: int = DEFAULT_LLM_REQUEST_RETRIES,
    request_retry_wait: float = DEFAULT_LLM_REQUEST_RETRY_WAIT,
) -> str:
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    compact_stats = build_compact_stats_summary(stats)

    obj_task = TEMPLATE_CLASSES[template]()
    knowledge = obj_task.generate_instructor_scene()

    task_description = (
        "Return only one JSON object.\n"
        "Use the following schema:\n"
        "{\n"
        '  "missing_mechanisms_analysis": {\n'
        '    "<variable_name>": {\n'
        '      "possible_related_variables": ["..."],\n'
        '      "description": "..."\n'
        "    }\n"
        "  },\n"
        '  "spatio_temporal_features": {\n'
        '    "temporal": "...",\n'
        '    "spatial": "..."\n'
        "  },\n"
        '  "bias_risk": {\n'
        '    "selection_bias": "...",\n'
        '    "information_bias": "..."\n'
        "  }\n"
        "}\n"
        "Do not add commentary outside the JSON object."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior data analysis expert. "
                "Infer plausible missingness mechanisms and risk notes from statistics and domain knowledge."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Domain knowledge:\n{knowledge}\n\n"
                f"Missingness statistics summary:\n{json.dumps(compact_stats, ensure_ascii=False, indent=2)}\n\n"
                f"Task:\n{task_description}"
            ),
        },
    ]

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=normalize_openai_base_url(os.getenv("OPENAI_BASE_URL")),
    )
    last_exc = None
    for attempt in range(1, max(1, int(request_retries)) + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= max(1, int(request_retries)):
                raise
            wait_s = float(request_retry_wait) * attempt
            print(
                f"[miss_reason] LLM request failed on attempt {attempt}/{request_retries}: {exc}. "
                f"Retrying in {wait_s:.1f}s."
            )
            time.sleep(wait_s)
    else:
        raise last_exc

    raw = extract_message_text(response).strip()
    raw_path = os.path.join(output_dir, "miss_reason_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
    try:
        result = extract_json_object(raw)
    except Exception as exc:
        print(f"[miss_reason] Failed to parse LLM JSON; using fallback missingness report: {exc}")
        result = fallback_missing_reason(stats)

    base = os.path.splitext(os.path.basename(stats_path))[0]
    filename = f"{base}_miss_reason.json"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return out_path


if __name__ == "__main__":
    raise SystemExit("Use run_all.py to drive miss_reason.py.")
