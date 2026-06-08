#!/usr/bin/env python3
"""
Standalone helper for generating a merged missingness report.
Prefer run_all.py for normal project usage.
"""

import argparse
import json
import os
from collections import OrderedDict

from dotenv import load_dotenv

from miss_calculate import run_miss_calculate
from miss_reason import run_miss_reason

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", default="ASI", choices=["ASI", "CRAC", "SAN", "ASIA", "Child"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL")):
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL must be set in the environment.")

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    calc_path = run_miss_calculate(args.data_file, args.out_dir)
    print("missing stats:", calc_path)

    temp_reason_path = run_miss_reason(
        stats_path=calc_path,
        output_dir=args.out_dir,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        template=args.template,
    )

    base = os.path.splitext(os.path.basename(args.data_file))[0]
    correct_reason = os.path.join(args.out_dir, f"{base}_miss_reason.json")
    if os.path.abspath(temp_reason_path) != os.path.abspath(correct_reason):
        os.replace(temp_reason_path, correct_reason)
    reason_path = correct_reason
    print("missing reason:", reason_path)

    final_path = os.path.join(args.out_dir, f"{base}_miss_report.json")
    with open(calc_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(reason_path, "r", encoding="utf-8") as f:
        reason = json.load(f)

    report = OrderedDict()
    report["introduction"] = "Merged missingness statistics and mechanism analysis."
    report["variable_missing_ratio"] = stats["variable_missing_ratio"]
    report["high_missing_variables"] = stats["high_missing_variables"]
    report["missing_mechanisms_analysis"] = (
        reason.get("missing_mechanisms_analysis") or reason.get("missing_mechanisms") or {}
    )

    patterns = stats["missing_patterns"]["data"]["counts"]
    top10 = dict(sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)[:10])
    report["missing_patterns"] = {
        "description": stats["missing_patterns"]["description"],
        "data": {"counts": top10},
    }
    report["spatio_temporal_features"] = reason.get("spatio_temporal_features", {})
    report["bias_risk"] = reason.get("bias_risk", {})

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print("merged report:", final_path)


if __name__ == "__main__":
    main()
