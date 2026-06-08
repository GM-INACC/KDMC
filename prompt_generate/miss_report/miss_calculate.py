#!/usr/bin/env python3

import json
import os
import warnings
from collections import Counter, defaultdict

import pandas as pd
from scipy.stats import pointbiserialr


def compute_missing_report(df: pd.DataFrame) -> dict:
    report = {}

    report["variable_missing_ratio"] = {
        "description": "Missing-value percentage for each variable.",
        "data": (df.isna().mean() * 100).round(2).to_dict(),
    }

    high = [var for var, pct in report["variable_missing_ratio"]["data"].items() if pct > 30.0]
    report["high_missing_variables"] = {
        "description": "Variables whose missing-value percentage exceeds 30%.",
        "data": high,
    }

    counts = df.isna().sum(axis=1)
    desc = counts.describe()
    report["sample_missing_stats"] = {
        "description": "Descriptive statistics for the number of missing variables per sample.",
        "data": {
            "mean": round(float(desc["mean"]), 2),
            "min": int(desc["min"]),
            "max": int(desc["max"]),
            "25%": int(desc["25%"]),
            "50%": int(desc["50%"]),
            "75%": int(desc["75%"]),
        },
    }

    patterns = []
    for _, row in df.isna().iterrows():
        missing_cols = list(row[row].index)
        patterns.append("|".join(sorted(missing_cols)) if missing_cols else "<no_missing>")
    cnt = Counter(patterns)
    total = max(1, len(df))
    report["missing_patterns"] = {
        "description": "Counts and percentages of row-wise missingness patterns.",
        "data": {
            "counts": dict(cnt),
            "proportions": {pat: round(c / total * 100, 2) for pat, c in cnt.items()},
        },
    }

    na = df.isna()
    cond = defaultdict(dict)
    for a in df.columns:
        mask = na[a]
        if int(mask.sum()) > 0:
            sub = na[mask]
            for b in df.columns:
                cond[a][b] = round(float(sub[b].mean()) * 100, 2)
        else:
            for b in df.columns:
                cond[a][b] = 0.0
    report["conditional_missing_ratios"] = {
        "description": "Conditional missingness percentage P(B missing | A missing).",
        "data": cond,
    }

    na_df = df.isna().astype(int)
    missing_vs_other_corr = defaultdict(dict)
    num_cols = df.select_dtypes(include="number").columns
    for a in df.columns:
        indicator = na_df[a]
        for b in num_cols:
            mask = df[b].notna()
            if int(mask.sum()) > 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        r, _ = pointbiserialr(indicator[mask], df[b][mask])
                        corr = 0.0 if pd.isna(r) else round(float(r), 3)
                    except Exception:
                        corr = 0.0
            else:
                corr = 0.0
            missing_vs_other_corr[a][b] = corr
    report["missing_value_correlations"] = {
        "description": "Point-biserial correlations between missingness indicators and observed numeric variables.",
        "data": missing_vs_other_corr,
    }

    return report


def run_miss_calculate(csv_path: str, output_dir: str) -> str:
    df = pd.read_csv(csv_path)
    report = compute_missing_report(df)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    filename = f"{base}_miss_calculate.json"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python miss_calculate.py <input_csv_path> <output_dir>")
        sys.exit(1)
    csv_file, out_dir = sys.argv[1:]
    path = run_miss_calculate(csv_file, out_dir)
    print("Missingness statistics saved to:", path)
