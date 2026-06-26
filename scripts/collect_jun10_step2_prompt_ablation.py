#!/usr/bin/env python3
"""Collect Jun10 Step2 prompt-ablation outputs.

The Step2 task is a retention gate, not a final classifier:
for each candidate piece, keep all tissue hypotheses still supported by H&E.

This collector summarizes whether each prompt:
1. keeps the true class,
2. narrows the candidate to fewer remaining classes,
3. avoids over-keeping broad fallback classes such as tumor, stroma, and immune.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_label(run_dir: Path, rows: list[dict[str, str]]) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            style = config.get("prompt_style") or rows[0].get("prompt_style")
            model = str(config.get("model", "")).split("/")[-1]
            mode = config.get("image_mode", "")
            if style:
                return "__".join(part for part in [style, model, mode] if part)
        except Exception:
            pass
    style = rows[0].get("prompt_style") if rows else ""
    return style or run_dir.name


def kept_classes(row: dict[str, str]) -> list[str]:
    return [item for item in str(row.get("he_retained_classes", "")).split(";") if item]


def summarize_run(run_dir: Path, rows: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    label = run_label(run_dir, rows)
    parsed = [row for row in rows if row.get("parse_status", "ok") == "ok"]
    n = len(parsed)
    retained = sum(str(row.get("true_retained_after_step2", "")).lower() == "true" for row in parsed)
    counts = [len(kept_classes(row)) for row in parsed]
    overall = {
        "run": label,
        "run_dir": str(run_dir),
        "n": n,
        "true_retained": retained,
        "true_retained_frac": round(retained / n, 4) if n else "",
        "mean_classes_left": round(statistics.mean(counts), 3) if counts else "",
        "median_classes_left": round(statistics.median(counts), 3) if counts else "",
        "max_classes_left": max(counts) if counts else "",
        "parse_ok": str(len(parsed) == len(rows)).lower(),
    }

    by_class = []
    false_rows = []
    false_counter_by_true: dict[str, Counter[str]] = defaultdict(Counter)
    kept_total = Counter()
    misses = []
    for cls in CLASS_KEYS:
        cls_rows = [row for row in parsed if row.get("true_class") == cls]
        cls_counts = [len(kept_classes(row)) for row in cls_rows]
        cls_retained = sum(str(row.get("true_retained_after_step2", "")).lower() == "true" for row in cls_rows)
        by_class.append(
            {
                "run": label,
                "true_class": cls,
                "n": len(cls_rows),
                "true_retained": cls_retained,
                "true_retained_frac": round(cls_retained / len(cls_rows), 4) if cls_rows else "",
                "mean_classes_left": round(statistics.mean(cls_counts), 3) if cls_counts else "",
            }
        )

    for row in parsed:
        true_class = row.get("true_class", "")
        kept = kept_classes(row)
        for cls in kept:
            kept_total[cls] += 1
            if cls != true_class:
                false_counter_by_true[true_class][cls] += 1
        if str(row.get("true_retained_after_step2", "")).lower() != "true":
            misses.append(
                {
                    "run": label,
                    "candidate_uid": row.get("candidate_uid", ""),
                    "true_class": true_class,
                    "he_retained_classes": ";".join(kept),
                    "he_retained_count": len(kept),
                    "true_class_he_support_score": row.get("true_class_he_support_score", ""),
                    "top_he_support_class": row.get("top_he_support_class", ""),
                    "top_he_support_score": row.get("top_he_support_score", ""),
                }
            )

    for true_class in CLASS_KEYS:
        for false_class in CLASS_KEYS:
            if false_class == true_class:
                continue
            false_rows.append(
                {
                    "run": label,
                    "true_class": true_class,
                    "false_class_kept": false_class,
                    "count": false_counter_by_true[true_class][false_class],
                }
            )
    for cls in CLASS_KEYS:
        false_rows.append(
            {
                "run": label,
                "true_class": "ALL",
                "false_class_kept": cls,
                "count": kept_total[cls],
            }
        )
    return overall, by_class, false_rows, misses


def find_run_dirs(input_roots: list[Path]) -> list[Path]:
    dirs = []
    for root in input_roots:
        if root.is_file() and root.name == "step2_candidate_retention_summary.csv":
            dirs.append(root.parent)
        elif (root / "step2_candidate_retention_summary.csv").exists():
            dirs.append(root)
        elif root.exists():
            dirs.extend(path.parent for path in root.rglob("step2_candidate_retention_summary.csv"))
    return sorted(set(dirs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weak58-baseline-retained", type=int, default=57)
    parser.add_argument("--weak58-baseline-mean", type=float, default=3.879)
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.input_root)
    if not run_dirs:
        raise SystemExit("No step2_candidate_retention_summary.csv files found")

    overall_rows: list[dict[str, object]] = []
    by_class_rows: list[dict[str, object]] = []
    false_rows: list[dict[str, object]] = []
    miss_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        rows = read_csv(run_dir / "step2_candidate_retention_summary.csv")
        overall, by_class, false_kept, misses = summarize_run(run_dir, rows)
        overall_rows.append(overall)
        by_class_rows.extend(by_class)
        false_rows.extend(false_kept)
        miss_rows.extend(misses)

    overall_rows = sorted(overall_rows, key=lambda row: (-int(row["true_retained"]), float(row["mean_classes_left"] or 999)))
    write_csv(args.output_dir / "step2_prompt_ablation_overall.csv", overall_rows)
    write_csv(args.output_dir / "step2_prompt_ablation_by_class.csv", by_class_rows)
    write_csv(args.output_dir / "step2_prompt_ablation_false_kept_matrix.csv", false_rows)
    write_csv(args.output_dir / "step2_prompt_ablation_misses.csv", miss_rows)

    lines = [
        "# Jun10 Step2 prompt-ablation collector",
        "",
        "Step2 is the H&E morphology verification step. It should retain the true tissue class while narrowing false tissue hypotheses.",
        "",
        "## Overall ranking",
        "",
        "| run | true retained | n | mean classes left | gate |",
        "|---|---:|---:|---:|---|",
    ]
    for row in overall_rows:
        gate = ""
        if int(row["n"]) == 58:
            passes = int(row["true_retained"]) >= args.weak58_baseline_retained - 1 and float(row["mean_classes_left"]) < args.weak58_baseline_mean
            gate = "expand candidate" if passes else "do not expand"
        lines.append(
            f"| {row['run']} | {row['true_retained']} | {row['n']} | {row['mean_classes_left']} | {gate} |"
        )
    lines.extend(
        [
            "",
            "Gate for weak58: expand only if it keeps at least 56/58 true classes and reduces the old mean classes-left baseline of 3.879.",
            "",
        ]
    )
    (args.output_dir / "step2_prompt_ablation_report.md").write_text("\n".join(lines))
    print(args.output_dir)


if __name__ == "__main__":
    main()
