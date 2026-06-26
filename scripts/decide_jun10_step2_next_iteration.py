#!/usr/bin/env python3
"""Decide the next Jun10 Step2 prompt iteration from collected sweep outputs.

Step2 is a retention gate, not a final classifier. The desired behavior is:
keep the true tissue class while reducing unsupported alternatives. This script
turns collected sweep CSVs into an explicit next-action recommendation so prompt
iteration is driven by measured failures rather than ad-hoc prompt edits.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
STRUCTURAL = {"bronchiola", "alveoli", "vessels"}
BROAD = {"tumor", "stroma", "immune_infiltration"}


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


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def as_float(value: object, default: float = 999.0) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def find_collector_dirs(root: Path) -> list[Path]:
    if (root / "step2_prompt_ablation_overall.csv").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("step2_prompt_ablation_overall.csv"))


def load_all(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    overall: list[dict[str, object]] = []
    by_class: list[dict[str, object]] = []
    false_rows: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    for collector in find_collector_dirs(root):
        source = collector.name
        for name, target in [
            ("step2_prompt_ablation_overall.csv", overall),
            ("step2_prompt_ablation_by_class.csv", by_class),
            ("step2_prompt_ablation_false_kept_matrix.csv", false_rows),
            ("step2_prompt_ablation_misses.csv", misses),
        ]:
            path = collector / name
            if not path.exists():
                continue
            for row in read_csv(path):
                row = dict(row)
                row["collector"] = source
                target.append(row)
    return overall, by_class, false_rows, misses


def unique_runs(overall: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in overall:
        run = str(row.get("run", ""))
        collector = str(row.get("collector", ""))
        n = as_int(row.get("n", 0))
        key = (collector, run, str(n))
        retained = as_int(row.get("true_retained", 0))
        mean_left = as_float(row.get("mean_classes_left", ""))
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = row
            continue
        if (retained, -mean_left) > (as_int(current.get("true_retained", 0)), -as_float(current.get("mean_classes_left", ""))):
            best_by_key[key] = row
    return list(best_by_key.values())


def score_run(row: dict[str, object]) -> float:
    n = as_int(row.get("n", 0))
    retained = as_int(row.get("true_retained", 0))
    if n <= 0:
        return -999
    mean_left = as_float(row.get("mean_classes_left", ""))
    retention = retained / n
    return retention * 100.0 - max(mean_left - 1.0, 0.0) * 2.5


def classify_run(row: dict[str, object], weak58_baseline_mean: float) -> str:
    n = as_int(row.get("n", 0))
    retained = as_int(row.get("true_retained", 0))
    mean_left = as_float(row.get("mean_classes_left", ""))
    if n == 58:
        if retained >= 56 and mean_left < weak58_baseline_mean:
            return "expand_to_full167"
        if retained < 56:
            return "too_many_true_classes_lost"
        return "retains_true_but_not_narrow_enough"
    if n == 167:
        if retained >= 162 and mean_left <= 3.4:
            return "candidate_full167_policy"
        if retained >= 162:
            return "retains_true_but_not_narrow_enough"
        return "too_many_true_classes_lost"
    return "diagnostic_only"


def family_for_false_keep(true_class: str, false_class: str) -> str:
    if true_class in STRUCTURAL and false_class in {"stroma", "immune_infiltration"}:
        return "structural_to_stroma_or_immune"
    if true_class in STRUCTURAL and false_class == "tumor":
        return "structural_to_tumor"
    if true_class == "immune_infiltration" and false_class == "stroma":
        return "immune_to_stroma"
    if true_class == "stroma" and false_class == "immune_infiltration":
        return "stroma_to_immune"
    if false_class in BROAD:
        return "broad_false_keep"
    return "other"


def summarize_failures(false_rows: list[dict[str, object]], misses: list[dict[str, object]], selected_runs: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    max_pair_counts: dict[tuple[str, str, str], int] = {}
    for row in false_rows:
        run = str(row.get("run", ""))
        if run not in selected_runs:
            continue
        true_class = str(row.get("true_class", ""))
        false_class = str(row.get("false_class_kept", ""))
        if true_class == "ALL" or true_class == false_class:
            continue
        key = (run, true_class, false_class)
        max_pair_counts[key] = max(max_pair_counts.get(key, 0), as_int(row.get("count", 0)))
    for (_, true_class, false_class), count in max_pair_counts.items():
        counts[family_for_false_keep(true_class, false_class)] += count
    seen_misses = set()
    for row in misses:
        run = str(row.get("run", ""))
        if run not in selected_runs:
            continue
        key = (run, row.get("candidate_uid", ""), row.get("true_class", ""), row.get("he_retained_classes", ""))
        if key in seen_misses:
            continue
        seen_misses.add(key)
        true_class = str(row.get("true_class", ""))
        retained = str(row.get("he_retained_classes", ""))
        top = str(row.get("top_he_support_class", ""))
        if true_class in STRUCTURAL and (top in BROAD or any(x in retained.split(";") for x in BROAD)):
            counts["structural_true_deleted_by_broad_label"] += 1
        elif true_class in STRUCTURAL:
            counts["structural_true_deleted"] += 1
        else:
            counts["nonstructural_true_deleted"] += 1
    return counts


def recommend(failure_counts: Counter[str], best_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    recs: list[dict[str, object]] = []
    if failure_counts["structural_to_stroma_or_immune"] or failure_counts["structural_true_deleted_by_broad_label"]:
        recs.append(
            {
                "priority": len(recs) + 1,
                "action": "run_prompt",
                "value": "comparative_alveoli_septa_rescue",
                "reason": "Structural bronchiola/alveoli/vessels pieces are being deleted or absorbed into stroma/immune.",
            }
        )
    if failure_counts["immune_to_stroma"] or failure_counts["stroma_to_immune"] or failure_counts["broad_false_keep"]:
        recs.append(
            {
                "priority": len(recs) + 1,
                "action": "run_prompt",
                "value": "comparative_stroma_immune_strict_veto",
                "reason": "Broad labels remain over-kept; require direct candidate-mask evidence for stroma/immune.",
            }
        )
    if any(str(row.get("decision", "")) == "too_many_true_classes_lost" for row in best_rows):
        recs.append(
            {
                "priority": len(recs) + 1,
                "action": "run_prompt",
                "value": "comparative_minimal_retention",
                "reason": "Some prompts are too strict; test whether fewer rules preserve true classes better.",
            }
        )
    recs.append(
        {
            "priority": len(recs) + 1,
            "action": "input_ablation",
            "value": "comparative_context_minimal_retention",
            "reason": "If isolated crop remains ambiguous, add local H&E context with minimal extra rules.",
        }
    )
    recs.append(
        {
            "priority": len(recs) + 1,
            "action": "model_ablation",
            "value": "Qwen3-VL-32B, Qwen3-VL-8B, InternVL3.5-8B, MedGemma-4B",
            "reason": "Check whether the bottleneck is prompt-specific or model-specific.",
        }
    )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-root", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_sweep_collected"))
    parser.add_argument("--fallback-collector", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_existing_outputs_collected"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_adaptive_iteration_decision"))
    parser.add_argument("--weak58-baseline-mean", type=float, default=3.879)
    args = parser.parse_args()

    root = args.collect_root if args.collect_root.exists() and find_collector_dirs(args.collect_root) else args.fallback_collector
    overall, by_class, false_rows, misses = load_all(root)
    if not overall:
        raise SystemExit(f"No collector outputs found under {root}")

    unique = unique_runs(overall)
    for row in unique:
        row["adaptive_score"] = round(score_run(row), 3)
        row["decision"] = classify_run(row, args.weak58_baseline_mean)
    ranked = sorted(unique, key=lambda row: (-as_float(row.get("adaptive_score", -999)), -as_int(row.get("true_retained", 0)), as_float(row.get("mean_classes_left", ""))))
    focus = ranked[:5]
    selected_runs = {str(row.get("run", "")) for row in focus}
    failure_counts = summarize_failures(false_rows, misses, selected_runs)
    recs = recommend(failure_counts, focus)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "adaptive_ranked_runs.csv", ranked)
    write_csv(args.output_dir / "adaptive_failure_family_counts.csv", [{"failure_family": k, "count": v} for k, v in failure_counts.most_common()])
    write_csv(args.output_dir / "adaptive_next_actions.csv", recs)

    lines = [
        "# Jun10 Step2 adaptive iteration decision",
        "",
        "## Experiment Design",
        "",
        "**Goal:** decide the next prompt/model/input iteration after Step2 sweep results are collected.",
        "",
        "**Step2 definition:** Step2 is a retention filter. A good run keeps the true tissue class and reduces unsupported alternatives; it does not need to force one final label.",
        "",
        f"**Evidence source:** `{root}`.",
        "",
        "## Best available runs",
        "",
        "| rank | run | n | true retained | mean classes left | decision |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {idx} | {row.get('run','')} | {row.get('n','')} | {row.get('true_retained','')} | {row.get('mean_classes_left','')} | {row.get('decision','')} |"
        )
    lines.extend(
        [
            "",
            "## Dominant failure families in focus runs",
            "",
            "| failure family | count |",
            "|---|---:|",
        ]
    )
    for key, value in failure_counts.most_common():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Next actions",
            "",
            "| priority | action | value | reason |",
            "|---:|---|---|---|",
        ]
    )
    for row in recs:
        lines.append(f"| {row['priority']} | {row['action']} | {row['value']} | {row['reason']} |")
    lines.append("")
    (args.output_dir / "adaptive_iteration_decision_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
