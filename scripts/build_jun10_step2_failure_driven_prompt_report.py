#!/usr/bin/env python3
"""Build a failure-driven prompt-iteration report for Jun10 Step2.

Step2 is a retention gate: keep the true tissue class while narrowing false
hypotheses. This script reads already-collected Step2 outputs and turns the
measured miss/false-keep patterns into concrete next prompt priorities.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
STRUCTURAL_CLASSES = {"bronchiola", "alveoli", "vessels"}
BROAD_CLASSES = {"tumor", "stroma", "immune_infiltration"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
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


def as_float(value: str, default: float = 999.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def rank_runs(overall_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    ranked = []
    for row in overall_rows:
        n = as_int(row.get("n", "0"))
        retained = as_int(row.get("true_retained", "0"))
        mean_left = as_float(row.get("mean_classes_left", ""))
        if n == 0:
            continue
        retained_frac = retained / n
        # Retention is the primary gate. Mean classes left matters only after
        # the run keeps most true labels.
        score = retained_frac * 100.0 - max(mean_left - 1.0, 0.0) * 2.0
        ranked.append(
            {
                **row,
                "retained_frac": round(retained_frac, 4),
                "priority_score": round(score, 3),
            }
        )
    return sorted(ranked, key=lambda row: (-as_float(str(row["priority_score"])), -as_int(str(row["true_retained"])), as_float(str(row["mean_classes_left"]))))


def unique_ranked_runs(ranked: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep one row per display run label to avoid duplicate historical reruns."""
    seen = set()
    out = []
    for row in ranked:
        label = str(row.get("run", ""))
        if label in seen:
            continue
        seen.add(label)
        out.append(row)
    return out


def summarize_false_keeps(false_rows: list[dict[str, str]], run: str) -> list[dict[str, object]]:
    pair_counts: dict[tuple[str, str], int] = {}
    for row in false_rows:
        if row.get("run") != run:
            continue
        true_class = row.get("true_class", "")
        false_class = row.get("false_class_kept", "")
        count = as_int(row.get("count", "0"))
        if true_class == "ALL" or count <= 0:
            continue
        key = (true_class, false_class)
        # Several historical runs share the same display label. Use max count
        # for the pair instead of summing duplicates, so a rerun does not inflate
        # the apparent failure mode.
        pair_counts[key] = max(pair_counts.get(key, 0), count)
    pairs = []
    for (true_class, false_class), count in pair_counts.items():
        pairs.append(
            {
                "run": run,
                "true_class": true_class,
                "false_class_kept": false_class,
                "count": count,
                "failure_family": failure_family(true_class, false_class),
            }
        )
    return sorted(pairs, key=lambda row: -as_int(str(row["count"])))


def failure_family(true_class: str, false_class: str) -> str:
    if true_class in STRUCTURAL_CLASSES and false_class in {"stroma", "immune_infiltration"}:
        return "structural_piece_overcalled_as_stroma_or_immune"
    if true_class in STRUCTURAL_CLASSES and false_class == "tumor":
        return "structural_piece_overcalled_as_tumor"
    if true_class == "stroma" and false_class == "immune_infiltration":
        return "stroma_immune_confusion"
    if true_class == "immune_infiltration" and false_class == "stroma":
        return "immune_stroma_confusion"
    return "other_false_keep"


def summarize_misses(miss_rows: list[dict[str, str]], run: str) -> list[dict[str, object]]:
    out = []
    seen = set()
    for row in miss_rows:
        if row.get("run") != run:
            continue
        key = (row.get("candidate_uid", ""), row.get("true_class", ""), row.get("he_retained_classes", ""), row.get("top_he_support_class", ""))
        if key in seen:
            continue
        seen.add(key)
        true_class = row.get("true_class", "")
        retained = [x for x in row.get("he_retained_classes", "").split(";") if x]
        top = row.get("top_he_support_class", "")
        family = "other_miss"
        if true_class in STRUCTURAL_CLASSES and (top in {"stroma", "immune_infiltration"} or any(x in {"stroma", "immune_infiltration"} for x in retained)):
            family = "structural_true_deleted_by_broad_label"
        elif true_class == "alveoli":
            family = "alveoli_deleted"
        elif true_class in STRUCTURAL_CLASSES:
            family = "structural_true_deleted"
        out.append(
            {
                "run": run,
                "candidate_uid": row.get("candidate_uid", ""),
                "true_class": true_class,
                "he_retained_classes": row.get("he_retained_classes", ""),
                "true_class_he_support_score": row.get("true_class_he_support_score", ""),
                "top_he_support_class": top,
                "top_he_support_score": row.get("top_he_support_score", ""),
                "failure_family": family,
            }
        )
    return out


def prompt_priorities(false_pairs: list[dict[str, object]], misses: list[dict[str, object]]) -> list[dict[str, object]]:
    family_counts = Counter()
    for row in false_pairs:
        family_counts[str(row["failure_family"])] += as_int(str(row["count"]))
    for row in misses:
        family_counts[str(row["failure_family"])] += 1

    priorities = []
    if family_counts["structural_piece_overcalled_as_stroma_or_immune"] or family_counts["structural_true_deleted_by_broad_label"]:
        priorities.append(
            {
                "priority": 1,
                "prompt_style": "comparative_alveoli_septa_rescue",
                "why": "Measured failures show structural pieces, especially alveoli/bronchiola/vessels, being replaced by stroma or immune.",
            }
        )
        priorities.append(
            {
                "priority": 2,
                "prompt_style": "comparative_stroma_immune_strict_veto",
                "why": "Measured false-kept labels are dominated by stroma and immune_infiltration, so broad labels need stricter candidate-mask evidence.",
            }
        )
    priorities.append(
        {
            "priority": 3,
            "prompt_style": "comparative_minimal_retention",
            "why": "If targeted rules still delete true labels, test whether a shorter prompt reduces over-control.",
        }
    )
    priorities.append(
        {
            "priority": 4,
            "prompt_style": "comparative_context_minimal_retention",
            "why": "If isolated crops remain ambiguous, test local H&E context with minimal extra rules.",
        }
    )
    return priorities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-dir", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_existing_outputs_collected"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_failure_driven_prompt_iteration"))
    parser.add_argument("--focus-top-runs", type=int, default=3)
    args = parser.parse_args()

    overall = read_csv(args.collector_dir / "step2_prompt_ablation_overall.csv")
    false_rows = read_csv(args.collector_dir / "step2_prompt_ablation_false_kept_matrix.csv")
    miss_rows = read_csv(args.collector_dir / "step2_prompt_ablation_misses.csv")

    ranked = rank_runs(overall)
    unique_ranked = unique_ranked_runs(ranked)
    top_runs = unique_ranked[: args.focus_top_runs]

    all_false_pairs: list[dict[str, object]] = []
    all_misses: list[dict[str, object]] = []
    for run in top_runs:
        label = str(run["run"])
        all_false_pairs.extend(summarize_false_keeps(false_rows, label)[:20])
        all_misses.extend(summarize_misses(miss_rows, label))

    priorities = prompt_priorities(all_false_pairs, all_misses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ranked_existing_step2_runs.csv", ranked)
    write_csv(args.output_dir / "ranked_existing_step2_runs_unique_labels.csv", unique_ranked)
    write_csv(args.output_dir / "top_run_false_keep_patterns.csv", all_false_pairs)
    write_csv(args.output_dir / "top_run_missed_true_classes.csv", all_misses)
    write_csv(args.output_dir / "next_prompt_priorities.csv", priorities)

    family_counts = Counter()
    for row in all_false_pairs:
        family_counts[str(row["failure_family"])] += as_int(str(row["count"]))
    for row in all_misses:
        family_counts[str(row["failure_family"])] += 1

    lines = [
        "# Jun10 Step2 failure-driven prompt iteration",
        "",
        "## Experiment Design",
        "",
        "**Goal:** use already-finished Step2 H&E verification results to decide the next prompt changes for weak-class filtering.",
        "",
        "**Step2 definition:** Step2 is a retention gate. The model should keep the true tissue class while removing unsupported alternatives. It is not required to choose one final class.",
        "",
        "**Ground truth use:** true class and annotation-derived labels are used only after model scoring to analyze misses and false-kept labels; they are not part of the prompt.",
        "",
        "## Best existing Step2 runs",
        "",
        "| rank | run | n | true retained | mean classes left | priority score |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(unique_ranked[:10], start=1):
        lines.append(
            f"| {idx} | {row['run']} | {row['n']} | {row['true_retained']} | {row['mean_classes_left']} | {row['priority_score']} |"
        )

    lines.extend(
        [
            "",
            "## Main measured failure families in the top runs",
            "",
            "| failure family | count |",
            "|---|---:|",
        ]
    )
    for family, count in family_counts.most_common():
        lines.append(f"| {family} | {count} |")

    lines.extend(
        [
            "",
            "## Next prompt priorities",
            "",
            "| priority | prompt style | why |",
            "|---:|---|---|",
        ]
    )
    for row in priorities:
        lines.append(f"| {row['priority']} | {row['prompt_style']} | {row['why']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The existing best runs keep many true labels, but they still leave too many broad false hypotheses. The recurring failure is not just general uncertainty: structural pieces are often over-kept or replaced by stroma and immune_infiltration. Therefore the next prompt sweep should prioritize targeted structural rescue and strict broad-label veto prompts before adding more image context.",
            "",
        ]
    )
    (args.output_dir / "failure_driven_prompt_iteration_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
