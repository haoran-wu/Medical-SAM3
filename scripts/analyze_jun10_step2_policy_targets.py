#!/usr/bin/env python3
"""Analyze post-filter policy targets for Jun10 Step2 H&E verifier scores.

This script does not claim a new model result. It uses existing Step2
per-hypothesis H&E scores to test what kind of narrowing rule a future prompt
should try to elicit from the VLM.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


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


def score_value(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("he_support_score", 0) or 0))
    except ValueError:
        return 0


def group_scores(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.get("parse_status", "ok") != "ok":
            continue
        uid = row["candidate_uid"]
        item = grouped.setdefault(
            uid,
            {
                "candidate_uid": uid,
                "row_index": row.get("row_index", ""),
                "true_class": row.get("true_class", ""),
                "scores": {},
                "raw": row,
            },
        )
        target = row.get("target_hypothesis_class", "")
        if target in CLASS_KEYS:
            item["scores"][target] = max(int(item["scores"].get(target, 0)), score_value(row))  # type: ignore[index]
    return grouped


def ranked(scores: dict[str, int], penalty: dict[str, int] | None = None) -> list[tuple[str, int, int]]:
    penalty = penalty or {}
    return sorted(
        [(cls, score, score - penalty.get(cls, 0)) for cls, score in scores.items()],
        key=lambda item: (item[2], item[1], item[0]),
        reverse=True,
    )


def keep_score_ge(threshold: int) -> Callable[[dict[str, int]], set[str]]:
    return lambda scores: {cls for cls, score in scores.items() if score >= threshold}


def keep_top_k(k: int, min_score: int = 0, penalty: dict[str, int] | None = None) -> Callable[[dict[str, int]], set[str]]:
    def policy(scores: dict[str, int]) -> set[str]:
        items = [item for item in ranked(scores, penalty) if item[1] >= min_score]
        return {cls for cls, _score, _adj in items[:k]}

    return policy


def keep_top_margin(max_k: int, margin: int, min_score: int, penalty: dict[str, int] | None = None) -> Callable[[dict[str, int]], set[str]]:
    def policy(scores: dict[str, int]) -> set[str]:
        items = [item for item in ranked(scores, penalty) if item[1] >= min_score]
        if not items:
            return set()
        top_adj = items[0][2]
        kept = {cls for cls, _score, adj in items if top_adj - adj <= margin}
        ordered = [cls for cls, _score, _adj in items if cls in kept]
        return set(ordered[:max_k])

    return policy


def structural_priority_policy(scores: dict[str, int]) -> set[str]:
    struct_items = [(cls, scores[cls]) for cls in STRUCTURAL if cls in scores]
    broad_items = [(cls, scores[cls]) for cls in BROAD if cls in scores]
    best_struct = max([score for _cls, score in struct_items], default=0)
    best_broad = max([score for _cls, score in broad_items], default=0)
    kept: set[str] = set()
    if best_struct >= 55:
        kept.update(cls for cls, score in struct_items if score >= max(50, best_struct - 15))
        kept.update(cls for cls, score in broad_items if score >= 75 and score >= best_struct + 10)
    elif best_broad >= 55:
        kept.update(cls for cls, score in broad_items if score >= max(55, best_broad - 10))
        kept.update(cls for cls, score in struct_items if score >= 70)
    else:
        kept.update(cls for cls, score in scores.items() if score >= 50)
    return set(list(sorted(kept, key=lambda cls: scores.get(cls, 0), reverse=True))[:3])


def broad_veto_policy(scores: dict[str, int]) -> set[str]:
    top = max(scores.values(), default=0)
    kept = set()
    for cls, score in scores.items():
        if cls in STRUCTURAL and score >= 50:
            kept.add(cls)
        elif cls in BROAD and score >= 70 and top - score <= 10:
            kept.add(cls)
    if not kept and scores:
        kept.add(max(scores, key=scores.get))
    return set(list(sorted(kept, key=lambda cls: scores.get(cls, 0), reverse=True))[:3])


def eval_policy(name: str, grouped: dict[str, dict[str, object]], fn: Callable[[dict[str, int]], set[str]]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    row_outputs: list[dict[str, object]] = []
    counts = []
    true_retained = 0
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    false_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for uid, item in grouped.items():
        scores = item["scores"]  # type: ignore[assignment]
        assert isinstance(scores, dict)
        kept = fn(scores)
        true_class = str(item["true_class"])
        retained = true_class in kept
        true_retained += int(retained)
        counts.append(len(kept))
        row = {
            "policy": name,
            "candidate_uid": uid,
            "true_class": true_class,
            "kept_classes": ";".join(cls for cls in CLASS_KEYS if cls in kept),
            "n_classes_left": len(kept),
            "true_retained": str(retained).lower(),
            **{f"score_{cls}": scores.get(cls, "") for cls in CLASS_KEYS},
        }
        row_outputs.append(row)
        by_class[true_class].append(row)
        for cls in kept:
            if cls != true_class:
                false_counter[true_class][cls] += 1

    n = len(row_outputs)
    overall = {
        "policy": name,
        "n": n,
        "true_retained": true_retained,
        "true_retained_frac": round(true_retained / n, 4) if n else "",
        "mean_classes_left": round(statistics.mean(counts), 3) if counts else "",
        "median_classes_left": round(statistics.median(counts), 3) if counts else "",
        "zero_class_rows": sum(count == 0 for count in counts),
    }
    by_class_rows = []
    for cls in CLASS_KEYS:
        rows = by_class.get(cls, [])
        if not rows:
            continue
        retained = sum(row["true_retained"] == "true" for row in rows)
        by_class_rows.append(
            {
                "policy": name,
                "true_class": cls,
                "n": len(rows),
                "true_retained": retained,
                "true_retained_frac": round(retained / len(rows), 4),
                "mean_classes_left": round(statistics.mean(int(row["n_classes_left"]) for row in rows), 3),
            }
        )
    false_rows = []
    for true_class in CLASS_KEYS:
        for false_class in CLASS_KEYS:
            if false_class == true_class:
                continue
            false_rows.append(
                {
                    "policy": name,
                    "true_class": true_class,
                    "false_class_kept": false_class,
                    "count": false_counter[true_class][false_class],
                }
            )
    return overall, by_class_rows, false_rows, row_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/he_prompt_selected_full167/step2_all_hypothesis_graded_stroma_lenient_qwen3vl32b_14418352/step2_hypothesis_verification_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_policy_target_analysis"))
    args = parser.parse_args()

    rows = read_csv(args.scores_csv)
    grouped = group_scores(rows)
    policies: list[tuple[str, Callable[[dict[str, int]], set[str]]]] = [
        ("all_step1_hypotheses", lambda scores: set(scores)),
        ("score_ge_50", keep_score_ge(50)),
        ("score_ge_60", keep_score_ge(60)),
        ("score_ge_65", keep_score_ge(65)),
        ("top1", keep_top_k(1)),
        ("top2_ge45", keep_top_k(2, 45)),
        ("top3_ge45", keep_top_k(3, 45)),
        ("top3_broadPenalty15_ge45", keep_top_k(3, 45, {cls: 15 for cls in BROAD})),
        ("top3_broadPenalty25_ge45", keep_top_k(3, 45, {cls: 25 for cls in BROAD})),
        ("top_margin15_ge45_max3", keep_top_margin(3, 15, 45)),
        ("top_margin20_broadPenalty15_ge45_max3", keep_top_margin(3, 20, 45, {cls: 15 for cls in BROAD})),
        ("structural_priority", structural_priority_policy),
        ("broad_veto", broad_veto_policy),
    ]

    overall_rows = []
    by_class_rows = []
    false_rows = []
    all_policy_rows = []
    for name, fn in policies:
        overall, by_class, false_kept, policy_rows = eval_policy(name, grouped, fn)
        overall_rows.append(overall)
        by_class_rows.extend(by_class)
        false_rows.extend(false_kept)
        all_policy_rows.extend(policy_rows)

    overall_rows = sorted(overall_rows, key=lambda row: (-int(row["true_retained"]), float(row["mean_classes_left"])))
    write_csv(args.output_dir / "policy_target_overall.csv", overall_rows)
    write_csv(args.output_dir / "policy_target_by_class.csv", by_class_rows)
    write_csv(args.output_dir / "policy_target_false_kept_matrix.csv", false_rows)
    write_csv(args.output_dir / "policy_target_rows.csv", all_policy_rows)

    report = [
        "# Jun10 Step2 policy target analysis",
        "",
        "This is not a new model run. It reuses existing H&E support scores to test what kind of prompt behavior would be useful.",
        "",
        "Best policies by retention then narrowness:",
    ]
    for row in overall_rows[:8]:
        report.append(
            f"- {row['policy']}: true_retained {row['true_retained']}/{row['n']}, mean_classes_left {row['mean_classes_left']}, zero_class_rows {row['zero_class_rows']}"
        )
    write_csv(args.output_dir / "policy_target_selected_candidates.csv", all_policy_rows)
    (args.output_dir / "policy_target_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.output_dir)
    for row in overall_rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
