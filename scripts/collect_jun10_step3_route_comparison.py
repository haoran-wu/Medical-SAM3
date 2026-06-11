#!/usr/bin/env python3
"""Compare Jun10 Step3 FICTURE filtering routes.

The three-step flow is:
1. FICTURE-derived cell-type text creates hypotheses.
2. H&E morphology keeps visually supported hypotheses.
3. FICTURE image consistency may optionally narrow the retained hypotheses.

This collector asks whether Step3 actually helps:
- Does it retain the true class?
- Does it reduce the number of remaining classes?
- Does it hurt structure classes such as bronchiola, alveoli, vessels, or immune?
"""

from __future__ import annotations

import argparse
import csv
import statistics
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


def split_classes(value: str) -> set[str]:
    return {item for item in str(value or "").split(";") if item}


def find_ficture_run_dirs(input_roots: list[Path]) -> list[Path]:
    dirs = []
    for root in input_roots:
        if root.is_file() and root.name == "ficture_candidate_retention_summary.csv":
            dirs.append(root.parent)
        elif (root / "ficture_candidate_retention_summary.csv").exists():
            dirs.append(root)
        elif root.exists():
            dirs.extend(path.parent for path in root.rglob("ficture_candidate_retention_summary.csv"))
    return sorted(set(dirs))


def load_he_policy(path: Path, policy: str) -> dict[str, set[str]]:
    rows = read_csv(path)
    out: dict[str, set[str]] = {}
    for row in rows:
        if row.get("policy") == policy:
            out[row["candidate_uid"]] = split_classes(row.get("kept_classes", ""))
    if not out:
        raise ValueError(f"No rows found for H&E policy {policy} in {path}")
    return out


def load_ficture_scores(run_dir: Path) -> dict[str, dict[str, int]]:
    path = run_dir / "ficture_hypothesis_verification_scores.csv"
    scores: dict[str, dict[str, int]] = {}
    if not path.exists():
        return scores
    for row in read_csv(path):
        uid = row["candidate_uid"]
        cls = row["target_hypothesis_class"]
        try:
            score = int(float(row.get("ficture_support_score") or 0))
        except Exception:
            score = 0
        scores.setdefault(uid, {})[cls] = score
    return scores


def summarize_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    overall = []
    by_class = []
    route_keys = sorted({(str(row["run"]), str(row["policy"])) for row in rows})
    for run, policy in route_keys:
        policy_rows = [row for row in rows if row["run"] == run and row["policy"] == policy]
        counts = [int(row["n_classes_left"]) for row in policy_rows]
        overall.append(
            {
                "run": run,
                "policy": policy,
                "true_retained": sum(row["true_retained"] == "true" for row in policy_rows),
                "n": len(policy_rows),
                "mean_classes_left": round(statistics.mean(counts), 3) if counts else "",
            }
        )
        for cls in CLASS_KEYS:
            cls_rows = [row for row in policy_rows if row["true_class"] == cls]
            cls_counts = [int(row["n_classes_left"]) for row in cls_rows]
            by_class.append(
                {
                    "run": run,
                    "policy": policy,
                    "true_class": cls,
                    "true_retained": sum(row["true_retained"] == "true" for row in cls_rows),
                    "n": len(cls_rows),
                    "mean_classes_left": round(statistics.mean(cls_counts), 3) if cls_counts else "",
                }
            )
    return (
        sorted(overall, key=lambda row: (-int(row["true_retained"]), float(row["mean_classes_left"] or 999))),
        by_class,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ficture-root", type=Path, action="append", required=True)
    parser.add_argument("--he-policy-csv", type=Path, required=True)
    parser.add_argument("--he-policy", default="best_class_specific_threshold")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tumor-stroma-ficture-threshold", type=int, default=50)
    args = parser.parse_args()

    he_map = load_he_policy(args.he_policy_csv, args.he_policy)
    all_rows: list[dict[str, object]] = []
    ficture_dirs = find_ficture_run_dirs(args.ficture_root)
    if not ficture_dirs:
        raise SystemExit("No ficture_candidate_retention_summary.csv files found")

    for run_dir in ficture_dirs:
        candidate_rows = read_csv(run_dir / "ficture_candidate_retention_summary.csv")
        ficture_scores = load_ficture_scores(run_dir)
        run_name = run_dir.name
        for row in candidate_rows:
            uid = row["candidate_uid"]
            true_class = row["true_class"]
            step1 = split_classes(row.get("step1_retained_classes", ""))
            he = he_map.get(uid, set())
            ficture = split_classes(row.get("ficture_retained_classes", ""))
            score_map = ficture_scores.get(uid, {})

            policies: dict[str, set[str]] = {
                "step1_text_only": step1,
                "step2_HE_policy": he,
                "step3_FICTURE_only": ficture,
                "HE_AND_FICTURE": he & ficture,
                "HE_OR_FICTURE": he | ficture,
            }

            tightened = set()
            for cls in he:
                if cls in {"tumor", "stroma"}:
                    if score_map.get(cls, -1) >= args.tumor_stroma_ficture_threshold:
                        tightened.add(cls)
                else:
                    tightened.add(cls)
            policies[f"HE_then_FICTURE_tighten_tumor_stroma_ge{args.tumor_stroma_ficture_threshold}"] = tightened

            for policy, kept in policies.items():
                all_rows.append(
                    {
                        "run": run_name,
                        "policy": policy,
                        "candidate_uid": uid,
                        "true_class": true_class,
                        "kept_classes": ";".join(sorted(kept)),
                        "n_classes_left": len(kept),
                        "true_retained": str(true_class in kept).lower(),
                    }
                )

    overall, by_class = summarize_rows(all_rows)
    write_csv(args.output_dir / "step3_route_comparison_rows.csv", all_rows)
    write_csv(args.output_dir / "step3_route_comparison_overall.csv", overall)
    write_csv(args.output_dir / "step3_route_comparison_by_class.csv", by_class)

    lines = [
        "# Jun10 Step3 route comparison",
        "",
        "Step3 uses the FICTURE image as an optional consistency check after the cell-type-text Step1 and H&E Step2.",
        "",
        "| run | policy | true retained | n | mean classes left |",
        "|---|---|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['run']} | {row['policy']} | {row['true_retained']} | {row['n']} | {row['mean_classes_left']} |"
        )
    lines.append("")
    lines.append("Interpretation: a useful Step3 policy should keep almost all true classes while reducing the mean number of remaining tissue hypotheses.")
    (args.output_dir / "step3_route_comparison_report.md").write_text("\n".join(lines))
    print(args.output_dir)


if __name__ == "__main__":
    main()
