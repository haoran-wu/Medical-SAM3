#!/usr/bin/env python3
"""Summarize Jun10 Step3 order/policy comparisons.

This script does not run a model. It uses already generated Step2 H&E and Step3
FICTURE verification outputs to answer a specific question: if we try both
orders, Step1->H&E->FICTURE and Step1->FICTURE->H&E, which final policy is
actually safer?
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ROOT = Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill")
OUT = ROOT / "step3_two_route_final_answer"


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


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def rename_policy(policy: str) -> tuple[str, str]:
    names = {
        "route_A_step1_then_ficture_supported": (
            "Step1 -> FICTURE image check",
            "Use FICTURE image as the first visual filter after cell-type text.",
        ),
        "route_B_step1_then_HE_threshold": (
            "Step1 -> H&E morphology check",
            "Use H&E morphology scores as the first visual filter after cell-type text.",
        ),
        "route_C_HE_threshold_AND_ficture_supported": (
            "Step1 -> H&E AND FICTURE",
            "Keep only classes supported by both H&E and FICTURE image checks.",
        ),
        "route_D_HE_threshold_OR_ficture_supported": (
            "Step1 -> H&E OR FICTURE",
            "Keep classes supported by either visual check; this is a rescue rule.",
        ),
        "route_I_HE_then_FIC_tighten_tumor_stroma_only": (
            "Step1 -> H&E, then FICTURE only tightens tumor/stroma",
            "Keep the H&E decision for structural classes; use FICTURE only to slightly tighten tumor/stroma.",
        ),
    }
    return names.get(policy, (policy, ""))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    local_overall_path = ROOT / "step3_two_route_local_logic_comparison/two_route_policy_overall.csv"
    local_by_class_path = ROOT / "step3_two_route_local_logic_comparison/two_route_policy_by_class.csv"
    selected_rows_path = ROOT / "step3_selected_prompt_vs_ficture_full167/step3_selected_prompt_order_comparison_rows.csv"
    selected_overall_path = ROOT / "step3_selected_prompt_vs_ficture_full167/step3_selected_prompt_order_comparison_overall.csv"
    selected_by_class_path = ROOT / "step3_selected_prompt_vs_ficture_full167/step3_selected_prompt_order_comparison_by_class.csv"

    local_overall = read_csv(local_overall_path)
    local_by_class = read_csv(local_by_class_path)
    selected_rows = read_csv(selected_rows_path)
    selected_overall = read_csv(selected_overall_path)
    selected_by_class = read_csv(selected_by_class_path)

    keep_policies = {
        "route_A_step1_then_ficture_supported",
        "route_B_step1_then_HE_threshold",
        "route_C_HE_threshold_AND_ficture_supported",
        "route_D_HE_threshold_OR_ficture_supported",
        "route_I_HE_then_FIC_tighten_tumor_stroma_only",
    }

    final_overall = []
    for row in local_overall:
        policy = row["policy"]
        if policy not in keep_policies:
            continue
        label, note = rename_policy(policy)
        final_overall.append(
            {
                "route": label,
                "true_retained": row["true_retained"],
                "n": row["n"],
                "mean_classes_left": row["mean_classes_left"],
                "note": note,
            }
        )

    order_diff = sum(
        row["selected_he_then_ficture_classes"] != row["ficture_then_selected_he_classes"]
        for row in selected_rows
    )
    order_check = [
        {
            "check": "hard_intersection_order_difference",
            "different_candidates": order_diff,
            "n": len(selected_rows),
            "interpretation": "0 means H&E->FICTURE and FICTURE->H&E give the same hard-intersection result.",
        }
    ]

    selected_map = index_by(selected_overall, "method")
    raw_comparison = []
    for method, label in [
        ("step1_celltype_only", "Step1 cell-type text only"),
        ("selected_he_prompt", "Step1 -> H&E check, raw retained classes"),
        ("ficture_image_filter", "Step1 -> FICTURE image check"),
        ("selected_he_AND_ficture", "Hard AND of H&E and FICTURE"),
        ("selected_he_OR_ficture", "OR rescue of H&E and FICTURE"),
    ]:
        row = selected_map[method]
        raw_comparison.append(
            {
                "method": label,
                "true_retained": row["true_retained"],
                "n": row["n"],
                "mean_classes_left": row["mean_candidate_classes_left"],
            }
        )

    # Keep selected by-class rows with clearer method names.
    selected_method_names = {
        "selected_he_prompt": "Step1 -> H&E check, raw retained classes",
        "ficture_image_filter": "Step1 -> FICTURE image check",
        "selected_he_AND_ficture": "Hard AND of H&E and FICTURE",
        "selected_he_OR_ficture": "OR rescue of H&E and FICTURE",
    }
    raw_by_class = []
    for row in selected_by_class:
        if row["method"] not in selected_method_names:
            continue
        raw_by_class.append(
            {
                "method": selected_method_names[row["method"]],
                "true_class": row["true_class"],
                "true_retained": row["true_retained"],
                "n": row["n"],
                "mean_classes_left": row["mean_candidate_classes_left"],
            }
        )

    final_by_class = []
    for row in local_by_class:
        if row["policy"] not in keep_policies:
            continue
        label, _note = rename_policy(row["policy"])
        final_by_class.append(
            {
                "route": label,
                "true_class": row["true_class"],
                "true_retained": row["true_retained"],
                "n": row["n"],
                "mean_classes_left": row["mean_classes_left"],
            }
        )

    write_csv(OUT / "step3_final_policy_overall.csv", final_overall)
    write_csv(OUT / "step3_final_policy_by_class.csv", final_by_class)
    write_csv(OUT / "step3_raw_order_overall.csv", raw_comparison)
    write_csv(OUT / "step3_raw_order_by_class.csv", raw_by_class)
    write_csv(OUT / "step3_order_commutativity_check.csv", order_check)

    lines = [
        "# Jun10 Step3 two-route final answer",
        "",
        "## Experiment Design",
        "",
        "**Goal:** compare whether Step3 should use the FICTURE image before or after the H&E morphology check.",
        "",
        "**Inputs:** the same 167 candidate pieces and the same Step1 cell-type-text hypotheses. Step2 uses H&E morphology. Step3 uses the FICTURE image as a visual consistency check.",
        "",
        "**Metric:** `true_retained` counts candidates where the true tissue class is still present in the retained hypothesis list. `mean_classes_left` is the average number of remaining tissue hypotheses per candidate; lower is narrower, but only useful if true_retained stays high.",
        "",
        "## Key finding",
        "",
        "Hard order does not matter: H&E->FICTURE and FICTURE->H&E are the same intersection when both checks must agree.",
        f"The direct order-difference check found **{order_diff}/{len(selected_rows)}** candidates with different hard-intersection outputs.",
        "",
        "## Final policy comparison",
        "",
        "| route | true retained | n | mean classes left | note |",
        "|---|---:|---:|---:|---|",
    ]
    for row in final_overall:
        lines.append(
            f"| {row['route']} | {row['true_retained']} | {row['n']} | {row['mean_classes_left']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Hard FICTURE filtering is not safe: it loses alveoli and vessels.",
            "- H&E-only thresholding keeps all true classes but remains broad.",
            "- The safest Step3 use is weak: use H&E as the main verifier and let FICTURE only lightly tighten tumor/stroma. It keeps all 167 true classes and narrows only a little.",
            "- OR rescue keeps true classes but expands the hypothesis list, so it is not a useful narrowing step.",
        ]
    )
    (OUT / "step3_two_route_final_answer.md").write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
