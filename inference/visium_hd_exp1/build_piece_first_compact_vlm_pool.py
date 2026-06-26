#!/usr/bin/env python3
"""Build a compact piece-first VLM source pool from selected candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DESCRIPTIONS = {
    "bronchiola": "bronchiolar airway tissue, airway-like lumen, epithelial lining.",
    "alveoli": "alveolar lung parenchyma, open air spaces, thin septa.",
    "vessels": "blood vessel or vascular wall, lumen-like vascular structure.",
    "tumor": "malignant epithelial tumor region.",
    "stroma": "stromal or mesenchymal tissue, collagen/fibroblast/smooth-muscle-like tissue.",
    "immune infiltration": "immune-cell-rich region, small round-cell aggregates.",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="medpt24_compact167_piece_funnel_20260603")
    args = parser.parse_args()

    selected = read_rows(args.selected_csv)
    if not selected:
        raise SystemExit("No selected rows")

    public_fields = [
        "label",
        "display",
        "target_label",
        "target_display",
        "source",
        "run",
        "setting",
        "candidate_id",
        "candidate_uid",
        "sample_bucket",
        "mask_path",
        "target_description",
        "target_factor_hints",
        "candidate_factor_composition",
        "he_crop_rel",
        "ficture_crop_rel",
    ]
    hidden_fields = public_fields + [
        "classification_true_label",
        "classification_true_display",
        "matched_annotation_component_id",
        "component_dice",
        "component_precision",
        "component_recall",
        "hidden_dice",
        "hidden_precision",
        "hidden_recall",
        "candidate_original_best_label",
        "candidate_original_best_dice",
        "component_sampling_rule",
        "union_policy",
        "pro_note",
        "best_component_label",
        "best_component_display",
        "best_component_dice",
        "piece_area",
        "piece_bbox",
        "parent_cluster_size",
        "parent_member_sources",
        "all_class_component_dice",
        "compact_funnel_score",
        "compact_funnel_important_score",
    ]

    public_rows: list[dict[str, str]] = []
    hidden_rows: list[dict[str, str]] = []
    for idx, row in enumerate(selected):
        display = row["display"]
        label = row["label"]
        base = {
            "label": label,
            "display": display,
            "target_label": label,
            "target_display": display,
            "source": "medpt24_compact_cluster_piece",
            "run": args.run_name,
            "setting": row["setting"],
            "candidate_id": str(idx),
            "candidate_uid": row["candidate_uid"],
            "sample_bucket": "COMPACT",
            "mask_path": row["mask_path"],
            "target_description": DESCRIPTIONS.get(display, display),
            "target_factor_hints": "",
            "candidate_factor_composition": "",
            "he_crop_rel": "",
            "ficture_crop_rel": "",
        }
        public_rows.append(base.copy())
        hidden = base.copy()
        hidden.update(
            {
                "classification_true_label": label,
                "classification_true_display": display,
                "matched_annotation_component_id": row.get(
                    "matched_annotation_component_id", ""
                ),
                "component_dice": row.get("component_dice", ""),
                "component_precision": row.get("component_precision", ""),
                "component_recall": row.get("component_recall", ""),
                "hidden_dice": row.get("full_dice", ""),
                "hidden_precision": row.get("full_precision", ""),
                "hidden_recall": row.get("full_recall", ""),
                "candidate_original_best_label": label,
                "candidate_original_best_dice": row.get("component_dice", ""),
                "component_sampling_rule": (
                    "compact balanced piece-first funnel over IoU-clustered "
                    "medical_official_points_step24 HE+FICTURE candidates; "
                    "annotation used only for retrospective coverage audit"
                ),
                "union_policy": "piece_first_no_union_input_compact167",
                "pro_note": "VLM input is one compact IoU-clustered candidate piece, not a final union mask.",
                "best_component_label": label,
                "best_component_display": display,
                "best_component_dice": row.get("component_dice", ""),
                "piece_area": row.get("candidate_area", ""),
                "piece_bbox": "",
                "parent_cluster_size": row.get("cluster_size", ""),
                "parent_member_sources": row.get("member_sources", ""),
                "all_class_component_dice": "",
                "compact_funnel_score": row.get("score", ""),
                "compact_funnel_important_score": row.get("important_score", ""),
            }
        )
        hidden_rows.append(hidden)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "public_vlm_requests.csv", public_rows, public_fields)
    write_rows(args.output_dir / "hidden_candidate_truth.csv", hidden_rows, hidden_fields)
    summary = args.output_dir / "compact_pool_summary.txt"
    counts: dict[str, int] = {}
    for row in public_rows:
        counts[row["display"]] = counts.get(row["display"], 0) + 1
    summary.write_text(
        "Compact piece-first VLM source pool\n"
        f"rows={len(public_rows)}\n"
        + "\n".join(f"{key}: {value}" for key, value in sorted(counts.items()))
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(public_rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
