#!/usr/bin/env python3
"""Archive final VisiumHD Exp1 results into stable Markdown and JSON files."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1")
SUM = ROOT / "summaries"


def rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(value: str | None, ndigits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{ndigits}f}"
    except ValueError:
        return str(value)


def compact_run(name: str) -> str:
    out = name
    replacements = {
        "gigapath_crop128_": "GigaPath ",
        "resnet50_crop128_": "ResNet50 ",
        "_11028639_": " #",
        "_11032366_": " #",
        "_11038859_": " #",
        "_retrieval_11037672_": " retrieval #",
        "_retrieval_11040135_": " retrieval #",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out.replace("_", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def main() -> None:
    SUM.mkdir(parents=True, exist_ok=True)
    train_csv = SUM / "latest_training_summary.csv"
    retr_csv = SUM / "latest_retrieval_summary.csv"
    snapshot = SUM / "latest_results_snapshot.json"

    train = rows_csv(train_csv)
    retr = rows_csv(retr_csv)
    train_loss = [
        r
        for r in train
        if r.get("group") == "loss_ablation"
        and ("gigapath_crop128" in r["run"] or "resnet50_crop128" in r["run"])
    ]
    train_strategy = [r for r in train if r.get("group") == "training_strategy"]
    retr_loss = [r for r in retr if r.get("group") == "loss_ablation"]
    retr_strategy = [r for r in retr if r.get("group") == "training_strategy"]

    payload = {
        "created_local_time": datetime.now().isoformat(timespec="seconds"),
        "project_results_root": str(ROOT),
        "stable_snapshot": str(snapshot),
        "latest_training_summary_csv": str(train_csv),
        "latest_retrieval_summary_csv": str(retr_csv),
        "counts_in_stable_snapshot": {
            "training_rows": len(train),
            "retrieval_rows": len(retr),
        },
        "hpc_status": {
            "squeue": "empty as checked 2026-05-07",
            "all_required_new_experiments_complete": True,
            "note": (
                "Original full-matrix retrieval tasks that OOMed were replaced "
                "by exact chunked top-k retry jobs 11039815, 11040086, and 11045911."
            ),
        },
        "training_rows": train,
        "retrieval_rows": retr,
    }

    archive_json = SUM / "final_group_meeting_results_20260507.json"
    archive_json.write_text(json.dumps(payload, indent=2))

    lines: list[str] = []
    lines.append("# Medical-SAM3 VisiumHD Exp1 Final Results Snapshot - 2026-05-07")
    lines.append("")
    lines.append("## Completion Status")
    lines.append("- Current `squeue -u hw646` is empty: no active running or pending Medical-SAM3 jobs remain for this batch.")
    lines.append("- All required overnight experiments are complete.")
    lines.append("- Original retrieval job `11037672` had OOM subtasks because the old evaluator materialized the full similarity matrix.")
    lines.append("- Those OOMed subtasks were completed with exact chunked top-k retry jobs: `11039815`, `11040086`, and `11045911`.")
    lines.append("- Stable snapshots were refreshed after completion.")
    lines.append("")
    lines.append("## Stable Files")
    for path in [snapshot, train_csv, retr_csv, archive_json]:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append(f"Stable summary counts: {len(train)} training rows, {len(retr)} retrieval rows.")
    lines.append("")

    lines.append("## Main Training Strategy Results")
    lines.append("| Run | Classes | Backbone | Blocks | Schedule | Expr acc | Expr macro-F1 | Image acc | Checkpoint/result dir |")
    lines.append("|---|---:|---|---:|---|---:|---:|---:|---|")
    for row in sorted(train_strategy, key=lambda x: x["run"]):
        lines.append(
            md_row(
                [
                    compact_run(row["run"]),
                    row.get("include_labels", ""),
                    row.get("image_backbone", ""),
                    row.get("trainable_backbone_blocks", ""),
                    row.get("lr_schedule", ""),
                    fnum(row.get("best_val_acc_expr")),
                    fnum(row.get("best_val_macro_f1_expr")),
                    fnum(row.get("best_val_acc_img")),
                    f"`{row.get('output_dir', '')}`",
                ]
            )
        )
    lines.append("")

    lines.append("## Loss Ablation Training Results")
    lines.append("| Run | Classes | Backbone | Align loss | Lambda | Expr acc | Expr macro-F1 | Image acc | Result dir |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---|")

    def train_sort(row: dict[str, str]) -> tuple[int, int, str]:
        run = row["run"]
        return (0 if "gigapath" in run else 1, 0 if "3class" in run else 1, run)

    for row in sorted(train_loss, key=train_sort):
        lines.append(
            md_row(
                [
                    compact_run(row["run"]),
                    row.get("include_labels", ""),
                    row.get("image_backbone", ""),
                    row.get("align_loss", ""),
                    fnum(row.get("align_weight"), 2),
                    fnum(row.get("best_val_acc_expr")),
                    fnum(row.get("best_val_macro_f1_expr")),
                    fnum(row.get("best_val_acc_img")),
                    f"`{row.get('output_dir', '')}`",
                ]
            )
        )
    lines.append("")

    lines.append("## Retrieval Results - Loss Ablation")
    lines.append("| Run | Group | R@1 | R@5 | R@10 | Best fused R@1 | Concat probe | Metrics dir |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    def retr_sort(row: dict[str, str]) -> tuple[int, int, str]:
        run = row["run"]
        return (0 if "gigapath" in run else 1, 0 if "3class" in run else 1, run)

    for row in sorted(retr_loss, key=retr_sort):
        lines.append(
            md_row(
                [
                    compact_run(row["run"]),
                    row.get("group", ""),
                    fnum(row.get("recall@1")),
                    fnum(row.get("recall@5")),
                    fnum(row.get("recall@10")),
                    fnum(row.get("fusion_best_recall@1")),
                    fnum(row.get("linear_concatenated")),
                    f"`{row.get('output_dir', '')}`",
                ]
            )
        )
    lines.append("")

    lines.append("## Retrieval Results - Training Strategy")
    lines.append("| Run | R@1 | R@5 | R@10 | Best fused R@1 | Concat probe | Metrics dir |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(retr_strategy, key=lambda x: x["run"]):
        lines.append(
            md_row(
                [
                    compact_run(row["run"]),
                    fnum(row.get("recall@1")),
                    fnum(row.get("recall@5")),
                    fnum(row.get("recall@10")),
                    fnum(row.get("fusion_best_recall@1")),
                    fnum(row.get("linear_concatenated")),
                    f"`{row.get('output_dir', '')}`",
                ]
            )
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("- `GigaPath crop128 3-class` is the strongest and most stable main line.")
    lines.append("- `CE-only` gives the best raw 3-class expression-to-image nearest-neighbor R@1 among the frozen loss-ablation models.")
    lines.append("- `CE + InfoNCE` improves broader top-k/fused retrieval in several settings, useful for candidate/proposal ranking rather than single nearest-neighbor lookup.")
    lines.append("- `InfoNCE-only` is not enough for this dataset because the expression tower is not a pretrained molecular anchor; `CE_expression` is still needed as semantic supervision.")
    lines.append("- Haiku-inspired last-block fine-tuning improves image branch and fused/probe readouts.")
    lines.append("- `last-2 warmup/cosine` is the best candidate for SAM3 proposal/mask ranking, while CE-only/last-1 remain strong raw retrieval baselines.")
    lines.append("- 8-class remains harder than 3-class, consistent with fine-label ambiguity, class imbalance, and rare labels.")
    lines.append("")

    lines.append("## Next Practical Checkpoint Choices")
    lines.append("- Main SAM3 ranking candidate: `training_runs/training_strategy/gigapath_crop128_3class_last2_warmcos_l005_11038859_3/best.pt`.")
    lines.append("- Raw nearest-neighbor baseline: `training_runs/loss_ablation/gigapath_crop128_3class_ce_only_11028639_0/best.pt`.")
    lines.append("- Last-1 retrieval baseline: `training_runs/training_strategy/gigapath_crop128_3class_last1_warmcos_l005_11038859_2/best.pt`.")
    lines.append("")

    archive_md = SUM / "final_group_meeting_results_20260507.md"
    archive_md.write_text("\n".join(lines) + "\n")
    print(archive_md)
    print(archive_json)
    print(f"training_rows={len(train)} retrieval_rows={len(retr)}")


if __name__ == "__main__":
    main()
