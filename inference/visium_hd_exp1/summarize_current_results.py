#!/usr/bin/env python3
"""Summarize VisiumHD Exp1 training/retrieval results into stable snapshot files."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def training_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        RESULTS / "training_runs" / "loss_ablation",
        RESULTS / "training_runs" / "training_strategy",
        RESULTS / "training_runs" / "musk_smoke",
    ]
    for root in roots:
        for path in sorted(root.glob("*/summary.json")):
            data = read_json(path)
            args = data.get("args", {})
            rows.append(
                {
                    "run": path.parent.name,
                    "group": root.name,
                    "best_epoch": data.get("best_epoch"),
                    "best_val_acc_expr": data.get("best_val_acc_expr"),
                    "best_val_macro_f1_expr": data.get("best_val_macro_f1_expr"),
                    "best_val_acc_img": data.get("best_val_acc_img"),
                    "image_backbone": args.get("image_backbone"),
                    "embed_dim": args.get("embed_dim"),
                    "include_labels": args.get("include_labels") or "8class",
                    "align_loss": args.get("align_loss"),
                    "align_weight": args.get("align_weight"),
                    "img_ce_weight": args.get("img_ce_weight"),
                    "expr_ce_weight": args.get("expr_ce_weight"),
                    "lr_schedule": args.get("lr_schedule"),
                    "warmup_ratio": args.get("warmup_ratio"),
                    "trainable_backbone_blocks": args.get("trainable_backbone_blocks"),
                    "freeze_epochs": args.get("freeze_epochs"),
                    "output_dir": str(path.parent),
                }
            )
    return rows


def retrieval_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        RESULTS / "retrieval_eval" / "loss_ablation",
        RESULTS / "retrieval_eval" / "training_strategy",
    ]
    for root in roots:
        for path in sorted(root.glob("*/metrics.json")):
            data = read_json(path)
            overall = data.get("retrieval_overall", {})
            linear = data.get("linear_probe", {})
            fusion = data.get("fusion_alpha_sweep") or {}
            rows.append(
                {
                    "run": path.parent.name,
                    "group": root.name,
                    "recall@1": overall.get("recall@1"),
                    "recall@5": overall.get("recall@5"),
                    "recall@10": overall.get("recall@10"),
                    "linear_image_only": linear.get("image_only"),
                    "linear_expression_only": linear.get("expression_only"),
                    "linear_concatenated": linear.get("concatenated"),
                    "fusion_best_alpha_by_recall@1": fusion.get("best_alpha_by_recall@1"),
                    "fusion_best_recall@1": fusion.get("best_recall@1"),
                    "n_train": data.get("n_train"),
                    "n_test": data.get("n_test"),
                    "checkpoint": data.get("checkpoint"),
                    "output_dir": str(path.parent),
                }
            )
    return rows


def main() -> None:
    out = RESULTS / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    train = training_rows()
    retrieval = retrieval_rows()
    payload = {
        "created_utc": stamp,
        "training": train,
        "retrieval": retrieval,
    }
    (out / "latest_results_snapshot.json").write_text(json.dumps(payload, indent=2))
    (out / f"results_snapshot_{stamp}.json").write_text(json.dumps(payload, indent=2))
    write_csv(out / "latest_training_summary.csv", train)
    write_csv(out / "latest_retrieval_summary.csv", retrieval)
    write_csv(out / f"training_summary_{stamp}.csv", train)
    write_csv(out / f"retrieval_summary_{stamp}.csv", retrieval)
    print(f"Wrote {len(train)} training rows and {len(retrieval)} retrieval rows to {out}")


if __name__ == "__main__":
    main()
