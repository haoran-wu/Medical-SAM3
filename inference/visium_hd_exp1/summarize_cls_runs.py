#!/usr/bin/env python3
"""Summarize classification-alignment runs from their output directories."""

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def summarize_run(run_dir: Path) -> dict:
    summary = load_json(run_dir / "summary.json")
    metrics = load_json(run_dir / "best_val_metrics.json")
    expr_metrics = metrics.get("expr", {})
    img_metrics = metrics.get("img", {})
    return {
        "run": run_dir.name,
        "best_epoch": summary.get("best_epoch"),
        "monitor": summary.get("monitor"),
        "best_monitor": summary.get("best_monitor"),
        "best_val_acc_expr": summary.get("best_val_acc_expr"),
        "best_val_macro_f1_expr": summary.get("best_val_macro_f1_expr"),
        "best_val_acc_img": summary.get("best_val_acc_img"),
        "best_val_macro_f1_img": img_metrics.get("macro_f1"),
        "expr_per_class": expr_metrics.get("per_class", {}),
        "img_per_class": img_metrics.get("per_class", {}),
        "exists": bool(summary),
    }


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    fields = [
        "run",
        "exists",
        "best_epoch",
        "monitor",
        "best_monitor",
        "best_val_acc_expr",
        "best_val_macro_f1_expr",
        "best_val_acc_img",
        "best_val_macro_f1_img",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_per_class_csv(rows: list[dict], output_path: Path, side: str) -> None:
    fields = ["run", "label", "support", "precision", "recall", "f1"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        key = f"{side}_per_class"
        for row in rows:
            for label, metrics in row.get(key, {}).items():
                writer.writerow({
                    "run": row["run"],
                    "label": label,
                    "support": metrics.get("support"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize classification run outputs.")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(path) for path in args.run_dirs]

    (args.output_dir / "classification_run_summary.json").write_text(json.dumps(rows, indent=2))
    write_summary_csv(rows, args.output_dir / "classification_run_summary.csv")
    write_per_class_csv(rows, args.output_dir / "classification_expr_per_class.csv", "expr")
    write_per_class_csv(rows, args.output_dir / "classification_img_per_class.csv", "img")

    print(f"Wrote summary for {len(rows)} runs to {args.output_dir}")
    for row in rows:
        status = "ok" if row["exists"] else "missing"
        print(
            f"{row['run']}: {status}, "
            f"expr_acc={row.get('best_val_acc_expr')}, "
            f"expr_macro_f1={row.get('best_val_macro_f1_expr')}"
        )


if __name__ == "__main__":
    main()
