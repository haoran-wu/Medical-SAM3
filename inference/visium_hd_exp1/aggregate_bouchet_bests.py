#!/usr/bin/env python3
"""Aggregate best_by_label.csv files across Bouchet VisiumHD experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FOCUS = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
}


def to_float(value: object) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/*/best_by_label.csv")):
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("label", "")
                    if not label:
                        continue
                    row = dict(row)
                    row["experiment_family"] = path.parent.parent.name
                    row["run"] = path.parent.name
                    row["path"] = str(path)
                    row["dice"] = f"{to_float(row.get('dice') or row.get('best Dice')):.6f}"
                    row["precision"] = f"{to_float(row.get('precision')):.6f}"
                    row["recall"] = f"{to_float(row.get('recall')):.6f}"
                    rows.append(row)
        except Exception:
            continue

    rows.sort(key=lambda r: (r["label"], -to_float(r["dice"])))
    fields = [
        "label",
        "display",
        "dice",
        "precision",
        "recall",
        "ranker",
        "top_k",
        "experiment_family",
        "run",
        "path",
        "selected",
    ]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    print("Current focus-class bests:")
    for label, display in FOCUS.items():
        label_rows = [r for r in rows if r["label"] == label]
        if not label_rows:
            print(f"- {display}: no result")
            continue
        best = label_rows[0]
        print(
            f"- {display}: Dice {to_float(best['dice']):.3f}, "
            f"P {to_float(best['precision']):.3f}, R {to_float(best['recall']):.3f}; "
            f"{best.get('ranker', '')} top-{best.get('top_k', '')}; {best['run']}"
        )
    print(f"Aggregated rows: {len(rows)}")


if __name__ == "__main__":
    main()
