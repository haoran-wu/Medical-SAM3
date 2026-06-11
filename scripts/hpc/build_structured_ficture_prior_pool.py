#!/usr/bin/env python3
"""Append compact structured FICTURE prior text to a VLM request pool."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FACTOR_GROUPS = {
    "airway": ["factor_7_fraction"],
    "at2_alveolar": ["factor_3_fraction"],
    "endothelial": ["factor_9_fraction"],
    "stroma": ["factor_1_fraction"],
    "immune": [
        "factor_4_fraction",
        "factor_6_fraction",
        "factor_8_fraction",
        "factor_10_fraction",
        "factor_11_fraction",
    ],
    "epithelial_tumor_like": [
        "factor_0_fraction",
        "factor_2_fraction",
        "factor_5_fraction",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    return "__".join(
        [
            row.get("target_label") or row.get("label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            str(row.get("candidate_id", "")),
        ]
    )


def group_fraction(row: dict[str, str], keys: list[str]) -> float:
    total = 0.0
    for key in keys:
        total += float(row.get(key, 0.0) or 0.0)
    return total


def summary_text(feature_row: dict[str, str]) -> str:
    values = {name: group_fraction(feature_row, keys) for name, keys in FACTOR_GROUPS.items()}
    return (
        "FICTURE numeric summary for the candidate mask: "
        f"airway={values['airway']:.3f}; "
        f"AT2/alveolar={values['at2_alveolar']:.3f}; "
        f"endothelial={values['endothelial']:.3f}; "
        f"stroma={values['stroma']:.3f}; "
        f"immune={values['immune']:.3f}; "
        f"epithelial_or_tumor_like={values['epithelial_tumor_like']:.3f}. "
        "Use this only as weak supporting evidence. Do not call tumor from epithelial_or_tumor_like alone; H&E morphology is primary."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--composition-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    pool_rows = read_csv(args.pool_csv)
    comp_rows = read_csv(args.composition_csv)
    comp_by_key = {row["row_key"]: row for row in comp_rows}

    out_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for row in pool_rows:
        key = row_key(row)
        comp = comp_by_key.get(key)
        if comp is None:
            missing.append(key)
            row["ficture_summary_text"] = ""
        else:
            row["ficture_summary_text"] = summary_text(comp)
        out_rows.append(row)
    if missing:
        raise SystemExit(f"Missing composition rows for {len(missing)} candidates; first={missing[0]}")
    write_csv(args.out_csv, out_rows)
    print(args.out_csv)


if __name__ == "__main__":
    main()
