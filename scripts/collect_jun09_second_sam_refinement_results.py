#!/usr/bin/env python3
"""Collect Jun09 second-SAM refinement results into the local report."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-or-local-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=BASE / "second_sam_refinement")
    args = parser.parse_args()

    src = args.remote_or_local_result_dir
    if not src.exists():
        raise SystemExit(f"Missing second-SAM result directory: {src}")
    required = ["second_sam_summary.csv", "second_sam_piece_results.csv", "run_config.json"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise SystemExit(f"Result directory is incomplete: missing {missing}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(src, args.output_dir)

    rows = read_csv(args.output_dir / "second_sam_summary.csv")
    if not rows:
        raise SystemExit("second_sam_summary.csv is empty")
    for row in rows:
        fig = args.output_dir / row.get("figure_rel", "")
        if row.get("figure_rel") and not fig.exists():
            raise SystemExit(f"Missing figure for {row.get('class')}: {fig}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/add_jun09_second_sam_refinement_section.py")],
        check=True,
        cwd=str(ROOT),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
