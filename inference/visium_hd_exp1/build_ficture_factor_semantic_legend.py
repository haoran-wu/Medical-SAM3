#!/usr/bin/env python3
"""Build the official FICTURE color/gene semantic legend.

This converts ``hex_12.k12.pixel.info.tsv`` into a stable JSON/CSV/Markdown
legend that downstream CLIP/VLM/ranking jobs can consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ficture_factor_semantics import build_semantic_legend, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info-tsv",
        type=Path,
        default=Path("pixel-level cell type image/visiumhd_exp1_hex12_k12/hex_12.k12.pixel.info.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/visium_hd_exp1/ficture_official_filtered_he_aligned"),
    )
    args = parser.parse_args()

    if not args.info_tsv.exists():
        raise SystemExit(f"Missing FICTURE factor info TSV: {args.info_tsv}")
    legend = build_semantic_legend(args.info_tsv)
    write_outputs(legend, args.output_dir)
    print(args.output_dir / "factor_semantic_legend.json")


if __name__ == "__main__":
    main()
