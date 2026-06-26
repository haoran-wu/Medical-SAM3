#!/usr/bin/env python3
"""Prepare small-core Xenium SAM prompt sweep task tables.

The first Xenium HD-style sweep reused VisiumHD-sized prompts.  The accepted
Xenium TMA crops are much smaller, so this companion bundle keeps the same
source images and hidden annotation masks but writes a denser, small-core prompt
manifest with smaller boxes and tighter point grids.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import xenium_prepare_hdstyle_sam_inputs as hd


DEFAULT_BASE_PREPARED = Path(
    "output/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_20260624"
)
DEFAULT_OUT = Path(
    "output/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_smallcore_20260625"
)


SMALLCORE_SETTINGS = [
    # Dense point prompts.  Images are about 700-900 px per side, so these
    # are intentionally much tighter than the old step24/32 defaults.
    (0, "base_smallpoint_g8_m1024_p4500", "base", "point", 1024, "--grid-step 8 --border 4 --max-points 4500"),
    (1, "base_smallpoint_g12_m1024_p3600", "base", "point", 1024, "--grid-step 12 --border 6 --max-points 3600"),
    (2, "base_smallpoint_g16_m1024_p2800", "base", "point", 1024, "--grid-step 16 --border 8 --max-points 2800"),
    (3, "base_smallpoint_g20_m1024_p2200", "base", "point", 1024, "--grid-step 20 --border 10 --max-points 2200"),
    (4, "base_smallpoint_g24_m1024_p1800", "base", "point", 1024, "--grid-step 24 --border 12 --max-points 1800"),
    (5, "base_smallpoint_g32_m1024_p1200", "base", "point", 1024, "--grid-step 32 --border 16 --max-points 1200"),
    (6, "medical_smallpoint_g8_m1024_p4500", "medical", "point", 1024, "--grid-step 8 --border 4 --max-points 4500"),
    (7, "medical_smallpoint_g12_m1024_p3600", "medical", "point", 1024, "--grid-step 12 --border 6 --max-points 3600"),
    (8, "medical_smallpoint_g16_m1024_p2800", "medical", "point", 1024, "--grid-step 16 --border 8 --max-points 2800"),
    (9, "medical_smallpoint_g20_m1024_p2200", "medical", "point", 1024, "--grid-step 20 --border 10 --max-points 2200"),
    (10, "medical_smallpoint_g24_m1024_p1800", "medical", "point", 1024, "--grid-step 24 --border 12 --max-points 1800"),
    (11, "medical_smallpoint_g32_m1024_p1200", "medical", "point", 1024, "--grid-step 32 --border 16 --max-points 1200"),
    # Smaller sliding boxes.  The previous box192/stride48 is retained only as
    # a comparison in older roots; this sweep focuses below 128 px.
    (12, "base_smallbox_b32_s8_m1024", "base", "box", 1024, "--box-size 32 --box-stride 8"),
    (13, "base_smallbox_b40_s10_m1024", "base", "box", 1024, "--box-size 40 --box-stride 10"),
    (14, "base_smallbox_b48_s12_m1024", "base", "box", 1024, "--box-size 48 --box-stride 12"),
    (15, "base_smallbox_b64_s16_m1024", "base", "box", 1024, "--box-size 64 --box-stride 16"),
    (16, "base_smallbox_b80_s20_m1024", "base", "box", 1024, "--box-size 80 --box-stride 20"),
    (17, "base_smallbox_b96_s24_m1024", "base", "box", 1024, "--box-size 96 --box-stride 24"),
    (18, "base_smallbox_b128_s32_m1024", "base", "box", 1024, "--box-size 128 --box-stride 32"),
    (19, "medical_smallbox_b32_s8_m1024", "medical", "box", 1024, "--box-size 32 --box-stride 8"),
    (20, "medical_smallbox_b40_s10_m1024", "medical", "box", 1024, "--box-size 40 --box-stride 10"),
    (21, "medical_smallbox_b48_s12_m1024", "medical", "box", 1024, "--box-size 48 --box-stride 12"),
    (22, "medical_smallbox_b64_s16_m1024", "medical", "box", 1024, "--box-size 64 --box-stride 16"),
    (23, "medical_smallbox_b80_s20_m1024", "medical", "box", 1024, "--box-size 80 --box-stride 20"),
    (24, "medical_smallbox_b96_s24_m1024", "medical", "box", 1024, "--box-size 96 --box-stride 24"),
    (25, "medical_smallbox_b128_s32_m1024", "medical", "box", 1024, "--box-size 128 --box-stride 32"),
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def copy_bundle(base_prepared: Path, out_root: Path, reset: bool) -> list[dict]:
    if reset and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ["source_manifest.csv", "label_manifest.csv", "no_label_eval_summary_missing.json"]:
        src = base_prepared / name
        if src.exists():
            shutil.copy2(src, out_root / name)
    src_tree = base_prepared / "per_tma"
    dst_tree = out_root / "per_tma"
    if dst_tree.exists() and reset:
        shutil.rmtree(dst_tree)
    if not dst_tree.exists():
        shutil.copytree(src_tree, dst_tree)
    return read_csv(out_root / "source_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prepared-root", type=Path, default=DEFAULT_BASE_PREPARED)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = copy_bundle(args.base_prepared_root, args.out_root, args.reset)
    settings = hd.settings_rows(SMALLCORE_SETTINGS)
    tasks = hd.task_rows(source_rows, SMALLCORE_SETTINGS)
    hd.write_csv(args.out_root / "settings_manifest_smallcore.csv", settings)
    hd.write_csv(args.out_root / "settings_manifest.csv", settings)
    hd.write_csv(args.out_root / "settings_manifest_48.csv", settings)
    hd.write_csv(args.out_root / "sam_array_tasks.csv", tasks)
    summary = {
        "base_prepared_root": str(args.base_prepared_root),
        "out_root": str(args.out_root),
        "profile": "smallcore",
        "n_tmas": len({int(row["tma"]) for row in source_rows}),
        "n_source_images": len(source_rows),
        "n_settings": len(settings),
        "n_sam_array_tasks": len(tasks),
        "settings": [row["setting_name"] for row in settings],
        "note": "Small-core prompt sweep: denser point grids and boxes below 128 px for Xenium TMA crops.",
    }
    (args.out_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_root / "README.md").write_text(
        "\n".join(
            [
                "# Xenium small-core SAM input bundle",
                "",
                "This bundle reuses the manually accepted H&E/FICTURE source crops and hidden annotation masks,",
                "but replaces the old VisiumHD-scale prompts with tighter prompts for smaller Xenium TMA cores.",
                "",
                f"- source images: {len(source_rows)}",
                f"- settings per source: {len(settings)}",
                f"- total SAM array tasks: {len(tasks)}",
                "",
                "No annotation mask is passed to SAM.  Annotation masks are hidden and used only by the scorer.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
