#!/usr/bin/env python3
"""Prepare SAM inputs from the current high-res H&E + molecule FICTURE report.

This intentionally avoids the older small-core bundle, which used low-resolution
manual H&E crops and deprecated cell/cell-count-like FICTURE-style images.

Inputs are read from:
  output/aaai_xenium_silica_20260623/tma_sample_overview_ficturedriven_highres_20260626

The resulting bundle contains only runtime model inputs in source_manifest.csv:
  - he
  - ficture_molecule_k12

Ground-truth masks are copied only into label_manifest.csv for hidden scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import xenium_prepare_hdstyle_sam_inputs as hd
import xenium_prepare_smallcore_hdstyle_sam_inputs as smallcore
import xenium_wholeslide_tif_he_translation_registration as ws_align

AAAI_ROOT = ROOT / "output/aaai_xenium_silica_20260623"
DEFAULT_REPORT_ROOT = AAAI_ROOT / "tma_sample_overview_ficturedriven_highres_20260626"
DEFAULT_OUT = AAAI_ROOT / "xenium_hdstyle_sam_inputs_highres_ficturedriven_20260626"


def slugify(text: str, max_len: int = 72) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "label"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rgb_from_hex(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def mask_for_label(rgb_path: Path, label: str, out_path: Path) -> tuple[int, tuple[int, int]]:
    color = rgb_from_hex(ws_align.base.GT_PALETTE[label])
    image = Image.open(rgb_path).convert("RGB")
    arr = np.asarray(image)
    mask = np.all(arr == np.asarray(color, dtype=np.uint8), axis=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(out_path)
    return int(mask.sum()), image.size


def labels_for_record(rec: dict) -> list[str]:
    meta = rec.get("gt_source_meta") or {}
    counts = meta.get("point_counts") or {}
    labels = [str(label) for label in counts if str(label) in ws_align.base.GT_PALETTE]
    if labels:
        return sorted(labels, key=lambda item: slugify(item))
    continuous = Image.open(rec["gt_continuous_mask_abs"]).convert("RGB")
    colors = {tuple(v) for v in np.asarray(continuous).reshape(-1, 3)}
    labels = []
    for label, hex_color in ws_align.base.GT_PALETTE.items():
        if rgb_from_hex(hex_color) in colors:
            labels.append(label)
    return sorted(labels, key=lambda item: slugify(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.report_root / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.reset and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict] = []
    label_rows: list[dict] = []
    qc_rows: list[dict] = []

    for rec in sorted(records, key=lambda row: int(row["tma_num"])):
        tma_num = int(rec["tma_num"])
        tma_name = f"TMA{tma_num:02d}"
        slide = str(rec["slide"])
        tma_dir = args.out_root / "per_tma" / tma_name
        source_dir = tma_dir / "source_images"

        he_src = Path(rec["he_abs"])
        fic_src = Path(rec["ficture_high_abs"])
        he_dst = source_dir / f"{tma_name}_he.png"
        fic_dst = source_dir / f"{tma_name}_ficture_molecule_k12.png"
        link_or_copy(he_src, he_dst)
        link_or_copy(fic_src, fic_dst)

        he_image = Image.open(he_dst).convert("RGB")
        fic_image = Image.open(fic_dst).convert("RGB")
        if he_image.size != fic_image.size:
            raise RuntimeError(f"{tma_name}: H&E and FICTURE size mismatch: {he_image.size} vs {fic_image.size}")

        source_rows.extend(
            [
                {
                    "tma": tma_num,
                    "slide": slide,
                    "source_key": "he",
                    "description": "Original-resolution H&E crop driven by complete K=12 molecule FICTURE footprint",
                    "image_rel": str(he_dst.relative_to(args.out_root)),
                    "width": he_image.width,
                    "height": he_image.height,
                    "source_abs": str(he_src),
                    "crop_policy": rec.get("crop_policy", ""),
                },
                {
                    "tma": tma_num,
                    "slide": slide,
                    "source_key": "ficture_molecule_k12",
                    "description": "K=12 molecule-level FICTURE/punkst map warped into the same high-res H&E canvas",
                    "image_rel": str(fic_dst.relative_to(args.out_root)),
                    "width": fic_image.width,
                    "height": fic_image.height,
                    "source_abs": str(fic_src),
                    "input_basis": "Xenium transcript/molecule locations and gene identities; not cell-count or cell-centroid map",
                    "color_basis": "FICTURE/punkst K=12 cmap48 first-12 RGB colors",
                    "not_cell_centroid_map": True,
                    "not_cell_count_weighted_map": True,
                },
            ]
        )

        labels = labels_for_record(rec)
        for idx, label in enumerate(labels, start=1):
            slug = slugify(label)
            point_path = tma_dir / "annotation_masks_original_points" / f"{idx:02d}_{slug}.png"
            dense_path = tma_dir / "annotation_masks_continuous_region" / f"{idx:02d}_{slug}.png"
            point_area, point_size = mask_for_label(Path(rec["gt_original_mask_abs"]), label, point_path)
            dense_area, dense_size = mask_for_label(Path(rec["gt_continuous_mask_abs"]), label, dense_path)
            if point_size != he_image.size or dense_size != he_image.size:
                raise RuntimeError(f"{tma_name} {label}: GT mask size mismatch")
            label_rows.append(
                {
                    "tma": tma_num,
                    "label_slug": slug,
                    "label": label,
                    "point_mask_rel": str(point_path.relative_to(args.out_root)),
                    "dense_mask_rel": str(dense_path.relative_to(args.out_root)),
                    "point_area_px": point_area,
                    "dense_area_px": dense_area,
                    "annotation_policy": "original_points_and_shape_preserving_continuous_region",
                    "continuous_region_policy": "matched cell_boundaries plus 8px label-wise closing and hole filling; no final dilation",
                    "gt_source": "new_annotation, hidden from SAM and used only for scoring",
                }
            )

        qc_rows.append(
            {
                "tma": tma_name,
                "slide": slide,
                "he_size": f"{he_image.width}x{he_image.height}",
                "ficture_size": f"{fic_image.width}x{fic_image.height}",
                "n_labels": len(labels),
                "labels": ";".join(labels),
                "raw_he_member": rec.get("raw_he_member", ""),
                "raw_he_member_px_unrotated": rec.get("raw_he_member_px_unrotated", ""),
                "raw_he_member_bytes": rec.get("raw_he_member_bytes", ""),
                "ficture_native_size": rec.get("ficture_native_size", ""),
                "old_he_size": rec.get("old_he_size", ""),
                "expanded_lowres_size": rec.get("expanded_lowres_size", ""),
            }
        )

    settings = hd.settings_rows(smallcore.SMALLCORE_SETTINGS)
    tasks = hd.task_rows(source_rows, smallcore.SMALLCORE_SETTINGS)
    write_csv(args.out_root / "source_manifest.csv", source_rows)
    write_csv(args.out_root / "label_manifest.csv", label_rows)
    write_csv(args.out_root / "source_pair_manifest.csv", [{"source_keys": "he ficture_molecule_k12"}])
    write_csv(args.out_root / "settings_manifest.csv", settings)
    write_csv(args.out_root / "settings_manifest_smallcore.csv", settings)
    write_csv(args.out_root / "settings_manifest_48.csv", settings)
    write_csv(args.out_root / "sam_array_tasks.csv", tasks)
    write_csv(args.out_root / "qc_manifest.csv", qc_rows)

    summary = {
        "report_root": str(args.report_root),
        "source_report_manifest": str(manifest_path),
        "out_root": str(args.out_root),
        "profile": "highres_ficturedriven_k12",
        "n_tmas": len(qc_rows),
        "n_source_images": len(source_rows),
        "source_keys": sorted({row["source_key"] for row in source_rows}),
        "n_labels": len(label_rows),
        "n_settings": len(settings),
        "n_sam_array_tasks": len(tasks),
        "sam_runtime_inputs": ["he", "ficture_molecule_k12"],
        "hidden_gt_policies": ["original_points", "continuous_region"],
        "default_scoring_annotation_policy": "dense",
        "old_lowres_smallcore_bundle_excluded": True,
        "deprecated_cell_proxy_ficture_excluded": True,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_root / "README.md").write_text(
        "\n".join(
            [
                "# Xenium high-res FICTURE-driven SAM input bundle",
                "",
                "This is the current SAM-ready input bundle for the Xenium Silica AAAI run.",
                "",
                "- Runtime inputs: `he` and `ficture_molecule_k12`.",
                "- H&E is the high-resolution crop from the original full H&E TIFF.",
                "- FICTURE is K=12 molecule-level FICTURE/punkst warped into the same H&E canvas.",
                "- GT masks are hidden scoring masks only; they are not passed to SAM.",
                "- `dense_mask_rel` is the current continuous-region GT.",
                "- `point_mask_rel` is the original sparse point GT.",
                "",
                "Run SAM candidate generation with `sam_array_tasks.csv` and score with source keys:",
                "",
                "```bash",
                "python scripts/xenium_score_hdstyle_sam_candidates.py \\",
                "  --prepared-root <this bundle> \\",
                "  --candidate-root <candidate root>/sam_candidate_masks \\",
                "  --source-keys he ficture_molecule_k12 \\",
                "  --annotation-policy dense \\",
                "  ...",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
