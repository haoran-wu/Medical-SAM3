#!/usr/bin/env python3
"""Prepare Xenium Silica TMA inputs for the old VisiumHD-style SAM sweep.

This script does not run SAM. It only packages the manually accepted Xenium
Silica crops into the input shape needed by the previous 48-setting candidate
proposal workflow:

- one H&E source image per TMA,
- three FICTURE-style density variants per TMA,
- per-tissue-class annotation masks,
- CSV manifests for cluster array jobs and downstream hidden-oracle scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_xenium_manual_aligned_annotation_tma_panels as render_panels


DEFAULT_MANUAL_ROOT = Path(
    "output/aaai_xenium_silica_20260623/manual_accepted_alignment_annotation_tmas_20260624"
)
DEFAULT_OUT = Path(
    "output/aaai_xenium_silica_20260623/xenium_hdstyle_sam_inputs_20260624"
)


LEGACY_SETTINGS = [
    (0, "base_official_points_step24", "base", "point", 2048, "--grid-step 24 --border 12 --max-points 3600"),
    (1, "base_official_points_step32", "base", "point", 2048, "--grid-step 32 --border 16 --max-points 3000"),
    (2, "base_official_boxes_b192_s48", "base", "box", 2048, "--box-size 192 --box-stride 48"),
    (3, "base_official_boxes_b384_s96", "base", "box", 2048, "--box-size 384 --box-stride 96"),
    (4, "medical_official_points_step24", "medical", "point", 2048, "--grid-step 24 --border 12 --max-points 3600"),
    (5, "medical_official_points_step32", "medical", "point", 2048, "--grid-step 32 --border 16 --max-points 3000"),
    (6, "medical_official_boxes_b192_s48", "medical", "box", 2048, "--box-size 192 --box-stride 48"),
    (7, "medical_official_boxes_b384_s96", "medical", "box", 2048, "--box-size 384 --box-stride 96"),
    (8, "base_box96_s48_m1536", "base", "box", 1536, "--box-size 96 --box-stride 48"),
    (9, "base_box128_s48_m1536", "base", "box", 1536, "--box-size 128 --box-stride 48"),
    (10, "base_box160_s64_m1536", "base", "box", 1536, "--box-size 160 --box-stride 64"),
    (11, "base_box192_s64_m1536", "base", "box", 1536, "--box-size 192 --box-stride 64"),
    (12, "base_box256_s96_m1536", "base", "box", 1536, "--box-size 256 --box-stride 96"),
    (13, "base_box384_s128_m1536", "base", "box", 1536, "--box-size 384 --box-stride 128"),
    (14, "base_box512_s160_m1536", "base", "box", 1536, "--box-size 512 --box-stride 160"),
    (15, "base_box96_s48_m2048", "base", "box", 2048, "--box-size 96 --box-stride 48"),
    (16, "base_box128_s48_m2048", "base", "box", 2048, "--box-size 128 --box-stride 48"),
    (17, "base_box160_s64_m2048", "base", "box", 2048, "--box-size 160 --box-stride 64"),
    (18, "base_box192_s64_m2048", "base", "box", 2048, "--box-size 192 --box-stride 64"),
    (19, "base_box256_s96_m2048", "base", "box", 2048, "--box-size 256 --box-stride 96"),
    (20, "base_box384_s128_m2048", "base", "box", 2048, "--box-size 384 --box-stride 128"),
    (21, "base_box512_s160_m2048", "base", "box", 2048, "--box-size 512 --box-stride 160"),
    (22, "medical_box96_s48_m1536", "medical", "box", 1536, "--box-size 96 --box-stride 48"),
    (23, "medical_box128_s48_m1536", "medical", "box", 1536, "--box-size 128 --box-stride 48"),
    (24, "medical_box160_s64_m1536", "medical", "box", 1536, "--box-size 160 --box-stride 64"),
    (25, "medical_box192_s64_m1536", "medical", "box", 1536, "--box-size 192 --box-stride 64"),
    (26, "medical_box256_s96_m1536", "medical", "box", 1536, "--box-size 256 --box-stride 96"),
    (27, "medical_box384_s128_m1536", "medical", "box", 1536, "--box-size 384 --box-stride 128"),
    (28, "medical_box512_s160_m1536", "medical", "box", 1536, "--box-size 512 --box-stride 160"),
    (29, "medical_box96_s48_m2048", "medical", "box", 2048, "--box-size 96 --box-stride 48"),
    (30, "medical_box128_s48_m2048", "medical", "box", 2048, "--box-size 128 --box-stride 48"),
    (31, "medical_box160_s64_m2048", "medical", "box", 2048, "--box-size 160 --box-stride 64"),
    (32, "medical_box192_s64_m2048", "medical", "box", 2048, "--box-size 192 --box-stride 64"),
    (33, "medical_box256_s96_m2048", "medical", "box", 2048, "--box-size 256 --box-stride 96"),
    (34, "medical_box384_s128_m2048", "medical", "box", 2048, "--box-size 384 --box-stride 128"),
    (35, "medical_box512_s160_m2048", "medical", "box", 2048, "--box-size 512 --box-stride 160"),
    (36, "base_point48_m1024_p900", "base", "point", 1024, "--grid-step 48 --border 24 --max-points 900"),
    (37, "base_point64_m1024_p700", "base", "point", 1024, "--grid-step 64 --border 32 --max-points 700"),
    (38, "base_point80_m1024_p500", "base", "point", 1024, "--grid-step 80 --border 40 --max-points 500"),
    (39, "base_point64_m1536_p900", "base", "point", 1536, "--grid-step 64 --border 32 --max-points 900"),
    (40, "base_point80_m1536_p700", "base", "point", 1536, "--grid-step 80 --border 40 --max-points 700"),
    (41, "base_point96_m1536_p500", "base", "point", 1536, "--grid-step 96 --border 48 --max-points 500"),
    (42, "medical_point48_m1024_p900", "medical", "point", 1024, "--grid-step 48 --border 24 --max-points 900"),
    (43, "medical_point64_m1024_p700", "medical", "point", 1024, "--grid-step 64 --border 32 --max-points 700"),
    (44, "medical_point80_m1024_p500", "medical", "point", 1024, "--grid-step 80 --border 40 --max-points 500"),
    (45, "medical_point64_m1536_p900", "medical", "point", 1536, "--grid-step 64 --border 32 --max-points 900"),
    (46, "medical_point80_m1536_p700", "medical", "point", 1536, "--grid-step 80 --border 40 --max-points 700"),
    (47, "medical_point96_m1536_p500", "medical", "point", 1536, "--grid-step 96 --border 48 --max-points 500"),
]

MISSING_SCREENSHOT_SETTINGS = [
    (48, "base_box64_s32_m1536", "base", "box", 1536, "--box-size 64 --box-stride 32"),
    (49, "base_box768_s256_m1536", "base", "box", 1536, "--box-size 768 --box-stride 256"),
    (50, "base_box64_s32_m2048", "base", "box", 2048, "--box-size 64 --box-stride 32"),
    (51, "base_box768_s256_m2048", "base", "box", 2048, "--box-size 768 --box-stride 256"),
    (52, "medical_box64_s32_m1536", "medical", "box", 1536, "--box-size 64 --box-stride 32"),
    (53, "medical_box768_s256_m1536", "medical", "box", 1536, "--box-size 768 --box-stride 256"),
    (54, "medical_box64_s32_m2048", "medical", "box", 2048, "--box-size 64 --box-stride 32"),
    (55, "medical_box768_s256_m2048", "medical", "box", 2048, "--box-size 768 --box-stride 256"),
    (56, "base_point40_m1024_p1200", "base", "point", 1024, "--grid-step 40 --border 20 --max-points 1200"),
    (57, "base_point40_m1536_p1600", "base", "point", 1536, "--grid-step 40 --border 20 --max-points 1600"),
    (58, "medical_point40_m1024_p1200", "medical", "point", 1024, "--grid-step 40 --border 20 --max-points 1200"),
    (59, "medical_point40_m1536_p1600", "medical", "point", 1536, "--grid-step 40 --border 20 --max-points 1600"),
]

SETTINGS = LEGACY_SETTINGS + MISSING_SCREENSHOT_SETTINGS


SOURCE_VARIANTS = [
    ("he", "he", "H&E no-margin manually aligned crop"),
    ("ficture_sparse", "ficture_style_celltype_sparse_points", "Sparse Xenium cellTypeFibSim point map remapped to June16 FICTURE RGB"),
    ("ficture_medium", "ficture_style_celltype", "Medium-density Xenium cellTypeFibSim map remapped to June16 FICTURE RGB"),
    ("ficture_full_dense", "ficture_style_celltype_full_dense_nearest", "Nearest-cell dense Xenium cellTypeFibSim map remapped to June16 FICTURE RGB"),
]


def slugify(text: str, max_len: int = 72) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "label"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def draw_point_masks(
    size: tuple[int, int],
    xy: np.ndarray,
    labels: list[str],
    *,
    radius: int,
) -> dict[str, np.ndarray]:
    width, height = size
    out: dict[str, Image.Image] = {}
    for (x, y), label in zip(xy, labels):
        label = render_panels.clean_label(label)
        if not label or not np.isfinite(x) or not np.isfinite(y):
            continue
        if label not in out:
            out[label] = Image.new("L", size, 0)
        draw = ImageDraw.Draw(out[label])
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if -radius <= xi < width + radius and -radius <= yi < height + radius:
            draw.ellipse((xi - radius, yi - radius, xi + radius, yi + radius), fill=255)
    return {label: np.asarray(image) > 0 for label, image in out.items()}


def dense_label_masks(
    he: Image.Image,
    xy: np.ndarray,
    labels: list[str],
    *,
    max_distance_px: float,
) -> dict[str, np.ndarray]:
    tissue = render_panels.make_tissue_mask_from_he(he)
    yy, xx = np.nonzero(tissue)
    if len(xx) == 0:
        return {}

    valid_idx: list[int] = []
    valid_labels: list[str] = []
    for idx, ((x, y), label) in enumerate(zip(xy, labels)):
        label = render_panels.clean_label(label)
        if not label or not np.isfinite(x) or not np.isfinite(y):
            continue
        valid_idx.append(idx)
        valid_labels.append(label)
    if not valid_idx:
        return {}

    pts = xy[np.asarray(valid_idx, dtype=int)].astype(float)
    tree = cKDTree(pts)
    query = np.column_stack([xx.astype(float), yy.astype(float)])
    dist, nearest = tree.query(query, k=1)
    keep = dist <= max_distance_px
    masks = {label: np.zeros(tissue.shape, dtype=bool) for label in sorted(set(valid_labels))}
    labels_arr = np.asarray(valid_labels, dtype=object)
    assigned_labels = labels_arr[nearest[keep]]
    assigned_x = xx[keep]
    assigned_y = yy[keep]
    for label in masks:
        select = assigned_labels == label
        masks[label][assigned_y[select], assigned_x[select]] = True
    return masks


def tma_points_and_labels(tma: int, source_root: Path, rec: dict, he: Image.Image) -> tuple[np.ndarray, list[str]]:
    slide = rec["slide"]
    manifest = render_panels.manual_tool.read_manifest(source_root, slide)
    params = render_panels.load_manual_params(source_root, slide, tma)
    rows = render_panels.ws_align.base.read_tma_frame(tma)
    xy = render_panels.points_for_tma(
        rows,
        manifest,
        params["crop_bbox_xyxy_in_oriented_he_grid"],
        he.width,
        he.height,
        params,
    )
    labels = [render_panels.clean_label(row.get(render_panels.ws_align.base.GT_COL, "")) for row in rows]
    return xy, labels


def copy_source_images(
    manual_root: Path,
    out_root: Path,
    rec: dict,
) -> tuple[list[dict], Image.Image]:
    tma = int(rec["tma"])
    slide = rec["slide"]
    tma_dir = out_root / "per_tma" / f"TMA{tma:02d}"
    source_dir = tma_dir / "source_images"
    source_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    he_image: Image.Image | None = None
    for source_key, manifest_key, description in SOURCE_VARIANTS:
        src = manual_root / rec["outputs"][manifest_key]
        dst = source_dir / f"TMA{tma:02d}_{source_key}.png"
        shutil.copy2(src, dst)
        image = Image.open(dst).convert("RGB")
        if source_key == "he":
            he_image = image.copy()
        rows.append(
            {
                "tma": tma,
                "slide": slide,
                "source_key": source_key,
                "description": description,
                "image_rel": str(dst.relative_to(out_root)),
                "width": image.width,
                "height": image.height,
                "status": rec.get("status", ""),
            }
        )
    if he_image is None:
        raise RuntimeError(f"TMA{tma} did not produce an H&E source image")
    return rows, he_image


def write_annotation_masks(
    out_root: Path,
    source_root: Path,
    rec: dict,
    he: Image.Image,
    *,
    point_radius: int,
    dense_max_distance_px: float,
) -> list[dict]:
    tma = int(rec["tma"])
    xy, labels = tma_points_and_labels(tma, source_root, rec, he)
    point_masks = draw_point_masks(he.size, xy, labels, radius=point_radius)
    dense_masks = dense_label_masks(he, xy, labels, max_distance_px=dense_max_distance_px)
    tma_dir = out_root / "per_tma" / f"TMA{tma:02d}"
    point_dir = tma_dir / "annotation_masks_points"
    dense_dir = tma_dir / "annotation_masks_dense"

    rows = []
    for idx, label in enumerate(sorted(set(point_masks) | set(dense_masks)), start=1):
        slug = slugify(label)
        point = point_masks.get(label, np.zeros((he.height, he.width), dtype=bool))
        dense = dense_masks.get(label, np.zeros((he.height, he.width), dtype=bool))
        point_path = point_dir / f"{idx:02d}_{slug}.png"
        dense_path = dense_dir / f"{idx:02d}_{slug}.png"
        save_binary_mask(point, point_path)
        save_binary_mask(dense, dense_path)
        rows.append(
            {
                "tma": tma,
                "label_slug": slug,
                "label": label,
                "point_mask_rel": str(point_path.relative_to(out_root)),
                "dense_mask_rel": str(dense_path.relative_to(out_root)),
                "point_area_px": int(point.sum()),
                "dense_area_px": int(dense.sum()),
                "annotation_policy": "dense_nearest_annotated_cell_with_point_mask_backup",
                "point_radius_px": point_radius,
                "dense_max_distance_px": dense_max_distance_px,
            }
        )
    return rows


def select_settings(mode: str) -> list[tuple[int, str, str, str, int, str]]:
    if mode == "legacy48":
        return LEGACY_SETTINGS
    if mode == "missing_screenshot":
        return MISSING_SCREENSHOT_SETTINGS
    if mode == "all":
        return SETTINGS
    raise ValueError(f"Unknown settings mode: {mode}")


def settings_rows(settings: list[tuple[int, str, str, str, int, str]]) -> list[dict]:
    rows = []
    for setting_id, name, checkpoint_kind, mode, max_side, extra in settings:
        rows.append(
            {
                "setting_id": setting_id,
                "setting_name": name,
                "checkpoint_kind": checkpoint_kind,
                "proposal_mode": mode,
                "max_side": max_side,
                "extra_args": extra,
            }
        )
    return rows


def task_rows(source_rows: list[dict], settings: list[tuple[int, str, str, str, int, str]]) -> list[dict]:
    rows = []
    for source_index, source in enumerate(source_rows):
        for setting in settings_rows(settings):
            task_id = len(rows)
            rows.append(
                {
                    "task_id": task_id,
                    "source_index": source_index,
                    "tma": source["tma"],
                    "slide": source["slide"],
                    "source_key": source["source_key"],
                    "image_rel": source["image_rel"],
                    "setting_id": setting["setting_id"],
                    "setting_name": setting["setting_name"],
                    "checkpoint_kind": setting["checkpoint_kind"],
                    "proposal_mode": setting["proposal_mode"],
                    "max_side": setting["max_side"],
                    "extra_args": setting["extra_args"],
                    "output_rel": f"sam_candidate_masks/TMA{int(source['tma']):02d}/{source['source_key']}/{setting['setting_name']}",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-root", type=Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Registration root used to recompute aligned annotation coordinates. Defaults to manifest['root'].",
    )
    parser.add_argument("--tmas", nargs="*", type=int, default=None)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--dense-max-distance-px", type=float, default=48.0)
    parser.add_argument(
        "--settings-mode",
        choices=["legacy48", "missing_screenshot", "all"],
        default="all",
        help="Which SAM candidate settings to write. Use missing_screenshot for the supplemental screenshot-only gap fill.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_root = args.manual_root
    out_root = args.out_root
    manifest_path = manual_root / "manual_accepted_alignment_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = args.source_root or Path(data["root"])
    wanted = set(args.tmas) if args.tmas else None

    source_rows: list[dict] = []
    label_rows: list[dict] = []
    records = []
    for rec in data["records"]:
        tma = int(rec["tma"])
        if wanted is not None and tma not in wanted:
            continue
        copied, he = copy_source_images(manual_root, out_root, rec)
        labels = write_annotation_masks(
            out_root,
            source_root,
            rec,
            he,
            point_radius=args.point_radius,
            dense_max_distance_px=args.dense_max_distance_px,
        )
        source_rows.extend(copied)
        label_rows.extend(labels)
        records.append(rec)

    selected_settings = select_settings(args.settings_mode)
    settings = settings_rows(selected_settings)
    tasks = task_rows(source_rows, selected_settings)
    write_csv(out_root / "source_manifest.csv", source_rows)
    write_csv(out_root / "label_manifest.csv", label_rows)
    write_csv(out_root / f"settings_manifest_{len(settings)}.csv", settings)
    write_csv(out_root / "settings_manifest.csv", settings)
    # Compatibility path for existing Bouchet scoring scripts, which read this
    # filename but compute task count from the file contents.
    write_csv(out_root / "settings_manifest_48.csv", settings)
    write_csv(out_root / "sam_array_tasks.csv", tasks)
    summary = {
        "manual_root": str(manual_root),
        "source_root": str(source_root),
        "out_root": str(out_root),
        "n_tmas": len(records),
        "n_source_images": len(source_rows),
        "n_labels": len(label_rows),
        "n_settings": len(settings),
        "settings_mode": args.settings_mode,
        "n_sam_array_tasks": len(tasks),
        "source_variants": SOURCE_VARIANTS,
        "annotation_policy": {
            "point_radius_px": args.point_radius,
            "dense_max_distance_px": args.dense_max_distance_px,
            "dense_mask": "nearest annotated cell label inside H&E tissue mask, limited by max distance",
            "point_mask": "direct annotated-cell point circles",
        },
        "sam_policy": {
            "copied_from": "old VisiumHD 48-setting H&E/FICTURE SAM candidate sweep",
            "run_sam_locally": False,
            "expected_array_tasks": len(tasks),
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# Xenium Silica HD-style SAM inputs",
                "",
                "Prepared input bundle for reusing the old VisiumHD 48-setting SAM candidate sweep.",
                "",
                f"- TMAs: {len(records)}",
                f"- source images: {len(source_rows)} (`he`, `ficture_sparse`, `ficture_medium`, `ficture_full_dense`)",
                f"- settings per source: {len(settings)}",
                f"- total SAM array tasks: {len(tasks)}",
                "",
                "Annotation masks:",
                "- `annotation_masks_dense/`: dense nearest-annotated-cell tissue-class masks inside the H&E tissue footprint.",
                "- `annotation_masks_points/`: direct annotated-cell point masks.",
                "",
                "The dense masks are hidden oracle masks for pool evaluation. They are not inputs to SAM candidate generation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
