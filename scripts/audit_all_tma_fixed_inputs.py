#!/usr/bin/env python3
"""Audit the ten registered inputs used by the frozen TMA39 transfer workflow."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TMAS = (
    "TMA07",
    "TMA24",
    "TMA29",
    "TMA30",
    "TMA31",
    "TMA34",
    "TMA36",
    "TMA39",
    "TMA41",
    "TMA42",
)
SLIDE_BY_TMA = {
    "TMA07": "TMA07_770-1",
    "TMA24": "TMA24_770-1",
    "TMA29": "TMA29_771-1",
    "TMA30": "TMA30_771-1",
    "TMA31": "TMA31_771-1",
    "TMA34": "TMA34_771-1",
    "TMA36": "TMA36_771-1",
    "TMA39": "TMA39_771-1",
    "TMA41": "TMA41_771-1",
    "TMA42": "TMA42_771-1",
}
EXPECTED_LEGEND_SHA256 = (
    "f89e0e599f6451ff37dbad24ff4b1a27b8c99bb66c033f5a40743268711181c6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--factor-info", type=Path, required=True)
    parser.add_argument("--overview-manifest", type=Path, required=True)
    parser.add_argument("--affine-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_rgb(path: Path) -> set[tuple[int, int, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12 or [int(row["Factor"]) for row in rows] != list(range(12)):
        raise RuntimeError("The fixed FICTURE legend must contain Factors 0 through 11")
    colors = {tuple(int(value) for value in ast.literal_eval(row["RGB"])) for row in rows}
    if len(colors) != 12:
        raise RuntimeError("The fixed FICTURE legend contains duplicate RGB values")
    return colors | {(0, 0, 0)}


def main() -> None:
    args = parse_args()
    if sha256(args.factor_info) != EXPECTED_LEGEND_SHA256:
        raise RuntimeError("The FICTURE factor/RGB legend is not the frozen TMA39 legend")
    overview_rows = json.loads(args.overview_manifest.read_text(encoding="utf-8"))
    overview_by_tma = {row["tma"]: row for row in overview_rows}
    if set(overview_by_tma) != set(TMAS):
        raise RuntimeError("The overview manifest does not contain exactly the ten expected TMAs")

    affine_rows = json.loads(args.affine_manifest.read_text(encoding="utf-8"))
    affine_by_tma = {row["tma"]: row for row in affine_rows}
    if set(affine_by_tma) != set(TMAS):
        raise RuntimeError("The affine-warp manifest does not contain exactly the ten expected TMAs")

    fixed_colors = allowed_rgb(args.factor_info)
    palette = np.asarray(sorted(fixed_colors - {(0, 0, 0)}), dtype=np.int16)
    audited: list[dict[str, Any]] = []
    for tma in TMAS:
        source = args.input_root / tma / "source_images"
        he_path = source / f"{tma}_he.png"
        ficture_path = source / f"{tma}_ficture_molecule_k12.png"
        if not he_path.is_file() or not ficture_path.is_file():
            raise FileNotFoundError(f"Missing registered input for {tma}")
        with Image.open(he_path) as he, Image.open(ficture_path) as ficture:
            he_size = he.size
            ficture_size = ficture.size
            if he_size != ficture_size:
                raise RuntimeError(f"Registered geometry differs for {tma}: {he_size} vs {ficture_size}")
            ficture_rgb = np.asarray(ficture.convert("RGB"))
        affine = affine_by_tma[tma]
        if list(affine["corrected_size"]) != list(he_size):
            raise RuntimeError(f"Affine-warp manifest geometry differs for {tma}")
        if Path(affine["corrected_ficture_rel"]).name != ficture_path.name:
            raise RuntimeError(f"Affine-warp manifest points to a different FICTURE image for {tma}")
        if affine.get("candidate_selection_used_annotation") is not False:
            raise RuntimeError(f"Affine-warp source for {tma} used annotation")
        if "strict official FICTURE Xenium K=12 rerun" not in affine["projection_policy"]:
            raise RuntimeError(f"{tma} is not the strict-official K=12 affine-warp source")

        # The strict source stores continuous factor mixtures, so edge and mixture
        # pixels need not equal one pure legend RGB. Audit a deterministic sample
        # against the same 12 anchor colors without altering the image.
        flat = ficture_rgb.reshape(-1, 3)
        stride = max(1, len(flat) // 1_000_000)
        sample = flat[::stride].astype(np.int32)
        nonblack = sample[np.max(sample, axis=1) > 8]
        if len(nonblack):
            distances = np.sum(
                (nonblack[:, None, :] - palette.astype(np.int32)[None, :, :]) ** 2,
                axis=2,
                dtype=np.int64,
            )
            nearest = np.sqrt(np.min(distances, axis=1))
            exact = np.any(np.all(nonblack[:, None, :] == palette[None, :, :], axis=2), axis=1)
            exact_fraction = float(exact.mean())
            nearest_median = float(np.median(nearest))
            nearest_p95 = float(np.quantile(nearest, 0.95))
        else:
            exact_fraction = nearest_median = nearest_p95 = 0.0
        audited.append(
            {
                "tma": tma,
                "slide": SLIDE_BY_TMA[tma],
                "registered_size": list(he_size),
                "he_path": str(he_path),
                "he_sha256": sha256(he_path),
                "ficture_path": str(ficture_path),
                "ficture_sha256": sha256(ficture_path),
                "ficture_color_basis": "unchanged continuous mixtures of the fixed 12 factor RGB anchors",
                "sampled_nonblack_exact_anchor_fraction": exact_fraction,
                "sampled_nearest_anchor_distance_median": nearest_median,
                "sampled_nearest_anchor_distance_p95": nearest_p95,
                "ficture_recolored_for_audit": False,
                "strict_affine_projection_policy": affine["projection_policy"],
                "overview_expanded_lowres_size": overview_by_tma[tma]["expanded_lowres_size"],
                "transcript_relative_path": (
                    "output/xenium_silica_true_punkst_input_20260626/"
                    f"{SLIDE_BY_TMA[tma]}/transcripts.sorted.tsv.gz"
                ),
            }
        )

    manifest = {
        "status": "valid",
        "workflow_id": "tma39-fixed-sam3-v1",
        "tma_count": len(audited),
        "factor_legend_path": str(args.factor_info),
        "factor_legend_sha256": EXPECTED_LEGEND_SHA256,
        "overview_manifest_path": str(args.overview_manifest),
        "overview_manifest_sha256": sha256(args.overview_manifest),
        "affine_manifest_path": str(args.affine_manifest),
        "affine_manifest_sha256": sha256(args.affine_manifest),
        "tmas": audited,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
