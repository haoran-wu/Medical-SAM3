#!/usr/bin/env python3
"""Run the fixed TMA39 independent H&E-gap rule on any registered TMA."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT_OVERRIDE", str(Path(__file__).resolve().parents[1]))
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inference.sam3_inference import SAM3Model
from run_tma39_he_gap_uniform_box_point import run_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tma", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prompt-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expansion-percent", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    all_mask_dir = args.output_dir / "all_prompt_consistent_masks"
    selected_mask_dir = args.output_dir / "selected_masks"
    all_mask_dir.mkdir()
    selected_mask_dir.mkdir()

    prompt_rows = read_csv(args.prompt_csv)
    prompt_by_region = {row["region_id"]: row for row in prompt_rows}
    if len(prompt_by_region) != len(prompt_rows):
        raise RuntimeError("Independent H&E-gap region IDs are not unique")
    crop_rows = json.loads((args.input_root / "region_crops/manifest.json").read_text())
    crop_by_region = {row["region_id"]: row for row in crop_rows}
    if set(prompt_by_region) != set(crop_by_region):
        raise RuntimeError("Prompt and crop region IDs differ")

    he_full = load_rgb(args.he_image)
    ficture_full = load_rgb(args.ficture_image)
    if he_full.shape != ficture_full.shape:
        raise RuntimeError("Registered H&E and FICTURE shapes differ")
    full_shape = he_full.shape[:2]

    if not prompt_rows:
        result_fields = [
            "tma", "region_id", "prompt_index", "prompt_type", "box_x1_full",
            "box_y1_full", "box_x2_full", "box_y2_full", "point_x_full",
            "point_y_full", "model_mask_count", "prompt_consistent_unique_mask_count",
            "rank_by_sam_score", "model_index", "sam_score", "area_pixels",
            "threshold_stability_offset0p5", "threshold_stability_offset1p0",
            "selected_by_blind_rule", "selection_used_annotation", "local_mask_path",
        ]
        selected_fields = result_fields + ["selected_mask_path", "selection_rule"]
        with (args.output_dir / "supplement_all_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.DictWriter(handle, fieldnames=result_fields).writeheader()
        with (args.output_dir / "selected_candidates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.DictWriter(handle, fieldnames=selected_fields).writeheader()
        manifest = {
            "tma": args.tma,
            "prompt_count": 0,
            "selected_candidate_count": 0,
            "prompt_type_for_every_region": "10% context box plus one automatic positive point",
            "expansion_percent_per_side": args.expansion_percent,
            "image_encoding_count": 0,
            "inference_calls_per_prompt": 1,
            "total_inference_calls": 0,
            "selection_rule": "highest SAM score among unique masks containing the positive point",
            "selection_used_annotation": False,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "he_image_path": str(args.he_image),
            "he_image_sha256": sha256(args.he_image),
            "ficture_image_path": str(args.ficture_image),
            "ficture_image_sha256": sha256(args.ficture_image),
            "prompt_csv_path": str(args.prompt_csv),
            "prompt_csv_sha256": sha256(args.prompt_csv),
            "zero_prompt_result": True,
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
        return

    sam3 = SAM3Model(
        confidence_threshold=0.1,
        checkpoint_path=str(args.checkpoint),
        device=args.device,
    )
    result_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompt_rows):
        region_id = prompt["region_id"]
        crop = crop_by_region[region_id]
        cx0, cy0, cx1, cy1 = (
            int(value) for value in crop["crop_xyxy_full_canvas"]
        )
        box_full = tuple(
            int(prompt[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2")
        )
        point_full = (int(prompt["point_x"]), int(prompt["point_y"]))
        box_crop = (
            box_full[0] - cx0,
            box_full[1] - cy0,
            box_full[2] - cx0,
            box_full[3] - cy0,
        )
        point_crop = (point_full[0] - cx0, point_full[1] - cy0)
        he_crop = he_full[cy0:cy1, cx0:cx1].copy()
        state = sam3.encode_image(he_crop)
        model_count, results = run_prompt(
            sam3, state, box_crop, point_crop, he_crop.shape[:2]
        )
        del state
        gc.collect()
        if not results:
            raise RuntimeError(f"{region_id} produced no mask containing its positive point")

        chosen = max(
            results,
            key=lambda row: (
                float(row["sam_score"]),
                float(row["threshold_stability_offset0p5"]),
                int(row["area_pixels"]),
            ),
        )
        for rank, result in enumerate(results, start=1):
            local_path = all_mask_dir / f"{region_id}_result_{rank:02d}.png"
            Image.fromarray(np.uint8(result["mask"]) * 255).save(local_path, optimize=True)
            row = {
                "tma": args.tma,
                "region_id": region_id,
                "prompt_index": prompt_index,
                "prompt_type": "10% context box plus one automatic positive point",
                "box_x1_full": box_full[0],
                "box_y1_full": box_full[1],
                "box_x2_full": box_full[2],
                "box_y2_full": box_full[3],
                "point_x_full": point_full[0],
                "point_y_full": point_full[1],
                "model_mask_count": model_count,
                "prompt_consistent_unique_mask_count": len(results),
                "rank_by_sam_score": rank,
                "model_index": result["model_index"],
                "sam_score": result["sam_score"],
                "area_pixels": result["area_pixels"],
                "threshold_stability_offset0p5": result["threshold_stability_offset0p5"],
                "threshold_stability_offset1p0": result["threshold_stability_offset1p0"],
                "selected_by_blind_rule": result is chosen,
                "selection_used_annotation": False,
                "local_mask_path": str(local_path),
            }
            result_rows.append(row)
            if result is not chosen:
                continue
            selected_full = np.zeros(full_shape, dtype=bool)
            selected_full[cy0:cy1, cx0:cx1] = np.asarray(chosen["mask"], dtype=bool)
            selected_path = selected_mask_dir / (
                f"{args.tma}_raw_he_gap_{region_id}_expand{args.expansion_percent}.png"
            )
            Image.fromarray(np.uint8(selected_full) * 255).save(selected_path, optimize=True)
            selected_rows.append(
                {
                    **row,
                    "selected_mask_path": str(selected_path),
                    "selection_rule": "highest SAM score among unique masks containing the positive point",
                }
            )
        print(
            f"[{prompt_index + 1}/{len(prompt_rows)}] {region_id}: "
            f"model={model_count} point-consistent={len(results)}",
            flush=True,
        )

    write_csv(args.output_dir / "supplement_all_results.csv", result_rows)
    write_csv(args.output_dir / "selected_candidates.csv", selected_rows)
    manifest = {
        "tma": args.tma,
        "prompt_count": len(prompt_rows),
        "selected_candidate_count": len(selected_rows),
        "prompt_type_for_every_region": "10% context box plus one automatic positive point",
        "expansion_percent_per_side": args.expansion_percent,
        "image_encoding_count": len(prompt_rows),
        "inference_calls_per_prompt": 1,
        "total_inference_calls": len(prompt_rows),
        "selection_rule": "highest SAM score among unique masks containing the positive point",
        "selection_used_annotation": False,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "he_image_path": str(args.he_image),
        "he_image_sha256": sha256(args.he_image),
        "ficture_image_path": str(args.ficture_image),
        "ficture_image_sha256": sha256(args.ficture_image),
        "prompt_csv_path": str(args.prompt_csv),
        "prompt_csv_sha256": sha256(args.prompt_csv),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
