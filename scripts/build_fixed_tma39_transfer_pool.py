#!/usr/bin/env python3
"""Freeze a cross-TMA pool with the final TMA39 FICTURE-primary rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HE_NOVELTY_THRESHOLD = 0.40
DEDUP_IOU_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tma", required=True)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--gap-root", type=Path, required=True)
    parser.add_argument("--method", default="genemap_support_f1")
    parser.add_argument("--output-dir", type=Path, required=True)
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


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if not mask.any():
        raise RuntimeError(f"Empty mask: {path}")
    return mask


def selected_by_method(root: Path, source: str, method: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(root / source / "selected_candidates.csv")
        if row["selection_method"] == method
    ]
    rows.sort(key=lambda row: int(row["prompt_index"]))
    if not rows:
        raise RuntimeError(f"No {source}/{method} masks found")
    if len({int(row["prompt_index"]) for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate prompt indices in {source}/{method}")
    for row in rows:
        row["resolved_mask_path"] = str(root / source / row["selected_mask_path"])
    return rows


def mask_record(path: Path) -> dict[str, Any]:
    mask = load_mask(path)
    yy, xx = np.nonzero(mask)
    bbox = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    x1, y1, x2, y2 = bbox
    return {
        "bbox": bbox,
        "crop": mask[y1:y2, x1:x2],
        "area": int(mask.sum()),
    }


def record_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    x1 = max(first["bbox"][0], second["bbox"][0])
    y1 = max(first["bbox"][1], second["bbox"][1])
    x2 = min(first["bbox"][2], second["bbox"][2])
    y2 = min(first["bbox"][3], second["bbox"][3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    fx1, fy1, _, _ = first["bbox"]
    sx1, sy1, _, _ = second["bbox"]
    shared = int(
        np.logical_and(
            first["crop"][y1 - fy1 : y2 - fy1, x1 - fx1 : x2 - fx1],
            second["crop"][y1 - sy1 : y2 - sy1, x1 - sx1 : x2 - sx1],
        ).sum()
    )
    return shared / max(1, first["area"] + second["area"] - shared)


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        record = mask_record(Path(row["source_mask_path"]))
        best_iou = 0.0
        best_id = ""
        for earlier, earlier_record in zip(kept, records):
            iou = record_iou(record, earlier_record)
            if iou > best_iou:
                best_iou, best_id = iou, earlier["candidate_id"]
        removed = best_iou >= DEDUP_IOU_THRESHOLD
        audit.append(
            {
                "candidate_id": row["candidate_id"],
                "source": row["source"],
                "highest_iou_with_earlier_retained_mask": best_iou,
                "closest_earlier_candidate": best_id,
                "duplicate_threshold": DEDUP_IOU_THRESHOLD,
                "decision": "removed_duplicate" if removed else "retained",
                "selection_used_annotation": False,
            }
        )
        if not removed:
            kept.append(row)
            records.append(record)
    return kept, audit


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir()

    ficture_rows = selected_by_method(args.sam_root, "ficture", args.method)
    he_rows = selected_by_method(args.sam_root, "he", args.method)
    ficture_manifest = json.loads(
        (args.sam_root / "ficture/run_manifest.json").read_text(encoding="utf-8")
    )
    he_manifest = json.loads(
        (args.sam_root / "he/run_manifest.json").read_text(encoding="utf-8")
    )
    prompt_count = int(ficture_manifest["prompt_count"])
    if int(he_manifest["prompt_count"]) != prompt_count:
        raise RuntimeError("FICTURE and H&E prompt counts differ")
    ficture_by_prompt = {int(row["prompt_index"]): row for row in ficture_rows}
    he_by_prompt = {int(row["prompt_index"]): row for row in he_rows}
    if len(ficture_by_prompt) != len(ficture_rows):
        raise RuntimeError("Duplicate FICTURE prompt indices")
    if len(he_by_prompt) != len(he_rows):
        raise RuntimeError("Duplicate H&E prompt indices")
    valid_prompt_indices = set(range(prompt_count))
    if not set(ficture_by_prompt).issubset(valid_prompt_indices):
        raise RuntimeError("FICTURE contains an out-of-range prompt index")
    if not set(he_by_prompt).issubset(valid_prompt_indices):
        raise RuntimeError("H&E contains an out-of-range prompt index")
    ficture_audit_by_prompt = {
        int(row["prompt_index"]): row
        for row in read_csv(args.sam_root / "ficture/prompt_audit.csv")
    }
    if set(ficture_audit_by_prompt) != valid_prompt_indices:
        raise RuntimeError("FICTURE prompt audit does not cover every prompt exactly once")

    first_mask = load_mask(Path(ficture_rows[0]["resolved_mask_path"]))
    ficture_union = np.zeros(first_mask.shape, dtype=bool)
    for row in ficture_rows:
        ficture_union |= load_mask(Path(row["resolved_mask_path"]))

    pair_rows: list[dict[str, Any]] = []
    pre_dedup: list[dict[str, Any]] = []
    for prompt_index in range(prompt_count):
        ficture_row = ficture_by_prompt.get(prompt_index)
        he_row = he_by_prompt.get(prompt_index)
        if ficture_row is None:
            prompt_meta = ficture_audit_by_prompt[prompt_index]
            if he_row is None:
                pair_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_id": prompt_meta["prompt_id"],
                        "gene_module": prompt_meta["gene_module"],
                        "ficture_area": "",
                        "he_area": "",
                        "shared_pixels": "",
                        "ficture_to_he_purity": "",
                        "he_to_ficture_purity": "",
                        "pair_iou": "",
                        "he_new_pixels_against_all_ficture": "",
                        "he_new_fraction_against_all_ficture": "",
                        "same_prompt_he_retained": False,
                        "same_prompt_he_threshold": HE_NOVELTY_THRESHOLD,
                        "same_prompt_he_status": "no eligible FICTURE or H&E mask for this prompt",
                        "selection_used_annotation": False,
                    }
                )
                continue
            he_mask = load_mask(Path(he_row["resolved_mask_path"]))
            new_he = int(np.logical_and(he_mask, ~ficture_union).sum())
            he_new_fraction = new_he / max(1, int(he_mask.sum()))
            keep_he = he_new_fraction >= HE_NOVELTY_THRESHOLD
            pair_rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_meta["prompt_id"],
                    "gene_module": prompt_meta["gene_module"],
                    "ficture_area": "",
                    "he_area": int(he_mask.sum()),
                    "shared_pixels": "",
                    "ficture_to_he_purity": "",
                    "he_to_ficture_purity": "",
                    "pair_iou": "",
                    "he_new_pixels_against_all_ficture": new_he,
                    "he_new_fraction_against_all_ficture": he_new_fraction,
                    "same_prompt_he_retained": keep_he,
                    "same_prompt_he_threshold": HE_NOVELTY_THRESHOLD,
                    "same_prompt_he_status": (
                        "no eligible FICTURE primary; available H&E mask evaluated by the 40% new-area rule"
                    ),
                    "selection_used_annotation": False,
                }
            )
            if keep_he:
                pre_dedup.append(
                    {
                        "candidate_id": f"{args.tma}:HE_SAME:P{prompt_index + 1:03d}",
                        "source": "Same-prompt H&E supplement",
                        "prompt_index": prompt_index,
                        "prompt_id": he_row["prompt_id"],
                        "gene_module": he_row["gene_module"],
                        "source_mask_path": he_row["resolved_mask_path"],
                        "sam_score": he_row["sam_score"],
                        "selection_method": args.method,
                        "selection_reason": "At least 40% of this H&E mask lies outside the complete FICTURE-primary pool.",
                        "selection_used_annotation": False,
                    }
                )
            continue

        ficture_mask = load_mask(Path(ficture_row["resolved_mask_path"]))
        if he_row is None:
            pair_rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": ficture_row["prompt_id"],
                    "gene_module": ficture_row["gene_module"],
                    "ficture_area": int(ficture_mask.sum()),
                    "he_area": "",
                    "shared_pixels": "",
                    "ficture_to_he_purity": "",
                    "he_to_ficture_purity": "",
                    "pair_iou": "",
                    "he_new_pixels_against_all_ficture": "",
                    "he_new_fraction_against_all_ficture": "",
                    "same_prompt_he_retained": False,
                    "same_prompt_he_threshold": HE_NOVELTY_THRESHOLD,
                    "same_prompt_he_status": "no eligible H&E mask containing the positive point",
                    "selection_used_annotation": False,
                }
            )
            pre_dedup.append(
                {
                    "candidate_id": f"{args.tma}:FICTURE:P{prompt_index + 1:03d}",
                    "source": "FICTURE primary",
                    "prompt_index": prompt_index,
                    "prompt_id": ficture_row["prompt_id"],
                    "gene_module": ficture_row["gene_module"],
                    "source_mask_path": ficture_row["resolved_mask_path"],
                    "sam_score": ficture_row["sam_score"],
                    "selection_method": args.method,
                    "selection_reason": "Primary candidate from the fixed GeneMap box+point prompt.",
                    "selection_used_annotation": False,
                }
            )
            continue
        he_mask = load_mask(Path(he_row["resolved_mask_path"]))
        shared = int(np.logical_and(ficture_mask, he_mask).sum())
        union = int(ficture_mask.sum()) + int(he_mask.sum()) - shared
        new_he = int(np.logical_and(he_mask, ~ficture_union).sum())
        he_new_fraction = new_he / max(1, int(he_mask.sum()))
        keep_he = he_new_fraction >= HE_NOVELTY_THRESHOLD
        pair_rows.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": ficture_row["prompt_id"],
                "gene_module": ficture_row["gene_module"],
                "ficture_area": int(ficture_mask.sum()),
                "he_area": int(he_mask.sum()),
                "shared_pixels": shared,
                "ficture_to_he_purity": shared / max(1, int(ficture_mask.sum())),
                "he_to_ficture_purity": shared / max(1, int(he_mask.sum())),
                "pair_iou": shared / max(1, union),
                "he_new_pixels_against_all_ficture": new_he,
                "he_new_fraction_against_all_ficture": he_new_fraction,
                "same_prompt_he_retained": keep_he,
                "same_prompt_he_threshold": HE_NOVELTY_THRESHOLD,
                "same_prompt_he_status": "eligible H&E mask evaluated by the 40% new-area rule",
                "selection_used_annotation": False,
            }
        )
        pre_dedup.append(
            {
                "candidate_id": f"{args.tma}:FICTURE:P{prompt_index + 1:03d}",
                "source": "FICTURE primary",
                "prompt_index": prompt_index,
                "prompt_id": ficture_row["prompt_id"],
                "gene_module": ficture_row["gene_module"],
                "source_mask_path": ficture_row["resolved_mask_path"],
                "sam_score": ficture_row["sam_score"],
                "selection_method": args.method,
                "selection_reason": "Primary candidate from the fixed GeneMap box+point prompt.",
                "selection_used_annotation": False,
            }
        )
        if keep_he:
            pre_dedup.append(
                {
                    "candidate_id": f"{args.tma}:HE_SAME:P{prompt_index + 1:03d}",
                    "source": "Same-prompt H&E supplement",
                    "prompt_index": prompt_index,
                    "prompt_id": he_row["prompt_id"],
                    "gene_module": he_row["gene_module"],
                    "source_mask_path": he_row["resolved_mask_path"],
                    "sam_score": he_row["sam_score"],
                    "selection_method": args.method,
                    "selection_reason": "At least 40% of this H&E mask lies outside the complete FICTURE-primary pool.",
                    "selection_used_annotation": False,
                }
            )

    gap_rows = read_csv(args.gap_root / "selected_candidates.csv")
    gap_rows.sort(key=lambda row: int(row["prompt_index"]))
    for row in gap_rows:
        region_id = row["region_id"]
        pre_dedup.append(
            {
                "candidate_id": f"{args.tma}:HE_GAP:{region_id}",
                "source": "Independent H&E supplement",
                "prompt_index": row["prompt_index"],
                "prompt_id": f"he_gap_{region_id}",
                "gene_module": "",
                "source_mask_path": row["selected_mask_path"],
                "sam_score": row["sam_score"],
                "selection_method": "point_constrained_sam_score",
                "selection_reason": "H&E tissue is present where registered FICTURE has little signal.",
                "selection_used_annotation": False,
            }
        )

    kept, dedup_rows = deduplicate(pre_dedup)
    for index, row in enumerate(kept, start=1):
        source_path = Path(row["source_mask_path"])
        output_name = f"candidate_{index:03d}_{row['candidate_id'].replace(':', '_')}.png"
        output_path = mask_dir / output_name
        shutil.copy2(source_path, output_path)
        row["mask_path"] = str(Path("masks") / output_name)
        row["mask_sha256"] = sha256(output_path)
        row["area_pixels"] = int(load_mask(output_path).sum())

    write_csv(args.output_dir / "same_prompt_he_audit.csv", pair_rows)
    write_csv(args.output_dir / "pre_dedup_pool.csv", pre_dedup)
    write_csv(args.output_dir / "dedup_audit.csv", dedup_rows)
    write_csv(args.output_dir / "final_pool.csv", kept)
    source_counts = {
        source: sum(row["source"] == source for row in kept)
        for source in (
            "FICTURE primary",
            "Same-prompt H&E supplement",
            "Independent H&E supplement",
        )
    }
    manifest = {
        "tma": args.tma,
        "main_selection_method": args.method,
        "main_prompt_count": prompt_count,
        "ficture_primary_count": len(ficture_rows),
        "ficture_missing_count": prompt_count - len(ficture_rows),
        "same_prompt_he_rule": "retain when at least 40% of H&E mask area lies outside the complete FICTURE-primary union",
        "same_prompt_he_threshold": HE_NOVELTY_THRESHOLD,
        "same_prompt_he_available_count": len(he_rows),
        "same_prompt_he_missing_count": prompt_count - len(he_rows),
        "independent_he_rule": "64-pixel cells; H&E tissue >=35%; FICTURE signal <=8%; at least 6 connected cells; 10% context box plus one automatic positive point",
        "independent_he_detected_count": len(gap_rows),
        "pre_dedup_count": len(pre_dedup),
        "dedup_iou_threshold": DEDUP_IOU_THRESHOLD,
        "removed_as_near_duplicate": len(pre_dedup) - len(kept),
        "final_candidate_count": len(kept),
        "final_source_counts": source_counts,
        "selection_used_annotation": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
