#!/usr/bin/env python3
"""Build an oracle component prompt pack for missed vessel components.

This is a diagnostic only: it uses the annotation component as the prompt mask
to ask whether SAM3 can segment the tiny vessel if the location is known.
If this succeeds, the missing layer is automatic proposal/location.  If this
fails, the issue is SAM/image resolution for that component.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
ANNOTATION = ROI_ROOT / "cropped_annotation_masks/05_lung_vessels_target_roi.png"
OUT = BASE / "vessels_component_oracle_prompt_diagnostic"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "oracle_component_masks"
    mask_dir.mkdir(exist_ok=True)

    ann = np.asarray(Image.open(ANNOTATION).convert("L")) > 127
    labels, n_comp = ndimage.label(ann, structure=ndimage.generate_binary_structure(2, 1))
    areas = np.bincount(labels.reshape(-1), minlength=n_comp + 1)
    component_id = args.component_id
    if component_id <= 0 or component_id > n_comp:
        raise SystemExit(f"component id {component_id} outside 1..{n_comp}")

    component = labels == component_id
    if not component.any():
        raise SystemExit(f"component id {component_id} is empty")
    out_mask = mask_dir / f"vessels_component_{component_id}_oracle_mask.png"
    Image.fromarray(component.astype(np.uint8) * 255).save(out_mask)

    ys, xs = np.where(component)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    selected = [
        {
            "class": "vessels",
            "candidate_uid": f"oracle_vessels_component_{component_id}",
            "rank": 1,
            "target_score": 1.0,
            "margin": 1.0,
            "mask_path": str(out_mask),
            "diagnostic_note": "annotation-component oracle prompt; not deployable",
        }
    ]
    metadata = [
        {
            "class": "vessels",
            "component_id": component_id,
            "component_pixels": int(component.sum()),
            "component_bbox_xyxy": bbox,
            "annotation_path": str(ANNOTATION),
            "oracle_mask_path": str(out_mask),
            "purpose": "test whether second-SAM can recover the tiny vessel if location is already known",
        }
    ]
    write_csv(args.output_dir / "selected_pieces_oracle_vessels_component.csv", selected)
    write_csv(args.output_dir / "oracle_component_metadata.csv", metadata)
    print(args.output_dir / "selected_pieces_oracle_vessels_component.csv")
    print(args.output_dir / "oracle_component_metadata.csv")


if __name__ == "__main__":
    main()
