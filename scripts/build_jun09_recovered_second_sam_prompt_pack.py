#!/usr/bin/env python3
"""Build a self-contained recovered-piece prompt pack for Jun09 second-SAM."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
SRC = BASE / "runtime_policy_second_sam_locator_pack/runtime_policy_selected_pieces_for_second_sam.csv"
OUT = BASE / "recovered_second_sam_prompt_pack"
MASK_DIR = OUT / "piece_masks"
ROI_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
HE_ROI = ROI_ROOT / "he_roi_matching_official_ficture_coverage.png"

RECOVERED = [
    {
        "class": "bronchiola",
        "candidate_uid": "recovered_bronchiola_component3_medium_box142",
        "rank": "recovered",
        "target_score": "",
        "runtime_policy_threshold": "",
        "runtime_policy_score": "",
        "source": "medium_boxes_b192_s64 candidate_142; recovers missing bottom bronchiola component",
        "mask_source": ROOT / "output/visium_hd_exp1/best_single_candidate_vs_annotation/source_candidate_runs/medium_boxes_b192_s64/candidate_masks/candidate_142.png",
    },
    {
        "class": "vessels",
        "candidate_uid": "recovered_vessels_component4_medium_box116",
        "rank": "recovered",
        "target_score": "",
        "runtime_policy_threshold": "",
        "runtime_policy_score": "",
        "source": "medium_boxes_b192_s64 candidate_116; recovers vessel component 4 with high recall but low precision",
        "mask_source": ROOT / "output/visium_hd_exp1/best_single_candidate_vs_annotation/source_candidate_runs/medium_boxes_b192_s64/candidate_masks/candidate_116.png",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def copy_mask_to_pack(src: Path, dest_name: str, target_size: tuple[int, int]) -> str:
    dest = MASK_DIR / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("L")
    if im.size != target_size:
        im = im.resize(target_size, Image.Resampling.NEAREST)
    im.save(dest)
    return str(dest.relative_to(OUT))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    he_size = Image.open(HE_ROI).size
    src_base = SRC.parent
    rows: list[dict[str, object]] = []

    for row in read_csv(SRC):
        if row["class"] not in {"bronchiola", "vessels"}:
            continue
        mask_path = Path(row["mask_path"])
        if not mask_path.is_absolute():
            mask_path = src_base / mask_path
        dest_name = f"{row['candidate_uid']}.png"
        rel = copy_mask_to_pack(mask_path, dest_name, he_size)
        new_row = dict(row)
        new_row["mask_path"] = rel
        new_row["selector_source"] = row.get("selector_source", "runtime policy selected piece")
        rows.append(new_row)

    for rec in RECOVERED:
        rel = copy_mask_to_pack(Path(rec["mask_source"]), f"{rec['candidate_uid']}.png", he_size)
        rows.append(
            {
                "class": rec["class"],
                "rank": rec["rank"],
                "candidate_uid": rec["candidate_uid"],
                "target_score": rec["target_score"],
                "margin": "",
                "runtime_policy_threshold": rec["runtime_policy_threshold"],
                "runtime_policy_score": rec["runtime_policy_score"],
                "mask_path": rel,
                "selector_source": rec["source"],
            }
        )

    write_csv(OUT / "selected_pieces_for_second_sam.csv", rows)
    (OUT / "run_config.json").write_text(
        json.dumps(
            {
                "purpose": "Recovered-piece prompt pack for true second-SAM refinement.",
                "labels": ["bronchiola", "vessels"],
                "source_selected_csv": str(SRC),
                "recovered_pieces": [
                    {k: str(v) for k, v in rec.items() if k != "mask_source"} | {"mask_source": str(rec["mask_source"])}
                    for rec in RECOVERED
                ],
                "roi_size": list(he_size),
                "selected_piece_count": len(rows),
                "note": "Recovered masks are locators/prompts, not final masks.",
            },
            indent=2,
        )
    )
    shutil.copy2(SRC, OUT / "source_runtime_policy_selected_pieces_for_second_sam.csv")
    print(OUT / "selected_pieces_for_second_sam.csv")


if __name__ == "__main__":
    main()
