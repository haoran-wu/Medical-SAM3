#!/usr/bin/env python3
"""Score hand-picked good/bad official FICTURE candidates with CLIP and VLM.

This is a small audit, not a final segmentation method. It picks a few GOOD,
MID, and BAD candidates from an existing hit-test truth table, computes fresh
CLIP scores for those exact masks, then joins already-produced VLM scores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from rank_multimodal_sam_candidates import (
    CLIP_TEXT_PROMPTS,
    Candidate,
    compute_clip_scores,
    load_semantic_legend,
    read_mask,
    resize_bool,
    semantic_clip_prompts,
)


FOCUS_LABELS = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]


def key(row: dict) -> Tuple[str, str, str, str, str]:
    return (row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"]))


def select_examples(rows: List[dict]) -> List[Tuple[str, dict]]:
    selected: List[Tuple[str, dict]] = []
    high = sorted(
        [row for row in rows if row["sample_bucket"] == "high_same_label"],
        key=lambda row: float(row["hidden_dice"]),
        reverse=True,
    )
    mid = sorted(
        [row for row in rows if row["sample_bucket"] == "mid_same_label"],
        key=lambda row: float(row["hidden_dice"]),
        reverse=True,
    )
    bad = sorted(
        [row for row in rows if row["sample_bucket"] == "bad_cross_label"],
        key=lambda row: float(row["hidden_dice"]),
    )
    if high:
        selected.append(("GOOD", high[0]))
    if mid:
        selected.append(("MID", mid[0]))
    if bad:
        selected.append(("BAD", bad[0]))
    return selected


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--vlm-csv", type=Path, required=True)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--factor-semantic-legend", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    args = parser.parse_args()

    truth_rows = list(csv.DictReader(args.truth_csv.open()))
    vlm_rows = {key(row): row for row in csv.DictReader(args.vlm_csv.open())}
    he = np.array(Image.open(args.he_image).convert("RGB"))
    ficture = np.array(Image.open(args.ficture_image).convert("RGB"))
    shape_hw = he.shape[:2]
    if ficture.shape[:2] != shape_hw:
        ficture = np.array(Image.fromarray(ficture).resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR))

    selected: List[Tuple[str, dict]] = []
    for label, _display in FOCUS_LABELS:
        selected.extend(select_examples([row for row in truth_rows if row["label"] == label]))

    candidates: List[Candidate] = []
    seen_paths: set[Path] = set()
    for _tag, row in selected:
        path = Path(row["mask_path"])
        if path in seen_paths:
            continue
        seen_paths.add(path)
        candidates.append(
            Candidate(
                source=row["source"],
                run=row["run"],
                setting=row["setting"],
                candidate_id=int(row["candidate_id"]),
                mask_path=path,
                report_path=Path("."),
                prompt_type="manual",
                area_reported=0,
            )
        )

    mask_cache: Dict[Path, np.ndarray] = {}

    def get_mask(path: Path) -> np.ndarray:
        if path not in mask_cache:
            mask_cache[path] = resize_bool(read_mask(path), shape_hw)
        return mask_cache[path]

    he_scores = compute_clip_scores(
        candidates,
        get_mask,
        he,
        model_name=args.clip_model,
        batch_size=args.clip_batch_size,
        device=args.clip_device,
        text_prompts_by_label=CLIP_TEXT_PROMPTS,
    )
    semantic_prompts = CLIP_TEXT_PROMPTS
    if args.factor_semantic_legend and args.factor_semantic_legend.exists():
        semantic = load_semantic_legend(args.factor_semantic_legend, n_factors=12)
        semantic_prompts = semantic_clip_prompts(CLIP_TEXT_PROMPTS, semantic)
    ficture_scores = compute_clip_scores(
        candidates,
        get_mask,
        ficture,
        model_name=args.clip_model,
        batch_size=args.clip_batch_size,
        device=args.clip_device,
        text_prompts_by_label=semantic_prompts,
    )

    rows_out: List[dict] = []
    for tag, row in selected:
        display = dict(FOCUS_LABELS)[row["label"]]
        vlm = vlm_rows.get(key(row), {})
        mask_path = Path(row["mask_path"])
        label = row["label"]
        clip_he = he_scores.get(mask_path, {}).get(label)
        clip_ficture = ficture_scores.get(mask_path, {}).get(label)
        clip_values = [v for v in [clip_he, clip_ficture] if v is not None]
        rows_out.append(
            {
                "label": display,
                "label_slug": label,
                "manual_pick": tag,
                "sample_bucket": row["sample_bucket"],
                "candidate": f"{row['source']}/{row['setting']}/{row['candidate_id']}",
                "original_best_for": row["candidate_original_best_label"],
                "true_dice": float(row["hidden_dice"]),
                "true_precision": float(row["hidden_precision"]),
                "true_recall": float(row["hidden_recall"]),
                "clip_he_score": clip_he,
                "clip_ficture_semantic_score": clip_ficture,
                "clip_mean_score": sum(clip_values) / len(clip_values) if clip_values else None,
                "vlm_score": float(vlm.get("vlm_score") or 0.0),
                "vlm_fused_score": float(vlm.get("vlm_fused_score") or 0.0),
                "vlm_reason": (vlm.get("reason") or "")[:220],
                "mask_path": str(mask_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "manual_good_bad_clip_vlm_fresh_clip.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    lines = [
        "# Manual GOOD/MID/BAD Candidate CLIP vs VLM",
        "",
        "Fresh CLIP scores were computed for the exact selected masks. VLM scores are from the matching Qwen2.5-VL-7B hit-test run.",
        "",
        "| label | pick | true Dice | P | R | CLIP H&E | CLIP FICTURE semantic | CLIP mean | VLM score | VLM fused | candidate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows_out:
        lines.append(
            f"| {row['label']} | {row['manual_pick']} | {row['true_dice']:.3f} | {row['true_precision']:.3f} | {row['true_recall']:.3f} | "
            f"{fmt(row['clip_he_score'])} | {fmt(row['clip_ficture_semantic_score'])} | {fmt(row['clip_mean_score'])} | "
            f"{row['vlm_score']:.3f} | {row['vlm_fused_score']:.3f} | `{row['candidate']}` |"
        )
    readme_path = args.output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    print(csv_path)
    print(readme_path)


if __name__ == "__main__":
    main()
