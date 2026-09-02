#!/usr/bin/env python3
"""Run one annotation-free box-plus-point rule for TMA39 H&E gap regions."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.sam3_inference import SAM3Model, normalize_bbox, normalize_points
from sam3.model.box_ops import box_xywh_to_cxcywh


RUN = (
    ROOT
    / "output/aaai_xenium_silica_20260623/TMA39_meeting_followup_20260813"
    / "genemap_ficture_primary_structure_20260819"
    / "independent_he_raw_gap_reprompt_20260819"
)
OUTPUT = RUN / "uniform_box_point_20260821"
CHECKPOINT = ROOT / "checkpoints/SAM3-base/sam3.pt"
REGION_IDS = ("R2", "R3", "R7")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def contour(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)
    color = np.asarray((0, 188, 220), dtype=np.float32)
    out[mask] = 0.62 * out[mask] + 0.38 * color
    out[contour(mask)] = (0, 45, 65)
    return np.clip(out, 0, 255).astype(np.uint8)


def threshold_stability(logits: np.ndarray, offset: float) -> float:
    low = logits > -offset
    high = logits > offset
    denominator = int(low.sum())
    return float(high.sum() / denominator) if denominator else 0.0


def run_prompt(
    sam3: SAM3Model,
    inference_state: dict,
    box: tuple[int, int, int, int],
    point: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[int, list[dict[str, object]]]:
    sam3.processor.reset_all_prompts(inference_state)
    if "language_features" not in inference_state["backbone_out"]:
        text_outputs = sam3.processor.model.backbone.forward_text(
            ["visual"], device=sam3.device
        )
        inference_state["backbone_out"].update(text_outputs)
    if "geometric_prompt" not in inference_state:
        inference_state["geometric_prompt"] = sam3.processor.model._get_dummy_prompt()

    image_h, image_w = image_shape
    x1, y1, x2, y2 = box
    box_xywh = torch.tensor(
        [x1, y1, x2 - x1, y2 - y1], dtype=torch.float32
    ).view(1, 4)
    normalized_box = normalize_bbox(
        box_xywh_to_cxcywh(box_xywh), image_w, image_h
    ).to(device=sam3.device, dtype=torch.float32).view(1, 1, 4)
    inference_state["geometric_prompt"].append_boxes(
        normalized_box,
        torch.ones((1, 1), device=sam3.device, dtype=torch.bool),
    )

    normalized_point = normalize_points([point], image_w, image_h)
    inference_state["geometric_prompt"].append_points(
        torch.tensor(
            normalized_point, device=sam3.device, dtype=torch.float32
        ).view(1, 1, 2),
        torch.ones((1, 1), device=sam3.device, dtype=torch.long),
    )

    state = sam3.processor._forward_grounding(inference_state)
    raw_masks = state.get("masks")
    raw_scores = state.get("scores")
    if raw_masks is None or raw_scores is None:
        return 0, []

    px, py = point
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for model_index in range(len(raw_masks)):
        logits_tensor = raw_masks[model_index]
        while logits_tensor.ndim > 2 and logits_tensor.shape[0] == 1:
            logits_tensor = logits_tensor[0]
        if logits_tensor.ndim != 2:
            raise ValueError(f"Unexpected SAM mask shape: {tuple(logits_tensor.shape)}")
        logits = logits_tensor.detach().float().cpu().numpy()
        mask = logits > 0
        if not (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]):
            continue
        digest = hashlib.sha1(np.packbits(mask, axis=None).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        score = float(raw_scores[model_index].detach().float().cpu().reshape(-1)[0])
        results.append(
            {
                "model_index": int(model_index),
                "sam_score": score,
                "area_pixels": int(mask.sum()),
                "threshold_stability_offset0p5": threshold_stability(logits, 0.5),
                "threshold_stability_offset1p0": threshold_stability(logits, 1.0),
                "mask": mask,
            }
        )
    results.sort(key=lambda row: float(row["sam_score"]), reverse=True)
    return int(len(raw_masks)), results


def draw_prompt_figure(
    he: np.ndarray,
    ficture: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    ficture_display = ficture.copy()
    ficture_display[np.max(ficture_display, axis=2) < 8] = (226, 231, 233)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (he, ficture_display),
        ("Unified prompts on registered H&E", "The same prompts on registered FICTURE"),
    ):
        axis.imshow(image)
        for row in rows:
            x1, y1, x2, y2 = row["box_full"]
            px, py = row["point_full"]
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    ec="#00877A",
                    lw=1.8,
                )
            )
            axis.scatter(
                [px],
                [py],
                s=42,
                facecolor="#111111",
                edgecolor="white",
                linewidth=0.9,
                zorder=6,
            )
            axis.text(
                x1 + 6,
                y1 + 28,
                str(row["region_id"]),
                fontsize=9,
                color="white",
                weight="bold",
                bbox=dict(facecolor="#00877A", edgecolor="none", pad=2),
            )
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_axis_off()
    fig.suptitle(
        "Every automatic H&E-present / FICTURE-low region uses the same box + point rule",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(
        OUTPUT / "TMA39_Uniform_HE_Gap_Prompts_G_Figure.png",
        dpi=190,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def draw_selected_masks(he: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    union = np.logical_or.reduce(list(masks.values()))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.8), constrained_layout=True)
    axes[0, 0].imshow(overlay(he, union))
    axes[0, 0].set_title("All three selected H&E supplements", weight="bold")
    for axis, region_id in zip(axes.flat[1:], REGION_IDS):
        mask = masks[region_id]
        yy, xx = np.where(mask)
        pad = 180
        x0, x1 = max(0, int(xx.min()) - pad), min(he.shape[1], int(xx.max()) + pad)
        y0, y1 = max(0, int(yy.min()) - pad), min(he.shape[0], int(yy.max()) + pad)
        axis.imshow(overlay(he[y0:y1, x0:x1], mask[y0:y1, x0:x1]))
        axis.set_title(f"{region_id}: automatic box + automatic point", weight="bold")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.savefig(
        OUTPUT / "TMA39_Uniform_HE_Gap_Candidates_M_Figure.png",
        dpi=190,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_mask_dir = OUTPUT / "all_prompt_consistent_masks"
    selected_mask_dir = OUTPUT / "selected_masks"
    all_mask_dir.mkdir(exist_ok=True)
    selected_mask_dir.mkdir(exist_ok=True)

    region_manifest = {
        item["region_id"]: item
        for item in json.loads((RUN / "region_crops/manifest.json").read_text())
    }
    detected = {row["region_id"]: row for row in read_csv(RUN / "detected_regions.csv")}
    prompt_manifest = json.loads((RUN / "prompt_manifest.json").read_text())
    he_full = load_rgb(Path(prompt_manifest["he_path"]))
    ficture_full = load_rgb(Path(prompt_manifest["ficture_path"]))

    sam3 = SAM3Model(
        confidence_threshold=0.1,
        checkpoint_path=str(CHECKPOINT),
        device="cpu",
    )
    rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    selected_masks: dict[str, np.ndarray] = {}
    full_shape = tuple(int(v) for v in prompt_manifest["canvas_height_width"])
    for region_id in REGION_IDS:
        item = region_manifest[region_id]
        crop_offset = tuple(int(v) for v in item["crop_xyxy_full_canvas"])
        box_crop = tuple(int(v) for v in item["prompt_box_xyxy_crop"])
        point_crop = tuple(int(v) for v in item["prompt_point_xy_crop"])
        x0, y0, x1, y1 = crop_offset
        point_full = (x0 + point_crop[0], y0 + point_crop[1])
        box_full = (x0 + box_crop[0], y0 + box_crop[1], x0 + box_crop[2], y0 + box_crop[3])
        prompt_rows.append(
            {
                "region_id": region_id,
                "box_full": box_full,
                "point_full": point_full,
            }
        )
        he_crop = load_rgb(RUN / f"region_crops/{region_id}/he_crop.png")
        inference_state = sam3.encode_image(he_crop)
        model_count, results = run_prompt(
            sam3, inference_state, box_crop, point_crop, he_crop.shape[:2]
        )
        del inference_state
        gc.collect()
        if not results:
            raise RuntimeError(f"{region_id} produced no mask containing its automatic point")
        for rank, result in enumerate(results, start=1):
            mask_path = all_mask_dir / f"{region_id}_result_{rank:02d}.png"
            Image.fromarray(np.uint8(result["mask"]) * 255).save(mask_path, optimize=True)
            rows.append(
                {
                    "region_id": region_id,
                    "prompt_type": "box_plus_one_automatic_positive_point",
                    "box_x1_crop": box_crop[0],
                    "box_y1_crop": box_crop[1],
                    "box_x2_crop": box_crop[2],
                    "box_y2_crop": box_crop[3],
                    "point_x_crop": point_crop[0],
                    "point_y_crop": point_crop[1],
                    "point_rule": "interior tissue plus automatic region-center preference",
                    "model_mask_count": model_count,
                    "prompt_consistent_unique_mask_count": len(results),
                    "rank_by_sam_score": rank,
                    "model_index": result["model_index"],
                    "sam_score": result["sam_score"],
                    "area_pixels": result["area_pixels"],
                    "threshold_stability_offset0p5": result["threshold_stability_offset0p5"],
                    "threshold_stability_offset1p0": result["threshold_stability_offset1p0"],
                    "selected_by_blind_rule": rank == 1,
                    "selection_used_annotation": False,
                    "mask_path": str(mask_path),
                }
            )
        selected_crop = np.asarray(results[0]["mask"], dtype=bool)
        selected_full = np.zeros(full_shape, dtype=bool)
        selected_full[y0:y1, x0:x1] = selected_crop
        selected_masks[region_id] = selected_full
        Image.fromarray(np.uint8(selected_full) * 255).save(
            selected_mask_dir / f"TMA39_raw_he_gap_{region_id}_uniform_box_point.png",
            optimize=True,
        )

    del sam3
    gc.collect()
    write_csv(OUTPUT / "uniform_box_point_all_results.csv", rows)
    draw_prompt_figure(he_full, ficture_full, prompt_rows)
    draw_selected_masks(he_full, selected_masks)
    summary = {
        "purpose": "Use one reusable prompt rule for all automatic H&E-present / FICTURE-low regions.",
        "region_ids": list(REGION_IDS),
        "prompt_type_for_every_region": "adaptive box plus one automatic positive point",
        "box_rule": "10 percent expansion around the automatically detected connected region",
        "point_rule": "inside H&E tissue, away from its edge, and close to the automatically detected region center",
        "candidate_rule": "highest SAM score among unique masks that contain the automatic positive point",
        "selection_used_annotation": False,
        "selected_candidate_count": len(selected_masks),
        "selected_masks": {
            region_id: str(
                selected_mask_dir
                / f"TMA39_raw_he_gap_{region_id}_uniform_box_point.png"
            )
            for region_id in REGION_IDS
        },
    }
    (OUTPUT / "frozen_before_annotation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
