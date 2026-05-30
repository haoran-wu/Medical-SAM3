#!/usr/bin/env python3
"""Assemble disconnected tissue masks from component-level candidate rankings.

This is the second step after `componentwise_candidate_oracle.py`.
The oracle scan tells us which candidates can cover each disconnected GT
component. This script tests a practical assembly rule: keep only reliable
component candidates, avoid redundant overlaps, then union the selected masks.

The current input ranking can be oracle-derived for method development. The
same selector can later consume VLM / FICTURE ranked candidates as long as the
CSV contains class, source, component, rank, score columns, and mask_path.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


LABEL_TO_MASK = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
}

POLICY_PRESETS = {
    "precision_focused": {
        "min_precision": 0.70,
        "min_component_dice": 0.50,
        "min_recall": 0.0,
        "max_overlap": 0.40,
    },
    "balanced": {
        "min_precision": 0.50,
        "min_component_dice": 0.45,
        "min_recall": 0.0,
        "max_overlap": 0.50,
    },
    "recall_oracle_all": {
        "min_precision": 0.0,
        "min_component_dice": 0.0,
        "min_recall": 0.0,
        "max_overlap": 1.00,
    },
}


@dataclass(frozen=True)
class SelectionPolicy:
    name: str
    min_precision: float
    min_component_dice: float
    min_recall: float
    max_overlap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-candidates",
        type=Path,
        required=True,
        help="CSV from componentwise_candidate_oracle.py.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
            "cropped_annotation_masks"
        ),
    )
    parser.add_argument(
        "--he-image",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
            "he_roi_matching_official_ficture_coverage.png"
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["bronchiola", "vessels"],
        choices=sorted(LABEL_TO_MASK),
    )
    parser.add_argument("--sources", nargs="+", default=["HE", "FICTURE"])
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help=(
            "Policy preset name or custom name:min_precision:min_component_dice:"
            "min_recall:max_overlap. May be used multiple times."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def parse_policies(entries: list[str]) -> list[SelectionPolicy]:
    if not entries:
        entries = ["precision_focused", "balanced", "recall_oracle_all"]
    policies: list[SelectionPolicy] = []
    for entry in entries:
        if entry in POLICY_PRESETS:
            params = POLICY_PRESETS[entry]
            policies.append(SelectionPolicy(entry, **params))
            continue
        parts = entry.split(":")
        if len(parts) != 5:
            raise ValueError(
                "--policy must be a preset or name:min_precision:min_component_dice:min_recall:max_overlap"
            )
        name, min_precision, min_component_dice, min_recall, max_overlap = parts
        policies.append(
            SelectionPolicy(
                name=name,
                min_precision=float(min_precision),
                min_component_dice=float(min_component_dice),
                min_recall=float(min_recall),
                max_overlap=float(max_overlap),
            )
        )
    return policies


def load_binary_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if expected_size is not None and image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "pred_area": float(pred_area),
        "gt_area": float(gt_area),
    }


def candidate_overlap(candidate: np.ndarray, selected_union: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 0.0
    return float(np.logical_and(candidate, selected_union).sum() / area)


def union_overlay(base: Image.Image, gt: np.ndarray, pred: np.ndarray, title: str) -> Image.Image:
    base = base.convert("RGBA")
    overlay = np.zeros((*gt.shape, 4), dtype=np.uint8)
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    overlay[tp] = (0, 180, 80, 135)
    overlay[fp] = (0, 90, 255, 135)
    overlay[fn] = (255, 0, 0, 150)
    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, min(1180, out.width), 36), fill=(255, 255, 255, 225))
    draw.text(
        (10, 10),
        title + " | green=correct, blue=extra candidate, red=missed annotation",
        fill=(0, 0, 0, 255),
        font=font,
    )
    return out.convert("RGB")


def write_selection_sheet(
    he_image: Image.Image,
    gt: np.ndarray,
    selected_masks: list[tuple[str, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    union_mask = np.zeros_like(gt, dtype=bool)
    for label, mask in selected_masks:
        union_mask |= mask
        thumbs.append((label, union_overlay(he_image, gt, mask, label)))
    thumbs.append(("final union", union_overlay(he_image, gt, union_mask, title)))

    thumb_width = 680
    margin = 24
    title_h = 34
    rendered: list[tuple[str, Image.Image]] = []
    for label, image in thumbs:
        scale = thumb_width / image.width
        rendered.append(
            (label, image.resize((thumb_width, int(image.height * scale)), Image.Resampling.BICUBIC))
        )
    width = thumb_width + 2 * margin
    height = margin + sum(title_h + image.height + margin for _, image in rendered)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = margin
    for label, image in rendered:
        draw.text((margin, y), label, fill=(0, 0, 0), font=font)
        y += title_h
        sheet.paste(image, (margin, y))
        y += image.height + margin
    sheet.save(out_path)


def select_for_group(
    group: pd.DataFrame,
    policy: SelectionPolicy,
    expected_size: tuple[int, int],
) -> tuple[list[dict[str, object]], np.ndarray]:
    selected_rows: list[dict[str, object]] = []
    selected_union = np.zeros((expected_size[1], expected_size[0]), dtype=bool)

    component_order = sorted(
        group["component"].unique(),
        key=lambda item: int(str(item).replace("C", "")),
    )
    for component in component_order:
        rows = group[group["component"] == component].sort_values("rank")
        accepted: dict[str, object] | None = None
        for _, row in rows.iterrows():
            dice = float(row["dice_vs_component"])
            precision = float(row["precision_vs_component"])
            recall = float(row["recall_vs_component"])
            if precision < policy.min_precision:
                continue
            if dice < policy.min_component_dice:
                continue
            if recall < policy.min_recall:
                continue
            mask = load_binary_mask(Path(row["mask_path"]), expected_size=expected_size)
            overlap = candidate_overlap(mask, selected_union)
            if overlap > policy.max_overlap:
                continue
            selected_union |= mask
            accepted = {
                "component": component,
                "rank": int(row["rank"]),
                "dice_vs_component": dice,
                "precision_vs_component": precision,
                "recall_vs_component": recall,
                "overlap_with_selected": overlap,
                "setting": row["setting"],
                "candidate": row["candidate"],
                "mask_path": row["mask_path"],
                "_mask": mask,
            }
            break
        if accepted is None:
            accepted = {
                "component": component,
                "rank": "",
                "dice_vs_component": "",
                "precision_vs_component": "",
                "recall_vs_component": "",
                "overlap_with_selected": "",
                "setting": "SKIPPED_BY_GATE",
                "candidate": "",
                "mask_path": "",
                "_mask": None,
            }
        selected_rows.append(accepted)
    return selected_rows, selected_union


def simple_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    visible_keys = [key for key in rows[0].keys() if not key.startswith("_")]
    head = "".join(f"<th>{html.escape(str(key))}</th>" for key in visible_keys)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in visible_keys)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(
    out_dir: Path,
    summary_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
    figures: list[Path],
) -> None:
    cards = []
    for path in figures:
        rel = path.relative_to(out_dir)
        cards.append(
            f"<figure><img src='{html.escape(str(rel))}'><figcaption>{html.escape(path.name)}</figcaption></figure>"
        )
    html_text = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Component-wise candidate assembly</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px; color: #1f2933; }}
h1 {{ margin-bottom: 8px; }}
p {{ line-height: 1.55; max-width: 1000px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px 10px; vertical-align: top; }}
th {{ background: #f7f7f8; }}
.note {{ background: #f3f4f6; border-left: 4px solid #2563eb; padding: 12px 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 22px; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
figcaption {{ font-size: 12px; color: #4b5563; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Component-wise candidate assembly</h1>
<p class="note">这个实验把 disconnected annotation 当成多个小组件处理：先用 component-level candidate ranking 找每一块的候选，再通过 Precision / Dice gate 过滤低可信候选，最后把通过 gate 的候选 mask union 成最终预测。当前 ranking 是 oracle-derived，用来验证这个拼接策略的上限和阈值选择；同一个 selector 后续可以接 VLM 或 FICTURE score。</p>

<h2>Assembly summary</h2>
{simple_table(summary_rows)}

<h2>Selected candidates</h2>
{simple_table(selection_rows)}

<h2>Figures</h2>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policies = parse_policies(args.policy)
    he_image = Image.open(args.he_image).convert("RGB")
    expected_size = he_image.size
    candidates = pd.read_csv(args.component_candidates)

    summary_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    figures: list[Path] = []

    for class_name in args.classes:
        gt_path = args.annotation_dir / LABEL_TO_MASK[class_name]
        gt = load_binary_mask(gt_path, expected_size=expected_size)
        class_dir = args.out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for source in args.sources:
            group = candidates[(candidates["class"] == class_name) & (candidates["source"] == source)]
            if group.empty:
                continue
            for policy in policies:
                selected, union_mask = select_for_group(group, policy, expected_size)
                m = metrics(union_mask, gt)
                selected_kept = [row for row in selected if row.get("_mask") is not None]
                summary_rows.append(
                    {
                        "class": class_name,
                        "source": source,
                        "policy": policy.name,
                        "Dice": f"{m['dice']:.4f}",
                        "Precision": f"{m['precision']:.4f}",
                        "Recall": f"{m['recall']:.4f}",
                        "selected_components": f"{len(selected_kept)}/{len(selected)}",
                        "min_precision": policy.min_precision,
                        "min_component_dice": policy.min_component_dice,
                        "max_overlap": policy.max_overlap,
                    }
                )
                for row in selected:
                    clean = {key: value for key, value in row.items() if key != "_mask"}
                    clean.update({"class": class_name, "source": source, "policy": policy.name})
                    selection_rows.append(clean)

                union_path = class_dir / f"{class_name}_{source}_{policy.name}_union_mask.png"
                Image.fromarray((union_mask.astype(np.uint8) * 255), mode="L").save(union_path)
                overlay_path = class_dir / f"{class_name}_{source}_{policy.name}_union_on_he.png"
                union_overlay(
                    he_image,
                    gt,
                    union_mask,
                    f"{class_name} {source} {policy.name}: Dice {m['dice']:.3f}, Precision {m['precision']:.3f}, Recall {m['recall']:.3f}",
                ).save(overlay_path)
                figures.append(overlay_path)

                selected_masks = [
                    (
                        f"{class_name} {source} {policy.name} {row['component']} "
                        f"P={row['precision_vs_component']} R={row['recall_vs_component']}",
                        row["_mask"],
                    )
                    for row in selected
                    if row.get("_mask") is not None
                ]
                sheet_path = class_dir / f"{class_name}_{source}_{policy.name}_selection_sheet.png"
                write_selection_sheet(
                    he_image,
                    gt,
                    selected_masks,
                    sheet_path,
                    f"{class_name} {source} {policy.name} final union",
                )
                figures.append(sheet_path)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "assembly_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.out_dir / "selected_candidates.csv", index=False)
    metadata = {
        "component_candidates": str(args.component_candidates),
        "annotation_dir": str(args.annotation_dir),
        "he_image": str(args.he_image),
        "classes": args.classes,
        "sources": args.sources,
        "policies": [policy.__dict__ for policy in policies],
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    write_html_report(args.out_dir, summary_rows, selection_rows, figures)
    print(args.out_dir)


if __name__ == "__main__":
    main()
