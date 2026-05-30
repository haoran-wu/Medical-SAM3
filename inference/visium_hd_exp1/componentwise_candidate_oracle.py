#!/usr/bin/env python3
"""Component-wise candidate oracle for multi-piece tissue annotations.

This script addresses cases where a class annotation contains several
disconnected objects, but the best single SAM candidate only captures one of
them. It can:

1. Split each target annotation into connected components.
2. Render component maps on the official H&E and FICTURE ROI.
3. Optionally scan candidate mask roots and choose the best mask per component.
4. Union the selected masks and evaluate the merged mask against the full GT.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


LABEL_TO_MASK = {
    "lung_bronchiola": "01_lung_bronchiola_target_roi.png",
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "lung_alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "lung_vessels": "05_lung_vessels_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
    "immune infiltration": "03_immune_infiltration_target_roi.png",
}

DISPLAY_LABEL = {
    "lung_bronchiola": "bronchiola",
    "bronchiola": "bronchiola",
    "lung_alveoli": "alveoli",
    "alveoli": "alveoli",
    "lung_vessels": "vessels",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
    "immune infiltration": "immune infiltration",
}

COMPONENT_COLORS = [
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
    (240, 228, 66),
    (128, 0, 128),
]


@dataclass(frozen=True)
class Component:
    component_id: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class CandidateMask:
    source: str
    path: Path
    setting: str
    candidate_id: str
    mask: np.ndarray
    resized_from: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--ficture-image",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
            "ficture_official_filtered_roi_rgb.png"
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["lung_bronchiola", "lung_vessels"],
        help="Labels to process. Currently mapped to official ROI mask filenames.",
    )
    parser.add_argument(
        "--candidate-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional candidate root to scan recursively for candidate_masks/*.png. "
            "Use multiple times for HE and FICTURE pools."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/final_deliverables/"
            "May28_componentwise_bronchiola_vessels"
        ),
    )
    parser.add_argument("--min-component-area", type=int, default=256)
    parser.add_argument(
        "--important-component-area-fraction",
        type=float,
        default=0.0,
        help=(
            "Only score components covering at least this fraction of the GT. "
            "Default 0 keeps the historical behavior of scoring every kept component."
        ),
    )
    parser.add_argument(
        "--max-components-per-label",
        type=int,
        default=0,
        help="Optional cap on scored components after sorting by area. 0 means no cap.",
    )
    parser.add_argument("--top-k-per-component", type=int, default=5)
    parser.add_argument(
        "--max-candidates-per-root",
        type=int,
        default=0,
        help="Debug cap. 0 means no cap.",
    )
    return parser.parse_args()


def load_binary_mask(path: Path, expected_size: tuple[int, int] | None = None) -> tuple[np.ndarray, str]:
    image = Image.open(path).convert("L")
    resized_from = ""
    if expected_size is not None and image.size != expected_size:
        resized_from = f"{image.size[0]}x{image.size[1]}"
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    arr = np.array(image) > 0
    return arr, resized_from


def split_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[Component]]:
    labeled, n_components = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndi.find_objects(labeled)
    components: list[Component] = []
    kept = np.zeros_like(labeled, dtype=np.int32)
    next_id = 1
    for original_id, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        component_mask = labeled[sl] == original_id
        area = int(component_mask.sum())
        if area < min_area:
            continue
        y0, y1 = int(sl[0].start), int(sl[0].stop)
        x0, x1 = int(sl[1].start), int(sl[1].stop)
        kept[sl][component_mask] = next_id
        components.append(Component(next_id, area, x0, y0, x1, y1))
        next_id += 1
    components.sort(key=lambda c: c.area, reverse=True)
    id_remap = {component.component_id: idx + 1 for idx, component in enumerate(components)}
    remapped = np.zeros_like(kept)
    remapped_components: list[Component] = []
    for component in components:
        new_id = id_remap[component.component_id]
        remapped[kept == component.component_id] = new_id
        remapped_components.append(
            Component(new_id, component.area, component.x0, component.y0, component.x1, component.y1)
        )
    return remapped, remapped_components


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


def metrics_from_counts(tp: int, pred_area: int, gt_area: int) -> dict[str, float]:
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


def candidate_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("candidate_"):
        return stem.replace("candidate_", "")
    return stem


def parse_candidate_roots(entries: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--candidate-root must be NAME=PATH, got: {entry}")
        name, value = entry.split("=", 1)
        root = Path(value)
        if not root.exists():
            raise FileNotFoundError(root)
        parsed.append((name, root))
    return parsed


def iter_candidate_masks(
    source: str,
    root: Path,
    expected_size: tuple[int, int],
    max_candidates: int,
) -> Iterable[CandidateMask]:
    paths = sorted(root.rglob("candidate_masks/candidate_*.png"))
    if max_candidates > 0:
        paths = paths[:max_candidates]
    for path in paths:
        mask, resized_from = load_binary_mask(path, expected_size=expected_size)
        setting = path.parent.parent.name if path.parent.name == "candidate_masks" else path.parent.name
        yield CandidateMask(
            source=source,
            path=path,
            setting=setting,
            candidate_id=candidate_id_from_path(path),
            mask=mask,
            resized_from=resized_from,
        )


def composite_overlay(
    base: Image.Image,
    component_map: np.ndarray,
    components: list[Component],
    title: str,
    alpha: float = 0.42,
) -> Image.Image:
    base = base.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_px = np.array(overlay)
    for component in components:
        color = COMPONENT_COLORS[(component.component_id - 1) % len(COMPONENT_COLORS)]
        mask = component_map == component.component_id
        overlay_px[mask] = (*color, int(255 * alpha))
    overlay = Image.fromarray(overlay_px, mode="RGBA")
    out = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, min(760, out.width), 30), fill=(255, 255, 255, 220))
    draw.text((10, 9), title, fill=(0, 0, 0, 255), font=font)
    for component in components:
        color = COMPONENT_COLORS[(component.component_id - 1) % len(COMPONENT_COLORS)]
        draw.rectangle(
            (component.x0, component.y0, component.x1, component.y1),
            outline=(*color, 255),
            width=6,
        )
        label = f"C{component.component_id} area={component.area}"
        tx = max(0, component.x0)
        ty = max(32, component.y0 - 20)
        draw.rectangle((tx, ty - 2, tx + 8 * len(label), ty + 14), fill=(255, 255, 255, 210))
        draw.text((tx + 3, ty), label, fill=(*color, 255), font=font)
    return out.convert("RGB")


def component_only_image(component_map: np.ndarray, components: list[Component], title: str) -> Image.Image:
    rgb = np.full((*component_map.shape, 3), 255, dtype=np.uint8)
    for component in components:
        color = COMPONENT_COLORS[(component.component_id - 1) % len(COMPONENT_COLORS)]
        rgb[component_map == component.component_id] = color
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, min(760, image.width), 30), fill=(255, 255, 255))
    draw.text((10, 9), title, fill=(0, 0, 0), font=font)
    for component in components:
        color = COMPONENT_COLORS[(component.component_id - 1) % len(COMPONENT_COLORS)]
        draw.rectangle((component.x0, component.y0, component.x1, component.y1), outline=color, width=6)
        draw.text((component.x0 + 4, max(32, component.y0 + 4)), f"C{component.component_id}", fill=color, font=font)
    return image


def union_overlay(base: Image.Image, gt: np.ndarray, pred: np.ndarray, title: str) -> Image.Image:
    base = base.convert("RGBA")
    overlay = np.zeros((*gt.shape, 4), dtype=np.uint8)
    tp = np.logical_and(gt, pred)
    fp = np.logical_and(~gt, pred)
    fn = np.logical_and(gt, ~pred)
    overlay[tp] = (0, 180, 80, 125)
    overlay[fp] = (0, 90, 255, 125)
    overlay[fn] = (255, 0, 0, 150)
    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(900, out.width), 30), fill=(255, 255, 255, 220))
    draw.text((10, 9), title + " | green=TP, blue=candidate-only, red=missed GT", fill=(0, 0, 0, 255))
    return out.convert("RGB")


def make_contact_sheet(images: list[tuple[str, Path]], out_path: Path, thumb_width: int = 720) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for title, path in images:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumb = image.resize((thumb_width, int(image.height * scale)), Image.Resampling.BICUBIC)
        thumbs.append((title, thumb))
    if not thumbs:
        return
    margin = 24
    title_h = 32
    width = thumb_width + 2 * margin
    height = sum(img.height + title_h + margin for _, img in thumbs) + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = margin
    for title, img in thumbs:
        draw.text((margin, y), title, fill=(0, 0, 0), font=font)
        y += title_h
        sheet.paste(img, (margin, y))
        y += img.height + margin
    sheet.save(out_path)


def write_html_report(
    out_dir: Path,
    component_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    union_rows: list[dict[str, object]],
    generated_images: dict[str, list[Path]],
) -> None:
    def table(rows: list[dict[str, object]]) -> str:
        if not rows:
            return "<p>未运行 candidate pool 扫描；这里只展示 GT component 拆分。</p>"
        keys = list(rows[0].keys())
        head = "".join(f"<th>{html.escape(str(k))}</th>" for k in keys)
        body = []
        for row in rows:
            cells = "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in keys)
            body.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    image_sections = []
    for label, paths in generated_images.items():
        cards = []
        for path in paths:
            rel = path.relative_to(out_dir)
            cards.append(
                f"<figure><img src='{html.escape(str(rel))}'><figcaption>{html.escape(path.name)}</figcaption></figure>"
            )
        image_sections.append(f"<h2>{html.escape(label)}</h2><div class='grid'>{''.join(cards)}</div>")

    html_text = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Component-wise candidate oracle</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px; color: #1f2933; }}
h1 {{ font-size: 30px; margin-bottom: 6px; }}
h2 {{ margin-top: 32px; }}
p {{ line-height: 1.55; max-width: 980px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px 10px; }}
th {{ background: #f7f7f8; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 22px; }}
figure {{ margin: 0; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
figcaption {{ font-size: 12px; color: #4b5563; margin-top: 4px; }}
.note {{ background: #f3f4f6; border-left: 4px solid #2563eb; padding: 12px 16px; }}
</style>
</head>
<body>
<h1>Component-wise candidate oracle check</h1>
<p class="note">问题：bronchiola / vessels 的 annotation 不是一个连续区域，而是多个 disconnected components。单个 best candidate 往往只覆盖最大的一块，所以 recall 被卡住。这个报告先把 GT 拆成 components；如果提供 full candidate masks，还会对每个 component 单独找 best mask，再 union 起来重新算 Dice / Precision / Recall。</p>

<h2>GT component summary</h2>
{table(component_rows)}

<h2>Component-wise candidate union summary</h2>
{table(union_rows)}

<h2>Top candidates per component</h2>
{table(candidate_rows)}

{''.join(image_sections)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    he_image = Image.open(args.he_image).convert("RGB")
    ficture_image = Image.open(args.ficture_image).convert("RGB")
    expected_size = he_image.size
    if ficture_image.size != expected_size:
        raise ValueError(f"H&E and FICTURE sizes differ: {he_image.size} vs {ficture_image.size}")

    component_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    union_rows: list[dict[str, object]] = []
    generated_images: dict[str, list[Path]] = {}

    candidate_roots = parse_candidate_roots(args.candidate_root)

    for label in args.labels:
        if label not in LABEL_TO_MASK:
            raise KeyError(f"No mask mapping for label: {label}")
        display = DISPLAY_LABEL[label]
        gt_path = args.annotation_dir / LABEL_TO_MASK[label]
        gt, _ = load_binary_mask(gt_path, expected_size=expected_size)
        component_map, components = split_components(gt, args.min_component_area)
        total_area = int(gt.sum())

        for component in components:
            component_rows.append(
                {
                    "class": display,
                    "component": f"C{component.component_id}",
                    "area_px": component.area,
                    "area_%_of_GT": f"{100 * component.area / total_area:.1f}",
                    "scored_for_union": (
                        component.area / total_area >= args.important_component_area_fraction
                        if args.important_component_area_fraction > 0
                        else True
                    ),
                    "bbox_xyxy": f"{component.x0},{component.y0},{component.x1},{component.y1}",
                    "bbox_size": f"{component.width}x{component.height}",
                }
            )

        label_dir = args.out_dir / display.replace(" ", "_")
        label_dir.mkdir(parents=True, exist_ok=True)
        component_only_path = label_dir / f"{display}_components_only.png"
        he_overlay_path = label_dir / f"{display}_components_on_he.png"
        ficture_overlay_path = label_dir / f"{display}_components_on_ficture.png"
        component_only_image(component_map, components, f"{display}: GT components only").save(component_only_path)
        composite_overlay(he_image, component_map, components, f"{display}: GT components on H&E ROI").save(he_overlay_path)
        composite_overlay(ficture_image, component_map, components, f"{display}: GT components on FICTURE ROI").save(ficture_overlay_path)
        contact_path = label_dir / f"{display}_component_contact_sheet.png"
        make_contact_sheet(
            [
                ("components only", component_only_path),
                ("components on H&E", he_overlay_path),
                ("components on FICTURE", ficture_overlay_path),
            ],
            contact_path,
        )
        generated_images[display] = [contact_path, component_only_path, he_overlay_path, ficture_overlay_path]

        scored_components = [
            component
            for component in components
            if args.important_component_area_fraction <= 0
            or component.area / total_area >= args.important_component_area_fraction
        ]
        if args.max_components_per_label > 0:
            scored_components = scored_components[: args.max_components_per_label]
        component_gt_masks = [component_map == component.component_id for component in scored_components]
        component_gt_crops = [
            component_gt[component.y0 : component.y1, component.x0 : component.x1]
            for component, component_gt in zip(scored_components, component_gt_masks)
        ]

        for source, root in candidate_roots:
            best_full: tuple[float, CandidateMask, dict[str, float]] | None = None
            per_component_top: list[list[tuple[float, CandidateMask, dict[str, float]]]] = [
                [] for _ in scored_components
            ]
            per_component_top1_masks: list[np.ndarray | None] = [None for _ in scored_components]
            n_seen = 0

            for candidate in iter_candidate_masks(source, root, expected_size, args.max_candidates_per_root):
                n_seen += 1
                pred_area = int(candidate.mask.sum())
                full_tp = int(candidate.mask[gt].sum())
                full_m = metrics_from_counts(full_tp, pred_area, total_area)
                if best_full is None or full_m["dice"] > best_full[0]:
                    best_full = (full_m["dice"], candidate, full_m)

                if n_seen % 1000 == 0:
                    print(
                        f"{display} {source}: scanned {n_seen} masks; "
                        f"current best full Dice={best_full[0]:.4f}",
                        flush=True,
                    )

                for idx, (component, component_gt_crop) in enumerate(
                    zip(scored_components, component_gt_crops)
                ):
                    pred_crop = candidate.mask[component.y0 : component.y1, component.x0 : component.x1]
                    component_tp = int(pred_crop[component_gt_crop].sum())
                    m = metrics_from_counts(component_tp, pred_area, component.area)
                    ranked = per_component_top[idx]
                    ranked.append((m["dice"], candidate, m))
                    ranked.sort(key=lambda item: item[0], reverse=True)
                    del ranked[args.top_k_per_component :]
                    if ranked and ranked[0][1].path == candidate.path:
                        per_component_top1_masks[idx] = candidate.mask.copy()

            if n_seen == 0 or best_full is None:
                continue

            _, best_full_candidate, best_full_metrics = best_full

            selected_candidates: list[CandidateMask] = []
            for idx, component in enumerate(scored_components):
                ranked = per_component_top[idx]
                if not ranked:
                    continue
                selected_candidates.append(ranked[0][1])
                for rank, (_, candidate, m) in enumerate(ranked, start=1):
                    row = {
                        "class": display,
                        "source": source,
                        "component": f"C{component.component_id}",
                        "rank": rank,
                        "dice_vs_component": f"{m['dice']:.4f}",
                        "precision_vs_component": f"{m['precision']:.4f}",
                        "recall_vs_component": f"{m['recall']:.4f}",
                        "setting": candidate.setting,
                        "candidate": candidate.candidate_id,
                        "resized_from": candidate.resized_from,
                        "mask_path": str(candidate.path),
                    }
                    candidate_rows.append(row)

            union_mask = np.zeros_like(gt, dtype=bool)
            for top1_mask in per_component_top1_masks:
                if top1_mask is not None:
                    union_mask |= top1_mask
            union_m = metrics(union_mask, gt)
            union_rows.append(
                {
                    "class": display,
                    "source": source,
                    "single_best_Dice": f"{best_full_metrics['dice']:.4f}",
                    "single_best_Precision": f"{best_full_metrics['precision']:.4f}",
                    "single_best_Recall": f"{best_full_metrics['recall']:.4f}",
                    "component_union_Dice": f"{union_m['dice']:.4f}",
                    "component_union_Precision": f"{union_m['precision']:.4f}",
                    "component_union_Recall": f"{union_m['recall']:.4f}",
                    "n_total_components": len(components),
                    "n_scored_components": len(scored_components),
                    "scored_component_ids": ", ".join(f"C{component.component_id}" for component in scored_components),
                    "selected_candidates": "; ".join(
                        f"C{component.component_id}:{candidate.setting}/candidate_{candidate.candidate_id}"
                        for component, candidate in zip(scored_components, selected_candidates)
                    ),
                    "single_best_candidate": f"{best_full_candidate.setting}/candidate_{best_full_candidate.candidate_id}",
                }
            )

            union_mask_path = label_dir / f"{display}_{source}_component_union_mask.png"
            Image.fromarray((union_mask.astype(np.uint8) * 255), mode="L").save(union_mask_path)
            union_overlay_path = label_dir / f"{display}_{source}_component_union_on_he.png"
            union_overlay(he_image, gt, union_mask, f"{display}: {source} component-wise union").save(union_overlay_path)
            generated_images[display].extend([union_mask_path, union_overlay_path])

    pd.DataFrame(component_rows).to_csv(args.out_dir / "gt_component_summary.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(args.out_dir / "component_top_candidates.csv", index=False)
    pd.DataFrame(union_rows).to_csv(args.out_dir / "component_union_summary.csv", index=False)

    metadata = {
        "annotation_dir": str(args.annotation_dir),
        "he_image": str(args.he_image),
        "ficture_image": str(args.ficture_image),
        "labels": args.labels,
        "candidate_roots": args.candidate_root,
        "min_component_area": args.min_component_area,
        "top_k_per_component": args.top_k_per_component,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    with (args.out_dir / "gt_component_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(component_rows)
    write_html_report(args.out_dir, component_rows, candidate_rows, union_rows, generated_images)
    print(args.out_dir)


if __name__ == "__main__":
    main()
