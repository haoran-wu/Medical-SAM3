#!/usr/bin/env python3
"""Render a CLIP vs VLM retrieval hit-test report for official FICTURE candidates."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]

LABEL_COLORS = {
    "lung_bronchiola": (31, 119, 180),
    "lung_alveoli_normal_adjacent": (245, 245, 220),
    "lung_vessels": (50, 205, 215),
    "tumor": (23, 190, 207),
    "stroma": (188, 189, 34),
    "immune_infiltration": (44, 160, 44),
}


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.array(image) > 127


def resize_rgb(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR)
    return np.array(image)


def load_gt_masks(summary_path: Path) -> Tuple[Dict[str, np.ndarray], Tuple[int, int]]:
    obj = json.loads(summary_path.read_text())
    wanted = {slug for slug, _display in LABEL_ORDER}
    masks: Dict[str, np.ndarray] = {}
    shape_hw: Tuple[int, int] | None = None
    for item in obj["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        if slug not in wanted:
            continue
        path = Path(item["mask_path"])
        if not path.exists():
            path = summary_path.parent / item["mask_path"]
        mask = np.array(Image.open(path).convert("L")) > 127
        shape_hw = mask.shape if shape_hw is None else shape_hw
        masks[slug] = mask
    if shape_hw is None:
        raise SystemExit(f"No masks found in {summary_path}")
    return masks, shape_hw


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def bbox(mask: np.ndarray, pad: int = 48) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    h, w = mask.shape
    return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))


def key_for(row: dict) -> Tuple[str, str, str, str, str]:
    return (
        row["label"],
        row["source"],
        row["run"],
        row["setting"],
        str(row["candidate_id"]),
    )


def parse_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return 0.0


def score_maps(hit_dir: Path, vlm_csv: Path | None) -> Dict[str, Dict[Tuple[str, str, str, str, str], float]]:
    maps: Dict[str, Dict[Tuple[str, str, str, str, str], float]] = {
        "CLIP H&E image-text": {},
        "CLIP FICTURE color-text": {},
        "CLIP mean": {},
    }
    for row in read_csv(hit_dir / "candidate_multimodal_scores.csv"):
        key = key_for(row)
        ranker = row.get("ranker", "")
        if ranker == "clip_he":
            maps["CLIP H&E image-text"][key] = parse_float(row, "score")
        elif ranker == "clip_ficture":
            maps["CLIP FICTURE color-text"][key] = parse_float(row, "score")
        elif ranker == "clip_mean":
            maps["CLIP mean"][key] = parse_float(row, "score")
    if vlm_csv and vlm_csv.exists():
        maps["VLM score"] = {}
        maps["VLM fused score"] = {}
        for row in read_csv(vlm_csv):
            key = key_for(row)
            maps["VLM score"][key] = parse_float(row, "vlm_score")
            maps["VLM fused score"][key] = parse_float(row, "vlm_fused_score")
    return maps


def render_card(
    he: np.ndarray,
    gt: np.ndarray,
    row: dict,
    score: float,
    rank: int,
    method: str,
    out_path: Path,
    shape_hw: Tuple[int, int],
) -> None:
    mask = read_mask(Path(row["mask_path"]), shape_hw)
    color = LABEL_COLORS[row["label"]]
    x1, y1, x2, y2 = bbox(np.logical_or(mask, gt), pad=36)
    candidate = overlay(he, mask, color, 0.50)[y1:y2, x1:x2]
    annotation = overlay(he, gt, (0, 185, 95), 0.45)[y1:y2, x1:x2]
    mask_rgb = np.zeros_like(candidate)
    mask_crop = mask[y1:y2, x1:x2]
    mask_rgb[mask_crop] = np.array(color, dtype=np.uint8)
    panels = [candidate, mask_rgb, annotation]
    panel_w = 360
    resized = []
    for panel in panels:
        img = Image.fromarray(panel)
        scale = panel_w / max(1, img.width)
        img = img.resize((panel_w, max(1, int(img.height * scale))), Image.Resampling.BILINEAR)
        resized.append(img)
    h = max(img.height for img in resized)
    canvas = Image.new("RGB", (panel_w * 3, h + 92), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"#{rank} {method} score={score:.3f} | true Dice={parse_float(row, 'hidden_dice'):.3f} "
        f"P={parse_float(row, 'hidden_precision'):.3f} R={parse_float(row, 'hidden_recall'):.3f}"
    )
    subtitle = f"{row['display']} | {row['sample_bucket']} | original best for {row['candidate_original_best_label']} | {row['setting']} / {row['candidate_id']}"
    draw.text((10, 8), title, fill="black")
    draw.text((10, 34), subtitle, fill="black")
    draw.text((10, 64), "candidate on H&E", fill="blue")
    draw.text((panel_w + 10, 64), "candidate mask", fill="blue")
    draw.text((panel_w * 2 + 10, 64), "annotation on H&E", fill="green")
    for i, img in enumerate(resized):
        canvas.paste(img, (i * panel_w, 92))
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-test-dir", type=Path, required=True)
    parser.add_argument("--vlm-csv", type=Path, default=None)
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    truth_rows = read_csv(args.hit_test_dir / "hidden_candidate_truth.csv")
    gt_masks, shape_hw = load_gt_masks(args.summary_path)
    he = resize_rgb(args.he_image, shape_hw)
    maps = score_maps(args.hit_test_dir, args.vlm_csv)

    by_label: Dict[str, List[dict]] = {slug: [] for slug, _display in LABEL_ORDER}
    for row in truth_rows:
        by_label.setdefault(row["label"], []).append(row)

    summary_rows: List[dict] = []
    html_lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>CLIP vs VLM candidate hit-test</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px} table{border-collapse:collapse;width:100%;margin:16px 0} th,td{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left} th{background:#f5f5f5} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}.card img{max-width:100%;border:1px solid #ccc}.muted{color:#666}</style>",
        "</head><body>",
        "<h1>CLIP vs VLM candidate-pool hit-test</h1>",
        "<p>This asks one question for both methods: can the scorer pull good masks above bad masks from the same small official FICTURE candidate set? Erythrocytes and pigment are excluded.</p>",
        "<p class='muted'>Truth Dice is used only after scoring, to audit retrieval quality.</p>",
    ]

    for label, display in LABEL_ORDER:
        rows = by_label.get(label, [])
        if not rows:
            continue
        html_lines.append(f"<h2>{html.escape(display)}</h2>")
        html_lines.append("<table><thead><tr><th>method</th><th>top_k</th><th>best true Dice in top-k</th><th>top-1 true Dice</th><th>top selected</th></tr></thead><tbody>")
        for method, scores in maps.items():
            scored = [
                (scores.get(key_for(row), float("-inf")), row)
                for row in rows
                if key_for(row) in scores
            ]
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            method_slug = slugify(method)
            for top_k in args.top_ks:
                top = scored[:top_k]
                best = max((row for _score, row in top), key=lambda row: parse_float(row, "hidden_dice"))
                top1_dice = parse_float(scored[0][1], "hidden_dice")
                selected_text = "; ".join(
                    f"{row['sample_bucket']} Dice={parse_float(row, 'hidden_dice'):.3f} score={score:.3f}"
                    for score, row in top[:3]
                )
                summary_rows.append(
                    {
                        "label": label,
                        "display": display,
                        "method": method,
                        "top_k": top_k,
                        "best_true_dice_in_topk": f"{parse_float(best, 'hidden_dice'):.9f}",
                        "top1_true_dice": f"{top1_dice:.9f}",
                        "top1_score": f"{scored[0][0]:.9f}",
                        "top1_bucket": scored[0][1]["sample_bucket"],
                        "top1_source": f"{scored[0][1]['source']}/{scored[0][1]['setting']}/{scored[0][1]['candidate_id']}",
                    }
                )
                html_lines.append(
                    f"<tr><td>{html.escape(method)}</td><td>{top_k}</td><td>{parse_float(best, 'hidden_dice'):.3f}</td>"
                    f"<td>{top1_dice:.3f}</td><td>{html.escape(selected_text)}</td></tr>"
                )
            html_lines.append("</tbody></table>")
            html_lines.append(f"<h3>{html.escape(method)} top selections</h3><div class='grid'>")
            for rank, (score, row) in enumerate(scored[:3], start=1):
                img_name = f"{label}_{method_slug}_{rank}.png"
                render_card(he, gt_masks[label], row, score, rank, method, fig_dir / img_name, shape_hw)
                html_lines.append(f"<div class='card'><img src='figures/{img_name}'></div>")
            html_lines.append("</div>")

    with (args.output_dir / "method_topk_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    html_lines.append("</body></html>")
    (args.output_dir / "index.html").write_text("\n".join(html_lines) + "\n")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
