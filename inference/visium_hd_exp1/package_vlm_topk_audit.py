#!/usr/bin/env python3
"""Package VLM top-k selections into a visual audit report.

The report answers: for each label/ranker/top_k, which candidate masks were
selected, which candidate-pool setting they came from, and what the visual
evidence looked like on H&E/FICTURE.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


LABEL_ORDER = [
    "lung_bronchiola",
    "lung_alveoli_normal_adjacent",
    "lung_vessels",
]

LABEL_DISPLAY = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
}

COLORS = {
    "lung_bronchiola": (31, 119, 180),
    "lung_alveoli_normal_adjacent": (245, 245, 220),
    "lung_vessels": (50, 205, 215),
}


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L")) > 127
    if mask.shape != shape_hw:
        h, w = shape_hw
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
    return mask


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_box(mask: np.ndarray, pad: int = 64) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox(mask)
    h, w = mask.shape
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def resize_for_thumb(img: Image.Image, max_side: int = 320) -> Image.Image:
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.BILINEAR)
    return img


def save_candidate_thumb(
    he: np.ndarray,
    ficture: np.ndarray,
    mask: np.ndarray,
    out_prefix: Path,
    label: str,
) -> Dict[str, Path]:
    color = COLORS.get(label, (0, 128, 255))
    x1, y1, x2, y2 = crop_box(mask)
    outs = {}
    panels = {
        "he": overlay(he, mask, color, 0.46),
        "mask": np.dstack([mask.astype(np.uint8) * color[0], mask.astype(np.uint8) * color[1], mask.astype(np.uint8) * color[2]]),
        "ficture": overlay(ficture, mask, color, 0.50),
    }
    for name, arr in panels.items():
        img = Image.fromarray(arr[y1:y2, x1:x2].astype(np.uint8))
        img = resize_for_thumb(img)
        path = out_prefix.with_name(out_prefix.name + f"_{name}.png")
        img.save(path)
        outs[name] = path
    return outs


def save_union_panel(
    he: np.ndarray,
    gt: np.ndarray | None,
    masks: Iterable[np.ndarray],
    out_path: Path,
    label: str,
    title: str,
) -> None:
    union = np.zeros(he.shape[:2], dtype=bool)
    for mask in masks:
        union |= mask
    focus = union.copy()
    if gt is not None:
        focus |= gt
    x1, y1, x2, y2 = crop_box(focus, pad=96)
    pred = overlay(he, union, COLORS.get(label, (0, 128, 255)), 0.48)
    if gt is not None:
        pred = overlay(pred, gt, (0, 190, 95), 0.36)
    crop = Image.fromarray(pred[y1:y2, x1:x2].astype(np.uint8))
    crop = resize_for_thumb(crop, max_side=520)
    canvas = Image.new("RGB", (crop.width, crop.height + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:120], fill=(0, 0, 0))
    canvas.paste(crop, (0, 34))
    canvas.save(out_path)


def load_gt_masks(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    out = {}
    for item in summary.get("labels", []):
        slug = item.get("slug") or slugify(item.get("label", ""))
        if slug not in LABEL_DISPLAY:
            continue
        path = Path(item["mask_path"])
        if not path.exists():
            path = summary_path.parent / item["mask_path"]
        out[slug] = read_mask(path, shape_hw)
    return out


def parse_log_params(log_path: Path | None) -> Dict[str, str]:
    keys = {
        "RUN_NAME",
        "MODEL_NAME",
        "FACTOR_SEMANTIC_LEGEND",
        "PER_LABEL",
        "TOP_KS",
        "MAX_OVERLAP",
        "MAX_FACTOR_CONTEXT_CHARS",
    }
    params: Dict[str, str] = {}
    if log_path is None or not log_path.exists():
        return params
    for line in log_path.read_text(errors="replace").splitlines():
        for key in keys:
            if line.startswith(key + "="):
                params[key] = line.split("=", 1)[1].strip()
        if "PER_LABEL=" in line and "TOP_KS=" in line:
            for part in line.strip().split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
        if line.startswith("Official score args: ") and line.endswith(".csv"):
            params.setdefault("SCORE_CSV", line.replace("Official score args: ", "").strip())
    return params


def find_log(project: Path, root: Path) -> Path | None:
    match = re.search(r"_(\d+)$", root.name)
    if not match:
        return None
    path = project / "results/visium_hd_exp1/logs" / f"vlm_judge_off_{match.group(1)}.log"
    return path if path.exists() else None


def candidate_key(row: dict) -> Tuple[str, str, str, str]:
    return row["label"], row["source"], row["setting"], str(row["candidate_id"])


def rel(path: Path, base: Path) -> str:
    return html.escape(str(path.relative_to(base)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--vlm-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Official FICTURE VLM top-k selection audit")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    input_dir = project / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
    summary_path = input_dir / "region_summary_official_filtered_roi_hpc.json"
    he_path = input_dir / "he_roi_matching_official_ficture_coverage.png"
    ficture_path = input_dir / "ficture_official_filtered_roi_rgb.png"
    official_summary = project / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
    official_status = json.loads(official_summary.read_text()).get("status") if official_summary.exists() else "MISSING"

    he = np.array(Image.open(he_path).convert("RGB"))
    ficture = np.array(Image.open(ficture_path).convert("RGB"))
    shape_hw = he.shape[:2]
    gt_masks = load_gt_masks(summary_path, shape_hw)

    rows_out: List[dict] = []
    html_parts: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(args.title)}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222} h1,h2,h3{margin-bottom:6px}"
        "table{border-collapse:collapse;width:100%;margin:10px 0 22px} th,td{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}"
        ".run{border:1px solid #bbb;border-radius:8px;padding:14px;margin:18px 0}.rowblk{border-top:2px solid #ddd;padding-top:12px;margin-top:18px}"
        ".cards{display:flex;gap:10px;flex-wrap:wrap}.card{border:1px solid #ccc;border-radius:6px;padding:8px;max-width:340px}"
        ".thumbs{display:flex;gap:4px}.thumbs img{max-width:100px;max-height:120px;border:1px solid #ddd}"
        ".union{max-width:520px;border:1px solid #aaa}.small{font-size:12px;color:#555}.metric{font-weight:bold}</style></head><body>",
        f"<h1>{html.escape(args.title)}</h1>",
        "<h2>Experiment Design</h2>",
        "<ul>",
        f"<li>Official FICTURE status: <b>{html.escape(str(official_status))}</b></li>",
        f"<li>Official candidate input: {html.escape(str(input_dir))}</li>",
        "<li>Rows show each <b>label / ranker / top_k</b>. top_k means the first k ranked candidates are selected, overlap-gated, and unioned for evaluation.</li>",
        "<li>Each candidate card shows source, candidate-pool setting, candidate id, candidate-level scores, and visual crops on H&E / mask-only / FICTURE.</li>",
        "<li>This is a VLM/CLIP top-k audit. It is not the full SaLIP second-stage SAM rerun.</li>",
        "</ul>",
    ]

    for root in args.vlm_root:
        root = (project / root).resolve() if not root.is_absolute() else root.resolve()
        run_slug = slugify(root.name)
        selection_path = root / "vlm_selection_metrics.csv"
        score_path = root / "vlm_candidate_scores.csv"
        if not selection_path.exists() or not score_path.exists():
            continue
        log_path = find_log(project, root)
        params = parse_log_params(log_path)
        scores = read_csv(score_path)
        score_map = {candidate_key(row): row for row in scores}

        html_parts.append(f"<section class='run'><h2>{html.escape(root.name)}</h2>")
        html_parts.append("<h3>Run Parameters</h3><table><tbody>")
        for key in ["MODEL_NAME", "RUN_NAME", "SCORE_CSV", "FACTOR_SEMANTIC_LEGEND", "PER_LABEL", "TOP_KS", "MAX_OVERLAP", "MAX_FACTOR_CONTEXT_CHARS"]:
            html_parts.append(f"<tr><th>{key}</th><td>{html.escape(params.get(key, 'NA'))}</td></tr>")
        html_parts.append(f"<tr><th>Output root</th><td>{html.escape(str(root))}</td></tr>")
        html_parts.append("</tbody></table>")

        selection_rows = [r for r in read_csv(selection_path) if r.get("label") in LABEL_ORDER]
        selection_rows.sort(key=lambda r: (LABEL_ORDER.index(r["label"]), r["ranker"], int(r["top_k"])))
        for row in selection_rows:
            label = row["label"]
            display = row.get("display") or LABEL_DISPLAY.get(label, label)
            ranker = row["ranker"]
            top_k = row["top_k"]
            selected = [s for s in row.get("selected", "").split(";") if s]
            masks = []
            cards = []
            for order, sel in enumerate(selected, start=1):
                parts = sel.split("/")
                if len(parts) != 3:
                    continue
                source, setting, cand_id = parts
                score = score_map.get((label, source, setting, cand_id))
                if score is None:
                    continue
                mask_path = Path(score["mask_path"])
                mask = read_mask(mask_path, shape_hw)
                masks.append(mask)
                prefix = asset_dir / f"{run_slug}_{slugify(label)}_{slugify(ranker)}_top{top_k}_{order}_{slugify(setting)}_{cand_id}"
                thumbs = save_candidate_thumb(he, ficture, mask, prefix, label)
                cards.append((order, sel, score, thumbs))
            union_path = asset_dir / f"{run_slug}_{slugify(label)}_{slugify(ranker)}_top{top_k}_union.png"
            save_union_panel(
                he,
                gt_masks.get(label),
                masks,
                union_path,
                label,
                f"{display} {ranker} top-{top_k}: D {float(row['dice']):.3f} P {float(row['precision']):.3f} R {float(row['recall']):.3f}",
            )

            rows_out.append({
                "run": root.name,
                "model": params.get("MODEL_NAME", ""),
                "label": display,
                "ranker": ranker,
                "top_k": top_k,
                "dice": row["dice"],
                "precision": row["precision"],
                "recall": row["recall"],
                "selected": row.get("selected", ""),
                "output_root": str(root),
            })

            html_parts.append("<div class='rowblk'>")
            html_parts.append(
                f"<h3>{html.escape(display)} | {html.escape(ranker)} | top_k={html.escape(top_k)}</h3>"
                f"<p class='metric'>Dice {float(row['dice']):.3f} | Precision {float(row['precision']):.3f} | Recall {float(row['recall']):.3f} | n_selected={html.escape(row.get('n_selected',''))}</p>"
                f"<img class='union' src='{rel(union_path, output_dir)}'>"
            )
            html_parts.append("<div class='cards'>")
            for order, sel, score, thumbs in cards:
                html_parts.append("<div class='card'>")
                html_parts.append(f"<b>#{order}: {html.escape(sel)}</b>")
                html_parts.append("<div class='small'>")
                html_parts.append(f"candidate pool setting: {html.escape(score.get('setting',''))}<br>")
                html_parts.append(f"candidate id: {html.escape(str(score.get('candidate_id','')))}<br>")
                html_parts.append(f"mask: {html.escape(score.get('mask_path',''))}<br>")
                for key in ["vlm_score", "vlm_fused_score", "base_score", "molecular_score", "factor_semantic_score", "he_score", "shape_score"]:
                    if key in score:
                        try:
                            html_parts.append(f"{key}: {float(score[key]):.3f}<br>")
                        except Exception:
                            html_parts.append(f"{key}: {html.escape(score[key])}<br>")
                html_parts.append("</div><div class='thumbs'>")
                for name in ["he", "mask", "ficture"]:
                    html_parts.append(f"<img title='{name}' src='{rel(thumbs[name], output_dir)}'>")
                html_parts.append("</div></div>")
            html_parts.append("</div></div>")
        html_parts.append("</section>")

    if rows_out:
        with (output_dir / "topk_selection_table.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    html_parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(html_parts))
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
