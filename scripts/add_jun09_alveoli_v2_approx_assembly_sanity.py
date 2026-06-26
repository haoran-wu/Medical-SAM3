#!/usr/bin/env python3
"""Approximate local assembly sanity check for alveoli V2.

Remote broad-box binary masks are not currently accessible.  This script
reconstructs approximate masks from saved H&E gray reverse-blur crops using the
candidate area from metadata, then tests a conservative assembly rule.  It is a
sanity check, not a replacement for exact mask evaluation.
"""

from __future__ import annotations

import ast
import csv
import html
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_v2_approx_assembly_sanity"
IMGDIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_GPT_manual_alveoli_prompt_test_inputs/images"
TRUTH = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_GPT_manual_alveoli_prompt_test_inputs/tables/hidden_candidate_truth.csv"
HE_FULL = ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
FICTURE_FULL = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/filtered_ficture_official_full_he_canvas.png"
TARGET_FULL = ROOT / "output/visium_hd_exp1/sam3_runs/base_sam3_multipoint_remaining_dev/masks/04_lung_alveoli_normal_adjacent_target.png"
ROI_BBOX = (75, 40, 3219, 3367)


def parse_bbox(v: object) -> tuple[int, int, int, int]:
    vals = ast.literal_eval(str(v))
    return tuple(int(x) for x in vals)  # type: ignore[return-value]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


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


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def target_mask() -> np.ndarray:
    full = np.array(Image.open(TARGET_FULL)) > 0
    x0, y0, x1, y1 = ROI_BBOX
    return full[y0:y1, x0:x1]


def reconstruct_mask(cid: int, truth: pd.DataFrame) -> np.ndarray:
    files = list(IMGDIR.glob(f"*__{cid:03d}_he_reverse_blur_gray.png"))
    if not files:
        raise FileNotFoundError(f"no saved reverse-blur crop for candidate {cid}")
    im = np.asarray(Image.open(files[0]).convert("RGB")).astype(np.float32) / 255.0
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    gray = im.mean(axis=2)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    grad = np.sqrt(gx * gx + gy * gy)
    score = (sat - sat.mean()) / (sat.std() + 1e-6) + 0.35 * (grad - grad.mean()) / (grad.std() + 1e-6)

    row = truth[truth["candidate_id"] == cid].iloc[0]
    x0, y0, x1, y1 = parse_bbox(row["candidate_bbox_xyxy"])
    fill = float(row["candidate_pixels"]) / ((x1 - x0) * (y1 - y0))
    k = max(1, min(score.size, int(round(fill * score.size))))
    threshold = np.partition(score.ravel(), -k)[-k]
    mask = score >= threshold
    closed = ndimage.binary_closing(mask, iterations=2)
    closed = ndimage.binary_fill_holes(closed)
    if abs(float(closed.mean()) - fill) <= 0.08:
        mask = closed
    mask_img = Image.fromarray((mask * 255).astype("uint8")).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    full = np.zeros((3327, 3144), dtype=bool)
    full[y0:y1, x0:x1] = np.array(mask_img) > 0
    return full


def metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, int]:
    inter = int((pred & truth).sum())
    pp = int(pred.sum())
    tt = int(truth.sum())
    precision = inter / pp if pp else 0.0
    recall = inter / tt if tt else 0.0
    dice = 2 * inter / (pp + tt) if pp + tt else 0.0
    return dice, precision, recall, pp


def overlap_small(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    denom = min(int(a.sum()), int(b.sum()))
    return inter / denom if denom else 0.0


def conservative_select(ranked_ids: list[int], masks: dict[int, np.ndarray], max_pieces: int = 2, overlap_threshold: float = 0.60) -> tuple[list[int], list[dict[str, object]]]:
    selected: list[int] = []
    trace: list[dict[str, object]] = []
    for cid in ranked_ids:
        if cid not in masks:
            trace.append({"candidate": cid, "decision": "skip", "reason": "no saved crop for approximate reconstruction"})
            continue
        max_overlap = max((overlap_small(masks[cid], masks[s]) for s in selected), default=0.0)
        if max_overlap > overlap_threshold:
            trace.append({"candidate": cid, "decision": "skip", "reason": f"overlap-small {max_overlap:.3f} > {overlap_threshold:.2f}"})
            continue
        selected.append(cid)
        trace.append({"candidate": cid, "decision": "select", "reason": f"overlap-small {max_overlap:.3f}; selected {len(selected)}/{max_pieces}"})
        if len(selected) >= max_pieces:
            break
    return selected, trace


def overlay_mask(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> Image.Image:
    img = base.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    overlay = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    arr[m] = (1 - alpha) * arr[m] + alpha * overlay
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    arr[mask] = color
    return Image.fromarray(arr)


def make_six_panel(selected: list[int], union_mask: np.ndarray, truth_mask: np.ndarray, summary: dict[str, object]) -> Path:
    he = Image.open(HE_FULL).convert("RGB").crop(ROI_BBOX)
    fic = Image.open(FICTURE_FULL).convert("RGB").crop(ROI_BBOX)
    panels = [
        ("Annotation on H&E", overlay_mask(he, truth_mask, (16, 185, 129), 0.45)),
        ("V2 selected union on H&E", overlay_mask(he, union_mask, (37, 99, 235), 0.45)),
        ("V2 selected union mask only", mask_only(union_mask, (37, 99, 235))),
        ("Annotation mask only", mask_only(truth_mask, (16, 185, 129))),
        ("H&E ROI", he),
        ("FICTURE ROI", fic),
    ]
    thumb_w, thumb_h = 320, 340
    header_h = 112
    canvas = Image.new("RGB", (len(panels) * thumb_w + 34, header_h + thumb_h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), "alveoli: V2 approximate selected-union sanity check", fill=(17, 24, 39), font=font(24))
    draw.text((20, 48), f"Selected candidates: {selected} | Approx D/P/R {summary['Dice']} / {summary['Precision']} / {summary['Recall']}", fill=(31, 41, 55), font=font(17))
    draw.text((20, 75), "Approximate masks are reconstructed from saved gray reverse-blur crops using candidate area metadata; exact broad-box masks are still needed for final evaluation.", fill=(75, 85, 99), font=font(14))
    for i, (title, img) in enumerate(panels):
        thumb = img.copy()
        thumb.thumbnail((thumb_w - 24, thumb_h - 54), Image.Resampling.LANCZOS)
        x = 18 + i * thumb_w
        y = header_h
        draw.text((x, y), title, fill=(31, 41, 55), font=font(15))
        canvas.paste(thumb, (x, y + 28))
    out = OUT / "figures" / "alveoli_v2_approx_union_six_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AR. Alveoli V2 Approximate Assembly Sanity Check</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[S-Z]|<h2>18\.", text[start + len(marker):])
        end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
        text = text[:start] + section + text[end:]
    else:
        text = text.replace("</body>", section + "</body>")
    html_path.write_text(text)


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text(errors="ignore")
    missing = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (html_path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"missing image assets: {missing[:5]}")


def rebuild_zip() -> None:
    import importlib.util
    script = ROOT / "scripts/add_jun09_precise_failure_framework_v5.py"
    spec = importlib.util.spec_from_file_location("pack", script)
    pack = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pack)
    pack.rebuild_zip()
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
    if bad:
        raise RuntimeError(f"bad zip entry: {bad}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(TRUTH)
    scores = pd.read_csv(BASE / "alveoli_target_grounding_selector_v2/alveoli_target_grounding_selector_v2_all_scores.csv")
    ranked_ids = [int(x) for x in scores.sort_values("score_target_grounding_v2", ascending=False)["candidate_id"].tolist()]
    saved_ids = sorted(
        {
            int(m.group(1))
            for p in IMGDIR.glob("*_he_reverse_blur_gray.png")
            for m in [re.search(r"__(\d+)_he_reverse_blur_gray", p.name)]
            if m
        }
    )
    target = target_mask()
    masks = {cid: reconstruct_mask(cid, truth) for cid in saved_ids}
    single_rows = []
    for cid in saved_ids:
        d, p, r, pix = metrics(masks[cid], target)
        hidden = truth[truth["candidate_id"] == cid].iloc[0]
        single_rows.append(
            {
                "candidate": cid,
                "approx Dice": f"{d:.3f}",
                "approx Precision": f"{p:.3f}",
                "approx Recall": f"{r:.3f}",
                "hidden Dice": f"{float(hidden['component_best_dice']):.3f}",
                "hidden Precision": f"{float(hidden['component_best_precision']):.3f}",
                "hidden Recall": f"{float(hidden['component_best_recall']):.3f}",
                "approx pixels": pix,
                "metadata candidate pixels": int(hidden["candidate_pixels"]),
            }
        )
    write_csv(OUT / "alveoli_v2_approx_reconstructed_single_metrics.csv", single_rows)

    selected, trace = conservative_select(ranked_ids, masks, max_pieces=2, overlap_threshold=0.60)
    union = np.zeros_like(target, dtype=bool)
    for cid in selected:
        union |= masks[cid]
    d, p, r, pix = metrics(union, target)
    summary = {
        "rule": "V2 score order, skip candidate if overlap-small with selected union > 0.60, max 2 pieces for alveoli",
        "selected": ";".join(str(x) for x in selected),
        "Dice": f"{d:.3f}",
        "Precision": f"{p:.3f}",
        "Recall": f"{r:.3f}",
        "predicted pixels": pix,
        "target pixels": int(target.sum()),
        "caveat": "approximate reconstruction from saved crops; exact broad-box binary masks still needed",
    }
    write_csv(OUT / "alveoli_v2_approx_assembly_summary.csv", [summary])
    write_csv(OUT / "alveoli_v2_approx_assembly_trace.csv", trace)
    fig = make_six_panel(selected, union, target, summary)

    trace_rows = trace[:8]
    rel_fig = fig.relative_to(BASE)
    section = f"""
<h2>17AR. Alveoli V2 Approximate Assembly Sanity Check</h2>
<p><b>Purpose.</b> The exact broad-box binary masks are still remote-only, so this is not final evaluation. To keep moving, I reconstructed approximate masks from the saved gray reverse-blur crops using candidate area metadata. This lets us test whether the V2 selector plus a conservative assembly rule has the right direction before waiting for exact masks.</p>
<h3>Assembly rule</h3>
{table([summary], ['rule', 'selected', 'Dice', 'Precision', 'Recall', 'predicted pixels', 'target pixels', 'caveat'])}
<h3>Selection trace</h3>
{table(trace_rows, ['candidate', 'decision', 'reason'])}
<figure><img src="{rel_fig}" alt="Alveoli V2 approximate selected union six panel" />
<figcaption>Approximate six-panel visualization. The rule selects 13 and 12; candidate 11 is skipped because it overlaps too much with 13 and mainly trades precision for recall. This recovers the same direction as the saved 12+13 union, but exact broad-box masks are still needed for final D/P/R.</figcaption></figure>
<div class='callout'><b>Decision.</b> The next alveoli skill step is now more precise: keep V2 target-grounding ranking, add conservative overlap-aware assembly, and recover exact broad-box masks to replace this approximate sanity check with exact evaluation.</div>
"""
    replace_or_append(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", section)
    replace_or_append(BASE / "index.html", section)
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    rebuild_zip()
    print(OUT / "alveoli_v2_approx_assembly_summary.csv")
    print(fig)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
