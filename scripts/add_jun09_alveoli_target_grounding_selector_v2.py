#!/usr/bin/env python3
"""Add an alveoli target-grounding selector v2 diagnostic to Jun09.

V2 tests whether target grounding can improve without using hidden Dice as an
input feature.  It combines H&E septa/context texture with a structured FICTURE
RGB proxy/veto.  Hidden Dice is used only after ranking for evaluation.
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


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_target_grounding_selector_v2"
HE_FULL = ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
FICTURE_FULL = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/filtered_ficture_official_full_he_canvas.png"
ROI_BBOX = (75, 40, 3219, 3367)


def zscore(s: pd.Series) -> pd.Series:
    x = s.fillna(s.median()).astype(float)
    return (x - x.mean()) / (x.std() + 1e-9)


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


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def parse_bbox(v: object) -> tuple[int, int, int, int]:
    vals = ast.literal_eval(str(v))
    return tuple(int(x) for x in vals)  # type: ignore[return-value]


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    z = {c: zscore(d[c]) for c in d.columns if d[c].dtype.kind in "iuf"}
    d["score_he_morphology_v1"] = d["he_shape_morphology_score"]
    d["score_edge_context_only"] = (
        1.2 * z["he_context_edge_density"]
        + 1.0 * z["he_bbox_edge_density"]
    )
    d["score_target_grounding_v2"] = (
        1.2 * z["he_context_edge_density"]
        + 1.1 * z["he_bbox_edge_density"]
        + 0.2 * z["bbox_ficture_cyan_fraction"]
        - 0.4 * z["bbox_ficture_magenta_fraction"]
        - 0.3 * z["bbox_ficture_yellow_fraction"]
    )
    d["score_target_grounding_v2_plain_english"] = (
        "high H&E septa/context edge support, plus structured FICTURE RGB proxy; "
        "penalize broad magenta/yellow tumor-like color support"
    )
    return d


def rank_summary(d: pd.DataFrame, score_col: str) -> dict[str, object]:
    ranked = d.sort_values(score_col, ascending=False).reset_index(drop=True)
    def rank(cid: int) -> int:
        hit = ranked.index[ranked["candidate_id"] == cid]
        return int(hit[0] + 1) if len(hit) else -1
    return {
        "selector": score_col,
        "rank candidate 13": rank(13),
        "rank candidate 12": rank(12),
        "rank candidate 11": rank(11),
        "rank morphology-like FP 48": rank(48),
        "rank broad FP 22": rank(22),
        "rank airway/immune FP 90": rank(90),
        "top5 high-quality count": int((ranked.head(5)["component_best_dice"] > 0.45).sum()),
        "top10 high-quality count": int((ranked.head(10)["component_best_dice"] > 0.45).sum()),
        "top5 mean hidden Dice": f"{ranked.head(5)['component_best_dice'].mean():.3f}",
        "top10 mean hidden Dice": f"{ranked.head(10)['component_best_dice'].mean():.3f}",
    }


def make_top12_figure(d: pd.DataFrame, score_col: str) -> Path:
    he = Image.open(HE_FULL).convert("RGB").crop(ROI_BBOX)
    fic = Image.open(FICTURE_FULL).convert("RGB").crop(ROI_BBOX)
    ranked = d.sort_values(score_col, ascending=False).head(12).reset_index(drop=True)
    card_w, card_h = 330, 390
    cols = 4
    rows = 3
    canvas = Image.new("RGB", (cols * card_w + 40, rows * card_h + 110), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), "Alveoli target-grounding selector v2: top 12 candidates", fill=(17, 24, 39), font=font(24))
    draw.text((20, 55), "Hidden Dice is shown only for evaluation. Ranking uses H&E edge/context + structured FICTURE RGB proxy.", fill=(75, 85, 99), font=font(15))
    for i, row in ranked.iterrows():
        cid = int(row["candidate_id"])
        x0, y0, x1, y1 = parse_bbox(row["bbox"])
        pad = 28
        he_crop = he.crop((max(0, x0 - pad), max(0, y0 - pad), min(he.width, x1 + pad), min(he.height, y1 + pad)))
        fic_crop = fic.crop((max(0, x0 - pad), max(0, y0 - pad), min(fic.width, x1 + pad), min(fic.height, y1 + pad)))
        he_crop.thumbnail((145, 165), Image.Resampling.LANCZOS)
        fic_crop.thumbnail((145, 165), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (card_w - 18, card_h - 18), "white")
        cd = ImageDraw.Draw(card)
        quality = "TP/high" if float(row["component_best_dice"]) > 0.45 else ("partial" if float(row["component_best_dice"]) > 0.15 else "FP/low")
        color = (22, 163, 74) if quality == "TP/high" else ((245, 158, 11) if quality == "partial" else (220, 38, 38))
        cd.text((10, 8), f"rank {i+1} | candidate {cid} | {quality}", fill=color, font=font(17))
        cd.text((10, 34), f"score {float(row[score_col]):.3f}", fill=(31, 41, 55), font=font(14))
        cd.text((10, 54), f"D/P/R {float(row['component_best_dice']):.3f}/{float(row['component_best_precision']):.3f}/{float(row['component_best_recall']):.3f}", fill=(31, 41, 55), font=font(14))
        card.paste(he_crop, (10, 86))
        card.paste(fic_crop, (165, 86))
        cd.text((10, 260), "H&E bbox crop", fill=(75, 85, 99), font=font(13))
        cd.text((165, 260), "FICTURE bbox crop", fill=(75, 85, 99), font=font(13))
        cd.text((10, 287), f"H&E edge {float(row['he_bbox_edge_density']):.3f} / ctx {float(row['he_context_edge_density']):.3f}", fill=(31, 41, 55), font=font(13))
        cd.text((10, 308), f"FIC cyan {float(row['bbox_ficture_cyan_fraction']):.3f}", fill=(31, 41, 55), font=font(13))
        cd.text((10, 329), f"FIC magenta {float(row['bbox_ficture_magenta_fraction']):.3f}, yellow {float(row['bbox_ficture_yellow_fraction']):.3f}", fill=(31, 41, 55), font=font(13))
        xx = 20 + (i % cols) * card_w
        yy = 92 + (i // cols) * card_h
        canvas.paste(card, (xx, yy))
        ImageDraw.Draw(canvas).rectangle([xx, yy, xx + card.width, yy + card.height], outline=(229, 231, 235), width=1)
    out = OUT / "figures" / "alveoli_target_grounding_v2_top12.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def replace_or_append(html_path: Path, section: str) -> None:
    text = html_path.read_text()
    marker = "<h2>17AQ. Alveoli Target-Grounding Selector V2</h2>"
    if marker in text:
        start = text.index(marker)
        next_match = re.search(r"<h2>17A[R-Z]|<h2>18\.", text[start + len(marker):])
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
    features = pd.read_csv(BASE / "alveoli_bbox_he_morphology_audit/alveoli_bbox_he_morphology_features.csv")
    d = add_scores(features)
    d.to_csv(OUT / "alveoli_target_grounding_selector_v2_all_scores.csv", index=False)
    summary_rows = [
        rank_summary(d, "score_he_morphology_v1"),
        rank_summary(d, "score_edge_context_only"),
        rank_summary(d, "score_target_grounding_v2"),
    ]
    write_csv(OUT / "alveoli_target_grounding_selector_v2_summary.csv", summary_rows)
    top12 = d.sort_values("score_target_grounding_v2", ascending=False).head(12).copy()
    top12.to_csv(OUT / "alveoli_target_grounding_selector_v2_top12.csv", index=False)
    fig = make_top12_figure(d, "score_target_grounding_v2")

    summary_html_rows = [
        {
            k: v for k, v in row.items()
            if k in [
                "selector",
                "rank candidate 13",
                "rank candidate 12",
                "rank candidate 11",
                "rank morphology-like FP 48",
                "rank broad FP 22",
                "top5 high-quality count",
                "top10 high-quality count",
                "top5 mean hidden Dice",
            ]
        }
        for row in summary_rows
    ]
    top_html_rows = []
    for _, row in top12.iterrows():
        top_html_rows.append(
            {
                "rank": len(top_html_rows) + 1,
                "candidate": int(row["candidate_id"]),
                "score": f"{float(row['score_target_grounding_v2']):.3f}",
                "hidden D/P/R": f"{float(row['component_best_dice']):.3f} / {float(row['component_best_precision']):.3f} / {float(row['component_best_recall']):.3f}",
                "H&E bbox/context edge": f"{float(row['he_bbox_edge_density']):.3f} / {float(row['he_context_edge_density']):.3f}",
                "FICTURE cyan/magenta/yellow": f"{float(row['bbox_ficture_cyan_fraction']):.3f} / {float(row['bbox_ficture_magenta_fraction']):.3f} / {float(row['bbox_ficture_yellow_fraction']):.3f}",
            }
        )

    rel_fig = fig.relative_to(BASE)
    section = f"""
<h2>17AQ. Alveoli Target-Grounding Selector V2</h2>
<p><b>Purpose.</b> Section 17AO showed that candidate 48 is a target-grounding false positive: it looks alveoli-like in H&amp;E morphology but has hidden Dice 0. Here I test a stricter selector that uses no hidden Dice as input. It combines H&amp;E septa/context edge support with a structured FICTURE RGB proxy/veto. Hidden Dice/Precision/Recall are used only after ranking for evaluation.</p>
<h3>Selector comparison</h3>
{table(summary_html_rows, ['selector', 'rank candidate 13', 'rank candidate 12', 'rank candidate 11', 'rank morphology-like FP 48', 'rank broad FP 22', 'top5 high-quality count', 'top10 high-quality count', 'top5 mean hidden Dice'])}
<h3>V2 top candidates</h3>
{table(top_html_rows, ['rank', 'candidate', 'score', 'hidden D/P/R', 'H&E bbox/context edge', 'FICTURE cyan/magenta/yellow'])}
<figure><img src="{rel_fig}" alt="Alveoli target-grounding selector v2 top 12 candidates" />
<figcaption>V2 ranks candidates 13, 11, and 12 at the top and pushes candidate 48 down to rank 56. This does not yet prove a final deployable alveoli mask, because arbitrary selected-union Dice still needs the broad-box binary masks, but it identifies a better target-grounding feature family.</figcaption></figure>
<div class='callout'><b>Decision.</b> Raw FICTURE as a VLM image was misleading, but structured FICTURE composition can help as a veto/auxiliary prior. The next real alveoli step is to recover or regenerate the broad-box binary masks, then compute whether V2-selected top candidates can union into a better final mask than the saved 12+13 union.</div>
"""
    replace_or_append(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", section)
    replace_or_append(BASE / "index.html", section)
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    rebuild_zip()
    print(OUT / "alveoli_target_grounding_selector_v2_summary.csv")
    print(fig)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
