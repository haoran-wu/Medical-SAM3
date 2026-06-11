#!/usr/bin/env python3
"""Append the vessels component-1 oracle prompt diagnostic to Jun09 report."""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import add_jun09_precise_failure_framework_v5 as pack


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "vessels_component_oracle_prompt_diagnostic"
ANNOTATION = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks/05_lung_vessels_target_roi.png"
HTML = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
INDEX = BASE / "index.html"


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
    bits = ["<table><thead><tr>"]
    bits.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    bits.append("</tr></thead><tbody>")
    for row in rows:
        bits.append("<tr>")
        for col in cols:
            bits.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        bits.append("</tr>")
    bits.append("</tbody></table>")
    return "".join(bits)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pa = int(pred.sum())
    ga = int(gt.sum())
    return {
        "dice": 2 * tp / (pa + ga) if pa + ga else 0.0,
        "precision": tp / pa if pa else 0.0,
        "recall": tp / ga if ga else 0.0,
        "pred_pixels": pa,
        "gt_pixels": ga,
        "tp_pixels": tp,
    }


def append_or_replace(section: str) -> None:
    marker = "<h2>17BD. Vessels Component-1 Oracle Prompt Diagnostic</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B[E-Z]|<h2>18\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            insert_before = "<h2>18."
            if insert_before in text:
                idx = text.index(insert_before)
                text = text[:idx] + section + text[idx:]
            else:
                text = text.replace("</body>", section + "</body>")
        path.write_text(text)


def verify(path: Path) -> None:
    text = path.read_text(errors="ignore")
    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"missing images in {path}: {missing[:8]}")


def main() -> None:
    ann = np.asarray(Image.open(ANNOTATION).convert("L")) > 127
    labels, _ = ndimage.label(ann, structure=ndimage.generate_binary_structure(2, 1))
    comp1 = labels == 1
    runs = [
        {
            "run": "oracle component mask before second-SAM",
            "rel": "oracle_component_masks/vessels_component_1_oracle_mask.png",
            "meaning": "annotation component itself; diagnostic upper bound, not deployable",
        },
        {
            "run": "wide oracle box, margin 0.50",
            "rel": "local_cpu_sam3_oracle_component1_box_m50/masks/vessels_oracle_vessels_component_1_second_sam.png",
            "meaning": "accurate location but loose box; tests whether SAM leaks into nearby tissue",
        },
        {
            "run": "tight oracle box, margin 0.05",
            "rel": "local_cpu_sam3_oracle_component1_tight_box_m05/masks/vessels_oracle_vessels_component_1_second_sam.png",
            "meaning": "accurate location and tight box; tests the best realistic prompt geometry",
        },
    ]
    rows: list[dict[str, object]] = []
    for row in runs:
        mask_path = OUT / row["rel"]
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        m = metrics(mask, comp1)
        rows.append(
            {
                "run": row["run"],
                "component-1 Dice / Precision / Recall": f"{m['dice']:.3f} / {m['precision']:.3f} / {m['recall']:.3f}",
                "pred pixels": m["pred_pixels"],
                "component pixels": m["gt_pixels"],
                "meaning": row["meaning"],
                "mask path": row["rel"],
            }
        )
    write_csv(OUT / "vessels_component1_oracle_prompt_metrics.csv", rows)

    figure_rel = "vessels_component_oracle_prompt_diagnostic/local_cpu_sam3_oracle_component1_tight_box_m05/figures/vessels_second_sam_union.png"
    preview_rel = "vessels_component_oracle_prompt_diagnostic/dry_run/figures/vessels_selected_piece_prompt_preview.png"
    section = f"""
<h2>17BD. Vessels Component-1 Oracle Prompt Diagnostic</h2>
<p><b>Goal.</b> Section 17BC identified vessels component 1 as the clean-proposal bottleneck.  This diagnostic asks a sharper question: if the location of that tiny vessel component were already known, can SAM3 recover it?</p>
<p><b>Method.</b> This is an oracle diagnostic, not a deployable method.  I use the hidden annotation component only to create a tiny prompt mask/box, then run local CPU SAM3 with a wide box and a tight box.  Metrics below are computed against vessels component 1 only, not the whole vessels annotation.</p>
{table(rows, ["run", "component-1 Dice / Precision / Recall", "pred pixels", "component pixels", "meaning"])}
<figure><img src="{html.escape(preview_rel)}" alt="vessels component 1 oracle prompt preview"><figcaption>Dry-run prompt preview for the tiny vessels component 1.  Full-class recall is tiny because this is one small component, so component-level metrics are the meaningful ones.</figcaption></figure>
<figure><img src="{html.escape(figure_rel)}" alt="vessels component 1 tight oracle second-SAM"><figcaption>Tight oracle box second-SAM for vessels component 1.  The tight box reaches component-level D/P/R 0.876/0.809/0.956, while the wide box leaks badly.</figcaption></figure>
<div class='callout'><b>Updated vessels failure diagnosis.</b> SAM3 can recover the missed tiny vessel if the locator is very tight.  Therefore the next skill refinement is not another tissue classifier; it is an automatic tiny-vessel locator/proposal skill that produces tight boxes or points for missed vessel components.</div>
<p class='small'>Machine-readable output: <code>vessels_component_oracle_prompt_diagnostic/vessels_component1_oracle_prompt_metrics.csv</code>.</p>
"""
    append_or_replace(section)
    verify(HTML)
    verify(INDEX)
    pack.rebuild_zip()
    with zipfile.ZipFile(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip") as handle:
        bad = handle.testzip()
        if bad:
            raise RuntimeError(f"bad zip entry: {bad}")
    print(HTML)
    print(INDEX)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
