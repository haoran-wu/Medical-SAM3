#!/usr/bin/env python3
"""Append FICTURE-local-locator diagnostic for tiny vessels component 1."""

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
OUT = BASE / "vessels_component_oracle_prompt_diagnostic/component1_ficture_local_locator"
SAM_OUT = BASE / "vessels_component_oracle_prompt_diagnostic/local_cpu_sam3_ficture_locator_component1_tight_box_m05"
ORACLE_OUT = BASE / "vessels_component_oracle_prompt_diagnostic/local_cpu_sam3_oracle_component1_tight_box_m05"
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
        "pixels": pa,
        "tp": tp,
    }


def append_or_replace(section: str) -> None:
    marker = "<h2>17BE. FICTURE Local Locator Diagnostic for Vessels Component 1</h2>"
    for path in [HTML, INDEX]:
        text = path.read_text()
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17B[F-Z]|<h2>18\.", text[start + len(marker) :])
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
    variants = [
        {
            "variant": "FICTURE local locator prior",
            "path": OUT / "component1_best_ficture_local_locator_mask.png",
            "meaning": "best local connected component from FICTURE vessel score near component 1",
        },
        {
            "variant": "FICTURE local locator + tight second-SAM",
            "path": SAM_OUT / "masks/vessels_ficture_local_locator_vessels_component1_second_sam.png",
            "meaning": "rough FICTURE locator used as a SAM3 box prompt",
        },
        {
            "variant": "oracle tight second-SAM",
            "path": ORACLE_OUT / "masks/vessels_oracle_vessels_component_1_second_sam.png",
            "meaning": "annotation-derived tight prompt upper bound",
        },
    ]
    rows: list[dict[str, object]] = []
    for variant in variants:
        mask = np.asarray(Image.open(variant["path"]).convert("L")) > 127
        m = metrics(mask, comp1)
        rows.append(
            {
                "variant": variant["variant"],
                "component-1 Dice / Precision / Recall": f"{m['dice']:.3f} / {m['precision']:.3f} / {m['recall']:.3f}",
                "pixels": m["pixels"],
                "meaning": variant["meaning"],
            }
        )
    write_csv(BASE / "vessels_component_oracle_prompt_diagnostic/component1_ficture_locator_vs_oracle_metrics.csv", rows)

    sweep_rows = []
    sweep_path = OUT / "component1_ficture_local_locator_sweep.csv"
    if sweep_path.exists():
        with sweep_path.open(newline="") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                if idx >= 8:
                    break
                sweep_rows.append(
                    {
                        "threshold": row["thresh"],
                        "close/open": f"{row['close']} / {row['open']}",
                        "component pixels": row["pixels"],
                        "D/P/R": f"{float(row['dice']):.3f} / {float(row['precision']):.3f} / {float(row['recall']):.3f}",
                    }
                )

    fig = SAM_OUT / "figures/vessels_second_sam_union.png"
    section = f"""
<h2>17BE. FICTURE Local Locator Diagnostic for Vessels Component 1</h2>
<p><b>Goal.</b> Section 17BD showed that a very tight oracle box lets SAM3 recover vessels component 1.  This section asks whether FICTURE can provide that locator automatically.</p>
<p><b>Method.</b> I inspected the official FICTURE vessel-score map around component 1.  The global semantic-vessel proposal dropped this tiny component because <code>keep_top=12</code> kept only large connected components.  Locally, factor 1 is present, but the connected component is still broader than the true tiny vessel.</p>
{table(rows, ["variant", "component-1 Dice / Precision / Recall", "pixels", "meaning"])}
<h3>Best local FICTURE locator candidates</h3>
{table(sweep_rows, ["threshold", "close/open", "component pixels", "D/P/R"])}
<figure><img src="{html.escape(str(fig.relative_to(BASE)))}" alt="FICTURE local locator second-SAM for vessels component 1"><figcaption>FICTURE rough locator plus tight second-SAM raises recall to 1.0 but keeps precision low.  FICTURE can find the neighborhood, but it does not yet provide the tight prompt needed for a clean tiny-vessel mask.</figcaption></figure>
<div class='callout'><b>Updated vessels refinement target.</b> The next skill should not be just a FICTURE threshold.  It needs a tiny-vessel locator that converts FICTURE vessel-support regions plus H&amp;E local morphology into tight boxes/points.  This is the missing subskill for vessels component 1.</div>
<p class='small'>Machine-readable output: <code>vessels_component_oracle_prompt_diagnostic/component1_ficture_locator_vs_oracle_metrics.csv</code>.</p>
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
