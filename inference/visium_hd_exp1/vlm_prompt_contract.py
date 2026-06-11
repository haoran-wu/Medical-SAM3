#!/usr/bin/env python3
"""Shared VLM prompt contract for VisiumHD Exp1 paired H&E/FICTURE crops."""

from __future__ import annotations

import csv
import re
import os
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_BUNDLE = PROJECT_ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs"
DEFAULT_POOL_CSV = DEFAULT_INPUT_BUNDLE / "public_vlm_requests.csv"
DEFAULT_FACTOR_LEGEND_CSV = DEFAULT_INPUT_BUNDLE / "ficture_factor_prompt_legend_from_html.csv"


SYSTEM_PROMPT = (
    "You are a careful pathology image classifier. "
    "You will see two aligned crops of the same candidate region: H&E and FICTURE. "
    "Score which tissue class the candidate region most likely belongs to. "
    "Return only valid JSON."
)


def _parse_rgb_display(text: str) -> str:
    values = [int(value) for value in re.findall(r"\d+", text)[:3]]
    if len(values) != 3:
        raise ValueError(f"Could not parse RGB from {text!r}")
    return f"({values[0]}, {values[1]}, {values[2]})"


def load_factor_legend_rows(path: Path = DEFAULT_FACTOR_LEGEND_CSV) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("Factor") or not row.get("RGB"):
                continue
            rows.append(
                {
                    "factor": int(row["Factor"]),
                    "rgb": _parse_rgb_display(row["RGB"]),
                    "cell_type": (
                        row.get("cell type") or row.get("Celltype2") or row.get("Celltype") or ""
                    ).strip(),
                }
            )
    rows.sort(key=lambda item: item["factor"])
    if [row["factor"] for row in rows] != list(range(12)):
        raise ValueError(f"Expected factors 0-11 in {path}, got {[row['factor'] for row in rows]}")
    missing = [row["factor"] for row in rows if not row["cell_type"]]
    if missing:
        raise ValueError(f"Missing cell type for factors {missing} in {path}")
    return rows


def build_ficture_legend_text(path: Path = DEFAULT_FACTOR_LEGEND_CSV) -> str:
    lines = []
    for row in load_factor_legend_rows(path):
        lines.append(
            "Color {factor}: RGB {rgb}; cell type: {cell_type}.".format(**row)
        )
    return "\n".join(lines)


def build_compact_ficture_hint_text() -> str:
    return """Class-specific FICTURE color support for Image 2:
bronchiola: Color 7, RGB (0, 170, 170), airway epithelial cells.
alveoli: Color 3, RGB (255, 84, 0), alveolar type II cells; Color 5, RGB (84, 0, 255), alveolar epithelial substate.
vessels: Color 9, RGB (255, 0, 127), endothelial cells; Color 1, RGB (0, 255, 255), smooth-muscle / fibroblast tissue can support vessel wall but is not vessel-specific.
tumor: Color 0, RGB (255, 204, 255), lung adenocarcinoma / epithelial tumor cells; Color 2, RGB (255, 255, 0), epithelial tumor-like subpopulation.
stroma: Color 1, RGB (0, 255, 255), fibroblasts and smooth-muscle / mesenchymal tissue.
immune_infiltration: Color 4, RGB (0, 255, 84), lymphocytes; Color 6, RGB (170, 0, 170), macrophages / monocytes; Colors 8, 10, and 11, plasma-cell states."""


DETAILED_TISSUE_CLASSES = """Tissue classes:
bronchiola = bronchiolar airway tissue. Look for airway epithelium, bronchiolar wall, folded epithelial lining, or airway-like structure. A partial airway wall can count even if the full lumen is not visible.
alveoli = alveolar lung parenchyma. Look for open air spaces, thin septa, delicate alveolar architecture, or alveolar epithelial region. Do not call dense tumor nests alveoli.
vessels = blood vessel or vascular wall. Look for endothelial-lined lumen, smooth-muscle vessel wall, red-blood-cell space, or partial vascular wall. A small vessel-wall piece can count even if the full vessel is not visible.
tumor = malignant epithelial tumor region. Look for dense atypical epithelial nests, crowded tumor glands, solid malignant epithelial tissue, or invasive tumor-like architecture. Do not call normal airway or alveolar epithelium tumor only because it is epithelial.
stroma = stromal or mesenchymal tissue. Look for collagen-rich tissue, fibroblast-rich tissue, smooth-muscle-like tissue, or supporting connective tissue. Do not use stroma as a default label for every uncertain region.
immune_infiltration = immune-cell-rich tissue. Look for lymphocyte-rich, macrophage-rich, plasma-cell-rich, or small round-cell infiltrates. Patchy immune-cell pieces can count even if they are not one large aggregate."""


NO_DESCRIPTION_TISSUE_CLASSES = """Tissue classes:
bronchiola
alveoli
vessels
tumor
stroma
immune_infiltration"""


SHORT_TISSUE_CLASSES = """Tissue classes:
bronchiola = bronchiolar airway tissue or partial airway wall / epithelium.
alveoli = alveolar lung parenchyma with open air spaces and thin septa.
vessels = blood vessel, vascular wall, endothelial-lined lumen, or partial vessel wall.
tumor = malignant epithelial tumor region with dense atypical epithelial nests or glands.
stroma = collagen-rich, fibroblast-rich, smooth-muscle-like, or mesenchymal supporting tissue.
immune_infiltration = immune-cell-rich tissue, including lymphocyte, macrophage, or plasma-cell-rich pieces."""


def build_mainline_style_prompt(color_context: str, tissue_classes: str = DETAILED_TISSUE_CLASSES) -> str:
    return f"""You are given two aligned crops of the same candidate region:

Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

{color_context}

Score how likely this candidate region belongs to each tissue class.

{tissue_classes}

Rules:
- Return exactly one JSON object.
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
- Each value must be an integer from 0 to 100.
- Higher means more likely.
- Use the full 0-100 range.
- Do not include explanation, markdown, code fences, or extra text.
- Do not give all classes the same score unless there is truly no visible evidence.
"""


def build_piece_level_prompt(
    *,
    color_context: str,
    probability_style: bool = False,
    tissue_classes: str = DETAILED_TISSUE_CLASSES,
) -> str:
    probability_rules = (
        "- The six values should sum to about 100.\n"
        "- If uncertain, split probability across plausible classes instead of using only 0 and 100.\n"
        if probability_style
        else "- Higher means more likely.\n- Use different scores when the evidence differs.\n"
    )
    return f"""You are given two aligned crops of the same candidate region:

Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Score the visible candidate region only. It may be a small part of a larger tissue structure.
Use H&E morphology as the main evidence. Use FICTURE colors as supporting molecular / cell-type evidence.
Do not classify by color alone.

{color_context}

{tissue_classes}

Rules:
- Return exactly one JSON object.
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
- Each value must be an integer from 0 to 100.
{probability_rules}- Do not include explanation, markdown, code fences, or extra text.
- Do not give all classes the same score unless there is truly no visible evidence.
"""


def build_user_prompt(path: Path = DEFAULT_FACTOR_LEGEND_CSV) -> str:
    style = os.environ.get("VLM_PROMPT_STYLE", "").strip().lower()
    if style in {"mainline", "mainline_full", "previous_mainline"}:
        legend = build_ficture_legend_text(path)
        return build_mainline_style_prompt(f"FICTURE color legend for Image 2:\n\n{legend}")
    if style in {"mainline_nodesc", "mainline_no_description", "previous_mainline_nodesc"}:
        legend = build_ficture_legend_text(path)
        return build_mainline_style_prompt(
            f"FICTURE color legend for Image 2:\n\n{legend}",
            tissue_classes=NO_DESCRIPTION_TISSUE_CLASSES,
        )
    if style in {"compact", "mainline_compact", "compact_color_support"}:
        return build_mainline_style_prompt(build_compact_ficture_hint_text())
    if style in {"compact_short", "compact_short_description", "compact_color_support_short"}:
        return build_mainline_style_prompt(
            build_compact_ficture_hint_text(),
            tissue_classes=SHORT_TISSUE_CLASSES,
        )
    if style in {"compact_nodesc", "compact_no_description", "compact_color_support_nodesc"}:
        return build_mainline_style_prompt(
            build_compact_ficture_hint_text(),
            tissue_classes=NO_DESCRIPTION_TISSUE_CLASSES,
        )
    if style in {"piece_simple", "simple", "no_legend"}:
        return build_piece_level_prompt(color_context="No written FICTURE color legend is provided.", probability_style=False)
    if style in {"piece_prob", "probability", "prob"}:
        return build_piece_level_prompt(color_context="No written FICTURE color legend is provided.", probability_style=True)
    if style in {"piece_full_color_legend", "full_color_legend", "full_legend"}:
        legend = build_ficture_legend_text(path)
        return build_piece_level_prompt(
            color_context=f"FICTURE color legend for Image 2:\n\n{legend}",
            probability_style=False,
        )
    if style in {"piece_compact_color_hints", "compact_color_hints", "compact_legend"}:
        return build_piece_level_prompt(
            color_context=build_compact_ficture_hint_text(),
            probability_style=False,
        )
    legend = build_ficture_legend_text(path)
    return build_mainline_style_prompt(f"FICTURE color legend for Image 2:\n\n{legend}")
