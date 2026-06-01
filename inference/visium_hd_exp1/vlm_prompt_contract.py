#!/usr/bin/env python3
"""Shared VLM prompt contract for VisiumHD Exp1 paired H&E/FICTURE crops."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_BUNDLE = PROJECT_ROOT / "data/visium_hd_exp1/current_ficture_vlm_inputs"
DEFAULT_POOL_CSV = DEFAULT_INPUT_BUNDLE / "public_vlm_requests.csv"
DEFAULT_FACTOR_LEGEND_CSV = DEFAULT_INPUT_BUNDLE / "ficture_factor_legend_for_prompt.csv"


SYSTEM_PROMPT = (
    "You are a careful pathology image classifier. "
    "You will see two aligned crops of the same candidate region: H&E and FICTURE. "
    "Score which tissue class the candidate region most likely belongs to. "
    "Return only valid JSON."
)


def _parse_rgb(text: str) -> str:
    values = [int(value) for value in re.findall(r"\d+", text)[:3]]
    if len(values) != 3:
        raise ValueError(f"Could not parse RGB from {text!r}")
    return ",".join(str(value) for value in values)


def load_factor_legend_rows(path: Path = DEFAULT_FACTOR_LEGEND_CSV) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("Factor") or not row.get("RGB"):
                continue
            rows.append(
                {
                    "factor": int(row["Factor"]),
                    "rgb": _parse_rgb(row["RGB"]),
                    "major": row.get("Major Compartment", "").strip(),
                    "cell_type": (row.get("Celltype2") or row.get("Celltype") or "").strip(),
                }
            )
    rows.sort(key=lambda item: item["factor"])
    if [row["factor"] for row in rows] != list(range(12)):
        raise ValueError(f"Expected factors 0-11 in {path}, got {[row['factor'] for row in rows]}")
    missing = [row["factor"] for row in rows if not row["major"] or not row["cell_type"]]
    if missing:
        raise ValueError(f"Missing Major Compartment or cell type for factors {missing} in {path}")
    return rows


def build_ficture_legend_text(path: Path = DEFAULT_FACTOR_LEGEND_CSV) -> str:
    lines = []
    for row in load_factor_legend_rows(path):
        lines.append(
            "Color {factor}: RGB {rgb}; Major Compartment: {major}; cell type: {cell_type}.".format(**row)
        )
    return "\n".join(lines)


def build_user_prompt(path: Path = DEFAULT_FACTOR_LEGEND_CSV) -> str:
    legend = build_ficture_legend_text(path)
    return f"""You are given two aligned crops of the same candidate region:

Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: official FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

The FICTURE colors use this source-matched legend:
This color legend applies only to Image 2, the FICTURE crop. It does not apply to Image 1.
The pink and purple colors in H&E are normal tissue staining, not FICTURE tumor colors.
Major Compartment and cell type are marker-gene-inferred fields from source_matched_factor_info_with_llm_inferred_celltypes.html.

{legend}

Score how likely this candidate region belongs to each tissue class.

Tissue classes:
bronchiola = bronchiolar airway tissue, airway-like lumen, epithelial lining.
alveoli = alveolar lung parenchyma, open air spaces, thin septa.
vessels = blood vessel or vascular wall, lumen-like vascular structure, smooth muscle vessel wall.
tumor = malignant epithelial tumor region. Prefer tumor-like epithelial morphology and tumor epithelial FICTURE color support.
stroma = stromal / mesenchymal tissue, collagen, fibroblast, smooth-muscle-like tissue. Treat stroma as a broad tissue compartment.
immune_infiltration = immune-cell-rich region, small round-cell aggregates, macrophage / lymphoid / plasma-cell FICTURE color support.

Rules:
- Return exactly one JSON object.
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
- Each value must be an integer from 0 to 100.
- Higher means more likely.
- Use the full 0-100 range.
- Do not include explanation, reason, precision, recall, Dice, markdown, code fences, or extra text.
- Do not give all classes the same score unless there is truly no visible evidence."""
