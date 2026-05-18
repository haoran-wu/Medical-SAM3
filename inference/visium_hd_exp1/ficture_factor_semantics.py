#!/usr/bin/env python3
"""FICTURE factor color/gene semantic helpers for official VisiumHD Exp1 runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


TARGET_LABELS = [
    "lung_bronchiola",
    "lung_alveoli_normal_adjacent",
    "lung_vessels",
    "tumor",
    "stroma",
    "immune_infiltration",
    "erythorocytes",
    "pigment",
]


# LLM/marker-gene interpretation of hex_12.k12.pixel.info.tsv.
# Keep this explicit and versionable: downstream ranking/VLM jobs should know
# exactly how each false-color factor was interpreted.
LLM_FACTOR_INTERPRETATIONS: Dict[int, dict] = {
    0: {
        "cell_type": "tumor-like epithelial / AT2-like malignant epithelial",
        "short_name": "tumor epithelial",
        "marker_basis": "SPINK1, CEACAM5, CEACAM6, NAPSA, SLC34A2, CXCL14",
        "target_scores": {
            "tumor": 0.85,
            "lung_alveoli_normal_adjacent": 0.30,
            "lung_bronchiola": 0.12,
            "immune_infiltration": 0.08,
        },
    },
    1: {
        "cell_type": "vascular smooth muscle / myofibroblast / vessel wall stroma",
        "short_name": "smooth muscle vessel wall",
        "marker_basis": "TAGLN, MYL9, MYH11, ACTA2, DES, COL1A1, COL3A1",
        "target_scores": {
            "lung_vessels": 1.00,
            "stroma": 0.80,
        },
    },
    2: {
        "cell_type": "epithelial tumor-like / mucinous-glandular epithelial",
        "short_name": "epithelial tumor-like",
        "marker_basis": "CEACAM5, MUC1, NKX2-1, SPP1, WSB1, LAMA5",
        "target_scores": {
            "tumor": 0.60,
            "lung_alveoli_normal_adjacent": 0.20,
            "immune_infiltration": 0.12,
        },
    },
    3: {
        "cell_type": "alveolar epithelial pneumocyte / AT2-like",
        "short_name": "alveolar epithelial",
        "marker_basis": "SFTPB, SFTPC, LPCAT1, SFTPA1, NPC2, NUPR1",
        "target_scores": {
            "lung_alveoli_normal_adjacent": 1.00,
            "lung_vessels": 0.12,
            "stroma": 0.10,
        },
    },
    4: {
        "cell_type": "lymphoid immune with alveolar epithelial admixture",
        "short_name": "immune/alveolar mixed",
        "marker_basis": "CXCR4, CCL19, IL7R, TRAC, MS4A1, SFTPC, SFTPA1",
        "target_scores": {
            "immune_infiltration": 0.85,
            "lung_alveoli_normal_adjacent": 0.45,
            "lung_bronchiola": 0.15,
        },
    },
    5: {
        "cell_type": "alveolar epithelial pneumocyte / AT1-AT2-like",
        "short_name": "alveolar epithelial",
        "marker_basis": "LPCAT1, PGGHG, ATP13A4, NKX2-1, SLC6A3, CXCL14",
        "target_scores": {
            "lung_alveoli_normal_adjacent": 0.80,
        },
    },
    6: {
        "cell_type": "SPP1/APOE macrophage",
        "short_name": "macrophage",
        "marker_basis": "SPP1, APOE, CD68, C1QA, C1QB, C1QC, LYZ, CHIT1",
        "target_scores": {
            "immune_infiltration": 0.75,
            "pigment": 0.35,
            "stroma": 0.10,
            "tumor": 0.08,
        },
    },
    7: {
        "cell_type": "bronchiolar secretory / club airway epithelium",
        "short_name": "bronchiolar secretory",
        "marker_basis": "SCGB1A1, SCGB3A1, BPIFB1, SLPI, MUC5B, TPPP3",
        "target_scores": {
            "lung_bronchiola": 1.00,
            "tumor": 0.12,
        },
    },
    8: {
        "cell_type": "plasma cell / B lineage",
        "short_name": "plasma cell",
        "marker_basis": "IGKC, IGHG1, IGLC1, IGHM, IGHA1, JCHAIN, CD79A",
        "target_scores": {
            "immune_infiltration": 0.90,
            "pigment": 0.10,
        },
    },
    9: {
        "cell_type": "pulmonary neuroendocrine / airway basal-like rare epithelial",
        "short_name": "neuroendocrine airway",
        "marker_basis": "GRP, CHGA, TPH1, KRT5, KRT17, CDK5R1",
        "target_scores": {
            "lung_bronchiola": 0.25,
            "tumor": 0.25,
        },
    },
    10: {
        "cell_type": "IgA plasma cell",
        "short_name": "IgA plasma cell",
        "marker_basis": "IGHA1, JCHAIN, IGKC, IGHG3, IGLC1, TXNDC5",
        "target_scores": {
            "immune_infiltration": 0.90,
            "lung_bronchiola": 0.10,
        },
    },
    11: {
        "cell_type": "IgM plasma/B cell",
        "short_name": "IgM plasma cell",
        "marker_basis": "IGHM, JCHAIN, IGKC, CD79A, MZB1, XBP1",
        "target_scores": {
            "immune_infiltration": 0.90,
        },
    },
}


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _split_genes(text: str, limit: int | None = None) -> List[str]:
    genes = [g.strip() for g in (text or "").split(",") if g.strip()]
    return genes if limit is None else genes[:limit]


def _parse_rgb(text: str) -> List[int]:
    return [int(float(x.strip())) for x in text.split(",")[:3]]


def build_semantic_legend(info_tsv: Path) -> dict:
    factors: List[dict] = []
    with info_tsv.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            factor = int(row["Factor"])
            interp = LLM_FACTOR_INTERPRETATIONS.get(
                factor,
                {
                    "cell_type": "unassigned FICTURE factor",
                    "short_name": "unassigned",
                    "marker_basis": row.get("TopGene_specific") or row.get("TopGene_pval") or "",
                    "target_scores": {},
                },
            )
            target_scores = {slug: float(interp.get("target_scores", {}).get(slug, 0.0)) for slug in TARGET_LABELS}
            factors.append(
                {
                    "factor": factor,
                    "rgb": _parse_rgb(row["RGB"]),
                    "rgb_text": row["RGB"],
                    "hex_color": "#" + "".join(f"{c:02X}" for c in _parse_rgb(row["RGB"])),
                    "weight": float(row.get("Weight") or 0.0),
                    "post_umi": int(float(row.get("PostUMI") or 0)),
                    "llm_inferred_cell_type": interp["cell_type"],
                    "short_name": interp["short_name"],
                    "marker_basis": interp["marker_basis"],
                    "top_genes_specific": _split_genes(row.get("TopGene_specific", ""), 20),
                    "top_genes_pval": _split_genes(row.get("TopGene_pval", ""), 20),
                    "target_scores": target_scores,
                }
            )
    factors.sort(key=lambda x: x["factor"])
    return {
        "source": str(info_tsv),
        "interpretation": "LLM/marker-gene inferred FICTURE false-color factor semantics for official filtered map.",
        "target_labels": TARGET_LABELS,
        "factors": factors,
    }


def load_semantic_legend(path: Path | None, n_factors: int | None = None) -> List[dict]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text())
    factors = sorted(data.get("factors", []), key=lambda item: int(item["factor"]))
    if n_factors is not None:
        present = {int(item["factor"]) for item in factors}
        for factor in range(n_factors):
            if factor not in present:
                factors.append(
                    {
                        "factor": factor,
                        "rgb": [0, 0, 0],
                        "rgb_text": "0,0,0",
                        "hex_color": "#000000",
                        "llm_inferred_cell_type": "unassigned FICTURE factor",
                        "short_name": "unassigned",
                        "marker_basis": "",
                        "target_scores": {slug: 0.0 for slug in TARGET_LABELS},
                    }
                )
        factors.sort(key=lambda item: int(item["factor"]))
    return factors


def factor_score_vectors(factors: Sequence[dict], labels: Iterable[str], n_factors: int) -> Dict[str, np.ndarray]:
    vectors = {slug: np.zeros(n_factors, dtype=np.float64) for slug in labels}
    for item in factors:
        factor = int(item["factor"])
        if factor < 0 or factor >= n_factors:
            continue
        scores = item.get("target_scores") or {}
        for slug, vec in vectors.items():
            vec[factor] = max(vec[factor], float(scores.get(slug, 0.0)))
    for slug, vec in vectors.items():
        if vec.max() > 0:
            vectors[slug] = vec / vec.max()
    return vectors


def label_factor_hints(factors: Sequence[dict], label: str, top_n: int = 4) -> str:
    ranked = [
        (
            float((item.get("target_scores") or {}).get(label, 0.0)),
            int(item["factor"]),
            item,
        )
        for item in factors
        if float((item.get("target_scores") or {}).get(label, 0.0)) > 0
    ]
    ranked.sort(reverse=True, key=lambda x: x[0])
    parts = []
    for score, factor, item in ranked[:top_n]:
        genes = ", ".join((item.get("top_genes_specific") or [])[:4])
        parts.append(
            f"factor {factor} RGB {item.get('rgb_text')} ({item.get('hex_color')}), "
            f"{item.get('short_name')} / {item.get('llm_inferred_cell_type')}, "
            f"score {score:.2f}, genes {genes or item.get('marker_basis', '')}"
        )
    return "; ".join(parts)


def legend_text(factors: Sequence[dict], compact: bool = True) -> str:
    lines = []
    for item in factors:
        genes = ", ".join((item.get("top_genes_specific") or [])[:5])
        line = (
            f"factor {item['factor']} RGB {item.get('rgb_text')} {item.get('hex_color')}: "
            f"{item.get('llm_inferred_cell_type')} (markers: {genes or item.get('marker_basis', '')})"
        )
        lines.append(line)
    if compact:
        return "\n".join(lines)
    return "\n".join(lines)


def factor_histogram_summary(hist: np.ndarray, factors: Sequence[dict], label: str | None = None, top_n: int = 5) -> str:
    by_factor = {int(item["factor"]): item for item in factors}
    pieces = []
    for factor in np.argsort(hist)[::-1][:top_n]:
        frac = float(hist[factor])
        if frac <= 0:
            continue
        item = by_factor.get(int(factor), {})
        score = ""
        if label is not None:
            semantic_score = float((item.get("target_scores") or {}).get(label, 0.0))
            score = f", target semantic score {semantic_score:.2f}"
        pieces.append(
            f"factor {int(factor)} {frac:.1%} RGB {item.get('rgb_text', 'NA')} "
            f"{item.get('short_name', 'unassigned')}{score}"
        )
    return "; ".join(pieces)


def semantic_clip_prompts(base_prompts: Dict[str, List[str]], factors: Sequence[dict]) -> Dict[str, List[str]]:
    prompts = {slug: list(texts) for slug, texts in base_prompts.items()}
    for slug in TARGET_LABELS:
        hint = label_factor_hints(factors, slug, top_n=3)
        if hint:
            prompts.setdefault(slug, []).extend(
                [
                    f"a FICTURE spatial transcriptomics factor-map crop for {slugify(slug).replace('_', ' ')} dominated by {hint}",
                    f"false-color FICTURE map regions matching {slugify(slug).replace('_', ' ')} factors: {hint}",
                ]
            )
    return prompts


def write_outputs(legend: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "factor_semantic_legend.json"
    csv_path = output_dir / "factor_semantic_legend.csv"
    md_path = output_dir / "factor_semantic_legend.md"
    json_path.write_text(json.dumps(legend, indent=2) + "\n")

    fields = [
        "factor",
        "rgb_text",
        "hex_color",
        "llm_inferred_cell_type",
        "short_name",
        "marker_basis",
        "target_scores_json",
        "top_genes_specific",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in legend["factors"]:
            writer.writerow(
                {
                    "factor": item["factor"],
                    "rgb_text": item["rgb_text"],
                    "hex_color": item["hex_color"],
                    "llm_inferred_cell_type": item["llm_inferred_cell_type"],
                    "short_name": item["short_name"],
                    "marker_basis": item["marker_basis"],
                    "target_scores_json": json.dumps(item["target_scores"], sort_keys=True),
                    "top_genes_specific": ", ".join(item["top_genes_specific"]),
                }
            )

    lines = [
        "# FICTURE Factor Semantic Legend",
        "",
        "LLM/marker-gene inferred meanings for the official filtered FICTURE false colors.",
        "",
        "| factor | RGB | color | inferred cell type | marker basis | main target support |",
        "|---:|---|---|---|---|---|",
    ]
    for item in legend["factors"]:
        top_targets = sorted(item["target_scores"].items(), key=lambda kv: kv[1], reverse=True)
        top_targets = [f"{k}:{v:.2f}" for k, v in top_targets if v > 0][:4]
        lines.append(
            f"| {item['factor']} | {item['rgb_text']} | {item['hex_color']} | "
            f"{item['llm_inferred_cell_type']} | {item['marker_basis']} | {', '.join(top_targets)} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default()
        row_h = 54
        width = 1600
        height = 46 + row_h * len(legend["factors"])
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((16, 12), "Official FICTURE factor semantic legend", fill="black", font=font)
        y = 42
        for item in legend["factors"]:
            rgb = tuple(int(x) for x in item["rgb"])
            draw.rectangle((16, y + 8, 52, y + 44), fill=rgb, outline="black")
            targets = sorted(item["target_scores"].items(), key=lambda kv: kv[1], reverse=True)
            target_text = ", ".join(f"{k}:{v:.2f}" for k, v in targets if v > 0)[:380]
            text = (
                f"F{item['factor']} RGB {item['rgb_text']} {item['hex_color']} | "
                f"{item['llm_inferred_cell_type']} | {item['marker_basis']} | {target_text}"
            )
            draw.text((64, y + 10), text, fill="black", font=font)
            y += row_h
        img.save(output_dir / "factor_semantic_legend.png")
    except Exception:
        pass
