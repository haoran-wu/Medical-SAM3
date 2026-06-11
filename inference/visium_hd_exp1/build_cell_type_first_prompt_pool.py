#!/usr/bin/env python3
"""Build text-only cell-type-first prompts for the Jun09 piece pool.

This creates the Step 1 input for the proposed skill:

1. Use only FICTURE-derived RGB / cell-type composition text.
2. Ask an LLM for tissue-class hypotheses.
3. Do not show H&E or FICTURE images at this stage.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
DEFAULT_BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
DEFAULT_PUBLIC = DEFAULT_BASE / "corrected_pool/public_vlm_requests.csv"
DEFAULT_FEATURES = DEFAULT_BASE / "candidate_skill_features.csv"
DEFAULT_LEGEND = (
    ROOT
    / "data/visium_hd_exp1/pixel_cell_type_image/final_correct_FICTURE_map/"
    / "final_factor_color_gene_table_with_llm_inferred_celltypes.csv"
)
DEFAULT_OUT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_uid(value: str) -> str:
    return str(value).strip()


def factor_key(value: str) -> int:
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"Cannot parse factor id from {value!r}")
    return int(match.group(0))


def split_genes(value: str, n: int = 8) -> list[str]:
    genes = [gene.strip() for gene in str(value or "").replace(";", ",").split(",")]
    return [gene for gene in genes if gene][:n]


def parse_factor_composition_top(text: str) -> list[dict[str, object]]:
    """Parse strings like 'Color 1: 0.437; RGB (...); compartment: ...'."""
    parts = [part.strip() for part in str(text or "").split("|") if part.strip()]
    rows: list[dict[str, object]] = []
    for part in parts:
        m = re.search(r"Color\s+(\d+)\s*:\s*([0-9.]+)", part)
        if not m:
            continue
        factor = int(m.group(1))
        proportion = float(m.group(2))
        rows.append({"factor": factor, "proportion": proportion, "raw": part})
    rows.sort(key=lambda item: float(item["proportion"]), reverse=True)
    return rows


def build_cell_type_list(composition_text: str, legend_by_factor: dict[int, dict[str, str]]) -> str:
    lines: list[str] = []
    parsed = parse_factor_composition_top(composition_text)
    for rank, item in enumerate(parsed, start=1):
        factor = int(item["factor"])
        proportion = float(item["proportion"])
        legend = legend_by_factor.get(factor, {})
        lines.append(
            f"{rank}. RGB {legend.get('RGB', 'not available')}; "
            f"proportion {proportion:.3f}; "
            f"cell type: {legend.get('Celltype2', legend.get('Celltype', 'not available'))}"
        )
    if not lines:
        return "No reliable FICTURE cell-type composition was available for this candidate."
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a spatial transcriptomics and lung pathology reasoning assistant. "
    "You will receive only a FICTURE-derived RGB and cell-type composition list for one candidate mask. "
    "Generate biological tissue-class hypotheses only. Do not use or imagine any image."
)


USER_TEMPLATE = """You are given one candidate mask from a human lung cancer tissue section.

At this step, you must use only the FICTURE-derived cell-type / marker-gene composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Based only on this ordered cell-type and marker-gene list, decide which tissue classes are biologically plausible, uncertain, or unlikely.

Important rules:
- This is only Step 1 of a larger skill. It is a hypothesis step, not the final tissue decision.
- Do not directly map one cell type to one tissue class.
- A tissue class can contain multiple cell types.
- The same cell type can appear in multiple tissue classes.
- H&E morphology will be checked in Step 2, so do not invent lumen, septa, tumor nests, or immune aggregates here.

Tissue classes:
bronchiola = bronchiolar airway tissue.
alveoli = alveolar lung parenchyma.
vessels = blood vessel or vascular wall.
tumor = malignant epithelial tumor region.
stroma = stromal or mesenchymal tissue.
immune_infiltration = immune-cell-rich region.

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }},
  "main_ambiguity": "one short sentence",
  "composition_warning": "one short sentence, or none"
}}
"""

USER_TEMPLATE_CONSERVATIVE_HYPOTHESIS = """You are given one candidate mask from a human lung cancer tissue section.

At this step, use only the FICTURE-derived RGB / cell-type / marker-gene composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Generate a broad but useful hypothesis set for later H&E verification.
This is a screening step. It is better to mark a class as "uncertain" than to incorrectly mark it as "unlikely" when the composition could be a partial wall, boundary, or mixed interface.

Decision meanings:
- plausible = the composition directly supports this class.
- uncertain = the composition does not directly prove this class, but this class remains biologically possible and should be checked by H&E.
- unlikely = the composition clearly argues against this class.

Important rules:
- Do not directly map one cell type to one tissue class.
- A tissue class can contain multiple cell types.
- The same cell type can appear in multiple tissue classes.
- H&E morphology will be checked in Step 2, so do not invent lumen, septa, tumor nests, or immune aggregates here.
- For small candidate pieces, bronchiola and vessels can appear as partial epithelial/stromal/smooth-muscle wall or boundary pieces. If the composition is mixed epithelial + stromal / smooth-muscle-like, do not mark bronchiola or vessels as unlikely; use uncertain unless clearly contradicted.
- If endothelial-like signal is absent, vessels may still be uncertain when stromal / smooth-muscle-like composition is present, because a vessel wall can be smooth-muscle-rich.
- If airway-specific signal is absent, bronchiola may still be uncertain when epithelial + stromal composition is present, because a candidate can be a partial airway wall.
- Tumor / stroma / immune can be plausible when their composition is directly supported, but their presence should not automatically make bronchiola or vessels unlikely in a mixed boundary candidate.

Tissue classes:
bronchiola = bronchiolar airway tissue.
alveoli = alveolar lung parenchyma.
vessels = blood vessel or vascular wall.
tumor = malignant epithelial tumor region.
stroma = stromal or mesenchymal tissue.
immune_infiltration = immune-cell-rich region.

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }},
  "main_ambiguity": "one short sentence",
  "composition_warning": "one short sentence, or none"
}}
"""

USER_TEMPLATE_TARGETED_WALL_RESCUE = """You are given one candidate mask from a human lung cancer tissue section.

At this step, use only the FICTURE-derived RGB / cell-type / marker-gene composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Generate tissue-class hypotheses for later H&E verification.
This is a screening step, but it should still be selective.

Decision meanings:
- plausible = the composition directly supports this class.
- uncertain = the composition gives partial or indirect support and H&E should check it.
- unlikely = the composition gives little support and there is no special reason to keep it.

Important rules:
- Do not directly map one cell type to one tissue class.
- A tissue class can contain multiple cell types.
- The same cell type can appear in multiple tissue classes.
- Do not mark all six classes as plausible or uncertain unless the composition genuinely supports all six.
- H&E morphology will be checked in Step 2, so do not invent lumen, septa, tumor nests, or immune aggregates here.

Targeted rescue rule for small wall / boundary candidates:
- Bronchiola can be kept as uncertain when the composition is mixed epithelial + stromal / smooth-muscle-like, even if airway-specific markers are not dominant. A small airway-wall candidate may not look compositionally pure.
- Vessels can be kept as uncertain when stromal / smooth-muscle-like composition is present, even if endothelial signal is weak or absent. A vessel-wall candidate may be smooth-muscle-rich.
- This rescue applies only to bronchiola and vessels. Do not use it to make every other class uncertain.

Class-specific support:
- alveoli: keep if AT2 / alveolar epithelial markers or alveolar-like epithelial composition is present; otherwise do not keep just because the tissue is lung.
- tumor: keep if tumor epithelial composition is prominent.
- stroma: keep if fibroblast, collagen, smooth-muscle-like, or mesenchymal composition is prominent.
- immune_infiltration: keep if lymphocyte, macrophage, plasma-cell, or other immune composition is prominent.

Tissue classes:
bronchiola = bronchiolar airway tissue.
alveoli = alveolar lung parenchyma.
vessels = blood vessel or vascular wall.
tumor = malignant epithelial tumor region.
stroma = stromal or mesenchymal tissue.
immune_infiltration = immune-cell-rich region.

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }},
  "main_ambiguity": "one short sentence",
  "composition_warning": "one short sentence, or none"
}}
"""

USER_TEMPLATE_RGB_CELLTYPE_WALL_RESCUE = """You are given one candidate mask from a human lung cancer tissue section.

At this step, use only the FICTURE-derived RGB and cell-type composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Generate tissue-class hypotheses for later H&E verification.
This is a screening step, but it should still be selective.

Decision meanings:
- plausible = the RGB/cell-type composition directly supports this class.
- uncertain = the composition gives partial or indirect support and H&E should check it.
- unlikely = the composition gives little support and there is no special reason to keep it.

Important rules:
- Do not directly map one cell type to one tissue class.
- A tissue class can contain multiple cell types.
- The same cell type can appear in multiple tissue classes.
- Do not mark all six classes as plausible or uncertain unless the composition genuinely supports all six.
- H&E morphology will be checked in Step 2, so do not invent lumen, septa, tumor nests, or immune aggregates here.

Targeted rescue rule for small wall / boundary candidates:
- Bronchiola can be kept as uncertain when the composition is mixed epithelial plus stromal or smooth-muscle-like. A small airway-wall candidate may not look compositionally pure.
- Vessels can be kept as uncertain when stromal or smooth-muscle-like composition is present, even if endothelial-like signal is weak or absent. A vessel-wall candidate may be smooth-muscle-rich.
- This rescue applies only to bronchiola and vessels. Do not use it to make every other class uncertain.

Class-specific support:
- alveoli: keep if alveolar epithelial or AT2-like cell type is present; otherwise do not keep just because the tissue is lung.
- tumor: keep if tumor epithelial composition is prominent.
- stroma: keep if fibroblast, smooth-muscle-like, or mesenchymal composition is prominent.
- immune_infiltration: keep if lymphocyte, macrophage, plasma-cell, or other immune composition is prominent.

Tissue classes:
bronchiola = bronchiolar airway tissue.
alveoli = alveolar lung parenchyma.
vessels = blood vessel or vascular wall.
tumor = malignant epithelial tumor region.
stroma = stromal or mesenchymal tissue.
immune_infiltration = immune-cell-rich region.

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }},
  "main_ambiguity": "one short sentence",
  "composition_warning": "one short sentence, or none"
}}
"""

USER_TEMPLATE_RGB_CELLTYPE_ALVEOLI_RESCUE = USER_TEMPLATE_RGB_CELLTYPE_WALL_RESCUE.replace(
    "- alveoli: keep if alveolar epithelial or AT2-like cell type is present; otherwise do not keep just because the tissue is lung.",
    "- alveoli: keep as uncertain when any AT2-like, alveolar-like, or lung epithelial signal appears, even at low proportion, because an alveolar septal piece can be compositionally mixed. Keep as plausible only when alveolar/AT2-like signal is prominent. Do not keep alveoli when there is no alveolar-like or lung epithelial signal.",
)

USER_TEMPLATE_MINIMAL_STATUS_ONLY = """You are given one candidate mask from a human lung cancer tissue section.

Use only the FICTURE-derived RGB / cell-type / marker-gene composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Decide which tissue classes should remain possible for later H&E verification.

Use:
- plausible = directly supported by the composition.
- uncertain = possible but not proven; keep it for later H&E verification.
- unlikely = clearly unsupported by the composition.

Important:
- This is only a first-pass screen, not the final diagnosis.
- Do not directly map one cell type to one tissue class.
- Do not discuss contamination, artifacts, or cross-contamination.
- The marker genes describe the FICTURE color groups; use them only as supporting hints.

Tissue classes:
bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }}
}}
"""

USER_TEMPLATE_BALANCED_EDGE_SCREEN = """You are given one candidate mask from a human lung cancer tissue section.

Use only the FICTURE-derived RGB / cell-type / marker-gene composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Generate a selective hypothesis set for later H&E verification.
Keep a class as uncertain when the composition could plausibly represent a small piece of that structure.
Do not keep all classes by default.

Decision meanings:
- plausible = directly supported by the composition.
- uncertain = indirect support, mixed interface support, or small-piece support.
- unlikely = no meaningful support.

Small-piece rules:
- Bronchiola may appear as a partial airway wall. Keep bronchiola as uncertain when epithelial signal is mixed with stromal / smooth-muscle-like signal.
- Vessels may appear as a partial vessel wall. Keep vessels as uncertain when endothelial signal OR stromal / smooth-muscle-like signal is present.
- Alveoli may appear as thin septa or adjacent parenchyma. Keep alveoli as uncertain when AT2 / alveolar marker genes are present anywhere in the listed color groups, or when the profile is mixed epithelial + stromal with no dominant vessel/airway evidence.

Direct support rules:
- tumor: tumor epithelial composition supports tumor.
- stroma: fibroblast / smooth-muscle / collagen-like composition supports stroma.
- immune_infiltration: lymphocyte / macrophage / plasma-cell composition supports immune_infiltration.

Important:
- The same cell type can appear in multiple tissue contexts.
- H&E morphology will decide the final class in Step 2.
- Do not discuss contamination, artifacts, or cross-contamination.
- The marker genes describe FICTURE color groups; use them only as supporting hints.

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }}
}}
"""

USER_TEMPLATE_THRESHOLD_SCREEN = """You are given one candidate mask from a human lung cancer tissue section.

Use only the FICTURE-derived RGB / cell-type / marker-gene composition text below.
No H&E image is shown.
No FICTURE image is shown.
Do not infer visual morphology.

Candidate cell type list, ordered by proportion inside the candidate mask:
{cell_type_list}

Task:
Apply the screening rules below and decide which tissue classes should remain possible for later H&E verification.

General thresholds:
- If a class has strong direct composition support, mark it plausible.
- If a class has weak direct support or reasonable small-piece/interface support, mark it uncertain.
- If a class has neither direct nor interface support, mark it unlikely.

Class support:
- bronchiola: airway epithelial cell type or airway-like marker genes; also uncertain for mixed epithelial + stromal / smooth-muscle-like profiles that could be a partial airway wall.
- alveoli: AT2 / alveolar marker genes such as SFTPC, SFTPA1, SFTPB, NAPSA; also uncertain for mixed epithelial + stromal profiles that could be alveolar septa or adjacent lung parenchyma.
- vessels: endothelial cell type; also uncertain for stromal / smooth-muscle-like profiles that could be a vessel wall.
- tumor: tumor epithelial composition.
- stroma: fibroblast, collagen, smooth-muscle, or mesenchymal composition.
- immune_infiltration: lymphocyte, macrophage, plasma-cell, or other immune composition.

Important:
- This is a hypothesis filter only.
- Keep uncertain classes when they are biologically plausible for a small candidate piece.
- Do not keep all six classes unless the profile truly supports all six.
- Do not discuss contamination, artifacts, or cross-contamination.
- The marker genes describe FICTURE color groups; use them only as supporting hints.

Return exactly one valid JSON object:
{{
  "hypotheses": {{
    "bronchiola": "plausible|uncertain|unlikely",
    "alveoli": "plausible|uncertain|unlikely",
    "vessels": "plausible|uncertain|unlikely",
    "tumor": "plausible|uncertain|unlikely",
    "stroma": "plausible|uncertain|unlikely",
    "immune_infiltration": "plausible|uncertain|unlikely"
  }}
}}
"""

STEP2_HE_TEMPLATE = """You are now in Step 2 of the tissue candidate judge skill.

Step 1 used only the FICTURE-derived RGB / cell-type list and produced biological hypotheses:
{step1_hypothesis_json}

Now inspect the H&E images for the same candidate mask.

Available images:
Image 1: H&E candidate crop. The candidate mask is the region being judged.
Image 2: H&E local context crop. Use it to see whether the candidate is part of a larger airway, vessel, alveolar region, tumor nest, stroma, or immune aggregate.

Task:
Use H&E morphology and local context to verify, reject, or revise the Step 1 hypotheses.

Important rules:
- Judge the candidate mask only, not the whole image.
- H&E morphology is the primary evidence for tissue structure.
- Step 1 RGB/cell-type hypotheses can be useful, but they can be wrong or ambiguous.
- Do not let cell-type composition override clear H&E morphology.

Return exactly one valid JSON object:
{{
  "he_supported_classes": ["class1", "class2"],
  "he_rejected_classes": ["class3"],
  "he_main_evidence": "one short sentence",
  "he_uncertainty": "one short sentence, or none"
}}
"""


STEP3_FICTURE_IMAGE_TEMPLATE = """You are now in Step 3 of the tissue candidate judge skill.

Step 1 RGB / cell-type hypotheses:
{step1_hypothesis_json}

Step 2 H&E verification:
{step2_he_json}

Now inspect the FICTURE image for the same candidate mask.

Available image:
Image 1: official FICTURE factor-color crop or local context crop for the candidate mask.

Task:
Use the FICTURE image only as a spatial consistency check.

Important rules:
- Do not classify the tissue by raw colors alone.
- Check whether the candidate mask lies in the expected FICTURE-colored region.
- Check whether the mask crosses multiple unrelated FICTURE compartments.
- Check whether the FICTURE image agrees with the Step 1 cell-type list.
- If FICTURE conflicts with H&E morphology, report the conflict instead of forcing a class.

Return exactly one valid JSON object:
{{
  "ficture_spatial_check": "consistent|mixed|conflicting|unclear",
  "ficture_evidence": "one short sentence",
  "final_scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }}
}}
"""


def build_html(out_dir: Path, rows: list[dict[str, object]]) -> None:
    preview = rows[:24]

    def table(items: list[dict[str, object]], cols: list[str]) -> str:
        out = ["<table><thead><tr>"]
        out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
        out.append("</tr></thead><tbody>")
        for row in items:
            out.append("<tr>")
            for col in cols:
                value = str(row.get(col, ""))
                if col == "cell_type_list":
                    value = "<pre>" + html.escape(value) + "</pre>"
                else:
                    value = html.escape(value)
                out.append(f"<td>{value}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Jun10 Cell-Type-First Skill Input Pack</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:30px; color:#20242a; line-height:1.55; }}
table {{ border-collapse:collapse; width:100%; margin:18px 0; }}
th,td {{ border-bottom:1px solid #ddd; padding:8px 10px; text-align:left; vertical-align:top; }}
pre {{ white-space:pre-wrap; margin:0; font-size:12px; line-height:1.4; }}
.note {{ max-width:1100px; }}
code {{ background:#f4f4f4; padding:1px 4px; border-radius:4px; }}
</style>
</head>
<body>
<h1>Jun10 Cell-Type-First Tissue Judge Skill: Step 1 Input Pack</h1>
<div class="note">
<p><b>Goal:</b> Before showing H&amp;E or FICTURE images, convert each candidate mask into an ordered FICTURE-derived RGB / cell-type list and ask Qwen3 for biological tissue-class hypotheses.</p>
<p><b>Why:</b> This follows the LLMiniST-style idea: translate spatial transcriptomics information into a text profile first, instead of asking a vision model to interpret raw false-color FICTURE images.</p>
<p><b>What the model sees in Step 1:</b> only text and numbers. No H&amp;E image, no FICTURE image, no hidden annotation, no Dice, no Precision, no Recall.</p>
<p><b>Important:</b> Step 1 is not final classification. H&amp;E morphology is checked in Step 2, and FICTURE image spatial consistency is checked in Step 3.</p>
<p><b>Rows:</b> {len(rows)} candidates from the Jun09 corrected 167-piece pool.</p>
<p><b>Prompt style:</b> {html.escape(str(rows[0].get("prompt_style", "baseline")) if rows else "baseline")}</p>
</div>

<h2>Prompt Template</h2>
<pre>{html.escape(str(rows[0].get("system_prompt", SYSTEM_PROMPT)) + "\\n\\n" + str(rows[0].get("user_prompt_template", USER_TEMPLATE)))}</pre>

<h2>Step 2 H&amp;E Verification Template</h2>
<pre>{html.escape(STEP2_HE_TEMPLATE)}</pre>

<h2>Step 3 FICTURE Image Spatial Check Template</h2>
<pre>{html.escape(STEP3_FICTURE_IMAGE_TEMPLATE)}</pre>

<h2>First 24 Candidate Text Profiles</h2>
{table(preview, ["row_index", "candidate_uid", "true_class", "cell_type_list"])}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-csv", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--legend-csv", type=Path, default=DEFAULT_LEGEND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--prompt-style",
        choices=[
            "baseline",
            "conservative_hypothesis",
            "targeted_wall_rescue",
            "rgb_celltype_wall_rescue",
            "rgb_celltype_alveoli_rescue",
            "minimal_status_only",
            "balanced_edge_screen",
            "threshold_screen",
        ],
        default="baseline",
    )
    args = parser.parse_args()

    public = read_csv(args.public_csv)
    features = read_csv(args.features_csv)
    legend_rows = read_csv(args.legend_csv)

    feature_by_uid = {normalize_uid(row["candidate_uid"]): row for row in features}
    legend_by_factor = {factor_key(row["Factor"]): row for row in legend_rows}

    rows: list[dict[str, object]] = []
    if args.prompt_style == "conservative_hypothesis":
        user_template = USER_TEMPLATE_CONSERVATIVE_HYPOTHESIS
    elif args.prompt_style == "targeted_wall_rescue":
        user_template = USER_TEMPLATE_TARGETED_WALL_RESCUE
    elif args.prompt_style == "rgb_celltype_wall_rescue":
        user_template = USER_TEMPLATE_RGB_CELLTYPE_WALL_RESCUE
    elif args.prompt_style == "rgb_celltype_alveoli_rescue":
        user_template = USER_TEMPLATE_RGB_CELLTYPE_ALVEOLI_RESCUE
    elif args.prompt_style == "minimal_status_only":
        user_template = USER_TEMPLATE_MINIMAL_STATUS_ONLY
    elif args.prompt_style == "balanced_edge_screen":
        user_template = USER_TEMPLATE_BALANCED_EDGE_SCREEN
    elif args.prompt_style == "threshold_screen":
        user_template = USER_TEMPLATE_THRESHOLD_SCREEN
    else:
        user_template = USER_TEMPLATE
    for idx, row in enumerate(public):
        uid = normalize_uid(row["candidate_uid"])
        feature = feature_by_uid.get(uid)
        if feature is None:
            raise KeyError(f"Missing candidate_skill_features row for candidate_uid={uid}")
        cell_type_list = build_cell_type_list(feature.get("factor_composition_top", ""), legend_by_factor)
        user_prompt = user_template.format(cell_type_list=cell_type_list)
        rows.append(
            {
                "row_index": idx,
                "candidate_uid": uid,
                "label": row.get("label", ""),
                "display": row.get("display", ""),
                "true_class": feature.get("true_class", row.get("display", "")),
                "source": row.get("source", ""),
                "run": row.get("run", ""),
                "setting": row.get("setting", ""),
                "candidate_id": row.get("candidate_id", ""),
                "mask_path": row.get("mask_path", ""),
                "he_crop_rel": row.get("he_crop_rel", ""),
                "ficture_crop_rel": row.get("ficture_crop_rel", ""),
                "component_dice": feature.get("component_dice", ""),
                "component_precision": feature.get("component_precision", ""),
                "component_recall": feature.get("component_recall", ""),
                "factor_composition_top": "",
                "cell_type_list": cell_type_list,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt_template": user_template,
                "user_prompt": user_prompt,
                "prompt_style": args.prompt_style,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_type_first_requests.csv", rows)
    (args.output_dir / "prompt_step1_system.txt").write_text(SYSTEM_PROMPT + "\n")
    (args.output_dir / "prompt_step1_user_template.txt").write_text(user_template)
    (args.output_dir / "prompt_step2_he_verification_template.txt").write_text(STEP2_HE_TEMPLATE)
    (args.output_dir / "prompt_step3_ficture_image_check_template.txt").write_text(STEP3_FICTURE_IMAGE_TEMPLATE)
    (args.output_dir / "prompt_step1_first_row.txt").write_text(
        rows[0]["system_prompt"] + "\n\n" + rows[0]["user_prompt"]
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "public_csv": str(args.public_csv),
                "features_csv": str(args.features_csv),
                "legend_csv": str(args.legend_csv),
                "n_candidates": len(rows),
                "input_rule": "Step 1 uses FICTURE-derived RGB / cell-type text only; no images.",
            },
            indent=2,
        )
    )
    build_html(args.output_dir, rows)
    print(args.output_dir)


if __name__ == "__main__":
    main()
