#!/usr/bin/env python3
"""Local VLM six-class scoring for multi-image piece-first inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List

from PIL import Image

from run_local_vlm_cross_label_top1_accuracy import (
    CLASS_KEYS,
    TRUE_LABEL_TO_CLASS,
    parse_scores,
    predicted_from_scores,
    row_key_string,
    write_accuracy_tables,
    write_same_class_retrieval_tables,
)
from run_paired_vlm_hit_test import generate, load_vlm, write_csv
from vlm_prompt_contract import (
    DEFAULT_FACTOR_LEGEND_CSV,
    DETAILED_TISSUE_CLASSES,
    NO_DESCRIPTION_TISSUE_CLASSES,
    SHORT_TISSUE_CLASSES,
    build_ficture_legend_text,
)


SYSTEM_PROMPT = (
    "You are a careful pathology image classifier. "
    "You will see multiple aligned images for the same candidate tissue piece. "
    "Score the candidate tissue piece only and return only valid JSON."
)


def context_image_mode() -> str:
    return os.environ.get("CONTEXT_IMAGE_MODE", "all5").strip().lower()


def build_image_description() -> str:
    mode = context_image_mode()
    if mode == "all6":
        return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: official FICTURE factor-color reverse-blur close-up for the same candidate piece.
Image 3: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure.
Image 4: official FICTURE local context crop. The same candidate piece is outlined so you can compare local FICTURE colors.
Image 5: low-resolution H&E ROI locator. The red box shows where this candidate is located in the whole ROI.
Image 6: low-resolution FICTURE ROI locator. The red box shows where this candidate is located in the whole FICTURE ROI."""
    if mode == "he_only":
        return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure.
Image 3: low-resolution H&E ROI locator. The yellow box shows where this candidate is located in the whole ROI."""
    if mode == "he_piece_local":
        return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure."""
    if mode == "ficture_piece_local":
        return """Available images:
Image 1: official FICTURE factor-color reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: official FICTURE local context crop. The same candidate piece is outlined so you can compare local FICTURE colors."""
    if mode == "piece_local":
        return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: FICTURE gray reverse-blur close-up. The same candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 3: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure.
Image 4: FICTURE local context crop. The same candidate piece is outlined so you can compare local FICTURE colors."""
    if mode == "no_ficture_closeup":
        return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure.
Image 3: FICTURE local context crop. The same candidate piece is outlined so you can compare local FICTURE colors.
Image 4: low-resolution H&E ROI locator. The yellow box shows where this candidate is located in the whole ROI."""
    if mode == "piece_only":
        return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: FICTURE gray reverse-blur close-up. The same candidate piece is sharp and full color; outside the piece is grayscale and blurred."""
    if mode == "local_only":
        return """Available images:
Image 1: H&E local context crop. The candidate piece is outlined so you can see nearby tissue structure.
Image 2: FICTURE local context crop. The same candidate piece is outlined so you can compare local FICTURE colors."""
    return """Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: FICTURE gray reverse-blur close-up. The same candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 3: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure.
Image 4: FICTURE local context crop. The same candidate piece is outlined so you can compare local FICTURE colors.
Image 5: low-resolution H&E ROI locator. The yellow box shows where this candidate is located in the whole ROI."""


def build_context_prompt(factor_legend_csv: Path) -> str:
    style = os.environ.get("CONTEXT_PROMPT_STYLE", "full").strip().lower()
    if style in {
        "he_semantic_ficture_map",
        "semantic_ficture_map",
        "he_semantic_ficture_position",
    }:
        return """You are given aligned images for one highlighted candidate tissue piece from a human lung cancer tissue section.

Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
Image 2: simplified semantic FICTURE map for the same close-up crop. It preserves where FICTURE-derived cell-type signals occur inside the candidate crop, but it is not the raw FICTURE color image.
Image 3: H&E local context crop. The same candidate piece is outlined so you can see nearby tissue structure.
Image 4: simplified semantic FICTURE map for the same local context crop. It preserves where FICTURE-derived cell-type signals occur around the candidate, but it is not the raw FICTURE color image.
Image 5: low-resolution H&E ROI locator. The yellow box shows where this candidate is located in the whole ROI.

Use H&E morphology and H&E context as the primary evidence.
Use the semantic FICTURE maps only as spatial cell-type support: they show where broad cell-type signals are located relative to the highlighted candidate.
Do not treat semantic FICTURE colors as direct tissue-class labels.
Do not call tumor from epithelial support alone; tumor should require H&E tumor morphology.

{SEMANTIC_FICTURE_LEGEND_TEXT}

Tissue classes:
bronchiola = bronchiolar airway tissue, airway-like lumen, epithelial lining.
alveoli = alveolar lung parenchyma, open air spaces, thin septa.
vessels = blood vessel or vascular wall, lumen-like vascular structure, smooth muscle vessel wall.
tumor = malignant epithelial tumor region, dense tumor nests or malignant glands.
stroma = stromal / mesenchymal tissue, collagen, fibroblast, smooth-muscle-like tissue.
immune_infiltration = immune-cell-rich region, small round-cell aggregates, macrophage / lymphoid / plasma-cell rich tissue.

Score how likely this highlighted candidate piece belongs to each tissue class.

Rules:
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
"""
    if style in {
        "he_located_ficture_text",
        "he_ficture_text",
        "no_ficture_image_structured_text",
        "he_located_ficture_text_vessel_guard",
        "he_ficture_grid_vessel_guard",
    }:
        vessel_guard = ""
        if style in {"he_located_ficture_text_vessel_guard", "he_ficture_grid_vessel_guard"}:
            vessel_guard = """
Important vessel-vs-stroma rule:
- A vessel candidate may be only a small wall fragment, not the whole vessel.
- Vessel wall can look stromal or smooth-muscle-like on H&E.
- If the local H&E context shows a lumen-like vascular structure, or the FICTURE-derived grid shows endothelial / vascular signal in or next to this candidate, score vessels high even if stroma is also plausible.
- Do not let broad stromal appearance override clear vascular-wall context.
"""
        return """You are given H&E image input for one highlighted candidate tissue piece from a human lung cancer tissue section.

Available images:
Image 1: H&E gray reverse-blur close-up. The candidate piece is sharp and full color; outside the piece is grayscale and blurred.
If additional H&E context images are provided, they show the same candidate in a larger local context or whole-ROI locator view.

No FICTURE color image is shown. Instead, structured FICTURE-derived cell-type information is provided below for this exact highlighted candidate mask and its nearby context.

Use H&E morphology and H&E location/context as the primary evidence.
Use the FICTURE-derived text only as molecular/cell-type supporting evidence.
Do not imagine or infer a FICTURE color image.
Do not call tumor from epithelial/tumor-like cell-type support alone; tumor should require H&E tumor morphology.
{VESSEL_GUARD}

Tissue classes:
bronchiola = bronchiolar airway tissue, airway-like lumen, epithelial lining.
alveoli = alveolar lung parenchyma, open air spaces, thin septa.
vessels = blood vessel or vascular wall, lumen-like vascular structure, smooth muscle vessel wall.
tumor = malignant epithelial tumor region, dense tumor nests or malignant glands.
stroma = stromal / mesenchymal tissue, collagen, fibroblast, smooth-muscle-like tissue.
immune_infiltration = immune-cell-rich region, small round-cell aggregates, macrophage / lymphoid / plasma-cell rich tissue.

{FICTURE_SUMMARY_TEXT}

Score how likely this highlighted candidate piece belongs to each tissue class.

Rules:
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
""".replace("{VESSEL_GUARD}", vessel_guard)
    use_legend = not style.startswith("no_legend")
    legend = build_ficture_legend_text(factor_legend_csv) if use_legend else ""
    if style in {"short", "concise", "no_legend_short", "no_legend_concise"}:
        tissue_classes = SHORT_TISSUE_CLASSES
    elif style in {"nodesc", "no_description", "names_only", "no_legend_nodesc", "no_legend_no_description"}:
        tissue_classes = NO_DESCRIPTION_TISSUE_CLASSES
    elif style in {
        "morphology_first",
        "no_legend_morphology",
        "no_legend_morphology_first",
        "structure_first",
        "no_legend_structure_first",
        "differential",
        "no_legend_differential",
        "exclusive",
        "no_legend_exclusive",
        "anti_tumor_strict",
        "no_legend_anti_tumor_strict",
        "lumen_first",
        "no_legend_lumen_first",
        "he_primary",
        "no_legend_he_primary",
        "partial_piece",
        "no_legend_partial_piece",
        "score_rubric",
        "no_legend_score_rubric",
        "forced_choice",
        "no_legend_forced_choice",
        "broad_group_first",
        "no_legend_broad_group_first",
        "concise_classifier",
        "no_legend_concise_classifier",
        "tumor_gate",
        "no_legend_tumor_gate",
        "stroma_guard",
        "no_legend_stroma_guard",
        "alveoli_stroma_guard",
        "no_legend_alveoli_stroma_guard",
        "tumor_not_default",
        "no_legend_tumor_not_default",
    }:
        tissue_classes = DETAILED_TISSUE_CLASSES
    elif style in {
        "forced_choice_short",
        "no_legend_forced_choice_short",
        "broad_group_short",
        "no_legend_broad_group_short",
    }:
        tissue_classes = SHORT_TISSUE_CLASSES
    elif style in {
        "forced_nodesc",
        "no_legend_forced_nodesc",
        "broad_group_nodesc",
        "no_legend_broad_group_nodesc",
        "primary_nodesc",
        "no_legend_primary_nodesc",
    }:
        tissue_classes = NO_DESCRIPTION_TISSUE_CLASSES
    else:
        tissue_classes = DETAILED_TISSUE_CLASSES
    legend_block = (
        f"FICTURE color legend for Image 2 and Image 4:\n{legend}"
        if use_legend
        else "No written FICTURE color legend is provided. Use FICTURE as visual spatial context only; do not infer a class from a written color mapping."
    )
    if style in {"guarded_compact", "compact_guarded", "compact_legend_guarded"}:
        use_legend = True
        legend_block = """Compact FICTURE support for Image 2 and Image 4:
- Airway/bronchiola support: airway epithelial colors, especially airway-like epithelium near a lumen.
- Alveoli support: alveolar epithelial / open-space parenchyma colors, but only if H&E also shows thin septa or air-space structure.
- Vessel support: endothelial or smooth-muscle/stromal colors around a lumen-like vascular structure.
- Tumor support: tumor-like epithelial colors only when H&E shows dense malignant nests, crowded tumor glands, or invasive tumor architecture.
- Stroma support: fibroblast, collagen, smooth muscle, or mesenchymal colors, especially when H&E shows connective tissue.
- Immune support: lymphocyte, macrophage, plasma-cell, or small-round-cell rich colors.
Important: epithelial color alone is not enough to call tumor. Normal airway and alveolar epithelium can also be epithelial."""
        tissue_classes = SHORT_TISSUE_CLASSES

    style_notes = ""
    if style in {"morphology_first", "no_legend_morphology", "no_legend_morphology_first"}:
        style_notes = """Decision style:
- Use H&E morphology as the main evidence.
- Use FICTURE only as weak spatial support.
- Do not call tumor unless H&E morphology supports malignant tumor architecture.
- A small partial wall can still be bronchiola or vessel if the local context shows airway or vascular structure."""
    elif style in {"structure_first", "no_legend_structure_first"}:
        style_notes = """Decision style:
- First decide whether the local context contains a lumen, airway wall, vessel wall, alveolar air spaces, dense tumor nest, stroma, or immune aggregate.
- Then score the candidate piece as part of that surrounding structure.
- The candidate mask may cover only a small segment of the larger structure."""
    elif style in {"differential", "no_legend_differential"}:
        style_notes = """Decision style:
- Mentally compare the two most plausible classes before scoring.
- Separate normal epithelium from tumor by H&E architecture, not by epithelial color alone.
- Separate vessel wall from stroma by lumen-like vascular structure and wall shape.
- Separate immune infiltration from tumor/stroma by small round-cell aggregate appearance."""
    elif style in {"exclusive", "no_legend_exclusive"}:
        style_notes = """Decision style:
- Choose one primary tissue class for the candidate piece.
- Give the primary class the highest score.
- Give close alternatives moderate scores only if genuinely ambiguous.
- Avoid selecting the same candidate for many unrelated classes."""
    elif style in {"anti_tumor_strict", "no_legend_anti_tumor_strict"}:
        style_notes = """Decision style:
- Be strict about tumor. Do not call a candidate tumor just because it is epithelial or dense.
- Call tumor only when H&E shows malignant tumor architecture such as crowded invasive nests, malignant glands, or solid tumor sheets.
- If local context shows airway lumen, vessel lumen, or alveolar air spaces, prefer bronchiola, vessels, or alveoli over tumor."""
    elif style in {"lumen_first", "no_legend_lumen_first"}:
        style_notes = """Decision style:
- First look for lumen or open-space context.
- Airway-like lumen plus epithelial lining supports bronchiola.
- Round/elongated lumen with vessel wall supports vessels.
- Many open air spaces with thin septa supports alveoli.
- Only after ruling out these normal structures should tumor, stroma, or immune be the primary class."""
    elif style in {"he_primary", "no_legend_he_primary"}:
        style_notes = """Decision style:
- Treat H&E morphology and local context as decisive.
- Treat FICTURE as optional supporting evidence, not as a direct label map.
- If H&E and FICTURE disagree, follow the H&E tissue structure."""
    elif style in {"partial_piece", "no_legend_partial_piece"}:
        style_notes = """Decision style:
- The candidate mask may be only a fragment of a larger tissue structure.
- A small wall fragment can still be bronchiola or vessel if the local context shows the larger airway or vascular structure.
- Score the tissue identity of the fragment, not whether the fragment covers the entire structure."""
    elif style in {"score_rubric", "no_legend_score_rubric"}:
        style_notes = """Decision style:
- Internally weigh three evidence sources: candidate texture, surrounding H&E structure, and FICTURE support if available.
- Penalize classes that fit only one evidence source while contradicting the surrounding structure.
- Use calibrated scores: one clear class should be high, plausible alternatives moderate, unlikely classes low."""
    elif style in {"forced_choice", "no_legend_forced_choice", "forced_choice_short", "no_legend_forced_choice_short"}:
        style_notes = """Decision style:
- Internally choose exactly one primary tissue class before assigning scores.
- The primary class should usually receive a high score, at least 80.
- Non-primary classes should usually be below 40 unless the candidate is genuinely ambiguous.
- Do not let one visual cue, such as density or epithelial appearance, override the surrounding H&E structure."""
    elif style in {"forced_nodesc", "no_legend_forced_nodesc", "primary_nodesc", "no_legend_primary_nodesc"}:
        style_notes = """Decision style:
- Internally choose exactly one primary class from the six class names.
- The primary class should receive a high score, usually at least 80.
- Non-primary classes should usually be below 35 unless genuinely ambiguous.
- Base the choice on the visible candidate piece and its local context, not on memorized class descriptions."""
    elif style in {"broad_group_first", "no_legend_broad_group_first", "broad_group_short", "no_legend_broad_group_short"}:
        style_notes = """Decision style:
- First choose the broad group: airway, alveolar parenchyma, vascular structure, tumor, stroma, or immune aggregate.
- Then map that broad group to one of the six tissue classes.
- A small piece should be classified by the larger local structure it belongs to."""
    elif style in {"broad_group_nodesc", "no_legend_broad_group_nodesc"}:
        style_notes = """Decision style:
- First choose the broad visual group using only image evidence: airway-like structure, air-space parenchyma, vascular structure, dense epithelial mass, connective tissue, or small-cell aggregate.
- Then map the broad visual group to the closest of the six class names.
- The candidate may be a small fragment of a larger structure."""
    elif style in {"concise_classifier", "no_legend_concise_classifier"}:
        style_notes = """Decision style:
- Act like a strict image classifier, not a report writer.
- Use local H&E structure first, candidate appearance second, and FICTURE only as support if present.
- Give a decisive high score to the best class and low scores to unlikely classes."""
    elif style in {"tumor_gate", "no_legend_tumor_gate"}:
        style_notes = """Decision style:
- Before scoring tumor, ask whether the candidate is better explained by normal airway, alveoli, vessel, stroma, or immune tissue.
- Score tumor high only if normal tissue structures do not explain the candidate and H&E supports malignant architecture.
- If tumor and immune are both plausible, use small round-cell aggregate appearance for immune and epithelial nest/gland architecture for tumor."""
    elif style in {"stroma_guard", "no_legend_stroma_guard"}:
        style_notes = """Decision style:
- Be careful not to overcall tumor.
- Stroma means connective or mesenchymal tissue: collagen-like pink matrix, fibroblast-rich tissue, smooth-muscle-like tissue, or supportive tissue around glands, vessels, and tumor.
- A stroma candidate can be broad or irregular; it does not need a lumen.
- Call tumor only when the candidate itself shows malignant epithelial nests, malignant glands, or solid tumor sheets.
- If the candidate is mostly supportive connective tissue around other structures, score stroma higher than tumor."""
    elif style in {"alveoli_stroma_guard", "no_legend_alveoli_stroma_guard"}:
        style_notes = """Decision style:
- First separate normal lung structure from tumor.
- Alveoli means normal lung parenchyma: open air spaces with thin septa; the candidate may cover septal tissue around empty spaces, not the empty space itself.
- Stroma means collagen/fibroblast/smooth-muscle-like supportive tissue, often broader and denser than alveolar septa.
- Do not call tumor just because the region is pink, epithelial-looking, or dense.
- Tumor should require malignant epithelial architecture in the candidate or immediately surrounding context."""
    elif style in {"tumor_not_default", "no_legend_tumor_not_default"}:
        style_notes = """Decision style:
- Tumor is not the default class.
- Before scoring tumor high, rule out airway wall, vessel wall, alveolar parenchyma, stromal connective tissue, and immune aggregate.
- Dense pink H&E alone is insufficient for tumor.
- If the candidate is a small fragment, classify it by the larger local structure that contains it.
- Use low tumor scores for fragments that look like vessel wall, airway wall, stroma, immune aggregate, or alveolar septa."""
    image_description = build_image_description()
    mode = context_image_mode()
    if mode == "piece_only":
        image_use_notes = """Use Image 1 to inspect H&E morphology inside the highlighted candidate piece.
Use Image 2 only as FICTURE-derived cell-type color support for the same candidate piece.
Only these two close-up images are provided in this run."""
        legend_block = legend_block.replace("Image 2 and Image 4", "Image 2")
    else:
        image_use_notes = """Use the close-up images to inspect the candidate piece.
Use the local context images to understand whether a small piece is part of an airway, vessel, alveolar region, tumor, stroma, or immune infiltrate.
Use the ROI locator only for broad location context."""
    return f"""You are given multiple aligned images for the same candidate tissue piece in a human lung cancer tissue section.

{image_description}

{image_use_notes}

{legend_block}

{tissue_classes}

{style_notes}

Score how likely this candidate piece belongs to each tissue class.

Rules:
- Use exactly these six keys:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
"""


def image_columns(row: dict[str, str]) -> List[str]:
    numbered = []
    for key, value in row.items():
        match = re.fullmatch(r"image(\d+)_rel", key)
        if match and value:
            numbered.append((int(match.group(1)), value))
    if numbered:
        by_number = {number: value for number, value in numbered}
        mode = context_image_mode()
        mode_numbers = {
            "all6": [1, 2, 3, 4, 5, 6],
            "all5": [1, 2, 3, 4, 5],
            "he_only": [1, 3, 5],
            "he_piece_local": [1, 3],
            "ficture_piece_local": [2, 4],
            "he_piece_text": [1],
            "he_text_piece": [1],
            "he_text_three": [1, 3, 5],
            "he_semantic_ficture": [1, 2, 3, 4, 5],
            "he_semantic_ficture_no_locator": [1, 2, 3, 4],
            "he_semantic_ficture_local": [1, 3, 4, 5],
            "he_semantic_ficture_piece": [1, 2, 3, 5],
            "piece_local": [1, 2, 3, 4],
            "no_ficture_closeup": [1, 3, 4, 5],
            "piece_only": [1, 2],
            "local_only": [3, 4],
        }.get(mode)
        if mode_numbers:
            selected = [by_number[number] for number in mode_numbers if number in by_number]
            if selected:
                return selected
        return [value for _, value in sorted(numbered)]
    return [row["he_crop_rel"], row["ficture_crop_rel"]]


def prompt_for_row(template: str, row: dict[str, str]) -> str:
    semantic_legend = row.get("semantic_ficture_legend_text", "").strip()
    if "{SEMANTIC_FICTURE_LEGEND_TEXT}" in template:
        template = template.replace("{SEMANTIC_FICTURE_LEGEND_TEXT}", semantic_legend)
    summary = row.get("ficture_summary_text", "").strip()
    final_json_instruction = ""
    if os.environ.get("CONTEXT_FORCE_FINAL_JSON", "").strip() == "1":
        final_json_instruction = (
            "\n\nAfter any thinking, end your answer with exactly one line beginning with "
            "Final JSON: followed by one JSON object using only these six keys: "
            "bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration."
        )
    if "{FICTURE_SUMMARY_TEXT}" in template:
        return template.replace("{FICTURE_SUMMARY_TEXT}", summary) + final_json_instruction
    if summary:
        return f"{template.rstrip()}\n\n{summary}\n{final_json_instruction}"
    return template + final_json_instruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--factor-legend-csv",
        type=Path,
        default=Path(os.environ.get("FACTOR_LEGEND_CSV", DEFAULT_FACTOR_LEGEND_CSV)),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--prompt-template-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    user_prompt = (
        args.prompt_template_file.read_text(encoding="utf-8")
        if args.prompt_template_file
        else build_context_prompt(args.factor_legend_csv)
    )
    pool_dir = args.pool_csv.parent
    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.limit:
        rows = rows[: args.limit]
    label_counts = Counter(row["label"] for row in rows)
    if not rows:
        raise SystemExit("Input pool has no rows")
    if set(label_counts) - set(TRUE_LABEL_TO_CLASS):
        raise SystemExit(f"Unexpected labels in input pool: {sorted(set(label_counts) - set(TRUE_LABEL_TO_CLASS))}")
    if not args.limit and len(label_counts) != len(CLASS_KEYS):
        raise SystemExit(f"Expected six labels in input pool, got {dict(label_counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (args.output_dir / "prompt_user_template.txt").write_text(user_prompt, encoding="utf-8")
    (args.output_dir / "prompt_user_first_row.txt").write_text(prompt_for_row(user_prompt, rows[0]), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(args.pool_csv),
                "factor_legend_csv": str(args.factor_legend_csv),
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "input_style": "five_image_context_aware_piece",
                "context_image_mode": context_image_mode(),
                "prompt_template_file": str(args.prompt_template_file) if args.prompt_template_file else "",
                "context_prompt_style": os.environ.get("CONTEXT_PROMPT_STYLE", "full"),
                "class_keys": CLASS_KEYS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cross_path = args.output_dir / "cross_label_scores.csv"
    pred_path = args.output_dir / "per_candidate_predictions.csv"
    failed_path = args.output_dir / "failed_rows.csv"
    responses_path = args.output_dir / "raw_responses.jsonl"

    prediction_rows: List[dict] = []
    done_keys = set()
    if args.resume and pred_path.exists():
        prediction_rows = list(csv.DictReader(pred_path.open()))
        done_keys = {row["row_key"] for row in prediction_rows if row.get("parse_status") == "ok"}

    failed_rows: List[dict] = []
    if args.resume and failed_path.exists():
        failed_rows = list(csv.DictReader(failed_path.open()))

    vlm = load_vlm(args.model, args.device)

    for idx, row in enumerate(rows, start=1):
        key = row_key_string(row)
        if key in done_keys:
            continue
        raw = ""
        try:
            images = [Image.open(pool_dir / rel).convert("RGB") for rel in image_columns(row)]
            raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt_for_row(user_prompt, row), args.max_new_tokens)
            scores, parsed_json = parse_scores(raw)
            predicted, top_tie, top_score = predicted_from_scores(scores)
            true_class = TRUE_LABEL_TO_CLASS[row["label"]]
            is_correct = bool(predicted and predicted == true_class and not top_tie)
            out = {
                "row_index": idx,
                "row_key": key,
                "candidate_uid": row["candidate_uid"],
                "true_label": row["label"],
                "true_class": true_class,
                "display": row["display"],
                "sample_bucket": row["sample_bucket"],
                "source": row["source"],
                "run": row["run"],
                "setting": row["setting"],
                "candidate_id": row["candidate_id"],
                "predicted_class": predicted,
                "top_score": top_score,
                "top_score_tie": "true" if top_tie else "false",
                "is_correct": "true" if is_correct else "false",
                "parse_status": "ok",
                **{class_key: scores[class_key] for class_key in CLASS_KEYS},
            }
            prediction_rows = [old for old in prediction_rows if old["row_key"] != key] + [out]
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            done_keys.add(key)
            with responses_path.open("a") as handle:
                handle.write(json.dumps({"row_key": key, "parsed_json": parsed_json, "raw_response": raw}) + "\n")
            print(f"Scored {idx}/{len(rows)} true={true_class} pred={predicted or 'TIE'} correct={is_correct}", flush=True)
        except Exception as exc:
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            failed_rows.append(
                {
                    "row_index": idx,
                    "row_key": key,
                    "candidate_uid": row["candidate_uid"],
                    "true_label": row["label"],
                    "sample_bucket": row["sample_bucket"],
                    "error": str(exc)[:2400],
                    "raw_response": raw[:2400],
                }
            )
            print(f"FAILED {idx}/{len(rows)} {key}: {str(exc)[:200]}", flush=True)

        prediction_rows = sorted(prediction_rows, key=lambda item: int(item["row_index"]))
        write_csv(pred_path, prediction_rows)
        cross_rows = [
            {
                "row_index": pred["row_index"],
                "row_key": pred["row_key"],
                "candidate_uid": pred["candidate_uid"],
                "true_class": pred["true_class"],
                "sample_bucket": pred["sample_bucket"],
                **{class_key: pred[class_key] for class_key in CLASS_KEYS},
            }
            for pred in prediction_rows
        ]
        write_csv(cross_path, cross_rows)
        write_csv(
            failed_path,
            failed_rows,
            fieldnames=["row_index", "row_key", "candidate_uid", "true_label", "sample_bucket", "error", "raw_response"],
        )

    if failed_rows:
        print(f"{len(failed_rows)} rows failed; rerun with --resume after fixing the issue.", flush=True)
    write_accuracy_tables(args.output_dir, prediction_rows)
    write_same_class_retrieval_tables(args.output_dir, prediction_rows)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
