#!/usr/bin/env python3
"""Step 2: verify Step 1 cell-type hypotheses with H&E morphology only.

Inputs:
- Step 1 cell-type-only hypotheses.
- H&E candidate reverse-blur crop.
- Optional H&E local context crop.

Outputs six H&E morphology scores.  This intentionally does not use the
FICTURE image; FICTURE image consistency is Step 3.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

from run_paired_vlm_hit_test import generate, load_vlm


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


SYSTEM_PROMPT = (
    "You are a careful lung pathology morphology verifier. "
    "You judge only H&E morphology for a highlighted candidate mask. "
    "Return only valid JSON."
)


USER_TEMPLATE = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only a FICTURE-derived RGB/cell-type list and produced these biological hypotheses:
{step1_hypothesis_json}

Now use H&E morphology to verify or reject those hypotheses.

Images:
Image 1: H&E candidate reverse-blur crop. The candidate region is sharp and full color; the outside region is grayscale and blurred.
Image 2: H&E local context crop. Use this to see whether the candidate is part of a larger airway, vessel, alveolar region, tumor nest, stromal matrix, or immune aggregate.

Judge the candidate mask only. Do not classify the whole image.

H&E morphology checklist:
- bronchiola: airway-like lumen, bronchiolar epithelial lining, airway wall or mucosal fold.
- alveoli: open alveolar air spaces and thin septa, not a dense solid region.
- vessels: vascular lumen or vessel wall, elongated/round vessel structure, smooth muscle vessel wall.
- tumor: dense malignant epithelial tumor nests, glands, or solid tumor-like cellular region.
- stroma: collagen/fibroblast/smooth-muscle-like matrix, stromal bands, mesenchymal tissue.
- immune_infiltration: dense small round-cell aggregates or immune-cell-rich clusters.

Important differential rules:
- This is a lung cancer tissue section. Do not call a candidate tumor just because the surrounding tissue is cancer, cellular, or atypical.
- Tumor should be high only when the candidate itself is a solid/nested/gland-forming epithelial tumor region.
- Bronchiolar epithelium can also be epithelial and cellular. If there is an airway-like lumen or lining, prefer bronchiola over tumor.
- Alveolar regions can be near tumor or inflammation. If the candidate mainly follows air spaces or thin septa, prefer alveoli over tumor/stroma.
- If the H&E view is too small or nonspecific, keep all uncertain classes moderate instead of forcing tumor.

Scoring rules:
- Return exactly one JSON object.
- Use the six scores only for H&E morphology, not for FICTURE color.
- Each score must be an integer from 0 to 100.
- Use high scores only when H&E shows defining structural evidence.
- If a class is merely possible but H&E is nonspecific, keep that score at 50 or lower.
- Use tumor score above 70 only when tumor architecture is clearly visible inside the candidate mask.
- Do not let Step 1 cell-type hypotheses override clear H&E morphology.

Return format:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "top_class": "one of the six class keys",
  "he_main_evidence": "one short H&E morphology sentence",
  "he_uncertainty": "one short sentence or none"
}}"""


USER_TEMPLATE_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only a FICTURE-derived RGB/cell-type list and produced these biological hypotheses:
{step1_hypothesis_json}

Now use H&E morphology to verify or reject those hypotheses.

Images:
Image 1: H&E candidate reverse-blur crop. The candidate region is sharp and full color; the outside region is grayscale and blurred.

Judge the candidate mask only. Do not classify the whole image.

H&E morphology checklist:
- bronchiola: airway-like lumen, bronchiolar epithelial lining, airway wall or mucosal fold.
- alveoli: open alveolar air spaces and thin septa, not a dense solid region.
- vessels: vascular lumen or vessel wall, elongated/round vessel structure, smooth muscle vessel wall.
- tumor: dense malignant epithelial tumor nests, glands, or solid tumor-like cellular region.
- stroma: collagen/fibroblast/smooth-muscle-like matrix, stromal bands, mesenchymal tissue.
- immune_infiltration: dense small round-cell aggregates or immune-cell-rich clusters.

Important differential rules:
- This is a lung cancer tissue section. Do not call a candidate tumor just because the tissue section is from cancer.
- Tumor should be high only when the candidate itself is a solid/nested/gland-forming epithelial tumor region.
- Bronchiolar epithelium can also be epithelial and cellular. If there is an airway-like lumen or lining inside the candidate, prefer bronchiola over tumor.
- Alveolar regions can be near tumor or inflammation. If the candidate mainly follows air spaces or thin septa, prefer alveoli over tumor/stroma.
- If this candidate crop is too small or nonspecific, keep all uncertain classes moderate instead of forcing tumor.

Scoring rules:
- Return exactly one JSON object.
- Use the six scores only for H&E morphology, not for FICTURE color.
- Each score must be an integer from 0 to 100.
- Use high scores only when H&E shows defining structural evidence.
- If a class is merely possible but H&E is nonspecific, keep that score at 50 or lower.
- Use tumor score above 70 only when tumor architecture is clearly visible inside the candidate mask.
- Do not let Step 1 cell-type hypotheses override clear H&E morphology.

Return format:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "top_class": "one of the six class keys",
  "he_main_evidence": "one short H&E morphology sentence",
  "he_uncertainty": "one short sentence or none"
}}"""


USER_TEMPLATE_SIMPLE_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and produced these hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Score how much the H&E appearance supports each tissue class for this candidate mask.
Judge only the highlighted candidate mask, not the background.

Tissue classes:
bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100."""


USER_TEMPLATE_TARGET_VERIFY_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and selected this target hypothesis:
{target_hypothesis}

Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Judge whether the H&E appearance supports the target hypothesis for this candidate mask.
Judge only the highlighted candidate mask, not the background.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported should be true only if H&E morphology supports the target hypothesis."""


USER_TEMPLATE_ONE_HYPOTHESIS_VERIFY_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Judge whether the H&E appearance supports this one hypothesis for the highlighted candidate mask.
Judge only the highlighted candidate mask, not the background.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported should be true only if H&E morphology supports this hypothesis."""


USER_TEMPLATE_ONE_HYPOTHESIS_SOFT_KEEP_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
This is a retention gate, not the final classifier.
Judge whether the H&E appearance is compatible with this one hypothesis for the highlighted candidate mask.
Judge only the highlighted candidate mask, not the background.

Important:
- Keep the hypothesis if H&E is compatible or not clearly contradictory.
- Reject only if H&E clearly contradicts this tissue type.
- A small candidate may show only part of a structure, such as one wall segment, one septal fragment, or one stromal band.
- Do not require a complete airway lumen, complete vessel lumen, or large alveolar field if the small piece is still compatible.
- Do not call everything tumor just because the section is lung cancer.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported should be true if H&E is compatible with this hypothesis, even if the evidence is incomplete."""


USER_TEMPLATE_ONE_HYPOTHESIS_PARTIAL_PIECE_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Judge whether this highlighted candidate piece could be a partial piece of the target tissue class.
This is not a whole-structure diagnosis. The candidate can be a small part of a larger structure.

Partial-piece rules:
- bronchiola: keep if the candidate could be airway epithelial lining, airway wall, or mucosal fold; a full lumen is helpful but not required.
- alveoli: keep if the candidate could be alveolar septa or open-air-space boundary; do not require a large alveolar field.
- vessels: keep if the candidate could be vessel wall, elongated vascular boundary, smooth-muscle wall, or a lumen edge; a full circular lumen is helpful but not required.
- tumor: keep only if the piece itself looks like crowded epithelial tumor, tumor nest, gland, or solid malignant epithelium.
- stroma: keep if the piece looks fibrous, collagen-rich, smooth-muscle-like, or mesenchymal.
- immune_infiltration: keep if the piece looks rich in small round immune cells or immune aggregates.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported should be true when this small piece is compatible with the target tissue class."""


USER_TEMPLATE_ONE_HYPOTHESIS_RETAIN_UNCERTAIN_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Decide whether H&E clearly rules out this one hypothesis.
This step is designed to avoid false negatives. It should keep uncertain but plausible hypotheses for later filtering.

Use this decision rule:
- supported = true if the target class is plausible, weakly supported, or uncertain from H&E.
- supported = false only if H&E clearly contradicts the target class.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100."""


USER_TEMPLATE_ONE_HYPOTHESIS_GRADED_COMPAT_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Give a graded H&E compatibility score for this one hypothesis.
Judge only the highlighted candidate mask, not the background.
This is a filter before later steps, so do not require a perfect whole structure, but also do not keep a class with no visible H&E support.

Score bands:
- 80-100: strong visible H&E support.
- 60-79: compatible partial piece; enough H&E evidence to keep.
- 40-59: nonspecific or uncertain; not enough H&E evidence to keep.
- 0-39: H&E contradicts this class.

Partial-piece guidance:
- bronchiola can be an airway wall or epithelial lining piece, not necessarily a full airway.
- alveoli can be thin septa or air-space boundary, not necessarily a large alveolar field.
- vessels can be vessel wall or lumen edge, not necessarily a full round vessel.
- stroma can be fibrous or smooth-muscle-like tissue.
- immune_infiltration can be small round-cell rich tissue.
- tumor should require visible tumor-like epithelial architecture inside the candidate.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

supported must be true only when he_support_score is 60 or higher."""


USER_TEMPLATE_ONE_HYPOTHESIS_VISIBLE_COMPATIBLE_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Check whether there is visible H&E morphology that is compatible with this target class.
Judge only the highlighted candidate mask, not the background.

Keep the target only when the candidate itself has visible compatible morphology:
- For bronchiola, a partial airway lining/wall can count.
- For alveoli, a septal or open-air-space boundary pattern can count.
- For vessels, a wall segment, lumen edge, or smooth-muscle-like vascular boundary can count.
- For tumor, the candidate itself should look like tumor epithelium, not just nearby cancer.
- For stroma, fibrous, collagenous, smooth-muscle-like, or spindle-cell tissue can count.
- For immune_infiltration, small round-cell rich tissue can count.

Do not keep a class only because Step 1 said it was possible.
If the H&E crop is too nonspecific, supported should be false.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100."""


USER_TEMPLATE_ONE_HYPOTHESIS_SOFT_NARROW_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
This is a conservative retention gate for small pieces.
Keep the target class if H&E is compatible with it and there is no stronger H&E reason to reject it.
Reject only clear mismatches, but do not mark every uncertain class as supported.

Practical rule:
- supported = true for clear or partial compatible evidence.
- supported = false for clearly wrong classes or purely nonspecific classes.
- If unsure between two plausible classes, keep both.
- If unsure among four or more classes, keep only the classes with the most visible morphology.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100."""


USER_TEMPLATE_ONE_HYPOTHESIS_GRADED_STROMA_LENIENT_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Give a graded H&E compatibility score for this one hypothesis.
Judge only the highlighted candidate mask, not the background.
This is a filter before later steps, so do not require a perfect whole structure, but also do not keep a class with no visible H&E support.

Score bands:
- 80-100: strong visible H&E support.
- 60-79: compatible partial piece; enough H&E evidence to keep.
- 40-59: nonspecific or uncertain; not enough H&E evidence to keep.
- 0-39: H&E contradicts this class.

Partial-piece guidance:
- bronchiola can be an airway wall or epithelial lining piece, not necessarily a full airway.
- alveoli can be thin septa or air-space boundary, not necessarily a large alveolar field.
- vessels can be vessel wall or lumen edge, not necessarily a full round vessel.
- stroma can be fibrous stroma, collagen-rich matrix, smooth-muscle-like tissue, or stromal band. If this morphology is visible, score at least 60.
- immune_infiltration can be small round-cell rich tissue.
- tumor should require visible tumor-like epithelial architecture inside the candidate.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

supported must be true only when he_support_score is 60 or higher."""


USER_TEMPLATE_ONE_HYPOTHESIS_THREE_LEVEL_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Verify this one hypothesis for the highlighted candidate mask.
This is a retention filter, not a final classifier.

Use three levels:
- strong support: clear H&E morphology for this class.
- weak support: the candidate is a small partial piece that is still compatible with this class.
- reject: H&E shows a different structure, or there is no visible support for this class.

Class cues:
- bronchiola: airway wall, epithelial lining, mucosal fold, or airway-like lumen.
- alveoli: open air-space boundary or thin septa; do not keep alveoli for a solid dense sheet.
- vessels: vessel wall, lumen edge, vascular boundary, or smooth-muscle-like vessel wall.
- tumor: crowded epithelial tumor nest, gland, or solid tumor-like epithelial architecture.
- stroma: collagen-rich, fibrous, spindle-cell, smooth-muscle-like, or mesenchymal tissue.
- immune_infiltration: small round-cell-rich tissue or immune aggregate.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

Score guide:
- 75-100 for strong support.
- 55-74 for weak but real partial-piece support.
- 0-54 for reject.
supported must be true only when he_support_score is 55 or higher."""


USER_TEMPLATE_ONE_HYPOTHESIS_STRUCTURE_EVIDENCE_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Decide whether the highlighted candidate mask has direct structural evidence for this one hypothesis.
Do not keep a class only because Step 1 says it is possible.
Do not keep a class only because the image is ambiguous.

Keep partial structures when they are visually specific:
- A bronchiola piece can be only airway lining or airway wall.
- A vessel piece can be only vessel wall or lumen edge.
- An immune piece can be only part of a dense small-cell cluster.
- A stroma piece can be only a fibrous or smooth-muscle-like band.

Be stricter for broad nonspecific classes:
- Alveoli needs visible open air-space / thin-septal pattern.
- Tumor needs visible tumor-like epithelial architecture inside the candidate.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported must be true only when there is direct H&E structural evidence."""


USER_TEMPLATE_ONE_HYPOTHESIS_ALVEOLI_NARROW_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Verify this one hypothesis for the highlighted candidate mask.
The main goal of this prompt is to avoid keeping alveoli just because a region is non-tumor or loose.

Rules:
- For alveoli, supported=true only if the highlighted mask follows open air spaces, thin septa, or alveolar parenchymal boundaries.
- Do not support alveoli for dense stroma, tumor nest, vessel wall, airway wall, or immune aggregate.
- For bronchiola and vessels, a partial wall or edge can still count.
- For stroma, fibrous, collagenous, smooth-muscle-like, or mesenchymal tissue can count.
- For immune infiltration, small round-cell-rich tissue can count.
- For tumor, require visible tumor-like epithelial architecture inside the mask.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported should be true when this target has visible compatible H&E evidence."""

USER_TEMPLATE_ONE_HYPOTHESIS_CLASS_SPECIFIC_RESCUE_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Verify this one target class for the highlighted candidate mask.
This is a retention gate for small candidate pieces, not a final classifier.
Keep a target class only when the highlighted mask has visible morphology that is compatible with that class.

Class-specific retention rules:
- bronchiola: supported=true if the mask could be airway epithelial lining, airway wall, mucosal fold, or a small wall-like piece near an airway. A full airway lumen is not required.
- vessels: supported=true if the mask could be vessel wall, lumen edge, elongated vascular boundary, or smooth-muscle-like vascular wall. A full round lumen is not required.
- alveoli: supported=true only if the mask follows open air-space boundaries, thin septa, or alveolar parenchymal edges. Do not keep alveoli for a solid dense tumor/stroma/immune sheet.
- tumor: supported=true if the mask contains crowded epithelial tumor, malignant glands, tumor nests, or solid tumor-like epithelial architecture, even if the piece is small.
- stroma: supported=true if the mask contains fibrous/collagenous matrix, spindle-cell tissue, smooth-muscle-like tissue, or supporting stromal band, including tumor-associated stroma.
- immune_infiltration: supported=true if the mask contains small round-cell-rich tissue, lymphoid/macrophage-rich patches, or an immune aggregate, even if it is a local patch rather than a large aggregate.

Reject rule:
- supported=false only when the highlighted mask is clearly another structure or has no visible compatible evidence for this target class.
- If the target is a plausible small partial structure and H&E is ambiguous but compatible, keep it.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported must match the rule above."""

USER_TEMPLATE_ONE_HYPOTHESIS_CONTEXT_RESCUE_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now inspect two H&E images.

Image 1: candidate crop. The highlighted mask is the exact candidate being judged.
Image 2: local context crop. Use this only to understand whether the candidate is part of a larger airway, vessel, alveolar region, tumor, stroma, or immune aggregate.

Task:
Verify this one target class for the highlighted candidate mask.
The final decision must be about the candidate mask, not the whole context image.

How to use context:
- If the candidate is a tiny wall or edge, use local context to decide whether it is attached to an airway-like lumen, vessel-like lumen, or alveolar air-space boundary.
- Do not mark a target as supported only because the context contains that tissue somewhere else.
- The candidate must be spatially connected to or immediately part of that larger structure.

Class-specific rules:
- bronchiola: keep if the candidate is a small epithelial/wall/mucosal piece attached to an airway-like lumen or airway wall in local context.
- vessels: keep if the candidate is a small wall/edge/boundary piece attached to a vessel-like lumen or vascular wall in local context.
- alveoli: keep if the candidate is part of thin septa or air-space boundary; reject solid dense sheets.
- tumor: keep if the candidate itself is tumor epithelium/tumor nest/gland/solid malignant epithelium.
- stroma: keep if the candidate itself is fibrous, collagenous, spindle-cell, smooth-muscle-like, or stromal band.
- immune_infiltration: keep if the candidate itself is small round-cell-rich or an immune-cell patch.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported=true means the candidate mask is compatible with the target after checking both crop and context."""

USER_TEMPLATE_ONE_HYPOTHESIS_BROAD_CLASS_VETO_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

Step 1 status for this target class: {target_step1_status}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Verify this one target class for the highlighted candidate mask.
This is a small-piece retention gate. Keep true partial structures, but do not use broad fallback labels when their defining morphology is absent.

Target evidence rules:
- bronchiola: keep if the mask could be airway epithelial lining, airway wall, mucosal fold, or airway-adjacent wall piece. A complete lumen is not required.
- alveoli: keep only if the mask follows open air-space boundaries or thin septa. Reject solid dense tissue.
- vessels: keep if the mask could be vessel wall, lumen edge, elongated vascular boundary, or smooth-muscle-like vascular wall. A complete lumen is not required.
- tumor: keep only if the mask itself contains cohesive epithelial tumor nest, malignant gland, or solid tumor-like epithelial architecture. Reject airway wall, vessel wall, collagen, immune cluster, and nonspecific cellular tissue.
- stroma: keep only if the mask itself contains collagen-rich matrix, fibrous band, spindle-cell tissue, smooth-muscle-like tissue, or mesenchymal support. Reject tumor epithelium, immune aggregate, airway epithelium, and vessel lumen edge unless stromal matrix is visible.
- immune_infiltration: keep only if the mask itself contains dense small round-cell-rich tissue or an immune aggregate. Reject scattered cells, tumor cellularity, collagen, airway wall, and vessel wall.

Score guide:
- 80-100: clear target morphology in the highlighted mask.
- 60-79: partial but visible target morphology.
- 40-59: weak/nonspecific; do not keep.
- 0-39: contradicts target.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported must be true only when he_support_score is 60 or higher."""

USER_TEMPLATE_ONE_HYPOTHESIS_DIRECT_EVIDENCE_MINRULES_HE_CROP_ONLY = """Step 1 proposed this possible tissue class:
{target_hypothesis}

Other Step 1 hypotheses:
{step1_hypothesis_json}

Look at the H&E candidate crop only.

Question:
Does the highlighted candidate mask show direct H&E evidence for this target class?

Use simple criteria:
- bronchiola: airway lining or airway wall.
- alveoli: open air-space boundary or thin septa.
- vessels: vessel wall or lumen edge.
- tumor: cohesive epithelial tumor nest, gland, or solid tumor region.
- stroma: collagen, fibrous band, spindle-cell, or smooth-muscle-like matrix.
- immune_infiltration: dense small round-cell aggregate.

Do not keep a class just because it is biologically possible from Step 1.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported=true only if the candidate mask shows direct visible support for the target class."""

USER_TEMPLATE_COMPARATIVE_HYPOTHESIS_VETO_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Compare the Step 1 hypotheses against each other using H&E morphology.
Judge only the highlighted candidate mask, not the background.

Why this is comparative:
- Do not evaluate each class in isolation.
- Keep only classes that are visibly supported relative to the other possible classes.
- Broad fallback classes such as tumor, stroma, and immune_infiltration should not be kept unless their defining morphology is visible.
- If the candidate is a small partial airway or vessel wall, bronchiola or vessels may be kept even without a complete lumen.

Class cues:
- bronchiola: airway lining, airway wall, mucosal fold, or airway-adjacent wall piece.
- alveoli: open air-space boundary or thin septa.
- vessels: vessel wall, lumen edge, elongated vascular boundary, or smooth-muscle-like vascular wall.
- tumor: cohesive epithelial tumor nest, malignant gland, or solid tumor-like epithelial region.
- stroma: collagen-rich matrix, fibrous band, spindle-cell tissue, or smooth-muscle-like matrix.
- immune_infiltration: dense small round-cell aggregate or immune-cell-rich cluster.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
Set keep=true only for classes with direct visible H&E support.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_SOFT_KEEP_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Compare the Step 1 hypotheses using H&E morphology and keep a small set of plausible classes.
This is still a retention gate, so do not force a single final label.

Rules:
- Keep a class if H&E has visible support or if the candidate is a plausible partial piece of that class.
- Reject broad fallback classes when they are only biologically possible but not visually supported.
- Prefer uncertainty between two or three plausible classes over keeping almost all classes.
- Avoid deleting bronchiola/vessels just because only a wall segment is visible.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_RANKED_RETENTION_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use H&E morphology to rank the Step 1 hypotheses by direct visual support for the highlighted candidate mask.
This is a retention gate, not a final diagnosis, but the output should be narrower than Step 1.

Decision policy:
- Usually keep only 1 to 3 classes.
- Keep bronchiola, vessels, or alveoli if the candidate is a plausible partial structural piece, even if the full lumen or full alveolar field is not visible.
- Keep tumor only if the highlighted mask itself looks like cohesive epithelial tumor, malignant gland, tumor nest, or solid tumor-like epithelium.
- Keep stroma only if the highlighted mask itself shows collagen, fibrous band, spindle-cell tissue, smooth-muscle-like matrix, or mesenchymal support.
- Keep immune_infiltration only if the highlighted mask itself shows dense small round-cell-rich tissue or an immune aggregate.
- Do not keep tumor, stroma, and immune_infiltration together just because all are biologically possible.
- H&E can overrule Step 1 when Step 1 is too broad.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_PATTERN_FIRST_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
First decide which H&E pattern the highlighted candidate mask most resembles. Then keep only the Step 1 hypotheses supported by that pattern.

H&E pattern guide:
- airway-wall pattern -> supports bronchiola.
- thin-septal / open-air-space boundary pattern -> supports alveoli.
- vessel-wall / lumen-edge / vascular smooth-muscle pattern -> supports vessels.
- cohesive epithelial nest / malignant gland / solid epithelial sheet pattern -> supports tumor.
- collagen / fibrous band / spindle-cell or smooth-muscle matrix pattern -> supports stroma.
- dense small-round-cell aggregate pattern -> supports immune_infiltration.

Important:
- A small wall segment can still be bronchiola or vessels.
- A thin boundary can still be alveoli.
- Broad classes must show their own pattern; do not keep them as generic fallback labels.
- Judge the highlighted candidate mask, not the gray background.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_PRIMARY_PATTERN_BUDGET_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Choose the dominant H&E morphology pattern of the highlighted candidate mask, then keep only the Step 1 hypotheses that match that pattern.

Decision budget:
- Keep 1 class when one pattern is clearly dominant.
- Keep 2 classes when two patterns are genuinely hard to separate in H&E.
- Keep 3 classes only when the crop is visibly mixed.
- Do not keep 4 or more classes unless the highlighted mask is truly nonspecific.

Pattern-to-class mapping:
- airway wall / airway lining / mucosal fold -> bronchiola.
- thin septum / open-air-space edge -> alveoli.
- vessel wall / lumen edge / vascular smooth muscle -> vessels.
- cohesive epithelial nest / malignant gland / solid epithelial sheet -> tumor.
- collagen / fibrous band / spindle-cell or smooth-muscle matrix -> stroma.
- dense small-round-cell aggregate -> immune_infiltration.

False-positive guard:
- Do not add tumor, stroma, or immune_infiltration as generic fallback labels.
- If a structural pattern is visible, prefer the structural class over broad fallback classes.
- Keep a broad class only when its own pattern is present inside the highlighted mask.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_STRUCTURAL_PRIORITY_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use H&E to decide whether the highlighted candidate is an anatomical structure piece or a broad tissue-compartment piece.

Priority rule:
If the candidate looks like a partial airway wall, vessel wall, or alveolar septum, keep the corresponding structural class even if the piece is small. Do not also keep broad labels unless their specific morphology is visible in the same highlighted mask.

Structural classes:
- bronchiola: airway lining, airway wall, mucosal fold, partial airway-adjacent epithelial/wall piece.
- alveoli: thin septa, open air-space boundary, alveolar parenchymal edge.
- vessels: vessel wall, lumen edge, elongated vascular boundary, vascular smooth muscle.

Broad classes:
- tumor: cohesive epithelial tumor nest, malignant gland, solid tumor-like epithelium.
- stroma: collagen, fibrous band, spindle-cell tissue, smooth-muscle-like matrix.
- immune_infiltration: dense small round-cell aggregate or immune-cell-rich cluster.

Known failure to avoid:
Small bronchiola or vessel wall pieces are often wrongly kept as tumor/stroma/immune. Do not do that unless the broad class has direct visible evidence inside the mask.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_FALSE_POSITIVE_GUARD_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Compare the Step 1 hypotheses and remove likely false-positive labels using H&E morphology.

Known false-positive patterns from previous runs:
- bronchiola candidates were often over-kept as tumor, stroma, and immune_infiltration.
- vessel candidates were often over-kept as stroma and immune_infiltration.
- alveoli candidates were often over-kept as every class.
- tumor, stroma, and immune_infiltration were often kept together even when only one broad pattern was visible.

Use these guards:
- If the mask is a wall/edge/septal piece, keep the matching structural class and reject broad labels without direct evidence.
- If the mask is a broad class, keep only the broad class or classes whose own morphology is visible.
- Immune_infiltration requires a dense small-round-cell aggregate, not just scattered inflammatory cells.
- Stroma requires collagen/fibrous/spindle/smooth-muscle matrix, not just adjacency to tissue.
- Tumor requires cohesive epithelial tumor architecture, not just cellularity.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_SOFT_BROAD_VETO_RETENTION_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use H&E morphology to keep all Step 1 hypotheses that remain visually plausible, but remove broad fallback labels that have no direct H&E evidence.

This is a retention gate:
- Do not force a single label.
- Do not use a fixed maximum number of kept classes.
- If two or three classes are genuinely plausible, keep them.
- If a candidate is weak or ambiguous but still compatible with a structural class, keep the structural class.

Broad-label veto:
- Tumor is not kept just because Step 1 found tumor-like cells. Keep tumor only when the highlighted mask itself has cohesive epithelial tumor, malignant glands, tumor nests, or solid tumor-like epithelium.
- Stroma is not kept just because the candidate is adjacent to tissue or looks nonspecific. Keep stroma only when collagen, fibrous band, spindle-cell tissue, smooth muscle matrix, or mesenchymal support is visible.
- Immune_infiltration is not kept just because immune cells may be present. Keep immune_infiltration only when the highlighted mask has dense small round-cell-rich tissue or an immune aggregate.

Structural rescue:
- Bronchiola can be a partial airway wall, lining, or mucosal fold.
- Vessels can be a partial vessel wall, lumen edge, or vascular smooth-muscle boundary.
- Alveoli can be a thin septal or open-air-space boundary.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_FEWSHOT_BROAD_VETO_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use H&E morphology to keep plausible Step 1 hypotheses. This is a retention gate, not a final single-label classifier.

Text examples of the intended behavior:
- Example A: If a small highlighted piece looks like airway wall or airway lining, keep bronchiola. Do not also keep tumor, stroma, or immune_infiltration unless those patterns are visible inside the same highlighted piece.
- Example B: If a small highlighted piece looks like a vessel wall or lumen edge, keep vessels. Do not keep stroma just because vessel wall can contain smooth muscle; keep stroma only if fibrous/stromal matrix is the visible target pattern.
- Example C: If a highlighted piece follows thin septa or open air-space boundaries, keep alveoli even if Step 1 also listed tumor or stroma.
- Example D: If the highlighted piece is a dense small-round-cell aggregate, keep immune_infiltration. If immune cells are only scattered or inferred from Step 1, do not keep immune_infiltration.
- Example E: If the highlighted piece is cohesive epithelial tumor or malignant gland, keep tumor. If it is only cellular or near cancer tissue, do not keep tumor.
- Example F: If the highlighted piece is collagen/fibrous/spindle-cell matrix, keep stroma. If it is just a wall edge or adjacent background, do not keep stroma.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_STRUCTURED_CHECKLIST_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use this internal checklist, then return only the final JSON.

Internal checklist:
1. Is the highlighted mask a partial anatomical wall/edge/septum?
   - airway wall/lining/fold -> bronchiola.
   - vessel wall/lumen edge/vascular boundary -> vessels.
   - thin septum/open air-space boundary -> alveoli.
2. Is the highlighted mask a broad tissue pattern?
   - cohesive epithelial tumor/gland/nest/solid tumor sheet -> tumor.
   - collagen/fibrous/spindle/smooth-muscle matrix -> stroma.
   - dense small round-cell aggregate -> immune_infiltration.
3. Are any Step 1 hypotheses only biologically possible but not visible in H&E?
   - If yes, reject those hypotheses.
4. If H&E is ambiguous, keep the plausible structural class rather than replacing it with a broad fallback label.

Output rule:
- Keep every Step 1 hypothesis with visible H&E support.
- Reject broad tumor/stroma/immune labels without visible H&E support.
- Do not force one label, and do not hard-cap the number of kept classes.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_MINIMAL_RETENTION_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Keep the Step 1 classes that are still visually plausible for the highlighted candidate mask in H&E.
Remove a class only when the H&E crop gives little or no support for that class.
This is a retention filter, not a final single-class diagnosis.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_STRUCTURAL_KEEP_BROAD_PRUNE_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use H&E morphology as a conservative filter. The main danger is deleting small true structural pieces, so keep bronchiola, alveoli, or vessels if the highlighted candidate could be part of that structure.

Weak structural classes:
- Keep bronchiola if the candidate could be airway wall, airway epithelial lining, mucosal fold, or a piece adjacent to an airway lumen.
- Keep alveoli if the candidate could be thin alveolar septa or open air-space boundary.
- Keep vessels if the candidate could be vessel wall, vascular lumen edge, or vascular smooth-muscle boundary.

Broad fallback classes:
- Keep tumor only when the candidate itself looks like cohesive epithelial tumor, malignant gland, or tumor nest.
- Keep stroma only when the candidate itself looks like collagen, fibrous band, spindle-cell tissue, or stromal matrix.
- Keep immune_infiltration only when the candidate itself is a dense small-round-cell aggregate.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_BRIEF_CLASS_DEFINITIONS_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use the short H&E definitions below to decide which Step 1 classes remain visually plausible for the highlighted candidate mask.

H&E definitions:
- bronchiola: airway-like lumen, airway epithelial lining, airway wall, mucosal fold.
- alveoli: open lung air spaces, thin septa, delicate parenchymal mesh.
- vessels: vascular lumen, vessel wall, endothelial-lined space, smooth-muscle vascular boundary.
- tumor: cohesive malignant epithelial tumor, tumor nests, malignant glands, solid crowded epithelial region.
- stroma: collagen, fibrous matrix, spindle-cell tissue, smooth-muscle-like stromal support.
- immune_infiltration: dense small round-cell aggregate or lymphoid/plasma-cell-rich cluster.

Output rule:
Keep a class when the highlighted mask has visible H&E support for that definition.
Reject Step 1 classes that are only biologically possible but not visible in the highlighted mask.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_ALVEOLI_SEPTA_RESCUE_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Use H&E morphology to keep the Step 1 classes that are still plausible. This prompt specifically protects alveolar and wall-like fragments from being over-called as stroma or immune.

Rescue rules:
- Keep alveoli if the highlighted mask follows thin septa, delicate parenchymal mesh, or the edge of open air spaces.
- Keep bronchiola if the highlighted mask is airway wall, epithelial lining, or mucosal fold.
- Keep vessels if the highlighted mask is vessel wall, vascular lumen edge, or smooth-muscle vascular boundary.

Do not over-call broad labels:
- Do not call thin septa or wall edges stroma just because they are fibrous-looking.
- Do not call scattered nuclei immune_infiltration; immune requires a dense small-round-cell aggregate.
- Do not call cellular epithelium tumor unless it is cohesive malignant tumor, malignant gland, or tumor nest.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_STROMA_IMMUNE_STRICT_VETO_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the H&E candidate crop.

Task:
Keep all Step 1 classes that remain visually plausible, but apply a strict evidence test to stroma and immune_infiltration because these two broad labels were often over-kept in prior runs.

Strict broad-label test:
- Keep stroma only if the highlighted mask is mostly collagen/fibrous matrix, spindle-cell tissue, smooth-muscle-like stromal band, or mesenchymal support tissue.
- Keep immune_infiltration only if the highlighted mask is a dense cluster of small round cells, lymphoid aggregate, plasma-cell-rich cluster, or macrophage-rich aggregate.

Important negatives:
- A vessel wall is not automatically stroma.
- An airway wall is not automatically stroma.
- Thin alveolar septa are not automatically stroma.
- Scattered nuclei are not automatically immune infiltration.
- Mixed nearby cells outside the highlighted mask are not enough.

Structural classes should stay if supported:
- bronchiola: airway wall, lining, fold, airway-like lumen edge.
- alveoli: open air-space boundary, thin septa, delicate parenchymal mesh.
- vessels: vascular wall, lumen edge, endothelial-lined space, vascular smooth muscle.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_CONTEXT_MINIMAL_RETENTION_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now inspect two H&E images.

Image 1: H&E candidate crop. The highlighted mask is the exact candidate being judged.
Image 2: H&E local context crop. Use this only to understand nearby structure.

Task:
Keep the Step 1 classes that are still visually plausible for the highlighted candidate mask.
Use the context image only if the candidate is a small piece of a larger airway, vessel, or alveolar structure.
Do not classify the whole context image.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_CONTEXT_SOFT_BROAD_VETO_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now inspect two H&E images.

Image 1: H&E candidate crop. The highlighted mask is the exact candidate being judged.
Image 2: H&E local context crop. Use it only to see whether the highlighted candidate is part of a larger airway, vessel, alveolar region, tumor nest, stromal matrix, or immune aggregate.

Task:
Use H&E morphology to keep all Step 1 hypotheses that remain visually plausible for the highlighted candidate mask, but remove broad fallback labels that have no direct H&E evidence.

How to use context:
- The final decision is about the highlighted candidate mask, not the whole context image.
- Context can support bronchiola, vessels, or alveoli when the candidate is a small wall/edge/septal piece attached to a larger structure.
- Do not keep a class only because that tissue appears somewhere else in the context crop.

Broad-label veto:
- Tumor requires cohesive epithelial tumor, malignant gland, tumor nest, or solid tumor-like epithelium in the highlighted mask.
- Stroma requires collagen, fibrous band, spindle-cell tissue, smooth-muscle matrix, or mesenchymal support in the highlighted mask.
- Immune_infiltration requires dense small round-cell-rich tissue or an immune aggregate in the highlighted mask.

Structural rescue:
- Bronchiola can be a partial airway wall, lining, or mucosal fold.
- Vessels can be a partial vessel wall, lumen edge, or vascular smooth-muscle boundary.
- Alveoli can be a thin septal or open-air-space boundary.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""

USER_TEMPLATE_COMPARATIVE_CONTEXT_STRUCTURED_CHECKLIST_HE_CROP_ONLY = """You are in Step 2 of a three-step tissue candidate judge.

Step 1 used only FICTURE-derived RGB/cell-type composition and kept these possible tissue classes:
{retained_hypotheses}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now inspect two H&E images.

Image 1: H&E candidate crop. The highlighted mask is the exact candidate being judged.
Image 2: H&E local context crop. Use it only to understand whether the candidate is connected to a larger structure.

Task:
Use this internal checklist, then return only the final JSON.

Internal checklist:
1. In Image 1, what does the highlighted mask itself look like?
   - airway wall/lining/fold -> bronchiola.
   - vessel wall/lumen edge/vascular boundary -> vessels.
   - thin septum/open air-space boundary -> alveoli.
   - cohesive epithelial tumor/gland/nest/solid tumor sheet -> tumor.
   - collagen/fibrous/spindle/smooth-muscle matrix -> stroma.
   - dense small-round-cell aggregate -> immune_infiltration.
2. In Image 2, is the candidate spatially attached to a larger structure?
   - Use this only to support a small partial wall/edge/septal candidate.
   - Do not classify the whole context image.
3. Which Step 1 hypotheses are only biologically possible but not visible in H&E?
   - Reject those hypotheses.

Output rule:
- Keep every Step 1 hypothesis with visible H&E support in the candidate mask.
- Reject broad tumor/stroma/immune labels without visible candidate-mask evidence.
- Do not force one label, and do not hard-cap the number of kept classes.

Return exactly one JSON object:
{{
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "keep": {{
    "bronchiola": false,
    "alveoli": false,
    "vessels": false,
    "tumor": false,
    "stroma": false,
    "immune_infiltration": false
  }},
  "top_class": "one of the six class keys"
}}

Each score must be an integer from 0 to 100.
For classes not retained by Step 1, score must be 0 and keep must be false."""


ALL_HYPOTHESIS_PROMPTS = {
    "all_hypothesis_verify": USER_TEMPLATE_ONE_HYPOTHESIS_VERIFY_HE_CROP_ONLY,
    "all_hypothesis_soft_keep": USER_TEMPLATE_ONE_HYPOTHESIS_SOFT_KEEP_HE_CROP_ONLY,
    "all_hypothesis_partial_piece": USER_TEMPLATE_ONE_HYPOTHESIS_PARTIAL_PIECE_HE_CROP_ONLY,
    "all_hypothesis_retain_uncertain": USER_TEMPLATE_ONE_HYPOTHESIS_RETAIN_UNCERTAIN_HE_CROP_ONLY,
    "all_hypothesis_graded_compat": USER_TEMPLATE_ONE_HYPOTHESIS_GRADED_COMPAT_HE_CROP_ONLY,
    "all_hypothesis_visible_compatible": USER_TEMPLATE_ONE_HYPOTHESIS_VISIBLE_COMPATIBLE_HE_CROP_ONLY,
    "all_hypothesis_soft_narrow": USER_TEMPLATE_ONE_HYPOTHESIS_SOFT_NARROW_HE_CROP_ONLY,
    "all_hypothesis_graded_stroma_lenient": USER_TEMPLATE_ONE_HYPOTHESIS_GRADED_STROMA_LENIENT_HE_CROP_ONLY,
    "all_hypothesis_three_level": USER_TEMPLATE_ONE_HYPOTHESIS_THREE_LEVEL_HE_CROP_ONLY,
    "all_hypothesis_structure_evidence": USER_TEMPLATE_ONE_HYPOTHESIS_STRUCTURE_EVIDENCE_HE_CROP_ONLY,
    "all_hypothesis_alveoli_narrow": USER_TEMPLATE_ONE_HYPOTHESIS_ALVEOLI_NARROW_HE_CROP_ONLY,
    "all_hypothesis_class_specific_rescue": USER_TEMPLATE_ONE_HYPOTHESIS_CLASS_SPECIFIC_RESCUE_HE_CROP_ONLY,
    "all_hypothesis_context_rescue": USER_TEMPLATE_ONE_HYPOTHESIS_CONTEXT_RESCUE_HE_CROP_ONLY,
    "all_hypothesis_broad_class_veto": USER_TEMPLATE_ONE_HYPOTHESIS_BROAD_CLASS_VETO_HE_CROP_ONLY,
    "all_hypothesis_direct_evidence_minrules": USER_TEMPLATE_ONE_HYPOTHESIS_DIRECT_EVIDENCE_MINRULES_HE_CROP_ONLY,
}

COMPARATIVE_PROMPTS = {
    "comparative_hypothesis_veto": USER_TEMPLATE_COMPARATIVE_HYPOTHESIS_VETO_HE_CROP_ONLY,
    "comparative_soft_keep": USER_TEMPLATE_COMPARATIVE_SOFT_KEEP_HE_CROP_ONLY,
    "comparative_ranked_retention": USER_TEMPLATE_COMPARATIVE_RANKED_RETENTION_HE_CROP_ONLY,
    "comparative_pattern_first": USER_TEMPLATE_COMPARATIVE_PATTERN_FIRST_HE_CROP_ONLY,
    "comparative_primary_pattern_budget": USER_TEMPLATE_COMPARATIVE_PRIMARY_PATTERN_BUDGET_HE_CROP_ONLY,
    "comparative_structural_priority": USER_TEMPLATE_COMPARATIVE_STRUCTURAL_PRIORITY_HE_CROP_ONLY,
    "comparative_false_positive_guard": USER_TEMPLATE_COMPARATIVE_FALSE_POSITIVE_GUARD_HE_CROP_ONLY,
    "comparative_soft_broad_veto_retention": USER_TEMPLATE_COMPARATIVE_SOFT_BROAD_VETO_RETENTION_HE_CROP_ONLY,
    "comparative_fewshot_broad_veto": USER_TEMPLATE_COMPARATIVE_FEWSHOT_BROAD_VETO_HE_CROP_ONLY,
    "comparative_structured_checklist": USER_TEMPLATE_COMPARATIVE_STRUCTURED_CHECKLIST_HE_CROP_ONLY,
    "comparative_minimal_retention": USER_TEMPLATE_COMPARATIVE_MINIMAL_RETENTION_HE_CROP_ONLY,
    "comparative_structural_keep_broad_prune": USER_TEMPLATE_COMPARATIVE_STRUCTURAL_KEEP_BROAD_PRUNE_HE_CROP_ONLY,
    "comparative_brief_class_definitions": USER_TEMPLATE_COMPARATIVE_BRIEF_CLASS_DEFINITIONS_HE_CROP_ONLY,
    "comparative_alveoli_septa_rescue": USER_TEMPLATE_COMPARATIVE_ALVEOLI_SEPTA_RESCUE_HE_CROP_ONLY,
    "comparative_stroma_immune_strict_veto": USER_TEMPLATE_COMPARATIVE_STROMA_IMMUNE_STRICT_VETO_HE_CROP_ONLY,
    "comparative_context_minimal_retention": USER_TEMPLATE_COMPARATIVE_CONTEXT_MINIMAL_RETENTION_HE_CROP_ONLY,
    "comparative_context_soft_broad_veto": USER_TEMPLATE_COMPARATIVE_CONTEXT_SOFT_BROAD_VETO_HE_CROP_ONLY,
    "comparative_context_structured_checklist": USER_TEMPLATE_COMPARATIVE_CONTEXT_STRUCTURED_CHECKLIST_HE_CROP_ONLY,
}


TARGET_HE_CHECKLISTS = {
    "bronchiola": "Look for an airway-like lumen, bronchiolar epithelial lining, airway wall, or mucosal fold.",
    "alveoli": "Look for open alveolar air spaces and thin septa; the candidate should not look like a solid dense sheet.",
    "vessels": "Look for a round or elongated vascular lumen, vascular wall, red-blood-cell-like lumen content, or smooth muscle vessel wall.",
    "tumor": "Look for dense epithelial tumor nests, malignant glands, or solid crowded epithelial tumor architecture.",
    "stroma": "Look for collagen-rich matrix, fibroblast or smooth-muscle-like tissue, fibrous bands, or elongated spindle-cell stroma.",
    "immune_infiltration": "Look for dense small round-cell aggregates or immune-cell-rich clusters.",
}


USER_TEMPLATE_TARGET_VERIFY_CHECKLIST_HE_CROP_ONLY = """Step 1 used only FICTURE-derived RGB/cell-type composition and selected this target hypothesis:
{target_hypothesis}

Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the H&E candidate crop.

Task:
Judge whether the H&E appearance supports the target hypothesis for this candidate mask.
Judge only the highlighted candidate mask, not the background.

Target-specific H&E checklist:
{target_checklist}

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "he_support_score": 0,
  "supported": false
}}

he_support_score must be an integer from 0 to 100.
supported should be true only if H&E morphology supports the target hypothesis."""


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


def extract_json(text: str) -> dict:
    candidates = re.findall(r"\{.*\}", text.strip(), flags=re.S)
    if not candidates:
        raise ValueError("No JSON object found")
    last_error: Exception | None = None
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_error = exc
            try:
                return ast.literal_eval(candidate)
            except Exception as exc2:
                last_error = exc2
    raise ValueError(f"JSON parse failed: {last_error}")


def clamp_int(value: object) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = 0
    return max(0, min(100, number))


def parse_response(text: str) -> dict[str, object]:
    data = extract_json(text)
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError("scores is not an object")
    out: dict[str, object] = {}
    missing = []
    for key in CLASS_KEYS:
        if key not in scores:
            missing.append(key)
        out[f"{key}_he_score"] = clamp_int(scores.get(key, 0))
    if missing:
        raise ValueError(f"Missing score keys: {missing}")
    ranked = sorted(CLASS_KEYS, key=lambda cls: int(out[f"{cls}_he_score"]), reverse=True)
    out["predicted_class_he"] = str(data.get("top_class") or ranked[0])
    if out["predicted_class_he"] not in CLASS_KEYS:
        out["predicted_class_he"] = ranked[0]
    out["top_score_he"] = int(out[f"{ranked[0]}_he_score"])
    out["second_score_he"] = int(out[f"{ranked[1]}_he_score"])
    out["top_margin_he"] = int(out["top_score_he"]) - int(out["second_score_he"])
    out["top_score_tie_he"] = str(int(out[f"{ranked[0]}_he_score"]) == int(out[f"{ranked[1]}_he_score"])).lower()
    out["he_main_evidence"] = str(data.get("he_main_evidence", ""))[:500]
    out["he_uncertainty"] = str(data.get("he_uncertainty", ""))[:500]
    return out


def parse_target_verify_response(text: str, target_class: str) -> dict[str, object]:
    data = extract_json(text)
    support_score = clamp_int(data.get("he_support_score", data.get("support_score", 0)))
    supported_raw = data.get("supported", False)
    if isinstance(supported_raw, str):
        supported = supported_raw.strip().lower() in {"true", "yes", "1", "supported"}
    else:
        supported = bool(supported_raw)
    return {
        "target_hypothesis_class": target_class,
        "he_support_score": support_score,
        "he_supported": str(supported).lower(),
        "predicted_class_he": target_class if supported else "unsupported",
        "top_score_he": support_score,
        "second_score_he": "",
        "top_margin_he": "",
        "top_score_tie_he": "false",
        "he_main_evidence": str(data.get("he_main_evidence", ""))[:500],
        "he_uncertainty": str(data.get("he_uncertainty", ""))[:500],
    }


def parse_comparative_response(text: str, retained: list[str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = extract_json(text)
    raw_scores = data.get("scores", {})
    raw_keep = data.get("keep", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    if not isinstance(raw_keep, dict):
        raw_keep = {}
    retained_set = set(retained)
    per_hypothesis = []
    for cls in CLASS_KEYS:
        score = clamp_int(raw_scores.get(cls, 0))
        keep_raw = raw_keep.get(cls, False)
        if isinstance(keep_raw, str):
            keep = keep_raw.strip().lower() in {"true", "yes", "1", "keep", "supported"}
        else:
            keep = bool(keep_raw)
        if cls not in retained_set:
            score = 0
            keep = False
        per_hypothesis.append(
            {
                "target_hypothesis_class": cls,
                "he_support_score": score,
                "he_supported": str(keep).lower(),
                "predicted_class_he": cls if keep else "unsupported",
                "top_score_he": score,
                "second_score_he": "",
                "top_margin_he": "",
                "top_score_tie_he": "false",
                "he_main_evidence": str(data.get("he_main_evidence", ""))[:500],
                "he_uncertainty": str(data.get("he_uncertainty", ""))[:500],
            }
        )
    ranked = sorted(CLASS_KEYS, key=lambda cls: clamp_int(raw_scores.get(cls, 0)), reverse=True)
    summary = {
        "predicted_class_he": str(data.get("top_class") or ranked[0]),
        "top_score_he": clamp_int(raw_scores.get(ranked[0], 0)),
        "second_score_he": clamp_int(raw_scores.get(ranked[1], 0)),
        "top_margin_he": clamp_int(raw_scores.get(ranked[0], 0)) - clamp_int(raw_scores.get(ranked[1], 0)),
        "top_score_tie_he": str(clamp_int(raw_scores.get(ranked[0], 0)) == clamp_int(raw_scores.get(ranked[1], 0))).lower(),
    }
    if summary["predicted_class_he"] not in CLASS_KEYS:
        summary["predicted_class_he"] = ranked[0]
    return per_hypothesis, summary


def step1_json(row: dict[str, str]) -> str:
    payload = {
        "hypotheses": {key: row.get(key, "uncertain") for key in CLASS_KEYS},
        "main_ambiguity": row.get("main_ambiguity", ""),
        "composition_warning": row.get("composition_warning", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def step1_hypotheses_json(row: dict[str, str]) -> str:
    return json.dumps({key: row.get(key, "uncertain") for key in CLASS_KEYS}, ensure_ascii=False)


def step1_target_hypothesis(row: dict[str, str]) -> str:
    for key in CLASS_KEYS:
        if str(row.get(key, "")).strip().lower() == "plausible":
            return key
    def score(key: str) -> float:
        try:
            return float(row.get(f"{key}_hypothesis_score", 0))
        except Exception:
            return 0.0
    return max(CLASS_KEYS, key=score)


def step1_retained_hypotheses(row: dict[str, str]) -> list[str]:
    retained = []
    for key in CLASS_KEYS:
        status = str(row.get(key, "")).strip().lower()
        if status in {"plausible", "uncertain"}:
            retained.append(key)
    return retained


def build_retention_summary(output_dir: Path, rows: list[dict[str, object]], failed: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("parse_status") == "ok"]
    summary = []
    for cls in CLASS_KEYS:
        cls_rows = [row for row in ok if row.get("true_class") == cls]
        after_step1 = sum(1 for row in cls_rows if row.get("true_retained_after_step1") == "true")
        after_step2 = sum(1 for row in cls_rows if row.get("true_retained_after_step2") == "true")
        summary.append(
            {
                "true_class": cls,
                "n": len(cls_rows),
                "after_step1_true_retained": after_step1,
                "after_step2_true_retained": after_step2,
            }
        )
    write_csv(output_dir / "step2_retention_by_true_class.csv", summary)
    write_csv(
        output_dir / "step2_retention_run_gate.csv",
        [
            {
                "parsed_candidates": len(ok),
                "failed_candidates": len(failed),
                "n_candidates": len(rows) + len(failed),
                "parse_ok": "true" if not failed else "false",
            }
        ],
    )


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_summary(output_dir: Path, rows: list[dict[str, object]], failed: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("parse_status") == "ok"]
    summary = []
    for cls in CLASS_KEYS:
        cls_rows = [row for row in ok if row.get("true_class") == cls]
        correct = sum(1 for row in cls_rows if row.get("predicted_class_he") == cls and row.get("top_score_tie_he") != "true")
        target_scores = [int(row.get(f"{cls}_he_score", 0)) for row in cls_rows]
        summary.append(
            {
                "true_class": cls,
                "n": len(cls_rows),
                "he_top1_correct": correct,
                "target_he_score_mean": round(sum(target_scores) / len(target_scores), 3) if target_scores else "",
                "target_he_score_min": min(target_scores) if target_scores else "",
                "target_he_score_max": max(target_scores) if target_scores else "",
            }
        )
    write_csv(output_dir / "step2_he_morphology_by_true_class.csv", summary)
    write_csv(
        output_dir / "step2_run_gate.csv",
        [
            {
                "parsed_rows": len(ok),
                "failed_rows": len(failed),
                "n_rows": len(rows) + len(failed),
                "parse_ok": "true" if not failed else "false",
            }
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-csv", type=Path, required=True)
    parser.add_argument("--step1-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--image-mode", choices=["he_crop_context", "he_crop_only"], default="he_crop_context")
    parser.add_argument(
        "--prompt-style",
        choices=[
            "detailed",
            "simple_verify",
            "target_verify",
            "target_verify_checklist",
            "all_hypothesis_verify",
            "all_hypothesis_soft_keep",
            "all_hypothesis_partial_piece",
            "all_hypothesis_retain_uncertain",
            "all_hypothesis_graded_compat",
            "all_hypothesis_visible_compatible",
            "all_hypothesis_soft_narrow",
            "all_hypothesis_graded_stroma_lenient",
            "all_hypothesis_three_level",
            "all_hypothesis_structure_evidence",
            "all_hypothesis_alveoli_narrow",
            "all_hypothesis_class_specific_rescue",
            "all_hypothesis_context_rescue",
            "all_hypothesis_broad_class_veto",
            "all_hypothesis_direct_evidence_minrules",
            "comparative_hypothesis_veto",
            "comparative_soft_keep",
            "comparative_ranked_retention",
            "comparative_pattern_first",
            "comparative_primary_pattern_budget",
            "comparative_structural_priority",
            "comparative_false_positive_guard",
            "comparative_soft_broad_veto_retention",
            "comparative_fewshot_broad_veto",
            "comparative_structured_checklist",
            "comparative_minimal_retention",
            "comparative_structural_keep_broad_prune",
            "comparative_brief_class_definitions",
            "comparative_alveoli_septa_rescue",
            "comparative_stroma_immune_strict_veto",
            "comparative_context_minimal_retention",
            "comparative_context_soft_broad_veto",
            "comparative_context_structured_checklist",
        ],
        default="detailed",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_rows = read_csv(args.requests_csv)
    step1_rows = {(row["row_index"], row["candidate_uid"]): row for row in read_csv(args.step1_csv)}
    joined = []
    for row in request_rows:
        key = (row["row_index"], row["candidate_uid"])
        if key not in step1_rows:
            raise KeyError(f"Missing Step 1 row for {key}")
        joined.append({**row, **{f"step1_{k}": v for k, v in step1_rows[key].items()}, "_step1": step1_rows[key]})
    if args.limit:
        joined = joined[: args.limit]

    out_path = args.output_dir / "step2_he_morphology_scores.csv"
    retention_path = args.output_dir / "step2_candidate_retention_summary.csv"
    hypothesis_path = args.output_dir / "step2_hypothesis_verification_scores.csv"
    failed_path = args.output_dir / "failed_rows.csv"
    existing_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    hypothesis_rows: list[dict[str, object]] = []
    done: set[str] = set()
    if args.resume and out_path.exists():
        existing_rows = read_csv(out_path)
        done = {str(row.get("row_index")) for row in existing_rows}
    if args.resume and retention_path.exists():
        existing_rows = read_csv(retention_path)
        done = {str(row.get("row_index")) for row in existing_rows}
    if args.resume and hypothesis_path.exists():
        hypothesis_rows = read_csv(hypothesis_path)

    vlm = load_vlm(args.model, args.device)
    all_out = list(existing_rows)
    for idx, row in enumerate(joined, start=1):
        if str(row["row_index"]) in done:
            continue
        uid = row["candidate_uid"]
        he_crop = args.image_root / row["he_crop_rel"]
        he_context = args.context_root / "views" / f"{uid}_he_local_context.jpg"
        try:
            if not he_crop.exists():
                raise FileNotFoundError(f"missing H&E crop: {he_crop}")
            if args.prompt_style in COMPARATIVE_PROMPTS:
                retained = step1_retained_hypotheses(row["_step1"])
                step1_map = step1_hypotheses_json(row["_step1"])
                prompt = COMPARATIVE_PROMPTS[args.prompt_style].format(
                    retained_hypotheses=", ".join(retained),
                    step1_hypothesis_json=step1_map,
                )
                images = [open_rgb(he_crop)]
                if "context" in args.prompt_style:
                    if not he_context.exists():
                        raise FileNotFoundError(f"missing H&E context: {he_context}")
                    images.append(open_rgb(he_context))
                raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
                per_candidate, parsed_summary = parse_comparative_response(raw, retained)
                true_class = str(row.get("true_class", ""))
                for parsed_target in per_candidate:
                    hyp_out = {
                        **{k: v for k, v in row.items() if k != "_step1"},
                        **parsed_target,
                        "parse_status": "ok",
                        "model": args.model,
                        "prompt_style": args.prompt_style,
                        "raw_response": raw,
                    }
                    hypothesis_rows.append(hyp_out)
                write_csv(hypothesis_path, hypothesis_rows)
                he_retained = [
                    str(item["target_hypothesis_class"])
                    for item in per_candidate
                    if item.get("he_supported") == "true"
                ]
                true_hyp = next((item for item in per_candidate if item.get("target_hypothesis_class") == true_class), None)
                top_hyp = max(per_candidate, key=lambda item: int(item.get("he_support_score", 0) or 0))
                out = {
                    **{k: v for k, v in row.items() if k != "_step1"},
                    **parsed_summary,
                    "step1_retained_classes": ";".join(retained),
                    "step1_retained_count": len(retained),
                    "true_retained_after_step1": str(true_class in retained).lower(),
                    "he_retained_classes": ";".join(he_retained),
                    "he_retained_count": len(he_retained),
                    "true_retained_after_step2": str(true_class in he_retained).lower(),
                    "true_class_he_support_score": true_hyp.get("he_support_score", "") if true_hyp else "",
                    "true_class_he_supported": true_hyp.get("he_supported", "false") if true_hyp else "false",
                    "top_he_support_class": top_hyp.get("target_hypothesis_class", ""),
                    "top_he_support_score": top_hyp.get("he_support_score", ""),
                    "parse_status": "ok",
                    "model": args.model,
                    "prompt_style": args.prompt_style,
                    "raw_response": raw,
                }
                all_out.append(out)
                write_csv(retention_path, all_out)
                print(
                    f"Step2 comparative {idx}/{len(joined)} {uid} true={true_class} "
                    f"step1_retained={true_class in retained} step2_retained={true_class in he_retained} ok",
                    flush=True,
                )
                continue
            if args.prompt_style in ALL_HYPOTHESIS_PROMPTS:
                retained = step1_retained_hypotheses(row["_step1"])
                step1_map = step1_hypotheses_json(row["_step1"])
                per_candidate = []
                for target_class in retained:
                    prompt = ALL_HYPOTHESIS_PROMPTS[args.prompt_style].format(
                        target_hypothesis=target_class,
                        target_step1_status=str(row["_step1"].get(target_class, "")),
                        step1_hypothesis_json=step1_map,
                    )
                    if args.prompt_style == "all_hypothesis_context_rescue":
                        if not he_context.exists():
                            raise FileNotFoundError(f"missing H&E context: {he_context}")
                        images = [open_rgb(he_crop), open_rgb(he_context)]
                    else:
                        images = [open_rgb(he_crop)]
                    raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
                    parsed_target = parse_target_verify_response(raw, target_class)
                    hyp_out = {
                        **{k: v for k, v in row.items() if k != "_step1"},
                        **parsed_target,
                        "parse_status": "ok",
                        "model": args.model,
                        "prompt_style": args.prompt_style,
                        "raw_response": raw,
                    }
                    hypothesis_rows.append(hyp_out)
                    write_csv(hypothesis_path, hypothesis_rows)
                    per_candidate.append(hyp_out)
                he_retained = [str(item["target_hypothesis_class"]) for item in per_candidate if item.get("he_supported") == "true"]
                true_class = str(row.get("true_class", ""))
                true_hyp = next((item for item in per_candidate if item.get("target_hypothesis_class") == true_class), None)
                top_hyp = max(
                    per_candidate,
                    key=lambda item: int(item.get("he_support_score", item.get("top_score_he", 0)) or 0),
                ) if per_candidate else {}
                out = {
                    **{k: v for k, v in row.items() if k != "_step1"},
                    "step1_retained_classes": ";".join(retained),
                    "step1_retained_count": len(retained),
                    "true_retained_after_step1": str(true_class in retained).lower(),
                    "he_retained_classes": ";".join(he_retained),
                    "he_retained_count": len(he_retained),
                    "true_retained_after_step2": str(true_class in he_retained).lower(),
                    "true_class_he_support_score": true_hyp.get("he_support_score", "") if true_hyp else "",
                    "true_class_he_supported": true_hyp.get("he_supported", "false") if true_hyp else "false",
                    "top_he_support_class": top_hyp.get("target_hypothesis_class", ""),
                    "top_he_support_score": top_hyp.get("he_support_score", ""),
                    "parse_status": "ok",
                    "model": args.model,
                    "prompt_style": args.prompt_style,
                }
                all_out.append(out)
                write_csv(retention_path, all_out)
                print(
                    f"Step2 {idx}/{len(joined)} {uid} true={true_class} "
                    f"step1_retained={true_class in retained} step2_retained={true_class in he_retained} ok",
                    flush=True,
                )
                continue
            if args.prompt_style == "simple_verify":
                prompt = USER_TEMPLATE_SIMPLE_HE_CROP_ONLY.format(
                    step1_hypothesis_json=step1_hypotheses_json(row["_step1"])
                )
                images = [open_rgb(he_crop)]
                raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
                parsed = parse_response(raw)
            elif args.prompt_style in {"target_verify", "target_verify_checklist"}:
                target_class = step1_target_hypothesis(row["_step1"])
                if args.prompt_style == "target_verify_checklist":
                    prompt = USER_TEMPLATE_TARGET_VERIFY_CHECKLIST_HE_CROP_ONLY.format(
                        target_hypothesis=target_class,
                        step1_hypothesis_json=step1_hypotheses_json(row["_step1"]),
                        target_checklist=TARGET_HE_CHECKLISTS[target_class],
                    )
                else:
                    prompt = USER_TEMPLATE_TARGET_VERIFY_HE_CROP_ONLY.format(
                        target_hypothesis=target_class,
                        step1_hypothesis_json=step1_hypotheses_json(row["_step1"]),
                    )
                images = [open_rgb(he_crop)]
                raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
                parsed = parse_target_verify_response(raw, target_class)
            elif args.image_mode == "he_crop_context":
                if not he_context.exists():
                    raise FileNotFoundError(f"missing H&E context: {he_context}")
                prompt = USER_TEMPLATE.format(step1_hypothesis_json=step1_json(row["_step1"]))
                images = [open_rgb(he_crop), open_rgb(he_context)]
                raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
                parsed = parse_response(raw)
            else:
                prompt = USER_TEMPLATE_HE_CROP_ONLY.format(step1_hypothesis_json=step1_json(row["_step1"]))
                images = [open_rgb(he_crop)]
                raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
                parsed = parse_response(raw)
            out = {
                **{k: v for k, v in row.items() if k != "_step1"},
                **parsed,
                "parse_status": "ok",
                "model": args.model,
                "raw_response": raw,
            }
            if args.prompt_style in {"target_verify", "target_verify_checklist"}:
                target_matches_truth = out.get("target_hypothesis_class") == row.get("true_class")
                supported = out.get("he_supported") == "true"
                out["target_matches_truth"] = str(target_matches_truth).lower()
                out["verification_decision_correct"] = str((target_matches_truth and supported) or ((not target_matches_truth) and (not supported))).lower()
                out["is_correct_he"] = str(target_matches_truth and supported).lower()
            else:
                out["is_correct_he"] = str(out.get("predicted_class_he") == row.get("true_class") and out.get("top_score_tie_he") != "true").lower()
            all_out.append(out)
            write_csv(out_path, all_out)
            print(f"Step2 {idx}/{len(joined)} {uid} true={row.get('true_class')} pred={out.get('predicted_class_he')} ok", flush=True)
        except Exception as exc:
            fail = {**{k: v for k, v in row.items() if k != "_step1"}, "error": str(exc)[:1000], "parse_status": "failed"}
            failed_rows.append(fail)
            write_csv(failed_path, failed_rows)
            print(f"FAILED Step2 {idx}/{len(joined)} {uid}: {exc}", flush=True)

    if args.prompt_style in ALL_HYPOTHESIS_PROMPTS or args.prompt_style in COMPARATIVE_PROMPTS:
        write_csv(retention_path, all_out)
        write_csv(hypothesis_path, hypothesis_rows)
        build_retention_summary(args.output_dir, all_out, failed_rows)
    else:
        write_csv(out_path, all_out)
        build_summary(args.output_dir, all_out, failed_rows)
    write_csv(failed_path, failed_rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "requests_csv": str(args.requests_csv),
                "step1_csv": str(args.step1_csv),
                "image_root": str(args.image_root),
                "context_root": str(args.context_root),
                "image_mode": args.image_mode,
                "prompt_style": args.prompt_style,
                "limit": args.limit,
            },
            indent=2,
        )
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
