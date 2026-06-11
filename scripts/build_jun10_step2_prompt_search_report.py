#!/usr/bin/env python3
"""Build a plain-language report for Jun10 Step2 prompt-search design."""

from __future__ import annotations

from pathlib import Path


OUT = Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_prompt_search_design_report.md")


ROWS = [
    (
        "all_hypothesis_broad_class_veto",
        "per-hypothesis structured rule",
        "Verify each Step1 class separately; stricter direct evidence for tumor/stroma/immune.",
        "Tests whether stronger broad-class veto reduces over-retention.",
    ),
    (
        "all_hypothesis_direct_evidence_minrules",
        "minimal-rule prompting",
        "Short prompt with only direct H&E evidence criteria.",
        "Tests whether earlier prompts were too complex and caused broad over-keeping.",
    ),
    (
        "comparative_hypothesis_veto",
        "comparative prompting",
        "Compare Step1 hypotheses together instead of judging each class in isolation.",
        "Targets the failure where many classes look plausible independently.",
    ),
    (
        "comparative_soft_keep",
        "comparative soft retention",
        "Compare hypotheses but keep a small plausible set without forcing one class.",
        "Balances true-class retention against narrowing.",
    ),
    (
        "comparative_ranked_retention",
        "ranked structured output",
        "Rank H&E support and usually keep only one to three classes.",
        "Tests a stronger narrowing rule; risky for weak classes.",
    ),
    (
        "comparative_pattern_first",
        "pattern-first reasoning",
        "Identify the visible H&E pattern first, then map pattern to class.",
        "Makes the skill more standardized than free-form class scoring.",
    ),
    (
        "comparative_primary_pattern_budget",
        "pattern-first with decision budget",
        "Keep only dominant or genuinely mixed patterns.",
        "A hard narrowing prompt; useful if broad over-retention dominates.",
    ),
    (
        "comparative_structural_priority",
        "class-priority prompting",
        "Prioritize partial airway/vessel/septal structures over broad fallback labels.",
        "Protects bronchiola/vessels/alveoli pieces from tumor/stroma/immune drift.",
    ),
    (
        "comparative_false_positive_guard",
        "failure-informed prompting",
        "Explicitly lists the current false-positive patterns and asks the model to avoid them.",
        "Directly targets the measured false-kept matrix.",
    ),
    (
        "comparative_soft_broad_veto_retention",
        "soft retention plus targeted veto",
        "No fixed top-k; keep plausible classes but reject tumor/stroma/immune without direct morphology.",
        "Best-designed candidate after offline policy analysis showed hard top-k loses too many true classes.",
    ),
    (
        "comparative_fewshot_broad_veto",
        "few-shot / counterexample prompting",
        "Gives text examples of how to treat wall pieces, septa, tumor, stroma, and immune aggregates.",
        "Tests whether examples calibrate the model better than rules alone.",
    ),
    (
        "comparative_structured_checklist",
        "checklist prompting",
        "Asks the model to internally check structure, broad pattern, and unsupported Step1 hypotheses.",
        "Tests whether a fixed skill checklist stabilizes decisions.",
    ),
    (
        "comparative_minimal_retention",
        "minimal prompting",
        "Short retention-gate instruction with almost no class-specific rules.",
        "Tests whether previous prompts were too rule-heavy and caused unnecessary deletion.",
    ),
    (
        "comparative_structural_keep_broad_prune",
        "weak-class preserving prompt",
        "Explicitly keeps possible bronchiola/alveoli/vessel structural fragments while pruning unsupported tumor/stroma/immune.",
        "Targets the weak-class failure where small wall/septal pieces are converted into broad fallback classes.",
    ),
    (
        "comparative_brief_class_definitions",
        "short class-description prompt",
        "Adds compact H&E definitions for all six classes without long examples.",
        "Tests whether a little more class description helps without making the prompt overcomplicated.",
    ),
    (
        "comparative_alveoli_septa_rescue",
        "failure-driven alveoli rescue",
        "Protects thin septa, delicate alveolar mesh, airway wall, and vessel edge pieces from being over-called as stroma or immune.",
        "Targets existing misses where alveoli/bronchiola/vessels were deleted and replaced by stroma or immune.",
    ),
    (
        "comparative_stroma_immune_strict_veto",
        "failure-driven broad-label veto",
        "Keeps stroma or immune only with direct candidate-mask evidence, not just nearby cells or generic wall tissue.",
        "Targets measured over-retention of stroma and immune_infiltration.",
    ),
    (
        "comparative_context_minimal_retention",
        "minimal context engineering",
        "Uses candidate crop plus local context with only a short instruction.",
        "Tests whether context itself helps, independent of heavier context-specific rules.",
    ),
    (
        "comparative_context_soft_broad_veto",
        "context engineering / two-image H&E input",
        "Uses candidate crop plus local H&E context, while judging only the candidate mask.",
        "Tests whether local structure context helps partial airway/vessel/septal pieces without overusing broad labels.",
    ),
    (
        "comparative_context_structured_checklist",
        "context-aware checklist prompting",
        "Runs a checklist on candidate appearance and whether it is attached to a larger structure in context.",
        "Tests whether context helps when the isolated piece lacks full lumen or septal structure.",
    ),
    (
        "all_hypothesis_context_rescue",
        "per-hypothesis context rescue",
        "Checks each Step1 hypothesis separately with candidate crop plus local H&E context.",
        "Tests whether small structural pieces are easier to retain when the model verifies one hypothesis at a time.",
    ),
]


def main() -> None:
    lines = [
        "# Jun10 Step2 prompt-search design",
        "",
        "## Experiment Design",
        "",
        "**Goal:** improve Step2 of the three-step tissue filtering flow.",
        "",
        "**Three-step flow:** Step1 uses FICTURE-derived cell-type/gene composition text to generate tissue hypotheses. Step2 uses H&E morphology to verify those hypotheses. Step3 optionally checks FICTURE image/spatial consistency.",
        "",
        "**Current failure:** Step2 keeps too many broad fallback classes, especially tumor, stroma, and immune infiltration, while strict prompts can delete true bronchiola, alveoli, and vessels.",
        "",
        "**Weak set:** 58 candidates: all bronchiola, alveoli, and vessels candidates plus five tumor, five stroma, and five immune infiltration candidates.",
        "",
        "**Ground truth use:** true class, Dice, Precision, and Recall are hidden from the model and used only for evaluation.",
        "",
        "**Input ablations:** the main prompt sweep uses one H&E gray reverse-blur crop. A separate context sweep uses two H&E images: candidate crop plus local context crop. Context is only allowed to support whether the candidate is part of a larger structure.",
        "",
        "## Prompt Variants",
        "",
        "| Prompt style | Method type | What it asks | Why test it |",
        "|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Adoption Gate",
            "",
            "A prompt is worth expanding to full167 only if it keeps at least 56/58 true classes and reduces the mean number of remaining classes compared with the weak58 baseline.",
            "",
            "## Current local status",
            "",
            "The prompt preview audit has 58 candidates and 12 main H&E-only prompt styles. A separate context-preview audit checks the context-aware prompts and confirms that the second image is available before submission.",
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
