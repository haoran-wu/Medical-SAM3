# VisiumHD Exp1 Report Manifest

This manifest lists the report folders that currently matter for the VisiumHD Exp1
project. It is a guide for what to cite, present, keep, archive, or avoid committing.

## Main reports to use now

The main result is:

```text
official FICTURE aligned to H&E same ROI
-> build H&E/FICTURE candidate pool
-> component-aware / precision-aware union
-> Test1 and Test2 VLM evaluation
```

This is the current storyline for presentation and research writing. Older CLIP-only,
prompt-ablation, pairwise, and smoke runs are supporting evidence or archive.

| Priority | Path | Use |
|---|---|---|
| Primary | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html` | Best current readable report: component-aware union, Test1, Test2, example prompt, and visualizations |
| Primary | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_unified_score_prompt_all_models_final/index.html` | Compact all-model summary for unified score prompt |
| Primary | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_GPT55_unified_score_prompt_full90/index.html` | GPT-5.5 full 90-candidate API result |
| Primary | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_Gemini31_unified_score_prompt_full90/index.html` | Gemini 3.1 full 90-candidate API result |
| Primary | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_precision_gated_component_assembly/index.html` | Component-aware assembly source report and visualizations |
| Primary | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May30_component_assembly_need_diagnostic_verified_maybe/index.html` | Decision logic for which classes should use component assembly |

## Historical reports worth keeping as archive

| Archive path | Why keep it |
|---|---|
| `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May28_aligned_candidate_pool_best_visualization/index.html` | Earlier aligned candidate-pool best visualization |
| `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May28_VLM_clean_direct_retrieval_report_with_GPT55_Gemini31/index.html` | Earlier VLM retrieval summary with GPT/Gemini |
| `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May27_GPT55_cross_label_top1_accuracy/index.html` | Previous GPT-5.5 cross-label result before unified prompt |
| `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May27_Gemini31_cross_label_top1_accuracy/index.html` | Previous Gemini 3.1 cross-label result before unified prompt |
| `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/final_deliverables/May22_OpenAI_CLIP_results` | CLIP baseline and earlier retrieval evidence |

## Debug or smoke outputs

These are useful only for troubleshooting and should not be treated as main results:

- Any folder containing `smoke`.
- Early May22-May25 pairwise or prompt-ablation folders unless specifically cited.
- Raw `api_responses.jsonl`.
- Intermediate crop pools that are only used to reproduce a report.

## What is already pushed to GitHub from `output/`

The repo currently tracks selected files under:

```text
output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report
output/visium_hd_exp1/final_deliverables/current_assets_dice08_masks
output/visium_hd_exp1/no_training_multirun_8class
output/visium_hd_exp1/saved_snapshots
output/visium_hd_exp1/weak_transfer_pseudoseg_competition_calibrated
```

Everything else under `output/` is ignored unless force-added intentionally.

## Main presentation narrative

Use this order for slides and reports:

1. Start from the concrete H&E + FICTURE paired example in the same ROI.
2. Build the H&E/FICTURE candidate mask pool.
3. Use component-aware union for bronchiola/vessels, and precision-aware subset
   selection for tumor/stroma/immune where taking every component can hurt Precision.
4. Run Test1: Cross-Label Tissue Classification, asking whether VLMs recognize tissue type.
5. Run Test2: Same-Class Candidate Mask Retrieval, asking whether VLMs can rank better masks higher.
6. Limitations: stroma and immune infiltration remain hard because candidate quality and label structure are weaker.

Short version:

```text
official alignment -> example -> candidate pool -> union -> Test1 -> Test2
```
