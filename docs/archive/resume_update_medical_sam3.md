# Resume Update Draft: Medical-SAM3 / Yale School of Medicine

## Recommended 5-Bullet Version

Research Assistant, Yale School of Medicine | Supervisors: Prof. Xiting Yan, Prof. Xiao Luo

- Developed end-to-end computational pathology pipelines for SAM3/Medical-SAM3 evaluation on TMA/Xenium and Visium HD histology, including text, box, point, hybrid, dense-proposal, and oracle-style prompting workflows.
- Converted spatial transcriptomics and pathology annotations into image-space supervision by mapping Xenium cell coordinates, rasterizing expert GeoJSON regions, projecting FICTURE molecular factor maps onto H&E, and exporting binary/non-overlapping masks for quantitative segmentation evaluation.
- Benchmarked SAM3 and Medical-SAM3 on histopathology region segmentation, showing that whole-image box prompts produced usable signal while text-only prompts and cross-region transfer were weak; used these results to redirect the project toward molecularly guided proposal ranking rather than direct text prompting.
- Built a slide-specific H&E + FICTURE multi-scale ExtraTrees segmentation pipeline for Visium HD Exp1, achieving original-resolution non-overlapping masks across 8 pathology regions with Dice scores from 0.803 to 0.987 and zero overlap pixels in the final mask package.
- Trained and evaluated cross-modal H&E-expression models with ResNet50 contrastive learning and GigaPath dual-tower classification/alignment, reaching 0.741 Recall@1 / 0.830 Recall@5 / 0.863 Recall@10 for 3-class expression-to-image retrieval and 0.833 concat linear-probe accuracy.

## Shorter 3-Bullet Version

Research Assistant, Yale School of Medicine | Supervisors: Prof. Xiting Yan, Prof. Xiao Luo

- Built SAM3/Medical-SAM3 histopathology segmentation pipelines across TMA/Xenium and Visium HD data, converting Xenium coordinates, FICTURE factors, and expert GeoJSON annotations into image-space masks and prompt/evaluation targets.
- Designed molecularly guided segmentation workflows combining H&E morphology and spatial transcriptomics; produced original-resolution non-overlapping Visium HD masks for 8 pathology regions with Dice 0.803-0.987 on the annotated Exp1 slide.
- Trained ResNet50 and GigaPath H&E-expression dual-tower models for cross-modal retrieval/classification, achieving 0.741 Recall@1, 0.863 Recall@10, and 0.833 concat linear-probe accuracy on 3-class expression-to-image retrieval.

## If You Want To Emphasize Engineering

- Built reproducible Python/HPC workflows for large-scale segmentation experiments, including SLURM batch scripts, result summarization, retrieval evaluation, visualization assets, PowerPoint report generation, and local HPC job-monitoring utilities.

## Conservative Claim Notes

- Safe to claim: one annotated Visium HD Exp1 slide reached high-quality reconstruction metrics using H&E + FICTURE + GeoJSON-derived supervision.
- Safe to claim: TMA24/SAM3 prompt experiments found box prompts useful, text-only prompts weak, and cross-region transfer near zero.
- Safe to claim: 25 TMA samples exist in the broader data setting, but current documents say cross-slide validated Dice above 0.8 has not been proven.
- Avoid saying: "validated across 25 TMAs" or "generalizes to all 24 unlabeled samples" unless new validation results are added.
