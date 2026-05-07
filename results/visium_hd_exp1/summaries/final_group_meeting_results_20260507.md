# Medical-SAM3 VisiumHD Exp1 Final Results Snapshot - 2026-05-07

## Completion Status

- Current HPC queue was checked and `squeue -u hw646` was empty.
- All required overnight experiments for the current group-meeting batch completed.
- Original retrieval job `11037672` had OOM subtasks because the old evaluator materialized a full query-by-gallery similarity matrix.
- OOMed retrieval subtasks were completed with exact chunked top-k retry jobs `11039815`, `11040086`, and `11045911`.
- HPC project archive with full tables is saved at:
  `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/summaries/final_group_meeting_results_20260507.md`
- Machine-readable archive is saved at:
  `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/summaries/final_group_meeting_results_20260507.json`

## Stable HPC Snapshot Files

- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/summaries/latest_results_snapshot.json`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/summaries/latest_training_summary.csv`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/summaries/latest_retrieval_summary.csv`

Final snapshot counts: 33 training rows and 28 retrieval rows.

## Presentation Files

- Final PPT:
  `/Users/haoranwu/Desktop/Yan_Lab_Research/Presentation/Medical_SAM3_GroupMeeting_20260507/Medical_SAM3_Training_Ablation_GroupMeeting_20260507.pptx`
- Project copy:
  `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/inference/visium_hd_exp1/Medical_SAM3_Training_Ablation_GroupMeeting_20260507.pptx`
- Notion page:
  `https://www.notion.so/3596bfe3ade481599fcbf2b2bd2f1eea`

## Visual Assets Checked Locally

- `results/visium_hd_exp1/figures/group_meeting/classification_loss_components.png`
- `results/visium_hd_exp1/figures/group_meeting/gigapath_training_curves.png`
- `results/visium_hd_exp1/figures/group_meeting/presentation_3class_total_loss_decomposition.png`
- `results/visium_hd_exp1/figures/group_meeting/presentation_3class_vs_8class_metrics.png`
- `results/visium_hd_exp1/figures/group_meeting/presentation_expression_ce_generalization_gap.png`
- `results/visium_hd_exp1/figures/group_meeting/retrieval_overall_recall.png`
- `results/visium_hd_exp1/figures/group_meeting/gigapath_3class_per_label_recall1.png`
- `results/visium_hd_exp1/figures/group_meeting/resnet_contrastive_loss.png`

The current PPT has 21 slides, 30 PowerPoint tables, and no `Missing figure` placeholder.

## Main Results To Present

### Training Strategy

| Run | Expr acc | Expr macro-F1 | Image acc | Interpretation |
|---|---:|---:|---:|---|
| Frozen constant | 0.7568 | 0.6799 | 0.6912 | baseline frozen GigaPath |
| Frozen warmup/cosine | 0.7532 | 0.6796 | 0.6908 | schedule alone did not help |
| Last-1 warmup/cosine | 0.7652 | 0.6929 | 0.7920 | fine-tuning helps image branch |
| Last-2 warmup/cosine | 0.7652 | 0.6945 | 0.8312 | best supervised/fused-probe candidate |
| 8-class last-2 warmup/cosine | 0.6316 | 0.4512 | 0.7084 | improves image branch but 8-class remains hard |

### 3-Class Retrieval

| Run | R@1 | R@5 | R@10 | Best fused R@1 |
|---|---:|---:|---:|---:|
| GigaPath CE-only | 0.745 | 0.811 | 0.848 | 0.856 |
| GigaPath InfoNCE 0.10 | 0.722 | 0.858 | 0.900 | 0.858 |
| Last-1 fine-tune | 0.741 | 0.833 | 0.870 | 0.874 |
| Last-2 fine-tune | 0.728 | 0.837 | 0.865 | 0.880 |

### GigaPath 8-Class Retrieval

| Run | R@1 | R@5 | R@10 | Best fused R@1 |
|---|---:|---:|---:|---:|
| CE-only | 0.605 | 0.719 | 0.754 | 0.842 |
| InfoNCE 0.05 | 0.601 | 0.729 | 0.768 | 0.843 |
| InfoNCE 0.10 | 0.613 | 0.730 | 0.769 | 0.841 |
| InfoNCE 0.20 | 0.603 | 0.737 | 0.777 | 0.842 |
| InfoNCE 0.50 | 0.587 | 0.720 | 0.757 | 0.844 |
| InfoNCE-only | 0.538 | 0.724 | 0.775 | 0.843 |

## Interpretation

- GigaPath crop128 3-class is the strongest and most stable main line.
- CE supervision is still needed because the expression tower is not a pretrained molecular anchor.
- InfoNCE helps broader top-k/fused retrieval more than raw nearest-neighbor R@1.
- Last-2 warmup/cosine is the best candidate for SAM3 proposal/mask ranking.
- CE-only and last-1 remain important raw retrieval baselines.
- 8-class remains harder than 3-class, consistent with fine-label ambiguity, class imbalance, and rare classes.
