# Visium HD Exp1 Training Summary Figures

Generated from saved project results. Key source files are `history.json`, `summary.json`, and `metrics.json` under the project results directory.

## Figures
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/gigapath_training_curves.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/classification_loss_components.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/classification_loss_components.csv`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/presentation_3class_vs_8class_metrics.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/presentation_3class_total_loss_decomposition.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/presentation_expression_ce_generalization_gap.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/resnet_contrastive_loss.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/retrieval_overall_recall.png`
- `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/figures/group_meeting/gigapath_3class_per_label_recall1.png`

## Key Completed Metrics
- GigaPath crop64 8-class: best epoch 3, val expr acc 0.5948, val image acc 0.6344, expr macro-F1 0.4368
- GigaPath crop128 8-class: best epoch 4, val expr acc 0.6100, val image acc 0.5328, expr macro-F1 0.4366
- GigaPath crop128 3-class: best epoch 2, val expr acc 0.7544, val image acc 0.6800, expr macro-F1 0.6724
- GigaPath crop128 3-class retrieval: recall@1 0.7406, recall@5 0.8301, recall@10 0.8634, concat linear probe 0.8325
- ResNet50 contrastive 8-class retrieval: recall@1 0.5571, recall@5 0.8219, recall@10 0.8851, concat linear probe 0.7266
