# Official Filtered FICTURE Final Bests

Official preflight: PASS_OFFICIAL; ROI 3144 x 3327; factor index 3327 x 3144; no manual shift; formula uses microns_per_pixel and tissue_hires_scalef.

## Overall Best By Dice

| tissue class | best Dice | Precision | Recall | source | method/run | ranker | top_k |
|---|---:|---:|---:|---|---|---|---:|
| bronchiola | 0.769 | 0.766 | 0.771 | multimodal_candidate_ranking | multimodal_candidate_ranking: official_ficture_consensus_fast_11946275 | molecular_only | 8 |
| alveoli | 0.604 | 0.742 | 0.509 | candidate_oracle | SAM on FICTURE ROI, box prompts (512 px box, 160 px stride) | candidate_by_dice | 1 |
| vessels | 0.807 | 0.738 | 0.892 | multimodal_candidate_ranking | multimodal_candidate_ranking: official_ficture_consensus_cpu_recheck_11956202 | alveoli_clip_precision | 3 |
| tumor | 0.577 | 0.646 | 0.520 | multimodal_candidate_ranking | multimodal_candidate_ranking: official_ficture_lowoverlap_quick_11949262 | high_recall | 20 |
| stroma | 0.528 | 0.371 | 0.911 | candidate_oracle | SAM on FICTURE ROI, point prompts (24 px spacing) | candidate_by_dice | 1 |
| immune infiltration | 0.396 | 0.350 | 0.457 | multimodal_candidate_ranking | multimodal_candidate_ranking: official_ficture_consensus_cpu_recheck_11956202 | alveoli_precision_prior | 8 |
| erythrocytes | 0.173 | 0.196 | 0.155 | candidate_oracle | SAM on FICTURE ROI, box prompts (512 px box, 160 px stride) | candidate_by_dice | 1 |
| pigment | 0.153 | 0.244 | 0.111 | candidate_oracle | Medical-SAM3 on FICTURE ROI, point prompts (24 px spacing) | candidate_by_dice | 1 |

## Candidate Oracle Focus Bests

- bronchiola: Dice 0.758, P 0.868, R 0.674; SAM on FICTURE ROI, point prompts (32 px spacing) (base_official_points_step32/303)
- alveoli: Dice 0.604, P 0.742, R 0.509; SAM on FICTURE ROI, box prompts (512 px box, 160 px stride) (base_box512_s160_m2048/31)
- vessels: Dice 0.626, P 0.922, R 0.474; Medical-SAM3 on FICTURE ROI, box prompts (512 px box, 160 px stride) (medical_box512_s160_m1536/25)

## Ranking/VLM Focus Bests

- bronchiola: Dice 0.769, P 0.766, R 0.771; official_ficture_consensus_fast_11946275 / molecular_only top-8
- alveoli: Dice 0.239, P 0.141, R 0.772; official_ficture_consensus_fast_11946275 / vessel_elongated_prior top-2
- vessels: Dice 0.807, P 0.738, R 0.892; official_ficture_consensus_cpu_recheck_11956202 / alveoli_clip_precision top-3

## Files

- `official_ficture_candidate_best_by_dice.csv`: single-candidate oracle table.
- `official_ranking_vlm_best_by_dice.csv`: best retrieve/VLM output per class.
- `official_overall_best_by_dice.csv`: best by Dice across candidate oracle and retrieve/VLM.
- `six_panel/`: eight six-panel candidate-oracle figures.
- Official summary: `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json`.
