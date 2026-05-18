# VisiumHD Exp1 Official Filtered FICTURE Results - 2026-05-17

This directory contains lightweight, git-trackable result artifacts for the
official filtered FICTURE rerun. Raw candidate pools and generated masks are too
large for normal GitHub storage, so they remain on Bouchet and the local
Desktop deliverable folder.

## Official Status

- Official FICTURE summary: `output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json`
- Required status: `PASS_OFFICIAL`
- Official candidate input: `output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi`
- ROI size: `3144 x 3327`
- Factor-index shape: `3327 x 3144`
- Official source: `pixel-level cell type image/visiumhd_exp1_hex12_k12/hex_12.k12.pixel.png`
- Transform: `np.fliplr(raw filtered PNG)`, then
  `he_x = y_um / microns_per_pixel * tissue_hires_scalef` and
  `he_y = x_um / microns_per_pixel * tissue_hires_scalef`
- No manual `dx/dy`, no full-canvas resize, no ROI-size substitution.

## Raw Candidate Pool Locations

These raw outputs are intentionally not committed because they contain many
generated masks and large CSVs.

- Official FICTURE candidate pool:
  `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/ficture_official_filtered_candidate_pool`
- Official FICTURE candidate input:
  `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi`
- Historical H&E candidate pool:
  `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/he_coord_scaled_hires_candidate_pool`

The historical H&E candidate pool is retained for traceability, but the
official FICTURE ranking script intentionally uses `CANDIDATE_MODE=ficture`
only because older H&E roots were generated against a different ROI size and
must not be mixed into official filtered-FICTURE final results.

## Pipeline Files

- Official FICTURE-to-H&E renderer:
  `inference/visium_hd_exp1/render_official_filtered_ficture_he_map.py`
- Official candidate-pool ROI builder:
  `inference/visium_hd_exp1/build_official_ficture_candidate_pool_input.py`
- Official FICTURE SAM/Medical-SAM3 candidate array:
  `scripts/hpc/visium_hd_exp1_ficture_official_candidate_array.sbatch`
- Official retrieve/ranking array:
  `scripts/hpc/visium_hd_exp1_multimodal_candidate_ranking_official_ficture.sbatch`
- Official VLM judge:
  `scripts/hpc/visium_hd_exp1_vlm_candidate_judge_official_ficture.sbatch`
- Official VLM direct grounding:
  `scripts/hpc/visium_hd_exp1_vlm_direct_grounding_official_ficture.sbatch`
- Official candidate-oracle report renderer:
  `inference/visium_hd_exp1/render_official_ficture_candidate_best_report.py`
- Final deliverable packager:
  `inference/visium_hd_exp1/finalize_official_ficture_deliverable.py`
- Guardrails:
  `docs/VISIUMHD_FICTURE_GUARDRAILS.md`

## Final Metrics

Ignoring erythrocytes and pigment for presentation, current retrieve/ranking
best-by-Dice results are:

| tissue class | Dice | Precision | Recall | source run | ranker | top-k |
|---|---:|---:|---:|---|---|---:|
| bronchiola | 0.769 | 0.766 | 0.771 | `official_ficture_consensus_fast_11946275` | `molecular_only` | 8 |
| alveoli | 0.239 | 0.141 | 0.772 | `official_ficture_consensus_fast_11946275` | `vessel_elongated_prior` | 2 |
| vessels | 0.807 | 0.738 | 0.892 | `official_ficture_consensus_cpu_recheck_11956202` | `alveoli_clip_precision` | 3 |
| tumor | 0.577 | 0.646 | 0.520 | `official_ficture_lowoverlap_quick_11949262` | `high_recall` | 20 |
| stroma | 0.475 | 0.554 | 0.415 | `official_ficture_plip_highrecall_quick_11950671` | `salip_clip` | 20 |
| immune infiltration | 0.396 | 0.350 | 0.457 | `official_ficture_consensus_cpu_recheck_11956202` | `alveoli_precision_prior` | 8 |

Overall best-by-Dice, allowing the candidate oracle table when retrieve
over-segments, keeps alveoli at the candidate-oracle single-mask best:

| tissue class | Dice | Precision | Recall | source |
|---|---:|---:|---:|---|
| bronchiola | 0.769 | 0.766 | 0.771 | retrieve/ranking |
| alveoli | 0.604 | 0.742 | 0.509 | candidate oracle |
| vessels | 0.807 | 0.738 | 0.892 | retrieve/ranking |

## CLIP and VLM Use

CLIP is used inside `rank_multimodal_sam_candidates.py` only when `CLIP_MODEL`
is set. Candidate H&E crops are compared to class text prompts, and the score
feeds rankers such as `salip_clip`, `salip_clip_molecular`,
`salip_clip_he`, and `alveoli_clip_precision`.

Completed CLIP runs:

- `official_ficture_plip_11944291`: `vinid/plip`, 2400 candidates
- `official_ficture_pubmedclip_11944292`: `flaviagiammarino/pubmed-clip-vit-base-patch32`, 2400 candidates
- `official_ficture_pubmedclip_highrecall_11948970`: PubMedCLIP, 4698 candidates
- `official_ficture_plip_highrecall_quick_11950671`: PLIP, 3332 candidates

VLM is a follow-up rerank/judge after retrieve/ranking. It uses
`Qwen/Qwen2.5-VL-7B-Instruct` via:

- `inference/visium_hd_exp1/vlm_candidate_judge.py`
- `inference/visium_hd_exp1/vlm_direct_grounding.py`

VLM was run on PubMedCLIP, PLIP, high-recall, and consensus-fast score CSVs,
but did not beat the best retrieve/ranking results in the final aggregate.

## Files in This Directory

- `official_ficture_candidate_best_by_dice.csv`: single-candidate oracle table.
- `official_ranking_vlm_best_by_dice.csv`: best retrieve/VLM row per class.
- `official_retrieve_best_by_dice_with_figures.csv`: retrieve table with linked local figure paths.
- `official_overall_best_by_dice.csv`: best by Dice across candidate oracle and retrieve/VLM.
- `official_*_table.png`: compact presentation tables.
- `final_manifest.json`: final packager manifest.
