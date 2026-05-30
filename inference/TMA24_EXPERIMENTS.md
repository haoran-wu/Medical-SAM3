# TMA24 Experiment Index

This file keeps the TMA24 prompt/proposal/generalization directions separated.

## Cross-Region Generalization

Entry point:

```bash
python inference/run_tma24_left_to_right_transfer.py \
  --direction left_to_right \
  --prompt-kind box
```

Purpose: use prompts derived from one pseudo-mask region and evaluate only on a
held-out region. This checks whether a prompt in one part of
`examples/legacy_examples/tma24/example1.jpg`
helps recover similar structures elsewhere, instead of merely segmenting the
prompted area.

Supported directions:

- `left_to_right`
- `right_to_left`
- `top_to_bottom`
- `bottom_to_top`

Supported prompt geometries:

- `--prompt-kind box`
- `--prompt-kind points`

Typical output folders:

- `output/tma24_left_to_right_transfer*/`
- `output/hpc_tma24_base_sam3_generalization_*/`

## Whole-Image Prompting

Entry point:

```bash
python inference/run_tma24_whole_image_prompts.py \
  --checkpoint checkpoints/Medical-SAM3/checkpoint.pt
```

Purpose: run direct full-image recognition on
`examples/legacy_examples/tma24/example1.jpg` with:

- whole-mask-derived box prompt
- text prompt
- joint box + text prompt

Default pseudo-mask source:

- `output/00_FINAL_tma24_example1_scale_0p55_shiftX_neg120_shiftY_620/summary.json`

Default output folder:

- `output/tma24_whole_image_prompts/`

## Dense Proposal Sweeps

Dense point proposals:

```bash
python inference/run_tma24_dense_point_proposals.py \
  --checkpoint checkpoints/SAM3-base/sam3.pt \
  --device cuda \
  --max-side 1024 \
  --grid-step 64 \
  --border 32
```

Dense box proposals:

```bash
python inference/run_tma24_dense_box_proposals.py \
  --checkpoint checkpoints/SAM3-base/sam3.pt \
  --device cuda \
  --max-side 1024 \
  --box-size 256 \
  --stride 128
```

Purpose: sweep the whole image with point or box prompts and ask whether any
proposal resembles a target label. This is not a generalization test; it is an
oracle-style proposal-generator test.

## Local Summary

To summarize downloaded outputs into a single CSV/Markdown table:

```bash
python inference/summarize_tma24_sam3_results.py
```

Current local summary outputs:

- `output/tma24_sam3_results_summary.csv`
- `output/tma24_sam3_results_summary.md`

## Current Readout

Direct whole-image box prompting has usable signal, text prompting is weak, and
dense box proposals recover partial structures when evaluated oracle-style.
Cross-region transfer remains the key generalization check: if held-out Dice
stays near zero, the model is acting more like an interactive/proposal segmenter
than an automatic recognizer of repeated pathology structures.
