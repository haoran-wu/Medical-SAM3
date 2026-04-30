# Exp1 Tile Size Ablation

Date: 2026-04-29

This note records a small SAM3-only tile-size ablation on Exp1 to test whether
the square tile artifacts are the main reason the oracle multipoint baseline is
weak.

## Setup

Dataset:

```text
/nfs/roberts/project/pi_xy48/hw646/Exp1
```

Model:

```text
SAM3-base
checkpoint: /nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/checkpoints/SAM3-base/sam3.pt
```

Prompt type:

```text
oracle multi-point prompts sampled from expert GeoJSON-derived masks
```

Label subset:

- `Lung Bronchiola`
- `lung vessels`
- `stroma`
- `tumor`

Fixed settings:

- positive points per prompted tile: `10`
- negative points per prompted tile: `5`

Variable settings:

- `1024` tile with `128` overlap
- `1536` tile with `256` overlap
- `2048` tile with `384` overlap

## Current Results

Status:

- `1024 / 128`: complete
- `1536 / 256`: complete
- `2048 / 384`: pending on Bouchet at time of writing

| Label | 1024 Dice | 1536 Dice | 1024 IoU | 1536 IoU | 1024 Precision | 1536 Precision | 1024 Recall | 1536 Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lung Bronchiola | 0.080 | 0.079 | 0.041 | 0.041 | 0.050 | 0.041 | 0.203 | 0.994 |
| lung vessels | 0.232 | 0.176 | 0.131 | 0.097 | 0.133 | 0.099 | 0.923 | 0.810 |
| stroma | 0.382 | 0.325 | 0.236 | 0.194 | 0.259 | 0.202 | 0.726 | 0.823 |
| tumor | 0.364 | 0.323 | 0.222 | 0.193 | 0.250 | 0.202 | 0.663 | 0.801 |

Tile counts:

| Tile size | overlap | number of tiles |
|---:|---:|---:|
| 1024 | 128 | 28 |
| 1536 | 256 | 15 |

## Interpretation

The larger tile size does not improve Dice on this subset.

Observed pattern:

- moving from `1024` to `1536` generally increases recall
- precision drops
- Dice decreases

This suggests that the larger field of view does not make SAM3 more accurate on
pathology-defined semantic tissue regions. Instead, it mainly encourages more
aggressive expansion of predicted masks.

Current takeaway:

> Tile boundary artifacts are not the only problem, and likely not the main
> problem. The dominant issue still appears to be a mismatch between SAM-style
> promptable object segmentation and semantic histopathology region
> segmentation.

## Output Paths

HPC:

```text
/home/hw646/Medical-SAM3/output/visium_hd_exp1/sam3_runs/tile_ablation_sam3_1024
/home/hw646/Medical-SAM3/output/visium_hd_exp1/sam3_runs/tile_ablation_sam3_1536
/home/hw646/Medical-SAM3/output/visium_hd_exp1/sam3_runs/tile_ablation_sam3_2048
```

Local:

```text
/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/sam3_runs/tile_ablation_sam3_1024
/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/sam3_runs/tile_ablation_sam3_1536
```

