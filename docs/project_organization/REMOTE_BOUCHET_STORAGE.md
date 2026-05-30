# Remote Bouchet Storage Layout

This file records the current remote storage state on Bouchet.

## Important limitation

The project filesystem is currently full:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3
```

At cleanup time, `/project` reported `100%` usage and rejected even a small test
directory with `Disk quota exceeded`. Because of that, the previous `/home/hw646/codex_*`
experiment folders could not be physically moved into the project filesystem.

## Current clean remote layout

The scattered home folders were consolidated into one project-named remote-run folder:

```text
/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530
```

This folder contains the previous `codex_*` VisiumHD experiment directories.

At cleanup time it contained 29 directories and used about 4.5 GB.

## Compatibility symlinks

The original paths remain as symlinks:

```text
/home/hw646/codex_salip_scripts -> /home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530/codex_salip_scripts
/home/hw646/codex_salip_outputs -> /home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530/codex_salip_outputs
...
```

This keeps older sbatch scripts, logs, and report paths working while the real files
are no longer scattered as separate top-level home directories.

## Current project-side large folders

Most remote storage is already under the project checkout, especially:

```text
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/output/visium_hd_exp1
```

The largest project-side folders observed during cleanup were:

- `results/visium_hd_exp1/training_runs`: about 216 GB.
- `output/visium_hd_exp1/multimodal_candidate_ranking`: about 95 GB.
- `results/visium_hd_exp1/retrieval_eval`: about 29 GB.
- `output_visium_hd_exp1_legacy`: about 28 GB.
- `checkpoints`: about 16 GB.

Do not delete or move these without explicit review.

## What is clean now

- No Slurm jobs were running for user `hw646` during the cleanup check.
- The old `/home/hw646/codex_*` paths now resolve through symlinks.
- Key compatibility checks passed:
  - `/home/hw646/codex_salip_scripts/rank_salip_clip_large_pool.py`
  - `/home/hw646/codex_salip_outputs`
  - the final H&E candidate report path under `/home/hw646/codex_he_official_roi_candidate_pool_b200`

## What is not fully ideal yet

The files are not physically inside `/project` yet because `/project` is full.

To make it fully ideal later:

1. Free or archive at least 5 GB from `/project`.
2. Move `/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530`
   into `/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/remote_runs/visium_hd_exp1/`.
3. Update the `/home/hw646/codex_*` symlinks to point to the new project-side location,
   or update scripts to use the project-side paths directly.

