# Compute Policy: Bouchet Compute-Node First

This project uses a Bouchet compute-node-first rule.

## Default Rule

Run all formal computation on Bouchet Slurm compute nodes unless Haoran
explicitly asks for a local run.

Do not run formal computation on the Bouchet login node. The login node is only
for SSH access, file checks, `sbatch`, `squeue`, `sacct`, log inspection, and
lightweight file organization.

Formal computation includes:

- candidate-pool generation or scanning.
- mask/component union search.
- Dice, Precision, Recall, and accuracy recomputation for reported results.
- CLIP, VLM, or API scoring runs.
- large image processing, crop rendering, or batch report generation.

## Candidate-Pool Source Rule

When rescanning candidate masks for the current VisiumHD Exp1 storyline, use the
candidate pools on Bouchet, not a small local exported subset:

```text
HE candidate pool:
/home/hw646/codex_he_official_roi_candidate_pool_b200/he_official_12445451

Official FICTURE candidate pool:
/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/ficture_official_filtered_candidate_pool
```

Small local/exported subsets, such as top-k component masks copied back for
inspection, are allowed only for explanation or debugging. They are not the
final search space unless the user explicitly asks for a small subset test.

## Partition Selection

Choose CPU or GPU based on the task before submitting the Slurm job.

Use CPU partitions for:

- candidate-pool scanning over existing masks.
- Dice, Precision, Recall, accuracy, or ranking aggregation.
- component-union selection and recall/precision threshold sweeps.
- CSV/HTML/report rendering when no model inference is needed.

Preferred CPU partitions:

- `day` or `day_amd` for full candidate scans and medium/large CPU jobs.
- `devel` for short smoke checks, pool building, and lightweight report jobs.
- `week` only for long non-GPU workflows that truly need it.

Use GPU partitions for:

- SAM or Medical-SAM3 mask generation/refinement.
- local VLM inference.
- CLIP or image-embedding inference when it is not CPU-only.
- neural-network training or feature extraction.

Preferred GPU partitions seen in this project:

- `gpu_h200,gpu_b200,gpu_rtx6000` for local VLM jobs and robust recovery jobs.
- `gpu_b200,gpu_h200,gpu_rtx6000` for high-memory or retry/recovery SAM jobs.
- `gpu_rtx6000` for standard SAM/FICTURE candidate arrays, CLIP/embedding jobs,
  and training/evaluation runs.
- `gpu_devel` for short GPU smoke tests or small GPU report/ranking jobs.
- `scavenge_gpu` only for opportunistic or non-final work where preemption risk
  is acceptable.

Every submitted job should record its job id, partition, resource request, log
path, output root, and purpose.

## What Local Is For

The local Mac checkout is mainly for:

- organizing files and Git commits.
- inspecting small outputs.
- editing scripts, prompts, and documentation.
- rendering lightweight HTML previews when the heavy inputs were already
  computed remotely.
- packaging final figures, reports, and presentation assets.

## What Bouchet Login Is For

The Bouchet login node is mainly for:

- submitting jobs with `sbatch`.
- checking jobs with `squeue`, `sacct`, `scontrol`, and log tails.
- copying small scripts or input bundles into the project run area.
- checking whether files exist.

Do not run Python mask scans, VLM/CLIP inference, SAM generation, or metric
recomputation directly on the login node.

## Exceptions

Local computation is allowed only when one of these is true:

- Haoran explicitly says to run it locally.
- It is a tiny sanity check that does not become a reported result.
- It is a lightweight local preview of remote-computed artifacts.

If a local sanity check is shown in a report or message, label it clearly as
local sanity-check output, not a final remote result.

## Reporting Rule

Final reported metrics should come from remote runs. If a number came from local
inspection or a login-node mistake, say that clearly and rerun it on a Slurm
compute node before treating it as a project result.
