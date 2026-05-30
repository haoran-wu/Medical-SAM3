# Medical-SAM3 Project Organization

This folder is the map for keeping the project usable.

The repository currently has three different kinds of things mixed together:

- Source code and reusable scripts.
- Local data, model checkpoints, and large intermediate outputs.
- Final reports and figures used for research discussion.

The main rule is: keep source code and small summary reports in GitHub; keep raw data,
model checkpoints, API raw responses, and large experiment outputs local unless they are
explicitly packaged for sharing.

## What is a branch?

A Git branch is a separate working line of the same project.

- `main` is usually the clean or official project line.
- A branch such as `codex-tma24-whole-image-prompts` is a working copy where we can
  save experimental scripts and reports without immediately changing `main`.
- When the branch is pushed to GitHub, it becomes recoverable and shareable.
- Later, we can decide whether to merge it into `main`, keep it as an experiment branch,
  or create a pull request for review.

The current active branch is:

```text
codex-tma24-whole-image-prompts
```

## Current repository map

| Type | Path | What it is | GitHub policy |
|---|---|---|---|
| Core code | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/inference` | Python pipelines for SAM, FICTURE, CLIP, VLM, candidate assembly, and reports | Track |
| HPC scripts | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/scripts` | Bouchet sbatch files and report helpers | Track |
| Documentation | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/docs` | Project notes and experiment design | Track |
| Examples | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/examples` | Current main example bundle plus legacy-example notes | Track |
| Original/local data | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/data` | Local datasets | Do not track as a whole |
| Model weights | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/checkpoints` | Large SAM/Medical-SAM3 checkpoints | Do not track |
| Main local outputs | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output` | Intermediate outputs, final reports, old experiments | Ignore by default; force-add only selected reports |
| Older manual output archive | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/archive/legacy_manual_outputs_20260507` | Older/manual rendered outputs from May 7 | Historical archive |
| Temporary files | `/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/tmp`, `__pycache__` | Runtime/debug files | Do not track |
| Presentation files | `/Users/haoranwu/Desktop/Yan_Lab_Research/Presentation` | Final slide assets and lab decks | Keep outside repo unless intentionally copied |

## Current main result

The current main result is the VisiumHD Exp1 H&E + FICTURE candidate-selection
pipeline, not the older CLIP-only or smoke-test runs.

Use this story line:

1. Use the same ROI with both H&E and official PASS_OFFICIAL FICTURE.
2. Build a candidate mask pool from H&E and FICTURE.
3. Show paired H&E/FICTURE crop examples for each candidate.
4. Use component-aware union to show that multiple disconnected tissue pieces can
   be assembled from candidate masks.
5. Run two VLM tests on the paired H&E/FICTURE candidate crops:
   - Test1: Cross-Label Tissue Classification.
   - Test2: Same-Class Candidate Mask Retrieval.

Everything else should be treated as supporting history, ablation, debugging, or
archive unless it directly supports this pipeline.

The current example bundle is:

```text
/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/examples/current_visium_hd_exp1
```

Older TMA24, spatialLIBD, and Kvasir examples now live under
`examples/legacy_examples/`, not in the repository root and not in the current
main story.

## Main problem

The source code is not the main source of confusion. The messy part is:

```text
/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1
```

That folder contains final reports, old reports, failed/smoke tests, intermediate masks,
alignment debug files, CLIP outputs, VLM outputs, and presentation assets. The safest
solution is to create a manifest and final-report index before moving or deleting anything.

## Recommended low-risk cleanup rule

1. Do not move original data or checkpoints.
2. Do not delete old experiment outputs until the final report and manifest are stable.
3. Keep a small curated set of final HTML/CSV/PNG reports in GitHub.
4. Keep large report folders, raw masks, API raw responses, and model outputs local.
5. Use new branches for cleanup work so cleanup does not disturb active experiments.
