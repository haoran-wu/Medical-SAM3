# Haoran Medical-SAM3 Project Index

This is the top-level map for the current research project in this checkout.

The upstream repository is Medical-SAM3. The active local research project is
VisiumHD Exp1: H&E + official FICTURE candidate pools, component-aware union, and
VLM evaluation.

## Start here

| File or folder | Purpose |
|---|---|
| `docs/project_organization/CURRENT_MAIN_RESULT.md` | Short explanation of the current main result and storyline |
| `docs/project_organization/VISIUMHD_REPORT_MANIFEST.md` | Which reports are current, historical, or debug-only |
| `docs/project_organization/CLEANUP_PLAN.md` | Conservative cleanup plan that avoids moving raw data |
| `docs/project_organization/GIT_AND_REMOTE.md` | Git branch, GitHub remote, and tracking policy |
| `docs/project_organization/REMOTE_BOUCHET_STORAGE.md` | Current Bouchet remote storage layout and cleanup status |
| `docs/project_organization/LOCAL_STORAGE_CLEANUP.md` | Local cleanup actions and what was intentionally left in place |
| `output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html` | Main current HTML report |

## Current main storyline

```text
Same ROI H&E + official FICTURE
  -> build H&E/FICTURE candidate mask pool
  -> show paired candidate crops
  -> component-aware union for disconnected tissue pieces
  -> Test1: Cross-Label Tissue Classification
  -> Test2: Same-Class Candidate Mask Retrieval
```

The current main result is not the older CLIP-only or smoke-test workflow. Those
older runs are useful as supporting evidence and archive, but the presentation
story should start from the paired H&E/FICTURE candidate-pool pipeline.

## Current primary deliverable

Use this report for the most complete current view:

```text
output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html
```

It contains:

- H&E + FICTURE paired crop examples.
- Component-aware union visualizations for disconnected structures.
- Test1 cross-label tissue classification results.
- Test2 same-class candidate mask retrieval results.
- Example prompts and plain-language experiment design.

## Important source areas

| Path | Role |
|---|---|
| `inference/` | Main Python workflows and VisiumHD-specific scripts |
| `scripts/` | HPC submission scripts, report builders, helper scripts |
| `docs/project_organization/` | Human-readable project map and cleanup policy |
| `output/visium_hd_exp1/final_deliverables/` | Final and historical HTML/CSV/figure deliverables |

## Local-only areas

These should not be committed wholesale:

- `data/`
- `checkpoints/`
- full `output/`
- raw API logs such as `api_responses.jsonl`
- local key files such as `.openrouter_api_key`
- temporary folders such as `tmp/` and `__pycache__/`

Selected small final reports can be force-added to Git when they are part of the
research record.

Local cleanup status is documented in:

```text
docs/project_organization/LOCAL_STORAGE_CLEANUP.md
```

## Branches and remotes

Current research branch:

```text
codex-tma24-whole-image-prompts
```

Cleanup/documentation branch:

```text
codex/cleanup-project-structure
```

Remote policy:

- `origin` is Haoran's GitHub fork: `haoran-wu/Medical-SAM3`.
- `upstream` is the original Medical-SAM3 repository: `AIM-Research-Lab/Medical-SAM3`.
- Push active research and cleanup branches to `origin`.
- Do not push large local data, checkpoints, or raw API artifacts.

## Bouchet remote storage

Remote Bouchet files are documented in:

```text
docs/project_organization/REMOTE_BOUCHET_STORAGE.md
```

The project filesystem was full during cleanup, so previous `/home/hw646/codex_*`
folders were consolidated into one project-named home folder with compatibility
symlinks. They should be moved into `/project` after project quota is freed.
