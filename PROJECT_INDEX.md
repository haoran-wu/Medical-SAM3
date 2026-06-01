# Haoran Medical-SAM3 Project Index

This is the top-level map for the current research project in this checkout.

The upstream repository is Medical-SAM3. The active local research project is
VisiumHD Exp1: H&E + official FICTURE candidate pools, component-aware union, and
VLM evaluation.

## Start here

| File or folder | Purpose |
|---|---|
| `docs/project_organization/CURRENT_MAIN_RESULT.md` | Short explanation of the current main result and storyline |
| `docs/project_organization/PRECISION_AWARE_COMPONENT_UNION.md` | Component-union policy, including precision-aware selection for tumor/stroma/immune |
| `docs/project_organization/VISIUMHD_REPORT_MANIFEST.md` | Which reports are current, historical, or debug-only |
| `docs/project_organization/CLEANUP_PLAN.md` | Conservative cleanup plan that avoids moving raw data |
| `docs/project_organization/GIT_AND_REMOTE.md` | Git branch, GitHub remote, and tracking policy |
| `docs/project_organization/REMOTE_BOUCHET_STORAGE.md` | Current Bouchet remote storage layout and cleanup status |
| `docs/project_organization/LOCAL_STORAGE_CLEANUP.md` | Local cleanup actions and what was intentionally left in place |
| `docs/project_organization/EXAMPLES_MANIFEST.md` | Current vs legacy examples |
| `data/visium_hd_exp1/current_ficture_vlm_inputs/` | Clean current VLM input bundle: source-matched FICTURE legend, 90-row crop pool, prompt snapshot |
| `examples/current_visium_hd_exp1/` | Main example bundle for the current H&E + FICTURE storyline |
| `output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html` | Main current HTML report |

## Current main storyline

```text
Official FICTURE aligned to H&E same ROI
  -> build H&E/FICTURE candidate mask pool
  -> show paired candidate crops
  -> component-aware / precision-aware union
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

## Current VLM input bundle

Use this folder as the clean input source for current Test1/Test2 reruns:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/
```

It contains the source-matched FICTURE HTML legend, the HTML-extracted CSV used to generate
prompt factor lines, the 90-row gray reverse-blur candidate table, and symlinks
to the crop images and official ROI assets. New VLM prompts should use
`ficture_factor_prompt_legend_from_html.csv`, which is extracted from
`source_matched_factor_info_with_llm_inferred_celltypes.html`.

For the next final-mask Test1/Test2 run, use the union-aware six-class input table:

```text
data/visium_hd_exp1/current_ficture_vlm_inputs/final_union_test_inputs/final_test_requests.csv
```

This table sends bronchiola and vessels as merged component-union masks, alveoli
as one single-best mask, and tumor/stroma/immune infiltration as the current
precision-aware single-mask fallbacks.

## Current component-union policy

Use this document for how to decide whether to union component candidates:

```text
docs/project_organization/PRECISION_AWARE_COMPONENT_UNION.md
```

Bronchiola and vessels use component-aware union to recover disconnected pieces.
Tumor, stroma, and immune infiltration should use precision-aware subset selection:
do not union every connected component just because it exists; only keep candidate
components that improve or preserve final Precision while giving useful Recall.

## Current example bundle

The current main example is collected here:

```text
examples/current_visium_hd_exp1/
```

Legacy examples were moved out of the root and now live under:

```text
examples/legacy_examples/
```

They are kept only for old scripts and smoke tests, not for the current VisiumHD
storyline.

## Important source areas

| Path | Role |
|---|---|
| `inference/` | Main Python workflows and VisiumHD-specific scripts |
| `scripts/` | HPC submission scripts, report builders, helper scripts |
| `data/visium_hd_exp1/current_ficture_vlm_inputs/` | Clean current input bundle for VLM Test1/Test2 |
| `docs/project_organization/` | Human-readable project map and cleanup policy |
| `output/visium_hd_exp1/final_deliverables/` | Final and historical HTML/CSV/figure deliverables |

## Root-level layout

The root is intentionally kept small: README/index files plus source, docs,
examples, scripts, and controlled data/output folders. Historical single-image
examples, Kvasir demo data, HPC dashboard helpers, and presentation helpers now
live in their own subfolders.

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
