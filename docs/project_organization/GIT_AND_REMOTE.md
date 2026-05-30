# Git and Remote Organization

This file explains how the current project is saved in Git and GitHub.

## Current remotes

```text
origin   https://github.com/haoran-wu/Medical-SAM3.git
upstream https://github.com/AIM-Research-Lab/Medical-SAM3.git
```

- `origin` is the user's GitHub copy. Push project work here.
- `upstream` is the original Medical-SAM3 repository. Use it only for reference
  or syncing upstream changes.

## Current important branches

| Branch | Use |
|---|---|
| `codex-tma24-whole-image-prompts` | Active research branch with the May30 VLM/union report |
| `codex/cleanup-project-structure` | Documentation and organization branch for making the project easier to navigate |

## What should be tracked

Track small, reproducible, human-readable project assets:

- source code and reusable scripts,
- HPC submission scripts,
- project documentation,
- selected final HTML/CSV/PNG reports,
- manifest files that explain where results came from.

## What should stay local

Do not commit these wholesale:

- raw datasets,
- model checkpoints,
- full candidate pools,
- large intermediate masks,
- raw API response logs,
- local API key files,
- temporary caches.

## Current main result already pushed

The May30 detailed report has already been committed and pushed on:

```text
codex-tma24-whole-image-prompts
```

Commit:

```text
e8b43e1 Add May30 detailed VLM union report
```

Main report:

```text
output/visium_hd_exp1/final_deliverables/May30_detailed_union_test1_test2_report/index.html
```

## Recommended workflow

1. Do experiments on a feature branch.
2. Package only the final selected reports.
3. Update `PROJECT_INDEX.md` and this folder's manifest.
4. Commit source, docs, and curated reports.
5. Push to `origin`.
6. Open a pull request only when the branch should be reviewed or merged.

## Bouchet remote note

Bouchet storage is documented separately in:

```text
docs/project_organization/REMOTE_BOUCHET_STORAGE.md
```

The important detail is that `/project` was full during cleanup. The remote `codex_*`
folders were consolidated under `/home/hw646/Medical-SAM3_remote_runs/` with
compatibility symlinks, but they are not physically inside `/project` yet.
