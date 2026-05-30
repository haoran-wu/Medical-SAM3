# Local Storage Cleanup

This file records the local cleanup status for this checkout.

## What was cleaned

Generated cache files were removed:

- `.DS_Store`
- `__pycache__/`

Local-only temporary material was moved out of the repository root:

```text
/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3_local_archive/20260530/tmp
```

The old root-level group meeting deck was moved to the presentation archive:

```text
/Users/haoranwu/Desktop/Yan_Lab_Research/Presentation/Archive_from_Medical-SAM3_root_20260530/medical_sam3_group_meeting.pptx
```

## What was intentionally not moved

These are large or important experiment assets and should not be moved without a
separate review:

```text
checkpoints/
data/
output/
outputs/
results/
pixel-level cell type image/
```

Some of `outputs/`, `results/`, and `pixel-level cell type image/` are already
tracked in Git, so moving them would create a larger structural change. They should
be handled only if we decide to reorganize old historical outputs into a new tracked
archive layout.

## Current root policy

The repository root should contain:

- README and project index files.
- source/helper scripts that are already tracked,
- source folders such as `inference/` and `scripts/`,
- documentation under `docs/`,
- local large data/output folders that are intentionally ignored or tracked as legacy.

The repository root should not contain:

- temporary clones under `tmp/`,
- presentation decks that belong in `/Users/haoranwu/Desktop/Yan_Lab_Research/Presentation`,
- generated Python/macOS cache files,
- raw API response logs,
- copied model checkpoints outside `checkpoints/`.

