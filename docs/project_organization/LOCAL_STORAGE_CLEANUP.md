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

Root-level legacy assets were moved into purpose-specific folders:

```text
example1.jpg -> examples/legacy_examples/tma24/example1.jpg
example2.png -> examples/legacy_examples/spatiallibd/example2.png
Kvasir-SEG/ -> examples/legacy_examples/kvasir_seg/Kvasir-SEG/
pixel-level cell type image/ -> data/visium_hd_exp1/pixel_cell_type_image/
hpc*.py and dashboard launchers -> scripts/hpc_dashboard/
make_group_meeting_ppt*.py -> scripts/presentation/
CLAUDE.md -> docs/agent_notes/CLAUDE.md
research_directions.md, resume_update_medical_sam3.md -> docs/archive/
outputs/ -> output/archive/legacy_manual_outputs_20260507/
```

## What was intentionally not moved

These are large or important experiment assets and should not be moved without a
separate review:

```text
checkpoints/
data/
output/
results/
```

Some of `output/` and `results/` are already tracked in Git, so moving them would
create a larger structural change. They should be handled only if we decide to
reorganize old historical outputs into a new tracked archive layout.

## Current root policy

The repository root should contain:

- README and project index files.
- source/helper scripts under `scripts/`,
- source folders such as `inference/` and `scripts/`,
- documentation under `docs/`,
- local large data/output folders that are intentionally ignored or tracked as legacy.

The repository root should not contain:

- temporary clones under `tmp/`,
- presentation decks that belong in `/Users/haoranwu/Desktop/Yan_Lab_Research/Presentation`,
- generated Python/macOS cache files,
- raw API response logs,
- copied model checkpoints outside `checkpoints/`.
