# Legacy Examples

These examples are kept because older scripts still use them as default inputs.

They should not be used as the main presentation/research storyline for the current
VisiumHD project.

## Root-level legacy files

| File | Historical role | Why it stays in root |
|---|---|---|
| `example1.jpg` | TMA24 / silicosis H&E example used by older prompt and proposal scripts | Multiple `inference/run_tma24_*` scripts use it as the default path |
| `example2.png` | spatialLIBD / prompt-inference example | `inference/run_spatiallibd_prompts.py` uses it as the default path |

## Current main example

Use this instead for the current project:

```text
examples/current_visium_hd_exp1/
```

