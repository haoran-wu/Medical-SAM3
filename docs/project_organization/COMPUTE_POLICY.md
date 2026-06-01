# Compute Policy: Remote First

This project uses a remote-first compute rule.

## Default Rule

Run all formal computation on Bouchet or another approved remote compute
environment unless Haoran explicitly asks for a local run.

Formal computation includes:

- candidate-pool generation or scanning.
- mask/component union search.
- Dice, Precision, Recall, and accuracy recomputation for reported results.
- CLIP, VLM, or API scoring runs.
- large image processing, crop rendering, or batch report generation.

## What Local Is For

The local Mac checkout is mainly for:

- organizing files and Git commits.
- inspecting small outputs.
- editing scripts, prompts, and documentation.
- rendering lightweight HTML previews when the heavy inputs were already
  computed remotely.
- packaging final figures, reports, and presentation assets.

## Exceptions

Local computation is allowed only when one of these is true:

- Haoran explicitly says to run it locally.
- It is a tiny sanity check that does not become a reported result.
- It is a lightweight local preview of remote-computed artifacts.

If a local sanity check is shown in a report or message, label it clearly as
local sanity-check output, not a final remote result.

## Reporting Rule

Final reported metrics should come from remote runs. If a number came from local
inspection, say that clearly and rerun remotely before treating it as a project
result.
