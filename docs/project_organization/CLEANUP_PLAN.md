# Low-Risk Cleanup Plan

This plan is intentionally conservative. It avoids moving or deleting raw data.

## Phase 1: Make the project navigable

- Keep this `docs/project_organization` folder as the human-readable map.
- Treat `VISIUMHD_REPORT_MANIFEST.md` as the current report index.
- Treat `README.md` in this folder as the file-type guide.
- Do not reorganize large folders until the manifest is accepted.
- Treat the current main result as:
  `H&E + FICTURE example -> candidate pool -> component-aware union -> Test1/Test2`.

## Phase 2: Separate final reports from local-only outputs

Future target structure:

```text
reports/
  visium_hd_exp1/
    final/
    archive/
    local_only_large/
```

Suggested policy:

- `reports/visium_hd_exp1/final/`: small final HTML/CSV/PNG summaries suitable for GitHub.
- `reports/visium_hd_exp1/archive/`: old but readable historical reports.
- `reports/visium_hd_exp1/local_only_large/`: large figures or raw outputs kept locally only.

For now, do not move the existing `output/` tree. Instead, copy only selected final
artifacts into a clean `reports/` folder when needed.

## Phase 3: GitHub policy

Track:

- Source code in `inference/`.
- HPC and helper scripts in `scripts/`.
- Documentation in `docs/`.
- Small curated reports: HTML, summary CSV, prompt text, and compressed figures.

Do not track:

- `checkpoints/`.
- Full `data/`.
- Full `output/`.
- Raw API responses such as `api_responses.jsonl`.
- API keys or local environment files.
- Multi-GB intermediate masks or candidate pools.

## Phase 4: Safe cleanup commands to run later

Only after confirming the manifest:

```bash
git status --ignored
git ls-files output/visium_hd_exp1/final_deliverables
du -sh output/visium_hd_exp1/* | sort -h
find output/visium_hd_exp1/final_deliverables -maxdepth 1 -type d | sort
```

Possible cleanup actions after review:

- Delete or move only `*_smoke*` folders after verifying they are not referenced.
- Move old May22-May27 readable reports into an archive folder.
- Keep May30 reports as current primary results.
- Preserve the latest H&E/FICTURE paired example, candidate-pool, union, Test1, and
  Test2 materials as the main result chain.
- Move `.openrouter_api_key` out of the repository root and into shell environment or keychain.

## Phase 5: Suggested next branch

If we do actual file moves later, use a separate branch:

```text
codex/cleanup-project-structure
```

That branch should contain only cleanup docs, manifests, and carefully selected small
report copies. It should not move raw data or checkpoints unless explicitly requested.
