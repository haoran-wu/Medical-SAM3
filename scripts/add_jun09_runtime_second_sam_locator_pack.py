#!/usr/bin/env python3
"""Build the 17T runtime-policy second-SAM locator pack.

This section is deliberately different from earlier second-SAM previews:
selection comes from the 17S runtime-only policy, not from hidden Dice or an
annotation-tuned VLM policy.  Annotation is used only after the selected pieces
are converted to box prompts, so the report can diagnose whether second-SAM
refinement is a reasonable next step.
"""

from __future__ import annotations

import csv
import html
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "runtime_policy_second_sam_locator_pack"
SECOND_SAM_SCRIPT = ROOT / "inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py"
RUNTIME_SELECTED = BASE / "runtime_policy_prototype/runtime_policy_selected_pieces.csv"
RUNTIME_SUMMARY = BASE / "runtime_policy_prototype/runtime_policy_summary.csv"
HIDDEN_TRUTH = BASE / "corrected_pool/hidden_candidate_truth.csv"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
DISPLAY_TO_CLASS = {"immune infiltration": "immune_infiltration", **{c: c for c in CLASS_KEYS}}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def display_class(name: str) -> str:
    return "immune infiltration" if name == "immune_infiltration" else name


def normalize_runtime_selected() -> Path:
    selected = pd.read_csv(RUNTIME_SELECTED)
    hidden = pd.read_csv(HIDDEN_TRUTH)[["candidate_uid", "mask_path"]]
    hidden_by_uid = hidden.set_index("candidate_uid")["mask_path"].to_dict()
    masks_dir = OUT / "piece_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        cls = DISPLAY_TO_CLASS[str(row["class"])]
        uid = str(row["candidate_uid"])
        source_mask = Path(hidden_by_uid.get(uid, ""))
        if not source_mask.exists():
            raise SystemExit(f"Missing source mask for runtime selected piece {uid}: {source_mask}")
        local_mask = masks_dir / f"{uid}.png"
        if not local_mask.exists():
            shutil.copy2(source_mask, local_mask)
        rows.append(
            {
                "class": cls,
                "rank": int(row["rank"]),
                "candidate_uid": uid,
                "target_score": f"{float(row['score']):.6f}",
                "margin": "",
                "runtime_policy_threshold": row.get("threshold", ""),
                "runtime_policy_score": f"{float(row['score']):.6f}",
                "mask_path": f"piece_masks/{uid}.png",
                "selector_source": "17S runtime-only policy; no hidden Dice/Precision/Recall used for selection",
            }
        )
    out_csv = OUT / "runtime_policy_selected_pieces_for_second_sam.csv"
    write_csv(out_csv, rows)
    return out_csv


def run_prompt_preview(selected_csv: Path) -> None:
    cmd = [
        sys.executable,
        str(SECOND_SAM_SCRIPT),
        "--selected-pieces-csv",
        str(selected_csv),
        "--hidden-truth-csv",
        str(HIDDEN_TRUTH),
        "--output-dir",
        str(OUT),
        "--labels",
        ",".join(CLASS_KEYS),
        "--dry-run-prompts-only",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_readme(summary_rows: list[dict[str, object]]) -> None:
    readme = [
        "# Jun09 Runtime Policy Second-SAM Locator Pack",
        "",
        "This folder is a prompt/locator pack, not a completed SAM rerun.",
        "",
        "Selection source: 17S runtime-only policy. The selected pieces were chosen using deployment-time scores only: H&E morphology, structured FICTURE composition prior, shape/location, and fixed class policies.",
        "",
        "Annotation/Dice/Precision/Recall are used only after selection for evaluation and gate decisions.",
        "",
        "Files:",
        "- `runtime_policy_selected_pieces_for_second_sam.csv`: selected pieces normalized for the second-SAM runner.",
        "- `second_sam_prompt_preview_rows.csv`: one row per selected piece plus one class-union row; includes box prompt coordinates.",
        "- `second_sam_prompt_preview_summary.csv`: per-class selected-piece union metrics before second-SAM.",
        "- `figures/*_selected_piece_prompt_preview.png`: six-panel visual checks for each class.",
        "",
        "Recommended next remote action:",
        "From the local Medical-SAM3 root, run `scripts/local/sync_and_submit_jun09_runtime_second_sam.sh` after Bouchet SSH/Duo works. By default it submits only `bronchiola,vessels`, because those are the 17T refinement-ready classes.",
        "",
        "Manual remote equivalent: run `inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py` without `--dry-run-prompts-only` using the same selected CSV and an available SAM/Medical-SAM checkpoint.",
        "",
        "Current gate verdicts:",
    ]
    for row in summary_rows:
        readme.append(
            f"- {row['class']}: {row['17T locator verdict']} "
            f"(selected pieces {row['selected_piece_count']}, preview D/P/R "
            f"{row['prompt_preview_dice']}/{row['prompt_preview_precision']}/{row['prompt_preview_recall']})."
        )
    (OUT / "README.md").write_text("\n".join(readme) + "\n")


def rebuild_zip() -> None:
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    include: list[Path] = []
    for rel in [
        "Jun09_SkillRanker_ComponentAwareMaskSelection.html",
        "index.html",
        "candidate_skill_features.csv",
        "piece_top1_all_methods.csv",
        "assembly_summary_all_methods.csv",
        "selected_pieces_all_methods.csv",
        "failure_attribution_by_class.csv",
        "run_config.json",
    ]:
        p = BASE / rel
        if p.exists():
            include.append(p)
    for root in [
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures",
        BASE / "refined_diagnostics",
        BASE / "guarded_loco_variant",
        BASE / "second_sam_refinement",
        BASE / "structured_veto_microscope",
        BASE / "quality_skill_diagnostic",
        BASE / "quality_adjusted_ranker",
        BASE / "proxy_quality_ranker",
        BASE / "failure_driven_proxy_refinement",
        BASE / "alveoli_generator_gate",
        BASE / "paper_informed_failure_engine",
        BASE / "remote_gate_preflight",
        BASE / "failure_gate_matrix",
        BASE / "skill_iteration_controller",
        BASE / "feature_space_upper_bound",
        BASE / "runtime_policy_prototype",
        OUT,
    ]:
        if root.exists():
            include.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for path in include:
            if path in seen:
                continue
            seen.add(path)
            handle.write(path, path.relative_to(BASE))


def build_section() -> str:
    runtime_summary = pd.read_csv(RUNTIME_SUMMARY)
    preview = read_csv(OUT / "second_sam_prompt_preview_summary.csv")
    preview_by = {row["class"]: row for row in preview}
    summary_rows: list[dict[str, object]] = []
    for _, row in runtime_summary.iterrows():
        cls = DISPLAY_TO_CLASS[str(row["class"])]
        p = preview_by[cls]
        dice = float(p["prompt_preview_dice"])
        precision = float(p["prompt_preview_precision"])
        recall = float(p["prompt_preview_recall"])
        if cls in {"bronchiola", "vessels"} and dice >= 0.70 and precision >= 0.85:
            verdict = "refinement-ready: use selected pieces as second-SAM locators"
            action = "run second-SAM box/point refinement once Bouchet access works"
        elif cls == "alveoli":
            verdict = "not ready: proposal generator failure"
            action = "add broad H&E box proposal before second-SAM"
        else:
            verdict = "exploratory only: selector/feature space still weak"
            action = "improve encoder/proposal or quality filter before treating as final"
        summary_rows.append(
            {
                "class": display_class(cls),
                "selected_piece_count": p["selected_piece_count"],
                "prompt_preview_dice": f"{dice:.3f}",
                "prompt_preview_precision": f"{precision:.3f}",
                "prompt_preview_recall": f"{recall:.3f}",
                "17S policy verdict": row["verdict"],
                "17T locator verdict": verdict,
                "next action": action,
                "figure_rel": f"runtime_policy_second_sam_locator_pack/{p['figure_rel']}",
            }
        )
    make_readme(summary_rows)

    preview_rows = [row for row in read_csv(OUT / "second_sam_prompt_preview_rows.csv") if row["candidate_uid"] != "__CLASS_UNION__"]
    for row in preview_rows:
        row["class"] = display_class(row["class"])

    section = f"""
<h2>17T. Runtime Policy To Second-SAM Locator Pack</h2>
<p>The 17S policy selected pieces without using hidden Dice, Precision, Recall, or annotation labels. 17T converts those selected pieces into <b>second-SAM locators</b>: each piece becomes a box prompt plus a preview union. This tests a SaLIP-style cascade idea: the skill ranker should first localize promising tissue pieces, then SAM/Medical-SAM can refine the actual mask boundary.</p>
<p>This section is a dry-run prompt pack. It does not load SAM locally. Annotation is shown only after the runtime policy has already selected pieces, so the metrics below are diagnostic, not part of selection.</p>
<h3>What this changes</h3>
<table><thead><tr><th>old direct union path</th><th>17T locator-refinement path</th></tr></thead><tbody>
<tr><td>select pieces, union their raw masks, stop</td><td>select pieces, convert each one to a box/point locator, rerun SAM/Medical-SAM, then union refined masks</td></tr>
<tr><td>raw piece boundaries decide final mask quality</td><td>the selected piece only needs to localize the correct structure; second-SAM can recover fuller local boundaries</td></tr>
<tr><td>good for high-precision local pieces</td><td>useful when recall is low because the selected mask is fragmented</td></tr>
</tbody></table>
<h3>Locator gate by class</h3>
{table(summary_rows, ['class', 'selected_piece_count', 'prompt_preview_dice', 'prompt_preview_precision', 'prompt_preview_recall', '17T locator verdict', 'next action'])}
<h3>Six-panel prompt previews</h3>
<div class='image-grid'>
{''.join(f"<figure><img src='{html.escape(row['figure_rel'])}'><figcaption>{html.escape(row['class'])}: selected-piece prompt preview D/P/R {html.escape(row['prompt_preview_dice'])}/{html.escape(row['prompt_preview_precision'])}/{html.escape(row['prompt_preview_recall'])}</figcaption></figure>" for row in summary_rows)}
</div>
<h3>Box prompt rows</h3>
<p>Each row below is one selected runtime-policy piece. The <code>bbox_xyxy</code> column is the box prompt that will be sent to SAM/Medical-SAM. The prior D/P/R columns are shown only as post-selection diagnostics.</p>
{table(preview_rows[:260], ['class', 'candidate_uid', 'rank', 'target_score', 'bbox_xyxy', 'prior_pixels', 'prior_dice', 'prior_precision', 'prior_recall', 'mask_path'])}
<p class='small'>Execution script: <code>scripts/local/sync_and_submit_jun09_runtime_second_sam.sh</code>. Remote sbatch: <code>scripts/hpc/jun09_runtime_policy_second_sam_refine.sbatch</code>. Default labels: <code>bronchiola,vessels</code>; override with <code>LABELS=...</code> only when a class passes its locator gate.</p>
<div class='callout'><b>Decision.</b> Bronchiola and vessels are the immediate second-SAM candidates because runtime-only selection is already high precision and close to the feature-space upper bound. Alveoli should not enter this refinement path yet: the current pool lacks the broad alveolar proposal. Tumor, stroma, and immune infiltration remain exploratory because the selector still admits too many broad or mixed pieces.</div>
"""
    return section


def insert_section(section: str) -> None:
    marker = "<h2>17T. Runtime Policy To Second-SAM Locator Pack</h2>"
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_csv = normalize_runtime_selected()
    run_prompt_preview(selected_csv)
    section = build_section()
    insert_section(section)
    rebuild_zip()
    print(OUT / "second_sam_prompt_preview_summary.csv")


if __name__ == "__main__":
    main()
