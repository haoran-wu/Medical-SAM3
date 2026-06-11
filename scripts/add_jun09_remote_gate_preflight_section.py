#!/usr/bin/env python3
"""Add the remote candidate-root preflight contract to the Jun09 report."""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "remote_gate_preflight"


ROOT_ROWS = [
    {
        "source": "he_medpt24",
        "setting": "medical_official_points_step24",
        "modality": "H&E",
        "role": "local piece generator that works well for bronchiola/vessels/immune pieces",
        "root": "/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530/codex_he_official_roi_candidate_pool_b200/he_official_12445451/medical_official_points_step24",
    },
    {
        "source": "ficture_medpt24",
        "setting": "medical_official_points_step24",
        "modality": "FICTURE",
        "role": "same point setting on official PASS_OFFICIAL FICTURE ROI",
        "root": "/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/ficture_official_filtered_candidate_pool_full_masks/ficture_official_medpt24_full_13707092/medical_official_points_step24",
    },
    {
        "source": "he_base_official_points_step24",
        "setting": "base_official_points_step24",
        "modality": "H&E",
        "role": "simpler point-prompt ablation for proposal diversity",
        "root": "/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/fullmask_repair_candidate_pools/he_fullmask_repair_13723426/base_official_points_step24",
    },
    {
        "source": "ficture_base_official_points_step24",
        "setting": "base_official_points_step24",
        "modality": "FICTURE",
        "role": "FICTURE counterpart of the simple point-prompt ablation",
        "root": "/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/fullmask_repair_candidate_pools/ficture_fullmask_repair_13723427/base_official_points_step24",
    },
    {
        "source": "he_base_box384_s128_m1536",
        "setting": "base_box384_s128_m1536",
        "modality": "H&E",
        "role": "broad box proposal that previously recovered alveoli better than medpt24",
        "root": "/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/codex_home_20260530/codex_he_official_roi_candidate_pool_b200/he_official_12445451/base_box384_s128_m1536",
    },
    {
        "source": "ficture_base_box384_s128_m1536",
        "setting": "base_box384_s128_m1536",
        "modality": "FICTURE",
        "role": "FICTURE counterpart of the broad box proposal",
        "root": "/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1/ficture_official_filtered_candidate_pool/ficture_official_11938227/base_box384_s128_m1536",
    },
]

CHECK_ROWS = [
    {
        "check": "root exists",
        "why it matters": "prevents silently scanning zero candidates from a stale or wrong path",
        "failure action": "stop before component scoring and fix the root path or rerun the missing pool",
    },
    {
        "check": "candidate_masks/candidate_*.png count > 0",
        "why it matters": "component oracle needs actual mask files, not only metadata",
        "failure action": "rerun full-mask generation; do not reuse save-mask-limit debug pools",
    },
    {
        "check": "candidate_metadata count matches mask count when row count is available",
        "why it matters": "keeps candidate IDs, source, and settings aligned with the actual masks",
        "failure action": "repair metadata/report or regenerate the pool",
    },
    {
        "check": "candidate_report count matches mask count when row count is available",
        "why it matters": "prevents reporting oracle metrics from an incomplete or truncated mask set",
        "failure action": "block the run and inspect the generator log",
    },
    {
        "check": "candidate mask IDs are contiguous and non-duplicated",
        "why it matters": "catches partial copies, interrupted repair arrays, and duplicate mask IDs",
        "failure action": "repair only the broken setting before rerunning the minimal-three gate",
    },
    {
        "check": "official same-ROI FICTURE input remains PASS_OFFICIAL",
        "why it matters": "makes H&E/FICTURE comparisons scientifically valid",
        "failure action": "do not use deprecated FICTURE roots; rebuild official aligned input",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
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
        OUT,
    ]:
        if root.exists():
            include.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include:
            zf.write(path, path.relative_to(BASE))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "expected_candidate_roots.csv", ROOT_ROWS)
    write_csv(OUT / "strict_preflight_checks.csv", CHECK_ROWS)

    section = f"""
<h2>17O. Remote Gate Preflight Contract: Bad Pools Stop Here</h2>
<p>This gate is now explicit in the component oracle script. Before the minimal-three proposal gate scores any tissue component, it writes <code>candidate_root_integrity_audit.csv</code>. In strict mode, one failed candidate root stops the entire run. This is important because an incomplete candidate pool can make a ranker look weak even when the real problem is missing masks.</p>
<p>The next proposal gate tests three settings, each with H&amp;E and FICTURE roots: <code>medical_official_points_step24</code>, <code>base_official_points_step24</code>, and <code>base_box384_s128_m1536</code>. The broad box setting is included because alveoli needs a proposal family that can cover a wider parenchymal region.</p>
<h3>Candidate roots expected on Bouchet</h3>
{table(ROOT_ROWS, ['source', 'setting', 'modality', 'role', 'root'])}
<h3>Strict preflight checks</h3>
{table(CHECK_ROWS, ['check', 'why it matters', 'failure action'])}
<div class='callout'><b>Implementation update.</b> <code>inference/visium_hd_exp1/componentwise_candidate_oracle.py</code> now supports <code>--strict-root-integrity</code>. The Jun09 minimal-three sbatch uses this flag. The run will not proceed to component-wise Dice/Precision/Recall if report, metadata, and masks are inconsistent. When Bouchet access is restored, run <code>scripts/local/sync_and_submit_jun09_minimal_three_gate.sh</code> from the local project root to sync this strict version and submit the gate.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17O. Remote Gate Preflight Contract: Bad Pools Stop Here</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
