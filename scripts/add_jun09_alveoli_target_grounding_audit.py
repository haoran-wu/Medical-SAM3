#!/usr/bin/env python3
"""Append an alveoli target-grounding audit to the Jun09 report.

The complete H&E bbox morphology audit showed a subtle but important failure:
candidate 48 ranks above true alveoli candidates under no-position morphology
even though its hidden Dice is 0.  This means the model can identify
alveoli-like texture, but not necessarily the specific target annotation region.

This section makes that distinction explicit: tissue appearance recognition is
not the same as target-region grounding.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_target_grounding_audit"


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


def fmt3(v: object) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE / "alveoli_bbox_he_morphology_audit/alveoli_bbox_he_morphology_features.csv")
    focus_ids = [48, 13, 12, 11, 22, 24, 37, 82, 90]
    cols = [
        "candidate_id",
        "component_best_dice",
        "component_best_precision",
        "component_best_recall",
        "he_shape_morphology_score",
        "he_airspace_texture_score",
        "he_bbox_edge_density",
        "he_bbox_bright_low_sat_fraction",
        "he_bbox_tissue_fraction",
        "candidate_pixels",
        "fill_ratio",
        "cx_norm",
        "cy_norm",
        "bbox",
    ]
    focus = df[df["candidate_id"].isin(focus_ids)][cols].copy()
    focus = focus.sort_values("he_shape_morphology_score", ascending=False)
    rows = []
    for _, row in focus.iterrows():
        cid = int(row["candidate_id"])
        if cid in {12, 13}:
            meaning = "target annotation region, useful union anchor"
        elif cid == 11:
            meaning = "target-adjacent useful recall partner"
        elif cid in {82, 90}:
            meaning = "known non-target false positive suppressed by other features"
        elif cid == 48:
            meaning = "alveoli-like morphology but hidden Dice 0: target-grounding failure example"
        else:
            meaning = "broad false positive or partial/neighboring region"
        rows.append(
            {
                "candidate": cid,
                "hidden D/P/R": f"{fmt3(row['component_best_dice'])} / {fmt3(row['component_best_precision'])} / {fmt3(row['component_best_recall'])}",
                "H&E morphology score": fmt3(row["he_shape_morphology_score"]),
                "texture score": fmt3(row["he_airspace_texture_score"]),
                "edge density": fmt3(row["he_bbox_edge_density"]),
                "bright-space fraction": fmt3(row["he_bbox_bright_low_sat_fraction"]),
                "bbox": row["bbox"],
                "meaning": meaning,
            }
        )

    failure_rows = [
        {
            "failure subtype": "tissue appearance success, target-region failure",
            "what it means": "The feature can find alveoli-like H&E texture, but the annotation is a specific spatial region. Similar-looking alveolar tissue elsewhere becomes false positive under the current metric.",
            "example": "candidate 48 ranks high by H&E morphology but has hidden Dice 0",
            "fix": "Add target-region grounding: full-ROI locator, spatial relation to annotation-free landmarks, or second-stage selection constrained by selected proposal family.",
        },
        {
            "failure subtype": "mask unavailable for arbitrary selected union",
            "what it means": "Ranking can be evaluated locally, but new selected-union Dice requires binary candidate masks from the broad-box pool.",
            "example": "we know 12+13 union from saved Jun07 result, but cannot recompute arbitrary top-k union locally",
            "fix": "Restore remote mask access or rebuild/copy the broad-box candidate masks locally.",
        },
        {
            "failure subtype": "H&E morphology not enough alone",
            "what it means": "No-position morphology gives useful signal, but still admits broad false positives.",
            "example": "top morphology list includes 48/22/31/9 with hidden Dice 0",
            "fix": "Use morphology as a locator score, then add grounding/assembly constraints rather than only retuning morphology weights.",
        },
    ]

    action_rows = [
        {
            "priority": 1,
            "action": "recover or regenerate broad-box binary candidate masks",
            "why": "needed to compute selector-selected union D/P/R and stop relying on saved 12+13 union only",
            "success check": "can union top-k selected candidates and generate six-panel figure",
        },
        {
            "priority": 2,
            "action": "add target-grounding feature",
            "why": "distinguish target annotation region from morphology-similar non-target regions",
            "success check": "candidate 48 falls below 12/13 without manually using hidden Dice",
        },
        {
            "priority": 3,
            "action": "test second-SAM on selected broad candidates",
            "why": "SaLIP-like cascade may turn good locator proposals into better final masks",
            "success check": "selected union/refined mask improves Dice or Precision/Recall balance over 12+13",
        },
    ]

    write_csv(OUT / "alveoli_target_grounding_focus_candidates.csv", rows)
    write_csv(OUT / "alveoli_target_grounding_failure_subtypes.csv", failure_rows)
    write_csv(OUT / "alveoli_target_grounding_next_actions.csv", action_rows)

    section = f"""
<h2>17AO. Alveoli Target-Grounding Audit</h2>
<p><b>Purpose.</b> The complete H&amp;E bbox morphology audit found real alveoli-like texture signal, but also exposed a subtler failure: finding tissue that looks like alveoli is not the same as finding the specific target alveoli annotation region.</p>
<h3>Focus candidates</h3>
{table(rows, ['candidate', 'hidden D/P/R', 'H&E morphology score', 'texture score', 'edge density', 'bright-space fraction', 'bbox', 'meaning'])}
<h3>Failure subtypes</h3>
{table(failure_rows, ['failure subtype', 'what it means', 'example', 'fix'])}
<h3>Next actions</h3>
{table(action_rows, ['priority', 'action', 'why', 'success check'])}
<div class='callout'><b>Decision.</b> The next alveoli improvement should not be another generic VLM prompt. It should first recover binary broad-box masks so selected unions can be measured, then add target-grounding/context constraints to separate morphology-similar non-target tissue from the actual target annotation region.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AO. Alveoli Target-Grounding Audit</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[P-Z]|<h2>18\\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "alveoli_target_grounding_next_actions.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


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
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he",
        BASE / "runtime_policy_prototype",
        BASE / "alveoli_broad_box_branch",
        BASE / "alveoli_deployable_selector_gate",
        BASE / "alveoli_feature_sufficiency_audit",
        BASE / "alveoli_bbox_he_morphology_audit",
        OUT,
        BASE / "semantic_ficture_proposal_generator",
        BASE / "failure_driven_hybrid_controller",
        BASE / "failure_hierarchy_v4",
        BASE / "immune_he_morphology_branch",
        BASE / "immune_loco_validation",
        BASE / "failure_analysis_protocol_v2",
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
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text()
    missing = []
    for src in re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", text):
        if src.startswith(("http://", "https://", "data:")):
            continue
        if not (BASE / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing HTML images: {missing[:20]}")


if __name__ == "__main__":
    main()
