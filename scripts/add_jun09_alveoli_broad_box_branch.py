#!/usr/bin/env python3
"""Append the 17U alveoli broad-box proposal branch to the Jun09 report.

The Jun09 skill ranker should not keep tuning the scorer when the generator
does not create a usable proposal.  This section imports the earlier Jun07
alveoli box384 evidence and turns it into a concrete generator-level branch:
medpt24 pieces fail alveoli, but a broad H&E box384 proposal can recover a
strong alveolar mask and should be tested before more VLM/API work.
"""

from __future__ import annotations

import csv
import html
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_broad_box_branch"

JUN07_RANK = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_alveoli_box384_setting_rank"
JUN07_INPUTS = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_GPT_manual_alveoli_prompt_test_inputs"
JUN07_MATRIX = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_best_setting_matrix/assets/all48_alveoli.png"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


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


def fmt3(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def copy_asset(src: Path, dst_rel: str) -> str:
    dst = OUT / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise SystemExit(f"Missing source asset: {src}")
    shutil.copy2(src, dst)
    return f"alveoli_broad_box_branch/{dst_rel}"


def read_text_if_exists(path: Path) -> str:
    return path.read_text() if path.exists() else ""


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
        BASE / "runtime_policy_second_sam_locator_pack",
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


def build_branch() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUT / "figures"
    figures_dir.mkdir(exist_ok=True)

    hidden = pd.read_csv(JUN07_INPUTS / "tables/hidden_candidate_truth.csv")
    ficture_text = pd.read_csv(JUN07_INPUTS / "tables/alveoli_box384_ficture_text_summary.csv")
    runtime = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    upper = pd.read_csv(BASE / "feature_space_upper_bound/feature_space_upper_bound_summary.csv")

    key_ids = [11, 12, 13, 82, 90]
    key = hidden[hidden["candidate_id"].isin(key_ids)].copy()
    ftxt = ficture_text[ficture_text["candidate_id"].isin(key_ids)].copy()
    ftxt_by_id = ftxt.set_index("candidate_id").to_dict("index")

    candidate_rows: list[dict[str, object]] = []
    for _, row in key.sort_values("candidate_id").iterrows():
        cid = int(row["candidate_id"])
        frow = ftxt_by_id.get(cid, {})
        role = {
            11: "true broad alveoli candidate; high recall partner",
            12: "true broad alveoli candidate; best union partner",
            13: "best single broad alveoli candidate",
            82: "false positive airway-like candidate",
            90: "false positive thin-wall / immune-heavy candidate",
        }[cid]
        decision = {
            11: "usable if recall needs to be pushed",
            12: "keep as union partner",
            13: "keep as anchor candidate",
            82: "reject; no annotation overlap and airway-heavy composition",
            90: "reject; no annotation overlap and immune/stroma-heavy composition",
        }[cid]
        candidate_rows.append(
            {
                "candidate_id": cid,
                "candidate_uid": row["candidate_uid"],
                "role": role,
                "component Dice": fmt3(row["component_best_dice"]),
                "component Precision": fmt3(row["component_best_precision"]),
                "component Recall": fmt3(row["component_best_recall"]),
                "candidate pixels": int(row["candidate_pixels"]),
                "bbox xyxy": row["candidate_bbox_xyxy"],
                "FICTURE inside composition": frow.get("inside_group_summary", ""),
                "FICTURE local composition": frow.get("local_group_summary", ""),
                "branch decision": decision,
            }
        )

    union_rows = [
        {
            "mask": "candidate 13",
            "Dice": "0.677",
            "Precision": "0.711",
            "Recall": "0.647",
            "interpretation": "best single mask from the broad H&E box384 setting",
        },
        {
            "mask": "candidate 11 + 12 union",
            "Dice": "0.729",
            "Precision": "0.582",
            "Recall": "0.976",
            "interpretation": "very high recall, but precision falls too much",
        },
        {
            "mask": "candidate 12 + 13 union",
            "Dice": "0.748",
            "Precision": "0.653",
            "Recall": "0.875",
            "interpretation": "best current compromise; this is the proposed alveoli branch target",
        },
        {
            "mask": "candidate 11 + 12 + 13 union",
            "Dice": "0.727",
            "Precision": "0.580",
            "Recall": "0.977",
            "interpretation": "recall is maximal but false-positive area grows",
        },
    ]

    medpt24_alv = runtime[runtime["class"] == "alveoli"].iloc[0]
    upper_alv = upper[upper["class"] == "alveoli"].iloc[0]
    generator_rows = [
        {
            "layer tested": "current medpt24 runtime-only policy",
            "evidence": f"D/P/R {fmt3(medpt24_alv['dice'])}/{fmt3(medpt24_alv['precision'])}/{fmt3(medpt24_alv['recall'])}",
            "diagnosis": "the selected pieces are too small/mixed; runtime scoring cannot create a missing broad proposal",
        },
        {
            "layer tested": "current medpt24 feature-space upper bound",
            "evidence": f"D/P/R {fmt3(upper_alv['dice'])}/{fmt3(upper_alv['precision'])}/{fmt3(upper_alv['recall'])}",
            "diagnosis": "even the best deployable-feature formula inside medpt24 cannot recover good alveoli",
        },
        {
            "layer tested": "H&E broad box384 candidate 13",
            "evidence": "D/P/R 0.677/0.711/0.647",
            "diagnosis": "a usable alveoli candidate exists when the proposal generator is broader",
        },
        {
            "layer tested": "H&E broad box384 candidate 12+13 union",
            "evidence": "D/P/R 0.748/0.653/0.875",
            "diagnosis": "broad box proposals plus constrained union solve the current alveoli failure better than ranker tuning",
        },
    ]

    ficture_rows = [
        {
            "candidate": "11",
            "structured FICTURE signal": "inside: alveolar 7.6%, tumor epithelial 59.3%, immune 26.7%; local: alveolar 15.9%",
            "use": "support, but H&E morphology remains primary",
        },
        {
            "candidate": "12",
            "structured FICTURE signal": "inside: alveolar 3.2%, tumor epithelial 64.0%, immune 27.4%",
            "use": "do not reject only because tumor-like epithelial color is high in lung cancer",
        },
        {
            "candidate": "13",
            "structured FICTURE signal": "inside: alveolar 3.8%, tumor epithelial 64.2%, immune 25.9%; local: alveolar 11.0%",
            "use": "anchor by H&E morphology; FICTURE is weakly supportive, not decisive",
        },
        {
            "candidate": "82",
            "structured FICTURE signal": "inside: airway epithelial 41.9%, no annotation overlap",
            "use": "reject as airway-like false positive",
        },
        {
            "candidate": "90",
            "structured FICTURE signal": "inside: immune 61.3%, stroma/endothelial 19.3%, no annotation overlap",
            "use": "reject as immune/stroma-heavy false positive",
        },
    ]

    decision_rows = [
        {
            "decision": "Do not send current medpt24 alveoli pieces to second-SAM as final locators",
            "reason": "their best runtime union is only Dice 0.299 with low precision",
        },
        {
            "decision": "Add a class-specific broad H&E box proposal branch for alveoli",
            "reason": "candidate 13 and union 12+13 show the missing mask exists under box384",
        },
        {
            "decision": "Use structured FICTURE text as a veto/support feature, not raw FICTURE image",
            "reason": "raw FICTURE image previously amplified false positives; composition text helps reject 82/90",
        },
        {
            "decision": "Next runnable gate",
            "reason": "minimal proposal pool = medpt24 points for bronchiola/vessels plus box384 H&E for alveoli, then rerun runtime-only policy",
        },
    ]

    write_csv(OUT / "alveoli_box384_candidate_evidence.csv", candidate_rows)
    write_csv(OUT / "alveoli_box384_union_evidence.csv", union_rows)
    write_csv(OUT / "alveoli_generator_diagnosis.csv", generator_rows)
    write_csv(OUT / "alveoli_structured_ficture_veto_evidence.csv", ficture_rows)
    write_csv(OUT / "alveoli_branch_decision.csv", decision_rows)

    union_fig = copy_asset(JUN07_RANK / "alveoli_box384_union_12_13_six_panel.png", "figures/alveoli_box384_union_12_13_six_panel.png")
    montage_fig = copy_asset(JUN07_RANK / "key_candidates/alveoli_key_candidates_montage.png", "figures/alveoli_key_candidates_montage.png")
    matrix_fig = copy_asset(JUN07_MATRIX, "figures/all48_alveoli_reference.png")
    he_sheet = copy_asset(JUN07_INPUTS / "alveoli_6candidate_HE_only_contact_sheet.png", "figures/alveoli_6candidate_HE_only_contact_sheet.png")
    hf_sheet = copy_asset(JUN07_INPUTS / "alveoli_6candidate_HE_FICTURE_contact_sheet.png", "figures/alveoli_6candidate_HE_FICTURE_contact_sheet.png")

    for cid in key_ids:
        copy_asset(
            JUN07_RANK / f"key_candidates/cid{cid:03d}_he_reverse_blur_gray.png",
            f"figures/key_candidates/cid{cid:03d}_he_reverse_blur_gray.png",
        )

    prompt_structured = html.escape(read_text_if_exists(JUN07_INPUTS / "prompts/prompt_3_HE_plus_structured_FICTURE_text_rank.txt"))

    section = f"""
<h2>17U. Alveoli Broad-Box Proposal Branch</h2>
<p>17T showed that bronchiola and vessels are ready for second-SAM locator refinement, but alveoli is not. This section makes the alveoli failure more precise. The current <code>medical_official_points_step24</code> piece pool is good for small local structures, but alveoli is a broad parenchymal region. A ranker cannot select a broad alveoli mask if the proposal generator mostly gives it small or mixed pieces.</p>
<p>The repair is a class-specific proposal branch: keep the current medpt24 pieces for classes where they work, but add a broader H&amp;E box proposal setting for alveoli. This is still annotation-hidden at selection time; annotation is used here only to diagnose which generator family actually contains a usable alveoli mask.</p>
<h3>Generator-layer evidence</h3>
{table(generator_rows, ['layer tested', 'evidence', 'diagnosis'])}
<h3>Key broad-box candidates</h3>
<p>All rows below come from the H&amp;E <code>base_box384_s128_m1536</code> setting. Candidate 13 is the best single broad alveoli mask. Candidate 12 is the best union partner. Candidates 82 and 90 are useful failure controls because they look tempting to visual models but have zero annotation overlap.</p>
{table(candidate_rows, ['candidate_id', 'role', 'component Dice', 'component Precision', 'component Recall', 'candidate pixels', 'bbox xyxy', 'branch decision'])}
<figure><img src='{html.escape(montage_fig)}'><figcaption>Key alveoli broad-box candidates. 11/12/13 are true broad parenchyma candidates; 82/90 are false-positive controls.</figcaption></figure>
<h3>Union choice</h3>
<p>The chosen branch target is not simply the highest recall union. Candidate 11+12+13 covers almost everything but loses too much precision. Candidate 12+13 is the best current precision/recall compromise.</p>
{table(union_rows, ['mask', 'Dice', 'Precision', 'Recall', 'interpretation'])}
<figure><img src='{html.escape(union_fig)}'><figcaption>Six-panel evidence for the proposed alveoli branch: candidate 12+13 union, D/P/R 0.748/0.653/0.875.</figcaption></figure>
<h3>How FICTURE should help this branch</h3>
<p>The key lesson from Jun07 is that raw FICTURE color images should not drive the alveoli selector. They can make false positives stronger because the model treats color as ordinary image texture. FICTURE is more useful after conversion into structured composition text: inside-mask and local-context RGB, major compartment, and cell-type proportions.</p>
{table(ficture_rows, ['candidate', 'structured FICTURE signal', 'use'])}
<figure><img src='{html.escape(he_sheet)}'><figcaption>H&amp;E-only manual check sheet for the same alveoli candidates.</figcaption></figure>
<figure><img src='{html.escape(hf_sheet)}'><figcaption>H&amp;E plus FICTURE manual check sheet. The conclusion is not to use raw FICTURE image as the main visual evidence for alveoli.</figcaption></figure>
<details><summary>Structured FICTURE-text prompt used in the Jun07 check</summary><pre>{prompt_structured}</pre></details>
<h3>Reference against earlier all-setting search</h3>
<p>This branch does not erase the old all-setting result. It explains which part of the old search was actually helpful for alveoli: broad H&amp;E box proposals.</p>
<figure><img src='{html.escape(matrix_fig)}'><figcaption>Earlier all-48 alveoli reference. The broad H&amp;E box family is the useful generator for this class.</figcaption></figure>
<h3>17U decision table</h3>
{table(decision_rows, ['decision', 'reason'])}
<div class='callout'><b>17U conclusion.</b> For alveoli, the current skill failure is a <b>proposal-generator failure</b>, not primarily a VLM/ranker failure. The next robust path is to add a broad H&amp;E box384 proposal branch, use H&amp;E morphology as the primary selector, use structured FICTURE composition as a false-positive veto, and then rerun the runtime-only policy before any API/VLM escalation.</div>
"""

    readme = [
        "# 17U Alveoli Broad-Box Proposal Branch",
        "",
        "Purpose: explain why alveoli failed in the Jun09 runtime skill and identify the next generator-level fix.",
        "",
        "Main result: the current medpt24 runtime-only policy reaches only D/P/R 0.299/0.206/0.546 for alveoli, while H&E base_box384_s128_m1536 candidate 13 reaches 0.677/0.711/0.647 and candidate 12+13 union reaches 0.748/0.653/0.875.",
        "",
        "Interpretation: alveoli needs broad H&E box proposals. It should not be debugged by further VLM prompt tuning inside the medpt24-only piece pool.",
        "",
        "FICTURE use: raw FICTURE images are not recommended as primary visual input for alveoli; structured composition text is useful as support/veto evidence.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n")
    return section


def append_section(section: str) -> None:
    marker = "<h2>17U. Alveoli Broad-Box Proposal Branch</h2>"
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if marker in text:
            start = text.index(marker)
            end = text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)


def main() -> None:
    section = build_branch()
    append_section(section)
    rebuild_zip()
    print(OUT / "alveoli_box384_candidate_evidence.csv")
    print(OUT / "alveoli_box384_union_evidence.csv")
    print(OUT / "alveoli_generator_diagnosis.csv")
    print(OUT / "alveoli_branch_decision.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
