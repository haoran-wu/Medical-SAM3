#!/usr/bin/env python3
"""Add a class-by-class failure microscope to the Jun09 skill-ranker report."""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
REF = BASE / "refined_diagnostics"

CLASS_ORDER = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in cols:
            parts.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def fnum(value: str | float | int, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def by_class(rows: list[dict[str, str]], key: str = "class") -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


def best_assembly_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        cls = row.get("class", "")
        if not cls:
            continue
        if cls not in out or fnum(row.get("dice")) > fnum(out[cls].get("dice")):
            out[cls] = row
    return out


def compact_piece_top1(piece_top1: str) -> tuple[int, int]:
    if "/" not in piece_top1:
        return 0, 0
    a, b = piece_top1.split("/", 1)
    return int(float(a)), int(float(b))


def build_rows() -> list[dict[str, object]]:
    refined = by_class(read_csv(REF / "refined_policy_summary_by_class.csv"))
    fail = by_class(read_csv(BASE / "failure_attribution_by_class.csv"))
    recog = by_class(read_csv(REF / "recognition_calibration_by_class.csv"))
    preview = by_class(read_csv(BASE / "second_sam_refinement/second_sam_prompt_preview_summary.csv"))
    best_asm = best_assembly_rows(read_csv(BASE / "assembly_summary_all_methods.csv"))

    rows: list[dict[str, object]] = []
    for cls in CLASS_ORDER:
        r = refined.get(cls, {})
        fa = fail.get(cls, {})
        rc = recog.get(cls, {})
        pr = preview.get(cls, {})
        ba = best_asm.get(cls, {})
        ann_components = int(fnum(r.get("annotation_components")))
        covered = int(fnum(r.get("pool_components_with_best_recall_ge_0.10")))
        missed = fnum(r.get("missed_component_fraction"))
        auc = fnum(r.get("target_vs_rest_auc"))
        top_correct, top_total = compact_piece_top1(r.get("piece_top1", "0/0"))
        current_d = fnum(r.get("current_dice"))
        refined_d = fnum(r.get("refined_dice"))
        preview_d = fnum(pr.get("prompt_preview_dice"))

        if missed >= 0.35:
            proposal_gate = f"weak: only {covered}/{ann_components} components have recall >=0.10"
        elif covered < ann_components:
            proposal_gate = f"partial: {covered}/{ann_components} components covered"
        else:
            proposal_gate = f"pass: {covered}/{ann_components} components covered"

        if auc >= 0.95 and top_total and top_correct / top_total >= 0.90:
            recognition_gate = f"pass in-sample: AUC {auc:.3f}, Piece Top1 {top_correct}/{top_total}"
        elif top_total:
            recognition_gate = f"weak: AUC {auc:.3f}, Piece Top1 {top_correct}/{top_total}"
        else:
            recognition_gate = "not measured"

        if refined_d - current_d > 0.10:
            assembly_gate = f"old assembly was limiting: Dice {current_d:.3f} -> {refined_d:.3f}"
        elif refined_d >= 0.70:
            assembly_gate = f"mostly working: refined Dice {refined_d:.3f}"
        elif missed >= 0.35:
            assembly_gate = f"cannot solve alone: refined Dice {refined_d:.3f} still limited by proposal coverage"
        else:
            assembly_gate = f"needs tuning/validation: refined Dice {refined_d:.3f}"

        best_method = ba.get("method", "")
        best_method_score = (
            f"{fnum(ba.get('dice')):.3f}/{fnum(ba.get('precision')):.3f}/{fnum(ba.get('recall')):.3f}"
            if ba
            else ""
        )

        if cls in {"bronchiola", "vessels"} and refined_d >= 0.70:
            next_action = "Run second-SAM focus: piece is a good locator; test whether SAM can clean boundary/recall."
        elif cls == "alveoli":
            next_action = "Add broader H&E box-style candidates before more VLM/ranker tuning."
        elif cls in {"tumor", "stroma"}:
            next_action = "Use broad-tissue generator/semantic compartment masks; small-piece union is not enough."
        elif cls == "immune_infiltration":
            next_action = "Keep recall-push many-piece assembly, but add NMS/audit to avoid tumor/stroma leakage."
        else:
            next_action = "Continue class-specific ablation."

        if auc >= 0.95 and missed >= 0.35:
            do_not_do = "Do not keep changing prompt first; the pool/coverage is the bottleneck."
        elif auc >= 0.95 and refined_d - current_d > 0.10:
            do_not_do = "Do not blame recognition first; assembly/refinement is the next gate."
        else:
            do_not_do = "Do not claim deployable until held-out/region-aware validation passes."

        rows.append(
            {
                "class": cls,
                "proposal gate": proposal_gate,
                "recognition gate": recognition_gate,
                "assembly gate": assembly_gate,
                "second-SAM preview": f"selected-piece union D/P/R {preview_d:.3f}/{fnum(pr.get('prompt_preview_precision')):.3f}/{fnum(pr.get('prompt_preview_recall')):.3f}",
                "best current assembly method": best_method,
                "best current D/P/R": best_method_score,
                "root cause": r.get("diagnosed_limiting_layer", fa.get("diagnosis", "")),
                "next action": next_action,
                "do not do next": do_not_do,
            }
        )
    return rows


def replace_section(text: str, title: str, section: str) -> str:
    marker = f"<h2>{title}</h2>"
    if marker in text:
        start = text.index(marker)
        match = re.search(r"<h2[^>]*>", text[start + 1 :])
        end = start + 1 + match.start() if match else text.index("</body>", start)
        return text[:start] + section + text[end:]
    return text.replace("</body>", section + "</body>")


def update_zip() -> None:
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
        BASE / "region_aware_piece_context_locator_pack",
        BASE / "region_aware_vlm_results",
        BASE / "second_sam_refinement",
        ROOT / "inference/visium_hd_exp1/run_jun09_selected_piece_second_sam_refine.py",
        ROOT / "scripts/hpc/jun09_selected_piece_second_sam_refine.sbatch",
        ROOT / "scripts/collect_jun09_second_sam_refinement_results.py",
        ROOT / "scripts/add_jun09_failure_microscope.py",
    ]:
        if root.is_file():
            include.append(root)
        elif root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for p in include:
            if p in seen:
                continue
            seen.add(p)
            try:
                arcname = p.relative_to(BASE)
            except ValueError:
                arcname = Path("support_scripts") / p.relative_to(ROOT)
            handle.write(p, arcname)


def main() -> None:
    rows = build_rows()
    out_csv = REF / "failure_microscope_by_class.csv"
    write_csv(out_csv, rows)
    cols = [
        "class",
        "proposal gate",
        "recognition gate",
        "assembly gate",
        "second-SAM preview",
        "root cause",
        "next action",
        "do not do next",
    ]
    section = f"""
<h2>17F. Failure Microscope by Class</h2>
<p>This table is the current decision layer for the skill. It separates four different failure modes: whether candidates cover the annotation components, whether the score recognizes the tissue, whether assembly/NMS selected the right pieces, and whether selected pieces are ready for second-SAM refinement. The goal is to change only the failing layer instead of blindly changing the prompt.</p>
{table(rows, cols)}
<div class='callout'><b>How to read this.</b> A high Piece Top1 or AUC does not mean the final segmentation is solved. It only means the scorer can recognize the candidate pieces in this pool. If the proposal gate is weak, fix the candidate generator. If recognition passes but assembly is weak, fix selection/NMS. If selected pieces already make a good locator, run second-SAM refinement.</div>
<p class='small'>Machine-readable table: <code>refined_diagnostics/failure_microscope_by_class.csv</code>.</p>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        html_path.write_text(replace_section(text, "17F. Failure Microscope by Class", section))
    update_zip()
    print(out_csv)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
