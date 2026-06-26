#!/usr/bin/env python3
"""Add a quantitative failure gate matrix to the Jun09 skill-ranker report."""

from __future__ import annotations

import csv
import html
import math
import re
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "failure_gate_matrix"
CLASS_ORDER = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
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


def display_to_key(name: str) -> str:
    return "immune_infiltration" if name == "immune infiltration" else name


def key_to_display(name: str) -> str:
    return "immune infiltration" if name == "immune_infiltration" else name


def parse_first_float(text: str) -> float:
    match = re.search(r"\d+\.\d+|\d+", str(text))
    return float(match.group(0)) if match else float("nan")


def parse_dpr(text: str) -> tuple[float, float, float]:
    values = [float(v) for v in re.findall(r"\d+\.\d+|\d+", str(text))]
    while len(values) < 3:
        values.append(float("nan"))
    return values[0], values[1], values[2]


def parse_fraction(text: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(text))
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def verdict_from_rate(rate: float, pass_threshold: float, partial_threshold: float) -> str:
    if math.isnan(rate):
        return "not estimable"
    if rate >= pass_threshold:
        return "pass"
    if rate >= partial_threshold:
        return "partial"
    return "fail"


def gate_summary(verdicts: dict[str, str]) -> str:
    order = ["proposal", "recognition", "quality", "assembly", "refinement"]
    for gate in order:
        if verdicts.get(gate) == "fail":
            return gate
    for gate in order:
        if verdicts.get(gate) == "partial":
            return gate
    return "validation"


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
        OUT,
    ]:
        if root.exists():
            include.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in include:
            zf.write(path, path.relative_to(BASE))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    iteration = read_csv(BASE / "refined_diagnostics/class_specific_iteration_plan.csv")
    failure = read_csv(BASE / "failure_attribution_by_class.csv")
    quality = read_csv(BASE / "quality_skill_diagnostic/quality_skill_summary_by_class.csv")
    refined = read_csv(BASE / "failure_driven_proxy_refinement/class_specific_refinement_summary.csv")

    iteration_by = {display_to_key(r["class"]): r for r in iteration}
    failure_by = {display_to_key(r["class"]): r for r in failure}
    quality_by = {display_to_key(r["class"]): r for r in quality}
    refined_by = {display_to_key(r["class"]): r for r in refined}

    rows: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    for cls in [display_to_key(c) for c in CLASS_ORDER]:
        it = iteration_by.get(cls, {})
        fa = failure_by.get(cls, {})
        qu = quality_by.get(cls, {})
        rf = refined_by.get(cls, {})

        medpt24_max = parse_first_float(it.get("medpt24 full max component Dice", "nan"))
        all48_dice, all48_p, all48_r = parse_dpr(it.get("current all-48 best D/P/R", "nan"))
        piece_ok, piece_total = parse_fraction(fa.get("combined_piece_top1", "0/0"))
        piece_rate = piece_ok / piece_total if piece_total else float("nan")
        oracle_d, oracle_p, oracle_r = parse_dpr(fa.get("oracle_greedy_dice_precision_recall", "nan"))
        runtime_d, runtime_p, runtime_r = parse_dpr(fa.get("combined_final_dice_precision_recall", "nan"))
        refined_d = parse_first_float(rf.get("dice", runtime_d))
        spearman = parse_first_float(qu.get("Spearman(score, Dice)", "nan"))

        proposal_gain = all48_dice - medpt24_max if not math.isnan(all48_dice) else float("nan")
        proposal_verdict = "fail" if medpt24_max < 0.60 and proposal_gain > 0.10 else verdict_from_rate(medpt24_max, 0.75, 0.55)
        recognition_verdict = verdict_from_rate(piece_rate, 0.85, 0.65)
        quality_verdict = verdict_from_rate(spearman, 0.65, 0.35)
        assembly_gap = oracle_d - runtime_d
        assembly_verdict = "fail" if assembly_gap > 0.15 else ("partial" if assembly_gap > 0.07 else "pass")
        refinement_gap = all48_dice - refined_d if not math.isnan(all48_dice) else float("nan")
        refinement_verdict = "fail" if refinement_gap > 0.20 else ("partial" if refinement_gap > 0.10 else "pass")

        verdicts = {
            "proposal": proposal_verdict,
            "recognition": recognition_verdict,
            "quality": quality_verdict,
            "assembly": assembly_verdict,
            "refinement": refinement_verdict,
        }
        primary = gate_summary(verdicts)

        if primary == "proposal":
            next_action = "change candidate generator before scorer tuning"
            do_not = "do not tune VLM/prompt on a pool without a strong proposal"
        elif primary == "recognition":
            next_action = "change region input: marked/context views or stronger tissue descriptors"
            do_not = "do not trust unmarked tiny crops"
        elif primary == "quality":
            next_action = "add deployable mask-quality proxy before assembly"
            do_not = "do not select by tissue probability alone"
        elif primary == "assembly":
            next_action = "tune class-specific thresholds, NMS, and piece count"
            do_not = "do not use one global assembly rule"
        elif primary == "refinement":
            next_action = "try second-SAM or broad-mask refinement from selected locators"
            do_not = "do not treat locator quality as final mask quality"
        else:
            next_action = "validate on minimal-three proposal gate and new ROI before claiming stability"
            do_not = "do not overclaim from one annotated ROI"

        rows.append(
            {
                "class": key_to_display(cls),
                "proposal gate": f"{proposal_verdict} (medpt24 max {medpt24_max:.3f}; all-48 best {all48_dice:.3f})",
                "recognition gate": f"{recognition_verdict} (piece top1 {piece_ok}/{piece_total})",
                "quality gate": f"{quality_verdict} (score-Dice Spearman {spearman:.3f})",
                "assembly gate": f"{assembly_verdict} (oracle-runtime Dice gap {assembly_gap:.3f})",
                "refinement gate": f"{refinement_verdict} (all-48-refined Dice gap {refinement_gap:.3f})",
                "primary failing layer": primary,
                "next action": next_action,
                "do not do": do_not,
            }
        )
        actions.append(
            {
                "class": key_to_display(cls),
                "first layer to fix": primary,
                "why": rows[-1][f"{primary} gate"] if primary in verdicts else "all basic gates pass; needs validation",
                "next action": next_action,
            }
        )

    write_csv(OUT / "quantitative_failure_gate_matrix.csv", rows)
    write_csv(OUT / "next_action_by_failing_layer.csv", actions)

    section = f"""
<h2>17P. Quantitative Failure Gate Matrix: Which Layer Failed?</h2>
<p>This table makes the skill-debugging rule explicit. For each tissue class, the method checks five layers in order: proposal generation, tissue recognition, mask-quality ranking, assembly, and final refinement. The next experiment changes only the first failing layer. This prevents a common mistake: changing the prompt when the missing piece is actually absent from the candidate pool, or changing the candidate pool when recognition is already working.</p>
{table(rows, ['class', 'proposal gate', 'recognition gate', 'quality gate', 'assembly gate', 'refinement gate', 'primary failing layer', 'next action', 'do not do'])}
<h3>Immediate next action by class</h3>
{table(actions, ['class', 'first layer to fix', 'why', 'next action'])}
<div class='callout'><b>Practical rule.</b> A class is not allowed to proceed to VLM/API tuning unless its proposal gate passes. A class is not allowed to proceed to final assembly unless its recognition and mask-quality gates are at least partial. This is how the skill becomes iterative and reliable rather than just runnable.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17P. Quantitative Failure Gate Matrix: Which Layer Failed?</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    print(OUT / "quantitative_failure_gate_matrix.csv")


if __name__ == "__main__":
    main()
