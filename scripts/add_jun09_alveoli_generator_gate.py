#!/usr/bin/env python3
"""Add the alveoli generator gate to the Jun09 skill-ranker report.

This section makes the failure analysis actionable: if a class is missing a
good proposal in the current medpt24 pool, scorer/prompt tuning should stop and
the proposal generator should change first.
"""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_generator_gate"


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


def parse_dice(metric: str) -> float:
    return float(str(metric).split("/")[0].strip())


def display_class(name: str) -> str:
    return "immune infiltration" if name == "immune_infiltration" else name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    refined = BASE / "refined_diagnostics"
    expanded = read_csv(refined / "expanded_pool_diagnostic.csv")
    minimal_top = read_csv(refined / "minimal_setting_top_by_class.csv")
    generator_gate = read_csv(refined / "class_specific_generator_gate.csv")
    proposed = read_csv(refined / "minimal_setting_proposed_next.csv")
    failure_refined = read_csv(BASE / "failure_driven_proxy_refinement/class_specific_refinement_comparison.csv")

    expanded_by_class = {r["class"].replace("immune infiltration", "immune_infiltration"): r for r in expanded}
    gate_by_class = {r["class"].replace("immune infiltration", "immune_infiltration"): r for r in generator_gate}
    failure_by_class = {r["class"].replace("immune infiltration", "immune_infiltration"): r for r in failure_refined}

    alveoli_tops = [r for r in minimal_top if r["class"] == "alveoli"][:5]
    alveoli_current = expanded_by_class["alveoli"]
    best_alveoli = alveoli_tops[0]

    class_layer_rows: list[dict[str, object]] = []
    for tissue_class in ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]:
        ex = expanded_by_class[tissue_class]
        gate = gate_by_class[tissue_class]
        refined_row = failure_by_class.get(tissue_class, {})
        refined_score = refined_row.get("failure_driven_proxy", "")
        current_max = float(ex["full max component Dice"])
        # This rule intentionally separates proposal failure from scoring and assembly failure.
        if tissue_class == "alveoli" and current_max < 0.60:
            layer = "proposal generator"
            action = "switch to broad H&E box proposals before more ranker tuning"
        elif tissue_class == "vessels" and refined_score.startswith("0.724"):
            layer = "class-specific scoring"
            action = "keep FICTURE-dominant vessel proxy and validate on broader candidates"
        elif current_max >= 0.80:
            layer = "assembly / mask-quality selection"
            action = "keep proposals; refine quality gating, NMS, and optional second-SAM"
        else:
            layer = gate["limiting layer"]
            action = gate["generator decision"]
        class_layer_rows.append(
            {
                "class": display_class(tissue_class),
                "current medpt24 full max Dice": ex["full max component Dice"],
                "current best full-pool oracle union D/P/R": ex["full oracle union D/P/R"],
                "latest runtime proxy D/P/R": refined_score,
                "failure layer": layer,
                "next action": action,
                "evidence": gate["why"],
            }
        )

    alveoli_evidence = [
        {
            "test": "Current medpt24 compact167 pool",
            "result": f"only {alveoli_current['compact167 true pieces']} true alveoli pieces; max component Dice {alveoli_current['compact max component Dice']}",
            "interpretation": "the funnel does not contain a strong alveoli piece",
        },
        {
            "test": "Current medpt24 full clustered pool",
            "result": f"{alveoli_current['full clustered candidates']} candidates but max component Dice still {alveoli_current['full max component Dice']}",
            "interpretation": "adding all medpt24 pieces does not fix alveoli, so the proposal family is wrong",
        },
        {
            "test": "Best broad H&E box setting",
            "result": f"{best_alveoli['setting']} reaches {best_alveoli['best component D/P/R']}",
            "interpretation": "alveoli needs broader H&E box-style proposals, not smaller point pieces",
        },
        {
            "test": "Failure-driven runtime proxy",
            "result": failure_by_class["alveoli"]["failure_driven_proxy"],
            "interpretation": "scoring improved recall a bit but cannot overcome the weak proposal pool",
        },
    ]

    next_steps = [
        {
            "step": "1",
            "gate": "Build a minimal three-setting proposal pool",
            "what changes": "combine medical_official_points_step24, base_official_points_step24, and base_box384_s128_m1536",
            "why": "keeps strong component pieces while adding the broad alveoli mask that medpt24 misses",
            "pass condition": "all six classes have a plausible component-level oracle before scorer training",
        },
        {
            "step": "2",
            "gate": "Run component-aware oracle on the three-setting pool",
            "what changes": "evaluate every candidate against annotation components; do not involve VLM yet",
            "why": "separates candidate-generator quality from ranker quality",
            "pass condition": "alveoli reaches the 0.65-0.68 Dice range and bronchiola/vessels do not regress",
        },
        {
            "step": "3",
            "gate": "Train/calibrate skill ranker only after proposal gate passes",
            "what changes": "use class-specific runtime formulas plus quality/shape gates",
            "why": "ranker cannot recover masks absent from the proposal pool",
            "pass condition": "selected-piece union stays close to each class oracle without annotation at selection time",
        },
        {
            "step": "4",
            "gate": "Optional second-SAM only for selected locators",
            "what changes": "use selected piece boxes/points as prompts to recover fuller local masks",
            "why": "SaLIP-style cascade can convert good locators into better masks",
            "pass condition": "recall improves without precision collapse",
        },
    ]

    write_csv(OUT / "class_failure_layer_decision.csv", class_layer_rows)
    write_csv(OUT / "alveoli_generator_evidence.csv", alveoli_evidence)
    write_csv(OUT / "next_three_setting_gate_plan.csv", next_steps)

    section = f"""
<h2>17M. Alveoli Generator Gate: Stop Tuning the Ranker Until the Proposal Exists</h2>
<p>This section turns the failure analysis into a decision rule. A skill ranker can only select masks that exist in the candidate pool. If the current proposal generator does not create a strong candidate for a tissue class, then another VLM prompt, CLIP score, or ranker formula cannot solve that class.</p>
<p>The cleanest example is <b>alveoli</b>. The medpt24 point-prompt pool contains useful local pieces for bronchiola and vessels, but it does not create a broad alveolar-region mask. The all-setting audit shows that alveoli is recovered by a broader H&amp;E box setting, especially <code>base_box384_s128_m1536</code>.</p>
<h3>Failure layer by class</h3>
{table(class_layer_rows, ['class', 'current medpt24 full max Dice', 'current best full-pool oracle union D/P/R', 'latest runtime proxy D/P/R', 'failure layer', 'next action', 'evidence'])}
<h3>Why alveoli is a generator failure</h3>
{table(alveoli_evidence, ['test', 'result', 'interpretation'])}
<h3>Minimal proposal pool to test next</h3>
{table(proposed, ['setting', 'main role', 'why keep it'])}
<h3>Next gate before more VLM/API work</h3>
{table(next_steps, ['step', 'gate', 'what changes', 'why', 'pass condition'])}
<div class='callout'><b>Decision.</b> The next iteration should not spend more effort on alveoli prompt tuning inside the medpt24 pool. First run the minimal three-setting proposal gate. If that gate passes, then the skill ranker can be evaluated fairly; if it fails, the method needs class-specific proposal generation rather than another scorer.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17M. Alveoli Generator Gate: Stop Tuning the Ranker Until the Proposal Exists</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

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
        OUT,
    ]:
        if root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for path in include:
            if path in seen:
                continue
            seen.add(path)
            handle.write(path, path.relative_to(BASE))

    print(OUT / "class_failure_layer_decision.csv")
    print(OUT / "alveoli_generator_evidence.csv")
    print(OUT / "next_three_setting_gate_plan.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
