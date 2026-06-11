#!/usr/bin/env python3
"""Add a paper-informed failure engine to the Jun09 report.

The goal is to move from one-off ablations to a repeatable diagnostic loop:
proposal -> recognition -> mask quality -> assembly -> refinement. Each class
gets a current failure state and a next action.
"""

from __future__ import annotations

import csv
import html
import math
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "paper_informed_failure_engine"


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


def table(rows: list[dict[str, object]], cols: list[str], link_col: str | None = None) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            val = str(row.get(col, ""))
            if link_col and col == link_col and val.startswith("http"):
                label = str(row.get("paper", val))
                out.append(f"<td><a href='{html.escape(val)}'>{html.escape(label)}</a></td>")
            else:
                out.append(f"<td>{html.escape(val)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def parse_dpr(text: str) -> tuple[float, float, float]:
    vals = []
    for part in str(text).replace("/", " ").split():
        try:
            vals.append(float(part))
        except ValueError:
            pass
    while len(vals) < 3:
        vals.append(float("nan"))
    return vals[0], vals[1], vals[2]


def normalize_class(name: str) -> str:
    if name == "immune infiltration":
        return "immune_infiltration"
    return name


def display_class(name: str) -> str:
    return "immune infiltration" if name == "immune_infiltration" else name


def status_from_spearman(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return "not estimable"
    if math.isnan(v):
        return "not estimable"
    if v >= 0.65:
        return "pass"
    if v >= 0.35:
        return "partial"
    return "fail"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    failure = pd.read_csv(BASE / "refined_diagnostics/failure_microscope_by_class.csv")
    quality = pd.read_csv(BASE / "quality_skill_diagnostic/quality_skill_summary_by_class.csv")
    refined = pd.read_csv(BASE / "failure_driven_proxy_refinement/class_specific_refinement_summary.csv")
    expanded = pd.read_csv(BASE / "refined_diagnostics/expanded_pool_diagnostic.csv")
    gate = pd.read_csv(BASE / "alveoli_generator_gate/class_failure_layer_decision.csv")
    coverage = pd.read_csv(BASE / "refined_diagnostics/component_level_proposal_coverage.csv")

    failure["key"] = failure["class"].map(normalize_class)
    quality["key"] = quality["class"].map(normalize_class)
    refined["key"] = refined["class"].map(normalize_class)
    expanded["key"] = expanded["class"].map(normalize_class)
    gate["key"] = gate["class"].map(normalize_class)
    coverage["key"] = coverage["class"].map(normalize_class)

    failure_by = failure.set_index("key").to_dict("index")
    quality_by = quality.set_index("key").to_dict("index")
    refined_by = refined.set_index("key").to_dict("index")
    expanded_by = expanded.set_index("key").to_dict("index")
    gate_by = gate.set_index("key").to_dict("index")

    class_rows: list[dict[str, object]] = []
    for tissue_class in ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]:
        q = quality_by.get(tissue_class, {})
        r = refined_by.get(tissue_class, {})
        e = expanded_by.get(tissue_class, {})
        g = gate_by.get(tissue_class, {})
        f = failure_by.get(tissue_class, {})

        full_max = float(e.get("full max component Dice", 0) or 0)
        latest_dice = float(r.get("dice", 0) or 0)
        latest_precision = float(r.get("precision", 0) or 0)
        latest_recall = float(r.get("recall", 0) or 0)
        spearman_dice = q.get("Spearman(score, Dice)", "")
        quality_status = status_from_spearman(spearman_dice)
        components = coverage[coverage["key"] == tissue_class]
        weak_components = int((components["best_recall"].astype(float) < 0.25).sum()) if len(components) else 0
        total_components = int(len(components))

        if tissue_class == "alveoli" and full_max < 0.60:
            primary_failure = "proposal"
            next_move = "switch to broad H&E box proposal and rerun component oracle"
            stop_doing = "stop prompt/ranker tuning on medpt24-only alveoli"
        elif tissue_class == "vessels" and latest_dice >= 0.70 and latest_precision >= 0.85:
            primary_failure = "validation/refinement"
            next_move = "freeze FICTURE-dominant vessel score and test second-SAM or minimal-three validation"
            stop_doing = "stop using one shared score formula for all tissues"
        elif quality_status in {"fail", "partial"} and full_max >= 0.80:
            primary_failure = "mask quality"
            next_move = "learn or proxy a deployable mask-quality score before assembly"
            stop_doing = "stop selecting by tissue probability alone"
        elif weak_components > max(2, total_components // 3):
            primary_failure = "component coverage"
            next_move = "expand proposal diversity or use broader/diffuse class generator"
            stop_doing = "stop treating this as an object-like small-piece class"
        else:
            primary_failure = str(g.get("failure layer", f.get("root cause", "mixed")))
            next_move = str(g.get("next action", f.get("next action", "")))
            stop_doing = str(f.get("do not do next", ""))

        class_rows.append(
            {
                "class": display_class(tissue_class),
                "proposal upper bound": f"full max Dice {full_max:.3f}",
                "weak components": f"{weak_components}/{total_components}",
                "recognition signal": str(f.get("recognition gate", "")),
                "quality correlation": f"Spearman(score,Dice)={spearman_dice}",
                "latest runtime union D/P/R": f"{latest_dice:.3f}/{latest_precision:.3f}/{latest_recall:.3f}",
                "primary failure": primary_failure,
                "next move": next_move,
                "do not do": stop_doing,
            }
        )

    paper_rows = [
        {
            "paper": "RegionGPT",
            "source": "https://arxiv.org/abs/2403.02330",
            "relevant lesson": "Generic VLMs have weak detailed region understanding because region features and instruction data are not naturally fine-grained.",
            "skill implication": "Do not trust a global image prompt for tiny tissue pieces; score marked regions or use explicit region features.",
            "concrete branch": "piece + local context + full ROI locator, or numeric region feature extractor",
        },
        {
            "paper": "Set-of-Mark Prompting",
            "source": "https://arxiv.org/abs/2310.11441",
            "relevant lesson": "Overlaying explicit marks/IDs helps multimodal models ground answers to the intended region.",
            "skill implication": "If a VLM is used, show candidate IDs/contours on a context view; avoid ambiguous unmarked crops.",
            "concrete branch": "marked candidate context pack for manual/API GPT checks",
        },
        {
            "paper": "MedSAM",
            "source": "https://pubmed.ncbi.nlm.nih.gov/38253604/",
            "relevant lesson": "Medical segmentation is stronger when the target is specified by a prompt such as a box rather than expecting a generic model to infer the final mask.",
            "skill implication": "Use ranker/VLM to choose region prompts, then prompt SAM/Medical-SAM again for final mask refinement.",
            "concrete branch": "selected-piece bbox/point second-SAM focus",
        },
        {
            "paper": "Alpha-CLIP",
            "source": "https://arxiv.org/abs/2312.03818",
            "relevant lesson": "Mask-indicated visual prompting can preserve context while telling the model which region matters.",
            "skill implication": "Treat candidate mask as an attention cue, not just a crop; keep surrounding H&E architecture visible.",
            "concrete branch": "mask-aware context scoring and shape/context features",
        },
        {
            "paper": "SaLIP",
            "source": "https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.html",
            "relevant lesson": "A practical cascade first finds promising regions, then reruns SAM with prompts rather than accepting the initial mask directly.",
            "skill implication": "Make the ranker a proposal selector; final mask quality can come from constrained assembly or second-SAM.",
            "concrete branch": "proposal-rerank-assembly/second-SAM cascade",
        },
    ]

    queue_rows = [
        {
            "priority": 1,
            "experiment": "minimal three-setting proposal gate",
            "changes one layer": "proposal generator",
            "why this is first": "alveoli is impossible in medpt24-only pool; no scorer can recover absent mask proposals",
            "success criterion": "alveoli reaches about 0.65-0.68 Dice without bronchiola/vessels regression",
            "next if success": "train/calibrate skill ranker on this smaller proposal pool",
            "next if failure": "class-specific alveoli generator branch",
        },
        {
            "priority": 2,
            "experiment": "marked/context region scorer",
            "changes one layer": "recognition input",
            "why this is first": "RegionGPT/SoM suggest ambiguous region grounding is a real VLM failure mode",
            "success criterion": "bronchiola/vessels Piece Top1 or target-vs-rest ranking improves without tumor collapse",
            "next if success": "use marked/context views only as a tissue-membership skill",
            "next if failure": "drop VLM for this layer and use structured numeric features",
        },
        {
            "priority": 3,
            "experiment": "deployable mask-quality proxy",
            "changes one layer": "quality skill",
            "why this is first": "quality-adjusted oracle head works, but annotation-trained quality is not deployable",
            "success criterion": "correlates with component Dice and selects high-quality vessels/bronchiola without compact_funnel_score",
            "next if success": "multiply/gate tissue score by quality score",
            "next if failure": "use second-SAM or stronger shape descriptors instead of quality regression",
        },
        {
            "priority": 4,
            "experiment": "second-SAM selected locator refinement",
            "changes one layer": "mask refinement",
            "why this is first": "selected pieces may locate the right tissue but have incomplete boundaries",
            "success criterion": "recall improves while precision stays acceptable",
            "next if success": "final method becomes proposal scorer plus prompted SAM",
            "next if failure": "keep raw union and tune assembly/NMS only",
        },
    ]

    write_csv(OUT / "class_failure_engine.csv", class_rows)
    write_csv(OUT / "paper_to_skill_design.csv", paper_rows)
    write_csv(OUT / "iterative_refinement_queue.csv", queue_rows)

    section = f"""
<h2>17N. Paper-Informed Failure Engine</h2>
<p>This section makes the skill iterative instead of merely runnable. The method now asks, for each tissue class, which layer failed: proposal generation, tissue recognition, mask-quality ranking, assembly/NMS, or final refinement. Only the failing layer should be changed. This avoids the earlier pattern of repeatedly changing prompts when the real problem is a missing candidate mask or an assembly policy.</p>
<h3>Paper ideas translated into skill design</h3>
{table(paper_rows, ['paper', 'source', 'relevant lesson', 'skill implication', 'concrete branch'], link_col='source')}
<h3>Current failure engine by class</h3>
{table(class_rows, ['class', 'proposal upper bound', 'weak components', 'recognition signal', 'quality correlation', 'latest runtime union D/P/R', 'primary failure', 'next move', 'do not do'])}
<h3>Refinement queue</h3>
{table(queue_rows, ['priority', 'experiment', 'changes one layer', 'why this is first', 'success criterion', 'next if success', 'next if failure'])}
<div class='callout'><b>Decision.</b> The next real test is still the minimal three-setting proposal gate. The reason is now sharper: alveoli is blocked before the scorer sees it. For vessels, the scorer layer is already improved by FICTURE-dominant scoring, so the next question is validation/refinement, not prompt tuning. For broad/diffuse classes, the skill should branch toward broader candidates or compartment priors instead of object-like small-piece union.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17N. Paper-Informed Failure Engine</h2>"
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
        BASE / "alveoli_generator_gate",
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

    print(OUT / "class_failure_engine.csv")
    print(OUT / "paper_to_skill_design.csv")
    print(OUT / "iterative_refinement_queue.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
