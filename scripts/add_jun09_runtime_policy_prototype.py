#!/usr/bin/env python3
"""Runtime-only policy prototype for Jun09.

This turns the 17R feature-space diagnosis into a concrete runtime policy.
Scores and selection do not use compact_funnel_score, component Dice,
Precision, Recall, or hidden labels. Annotation is used only after the policy
selects pieces, to evaluate the final union.

This is a prototype, not a completed external validation: formulas and class
caps are paper/biology-inspired rules written down from the current failure
analysis, then evaluated on this one ROI.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "runtime_policy_prototype"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"

spec = importlib.util.spec_from_file_location("qa_helpers", QA_SCRIPT)
qa = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qa)

CLASS_KEYS = qa.CLASS_KEYS


POLICIES = {
    "bronchiola": {
        "formula": "score_fig_comp_quality",
        "max_pieces": 3,
        "max_overlap": 0.90,
        "threshold_rule": "top_score_fraction",
        "threshold_param": 0.35,
        "reason": "airway epithelial FICTURE composition plus shape/area quality; few disconnected airway pieces expected",
    },
    "alveoli": {
        "formula": "score_shape_only",
        "max_pieces": 3,
        "max_overlap": 0.35,
        "threshold_rule": "top_score_fraction",
        "threshold_param": 0.45,
        "reason": "alveoli has weak FICTURE prior in this pool; shape is kept only as a diagnostic fallback",
    },
    "vessels": {
        "formula": "score_fig_only",
        "max_pieces": 8,
        "max_overlap": 0.35,
        "threshold_rule": "top_score_fraction",
        "threshold_param": 0.55,
        "reason": "endothelial FICTURE prior is the strongest runtime vessel signal; low-overlap NMS keeps spatial diversity",
    },
    "tumor": {
        "formula": "score_fig_shape",
        "max_pieces": 24,
        "max_overlap": 0.90,
        "threshold_rule": "top_score_fraction",
        "threshold_param": 0.65,
        "reason": "broad epithelial tumor compartment; use FICTURE plus shape but keep this branch conservative",
    },
    "stroma": {
        "formula": "score_fig_shape",
        "max_pieces": 48,
        "max_overlap": 0.90,
        "threshold_rule": "top_score_fraction",
        "threshold_param": 0.50,
        "reason": "broad stromal compartment; allow more pieces but do not use annotation-derived quality",
    },
    "immune_infiltration": {
        "formula": "score_fig_comp_quality",
        "max_pieces": 48,
        "max_overlap": 0.75,
        "threshold_rule": "top_score_fraction",
        "threshold_param": 0.55,
        "reason": "many small immune pieces; structured immune composition plus quality proxy, with moderate NMS",
    },
}


def display_class(name: str) -> str:
    return "immune infiltration" if name == "immune_infiltration" else name


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


def threshold_for_scores(scores: pd.Series, rule: str, param: float) -> float:
    scores = scores.astype(float)
    if len(scores) == 0:
        return 1.0
    if rule == "top_score_fraction":
        return float(scores.max()) * float(param)
    if rule == "quantile":
        return float(scores.quantile(param))
    raise ValueError(rule)


def select_runtime_pieces(
    class_df: pd.DataFrame,
    score_col: str,
    get_mask,
    canvas_shape: tuple[int, int],
    threshold: float,
    max_overlap: float,
    max_pieces: int,
) -> tuple[list[pd.Series], np.ndarray]:
    union = np.zeros(canvas_shape, dtype=bool)
    selected: list[pd.Series] = []
    for _, row in class_df.sort_values(score_col, ascending=False).iterrows():
        if float(row[score_col]) < threshold:
            break
        mask = get_mask(row["row_key"])
        if qa.overlap_fraction(mask, union) > max_overlap:
            continue
        selected.append(row)
        union |= mask
        if len(selected) >= max_pieces:
            break
    return selected, union


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(qa.HE_ROI).size
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(qa.row_key, axis=1)
    public_by_key = public.set_index("row_key").to_dict("index")
    pair = pd.read_csv(BASE / "feature_space_upper_bound/feature_space_pair_scores.csv")
    upper = pd.read_csv(BASE / "feature_space_upper_bound/feature_space_upper_bound_summary.csv")
    upper_by = {str(r["class"]).replace("immune infiltration", "immune_infiltration"): r for _, r in upper.iterrows()}

    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        if str(path) not in mask_cache:
            mask_cache[str(path)] = qa.load_mask(path, size)
        return mask_cache[str(path)]

    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        policy = POLICIES[tissue_class]
        score_col = policy["formula"]
        class_df = pair[pair["target_class"] == tissue_class].copy()
        threshold = threshold_for_scores(
            class_df[score_col],
            str(policy["threshold_rule"]),
            float(policy["threshold_param"]),
        )
        selected, union = select_runtime_pieces(
            class_df,
            score_col,
            get_mask,
            (size[1], size[0]),
            threshold,
            float(policy["max_overlap"]),
            int(policy["max_pieces"]),
        )
        d, p, r = qa.metrics(union, annotations[tissue_class])
        upper_row = upper_by.get(tissue_class, {})
        upper_d = float(upper_row.get("dice", np.nan))
        gap = upper_d - d if not np.isnan(upper_d) else np.nan
        if tissue_class == "alveoli":
            verdict = "proposal failure remains; runtime policy cannot solve absent broad alveoli proposal"
        elif gap <= 0.05 and d >= 0.65:
            verdict = "runtime policy is close to feature-space upper bound; next step is validation/refinement"
        elif d >= 0.45:
            verdict = "runtime policy partially works; next step is calibrating quality/threshold or second-SAM"
        else:
            verdict = "runtime policy weak; move to new encoder/proposal/context branch"
        fig = OUT / "figures" / f"{tissue_class}_runtime_policy_union.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            f"{tissue_class}: runtime-only policy prototype",
            f"{policy['formula']}, threshold={threshold:.3f}, overlap<={policy['max_overlap']}, max_pieces={policy['max_pieces']}. Score/selection do not use annotation.",
        )
        summary_rows.append(
            {
                "class": display_class(tissue_class),
                "formula": score_col.replace("score_", ""),
                "selected_piece_count": len(selected),
                "threshold": f"{threshold:.3f}",
                "dice": f"{d:.3f}",
                "precision": f"{p:.3f}",
                "recall": f"{r:.3f}",
                "feature_upper_bound_dice": f"{upper_d:.3f}",
                "upper_bound_gap": f"{gap:.3f}",
                "verdict": verdict,
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        policy_rows.append(
            {
                "class": display_class(tissue_class),
                "formula": score_col.replace("score_", ""),
                "threshold_rule": policy["threshold_rule"],
                "threshold_param": policy["threshold_param"],
                "max_overlap": policy["max_overlap"],
                "max_pieces": policy["max_pieces"],
                "reason": policy["reason"],
            }
        )
        for rank, row in enumerate(selected[:120], 1):
            md, mp, mr = qa.metrics(get_mask(row["row_key"]), annotations[tissue_class])
            selected_rows.append(
                {
                    "class": display_class(tissue_class),
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "score": f"{float(row[score_col]):.4f}",
                    "threshold": f"{threshold:.4f}",
                    "true_class_hidden": row["true_class"],
                    "target_component_dice": f"{float(row['component_dice']):.3f}",
                    "fullres_target_dice": f"{md:.3f}",
                    "fullres_target_precision": f"{mp:.3f}",
                    "fullres_target_recall": f"{mr:.3f}",
                    "he": f"{float(row['he']):.3f}",
                    "fig": f"{float(row['fig']):.3f}",
                    "shape": f"{float(row['shape']):.3f}",
                    "comp": f"{float(row['comp']):.3f}",
                    "area_ok": f"{float(row['area_ok']):.3f}",
                }
            )

    write_csv(OUT / "runtime_policy_definition.csv", policy_rows)
    write_csv(OUT / "runtime_policy_summary.csv", summary_rows)
    write_csv(OUT / "runtime_policy_selected_pieces.csv", selected_rows)
    state = {
        "status": "prototype_not_final",
        "selection_uses_hidden_metrics": False,
        "evaluation_uses_annotation": True,
        "policy_source": "paper/biology/failure-analysis rules from 17Q-17R, not optimized by per-candidate hidden Dice",
        "summary": summary_rows,
    }
    (OUT / "runtime_policy_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    section = f"""
<h2>17S. Runtime-Only Policy Prototype: Can The Skill Select Masks Without Hidden Dice?</h2>
<p>This section converts the 17R upper-bound diagnosis into a concrete runtime policy. The policy uses only deployment-time scores: H&amp;E morphology, structured FICTURE composition prior, shape/location, and simple score thresholds. It does <b>not</b> use <code>compact_funnel_score</code>, component Dice, Precision, Recall, or hidden labels for selection.</p>
<p>The policy is still a prototype: formulas and class caps are chosen from tissue biology and the failure-analysis controller, then evaluated on this one ROI. Annotation is used only after selection to compute Dice / Precision / Recall.</p>
<h3>Policy definition</h3>
{table(policy_rows, ['class', 'formula', 'threshold_rule', 'threshold_param', 'max_overlap', 'max_pieces', 'reason'])}
<h3>Runtime-policy result</h3>
{table(summary_rows, ['class', 'formula', 'selected_piece_count', 'threshold', 'dice', 'precision', 'recall', 'feature_upper_bound_dice', 'upper_bound_gap', 'verdict'])}
<h3>Six-panel visual check</h3>
<div class='image-grid'>
{''.join(f"<figure><img src='{html.escape(row['figure_rel'])}'><figcaption>{html.escape(row['class'])}: {html.escape(row['dice'])}/{html.escape(row['precision'])}/{html.escape(row['recall'])}</figcaption></figure>" for row in summary_rows)}
</div>
<h3>Selected pieces</h3>
{table(selected_rows[:220], ['class', 'rank', 'candidate_uid', 'score', 'threshold', 'true_class_hidden', 'target_component_dice', 'fullres_target_dice', 'fullres_target_precision', 'fullres_target_recall', 'he', 'fig', 'shape', 'comp', 'area_ok'])}
<div class='callout'><b>Interpretation.</b> If this runtime policy is close to the 17R upper bound, the remaining work is validation and refinement. If it is far below 17R, the feature scores may contain useful information but the policy needs calibration. If both this and 17R are weak, the next iteration must change proposal generation, add a pathology encoder, or use second-SAM refinement.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17S. Runtime-Only Policy Prototype: Can The Skill Select Masks Without Hidden Dice?</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    print(OUT / "runtime_policy_summary.csv")


if __name__ == "__main__":
    main()
