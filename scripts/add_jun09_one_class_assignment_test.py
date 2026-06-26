#!/usr/bin/env python3
"""Test a one-class assignment constraint for the Jun09 runtime skill.

The 17W audit showed many selected false positives caused by the same candidate
being attractive to multiple tissue classes.  This diagnostic adds a deployable
constraint: each candidate can be assigned to at most one target class before
per-class NMS/union.  Annotation is used only after selection to evaluate the
effect, and a small margin grid is reported as diagnostic evidence.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "one_class_assignment_test"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"
RUNTIME_SCRIPT = ROOT / "scripts/add_jun09_runtime_policy_prototype.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qa = import_module(QA_SCRIPT, "qa_helpers_oneclass")
runtime = import_module(RUNTIME_SCRIPT, "runtime_helpers_oneclass")

CLASS_KEYS = qa.CLASS_KEYS


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
        BASE / "alveoli_broad_box_branch",
        BASE / "failure_debugger_v2",
        BASE / "runtime_candidate_error_auditor",
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


def build_eligible(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for tissue_class in CLASS_KEYS:
        policy = runtime.POLICIES[tissue_class]
        score_col = str(policy["formula"])
        cls_df = pair[pair["target_class"] == tissue_class].copy()
        threshold = runtime.threshold_for_scores(
            cls_df[score_col],
            str(policy["threshold_rule"]),
            float(policy["threshold_param"]),
        )
        cls_df["runtime_score"] = cls_df[score_col].astype(float)
        cls_df["runtime_threshold"] = threshold
        cls_df["normalized_score"] = cls_df["runtime_score"] / max(threshold, 1e-8)
        cls_df["score_formula"] = score_col
        cls_df = cls_df[cls_df["runtime_score"] >= threshold]
        rows.append(cls_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def assign_one_class(eligible: pd.DataFrame, margin_min: float) -> pd.DataFrame:
    assigned_rows = []
    for _, group in eligible.groupby("candidate_uid"):
        group = group.sort_values("normalized_score", ascending=False)
        best = group.iloc[0].copy()
        second = float(group.iloc[1]["normalized_score"]) if len(group) > 1 else 0.0
        margin = float(best["normalized_score"]) - second
        if margin < margin_min:
            continue
        best["assignment_margin"] = margin
        best["second_best_normalized_score"] = second
        assigned_rows.append(best)
    if not assigned_rows:
        return pd.DataFrame()
    return pd.DataFrame(assigned_rows)


def run_assembly(assigned: pd.DataFrame, margin_min: float, public_by_key: dict[str, dict[str, object]], annotations: dict[str, np.ndarray], get_mask, canvas_shape: tuple[int, int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        policy = runtime.POLICIES[tissue_class]
        cls_df = assigned[assigned["target_class"] == tissue_class].copy()
        union = np.zeros(canvas_shape, dtype=bool)
        selected: list[pd.Series] = []
        for _, row in cls_df.sort_values("normalized_score", ascending=False).iterrows():
            mask = get_mask(str(row["row_key"]))
            if qa.overlap_fraction(mask, union) > float(policy["max_overlap"]):
                continue
            selected.append(row)
            union |= mask
            if len(selected) >= int(policy["max_pieces"]):
                break
        d, p, r = qa.metrics(union, annotations[tissue_class])
        fig = OUT / "figures" / f"{tissue_class}_one_class_assignment_margin{margin_min:.2f}.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            "One-class assignment runtime union",
            f"Each candidate can enter at most one class; assignment margin >= {margin_min:.2f}.",
        )
        selected_true = sum(1 for row in selected if str(row["true_class"]) == tissue_class)
        selected_false = len(selected) - selected_true
        summary_rows.append(
            {
                "margin": margin_min,
                "class": display_class(tissue_class),
                "selected_piece_count": len(selected),
                "selected_true_hidden": selected_true,
                "selected_false_positive_hidden": selected_false,
                "dice": round(float(d), 3),
                "precision": round(float(p), 3),
                "recall": round(float(r), 3),
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        for rank, row in enumerate(selected, 1):
            mask = get_mask(str(row["row_key"]))
            sd, sp, sr = qa.metrics(mask, annotations[tissue_class])
            selected_rows.append(
                {
                    "margin": margin_min,
                    "class": display_class(tissue_class),
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "assigned_target_class": display_class(str(row["target_class"])),
                    "true_class_hidden": display_class(str(row["true_class"])),
                    "runtime_score": round(float(row["runtime_score"]), 4),
                    "normalized_score": round(float(row["normalized_score"]), 4),
                    "assignment_margin": round(float(row["assignment_margin"]), 4),
                    "single_piece_target_dice": round(float(sd), 3),
                    "single_piece_target_precision": round(float(sp), 3),
                    "single_piece_target_recall": round(float(sr), 3),
                    "he_crop_rel": public_by_key[str(row["row_key"])]["he_crop_rel"],
                    "ficture_crop_rel": public_by_key[str(row["row_key"])]["ficture_crop_rel"],
                }
            )
    return summary_rows, selected_rows


def append_report(section: str) -> None:
    marker = "<h2>17X. One-Class Assignment Test: Can Mutual Exclusion Reduce False Positives?</h2>"
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
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(qa.HE_ROI).size
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(qa.row_key, axis=1)
    public_by_key = public.set_index("row_key").to_dict("index")
    pair = pd.read_csv(BASE / "feature_space_upper_bound/feature_space_pair_scores.csv")
    runtime_summary = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    runtime_summary["margin"] = "no one-class constraint"

    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        path_s = str(path)
        if path_s not in mask_cache:
            mask_cache[path_s] = qa.load_mask(path, size)
        return mask_cache[path_s]

    eligible = build_eligible(pair)
    write_csv(OUT / "one_class_assignment_eligible_pairs.csv", eligible.to_dict("records"))

    all_summary: list[dict[str, object]] = []
    all_selected: list[dict[str, object]] = []
    for margin in [0.0, 0.05, 0.10, 0.20, 0.35]:
        assigned = assign_one_class(eligible, margin)
        write_csv(OUT / f"one_class_assigned_margin{margin:.2f}.csv", assigned.to_dict("records"))
        summary, selected = run_assembly(
            assigned,
            margin,
            public_by_key,
            annotations,
            get_mask,
            (size[1], size[0]),
        )
        all_summary.extend(summary)
        all_selected.extend(selected)

    write_csv(OUT / "one_class_assignment_summary_grid.csv", all_summary)
    write_csv(OUT / "one_class_assignment_selected_pieces.csv", all_selected)

    fixed = [r for r in all_summary if r["margin"] == 0.05]
    best_by_class = []
    for tissue_class in [display_class(c) for c in CLASS_KEYS]:
        cls_rows = [r for r in all_summary if r["class"] == tissue_class]
        best = max(cls_rows, key=lambda row: row["dice"])
        best_by_class.append(best)
    write_csv(OUT / "one_class_assignment_fixed_margin005_summary.csv", fixed)
    write_csv(OUT / "one_class_assignment_oracle_best_margin_by_class.csv", best_by_class)

    comparison_rows: list[dict[str, object]] = []
    runtime_by = {str(r["class"]).replace("_", " "): r for _, r in runtime_summary.iterrows()}
    fixed_by = {r["class"]: r for r in fixed}
    best_by = {r["class"]: r for r in best_by_class}
    for cls in [display_class(c) for c in CLASS_KEYS]:
        old = runtime_by.get(cls, {})
        new = fixed_by.get(cls, {})
        best = best_by.get(cls, {})
        comparison_rows.append(
            {
                "class": cls,
                "runtime D/P/R": f"{float(old.get('dice', 0)):.3f}/{float(old.get('precision', 0)):.3f}/{float(old.get('recall', 0)):.3f}",
                "one-class fixed 0.05 D/P/R": f"{float(new.get('dice', 0)):.3f}/{float(new.get('precision', 0)):.3f}/{float(new.get('recall', 0)):.3f}",
                "fixed selected true/false": f"{new.get('selected_true_hidden', 0)}/{new.get('selected_false_positive_hidden', 0)}",
                "oracle best margin D/P/R": f"{float(best.get('dice', 0)):.3f}/{float(best.get('precision', 0)):.3f}/{float(best.get('recall', 0)):.3f}",
                "oracle best margin": best.get("margin", ""),
                "verdict": one_class_verdict(float(old.get("dice", 0)), float(new.get("dice", 0)), float(best.get("dice", 0))),
            }
        )
    write_csv(OUT / "one_class_assignment_runtime_comparison.csv", comparison_rows)

    section = f"""
<h2>17X. One-Class Assignment Test: Can Mutual Exclusion Reduce False Positives?</h2>
<p><b>Goal.</b> Section 17W showed that several classes selected many cross-class false positives. This test adds one deployable constraint before union: one candidate piece can enter at most one tissue class. The assigned class is the class whose runtime score is strongest relative to its own class threshold.</p>
<p><b>Rule tested.</b> First, each class applies the same runtime formula and threshold as 17T. Then each candidate keeps only its best normalized class score. If the best score is not at least a small margin above the second-best score, the candidate is rejected. Finally the usual class-specific overlap filtering and max-piece cap are applied.</p>
<p><b>Ground truth use.</b> Annotation is hidden during assignment and selection; Dice/Precision/Recall and true/false counts below are post-hoc evaluation. The fixed margin 0.05 row is the deployable-style test. The oracle-best margin column is diagnostic only and tells us whether margin calibration could help.</p>
<h3>Runtime vs one-class assignment</h3>
{table(comparison_rows, ['class', 'runtime D/P/R', 'one-class fixed 0.05 D/P/R', 'fixed selected true/false', 'oracle best margin D/P/R', 'oracle best margin', 'verdict'])}
<h3>Fixed margin 0.05 selected-union figures</h3>
<div class='figure-grid'>
{''.join(f"<figure><img src='{html.escape(str(Path(r['figure_rel']).as_posix()))}' alt='{html.escape(str(r['class']))} one-class union'><figcaption>{html.escape(str(r['class']))}: D/P/R {r['dice']:.3f}/{r['precision']:.3f}/{r['recall']:.3f}</figcaption></figure>" for r in fixed)}
</div>
<div class='callout'><b>17X interpretation.</b> If fixed one-class assignment improves precision but harms recall, the problem is not solved by mutual exclusion alone; we need class-specific recall recovery or second-SAM. If oracle-best margin improves a class, the next iteration is threshold calibration. If both are weak, return to proposal or quality features rather than prompt tuning.</div>
"""
    append_report(section)
    rebuild_zip()
    print(OUT / "one_class_assignment_runtime_comparison.csv")
    print(OUT / "one_class_assignment_summary_grid.csv")
    print(OUT / "one_class_assignment_selected_pieces.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


def one_class_verdict(runtime_dice: float, fixed_dice: float, oracle_dice: float) -> str:
    if fixed_dice > runtime_dice + 0.03:
        return "fixed mutual-exclusion helps; keep it as runtime rule candidate"
    if oracle_dice > runtime_dice + 0.05:
        return "margin calibration may help, but fixed rule is not enough"
    if fixed_dice < runtime_dice - 0.05:
        return "mutual-exclusion alone hurts recall; use only as veto/guard"
    return "neutral; keep diagnosis but do not rely on it as main fix"


if __name__ == "__main__":
    main()
