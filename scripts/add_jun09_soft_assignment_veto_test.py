#!/usr/bin/env python3
"""Test a soft cross-class veto for the Jun09 runtime skill.

Hard one-class assignment reduced recall badly for several classes.  This
variant keeps the per-class runtime selector but rejects a target assignment
only when the target score is far below the candidate's best score for another
class.  It is a deployable guard because it uses only runtime scores.
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
OUT = BASE / "soft_assignment_veto_test"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"
RUNTIME_SCRIPT = ROOT / "scripts/add_jun09_runtime_policy_prototype.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qa = import_module(QA_SCRIPT, "qa_helpers_softveto")
runtime = import_module(RUNTIME_SCRIPT, "runtime_helpers_softveto")
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
        BASE / "one_class_assignment_test",
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


def normalized_pair_scores(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for tissue_class in CLASS_KEYS:
        policy = runtime.POLICIES[tissue_class]
        score_col = str(policy["formula"])
        cls = pair[pair["target_class"] == tissue_class].copy()
        threshold = runtime.threshold_for_scores(cls[score_col], str(policy["threshold_rule"]), float(policy["threshold_param"]))
        cls["runtime_score"] = cls[score_col].astype(float)
        cls["runtime_threshold"] = threshold
        cls["normalized_score"] = cls["runtime_score"] / max(threshold, 1e-8)
        cls["score_formula"] = score_col
        rows.append(cls)
    out = pd.concat(rows, ignore_index=True)
    best = out.groupby("candidate_uid")["normalized_score"].max().rename("best_normalized_score")
    out = out.merge(best, on="candidate_uid", how="left")
    out["target_vs_best_ratio"] = out["normalized_score"] / out["best_normalized_score"].clip(lower=1e-8)
    return out


def run_soft_veto(scores: pd.DataFrame, ratio_min: float, public_by_key: dict[str, dict[str, object]], annotations: dict[str, np.ndarray], get_mask, canvas_shape: tuple[int, int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        policy = runtime.POLICIES[tissue_class]
        cls = scores[
            (scores["target_class"] == tissue_class)
            & (scores["runtime_score"] >= scores["runtime_threshold"])
            & (scores["target_vs_best_ratio"] >= ratio_min)
        ].copy()
        union = np.zeros(canvas_shape, dtype=bool)
        selected = []
        for _, row in cls.sort_values("normalized_score", ascending=False).iterrows():
            mask = get_mask(str(row["row_key"]))
            if qa.overlap_fraction(mask, union) > float(policy["max_overlap"]):
                continue
            selected.append(row)
            union |= mask
            if len(selected) >= int(policy["max_pieces"]):
                break
        d, p, r = qa.metrics(union, annotations[tissue_class])
        fig = OUT / "figures" / f"{tissue_class}_soft_veto_ratio{ratio_min:.2f}.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            "Soft cross-class veto runtime union",
            f"Keep target if target normalized score / best normalized class score >= {ratio_min:.2f}.",
        )
        selected_true = sum(1 for row in selected if str(row["true_class"]) == tissue_class)
        selected_false = len(selected) - selected_true
        summary_rows.append(
            {
                "ratio_min": ratio_min,
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
                    "ratio_min": ratio_min,
                    "class": display_class(tissue_class),
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "true_class_hidden": display_class(str(row["true_class"])),
                    "runtime_score": round(float(row["runtime_score"]), 4),
                    "normalized_score": round(float(row["normalized_score"]), 4),
                    "target_vs_best_ratio": round(float(row["target_vs_best_ratio"]), 4),
                    "single_piece_target_dice": round(float(sd), 3),
                    "single_piece_target_precision": round(float(sp), 3),
                    "single_piece_target_recall": round(float(sr), 3),
                    "he_crop_rel": public_by_key[str(row["row_key"])]["he_crop_rel"],
                    "ficture_crop_rel": public_by_key[str(row["row_key"])]["ficture_crop_rel"],
                }
            )
    return summary_rows, selected_rows


def verdict(runtime_dice: float, fixed_dice: float, best_dice: float) -> str:
    if fixed_dice > runtime_dice + 0.03:
        return "soft veto helps as a deployable guard"
    if best_dice > runtime_dice + 0.04:
        return "soft veto may help but ratio needs calibration"
    if fixed_dice < runtime_dice - 0.05:
        return "soft veto hurts recall; keep original selector for now"
    return "neutral; not the main fix"


def append_report(section: str) -> None:
    marker = "<h2>17Y. Soft Cross-Class Veto: A Less Destructive Alternative To One-Class Assignment</h2>"
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
    scores = normalized_pair_scores(pair)
    write_csv(OUT / "soft_veto_normalized_pair_scores.csv", scores.to_dict("records"))

    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        path_s = str(path)
        if path_s not in mask_cache:
            mask_cache[path_s] = qa.load_mask(path, size)
        return mask_cache[path_s]

    all_summary: list[dict[str, object]] = []
    all_selected: list[dict[str, object]] = []
    for ratio in [0.00, 0.40, 0.55, 0.70, 0.85]:
        summary, selected = run_soft_veto(scores, ratio, public_by_key, annotations, get_mask, (size[1], size[0]))
        all_summary.extend(summary)
        all_selected.extend(selected)
    write_csv(OUT / "soft_veto_summary_grid.csv", all_summary)
    write_csv(OUT / "soft_veto_selected_pieces.csv", all_selected)

    runtime_summary = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    runtime_by = {display_class(str(r["class"])): r for _, r in runtime_summary.iterrows()}
    fixed = [r for r in all_summary if r["ratio_min"] == 0.55]
    fixed_by = {r["class"]: r for r in fixed}
    best_by_class = []
    for cls in [display_class(c) for c in CLASS_KEYS]:
        cls_rows = [r for r in all_summary if r["class"] == cls]
        best_by_class.append(max(cls_rows, key=lambda row: row["dice"]))
    best_by = {r["class"]: r for r in best_by_class}
    comparison_rows: list[dict[str, object]] = []
    for cls in [display_class(c) for c in CLASS_KEYS]:
        old = runtime_by[cls]
        new = fixed_by[cls]
        best = best_by[cls]
        comparison_rows.append(
            {
                "class": cls,
                "runtime D/P/R": f"{float(old['dice']):.3f}/{float(old['precision']):.3f}/{float(old['recall']):.3f}",
                "soft-veto ratio 0.55 D/P/R": f"{float(new['dice']):.3f}/{float(new['precision']):.3f}/{float(new['recall']):.3f}",
                "ratio 0.55 selected true/false": f"{new['selected_true_hidden']}/{new['selected_false_positive_hidden']}",
                "oracle best ratio D/P/R": f"{float(best['dice']):.3f}/{float(best['precision']):.3f}/{float(best['recall']):.3f}",
                "oracle best ratio": best["ratio_min"],
                "verdict": verdict(float(old["dice"]), float(new["dice"]), float(best["dice"])),
            }
        )
    write_csv(OUT / "soft_veto_runtime_comparison.csv", comparison_rows)

    section = f"""
<h2>17Y. Soft Cross-Class Veto: A Less Destructive Alternative To One-Class Assignment</h2>
<p><b>Goal.</b> Hard one-class assignment hurt recall because small tissue pieces can be ambiguous or share features across classes. This test keeps the original per-class runtime selector but rejects a target assignment only when that target is much weaker than the candidate's best class score.</p>
<p><b>Rule tested.</b> For every candidate-class pair, compute target normalized score divided by the candidate's best normalized class score. A ratio of 0.55 means the target class must be at least 55% as plausible as the best class. This is a soft veto, not a forced unique assignment.</p>
<p><b>Ground truth use.</b> Annotation is hidden during scoring and vetoing; it is used only after union for evaluation.</p>
<h3>Runtime vs soft-veto assignment</h3>
{table(comparison_rows, ['class', 'runtime D/P/R', 'soft-veto ratio 0.55 D/P/R', 'ratio 0.55 selected true/false', 'oracle best ratio D/P/R', 'oracle best ratio', 'verdict'])}
<h3>Soft-veto ratio 0.55 selected-union figures</h3>
<div class='figure-grid'>
{''.join(f"<figure><img src='{html.escape(str(Path(r['figure_rel']).as_posix()))}' alt='{html.escape(str(r['class']))} soft veto union'><figcaption>{html.escape(str(r['class']))}: D/P/R {r['dice']:.3f}/{r['precision']:.3f}/{r['recall']:.3f}</figcaption></figure>" for r in fixed)}
</div>
<div class='callout'><b>17Y interpretation.</b> Soft veto is only useful if it removes false positives without destroying recall. If it is neutral or harmful, the next improvement must come from a better quality signal, proposal branch, or second-SAM refinement rather than cross-class competition rules.</div>
"""
    append_report(section)
    rebuild_zip()
    print(OUT / "soft_veto_runtime_comparison.csv")
    print(OUT / "soft_veto_summary_grid.csv")
    print(OUT / "soft_veto_selected_pieces.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
