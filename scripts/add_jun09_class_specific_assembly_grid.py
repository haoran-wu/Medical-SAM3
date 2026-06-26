#!/usr/bin/env python3
"""Class-specific assembly grid for Jun09 skill ranker.

This diagnostic asks whether the current deployable score columns are already
good enough if each tissue class gets its own formula, score threshold,
overlap/NMS threshold, and max-piece cap.  Candidate selection in every grid
cell uses only deployable scores.  Annotation is used after selection to choose
diagnostic best settings and to decide whether the bottleneck is assembly or an
earlier layer.
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
OUT = BASE / "class_specific_assembly_grid"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qa = import_module(QA_SCRIPT, "qa_helpers_class_grid")
CLASS_KEYS = qa.CLASS_KEYS

FORMULAS = [
    "score_he_only",
    "score_fig_only",
    "score_shape_only",
    "score_comp_only",
    "score_he_shape",
    "score_fig_shape",
    "score_he_fig_shape",
    "score_he_shape_quality",
    "score_fig_comp_quality",
    "score_balanced_static",
    "score_margin_balanced",
    "score_quality_gate_static",
]

MIN_PRECISION = {
    "bronchiola": 0.85,
    "alveoli": 0.35,
    "vessels": 0.85,
    "tumor": 0.55,
    "stroma": 0.35,
    "immune_infiltration": 0.40,
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


def threshold_for(scores: pd.Series, fraction: float) -> float:
    if scores.empty:
        return 1.0
    return float(scores.max()) * fraction


def select_candidates(
    class_df: pd.DataFrame,
    formula: str,
    threshold_fraction: float,
    max_overlap: float,
    max_pieces: int,
    get_mask,
    canvas_shape: tuple[int, int],
) -> tuple[list[pd.Series], np.ndarray, float]:
    threshold = threshold_for(class_df[formula].astype(float), threshold_fraction)
    union = np.zeros(canvas_shape, dtype=bool)
    selected: list[pd.Series] = []
    for _, row in class_df.sort_values(formula, ascending=False).iterrows():
        if float(row[formula]) < threshold:
            break
        mask = get_mask(str(row["row_key"]))
        if qa.overlap_fraction(mask, union) > max_overlap:
            continue
        selected.append(row)
        union |= mask
        if len(selected) >= max_pieces:
            break
    return selected, union, threshold


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
        BASE / "soft_assignment_veto_test",
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


def choose_best(rows: list[dict[str, object]], tissue_class: str) -> dict[str, object]:
    cls_rows = [row for row in rows if row["class_key"] == tissue_class]
    safe = [row for row in cls_rows if float(row["precision"]) >= MIN_PRECISION[tissue_class]]
    pool = safe if safe else cls_rows
    return max(pool, key=lambda row: (float(row["dice"]), float(row["recall"]), float(row["precision"])))


def verdict(runtime_dice: float, best: dict[str, object], tissue_class: str) -> str:
    best_dice = float(best["dice"])
    if tissue_class == "alveoli":
        return "assembly grid cannot fix the narrow medpt24 proposal; keep proposal-generator branch as first priority"
    if best_dice > runtime_dice + 0.05:
        return "class-specific assembly improves this class; promote this policy only after validation"
    if best_dice > runtime_dice + 0.015:
        return "small assembly improvement; useful diagnostic but not a full fix"
    return "assembly hyperparameters are not the main bottleneck"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(qa.HE_ROI).size
    small_size = (512, int(round(512 * size[1] / size[0])))
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    small_annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], small_size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(qa.row_key, axis=1)
    public_by_key = public.set_index("row_key").to_dict("index")
    pair = pd.read_csv(BASE / "feature_space_upper_bound/feature_space_pair_scores.csv")
    runtime_summary = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    runtime_by = {str(r["class"]).replace("immune infiltration", "immune_infiltration"): r for _, r in runtime_summary.iterrows()}

    mask_cache: dict[str, np.ndarray] = {}
    small_mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        path_s = str(path)
        if path_s not in mask_cache:
            mask_cache[path_s] = qa.load_mask(path, size)
        return mask_cache[path_s]

    def get_small_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        path_s = str(path)
        if path_s not in small_mask_cache:
            small_mask_cache[path_s] = qa.load_mask(path, small_size)
        return small_mask_cache[path_s]

    grid_rows: list[dict[str, object]] = []
    threshold_fracs = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    overlap_values = [0.35, 0.55, 0.75, 0.90]
    max_piece_values = [2, 3, 5, 8, 12, 24, 48]

    for tissue_class in CLASS_KEYS:
        class_df = pair[pair["target_class"] == tissue_class].copy()
        for formula in FORMULAS:
            for threshold_fraction in threshold_fracs:
                for max_overlap in overlap_values:
                    for max_pieces in max_piece_values:
                        selected, union, threshold = select_candidates(
                            class_df,
                            formula,
                            threshold_fraction,
                            max_overlap,
                            max_pieces,
                            get_small_mask,
                            (small_size[1], small_size[0]),
                        )
                        d, p, r = qa.metrics(union, small_annotations[tissue_class])
                        selected_true = sum(1 for row in selected if str(row["true_class"]) == tissue_class)
                        selected_false = len(selected) - selected_true
                        grid_rows.append(
                            {
                                "class_key": tissue_class,
                                "class": display_class(tissue_class),
                                "formula": formula,
                                "threshold_fraction": threshold_fraction,
                                "threshold": round(float(threshold), 6),
                                "max_overlap": max_overlap,
                                "max_pieces": max_pieces,
                                "selected_piece_count": len(selected),
                                "selected_true_hidden": selected_true,
                                "selected_false_positive_hidden": selected_false,
                                "screening_dice": round(float(d), 4),
                                "screening_precision": round(float(p), 4),
                                "screening_recall": round(float(r), 4),
                            }
                        )
    write_csv(OUT / "class_specific_assembly_grid_all.csv", grid_rows)

    full_eval_rows: list[dict[str, object]] = []
    # Full-resolution evaluation is reserved for the best low-resolution
    # policies per class.  This keeps the grid exhaustive without rereading
    # full-size masks for every possible assembly rule.
    for tissue_class in CLASS_KEYS:
        cls_rows = [row for row in grid_rows if row["class_key"] == tissue_class]
        safe = [row for row in cls_rows if float(row["screening_precision"]) >= MIN_PRECISION[tissue_class]]
        candidates = safe if safe else cls_rows
        top = sorted(
            candidates,
            key=lambda row: (float(row["screening_dice"]), float(row["screening_recall"]), float(row["screening_precision"])),
            reverse=True,
        )[:80]
        # Also keep the top pure-Dice rows, because precision floors can be too
        # harsh for diffuse classes in a one-ROI diagnostic.
        top += sorted(
            cls_rows,
            key=lambda row: (float(row["screening_dice"]), float(row["screening_recall"]), float(row["screening_precision"])),
            reverse=True,
        )[:25]
        seen: set[tuple[str, str, float, float, int]] = set()
        for row in top:
            key = (
                str(row["class_key"]),
                str(row["formula"]),
                float(row["threshold_fraction"]),
                float(row["max_overlap"]),
                int(row["max_pieces"]),
            )
            if key in seen:
                continue
            seen.add(key)
            class_df = pair[pair["target_class"] == tissue_class].copy()
            selected, union, threshold = select_candidates(
                class_df,
                str(row["formula"]),
                float(row["threshold_fraction"]),
                float(row["max_overlap"]),
                int(row["max_pieces"]),
                get_mask,
                (size[1], size[0]),
            )
            d, p, r = qa.metrics(union, annotations[tissue_class])
            selected_true = sum(1 for sel in selected if str(sel["true_class"]) == tissue_class)
            selected_false = len(selected) - selected_true
            full_eval_rows.append(
                {
                    **row,
                    "threshold": round(float(threshold), 6),
                    "selected_piece_count": len(selected),
                    "selected_true_hidden": selected_true,
                    "selected_false_positive_hidden": selected_false,
                    "dice": round(float(d), 4),
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                }
            )
    write_csv(OUT / "class_specific_assembly_grid_fullres_evaluated.csv", full_eval_rows)

    best_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        best = choose_best(full_eval_rows, tissue_class)
        key = (
            tissue_class,
            str(best["formula"]),
            float(best["threshold_fraction"]),
            float(best["max_overlap"]),
            int(best["max_pieces"]),
        )
        class_df = pair[pair["target_class"] == tissue_class].copy()
        selected, union, threshold = select_candidates(
            class_df,
            str(best["formula"]),
            float(best["threshold_fraction"]),
            float(best["max_overlap"]),
            int(best["max_pieces"]),
            get_mask,
            (size[1], size[0]),
        )
        fig = OUT / "figures" / f"{tissue_class}_class_specific_assembly_grid_best.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            "Class-specific assembly grid best",
            f"Formula {best['formula']}; threshold fraction {best['threshold_fraction']}; overlap <= {best['max_overlap']}; max pieces {best['max_pieces']}.",
        )
        runtime_row = runtime_by[tissue_class]
        best_row = {
            "class_key": tissue_class,
            "class": display_class(tissue_class),
            "runtime D/P/R": f"{float(runtime_row['dice']):.3f}/{float(runtime_row['precision']):.3f}/{float(runtime_row['recall']):.3f}",
            "grid-best D/P/R": f"{float(best['dice']):.3f}/{float(best['precision']):.3f}/{float(best['recall']):.3f}",
            "selected true/false": f"{best['selected_true_hidden']}/{best['selected_false_positive_hidden']}",
            "formula": best["formula"],
            "threshold_fraction": best["threshold_fraction"],
            "max_overlap": best["max_overlap"],
            "max_pieces": best["max_pieces"],
            "verdict": verdict(float(runtime_row["dice"]), best, tissue_class),
            "figure_rel": str(fig.relative_to(BASE)),
        }
        best_rows.append(best_row)
        for rank, row in enumerate(selected, 1):
            mask = get_mask(str(row["row_key"]))
            sd, sp, sr = qa.metrics(mask, annotations[tissue_class])
            selected_rows.append(
                {
                    "class": display_class(tissue_class),
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "true_class_hidden": display_class(str(row["true_class"])),
                    "formula": best["formula"],
                    "score": round(float(row[str(best["formula"])]), 4),
                    "single_piece_target_dice": round(float(sd), 3),
                    "single_piece_target_precision": round(float(sp), 3),
                    "single_piece_target_recall": round(float(sr), 3),
                    "he_crop_rel": public_by_key[str(row["row_key"])]["he_crop_rel"],
                    "ficture_crop_rel": public_by_key[str(row["row_key"])]["ficture_crop_rel"],
                }
            )
    write_csv(OUT / "class_specific_assembly_grid_best_by_class.csv", best_rows)
    write_csv(OUT / "class_specific_assembly_grid_selected_pieces.csv", selected_rows)

    section = f"""
<h2>17Z. Class-Specific Assembly Grid: Can The Existing Scores Be Assembled Better?</h2>
<p><b>Goal.</b> After 17X and 17Y showed that cross-class competition is not the main fix, this section tests whether each tissue class simply needs its own assembly rule. The grid changes formula, score threshold, overlap filtering, and max-piece count while keeping the same candidate pool and the same deployable score columns.</p>
<p><b>What the grid is allowed to use.</b> During candidate selection, the grid uses only runtime score columns such as H&amp;E score, FICTURE prior, shape score, and static quality gates. It does not use hidden Dice, Precision, Recall, or annotation labels. Annotation is used only after selection to evaluate each grid cell and to diagnose whether assembly is the real bottleneck.</p>
<p><b>How to read this table.</b> Runtime D/P/R is the current deployable policy. Grid-best D/P/R is the best current-ROI diagnostic setting under a class-specific precision floor. If grid-best improves strongly, assembly policy is worth refining. If it does not, the failure is earlier: proposal generation, quality scoring, or feature representation.</p>
{table(best_rows, ['class', 'runtime D/P/R', 'grid-best D/P/R', 'selected true/false', 'formula', 'threshold_fraction', 'max_overlap', 'max_pieces', 'verdict'])}
<h3>Grid-best selected-union figures</h3>
<div class='figure-grid'>
{''.join(f"<figure><img src='{html.escape(str(Path(r['figure_rel']).as_posix()))}' alt='{html.escape(str(r['class']))} class-specific grid best'><figcaption>{html.escape(str(r['class']))}: {html.escape(str(r['grid-best D/P/R']))}</figcaption></figure>" for r in best_rows)}
</div>
<div class='callout'><b>17Z decision.</b> If this section cannot improve a class using the current score columns, the next step should not be more threshold tuning. It should be the earliest failing layer from 17V: proposal generator for alveoli, deployable quality proxy/pathology encoder for vessels/tumor/immune, and second-SAM/localizer refinement for bronchiola/vessels.</div>
"""

    marker = "<h2>17Z. Class-Specific Assembly Grid: Can The Existing Scores Be Assembled Better?</h2>"
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if marker in text:
            start = text.index(marker)
            end = text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    print(OUT / "class_specific_assembly_grid_best_by_class.csv")
    print(OUT / "class_specific_assembly_grid_all.csv")
    print(OUT / "class_specific_assembly_grid_selected_pieces.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
