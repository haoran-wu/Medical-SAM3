#!/usr/bin/env python3
"""Structured quality-v2 diagnostic for the Jun09 skill ranker.

This is an iteration controller experiment, not a final deployable claim.
It tests whether deployment-visible features can make a better mask-quality
skill: target composition inside the mask, contrast against the surrounding
ring, H&E/shape margins, and simple geometry.  Annotation is used only after
selection to choose the diagnostic best policy and decide whether this path is
worth continuing.
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
OUT = BASE / "structured_quality_v2"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qa = import_module(QA_SCRIPT, "qa_helpers_structured_quality_v2")
CLASS_KEYS = qa.CLASS_KEYS

FORMULAS = [
    "score_inside_contrast",
    "score_context_enrichment",
    "score_quality_v2",
    "score_membership_x_quality_v2",
    "score_ficture_context_quality",
    "score_he_context_quality",
    "score_margin_context_quality",
    "score_diffuse_safe",
]

MIN_PRECISION = {
    "bronchiola": 0.85,
    "alveoli": 0.35,
    "vessels": 0.85,
    "tumor": 0.55,
    "stroma": 0.35,
    "immune_infiltration": 0.40,
}


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


def display_class(name: str) -> str:
    return "immune infiltration" if name == "immune_infiltration" else name


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def add_structured_quality_scores(pair: pd.DataFrame) -> pd.DataFrame:
    data = pair.copy()
    comp = data["comp"].astype(float).clip(0, 1)
    ring = data["ring_comp"].astype(float).clip(0, 1)
    comp_delta = (comp - ring).clip(lower=0, upper=1)
    comp_enrichment = (comp / (ring + 0.02)).clip(0, 8) / 8.0
    margin_mean = (
        data["he_margin"].astype(float).clip(0, 1)
        + data["fig_margin"].astype(float).clip(0, 1)
        + data["shape_margin"].astype(float).clip(0, 1)
    ) / 3.0
    quality_v2 = (
        0.20 * data["area_ok"].astype(float).clip(0, 1)
        + 0.15 * data["solidity_ok"].astype(float).clip(0, 1)
        + 0.15 * data["thinness_ok"].astype(float).clip(0, 1)
        + 0.15 * data["shape_margin"].astype(float).clip(0, 1)
        + 0.15 * data["he_margin"].astype(float).clip(0, 1)
        + 0.10 * data["fig_margin"].astype(float).clip(0, 1)
        + 0.10 * comp_delta
    ).clip(0, 1)
    membership = (
        0.40 * data["he"].astype(float).clip(0, 1)
        + 0.25 * data["fig"].astype(float).clip(0, 1)
        + 0.20 * data["shape"].astype(float).clip(0, 1)
        + 0.15 * comp
    ).clip(0, 1)
    data["comp_delta"] = comp_delta
    data["comp_enrichment"] = comp_enrichment
    data["quality_v2"] = quality_v2
    data["score_inside_contrast"] = (
        0.45 * comp + 0.25 * comp_delta + 0.15 * data["fig"].astype(float) + 0.15 * data["shape"].astype(float)
    ).clip(0, 1)
    data["score_context_enrichment"] = (
        0.35 * data["fig"].astype(float)
        + 0.25 * comp_delta
        + 0.15 * comp_enrichment
        + 0.15 * data["shape_margin"].astype(float)
        + 0.10 * data["area_ok"].astype(float)
    ).clip(0, 1)
    data["score_quality_v2"] = quality_v2
    data["score_membership_x_quality_v2"] = (membership * (0.30 + 0.70 * quality_v2)).clip(0, 1)
    data["score_ficture_context_quality"] = (
        (0.55 * data["fig"].astype(float) + 0.25 * comp_delta + 0.20 * data["shape"].astype(float))
        * (0.25 + 0.75 * quality_v2)
    ).clip(0, 1)
    data["score_he_context_quality"] = (
        (0.55 * data["he"].astype(float) + 0.25 * data["shape"].astype(float) + 0.20 * comp_delta)
        * (0.25 + 0.75 * quality_v2)
    ).clip(0, 1)
    data["score_margin_context_quality"] = (
        (0.45 * margin_mean + 0.30 * data["shape"].astype(float) + 0.25 * comp_delta)
        * (0.25 + 0.75 * quality_v2)
    ).clip(0, 1)
    data["score_diffuse_safe"] = (
        0.30 * data["fig"].astype(float)
        + 0.25 * data["he"].astype(float)
        + 0.20 * data["shape"].astype(float)
        + 0.15 * comp
        + 0.10 * quality_v2
    ).clip(0, 1)
    return data


def select_candidates(
    class_df: pd.DataFrame,
    formula: str,
    threshold_fraction: float,
    max_overlap: float,
    max_pieces: int,
    get_mask,
    canvas_shape: tuple[int, int],
) -> tuple[list[pd.Series], np.ndarray, float]:
    scores = class_df[formula].astype(float)
    threshold = float(scores.max()) * float(threshold_fraction) if len(scores) else 1.0
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


def choose_best(rows: list[dict[str, object]], tissue_class: str) -> dict[str, object]:
    cls_rows = [row for row in rows if row["class_key"] == tissue_class]
    safe = [row for row in cls_rows if float(row["precision"]) >= MIN_PRECISION[tissue_class]]
    pool = safe if safe else cls_rows
    return max(pool, key=lambda row: (float(row["dice"]), float(row["recall"]), float(row["precision"])))


def dpr_to_tuple(value: str) -> tuple[float, float, float]:
    a, b, c = str(value).split("/")
    return float(a), float(b), float(c)


def verdict(cls: str, runtime_d: float, best_d: float, best_p: float, best_r: float) -> str:
    delta = best_d - runtime_d
    if cls == "alveoli":
        return "proposal bottleneck remains; quality-v2 cannot create the missing broad alveoli proposal"
    if delta >= 0.05:
        return "promising quality-v2 improvement; validate on broader proposal pool or second ROI before promoting"
    if delta >= 0.015:
        return "small quality-v2 improvement; useful as an auxiliary feature, not enough as the main fix"
    if best_p >= MIN_PRECISION[cls] and best_r < 0.45:
        return "quality-v2 is precision-biased and loses recall; combine with second-SAM or recall branch"
    return "quality-v2 does not materially improve this class; stop hand-crafted proxy search here"


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
        BASE / "class_specific_assembly_grid",
        BASE / "iteration_state_after_grid",
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
    small_size = (512, int(round(512 * size[1] / size[0])))
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    small_annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], small_size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(qa.row_key, axis=1)
    public_by_key = public.set_index("row_key").to_dict("index")
    pair = add_structured_quality_scores(pd.read_csv(BASE / "feature_space_upper_bound/feature_space_pair_scores.csv"))
    pair.to_csv(OUT / "structured_quality_v2_pair_scores.csv", index=False)
    runtime = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    runtime_by = {str(r["class"]).replace("immune infiltration", "immune_infiltration"): r for _, r in runtime.iterrows()}

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
    threshold_fracs = [0.20, 0.35, 0.50, 0.65, 0.80]
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
                                "selected_false_positive_hidden": len(selected) - selected_true,
                                "screening_dice": round(float(d), 4),
                                "screening_precision": round(float(p), 4),
                                "screening_recall": round(float(r), 4),
                            }
                        )
    write_csv(OUT / "structured_quality_v2_screening_grid.csv", grid_rows)

    full_eval_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        cls_rows = [row for row in grid_rows if row["class_key"] == tissue_class]
        safe = [row for row in cls_rows if float(row["screening_precision"]) >= MIN_PRECISION[tissue_class]]
        pool = safe if safe else cls_rows
        top = sorted(
            pool,
            key=lambda row: (float(row["screening_dice"]), float(row["screening_recall"]), float(row["screening_precision"])),
            reverse=True,
        )[:16]
        class_df = pair[pair["target_class"] == tissue_class].copy()
        for candidate in top:
            selected, union, threshold = select_candidates(
                class_df,
                str(candidate["formula"]),
                float(candidate["threshold_fraction"]),
                float(candidate["max_overlap"]),
                int(candidate["max_pieces"]),
                get_mask,
                (size[1], size[0]),
            )
            d, p, r = qa.metrics(union, annotations[tissue_class])
            selected_true = sum(1 for row in selected if str(row["true_class"]) == tissue_class)
            row = {
                **candidate,
                "threshold": round(float(threshold), 6),
                "dice": round(float(d), 4),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "selected_piece_count": len(selected),
                "selected_true_hidden": selected_true,
                "selected_false_positive_hidden": len(selected) - selected_true,
            }
            full_eval_rows.append(row)

        best = choose_best(full_eval_rows, tissue_class)
        selected, union, threshold = select_candidates(
            class_df,
            str(best["formula"]),
            float(best["threshold_fraction"]),
            float(best["max_overlap"]),
            int(best["max_pieces"]),
            get_mask,
            (size[1], size[0]),
        )
        d, p, r = qa.metrics(union, annotations[tissue_class])
        fig = OUT / "figures" / f"{tissue_class}_structured_quality_v2_best.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            f"{display_class(tissue_class)}: structured quality-v2 diagnostic",
            f"Formula={best['formula']}; threshold={threshold:.3f}; overlap<={best['max_overlap']}; max_pieces={best['max_pieces']}",
        )
        rt = runtime_by[tissue_class]
        runtime_d = float(rt["dice"])
        runtime_p = float(rt["precision"])
        runtime_r = float(rt["recall"])
        best_row = {
            "class_key": tissue_class,
            "class": display_class(tissue_class),
            "runtime D/P/R": f"{runtime_d:.3f}/{runtime_p:.3f}/{runtime_r:.3f}",
            "structured quality-v2 D/P/R": f"{d:.3f}/{p:.3f}/{r:.3f}",
            "delta Dice vs runtime": f"{d - runtime_d:+.3f}",
            "formula": best["formula"],
            "threshold_fraction": best["threshold_fraction"],
            "max_overlap": best["max_overlap"],
            "max_pieces": best["max_pieces"],
            "selected true/false hidden": f"{sum(1 for row in selected if str(row['true_class']) == tissue_class)}/{sum(1 for row in selected if str(row['true_class']) != tissue_class)}",
            "verdict": verdict(tissue_class, runtime_d, d, p, r),
            "figure_rel": str(fig.relative_to(BASE)),
        }
        best_rows.append(best_row)
        for rank, row in enumerate(selected[:80], 1):
            fd, fp, fr = qa.metrics(get_mask(str(row["row_key"])), annotations[tissue_class])
            selected_rows.append(
                {
                    "class": display_class(tissue_class),
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "formula": best["formula"],
                    "score": f"{float(row[str(best['formula'])]):.4f}",
                    "he": f"{float(row['he']):.3f}",
                    "fig": f"{float(row['fig']):.3f}",
                    "shape": f"{float(row['shape']):.3f}",
                    "comp": f"{float(row['comp']):.3f}",
                    "ring_comp": f"{float(row['ring_comp']):.3f}",
                    "comp_delta": f"{float(row['comp_delta']):.3f}",
                    "quality_v2": f"{float(row['quality_v2']):.3f}",
                    "true_class_hidden": row["true_class"],
                    "target_piece D/P/R": f"{fd:.3f}/{fp:.3f}/{fr:.3f}",
                }
            )

    write_csv(OUT / "structured_quality_v2_fullres_evaluated.csv", full_eval_rows)
    write_csv(OUT / "structured_quality_v2_best_by_class.csv", best_rows)
    write_csv(OUT / "structured_quality_v2_selected_pieces.csv", selected_rows)

    stop_continue_rows = []
    for row in best_rows:
        cls = str(row["class_key"])
        delta = float(row["delta Dice vs runtime"])
        if cls == "alveoli":
            decision = "stop same-pool quality tuning; proposal generator is mandatory"
        elif delta >= 0.05:
            decision = "continue this feature family but validate outside the diagnostic grid"
        elif delta >= 0.015:
            decision = "keep as auxiliary feature only"
        else:
            decision = "stop hand-crafted quality proxy search for this class"
        stop_continue_rows.append(
            {
                "class": row["class"],
                "decision": decision,
                "reason": row["verdict"],
                "next change if continuing": "pathology encoder / second-SAM / broader proposal, not another global prompt",
            }
        )
    write_csv(OUT / "structured_quality_v2_stop_continue_decision.csv", stop_continue_rows)

    section = f"""
<h2>17AB. Structured Quality-V2: More Precise Failure Test After 17Z</h2>
<p><b>Goal.</b> This is a paper-guided failure test for the current skill ranker. Region-aware papers warn that a small region needs explicit region/context signals, and SaLIP-style cascades suggest that the ranker should often choose prompts rather than directly output the final mask. Here I test whether a more explicit <b>mask-quality skill</b> can improve selection before moving to second-SAM or a new encoder.</p>
<p><b>Inputs used for selection.</b> Only deployment-visible signals are used: H&amp;E score, structured FICTURE prior, shape score, target color/cell-type fraction inside the mask, the same fraction in the surrounding ring, inside-vs-ring contrast, area plausibility, solidity, and thinness. Annotation is hidden from selection and used only after selection for Dice / Precision / Recall.</p>
<p><b>What changed from 17Z.</b> 17Z only varied formulas and thresholds over existing score columns. This section adds a more precise structured quality feature: a candidate is trusted more when the target FICTURE/cell-type signal is enriched inside the mask compared with its nearby ring, and when its geometry looks plausible for that tissue.</p>
<h3>Best diagnostic policy per class</h3>
{table(best_rows, ['class', 'runtime D/P/R', 'structured quality-v2 D/P/R', 'delta Dice vs runtime', 'formula', 'threshold_fraction', 'max_overlap', 'max_pieces', 'selected true/false hidden', 'verdict'])}
<h3>Selected pieces under the best policy</h3>
{table(selected_rows[:220], ['class', 'rank', 'candidate_uid', 'formula', 'score', 'he', 'fig', 'shape', 'comp', 'ring_comp', 'comp_delta', 'quality_v2', 'true_class_hidden', 'target_piece D/P/R'])}
<h3>Stop / continue decision</h3>
{table(stop_continue_rows, ['class', 'decision', 'reason', 'next change if continuing'])}
<div class='grid'>
{''.join(f"<figure><img src='{html.escape(row['figure_rel'])}'><figcaption>{html.escape(row['class'])}: structured quality-v2 best</figcaption></figure>" for row in best_rows)}
</div>
<div class='callout'><b>17AB decision.</b> This test is deliberately stricter than another prompt trial. If structured quality-v2 does not materially improve a class, the next move is not more hand-crafted threshold search. It is either a proposal-generator change, a pathology/mask-aware encoder, or second-SAM refinement depending on the class-specific failing layer.</div>
"""
    marker = "<h2>17AB. Structured Quality-V2: More Precise Failure Test After 17Z</h2>"
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
    print(OUT / "structured_quality_v2_best_by_class.csv")
    print(OUT / "structured_quality_v2_stop_continue_decision.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
