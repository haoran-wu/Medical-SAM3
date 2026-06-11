#!/usr/bin/env python3
"""Feature-space upper-bound search for Jun09 skill ranker.

This asks a narrow but important question: if we restrict ourselves to features
available at deployment time, is there enough signal to rank candidate pieces
well? Annotation is used only after scoring to choose a diagnostic best policy
and measure the result. Scores do not use compact_funnel_score, component Dice,
Precision, Recall, or hidden class labels.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "feature_space_upper_bound"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"

spec = importlib.util.spec_from_file_location("qa_helpers", QA_SCRIPT)
qa = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qa)

CLASS_KEYS = qa.CLASS_KEYS
DISPLAY = {"immune_infiltration": "immune infiltration"}


def display_class(name: str) -> str:
    return DISPLAY.get(name, name)


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


def norm01(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def area_plausibility(area_frac: pd.Series, tissue_class: str) -> pd.Series:
    log_area = np.log10(np.maximum(area_frac.astype(float), 1e-7))
    if tissue_class in {"bronchiola", "vessels", "immune_infiltration"}:
        center, width = -3.0, 1.0
    elif tissue_class == "alveoli":
        center, width = -2.0, 0.9
    else:
        center, width = -2.35, 1.1
    return pd.Series(np.clip(np.exp(-((log_area - center) / width) ** 2), 0, 1), index=area_frac.index)


def class_composition(row: pd.DataFrame, tissue_class: str, ring: bool = False) -> pd.Series:
    prefix = "ring_frac" if ring else "frac"
    mapping = {
        "bronchiola": "airway_epithelial",
        "alveoli": "at2",
        "vessels": "endothelial",
        "tumor": "epithelial_tumor",
        "stroma": "stroma",
        "immune_infiltration": "immune",
    }
    return row[f"{prefix}_{mapping[tissue_class]}"].astype(float).fillna(0)


def margin(df: pd.DataFrame, prefix: str, tissue_class: str) -> pd.Series:
    target = df[f"{prefix}_{tissue_class}"].astype(float).fillna(0) / 100.0
    other_cols = [f"{prefix}_{c}" for c in CLASS_KEYS if c != tissue_class]
    others = df[other_cols].astype(float).fillna(0).max(axis=1) / 100.0
    return pd.Series(np.clip((target - others + 1.0) / 2.0, 0, 1), index=df.index)


def build_pair_features(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for tissue_class in CLASS_KEYS:
        df = features.copy()
        df["target_class"] = tissue_class
        df["he"] = df[f"he_clip_large_{tissue_class}"].astype(float).fillna(0) / 100.0
        df["fig"] = df[f"ficture_prior_{tissue_class}"].astype(float).fillna(0) / 100.0
        df["shape"] = df[f"shape_skill_{tissue_class}"].astype(float).fillna(0) / 100.0
        df["comp"] = class_composition(df, tissue_class, ring=False)
        df["ring_comp"] = class_composition(df, tissue_class, ring=True)
        df["he_margin"] = margin(df, "he_clip_large", tissue_class)
        df["fig_margin"] = margin(df, "ficture_prior", tissue_class)
        df["shape_margin"] = margin(df, "shape_skill", tissue_class)
        df["area_ok"] = area_plausibility(df["area_frac"], tissue_class)
        df["solidity_ok"] = np.clip(df["solidity"].astype(float).fillna(0), 0, 1)
        df["thinness_ok"] = np.clip(1.0 - np.abs(df["thinness"].astype(float).fillna(0) - 0.045) / 0.08, 0, 1)
        rows.append(
            df[
                [
                    "row_key",
                    "candidate_uid",
                    "true_class",
                    "target_class",
                    "component_dice",
                    "component_precision",
                    "component_recall",
                    "he",
                    "fig",
                    "shape",
                    "comp",
                    "ring_comp",
                    "he_margin",
                    "fig_margin",
                    "shape_margin",
                    "area_ok",
                    "solidity_ok",
                    "thinness_ok",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def add_score_formulas(pair_df: pd.DataFrame) -> dict[str, str]:
    formulas = {
        "he_only": "he",
        "fig_only": "fig",
        "shape_only": "shape",
        "comp_only": "comp",
        "he_shape": "0.65*he + 0.35*shape",
        "fig_shape": "0.65*fig + 0.35*shape",
        "he_fig_shape": "0.45*he + 0.30*fig + 0.25*shape",
        "he_shape_quality": "0.55*he + 0.20*shape + 0.15*area_ok + 0.10*solidity_ok",
        "fig_comp_quality": "0.45*fig + 0.25*comp + 0.15*ring_comp + 0.10*area_ok + 0.05*solidity_ok",
        "balanced_static": "0.30*he + 0.20*fig + 0.20*shape + 0.15*comp + 0.10*area_ok + 0.05*solidity_ok",
        "margin_balanced": "0.25*he + 0.20*fig + 0.15*shape + 0.15*he_margin + 0.15*fig_margin + 0.10*shape_margin",
        "quality_gate_static": "(0.55*he + 0.25*fig + 0.20*shape) * (0.35 + 0.65*(0.35*area_ok + 0.25*solidity_ok + 0.20*shape_margin + 0.20*thinness_ok))",
    }
    namespace = {col: pair_df[col] for col in ["he", "fig", "shape", "comp", "ring_comp", "he_margin", "fig_margin", "shape_margin", "area_ok", "solidity_ok", "thinness_ok"]}
    for name, expr in formulas.items():
        pair_df[f"score_{name}"] = np.clip(eval(expr, {"__builtins__": {}}, namespace), 0, 1)
    return formulas


def select_with_policy(
    class_df: pd.DataFrame,
    score_col: str,
    get_mask,
    canvas_shape: tuple[int, int],
    min_score: float,
    max_overlap: float,
    max_pieces: int,
) -> tuple[list[pd.Series], np.ndarray]:
    union = np.zeros(canvas_shape, dtype=bool)
    selected: list[pd.Series] = []
    for _, row in class_df.sort_values(score_col, ascending=False).iterrows():
        if float(row[score_col]) < min_score:
            break
        mask = get_mask(row["row_key"])
        if qa.overlap_fraction(mask, union) > max_overlap:
            continue
        selected.append(row)
        union |= mask
        if len(selected) >= max_pieces:
            break
    return selected, union


def objective_for_class(tissue_class: str, d: float, p: float, r: float) -> float:
    if tissue_class in {"bronchiola", "vessels"}:
        return d + 0.15 * p + 0.10 * r
    if tissue_class == "immune_infiltration":
        return d + 0.10 * p + 0.20 * r
    if tissue_class == "alveoli":
        return d + 0.12 * p + 0.12 * r
    return d + 0.20 * p + 0.06 * r


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
    features = pd.read_csv(BASE / "candidate_skill_features.csv")
    pair_df = build_pair_features(features)
    formulas = add_score_formulas(pair_df)
    pair_df.to_csv(OUT / "feature_space_pair_scores.csv", index=False)

    corr_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        sub = pair_df[pair_df["target_class"] == tissue_class]
        for feature in [
            "he",
            "fig",
            "shape",
            "comp",
            "ring_comp",
            "he_margin",
            "fig_margin",
            "shape_margin",
            "area_ok",
            "solidity_ok",
            "thinness_ok",
        ]:
            corr_rows.append(
                {
                    "class": display_class(tissue_class),
                    "feature": feature,
                    "Spearman(feature,Dice)": f"{sub[feature].corr(sub['component_dice'], method='spearman'):.3f}",
                    "Spearman(feature,Precision)": f"{sub[feature].corr(sub['component_precision'], method='spearman'):.3f}",
                    "Spearman(feature,Recall)": f"{sub[feature].corr(sub['component_recall'], method='spearman'):.3f}",
                }
            )
    write_csv(OUT / "deployable_feature_correlations.csv", corr_rows)

    full_mask_cache: dict[str, np.ndarray] = {}
    small_mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        if str(path) not in full_mask_cache:
            full_mask_cache[str(path)] = qa.load_mask(path, size)
        return full_mask_cache[str(path)]

    def get_small_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        if str(path) not in small_mask_cache:
            small_mask_cache[str(path)] = qa.load_mask(path, small_size)
        return small_mask_cache[str(path)]

    score_cols = [f"score_{name}" for name in formulas]
    min_scores = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
    overlaps = [0.35, 0.45, 0.60, 0.75, 0.90]
    piece_options = {
        "bronchiola": [1, 2, 3, 4, 6, 8],
        "alveoli": [1, 2, 3, 4, 6],
        "vessels": [2, 4, 6, 8, 12, 18],
        "tumor": [4, 8, 12, 24, 36, 48],
        "stroma": [4, 8, 12, 24, 36, 48],
        "immune_infiltration": [8, 12, 24, 36, 48, 64],
    }

    grid_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for tissue_class in CLASS_KEYS:
        class_df = pair_df[pair_df["target_class"] == tissue_class]
        best: tuple[float, str, float, float, int, float, float, float, list[pd.Series], np.ndarray] | None = None
        for score_col, min_score, max_overlap, max_pieces in itertools.product(
            score_cols,
            min_scores,
            overlaps,
            piece_options[tissue_class],
        ):
            selected, union = select_with_policy(
                class_df,
                score_col,
                get_small_mask,
                (small_size[1], small_size[0]),
                min_score,
                max_overlap,
                max_pieces,
            )
            d, p, r = qa.metrics(union, small_annotations[tissue_class])
            obj = objective_for_class(tissue_class, d, p, r)
            grid_rows.append(
                {
                    "class": display_class(tissue_class),
                    "formula": score_col.replace("score_", ""),
                    "min_score": min_score,
                    "max_overlap": max_overlap,
                    "max_pieces": max_pieces,
                    "screening_selected": len(selected),
                    "screening_dice": f"{d:.4f}",
                    "screening_precision": f"{p:.4f}",
                    "screening_recall": f"{r:.4f}",
                    "objective": f"{obj:.4f}",
                }
            )
            if best is None or obj > best[0]:
                best = (obj, score_col, min_score, max_overlap, max_pieces, d, p, r, selected, union)
        assert best is not None
        _, score_col, min_score, max_overlap, max_pieces, _, _, _, _, _ = best
        selected, union = select_with_policy(
            class_df,
            score_col,
            get_mask,
            (size[1], size[0]),
            min_score,
            max_overlap,
            max_pieces,
        )
        d, p, r = qa.metrics(union, annotations[tissue_class])
        formula = score_col.replace("score_", "")
        fig = OUT / "figures" / f"{tissue_class}_feature_space_upper_bound_union.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            f"{tissue_class}: feature-space upper-bound assembly",
            f"Formula={formula}; policy score>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}. This is diagnostic: formula/policy selected using this ROI annotation.",
        )
        if d >= 0.65:
            verdict = "feature space is likely sufficient; focus on deployable policy learning/validation"
        elif tissue_class == "alveoli":
            verdict = "feature space still cannot solve alveoli inside this pool; proposal generator must change"
        elif d >= 0.45:
            verdict = "partial signal; add better quality proxy, pathology encoder, or second-SAM refinement"
        else:
            verdict = "feature space weak; needs new proposal/encoder/context branch"
        best_rows.append(
            {
                "class": display_class(tissue_class),
                "best_formula": formula,
                "selected_piece_count": len(selected),
                "dice": f"{d:.3f}",
                "precision": f"{p:.3f}",
                "recall": f"{r:.3f}",
                "policy": f"score>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}",
                "verdict": verdict,
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        for rank, row in enumerate(selected[:120], 1):
            md, mp, mr = qa.metrics(get_mask(row["row_key"]), annotations[tissue_class])
            selected_rows.append(
                {
                    "class": display_class(tissue_class),
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "formula": formula,
                    "score": f"{float(row[score_col]):.4f}",
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

    write_csv(OUT / "feature_space_policy_grid.csv", grid_rows)
    write_csv(OUT / "feature_space_upper_bound_summary.csv", best_rows)
    write_csv(OUT / "feature_space_selected_pieces.csv", selected_rows)

    previous = []
    for path, label in [
        (BASE / "failure_driven_proxy_refinement/class_specific_refinement_summary.csv", "17L runtime proxy"),
        (BASE / "failure_gate_matrix/quantitative_failure_gate_matrix.csv", "17P failure gate"),
    ]:
        if path.exists():
            previous.append({"source": label, "file": str(path.relative_to(BASE))})
    write_csv(OUT / "comparison_sources.csv", previous)

    section = f"""
<h2>17R. Deployable Feature-Space Upper Bound: Is There Enough Signal Without Hidden Quality?</h2>
<p>This section tests whether the current deployable feature space is strong enough. The scores use only features available at runtime: H&amp;E CLIP-style morphology score, structured FICTURE composition prior, shape/location, area, solidity, and simple margins. The score does <b>not</b> use <code>compact_funnel_score</code>, component Dice, Precision, Recall, or hidden labels.</p>
<p>Annotation is used only after scoring to choose the diagnostic best formula/policy and measure Dice / Precision / Recall. Therefore this is an <b>upper-bound diagnosis</b>, not a final deployable method. If this upper bound is weak, the next change must be a new proposal, pathology encoder, context cue, or second-SAM refinement rather than another hand-tuned formula.</p>
<h3>Best feature-space upper-bound result by class</h3>
{table(best_rows, ['class', 'best_formula', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy', 'verdict'])}
<h3>Six-panel visual check</h3>
<div class='image-grid'>
{''.join(f"<figure><img src='{html.escape(row['figure_rel'])}'><figcaption>{html.escape(row['class'])}: {html.escape(row['dice'])}/{html.escape(row['precision'])}/{html.escape(row['recall'])}</figcaption></figure>" for row in best_rows)}
</div>
<h3>Feature correlation microscope</h3>
{table(corr_rows, ['class', 'feature', 'Spearman(feature,Dice)', 'Spearman(feature,Precision)', 'Spearman(feature,Recall)'])}
<h3>Selected pieces from the upper-bound policy</h3>
{table(selected_rows[:220], ['class', 'rank', 'candidate_uid', 'formula', 'score', 'true_class_hidden', 'target_component_dice', 'fullres_target_dice', 'fullres_target_precision', 'fullres_target_recall', 'he', 'fig', 'shape', 'comp', 'area_ok'])}
<div class='callout'><b>How to read this.</b> A high result here means the current features contain enough information but need a deployable calibrated policy. A low result means the representation is not enough; the next iteration should move to the 17Q branches: minimal-three proposal gate, pathology/mask-aware H&amp;E encoder, or second-SAM locator refinement.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17R. Deployable Feature-Space Upper Bound: Is There Enough Signal Without Hidden Quality?</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    state = {
        "status": "diagnostic_complete_not_final",
        "score_uses_hidden_quality": False,
        "policy_selected_with_annotation": True,
        "summary": best_rows,
        "next_rule": "If feature-space upper bound is below the class target, move to proposal/encoder/refinement branch instead of tuning formula.",
    }
    (OUT / "feature_space_upper_bound_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
    rebuild_zip()
    print(OUT / "feature_space_upper_bound_summary.csv")


if __name__ == "__main__":
    main()
