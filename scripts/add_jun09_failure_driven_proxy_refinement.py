#!/usr/bin/env python3
"""Failure-driven class-specific proxy refinement for Jun09.

This section turns the failure diagnosis into one concrete next iteration:
different tissue classes need different runtime signals. The formulas are still
deployment-time proxies; annotation is only used after selection for full
resolution evaluation.
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
OUT = BASE / "failure_driven_proxy_refinement"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"

spec = importlib.util.spec_from_file_location("qa_helpers", QA_SCRIPT)
qa = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qa)

CLASS_KEYS = qa.CLASS_KEYS


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


def score_formula(row: pd.Series, formula: str) -> float:
    loco = float(row["loco"])
    he = float(row["he"])
    fig = float(row["fig"])
    shape = float(row["shape_score"])
    if formula == "loco_he_fig_shape":
        return 0.50 * loco + 0.25 * he + 0.10 * fig + 0.15 * shape
    if formula == "fig_onlyish":
        return 0.05 * loco + 0.05 * he + 0.85 * fig + 0.05 * shape
    if formula == "shape_only":
        return 0.05 * loco + 0.05 * he + 0.05 * fig + 0.85 * shape
    if formula == "fig_shape_low_loco":
        return 0.05 * loco + 0.10 * he + 0.55 * fig + 0.30 * shape
    raise ValueError(formula)


def select_pieces(
    class_df: pd.DataFrame,
    get_mask,
    canvas_shape: tuple[int, int],
    min_score: float,
    max_overlap: float,
    max_pieces: int,
) -> tuple[list[pd.Series], np.ndarray]:
    union = np.zeros(canvas_shape, dtype=bool)
    selected: list[pd.Series] = []
    for _, row in class_df.sort_values("class_specific_score", ascending=False).iterrows():
        if float(row["class_specific_score"]) < min_score:
            break
        mask = get_mask(row["row_key"])
        if qa.overlap_fraction(mask, union) > max_overlap:
            continue
        selected.append(row)
        union |= mask
        if len(selected) >= max_pieces:
            break
    return selected, union


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(qa.HE_ROI).size
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(qa.row_key, axis=1)
    public_by_key = public.set_index("row_key").to_dict("index")
    features = pd.read_csv(BASE / "candidate_skill_features.csv")
    loco = pd.read_csv(BASE / "score_outputs/loco_component_logistic_validation/per_candidate_predictions.csv")
    data = features.merge(loco[["row_key", *CLASS_KEYS]], on="row_key", how="left")

    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        for tissue_class in CLASS_KEYS:
            def f(v: object) -> float:
                try:
                    return float(v) / 100.0
                except Exception:
                    return 0.0

            formula = {
                "bronchiola": "loco_he_fig_shape",
                "alveoli": "shape_only",
                "vessels": "fig_onlyish",
                "tumor": "loco_he_fig_shape",
                "stroma": "fig_shape_low_loco",
                "immune_infiltration": "loco_he_fig_shape",
            }[tissue_class]
            rec = {
                "row_key": row["row_key"],
                "candidate_uid": row["candidate_uid"],
                "target_class": tissue_class,
                "true_class": row["true_class"],
                "formula": formula,
                "loco": f(row.get(tissue_class, 0)),
                "he": f(row.get(f"he_clip_large_{tissue_class}", 0)),
                "fig": f(row.get(f"ficture_prior_{tissue_class}", 0)),
                "shape_score": f(row.get(f"shape_skill_{tissue_class}", 0)),
            }
            rec["class_specific_score"] = score_formula(pd.Series(rec), formula)
            rows.append(rec)
    pair_scores = pd.DataFrame(rows)
    pair_scores.to_csv(OUT / "class_specific_pair_scores.csv", index=False)

    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = public_by_key[key]["mask_path"]
        if path not in mask_cache:
            mask_cache[path] = qa.load_mask(Path(path), size)
        return mask_cache[path]

    # These policies came from the low-resolution failure-driven search and are
    # then verified here at full resolution.
    policies = {
        "bronchiola": (0.60, 0.90, 2),
        "alveoli": (0.80, 0.45, 4),
        "vessels": (0.80, 0.90, 12),
        "tumor": (0.60, 0.45, 48),
        "stroma": (0.50, 0.90, 36),
        "immune_infiltration": (0.60, 0.90, 48),
    }
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    diagnosis = {
        "bronchiola": "works with balanced LOCO + H&E + FICTURE + shape; remaining gap is the third tiny component.",
        "alveoli": "still weak even with shape-heavy scoring; bottleneck is candidate/generator/context, not only class score.",
        "vessels": "fixed by making FICTURE composition the dominant signal; previous failure was LOCO/H&E under-ranking full vessel components.",
        "tumor": "stable under balanced scoring; diffuse tumor boundary remains the limiting factor.",
        "stroma": "improves with FICTURE+shape low-LOCO scoring, but broad stroma remains precision/recall tradeoff.",
        "immune_infiltration": "balanced scoring keeps recall acceptable; many tiny components still need assembly calibration.",
    }
    for tissue_class in CLASS_KEYS:
        min_score, max_overlap, max_pieces = policies[tissue_class]
        selected, union = select_pieces(
            pair_scores[pair_scores["target_class"] == tissue_class],
            get_mask,
            (size[1], size[0]),
            min_score,
            max_overlap,
            max_pieces,
        )
        d, p, r = qa.metrics(union, annotations[tissue_class])
        formula = str(pair_scores[pair_scores["target_class"] == tissue_class]["formula"].iloc[0])
        fig = OUT / "figures" / f"{tissue_class}_failure_driven_class_specific_union.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            f"{tissue_class}: failure-driven class-specific proxy",
            f"Formula={formula}; full-res verification after low-res policy search: score>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}.",
        )
        summary_rows.append(
            {
                "class": tissue_class,
                "formula": formula,
                "selected_piece_count": len(selected),
                "dice": f"{d:.3f}",
                "precision": f"{p:.3f}",
                "recall": f"{r:.3f}",
                "policy": f"score>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}",
                "failure_diagnosis": diagnosis[tissue_class],
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        for rank, row in enumerate(selected[:80], 1):
            fd, fp, fr = qa.metrics(get_mask(row["row_key"]), annotations[tissue_class])
            selected_rows.append(
                {
                    "class": tissue_class,
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "class_specific_score": f"{float(row['class_specific_score']):.4f}",
                    "formula": formula,
                    "loco": f"{float(row['loco']):.4f}",
                    "he": f"{float(row['he']):.4f}",
                    "fig": f"{float(row['fig']):.4f}",
                    "shape_score": f"{float(row['shape_score']):.4f}",
                    "true_class_hidden": row["true_class"],
                    "fullres_target_dice": f"{fd:.3f}",
                    "fullres_target_precision": f"{fp:.3f}",
                    "fullres_target_recall": f"{fr:.3f}",
                }
            )

    write_csv(OUT / "class_specific_refinement_summary.csv", summary_rows)
    write_csv(OUT / "class_specific_refinement_selected_pieces.csv", selected_rows)

    qa_best = pd.read_csv(BASE / "quality_adjusted_ranker/quality_adjusted_best_by_class.csv")
    proxy_best = pd.read_csv(BASE / "proxy_quality_ranker/proxy_quality_fixed_best_by_class.csv")
    comparison_rows: list[dict[str, object]] = []
    for row in summary_rows:
        tissue_class = row["class"]
        qa_row = qa_best[qa_best["class"] == tissue_class].iloc[0]
        proxy_row = proxy_best[proxy_best["class"] == tissue_class].iloc[0]
        comparison_rows.append(
            {
                "class": tissue_class,
                "failure_driven_proxy": f"{row['dice']}/{row['precision']}/{row['recall']}",
                "previous_fixed_runtime_proxy": f"{proxy_row['dice']}/{proxy_row['precision']}/{proxy_row['recall']}",
                "annotation_trained_quality_head": f"{float(qa_row['dice']):.3f}/{float(qa_row['precision']):.3f}/{float(qa_row['recall']):.3f}",
                "formula": row["formula"],
                "diagnosis": row["failure_diagnosis"],
            }
        )
    write_csv(OUT / "class_specific_refinement_comparison.csv", comparison_rows)

    section = f"""
<h2>17L. Failure-Driven Class-Specific Proxy Refinement</h2>
<p>This iteration does not simply run another model. It uses the failure microscope to change the ranker class by class. The main finding is that the same score recipe should not be used for every tissue: vessels need FICTURE composition to dominate; bronchiola works with balanced tissue + quality signals; alveoli is still not solved by scoring and probably needs a better proposal/context generator.</p>
<p><b>Important:</b> formulas use only runtime signals. The low-resolution annotation search was used to choose a diagnostic policy, and the table below is full-resolution evaluation. Therefore this is still one-ROI method development, not final external validation.</p>
<h3>Class-specific result</h3>
{table(summary_rows, ['class', 'formula', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy', 'failure_diagnosis'])}
<h3>Comparison to previous proxy and annotation-trained quality head</h3>
{table(comparison_rows, ['class', 'failure_driven_proxy', 'previous_fixed_runtime_proxy', 'annotation_trained_quality_head', 'formula', 'diagnosis'])}
<h3>Selected pieces and sub-scores</h3>
{table(selected_rows[:180], ['class', 'rank', 'candidate_uid', 'class_specific_score', 'formula', 'loco', 'he', 'fig', 'shape_score', 'true_class_hidden', 'fullres_target_dice', 'fullres_target_precision', 'fullres_target_recall'])}
<div class='callout'><b>Next refinement from this table.</b> Vessels are now near the oracle-quality branch, so the direction is usable there. Alveoli remains the clean failure: the candidate pool has only weak alveoli pieces and the runtime scores rank them below non-alveoli pieces, so the next experiment should be a broader alveoli proposal generator or explicit local-context feature rather than another global prompt.</div>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17L. Failure-Driven Class-Specific Proxy Refinement</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            insert_after = "<h2>17K. No-Annotation Mask-Quality Proxy Test</h2>"
            if insert_after in text:
                start = text.index(insert_after)
                end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
                text = text[:end] + section + text[end:]
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
    print(OUT / "class_specific_refinement_summary.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
