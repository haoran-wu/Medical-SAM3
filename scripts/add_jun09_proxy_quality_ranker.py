#!/usr/bin/env python3
"""No-annotation quality proxy diagnostic for Jun09.

The quality-adjusted branch proves that a mask-quality head helps, but it is
trained from annotation-derived quality labels. This script tests the next
question: how far can we get with quality signals available at deployment time?

The scoring formulas below do not use annotation. Annotation is used only after
selection to measure Dice, Precision, and Recall.
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
OUT = BASE / "proxy_quality_ranker"
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


def norm01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def margin(row: pd.Series, prefix: str, tissue_class: str) -> float:
    target = float(row.get(f"{prefix}_{tissue_class}", 0) or 0) / 100.0
    others = [float(row.get(f"{prefix}_{c}", 0) or 0) / 100.0 for c in CLASS_KEYS if c != tissue_class]
    return norm01((target - max(others or [0.0]) + 1.0) / 2.0)


def area_plausibility(area_frac: float, tissue_class: str) -> float:
    """Deployment-time size prior, intentionally broad.

    This does not know annotation. It only prevents extreme speckles or very
    large background-like masks from dominating a class score.
    """
    log_area = np.log10(max(float(area_frac), 1e-7))
    if tissue_class in {"bronchiola", "vessels", "immune_infiltration"}:
        center, width = -3.0, 1.0
    else:
        center, width = -2.4, 1.1
    return norm01(np.exp(-((log_area - center) / width) ** 2))


def compute_proxy_scores(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        for tissue_class in CLASS_KEYS:
            loco = float(row.get(tissue_class, 0) or 0) / 100.0
            he = float(row.get(f"he_clip_large_{tissue_class}", 0) or 0) / 100.0
            fig = float(row.get(f"ficture_prior_{tissue_class}", 0) or 0) / 100.0
            shape = float(row.get(f"shape_skill_{tissue_class}", 0) or 0) / 100.0
            he_margin = margin(row, "he_clip_large", tissue_class)
            fig_margin = margin(row, "ficture_prior", tissue_class)
            shape_margin = margin(row, "shape_skill", tissue_class)
            area_ok = area_plausibility(float(row.get("area_frac", 0) or 0), tissue_class)
            solidity = norm01(float(row.get("solidity", 0) or 0))
            thinness = float(row.get("thinness", 0) or 0)
            thinness_ok = norm01(1.0 - min(abs(thinness - 0.045) / 0.08, 1.0))

            runtime_quality_proxy = norm01(
                0.25 * area_ok
                + 0.20 * shape
                + 0.15 * shape_margin
                + 0.15 * he_margin
                + 0.10 * fig_margin
                + 0.10 * solidity
                + 0.05 * thinness_ok
            )
            membership_proxy = norm01(0.50 * loco + 0.25 * he + 0.15 * shape + 0.10 * fig)
            he_shape_membership = norm01(0.55 * loco + 0.30 * he + 0.15 * shape)
            structured_membership = norm01(0.45 * loco + 0.20 * he + 0.20 * fig + 0.15 * shape)
            quality_times_membership = norm01(membership_proxy * (0.35 + 0.65 * runtime_quality_proxy))
            proxy_blend = norm01(0.65 * membership_proxy + 0.35 * runtime_quality_proxy)
            rows.append(
                {
                    "row_key": row["row_key"],
                    "candidate_uid": row["candidate_uid"],
                    "target_class": tissue_class,
                    "loco_class_score": loco,
                    "he_clip_target": he,
                    "ficture_prior_target": fig,
                    "shape_skill_target": shape,
                    "area_plausibility": area_ok,
                    "he_margin_proxy": he_margin,
                    "ficture_margin_proxy": fig_margin,
                    "shape_margin_proxy": shape_margin,
                    "runtime_quality_proxy": runtime_quality_proxy,
                    "score_membership_proxy": membership_proxy,
                    "score_he_shape_membership": he_shape_membership,
                    "score_structured_membership": structured_membership,
                    "score_quality_times_membership": quality_times_membership,
                    "score_proxy_blend": proxy_blend,
                }
            )
    return pd.DataFrame(rows)


def select_pieces(
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

    mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = public_by_key[key]["mask_path"]
        if path not in mask_cache:
            mask_cache[path] = qa.load_mask(Path(path), size)
        return mask_cache[path]

    proxy_df = compute_proxy_scores(data)
    proxy_df.to_csv(OUT / "proxy_quality_pair_scores.csv", index=False)

    fixed_policy = {
        "bronchiola": (0.45, 0.60, 4),
        "alveoli": (0.40, 0.45, 2),
        "vessels": (0.45, 0.75, 8),
        "tumor": (0.45, 0.45, 10),
        "stroma": (0.40, 0.90, 36),
        "immune_infiltration": (0.40, 0.90, 48),
    }
    score_methods = {
        "runtime_proxy_fixed_quality_times_membership": "score_quality_times_membership",
        "runtime_proxy_fixed_proxy_blend": "score_proxy_blend",
        "runtime_proxy_fixed_he_shape": "score_he_shape_membership",
    }
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    def record(method: str, tissue_class: str, selected: list[pd.Series], union: np.ndarray, policy: str, score_col: str) -> None:
        d, p, r = qa.metrics(union, annotations[tissue_class])
        fig = OUT / "figures" / f"{tissue_class}_{method}_union.png"
        qa.render_six_panel(
            fig,
            tissue_class,
            union,
            annotations[tissue_class],
            f"{tissue_class}: no-annotation quality proxy ({method})",
            policy,
        )
        summary_rows.append(
            {
                "method": method,
                "class": tissue_class,
                "selected_piece_count": len(selected),
                "dice": f"{d:.3f}",
                "precision": f"{p:.3f}",
                "recall": f"{r:.3f}",
                "policy": policy,
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        for rank, row in enumerate(selected[:80], 1):
            fd, fp, fr = qa.metrics(get_mask(row["row_key"]), annotations[tissue_class])
            selected_rows.append(
                {
                    "method": method,
                    "class": tissue_class,
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "score": f"{float(row[score_col]):.4f}",
                    "runtime_quality_proxy": f"{float(row['runtime_quality_proxy']):.4f}",
                    "loco_class_score": f"{float(row['loco_class_score']):.4f}",
                    "he_clip_target": f"{float(row['he_clip_target']):.4f}",
                    "ficture_prior_target": f"{float(row['ficture_prior_target']):.4f}",
                    "shape_skill_target": f"{float(row['shape_skill_target']):.4f}",
                    "fullres_target_dice": f"{fd:.3f}",
                    "fullres_target_precision": f"{fp:.3f}",
                    "fullres_target_recall": f"{fr:.3f}",
                }
            )

    for method, score_col in score_methods.items():
        for tissue_class in CLASS_KEYS:
            min_score, max_overlap, max_pieces = fixed_policy[tissue_class]
            selected, union = select_pieces(
                proxy_df[proxy_df["target_class"] == tissue_class],
                score_col,
                get_mask,
                (size[1], size[0]),
                min_score,
                max_overlap,
                max_pieces,
            )
            record(
                method,
                tissue_class,
                selected,
                union,
                f"fixed runtime policy: {score_col}>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}",
                score_col,
            )

    # Same no-annotation scores, but threshold/NMS policy is tuned against the
    # annotation. This is only an upper-bound microscope for assembly policy.
    oracle_methods = {
        "oracle_threshold_proxy_quality_times_membership": "score_quality_times_membership",
        "oracle_threshold_proxy_blend": "score_proxy_blend",
    }
    for method, score_col in oracle_methods.items():
        for tissue_class in CLASS_KEYS:
            best = None
            class_df = proxy_df[proxy_df["target_class"] == tissue_class]
            for min_score in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
                for max_overlap in [0.45, 0.60, 0.75, 0.90]:
                    for max_pieces in [2, 4, 8, 12, 24, 36, 48]:
                        selected, union = select_pieces(
                            class_df,
                            score_col,
                            get_mask,
                            (size[1], size[0]),
                            min_score,
                            max_overlap,
                            max_pieces,
                        )
                        d, p, r = qa.metrics(union, annotations[tissue_class])
                        if tissue_class in {"bronchiola", "vessels"}:
                            objective = d + 0.20 * p + 0.10 * r
                        elif tissue_class == "immune_infiltration":
                            objective = d + 0.10 * p + 0.20 * r
                        else:
                            objective = d + 0.25 * p + 0.05 * r
                        if best is None or objective > best[0]:
                            best = (objective, min_score, max_overlap, max_pieces, selected, union, d, p, r)
            assert best is not None
            _, min_score, max_overlap, max_pieces, selected, union, _, _, _ = best
            record(
                method,
                tissue_class,
                selected,
                union,
                f"annotation-tuned upper bound: {score_col}>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}",
                score_col,
            )

    write_csv(OUT / "proxy_quality_assembly_summary.csv", summary_rows)
    write_csv(OUT / "proxy_quality_selected_pieces.csv", selected_rows)

    best_by_class: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        rows = [r for r in summary_rows if r["class"] == tissue_class and str(r["method"]).startswith("runtime_proxy_fixed")]
        rows.sort(key=lambda r: float(r["dice"]) + 0.05 * float(r["precision"]), reverse=True)
        best_by_class.append(rows[0])
    write_csv(OUT / "proxy_quality_fixed_best_by_class.csv", best_by_class)

    qa_best = pd.read_csv(BASE / "quality_adjusted_ranker/quality_adjusted_best_by_class.csv")
    structured = pd.read_csv(BASE / "structured_veto_microscope/structured_veto_summary.csv")
    comparison_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        fixed = next(r for r in best_by_class if r["class"] == tissue_class)
        oracle_rows = [r for r in summary_rows if r["class"] == tissue_class and str(r["method"]).startswith("oracle_threshold")]
        oracle_rows.sort(key=lambda r: float(r["dice"]) + 0.05 * float(r["precision"]), reverse=True)
        qa_row = qa_best[qa_best["class"] == tissue_class].iloc[0]
        st_row = structured[structured["class"] == tissue_class].iloc[0]
        comparison_rows.append(
            {
                "class": tissue_class,
                "best_fixed_runtime_proxy": f"{fixed['dice']}/{fixed['precision']}/{fixed['recall']}",
                "best_fixed_method": fixed["method"],
                "oracle_threshold_proxy_upper_bound": f"{oracle_rows[0]['dice']}/{oracle_rows[0]['precision']}/{oracle_rows[0]['recall']}",
                "quality_adjusted_annotation_trained": f"{float(qa_row['dice']):.3f}/{float(qa_row['precision']):.3f}/{float(qa_row['recall']):.3f}",
                "structured_veto": f"{float(st_row['dice']):.3f}/{float(st_row['precision']):.3f}/{float(st_row['recall']):.3f}",
            }
        )
    write_csv(OUT / "proxy_quality_comparison.csv", comparison_rows)

    fixed_rows = [r for r in summary_rows if str(r["method"]).startswith("runtime_proxy_fixed")]
    oracle_rows = [r for r in summary_rows if str(r["method"]).startswith("oracle_threshold")]
    section = f"""
<h2>17K. No-Annotation Mask-Quality Proxy Test</h2>
<p>This section tests whether the quality skill can be approximated without seeing annotation. The proxy uses only deployment-time signals: H&amp;E morphology score, structured FICTURE composition prior, shape/location score, mask size and shape plausibility, and score margins. It deliberately excludes <code>compact_funnel_score</code>, because that score is almost perfectly correlated with hidden component Dice and is therefore an oracle/funnel diagnostic signal, not a deployable input.</p>
<p><b>How to read this section:</b> <code>runtime_proxy_fixed_*</code> uses fixed thresholds that do not look at annotation. <code>oracle_threshold_proxy_*</code> uses the same no-annotation scores, but tunes the final threshold/overlap policy against annotation, so it is only an upper-bound microscope for assembly policy.</p>
<h3>Best fixed runtime proxy by class</h3>
{table(best_by_class, ['method', 'class', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy'])}
<h3>Comparison Against Other Branches</h3>
{table(comparison_rows, ['class', 'best_fixed_runtime_proxy', 'best_fixed_method', 'oracle_threshold_proxy_upper_bound', 'quality_adjusted_annotation_trained', 'structured_veto'])}
<h3>All fixed runtime proxy runs</h3>
{table(fixed_rows, ['method', 'class', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy'])}
<h3>Oracle-threshold upper-bound runs</h3>
{table(oracle_rows, ['method', 'class', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy'])}
<h3>Selected pieces with deployable proxy sub-scores</h3>
{table(selected_rows[:160], ['method', 'class', 'rank', 'candidate_uid', 'score', 'runtime_quality_proxy', 'loco_class_score', 'he_clip_target', 'ficture_prior_target', 'shape_skill_target', 'fullres_target_dice', 'fullres_target_precision', 'fullres_target_recall'])}
<div class='callout'><b>Failure diagnosis rule.</b> If the fixed runtime proxy is much worse than the annotation-trained quality head, the missing piece is not another prompt; it is a deployable mask-quality estimator. If oracle-threshold proxy improves sharply over fixed proxy, the score is usable but the assembly policy needs calibration. If both are poor, the candidate generator or tissue feature itself is the bottleneck.</div>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17K. No-Annotation Mask-Quality Proxy Test</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("<h2>18.", start) if "<h2>18." in text[start:] else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            insert_after = "<h2>17J. Quality-Adjusted Ranker Test</h2>"
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
    print(OUT / "proxy_quality_comparison.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
