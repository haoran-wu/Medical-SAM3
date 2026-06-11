#!/usr/bin/env python3
"""Semantic FICTURE proposal-generator diagnostic for Jun09.

This branch asks a different failure question from the earlier VLM/ranker
experiments: if the 167 piece pool cannot assemble a good mask for some
classes, can the official FICTURE factor semantics themselves produce a better
candidate proposal?

The sweep below uses hidden annotations only to evaluate and choose diagnostic
settings. It is not a deployable selector; it is a controller test for whether
the next iteration should invest in FICTURE-derived proposal generation.
"""

from __future__ import annotations

import csv
import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

import add_jun09_quality_adjusted_ranker as qa


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "semantic_ficture_proposal_generator"
INPUT_ROOT = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
SUMMARY_OFFICIAL = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
LEGEND_CSV = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/factor_semantic_legend.csv"
FACTOR_INDEX = INPUT_ROOT / "ficture_official_filtered_roi_factor_index.npy"


TARGET_SCORE_KEYS = {
    "bronchiola": "lung_bronchiola",
    "alveoli": "lung_alveoli_normal_adjacent",
    "vessels": "lung_vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
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


def official_gate() -> list[dict[str, object]]:
    summary = json.loads(SUMMARY_OFFICIAL.read_text())
    region = json.loads((INPUT_ROOT / "region_summary_official_filtered_roi.json").read_text())
    he_size = Image.open(qa.HE_ROI).size
    fic_size = Image.open(qa.FICTURE_ROI).size
    factor_shape = np.load(FACTOR_INDEX, mmap_mode="r").shape
    bbox = region.get("bbox_xyxy") or region.get("crop_bbox_xyxy_exclusive")
    rows = [
        {"check": "official FICTURE status", "expected": "PASS_OFFICIAL", "observed": summary.get("status"), "pass": summary.get("status") == "PASS_OFFICIAL"},
        {"check": "candidate input root", "expected": str(INPUT_ROOT), "observed": str(INPUT_ROOT), "pass": INPUT_ROOT.exists()},
        {"check": "H&E ROI size", "expected": "(3144, 3327)", "observed": str(he_size), "pass": he_size == (3144, 3327)},
        {"check": "FICTURE ROI size", "expected": "(3144, 3327)", "observed": str(fic_size), "pass": fic_size == (3144, 3327)},
        {"check": "factor index shape", "expected": "(3327, 3144)", "observed": str(factor_shape), "pass": factor_shape == (3327, 3144)},
        {"check": "official crop bbox", "expected": "[75, 40, 3219, 3367]", "observed": str(bbox), "pass": bbox == [75, 40, 3219, 3367]},
    ]
    if not all(bool(row["pass"]) for row in rows):
        write_csv(OUT / "official_gate_failed.csv", rows)
        raise RuntimeError(f"Official FICTURE gate failed: {rows}")
    return rows


def factor_scores() -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    legend = pd.read_csv(LEGEND_CSV)
    rows: list[dict[str, object]] = []
    scores: dict[int, dict[str, float]] = {}
    for _, row in legend.iterrows():
        factor = int(row["factor"])
        raw_scores = json.loads(row["target_scores_json"])
        class_scores: dict[str, float] = {}
        rec: dict[str, object] = {
            "factor": factor,
            "RGB": row["rgb_text"],
            "hex": row["hex_color"],
            "cell_type": row["llm_inferred_cell_type"],
            "short_name": row["short_name"],
            "marker_basis": row["marker_basis"],
        }
        for cls, target_key in TARGET_SCORE_KEYS.items():
            value = float(raw_scores.get(target_key, 0.0))
            class_scores[cls] = value
            rec[cls] = value
        rows.append(rec)
        scores[factor] = class_scores
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "official_factor_semantic_scores_used.csv", index=False)
    return out_df, scores


def remove_small_and_keep(mask: np.ndarray, min_area: int, keep_top: int | None) -> tuple[np.ndarray, int, int]:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return mask, 0, 0
    sizes = np.bincount(labels.ravel())
    valid = np.flatnonzero(sizes >= min_area)
    valid = valid[valid != 0]
    if keep_top is not None and len(valid) > keep_top:
        valid = valid[np.argsort(sizes[valid])[-keep_top:]]
    if len(valid) == 0:
        return np.zeros_like(mask, dtype=bool), n_labels, 0
    kept = np.isin(labels, valid)
    return kept, n_labels, len(valid)


def build_mask(score_map: np.ndarray, threshold: float, close_iter: int, open_iter: int, fill: bool, min_area: int, keep_top: int | None) -> tuple[np.ndarray, int, int]:
    mask = score_map >= threshold
    if close_iter:
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iter)
    if open_iter:
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=open_iter)
    if fill:
        mask = ndi.binary_fill_holes(mask)
    return remove_small_and_keep(mask.astype(bool), min_area=min_area, keep_top=keep_top)


def class_grid(cls: str) -> dict[str, list[object]]:
    if cls == "bronchiola":
        return {
            "threshold": [0.25, 0.5, 0.75, 0.9],
            "close_iter": [0, 2, 5],
            "open_iter": [0, 1],
            "fill": [False, True],
            "min_area": [200, 800, 2500],
            "keep_top": [None, 3, 8, 20],
        }
    if cls == "vessels":
        return {
            "threshold": [0.25, 0.5, 0.75, 0.9],
            "close_iter": [0, 2, 5],
            "open_iter": [0, 1],
            "fill": [False, True],
            "min_area": [200, 1000, 4000],
            "keep_top": [None, 5, 12, 30],
        }
    if cls == "immune_infiltration":
        return {
            "threshold": [0.25, 0.5, 0.7, 0.85],
            "close_iter": [0, 1, 3],
            "open_iter": [0, 1],
            "fill": [False],
            "min_area": [50, 200, 800],
            "keep_top": [None, 20, 60, 120],
        }
    if cls == "alveoli":
        return {
            "threshold": [0.15, 0.25, 0.4, 0.6, 0.75],
            "close_iter": [0, 3, 8],
            "open_iter": [0, 1, 3],
            "fill": [False, True],
            "min_area": [1000, 8000, 30000],
            "keep_top": [None, 3, 8, 20],
        }
    if cls == "tumor":
        return {
            "threshold": [0.25, 0.5, 0.7, 0.84],
            "close_iter": [0, 3, 8],
            "open_iter": [0, 1, 3],
            "fill": [False, True],
            "min_area": [1000, 8000, 30000],
            "keep_top": [None, 2, 5, 12],
        }
    return {
        "threshold": [0.25, 0.5, 0.7],
        "close_iter": [0, 3, 8],
        "open_iter": [0, 1, 3],
        "fill": [False, True],
        "min_area": [1000, 8000, 30000],
        "keep_top": [None, 2, 5, 12],
    }


def diagnostic_objective(cls: str, dice: float, precision: float, recall: float) -> float:
    if cls in {"bronchiola", "vessels"}:
        return dice + 0.10 * precision + 0.10 * recall
    if cls == "immune_infiltration":
        return dice + 0.05 * precision + 0.15 * recall
    if cls == "alveoli":
        return dice + 0.05 * precision + 0.15 * recall
    return dice + 0.15 * precision + 0.05 * recall


def score_maps(factor_index: np.ndarray, score_lookup: dict[int, dict[str, float]]) -> dict[str, np.ndarray]:
    maps = {cls: np.zeros(factor_index.shape, dtype=np.float32) for cls in qa.CLASS_KEYS}
    valid = factor_index >= 0
    for factor, class_scores in score_lookup.items():
        loc = valid & (factor_index == factor)
        if not np.any(loc):
            continue
        for cls, score in class_scores.items():
            if score:
                maps[cls][loc] = score
    return maps


def resize_score_map(score_map: np.ndarray, small_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((np.clip(score_map, 0, 1) * 255).astype(np.uint8), mode="L")
    return np.array(image.resize(small_size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0


def parse_metric_triplet(value: str) -> tuple[float, float, float]:
    parts = [float(x) for x in value.split("/")]
    return parts[0], parts[1], parts[2]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_rows = official_gate()
    write_csv(OUT / "official_ficture_gate.csv", gate_rows)
    factor_table, score_lookup = factor_scores()
    size = Image.open(qa.HE_ROI).size
    small_size = (512, int(round(512 * size[1] / size[0])))
    factor_index = np.load(FACTOR_INDEX)
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in qa.CLASS_KEYS}
    small_annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], small_size) for c in qa.CLASS_KEYS}
    maps = score_maps(factor_index, score_lookup)
    small_maps = {c: resize_score_map(maps[c], small_size) for c in qa.CLASS_KEYS}

    summary_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []

    for cls in qa.CLASS_KEYS:
        grid = class_grid(cls)
        best: dict[str, object] | None = None
        best_mask: np.ndarray | None = None
        top_for_class: list[dict[str, object]] = []
        lowres_rows: list[dict[str, object]] = []
        for threshold in grid["threshold"]:
            for close_iter in grid["close_iter"]:
                for open_iter in grid["open_iter"]:
                    for fill in grid["fill"]:
                        for min_area in grid["min_area"]:
                            for keep_top in grid["keep_top"]:
                                mask, components_before, components_after = build_mask(
                                    small_maps[cls],
                                    threshold=float(threshold),
                                    close_iter=int(close_iter),
                                    open_iter=int(open_iter),
                                    fill=bool(fill),
                                    min_area=max(1, int(round(int(min_area) * (small_size[0] * small_size[1]) / (size[0] * size[1])))),
                                    keep_top=keep_top if keep_top is None else int(keep_top),
                                )
                                dice, precision, recall = qa.metrics(mask, small_annotations[cls])
                                objective = diagnostic_objective(cls, dice, precision, recall)
                                rec: dict[str, object] = {
                                    "phase": "lowres_parameter_search",
                                    "class": cls,
                                    "threshold": threshold,
                                    "close_iter": close_iter,
                                    "open_iter": open_iter,
                                    "fill_holes": fill,
                                    "min_area": min_area,
                                    "keep_top": "all" if keep_top is None else keep_top,
                                    "components_before_filter": components_before,
                                    "components_after_filter": components_after,
                                    "area_pixels": int(mask.sum()),
                                    "dice": f"{dice:.4f}",
                                    "precision": f"{precision:.4f}",
                                    "recall": f"{recall:.4f}",
                                    "objective": f"{objective:.4f}",
                                }
                                sweep_rows.append(rec)
                                lowres_rows.append(rec)
        lowres_rows.sort(key=lambda r: float(r["objective"]), reverse=True)
        # Full-resolution morphology is expensive, so only recompute the most
        # promising low-res settings. This preserves exact final metrics while
        # avoiding a long full-ROI grid search.
        seen_settings: set[tuple[object, ...]] = set()
        top_settings: list[dict[str, object]] = []
        for rec in lowres_rows:
            key = (rec["threshold"], rec["close_iter"], rec["open_iter"], rec["fill_holes"], rec["min_area"], rec["keep_top"])
            if key in seen_settings:
                continue
            seen_settings.add(key)
            top_settings.append(rec)
            if len(top_settings) >= 12:
                break
        for rec in top_settings:
            keep_top_value = None if rec["keep_top"] == "all" else int(rec["keep_top"])
            mask, components_before, components_after = build_mask(
                maps[cls],
                threshold=float(rec["threshold"]),
                close_iter=int(rec["close_iter"]),
                open_iter=int(rec["open_iter"]),
                fill=bool(rec["fill_holes"]),
                min_area=int(rec["min_area"]),
                keep_top=keep_top_value,
            )
            dice, precision, recall = qa.metrics(mask, annotations[cls])
            objective = diagnostic_objective(cls, dice, precision, recall)
            full_rec = {
                **rec,
                "phase": "fullres_top_recompute",
                "components_before_filter": components_before,
                "components_after_filter": components_after,
                "area_pixels": int(mask.sum()),
                "dice": f"{dice:.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "objective": f"{objective:.4f}",
            }
            sweep_rows.append(full_rec)
            top_for_class.append(full_rec)
            if best is None or objective > float(best["objective"]):
                best = full_rec
                best_mask = mask
        assert best is not None and best_mask is not None
        fig = OUT / "figures" / f"{cls}_semantic_ficture_best.png"
        qa.render_six_panel(
            fig,
            cls,
            best_mask,
            annotations[cls],
            f"{cls}: semantic FICTURE proposal generator",
            (
                "Diagnostic sweep over official factor target scores. "
                f"threshold={best['threshold']}, close={best['close_iter']}, open={best['open_iter']}, "
                f"fill={best['fill_holes']}, min_area={best['min_area']}, keep_top={best['keep_top']}."
            ),
        )
        best = dict(best)
        best["figure_rel"] = str(fig.relative_to(BASE))
        summary_rows.append(best)
        figure_rows.append(
            {
                "class": cls,
                "figure_rel": str(fig.relative_to(BASE)),
                "caption": f"{cls} best semantic FICTURE proposal",
            }
        )
        top_for_class.sort(key=lambda r: float(r["objective"]), reverse=True)
        write_csv(OUT / f"{cls}_top_sweep_rows.csv", top_for_class[:30])

    write_csv(OUT / "semantic_ficture_sweep_all_rows.csv", sweep_rows)
    write_csv(OUT / "semantic_ficture_best_by_class.csv", summary_rows)

    oracle_path = BASE / "oracle_coverage_upper_bound/oracle_coverage_summary.csv"
    compare_rows: list[dict[str, object]] = []
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        by_class = {str(r["class"]).replace("immune infiltration", "immune_infiltration"): r for _, r in oracle.iterrows()}
        for row in summary_rows:
            cls = str(row["class"])
            oracle_row = by_class.get(cls)
            runtime_d = runtime_p = runtime_r = oracle_d = oracle_p = oracle_r = None
            if oracle_row is not None:
                runtime_d, runtime_p, runtime_r = parse_metric_triplet(str(oracle_row["runtime D/P/R"]))
                oracle_d, oracle_p, oracle_r = parse_metric_triplet(str(oracle_row["all-candidate oracle D/P/R"]))
            semantic_d = float(row["dice"])
            semantic_p = float(row["precision"])
            semantic_r = float(row["recall"])
            compare_rows.append(
                {
                    "class": cls,
                    "current_runtime_D/P/R": "" if runtime_d is None else f"{runtime_d:.3f}/{runtime_p:.3f}/{runtime_r:.3f}",
                    "167_pool_oracle_D/P/R": "" if oracle_d is None else f"{oracle_d:.3f}/{oracle_p:.3f}/{oracle_r:.3f}",
                    "semantic_ficture_D/P/R": f"{semantic_d:.3f}/{semantic_p:.3f}/{semantic_r:.3f}",
                    "semantic_vs_runtime_dice_delta": "" if runtime_d is None else f"{semantic_d - runtime_d:+.3f}",
                    "semantic_vs_167_oracle_dice_delta": "" if oracle_d is None else f"{semantic_d - oracle_d:+.3f}",
                    "decision": decision_for_row(cls, semantic_d, runtime_d, oracle_d),
                }
            )
    write_csv(OUT / "semantic_ficture_vs_oracle_comparison.csv", compare_rows)

    section = build_html_section(gate_rows, factor_table, summary_rows, compare_rows, figure_rows)
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AF. Semantic FICTURE Proposal Generator</h2>"
        if marker in text:
            start = text.index(marker)
            # This is the last current section; preserve any future section if present.
            next_marker_pos = text.find("<h2>17AG.", start + len(marker))
            end = next_marker_pos if next_marker_pos != -1 else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "semantic_ficture_vs_oracle_comparison.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


def decision_for_row(cls: str, semantic_d: float, runtime_d: float | None, oracle_d: float | None) -> str:
    if runtime_d is None or oracle_d is None:
        return "semantic proposal generated; no oracle comparison available"
    if semantic_d > max(runtime_d, oracle_d) + 0.05:
        return "promising: semantic FICTURE proposal beats current piece-pool evidence"
    if semantic_d > runtime_d + 0.05:
        return "useful diagnostic: improves runtime but does not exceed oracle ceiling"
    if cls in {"bronchiola", "vessels"}:
        return "do not replace current selector; current piece pool is already near oracle"
    return "not enough: semantic FICTURE alone does not solve this class"


def build_html_section(
    gate_rows: list[dict[str, object]],
    factor_table: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    compare_rows: list[dict[str, object]],
    figure_rows: list[dict[str, object]],
) -> str:
    factor_rows = factor_table.to_dict("records")
    figure_html = []
    for row in figure_rows:
        rel = html.escape(str(row["figure_rel"]))
        figure_html.append(
            f"<figure class='wide-figure'><img src='{rel}' alt='{html.escape(str(row['class']))} semantic FICTURE proposal'>"
            f"<figcaption>{html.escape(str(row['caption']))}</figcaption></figure>"
        )
    return f"""
<h2>17AF. Semantic FICTURE Proposal Generator</h2>
<p><b>Goal.</b> This diagnostic tests a different failure layer. The 17AD oracle showed that the 167 piece pool is already saturated for bronchiola and vessels, but weak for alveoli, tumor, stroma, and immune infiltration. Instead of tuning the VLM again, here I convert the official FICTURE color/cell-type legend into six tissue-specific score maps and ask whether those maps can generate better proposals.</p>
<p><b>What the method sees.</b> It uses only the official same-ROI FICTURE factor index and the current source-matched semantic legend. For each FICTURE factor, the legend provides marker-gene-inferred cell type and six target scores. The script paints those scores back onto the ROI, thresholds the score map, applies simple morphology, removes tiny components, and evaluates the resulting proposal against the hidden annotation.</p>
<p><b>Important caveat.</b> The threshold and morphology sweep below is an annotation-guided diagnostic. It tells us whether FICTURE semantics can provide a better proposal in principle. It is not yet a deployable selector because annotation is used to choose the best row.</p>
<h3>Official-data gate</h3>
{table(gate_rows, ['check', 'expected', 'observed', 'pass'])}
<h3>Factor semantics used</h3>
{table(factor_rows, ['factor', 'RGB', 'hex', 'short_name', 'cell_type', 'bronchiola', 'alveoli', 'vessels', 'tumor', 'stroma', 'immune_infiltration'])}
<h3>Best semantic FICTURE proposal by class</h3>
{table(summary_rows, ['class', 'threshold', 'close_iter', 'open_iter', 'fill_holes', 'min_area', 'keep_top', 'components_after_filter', 'area_pixels', 'dice', 'precision', 'recall'])}
<h3>Comparison with current piece-pool evidence</h3>
{table(compare_rows, ['class', 'current_runtime_D/P/R', '167_pool_oracle_D/P/R', 'semantic_ficture_D/P/R', 'semantic_vs_runtime_dice_delta', 'semantic_vs_167_oracle_dice_delta', 'decision'])}
<div class='callout'><b>How to read this.</b> If semantic FICTURE beats both the current runtime and the 167-pool oracle, then the next skill iteration should add a FICTURE proposal generator. If it only improves runtime but not oracle, the current selector was weak but the 167 pool still limits the method. If it stays below both, then raw FICTURE semantics should remain an auxiliary prior rather than the proposal source.</div>
{''.join(figure_html)}
"""


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
        BASE / "structured_quality_v2",
        BASE / "failure_taxonomy_v3_after_quality_v2",
        BASE / "oracle_coverage_upper_bound",
        BASE / "controller_after_oracle_coverage",
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
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text()
    import re

    missing: list[str] = []
    for src in re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", text):
        if src.startswith(("http://", "https://", "data:")):
            continue
        if not (BASE / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing HTML images: {missing[:20]}")


if __name__ == "__main__":
    main()
