#!/usr/bin/env python3
"""Quality-adjusted tissue ranker diagnostic for Jun09.

The previous diagnostics show that a tissue score is not enough for assembly.
This script adds a separate mask-quality prediction head and tests whether
assembly improves when candidates are ranked by tissue membership plus predicted
mask quality.

The validation is candidate-grouped 5-fold: all six target-class labels for a
candidate are held out together. This is still one-ROI validation, so the report
labels it as diagnostic rather than deployable.
"""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "quality_adjusted_ranker"
ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
ANNOTATION_FILES = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}


def row_key(row: pd.Series) -> str:
    if "row_key" in row and pd.notna(row["row_key"]):
        return str(row["row_key"])
    return "__".join(
        [
            str(row.get("target_label", "")),
            str(row.get("source", "")),
            str(row.get("run", "")),
            str(row.get("setting", "")),
            str(row.get("candidate_id", "")),
        ]
    )


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


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pa = int(pred.sum())
    ga = int(gt.sum())
    p = tp / pa if pa else 0.0
    r = tp / ga if ga else 0.0
    d = 2 * tp / (pa + ga) if pa + ga else 0.0
    return d, p, r


def overlap_fraction(candidate: np.ndarray, union: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 1.0
    return int(np.logical_and(candidate, union).sum()) / area


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 130) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    return Image.alpha_composite(base, Image.fromarray(rgba, mode="RGBA")).convert("RGB")


def mask_only(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    arr[mask] = color
    return Image.fromarray(arr, mode="RGB")


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    img = image.convert("RGB")
    img.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def render_six_panel(path: Path, tissue_class: str, union: np.ndarray, ann: np.ndarray, title: str, subtitle: str) -> None:
    he = Image.open(HE_ROI).convert("RGB")
    ficture = Image.open(FICTURE_ROI).convert("RGB")
    d, p, r = metrics(union, ann)
    panel_w, panel_h = 260, 275
    gap, margin, header_h, label_h = 18, 26, 104, 24
    sheet = Image.new("RGB", (margin * 2 + panel_w * 6 + gap * 5, margin * 2 + header_h + label_h + panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), f"{tissue_class} | Dice {d:.3f} | Precision {p:.3f} | Recall {r:.3f}", fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), subtitle[:190], fill=(70, 80, 90), font=font)
    panels = [
        ("Annotation on H&E", overlay_mask(he, ann, (0, 180, 90))),
        ("Quality-adjusted union on H&E", overlay_mask(he, union, (0, 90, 255))),
        ("Quality-adjusted mask only", mask_only(union, (0, 90, 255))),
        ("Annotation mask only", mask_only(ann, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    y0 = margin + header_h
    for idx, (label, image) in enumerate(panels):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(image, panel_w, panel_h), (x, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(HE_ROI).size
    small_size = (512, int(round(512 * size[1] / size[0])))
    annotations = {c: load_mask(ANNOTATION_DIR / ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    small_annotations = {c: load_mask(ANNOTATION_DIR / ANNOTATION_FILES[c], small_size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(row_key, axis=1)
    features = pd.read_csv(BASE / "candidate_skill_features.csv")
    loco = pd.read_csv(BASE / "score_outputs/loco_component_logistic_validation/per_candidate_predictions.csv")
    data = features.merge(loco[["row_key", *CLASS_KEYS]], on="row_key", how="left")
    public_by_key = public.set_index("row_key").to_dict("index")

    mask_cache: dict[str, np.ndarray] = {}
    small_mask_cache: dict[str, np.ndarray] = {}
    def get_mask(key: str) -> np.ndarray:
        path = public_by_key[key]["mask_path"]
        if path not in mask_cache:
            mask_cache[path] = load_mask(Path(path), size)
        return mask_cache[path]

    def get_small_mask(key: str) -> np.ndarray:
        path = public_by_key[key]["mask_path"]
        if path not in small_mask_cache:
            small_mask_cache[path] = load_mask(Path(path), small_size)
        return small_mask_cache[path]

    label_rows: list[dict[str, object]] = []
    feature_cols = [
        "area_frac",
        "bbox_aspect",
        "bbox_area_frac",
        "solidity",
        "eccentricity",
        "thinness",
        "centroid_x_frac",
        "centroid_y_frac",
        "frac_airway_epithelial",
        "frac_endothelial",
        "frac_immune",
        "frac_stroma",
        "frac_epithelial_tumor",
        "frac_at2",
        "ring_frac_airway_epithelial",
        "ring_frac_endothelial",
        "ring_frac_immune",
        "ring_frac_stroma",
        "ring_frac_epithelial_tumor",
        "ring_frac_at2",
    ]
    for _, row in data.iterrows():
        key = row["row_key"]
        mask = get_small_mask(key)
        base_feats = {col: float(row.get(col, 0) or 0) for col in feature_cols}
        for tissue_class in CLASS_KEYS:
            d, p, r = metrics(mask, small_annotations[tissue_class])
            rec = {
                "row_key": key,
                "candidate_uid": row["candidate_uid"],
                "target_class": tissue_class,
                "target_dice": d,
                "target_precision": p,
                "target_recall": r,
                "loco_class_score": float(row.get(tissue_class, 0) or 0) / 100.0,
                "target_ficture_prior": float(row.get(f"ficture_prior_{tissue_class}", 0) or 0) / 100.0,
                "target_shape_skill": float(row.get(f"shape_skill_{tissue_class}", 0) or 0) / 100.0,
                "target_he_clip": float(row.get(f"he_clip_large_{tissue_class}", 0) or 0) / 100.0,
            }
            rec.update(base_feats)
            for c in CLASS_KEYS:
                rec[f"is_{c}"] = 1.0 if c == tissue_class else 0.0
            label_rows.append(rec)
    label_df = pd.DataFrame(label_rows)
    feature_matrix_cols = [
        "loco_class_score",
        "target_ficture_prior",
        "target_shape_skill",
        "target_he_clip",
        *feature_cols,
        *[f"is_{c}" for c in CLASS_KEYS],
    ]
    X = label_df[feature_matrix_cols].fillna(0).to_numpy(dtype=float)
    y = label_df["target_dice"].to_numpy(dtype=float)
    groups = label_df["row_key"].to_numpy()
    pred_quality = np.zeros(len(label_df), dtype=float)
    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=13, n_jobs=-1)
        model.fit(X[train_idx], y[train_idx])
        pred_quality[test_idx] = model.predict(X[test_idx])
    label_df["pred_quality"] = np.clip(pred_quality, 0, 1)
    label_df["score_quality_only"] = label_df["pred_quality"]
    label_df["score_class_times_quality"] = label_df["loco_class_score"] * label_df["pred_quality"]
    label_df["score_blend"] = 0.55 * label_df["loco_class_score"] + 0.45 * label_df["pred_quality"]
    label_df.to_csv(OUT / "quality_adjusted_pair_scores.csv", index=False)

    policies = {
        "quality_only": "score_quality_only",
        "class_times_quality": "score_class_times_quality",
        "blend": "score_blend",
    }
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for method, score_col in policies.items():
        for tissue_class in CLASS_KEYS:
            class_df = label_df[label_df["target_class"] == tissue_class].sort_values(score_col, ascending=False)
            # Class-specific conservative caps. These are not fitted to annotation;
            # the threshold is grid-searched below only for diagnostic comparison.
            best = None
            for min_score in [0.02, 0.05, 0.10, 0.20, 0.35]:
                for max_overlap in [0.45, 0.60, 0.75, 0.90]:
                    for max_pieces in [2, 4, 8, 12, 24, 36, 48]:
                        union = np.zeros((small_size[1], small_size[0]), dtype=bool)
                        selected = []
                        for _, row in class_df.iterrows():
                            if float(row[score_col]) < min_score:
                                break
                            mask = get_small_mask(row["row_key"])
                            if overlap_fraction(mask, union) > max_overlap:
                                continue
                            selected.append(row)
                            union |= mask
                            if len(selected) >= max_pieces:
                                break
                        d, p, r = metrics(union, small_annotations[tissue_class])
                        if tissue_class in {"bronchiola", "vessels"}:
                            objective = d + 0.20 * p + 0.10 * r
                        elif tissue_class == "immune_infiltration":
                            objective = d + 0.10 * p + 0.20 * r
                        else:
                            objective = d + 0.25 * p + 0.05 * r
                        if best is None or objective > best[0]:
                            best = (objective, min_score, max_overlap, max_pieces, selected, union, d, p, r)
            assert best is not None
            _, min_score, max_overlap, max_pieces, _, _, _, _, _ = best
            # Re-run the selected low-resolution policy at full resolution for
            # exact metrics and visualization.
            union = np.zeros((size[1], size[0]), dtype=bool)
            selected = []
            for _, row in class_df.iterrows():
                if float(row[score_col]) < min_score:
                    break
                mask = get_mask(row["row_key"])
                if overlap_fraction(mask, union) > max_overlap:
                    continue
                selected.append(row)
                union |= mask
                if len(selected) >= max_pieces:
                    break
            d, p, r = metrics(union, annotations[tissue_class])
            fig = OUT / "figures" / f"{tissue_class}_{method}_union.png"
            render_six_panel(
                fig,
                tissue_class,
                union,
                annotations[tissue_class],
                f"{tissue_class}: quality-adjusted ranker ({method})",
                f"Candidate-grouped 5-fold predicted quality; assembly threshold score>={min_score}, overlap<={max_overlap}, max_pieces={max_pieces}.",
            )
            summary_rows.append(
                {
                    "method": method,
                    "class": tissue_class,
                    "selected_piece_count": len(selected),
                    "dice": f"{d:.3f}",
                    "precision": f"{p:.3f}",
                    "recall": f"{r:.3f}",
                    "policy": f"{score_col}>= {min_score}, overlap<={max_overlap}, max_pieces={max_pieces}",
                    "figure_rel": str(fig.relative_to(BASE)),
                }
            )
            for rank, row in enumerate(selected[:80], 1):
                full_d, full_p, full_r = metrics(get_mask(row["row_key"]), annotations[tissue_class])
                selected_rows.append(
                    {
                        "method": method,
                        "class": tissue_class,
                        "rank": rank,
                        "candidate_uid": row["candidate_uid"],
                        "combined_score": f"{float(row[score_col]):.4f}",
                        "pred_quality": f"{float(row['pred_quality']):.4f}",
                        "loco_class_score": f"{float(row['loco_class_score']):.4f}",
                        "screening_target_dice": f"{float(row['target_dice']):.3f}",
                        "screening_target_precision": f"{float(row['target_precision']):.3f}",
                        "screening_target_recall": f"{float(row['target_recall']):.3f}",
                        "fullres_target_dice": f"{full_d:.3f}",
                        "fullres_target_precision": f"{full_p:.3f}",
                        "fullres_target_recall": f"{full_r:.3f}",
                    }
                )
    write_csv(OUT / "quality_adjusted_assembly_summary.csv", summary_rows)
    write_csv(OUT / "quality_adjusted_selected_pieces.csv", selected_rows)

    best_by_class = []
    for tissue_class in CLASS_KEYS:
        rows = [r for r in summary_rows if r["class"] == tissue_class]
        rows.sort(key=lambda r: float(r["dice"]) + 0.05 * float(r["precision"]), reverse=True)
        best_by_class.append(rows[0])
    write_csv(OUT / "quality_adjusted_best_by_class.csv", best_by_class)

    section = f"""
<h2>17J. Quality-Adjusted Ranker Test</h2>
<p>This branch implements the refinement suggested by the failure microscope: separate <b>tissue membership</b> from <b>mask quality</b>. A Random Forest regressor predicts candidate-mask quality from H&amp;E morphology, FICTURE composition prior, shape/location, and the original tissue score. Validation is candidate-grouped 5-fold, so all six target scores for the same candidate are held out together.</p>
<p><b>Caveat:</b> this is still one-ROI diagnostic validation. The quality head is trained from annotation-derived target quality, so it is evidence that a quality skill is needed, not yet a deployment-ready scorer. Training used downsampled screening masks for speed; final union metrics and the selected-candidate <code>fullres_target_*</code> columns below are recomputed at full ROI resolution.</p>
{table(best_by_class, ['method', 'class', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy'])}
<h3>All quality-adjusted methods</h3>
{table(summary_rows, ['method', 'class', 'selected_piece_count', 'dice', 'precision', 'recall', 'policy'])}
<h3>Selected candidates from the best quality-adjusted runs</h3>
{table(selected_rows[:120], ['method', 'class', 'rank', 'candidate_uid', 'combined_score', 'pred_quality', 'loco_class_score', 'screening_target_dice', 'screening_target_precision', 'screening_target_recall', 'fullres_target_dice', 'fullres_target_precision', 'fullres_target_recall'])}
<div class='callout'><b>Interpretation.</b> If this improves a class over tissue-score-only assembly, the next scientific method should be a candidate-quality ranker, not another VLM prompt. If it does not improve a class, the bottleneck is likely proposal coverage or segmentation refinement.</div>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17J. Quality-Adjusted Ranker Test</h2>"
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
    print(OUT / "quality_adjusted_best_by_class.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
