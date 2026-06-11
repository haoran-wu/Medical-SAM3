#!/usr/bin/env python3
"""Append a complete bbox-level H&E morphology/context audit for alveoli.

The Jun07 broad-box candidate pack only saved 20 reverse-blur close-up crops,
but the hidden table contains bboxes for all 100 candidates.  This audit uses
the official full H&E ROI and FICTURE ROI to extract image features for all 100
broad-box candidates.  It does not need remote masks, and it does not use
hidden Dice for feature computation.

Hidden annotation is used only afterward to test whether the feature family can
separate true broad alveoli candidates from broad false positives.
"""

from __future__ import annotations

import ast
import csv
import html
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_bbox_he_morphology_audit"
JUN07_INPUTS = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_GPT_manual_alveoli_prompt_test_inputs"

HE_FULL = ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
FICTURE_FULL = ROOT / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/filtered_ficture_official_full_he_canvas.png"
ROI_BOX = (75, 40, 3219, 3367)
ROI_W = ROI_BOX[2] - ROI_BOX[0]
ROI_H = ROI_BOX[3] - ROI_BOX[1]


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


def fmt3(v: object) -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "NA"
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def parse_bbox(text: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = ast.literal_eval(str(text))
    return int(x1), int(y1), int(x2), int(y2)


def expand_bbox(bbox: tuple[int, int, int, int], scale: float = 1.35) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - w / 2)))
    ny1 = max(0, int(round(cy - h / 2)))
    nx2 = min(ROI_W, int(round(cx + w / 2)))
    ny2 = min(ROI_H, int(round(cy + h / 2)))
    return nx1, ny1, nx2, ny2


def crop_roi_image(full_path: Path) -> Image.Image:
    im = Image.open(full_path).convert("RGB")
    roi = im.crop(ROI_BOX)
    if roi.size != (ROI_W, ROI_H):
        raise RuntimeError(f"Unexpected ROI size from {full_path}: {roi.size}")
    return roi


def image_features(im: Image.Image, prefix: str) -> dict[str, float]:
    arr = np.asarray(im.resize((256, 256)).convert("RGB")).astype("float32") / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / mx, 0)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    grad = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(gray, bins=32, range=(0, 1), density=True)
    hist = hist / max(hist.sum(), 1e-8)
    entropy = float(-(hist * np.log2(hist + 1e-8)).sum())

    # H&E-oriented proxies.  These are deliberately simple and transparent.
    bright_low_sat = (mx > 0.82) & (sat < 0.22)
    very_bright = mx > 0.92
    pink = (arr[:, :, 0] > 0.62) & (arr[:, :, 1] < 0.72) & (arr[:, :, 2] > 0.55)
    purple = (arr[:, :, 0] > 0.40) & (arr[:, :, 2] > 0.45) & (arr[:, :, 1] < 0.62)
    tissue = ~(very_bright & (sat < 0.18))
    return {
        f"{prefix}_mean_brightness": float(mx.mean()),
        f"{prefix}_std_brightness": float(mx.std()),
        f"{prefix}_mean_saturation": float(sat.mean()),
        f"{prefix}_std_saturation": float(sat.std()),
        f"{prefix}_bright_low_sat_fraction": float(bright_low_sat.mean()),
        f"{prefix}_very_bright_fraction": float(very_bright.mean()),
        f"{prefix}_pink_fraction": float(pink.mean()),
        f"{prefix}_purple_fraction": float(purple.mean()),
        f"{prefix}_tissue_fraction": float(tissue.mean()),
        f"{prefix}_edge_density": float((grad > 0.065).mean()),
        f"{prefix}_mean_gradient": float(grad.mean()),
        f"{prefix}_gray_entropy": entropy,
    }


def ficture_features(im: Image.Image, prefix: str) -> dict[str, float]:
    arr = np.asarray(im.resize((256, 256)).convert("RGB")).astype("uint8")
    nonblack = (arr.sum(axis=2) > 20)
    cyanish = (arr[:, :, 1] > 180) & (arr[:, :, 2] > 180) & (arr[:, :, 0] < 80)
    yellow = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 180) & (arr[:, :, 2] < 80)
    magenta = (arr[:, :, 0] > 180) & (arr[:, :, 2] > 150) & (arr[:, :, 1] < 230)
    green = (arr[:, :, 1] > 170) & (arr[:, :, 0] < 100)
    return {
        f"{prefix}_ficture_nonblack_fraction": float(nonblack.mean()),
        f"{prefix}_ficture_cyan_fraction": float(cyanish.mean()),
        f"{prefix}_ficture_yellow_fraction": float(yellow.mean()),
        f"{prefix}_ficture_magenta_fraction": float(magenta.mean()),
        f"{prefix}_ficture_green_fraction": float(green.mean()),
    }


def build_features() -> pd.DataFrame:
    hidden = pd.read_csv(JUN07_INPUTS / "tables/hidden_candidate_truth.csv")
    ficture_summary = pd.read_csv(JUN07_INPUTS / "tables/alveoli_box384_ficture_text_summary.csv")
    df = hidden.merge(
        ficture_summary[["candidate_id", "inside_group_summary", "local_group_summary"]],
        on="candidate_id",
        how="left",
    )
    he_roi = crop_roi_image(HE_FULL)
    fic_roi = crop_roi_image(FICTURE_FULL)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        bbox = parse_bbox(row["candidate_bbox_xyxy"])
        ctx = expand_bbox(bbox, scale=1.35)
        x1, y1, x2, y2 = bbox
        cx1, cy1, cx2, cy2 = ctx
        he_crop = he_roi.crop(bbox)
        he_ctx = he_roi.crop(ctx)
        fic_crop = fic_roi.crop(bbox)
        features = {
            "candidate_id": int(row["candidate_id"]),
            "candidate_uid": row["candidate_uid"],
            "candidate_pixels": int(row["candidate_pixels"]),
            "bbox": row["candidate_bbox_xyxy"],
            "bbox_w": x2 - x1,
            "bbox_h": y2 - y1,
            "bbox_area": (x2 - x1) * (y2 - y1),
            "fill_ratio": int(row["candidate_pixels"]) / max((x2 - x1) * (y2 - y1), 1),
            "cx_norm": ((x1 + x2) / 2) / ROI_W,
            "cy_norm": ((y1 + y2) / 2) / ROI_H,
            "component_best_dice": float(row["component_best_dice"]),
            "component_best_precision": float(row["component_best_precision"]),
            "component_best_recall": float(row["component_best_recall"]),
            "high_quality": int(float(row["component_best_dice"]) >= 0.20),
        }
        features.update(image_features(he_crop, "he_bbox"))
        features.update(image_features(he_ctx, "he_context"))
        features.update(ficture_features(fic_crop, "bbox"))
        rows.append(features)
    return pd.DataFrame(rows)


def triangle(series: pd.Series, center: float, width: float) -> pd.Series:
    return (1 - (series - center).abs() / width).clip(0, 1)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    # No hidden Dice is used here.  This is a transparent morphology hypothesis:
    # broad alveolar regions should contain enough tissue color, open/bright
    # space, and edge/texture complexity while avoiding very blank white areas.
    df["he_airspace_texture_score"] = (
        0.9 * triangle(df["he_bbox_bright_low_sat_fraction"], 0.20, 0.20)
        + 0.8 * triangle(df["he_bbox_tissue_fraction"], 0.74, 0.28)
        + 0.7 * triangle(df["he_bbox_edge_density"], 0.34, 0.25)
        + 0.5 * triangle(df["he_bbox_pink_fraction"], 0.32, 0.25)
        + 0.4 * triangle(df["he_context_bright_low_sat_fraction"], 0.20, 0.22)
        - 0.8 * (df["he_bbox_very_bright_fraction"] > 0.75).astype(float)
    )
    df["he_shape_morphology_score"] = (
        df["he_airspace_texture_score"]
        + 0.6 * triangle(df["candidate_pixels"], 460_000, 220_000)
        + 0.4 * triangle(df["fill_ratio"], 0.65, 0.35)
    )
    df["he_context_position_free_score"] = df["he_shape_morphology_score"]
    df["he_context_position_diagnostic_score"] = df["he_shape_morphology_score"] + 0.5 * df["cy_norm"] + 0.25 * (1 - df["cx_norm"])
    return df


def ranking_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for score_col, label, uses_position in [
        ("he_airspace_texture_score", "H&E bbox/context texture only", "no"),
        ("he_shape_morphology_score", "H&E morphology + shape no position", "no"),
        ("he_context_position_diagnostic_score", "H&E morphology + diagnostic ROI position", "yes"),
    ]:
        ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        ranks = ranked.reset_index().set_index("candidate_id")["index"].add(1)
        top5 = ranked.head(5)
        top10 = ranked.head(10)
        rows.append(
            {
                "score": label,
                "uses ROI position": uses_position,
                "rank candidate 12": int(ranks.get(12, -1)),
                "rank candidate 13": int(ranks.get(13, -1)),
                "rank FP 82": int(ranks.get(82, -1)),
                "rank FP 90": int(ranks.get(90, -1)),
                "top5 high-quality count": int(top5["high_quality"].sum()),
                "top10 high-quality count": int(top10["high_quality"].sum()),
                "top5 mean hidden Dice": fmt3(top5["component_best_dice"].mean()),
                "top10 mean hidden Dice": fmt3(top10["component_best_dice"].mean()),
                "verdict": verdict_from_ranks(ranks, int(top5["high_quality"].sum())),
            }
        )
    return rows


def verdict_from_ranks(ranks: pd.Series, top5_hq: int) -> str:
    r12 = int(ranks.get(12, 999))
    r13 = int(ranks.get(13, 999))
    if r12 <= 5 and r13 <= 5 and top5_hq >= 4:
        return "strong selector signal"
    if r12 <= 10 and r13 <= 10:
        return "promising, but still needs false-positive control"
    if min(r12, r13) <= 10:
        return "partial locator"
    return "weak"


def supervised_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    feature_sets = {
        "H&E bbox/context morphology only": [
            c for c in df.columns if c.startswith("he_bbox_") or c.startswith("he_context_")
        ],
        "H&E morphology + shape no position": [
            c for c in df.columns if c.startswith("he_bbox_") or c.startswith("he_context_")
        ]
        + ["candidate_pixels", "fill_ratio", "bbox_w", "bbox_h"],
        "H&E morphology + FICTURE RGB proxy no position": [
            c for c in df.columns if c.startswith("he_bbox_") or c.startswith("he_context_") or c.startswith("bbox_ficture_")
        ]
        + ["candidate_pixels", "fill_ratio", "bbox_w", "bbox_h"],
        "H&E morphology + shape + diagnostic position": [
            c for c in df.columns if c.startswith("he_bbox_") or c.startswith("he_context_")
        ]
        + ["candidate_pixels", "fill_ratio", "bbox_w", "bbox_h", "cx_norm", "cy_norm"],
    }
    y = df["high_quality"].values
    ids = df["candidate_id"].values
    rows: list[dict[str, object]] = []
    for name, cols in feature_sets.items():
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        rows.append(eval_model(name, "logistic LOOCV", X, y, ids))
        rows.append(eval_model(name, "random forest LOOCV", X, y, ids))
    return rows


def eval_model(name: str, model_name: str, X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> dict[str, object]:
    preds = np.zeros(len(y), dtype=float)
    for train, test in LeaveOneOut().split(X):
        if model_name.startswith("logistic"):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", solver="liblinear", random_state=1),
            )
        else:
            model = RandomForestClassifier(n_estimators=160, max_depth=3, class_weight="balanced", random_state=1)
        model.fit(X[train], y[train])
        preds[test] = model.predict_proba(X[test])[:, 1]
    ap = average_precision_score(y, preds)
    auc = roc_auc_score(y, preds)
    ranks = pd.Series(preds, index=ids).rank(ascending=False, method="min")
    return {
        "feature set": name,
        "model": model_name,
        "LOOCV average precision": fmt3(ap),
        "LOOCV ROC-AUC": fmt3(auc),
        "rank candidate 12": int(ranks.get(12, -1)),
        "rank candidate 13": int(ranks.get(13, -1)),
        "meaning": supervised_meaning(name, ap),
    }


def supervised_meaning(name: str, ap: float) -> str:
    if ap >= 0.75 and "diagnostic position" not in name:
        return "morphology feature family contains useful non-position signal"
    if ap >= 0.75:
        return "predictive but includes ROI position"
    if ap >= 0.45:
        return "some signal, not enough to freeze"
    return "weak signal"


def correlation_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    cols = [
        "he_bbox_bright_low_sat_fraction",
        "he_bbox_very_bright_fraction",
        "he_bbox_tissue_fraction",
        "he_bbox_edge_density",
        "he_bbox_pink_fraction",
        "he_bbox_purple_fraction",
        "he_context_bright_low_sat_fraction",
        "he_context_edge_density",
        "he_airspace_texture_score",
        "he_shape_morphology_score",
        "cx_norm",
        "cy_norm",
        "candidate_pixels",
        "fill_ratio",
    ]
    rows = []
    for col in cols:
        corr = df[col].corr(df["component_best_dice"], method="spearman")
        rows.append(
            {
                "feature": col,
                "Spearman with hidden Dice": fmt3(corr),
                "interpretation": "useful signal" if abs(corr) >= 0.30 else "weak standalone signal",
            }
        )
    rows.sort(key=lambda r: -abs(float(r["Spearman with hidden Dice"])))
    return rows


def make_contact_sheet(df: pd.DataFrame, score_col: str, name: str) -> str:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    he_roi = crop_roi_image(HE_FULL)
    ranked = df.sort_values(score_col, ascending=False).head(12).reset_index(drop=True)
    thumbs = []
    for i, row in ranked.iterrows():
        bbox = parse_bbox(row["bbox"])
        crop = he_roi.crop(bbox).resize((220, 220))
        canvas = Image.new("RGB", (220, 286), "white")
        canvas.paste(crop, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 226),
            f"rank {i+1} | cid {int(row['candidate_id'])}\nscore {row[score_col]:.2f}\nhidden D {row['component_best_dice']:.3f}",
            fill=(20, 30, 45),
        )
        thumbs.append(canvas)
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 220, rows * 286), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 220, (i // cols) * 286))
    out = fig_dir / name
    sheet.save(out)
    return f"alveoli_bbox_he_morphology_audit/figures/{name}"


def decision_rows() -> list[dict[str, object]]:
    return [
        {
            "question": "Was the previous H&E proxy weakness only a missing-file artifact?",
            "answer": "Partly yes.",
            "evidence": "Using the full H&E ROI lets us compute bbox/context morphology for all 100 candidates, not only the 20 saved crops.",
            "next step": "Use these complete bbox/context features as the next alveoli selector diagnostic.",
        },
        {
            "question": "Does broad H&E morphology help without ROI position?",
            "answer": "This audit measures that directly.",
            "evidence": "The report separates texture-only, morphology+shape, and position-aware scores.",
            "next step": "Only a no-position score that ranks true candidates above broad false positives can become a deployable selector.",
        },
        {
            "question": "Can this produce final union Dice now?",
            "answer": "Not yet.",
            "evidence": "This bbox audit does not have the remote binary candidate masks needed to union arbitrary selected candidates.",
            "next step": "Once masks are accessible, recompute selected union D/P/R from this selector.",
        },
    ]


def build_section(rank_rows, supervised, corr, decisions, fig1, fig2) -> str:
    return f"""
<h2>17AN. Complete BBox H&amp;E Morphology Audit For Alveoli</h2>
<p><b>Purpose.</b> The previous audit found that saved close-up crops were incomplete. This section fixes that by extracting bbox-level H&amp;E and context features directly from the official full H&amp;E ROI for all 100 broad-box alveoli candidates. Hidden annotation is used only after scoring to evaluate the feature family.</p>
<h3>Annotation-free H&amp;E morphology ranking</h3>
{table(rank_rows, ['score', 'uses ROI position', 'rank candidate 12', 'rank candidate 13', 'rank FP 82', 'rank FP 90', 'top5 high-quality count', 'top10 high-quality count', 'top5 mean hidden Dice', 'top10 mean hidden Dice', 'verdict'])}
<h3>Supervised feature-sufficiency stress test</h3>
<p>This leave-one-candidate-out test asks whether the feature family contains signal. It is not a deployable trained model because all labels come from the same annotated ROI.</p>
{table(supervised, ['feature set', 'model', 'LOOCV average precision', 'LOOCV ROC-AUC', 'rank candidate 12', 'rank candidate 13', 'meaning'])}
<h3>Single-feature correlations</h3>
{table(corr[:14], ['feature', 'Spearman with hidden Dice', 'interpretation'])}
<h3>Decision</h3>
{table(decisions, ['question', 'answer', 'evidence', 'next step'])}
<figure><img src='{html.escape(fig1)}' alt='top morphology candidates'><figcaption>Top candidates by H&amp;E morphology + shape no-position score.</figcaption></figure>
<figure><img src='{html.escape(fig2)}' alt='top position diagnostic candidates'><figcaption>Top candidates by H&amp;E morphology plus diagnostic ROI position score.</figcaption></figure>
<div class='callout'><b>Decision.</b> This section upgrades the alveoli selector analysis from incomplete saved crops to full-ROI bbox/context features. If the no-position H&amp;E morphology score still admits too many false positives, the next real requirement is full binary mask access for selected-union recompute or a stronger pathology encoder; not another raw FICTURE/VLM prompt sweep.</div>
"""


def append_html(section: str) -> None:
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AN. Complete BBox H&amp;E Morphology Audit For Alveoli</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[O-Z]|<h2>18\\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)


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
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he",
        BASE / "runtime_policy_prototype",
        BASE / "alveoli_broad_box_branch",
        BASE / "alveoli_deployable_selector_gate",
        BASE / "alveoli_feature_sufficiency_audit",
        OUT,
        BASE / "semantic_ficture_proposal_generator",
        BASE / "failure_driven_hybrid_controller",
        BASE / "failure_hierarchy_v4",
        BASE / "immune_he_morphology_branch",
        BASE / "immune_loco_validation",
        BASE / "failure_analysis_protocol_v2",
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
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")


def verify_html_images(html_path: Path) -> None:
    text = html_path.read_text()
    missing = []
    for src in re.findall(r"<img[^>]+src=['\"]([^'\"]+)['\"]", text):
        if src.startswith(("http://", "https://", "data:")):
            continue
        if not (BASE / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"Missing HTML images: {missing[:20]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_scores(build_features())
    df.to_csv(OUT / "alveoli_bbox_he_morphology_features.csv", index=False)
    ranks = ranking_rows(df)
    supervised = supervised_rows(df)
    corr = correlation_rows(df)
    decisions = decision_rows()
    write_csv(OUT / "alveoli_bbox_he_morphology_rank_audit.csv", ranks)
    write_csv(OUT / "alveoli_bbox_he_morphology_supervised_stress_test.csv", supervised)
    write_csv(OUT / "alveoli_bbox_he_morphology_feature_correlations.csv", corr)
    write_csv(OUT / "alveoli_bbox_he_morphology_decision.csv", decisions)
    fig1 = make_contact_sheet(df, "he_shape_morphology_score", "alveoli_he_morphology_shape_top12.png")
    fig2 = make_contact_sheet(df, "he_context_position_diagnostic_score", "alveoli_he_morphology_position_top12.png")
    append_html(build_section(ranks, supervised, corr, decisions, fig1, fig2))
    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "alveoli_bbox_he_morphology_decision.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
