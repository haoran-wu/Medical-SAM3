#!/usr/bin/env python3
"""Append an alveoli feature-sufficiency audit to the Jun09 report.

The deployable-selector gate showed that candidates 12/13 can rank high, but
the working score still included a one-ROI spatial prior.  This audit asks a
more precise failure-analysis question:

Can annotation-free features identify good broad alveoli candidates without
relying on ROI position?  If not, the next layer to improve is a true H&E
morphology/context encoder or retrieval of full candidate masks/images.
"""

from __future__ import annotations

import csv
import html
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_feature_sufficiency_audit"


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


def fmt3(value: object) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "NA"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def load_data() -> pd.DataFrame:
    p = BASE / "alveoli_deployable_selector_gate/alveoli_deployable_selector_candidate_scores.csv"
    df = pd.read_csv(p)
    # Hidden labels are evaluation-only.  They are not part of any deployable
    # selector; here they define whether a feature set has any diagnostic power.
    df["high_quality"] = (df["component_best_dice"] >= 0.20).astype(int)
    df["target_pair"] = df["candidate_id"].isin([12, 13]).astype(int)
    return df


def feature_sets() -> dict[str, list[str]]:
    ficture_cols = [
        "inside_alveolar_AT2_F3_F5",
        "inside_tumor_epithelial_F0_F2",
        "inside_stroma_endothelial_F1_F9",
        "inside_airway_epithelial_F7",
        "inside_immune_F4_F6_F8_F10_F11",
        "local_alveolar_AT2_F3_F5",
        "local_tumor_epithelial_F0_F2",
        "local_stroma_endothelial_F1_F9",
        "local_airway_epithelial_F7",
        "local_immune_F4_F6_F8_F10_F11",
    ]
    shape_no_pos = ["candidate_pixels", "fill_ratio"]
    # Some candidate bbox features were not saved in the previous gate CSV, so
    # this audit uses the available stable fields plus derived score columns.
    position = ["shape_location_score", "lower_left_broad_score"]
    he_proxy = ["he_color_fraction", "he_bright_fraction", "he_pink_fraction"]
    return {
        "FICTURE composition only": ficture_cols,
        "shape no position": shape_no_pos,
        "position diagnostic only": position,
        "FICTURE + shape no position": ficture_cols + shape_no_pos,
        "combined no position plus available H&E proxies": ficture_cols + shape_no_pos + he_proxy,
        "combined all diagnostic": ficture_cols + shape_no_pos + he_proxy + position,
    }


def rank_metric_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, cols in feature_sets().items():
        score = simple_score(df, name, cols)
        ranked = df.assign(score=score).sort_values("score", ascending=False).reset_index(drop=True)
        ranks = ranked.reset_index().set_index("candidate_id")["index"].add(1)
        top5 = ranked.head(5)
        top10 = ranked.head(10)
        rows.append(
            {
                "feature set": name,
                "uses ROI position": "yes" if uses_roi_position(name) else "no",
                "uses incomplete H&E image pack": "yes" if uses_incomplete_he_pack(name) else "no",
                "rank candidate 12": int(ranks.get(12, -1)),
                "rank candidate 13": int(ranks.get(13, -1)),
                "rank false-positive 82": int(ranks.get(82, -1)),
                "rank false-positive 90": int(ranks.get(90, -1)),
                "top5 high-quality count": int(top5["high_quality"].sum()),
                "top10 high-quality count": int(top10["high_quality"].sum()),
                "top5 mean hidden Dice": fmt3(top5["component_best_dice"].mean()),
                "top10 mean hidden Dice": fmt3(top10["component_best_dice"].mean()),
                "verdict": rank_verdict(name, ranks, int(top5["high_quality"].sum())),
            }
        )
    return rows


def simple_score(df: pd.DataFrame, name: str, cols: list[str]) -> pd.Series:
    """Annotation-free heuristic score per feature set.

    This is not trained on hidden labels.  It is a deterministic diagnostic of
    whether the available feature family contains the needed signal.
    """

    if name == "FICTURE composition only":
        return (
            0.8 * (df["local_alveolar_AT2_F3_F5"] / 20).clip(0, 1)
            + 0.3 * (df["inside_tumor_epithelial_F0_F2"] / 65).clip(0, 1)
            - 1.5 * (df["inside_airway_epithelial_F7"] / 25).clip(0, 1)
            - 1.2 * (df["inside_immune_F4_F6_F8_F10_F11"] / 55).clip(0, 1)
            - 1.0 * (df["inside_stroma_endothelial_F1_F9"] / 25).clip(0, 1)
        )
    if name == "shape no position":
        return triangle(df["candidate_pixels"], 460_000, 220_000) + 0.7 * triangle(df["fill_ratio"], 0.65, 0.35)
    if name == "position diagnostic only":
        return 0.55 * df["lower_left_broad_score"] + 0.45 * df["shape_location_score"]
    if name == "FICTURE + shape no position":
        return simple_score(df, "FICTURE composition only", []) + 0.7 * simple_score(df, "shape no position", [])
    if name == "combined no position plus available H&E proxies":
        he = (
            0.6 * triangle(df["he_color_fraction"].fillna(0), 0.38, 0.22)
            + 0.5 * triangle(df["he_bright_fraction"].fillna(0), 0.48, 0.28)
            + 0.3 * triangle(df["he_pink_fraction"].fillna(0), 0.33, 0.20)
        )
        return simple_score(df, "FICTURE + shape no position", []) + he * df["he_image_available"].fillna(0)
    if name == "combined all diagnostic":
        return (
            simple_score(df, "combined no position plus available H&E proxies", [])
            + 0.8 * simple_score(df, "position diagnostic only", [])
        )
    raise ValueError(name)


def triangle(series: pd.Series, center: float, width: float) -> pd.Series:
    return (1 - (series - center).abs() / width).clip(0, 1)


def rank_verdict(name: str, ranks: pd.Series, top5_high_quality_count: int) -> str:
    r12 = int(ranks.get(12, 999))
    r13 = int(ranks.get(13, 999))
    if r12 <= 5 and r13 <= 5 and not uses_roi_position(name) and not uses_incomplete_he_pack(name):
        if top5_high_quality_count >= 4:
            return "strong: target pair ranks high without ROI position"
        return "promising locator: target pair ranks high, but broad false positives remain"
    if r12 <= 5 and r13 <= 5 and uses_incomplete_he_pack(name):
        return "diagnostic only: works, but H&E crop coverage is incomplete"
    if r12 <= 5 and r13 <= 5:
        return "diagnostic only: works, but depends on ROI position"
    if min(r12, r13) <= 10:
        return "partial: finds one target but not a robust selector"
    return "weak: does not recover the target pair"


def uses_roi_position(feature_name: str) -> bool:
    return feature_name in {"position diagnostic only", "combined all diagnostic"}


def uses_incomplete_he_pack(feature_name: str) -> bool:
    return "H&E proxies" in feature_name


def supervised_sufficiency_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y = df["high_quality"].values
    for name, cols in feature_sets().items():
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        rows.append(evaluate_model(name, "logistic LOOCV", X, y, df["candidate_id"].values))
        rows.append(evaluate_model(name, "random forest LOOCV", X, y, df["candidate_id"].values))
    return rows


def evaluate_model(feature_name: str, model_name: str, X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> dict[str, object]:
    if len(np.unique(y)) < 2:
        return {
            "feature set": feature_name,
            "model": model_name,
            "LOOCV average precision": "NA",
            "LOOCV ROC-AUC": "NA",
            "rank candidate 12": "NA",
            "rank candidate 13": "NA",
            "meaning": "not enough labels",
        }
    preds = np.zeros(len(y), dtype=float)
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        if model_name.startswith("logistic"):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", solver="liblinear", random_state=0),
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                class_weight="balanced",
                random_state=0,
            )
        model.fit(X[train], y[train])
        if hasattr(model, "predict_proba"):
            preds[test] = model.predict_proba(X[test])[:, 1]
        else:
            preds[test] = model.decision_function(X[test])
    ap = average_precision_score(y, preds)
    try:
        auc = roc_auc_score(y, preds)
    except Exception:
        auc = float("nan")
    rank_order = pd.Series(preds, index=ids).rank(ascending=False, method="min")
    return {
        "feature set": feature_name,
        "model": model_name,
        "LOOCV average precision": fmt3(ap),
        "LOOCV ROC-AUC": fmt3(auc),
        "rank candidate 12": int(rank_order.get(12, -1)),
        "rank candidate 13": int(rank_order.get(13, -1)),
        "meaning": supervised_meaning(feature_name, ap),
    }


def supervised_meaning(feature_name: str, ap: float) -> str:
    if ap >= 0.7 and not uses_roi_position(feature_name) and not uses_incomplete_he_pack(feature_name):
        return "feature family may be enough without location"
    if ap >= 0.7 and uses_incomplete_he_pack(feature_name):
        return "predictive but H&E crop coverage is incomplete"
    if ap >= 0.7:
        return "predictive but may be one-ROI/location-biased"
    if ap >= 0.45:
        return "some signal, not reliable enough"
    return "weak signal"


def correlation_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    cols = [
        "candidate_pixels",
        "fill_ratio",
        "inside_alveolar_AT2_F3_F5",
        "inside_tumor_epithelial_F0_F2",
        "inside_airway_epithelial_F7",
        "inside_immune_F4_F6_F8_F10_F11",
        "inside_stroma_endothelial_F1_F9",
        "local_alveolar_AT2_F3_F5",
        "local_immune_F4_F6_F8_F10_F11",
        "shape_location_score",
        "lower_left_broad_score",
        "he_color_fraction",
        "he_bright_fraction",
        "he_pink_fraction",
    ]
    rows = []
    for col in cols:
        if col not in df:
            continue
        valid = df[[col, "component_best_dice"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 5:
            corr = float("nan")
        else:
            corr = valid[col].corr(valid["component_best_dice"], method="spearman")
        rows.append(
            {
                "feature": col,
                "Spearman with hidden Dice": fmt3(corr),
                "available rows": len(valid),
                "interpretation": corr_interpretation(col, corr, len(valid)),
            }
        )
    rows.sort(key=lambda r: -abs(float(r["Spearman with hidden Dice"]) if r["Spearman with hidden Dice"] != "NA" else 0))
    return rows


def corr_interpretation(col: str, corr: float, n: int) -> str:
    if n < 50 and col.startswith("he_"):
        return "H&E proxy is incomplete in the local pack"
    if not np.isfinite(corr):
        return "not enough data"
    if abs(corr) >= 0.45:
        return "strong diagnostic signal, but check whether it is location/proxy leakage"
    if abs(corr) >= 0.25:
        return "weak-to-moderate signal"
    return "little standalone signal"


def decision_rows() -> list[dict[str, object]]:
    return [
        {
            "question": "Can current structured FICTURE composition alone solve alveoli?",
            "answer": "No.",
            "evidence": "It does not rank 12/13 together in the top five, because true alveoli candidates in this tumor section are not simply high-alveolar-color regions.",
            "next step": "Use FICTURE as veto/support only.",
        },
        {
            "question": "Can non-position features replace the diagnostic selector?",
            "answer": "Partly, but not enough.",
            "evidence": "Area/fill features can rank 12/13 high, but still admit broad false positives; H&E proxy looks strong only on the 20 saved crops, so it needs a complete image/mask pack before it can be trusted.",
            "next step": "Build or retrieve full broad-candidate H&E/context crops and masks for a real morphology encoder.",
        },
        {
            "question": "Is the current selector deployable?",
            "answer": "No, not frozen.",
            "evidence": "It can rank the target pair high with area/fill and diagnostic spatial cues, but false positives remain and the strongest H&E-proxy signal is based on only 20 saved crops.",
            "next step": "The next real improvement is a broad-candidate morphology/context encoder or second-SAM selector, then rerun union Dice/P/R.",
        },
    ]


def build_section(rank_rows, supervised_rows, corr_rows, decisions) -> str:
    return f"""
<h2>17AM. Alveoli Feature-Sufficiency Audit</h2>
<p><b>Purpose.</b> Section 17AK showed that a diagnostic selector can rank the useful broad alveoli candidates high. This audit asks whether that success comes from real tissue/mask features or from one-ROI spatial bias. Hidden Dice is used only as an evaluation label in this audit.</p>
<h3>Annotation-free heuristic feature sets</h3>
{table(rank_rows, ['feature set', 'uses ROI position', 'uses incomplete H&E image pack', 'rank candidate 12', 'rank candidate 13', 'rank false-positive 82', 'rank false-positive 90', 'top5 high-quality count', 'top10 high-quality count', 'top5 mean hidden Dice', 'top10 mean hidden Dice', 'verdict'])}
<h3>Supervised feature-sufficiency stress test</h3>
<p>This leave-one-candidate-out test intentionally uses hidden labels only to test whether the feature family contains enough signal. It is not a deployable trained model because all labels come from one ROI.</p>
{table(supervised_rows, ['feature set', 'model', 'LOOCV average precision', 'LOOCV ROC-AUC', 'rank candidate 12', 'rank candidate 13', 'meaning'])}
<h3>Single-feature correlations</h3>
{table(corr_rows[:14], ['feature', 'Spearman with hidden Dice', 'available rows', 'interpretation'])}
<h3>Decision</h3>
{table(decisions, ['question', 'answer', 'evidence', 'next step'])}
<div class='callout'><b>Decision.</b> The current features are enough to explain why the broad H&amp;E proposal branch matters, but they are not enough to freeze a deployable alveoli selector. The next iteration should create a complete broad-candidate morphology/context feature table from the full masks/crops, then recompute the selected union. If remote masks remain inaccessible, this layer remains blocked, not solved.</div>
"""


def append_html(section: str) -> None:
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AM. Alveoli Feature-Sufficiency Audit</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[N-Z]|<h2>18\\.", text[start + len(marker) :])
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
    df = load_data()
    rank_rows = rank_metric_rows(df)
    supervised_rows = supervised_sufficiency_rows(df)
    corr_rows = correlation_rows(df)
    decisions = decision_rows()

    write_csv(OUT / "alveoli_feature_set_rank_audit.csv", rank_rows)
    write_csv(OUT / "alveoli_supervised_sufficiency_stress_test.csv", supervised_rows)
    write_csv(OUT / "alveoli_single_feature_correlations.csv", corr_rows)
    write_csv(OUT / "alveoli_feature_sufficiency_decision.csv", decisions)

    append_html(build_section(rank_rows, supervised_rows, corr_rows, decisions))
    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "alveoli_feature_sufficiency_decision.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
