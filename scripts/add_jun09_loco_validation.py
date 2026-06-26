#!/usr/bin/env python3
"""Add leave-one-component-out validation to the Jun09 report."""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
REFINE_DIR = BASE / "refined_diagnostics"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def predicted(scores: dict[str, int]) -> tuple[str, int, bool]:
    max_score = max(scores.values())
    winners = [c for c, s in scores.items() if s == max_score]
    return winners[0], max_score, len(winners) > 1


def html_table(rows: list[dict[str, object]], cols: list[str]) -> str:
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
    rows = read_csv(BASE / "candidate_skill_features.csv")
    feature_names: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith(("ficture_", "shape_", "frac_", "ring_frac_", "he_clip_large_")) or key in {
                "area_frac",
                "bbox_aspect",
                "bbox_area_frac",
                "solidity",
                "eccentricity",
                "thinness",
                "centroid_x_frac",
                "centroid_y_frac",
                "compact_funnel_score",
                "compact_funnel_important_score",
            }:
                try:
                    float(value)
                except ValueError:
                    continue
                if key not in feature_names:
                    feature_names.append(key)
    X = np.array([[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows])
    y = np.array([CLASS_KEYS.index(row["true_class"]) for row in rows])
    groups = np.array([f"{row['true_class']}::{row.get('matched_annotation_component_id','')}" for row in rows])
    weights = np.array([0.25 + float(row.get("component_dice", 0.0) or 0.0) for row in rows])
    unique_groups = sorted(set(groups))

    pred_rows: list[dict[str, object]] = []
    for group in unique_groups:
        test_idx = np.flatnonzero(groups == group)
        train_idx = np.flatnonzero(groups != group)
        train_classes = set(y[train_idx])
        true_class = group.split("::", 1)[0]
        estimable = CLASS_KEYS.index(true_class) in train_classes
        if len(train_classes) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=9),
        )
        model.fit(X[train_idx], y[train_idx], logisticregression__sample_weight=weights[train_idx])
        proba = model.predict_proba(X[test_idx])
        classes = list(model.classes_)
        for local_i, row_index in enumerate(test_idx):
            scores = {c: 0 for c in CLASS_KEYS}
            for class_index, p in zip(classes, proba[local_i]):
                scores[CLASS_KEYS[int(class_index)]] = int(round(100 * float(p)))
            pred, top, tie = predicted(scores)
            pred_rows.append(
                {
                    "row_key": rows[row_index]["row_key"],
                    "candidate_uid": rows[row_index]["candidate_uid"],
                    "true_class": rows[row_index]["true_class"],
                    "heldout_group": group,
                    "heldout_class_estimable": str(estimable).lower(),
                    "predicted_class": pred,
                    "top_score": top,
                    "tie": str(tie).lower(),
                    "correct": str((pred == rows[row_index]["true_class"]) and not tie).lower(),
                    **scores,
                }
            )
    write_csv(REFINE_DIR / "loco_component_validation_predictions.csv", pred_rows)

    summary: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        subset = [row for row in pred_rows if row["true_class"] == c]
        estimable = [row for row in subset if row["heldout_class_estimable"] == "true"]
        all_correct = sum(row["correct"] == "true" for row in subset)
        est_correct = sum(row["correct"] == "true" for row in estimable)
        summary.append(
            {
                "class": c,
                "LOCO all rows": f"{all_correct}/{len(subset)}",
                "LOCO estimable rows": f"{est_correct}/{len(estimable)}",
                "note": "not estimable means this class had no other component left for training in that fold" if len(estimable) < len(subset) else "",
            }
        )
    write_csv(REFINE_DIR / "loco_component_validation_summary.csv", summary)

    section = f"""
<h2>11. Stability Check: Leave-One-Component-Out</h2>
<p>The high in-sample ranker score can overfit one annotated ROI. To test stability, this check holds out all candidates from one annotation component, trains on the remaining components, and predicts the held-out component. If a class has only one component in this 167-piece pool, that fold cannot estimate that class fairly.</p>
{html_table(summary, ['class', 'LOCO all rows', 'LOCO estimable rows', 'note'])}
<div class='callout warn'><b>Interpretation.</b> This is stricter than the in-sample RF+HE result. A class with poor LOCO performance should not be claimed as solved; it needs either more annotated ROIs, better non-annotation heuristics, or a frozen rule based on interpretable morphology instead of a trained ranker.</div>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>11. Stability Check: Leave-One-Component-Out</h2>"
        if marker in text:
            start = text.index(marker)
            if "<h2>11. Interpretation</h2>" in text[start:]:
                end = text.index("<h2>11. Interpretation</h2>", start)
                text = text[:start] + section + text[end:]
            elif "<h2>12. Interpretation</h2>" in text[start:]:
                end = text.index("<h2>12. Interpretation</h2>", start)
                text = text[:start] + section + text[end:]
        elif "<h2>11. Interpretation</h2>" in text:
            text = text.replace("<h2>11. Interpretation</h2>", section + "<h2>12. Interpretation</h2>")
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    print(REFINE_DIR / "loco_component_validation_summary.csv")


if __name__ == "__main__":
    main()
