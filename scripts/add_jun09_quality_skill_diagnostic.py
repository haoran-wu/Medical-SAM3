#!/usr/bin/env python3
"""Diagnose whether tissue recognition scores also rank mask quality.

The Jun09 skill ranker originally outputs six tissue scores. This script checks
whether those scores are good enough for assembly, where the real question is:
"among pieces that are the right tissue, which mask is worth selecting?"
"""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "quality_skill_diagnostic"
CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


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


def rank_text(items: list[dict[str, object]], score_key: str = "target_score") -> str:
    parts = []
    for row in items[:5]:
        parts.append(
            f"{row['candidate_uid']} ({score_key}={row[score_key]}, Dice={row['component_dice']:.3f}, P={row['component_precision']:.3f}, R={row['component_recall']:.3f})"
        )
    return "; ".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(BASE / "candidate_skill_features.csv")
    loco = pd.read_csv(BASE / "score_outputs/loco_component_logistic_validation/per_candidate_predictions.csv")
    truth = features.merge(
        loco[["row_key", "predicted_label", "top_score", "top_score_tie", "is_correct", *CLASS_KEYS]],
        on="row_key",
        how="left",
    )
    summary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        target_rows = truth[truth["true_class"] == tissue_class].copy()
        all_rows = truth.copy()
        target_rows["target_score"] = pd.to_numeric(target_rows[tissue_class], errors="coerce").fillna(0)
        all_rows["target_score"] = pd.to_numeric(all_rows[tissue_class], errors="coerce").fillna(0)
        for col in ["component_dice", "component_precision", "component_recall"]:
            target_rows[col] = pd.to_numeric(target_rows[col], errors="coerce").fillna(0)
        if len(target_rows) > 1:
            corr_dice = target_rows["target_score"].corr(target_rows["component_dice"], method="spearman")
            corr_recall = target_rows["target_score"].corr(target_rows["component_recall"], method="spearman")
        else:
            corr_dice = float("nan")
            corr_recall = float("nan")

        ranked_by_score = target_rows.sort_values(["target_score", "component_dice"], ascending=[False, False])
        ranked_by_quality = target_rows.sort_values(["component_dice", "component_recall"], ascending=[False, False])
        best_quality_uid = ranked_by_quality.iloc[0]["candidate_uid"] if len(ranked_by_quality) else ""
        if best_quality_uid:
            score_rank = int(ranked_by_score.reset_index(drop=True).index[ranked_by_score.reset_index(drop=True)["candidate_uid"] == best_quality_uid][0] + 1)
            best_row = ranked_by_quality.iloc[0]
        else:
            score_rank = 0
            best_row = None

        top_all = all_rows.sort_values(["target_score"], ascending=False).head(10)
        false_in_top10 = int((top_all["true_class"] != tissue_class).sum())
        top1_is_false = bool(len(top_all) and top_all.iloc[0]["true_class"] != tissue_class)
        if pd.isna(corr_dice):
            diagnosis = "not estimable"
        elif corr_dice < 0.25 or score_rank > 5:
            diagnosis = "mask-quality ranking failure"
        elif false_in_top10 >= 4:
            diagnosis = "false-positive contamination"
        else:
            diagnosis = "recognition score mostly usable for assembly"

        summary_rows.append(
            {
                "class": tissue_class,
                "candidate pieces": len(target_rows),
                "Spearman(score, Dice)": f"{corr_dice:.3f}" if not pd.isna(corr_dice) else "NA",
                "Spearman(score, Recall)": f"{corr_recall:.3f}" if not pd.isna(corr_recall) else "NA",
                "best quality candidate": best_quality_uid,
                "best quality candidate score-rank": score_rank,
                "best quality D/P/R": (
                    f"{best_row['component_dice']:.3f}/{best_row['component_precision']:.3f}/{best_row['component_recall']:.3f}"
                    if best_row is not None
                    else ""
                ),
                "false candidates in top10 target score": false_in_top10,
                "top1 target-score is false": top1_is_false,
                "diagnosis": diagnosis,
                "needed skill refinement": (
                    "Add a mask-quality score trained/calibrated to predict component Dice/Recall, then multiply or gate tissue score by quality."
                    if diagnosis == "mask-quality ranking failure"
                    else "Add class-specific veto/NMS or generator repair before more prompt tuning."
                    if diagnosis == "false-positive contamination"
                    else "Keep current scorer; focus on assembly/generator validation."
                ),
            }
        )

        for rank, row in enumerate(ranked_by_score.head(8).to_dict("records"), 1):
            top_rows.append(
                {
                    "class": tissue_class,
                    "list": "top by tissue score among true pieces",
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "target_score": int(row["target_score"]),
                    "component_dice": f"{row['component_dice']:.3f}",
                    "component_precision": f"{row['component_precision']:.3f}",
                    "component_recall": f"{row['component_recall']:.3f}",
                }
            )
        for rank, row in enumerate(ranked_by_quality.head(8).to_dict("records"), 1):
            top_rows.append(
                {
                    "class": tissue_class,
                    "list": "top by hidden mask quality",
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "target_score": int(row["target_score"]),
                    "component_dice": f"{row['component_dice']:.3f}",
                    "component_precision": f"{row['component_precision']:.3f}",
                    "component_recall": f"{row['component_recall']:.3f}",
                }
            )

    write_csv(OUT / "quality_skill_summary_by_class.csv", summary_rows)
    write_csv(OUT / "quality_skill_top_candidates.csv", top_rows)

    section = f"""
<h2>17I. Mask-Quality Skill Diagnostic</h2>
<p>This is the most important refinement after the first skill-ranker pass. A six-class tissue score answers <i>what tissue is this piece?</i>, but assembly needs a second question: <i>is this piece a good mask worth adding to the final union?</i> A correct tissue label can still be a low-quality mask if it covers only a tiny wall segment, has poor boundary, or misses the component.</p>
<p>Here, annotation is used only after scoring to audit whether the tissue score is correlated with hidden component Dice/Recall. Low correlation means the next model should explicitly learn a <b>mask-quality skill</b>, not just a tissue-recognition skill.</p>
{table(summary_rows, ['class', 'candidate pieces', 'Spearman(score, Dice)', 'Spearman(score, Recall)', 'best quality candidate', 'best quality candidate score-rank', 'best quality D/P/R', 'false candidates in top10 target score', 'diagnosis', 'needed skill refinement'])}
<h3>Why this matters for selection</h3>
{table(top_rows[:96], ['class', 'list', 'rank', 'candidate_uid', 'target_score', 'component_dice', 'component_precision', 'component_recall'])}
<div class='callout'><b>Decision.</b> The next skill ranker should output two scores per class: tissue membership and mask quality. Assembly should rank by a combined score, for example <code>class_score * quality_score</code>, with class-specific NMS. This is closer to a usable candidate ranker than asking a VLM for only six class probabilities.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17I. Mask-Quality Skill Diagnostic</h2>"
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

    print(OUT / "quality_skill_summary_by_class.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
