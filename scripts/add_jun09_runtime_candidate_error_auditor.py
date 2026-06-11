#!/usr/bin/env python3
"""Add a candidate-level runtime error auditor to the Jun09 skill report.

The previous debugger identifies the first failing layer by class.  This script
drills one level deeper: which selected candidates are false positives, which
high-quality true candidates were missed, and which deployable skill signals
look responsible.  It does not train a new model and it does not use hidden
annotation at selection time; hidden labels are only used here for diagnosis.
"""

from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "runtime_candidate_error_auditor"
CLASSES = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}


def norm_class(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def td(value: object) -> str:
    return f"<td>{html.escape(fmt(value))}</td>"


def simple_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(c)}</th>" for c in columns)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in columns:
            out.append(td(row.get(col, "")))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def candidate_card(row: dict[str, object], target_class: str) -> str:
    he_rel = f"corrected_pool/{row.get('he_crop_rel', '')}"
    fig_rel = f"corrected_pool/{row.get('ficture_crop_rel', '')}"
    uid = html.escape(str(row.get("candidate_uid", "")))
    true_cls = html.escape(str(row.get("true_class", row.get("true_class_hidden", ""))))
    status = html.escape(str(row.get("status", "")))
    return f"""
<div class='candidate-card'>
  <div class='candidate-title'>{uid}</div>
  <div class='candidate-meta'>target={html.escape(DISPLAY[target_class])} | true={true_cls} | {status}</div>
  <div class='candidate-imgs'>
    <figure><img src='{html.escape(he_rel)}' alt='{uid} H&amp;E'><figcaption>H&amp;E piece</figcaption></figure>
    <figure><img src='{html.escape(fig_rel)}' alt='{uid} FICTURE'><figcaption>FICTURE piece</figcaption></figure>
  </div>
  <div class='candidate-meta'>
    rank={html.escape(fmt(row.get('rank')))} |
    runtime score={html.escape(fmt(row.get('runtime_score')))} |
    component Dice={html.escape(fmt(row.get('component_dice')))} |
    P/R={html.escape(fmt(row.get('component_precision')))}/{html.escape(fmt(row.get('component_recall')))}
  </div>
  <div class='candidate-meta'>
    H&amp;E skill={html.escape(fmt(row.get('target_he_skill')))} |
    FICTURE prior={html.escape(fmt(row.get('target_ficture_prior')))} |
    shape skill={html.escape(fmt(row.get('target_shape_skill')))} |
    area frac={html.escape(fmt(row.get('area_frac'), 4))}
  </div>
</div>
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
        path = BASE / rel
        if path.exists():
            include.append(path)
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


def load_runtime_rows(features: pd.DataFrame, selected: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_uid = features.set_index("candidate_uid", drop=False)
    selected = selected.copy()
    selected["class_norm"] = selected["class"].map(norm_class)
    selected_rows: list[dict[str, object]] = []
    false_positive_rows: list[dict[str, object]] = []
    missed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for target in CLASSES:
        he_col = f"he_clip_large_{target}"
        fig_col = f"ficture_prior_{target}"
        shape_col = f"shape_skill_{target}"

        cls_selected = selected[selected["class_norm"] == target].copy()
        cls_selected_uids = set(cls_selected["candidate_uid"])
        selected_true = 0
        selected_wrong = 0

        for _, sel in cls_selected.iterrows():
            uid = sel["candidate_uid"]
            if uid not in by_uid.index:
                continue
            feat = by_uid.loc[uid]
            row = {
                "class": target,
                "rank": int(sel["rank"]),
                "candidate_uid": uid,
                "status": "selected_true" if feat["true_class"] == target else "selected_false_positive",
                "runtime_score": float(sel["score"]),
                "threshold": float(sel["threshold"]),
                "true_class": feat["true_class"],
                "component_dice": float(feat["component_dice"]),
                "component_precision": float(feat["component_precision"]),
                "component_recall": float(feat["component_recall"]),
                "target_he_skill": float(feat[he_col]),
                "target_ficture_prior": float(feat[fig_col]),
                "target_shape_skill": float(feat[shape_col]),
                "area_frac": float(feat["area_frac"]),
                "thinness": float(feat["thinness"]),
                "solidity": float(feat["solidity"]),
                "bbox_aspect": float(feat["bbox_aspect"]),
                "factor_composition_top": feat.get("factor_composition_top", ""),
                "he_crop_rel": feat["he_crop_rel"],
                "ficture_crop_rel": feat["ficture_crop_rel"],
            }
            selected_rows.append(row)
            if feat["true_class"] == target:
                selected_true += 1
            else:
                selected_wrong += 1
                false_positive_rows.append(row)

        cls_true = features[features["true_class"] == target].sort_values("component_dice", ascending=False).head(10)
        missed_count = 0
        top_true_selected = 0
        for _, feat in cls_true.iterrows():
            uid = feat["candidate_uid"]
            status = "top_true_selected" if uid in cls_selected_uids else "missed_top_true"
            if status == "top_true_selected":
                top_true_selected += 1
            else:
                missed_count += 1
            missed_rows.append(
                {
                    "class": target,
                    "candidate_uid": uid,
                    "status": status,
                    "runtime_score": float(cls_selected[cls_selected["candidate_uid"] == uid]["score"].iloc[0]) if uid in cls_selected_uids else "",
                    "rank": int(cls_selected[cls_selected["candidate_uid"] == uid]["rank"].iloc[0]) if uid in cls_selected_uids else "",
                    "true_class": feat["true_class"],
                    "component_dice": float(feat["component_dice"]),
                    "component_precision": float(feat["component_precision"]),
                    "component_recall": float(feat["component_recall"]),
                    "target_he_skill": float(feat[he_col]),
                    "target_ficture_prior": float(feat[fig_col]),
                    "target_shape_skill": float(feat[shape_col]),
                    "area_frac": float(feat["area_frac"]),
                    "thinness": float(feat["thinness"]),
                    "solidity": float(feat["solidity"]),
                    "bbox_aspect": float(feat["bbox_aspect"]),
                    "factor_composition_top": feat.get("factor_composition_top", ""),
                    "he_crop_rel": feat["he_crop_rel"],
                    "ficture_crop_rel": feat["ficture_crop_rel"],
                }
            )

        if selected_wrong:
            first_error = "selection allows cross-class false positives"
        elif missed_count:
            first_error = "selector misses high-Dice true pieces"
        elif target == "alveoli":
            first_error = "selected top pieces exist but proposal family is too narrow"
        else:
            first_error = "candidate set selected cleanly; next check boundary/refinement"

        summary_rows.append(
            {
                "class": DISPLAY[target],
                "runtime selected": len(cls_selected),
                "selected true": selected_true,
                "selected false positives": selected_wrong,
                "top-10 true pieces selected": top_true_selected,
                "top-10 true pieces missed": missed_count,
                "best available component Dice": round(float(cls_true["component_dice"].max()), 3) if len(cls_true) else "",
                "first concrete error": first_error,
                "next skill change": next_skill_change(target, selected_wrong, missed_count),
            }
        )

    return summary_rows, selected_rows, false_positive_rows, missed_rows


def next_skill_change(target: str, selected_wrong: int, missed_count: int) -> str:
    if target == "alveoli":
        return "change proposal branch first: add broad H&E box384 candidates, then rerun component oracle"
    if selected_wrong > 0:
        return "add one-class-only assignment and false-positive veto before union"
    if missed_count > 0 and target in {"bronchiola", "vessels", "immune_infiltration"}:
        return "loosen recall only after low-overlap/NMS check, then test second-SAM locator refinement"
    if missed_count > 0:
        return "add class-specific assembly threshold; avoid one global max-piece rule"
    return "freeze selector for this class and validate/refine boundary"


def feature_delta_rows(selected_rows: list[dict[str, object]], missed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected_df = pd.DataFrame(selected_rows)
    missed_df = pd.DataFrame(missed_rows)
    for target in CLASSES:
        groups = {
            "selected_true": selected_df[(selected_df["class"] == target) & (selected_df["status"] == "selected_true")],
            "selected_false_positive": selected_df[(selected_df["class"] == target) & (selected_df["status"] == "selected_false_positive")],
            "missed_top_true": missed_df[(missed_df["class"] == target) & (missed_df["status"] == "missed_top_true")],
        }
        for group_name, group in groups.items():
            if group.empty:
                rows.append({"class": DISPLAY[target], "group": group_name, "n": 0})
                continue
            rows.append(
                {
                    "class": DISPLAY[target],
                    "group": group_name,
                    "n": len(group),
                    "mean component Dice": round(float(group["component_dice"].mean()), 3),
                    "mean H&E skill": round(float(group["target_he_skill"].mean()), 3),
                    "mean FICTURE prior": round(float(group["target_ficture_prior"].mean()), 3),
                    "mean shape skill": round(float(group["target_shape_skill"].mean()), 3),
                    "mean area frac": round(float(group["area_frac"].mean()), 5),
                    "mean thinness": round(float(group["thinness"].mean()), 3),
                    "mean solidity": round(float(group["solidity"].mean()), 3),
                }
            )
    return rows


def class_section(target: str, selected_rows: list[dict[str, object]], false_positive_rows: list[dict[str, object]], missed_rows: list[dict[str, object]]) -> str:
    sel = [r for r in selected_rows if r["class"] == target]
    fp = [r for r in false_positive_rows if r["class"] == target][:4]
    missed = [r for r in missed_rows if r["class"] == target and r["status"] == "missed_top_true"][:4]
    selected_cards = "".join(candidate_card(row, target) for row in sel[:6])
    fp_cards = "".join(candidate_card(row, target) for row in fp) if fp else "<p class='muted'>No selected false positives in the runtime policy for this class.</p>"
    missed_cards = "".join(candidate_card(row, target) for row in missed) if missed else "<p class='muted'>No missed top-10 true pieces under this runtime policy.</p>"
    return f"""
<details open>
  <summary><b>{html.escape(DISPLAY[target])}</b>: selected pieces, false positives, and missed top true pieces</summary>
  <h4>Runtime selected pieces (first six shown)</h4>
  <div class='candidate-grid'>{selected_cards}</div>
  <h4>Selected false positives</h4>
  <div class='candidate-grid'>{fp_cards}</div>
  <h4>High-quality true pieces missed by runtime selection</h4>
  <div class='candidate-grid'>{missed_cards}</div>
</details>
"""


def append_report(section: str) -> None:
    marker = "<h2>17W. Runtime Candidate Error Auditor: Exact Selected And Missed Pieces</h2>"
    style = """
<style>
.candidate-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;margin:10px 0 18px}
.candidate-card{border:1px solid #dde3ea;border-radius:8px;padding:10px;background:#fff}
.candidate-title{font-weight:700;color:#1f2937;word-break:break-all}
.candidate-meta{font-size:12px;color:#52606d;margin:4px 0}
.candidate-imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px;align-items:start}
.candidate-imgs figure{margin:0}
.candidate-imgs img{width:100%;max-height:220px;object-fit:contain;border:1px solid #e5e7eb;background:#f8fafc}
.candidate-imgs figcaption{font-size:11px;color:#667085;margin-top:3px}
</style>
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if ".candidate-grid" not in text:
            text = text.replace("</head>", style + "</head>")
        if marker in text:
            start = text.index(marker)
            end = text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(BASE / "candidate_skill_features.csv")
    runtime_selected = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_selected_pieces.csv")

    summary_rows, selected_rows, false_positive_rows, missed_rows = load_runtime_rows(features, runtime_selected)
    delta_rows = feature_delta_rows(selected_rows, missed_rows)

    write_csv(OUT / "runtime_selection_summary_by_class.csv", summary_rows)
    write_csv(OUT / "runtime_selected_piece_audit.csv", selected_rows)
    write_csv(OUT / "runtime_selected_false_positives.csv", false_positive_rows)
    write_csv(OUT / "runtime_missed_top_true_pieces.csv", missed_rows)
    write_csv(OUT / "runtime_feature_delta_by_error_type.csv", delta_rows)

    section = f"""
<h2>17W. Runtime Candidate Error Auditor: Exact Selected And Missed Pieces</h2>
<p><b>Goal.</b> This is the candidate-level failure microscope for the current deployable runtime policy. It asks: after the skill assigns H&amp;E, FICTURE-prior, shape, and composition scores, which exact pieces did the policy select, which selected pieces are wrong, and which high-quality true pieces were left out?</p>
<p><b>Important guardrail.</b> Hidden annotation labels and Dice/Precision/Recall are used only here for post-hoc diagnosis. They are not used by the runtime selector. This section is meant to tell us which skill layer to change next, not to claim a deployable final result.</p>
<h3>Class-level candidate error summary</h3>
{simple_table(summary_rows, ['class', 'runtime selected', 'selected true', 'selected false positives', 'top-10 true pieces selected', 'top-10 true pieces missed', 'best available component Dice', 'first concrete error', 'next skill change'])}
<h3>Feature pattern behind selected and missed pieces</h3>
<p>If selected false positives have high H&amp;E/FICTURE/shape skill but low hidden component Dice, the failure is a <b>quality skill</b> failure. If top true pieces are missed despite high component Dice, the failure is an <b>assembly or threshold</b> failure. If the best available component Dice is low, the failure is a <b>proposal-generator</b> failure.</p>
{simple_table(delta_rows, ['class', 'group', 'n', 'mean component Dice', 'mean H&E skill', 'mean FICTURE prior', 'mean shape skill', 'mean area frac', 'mean thinness', 'mean solidity'])}
<h3>Candidate-level microscope</h3>
{''.join(class_section(cls, selected_rows, false_positive_rows, missed_rows) for cls in CLASSES)}
<div class='callout'><b>17W decision.</b> The next skill iteration should not be a larger prompt. For alveoli, change the proposal generator. For vessels/tumor/immune, improve deployable quality scoring and one-class assignment. For bronchiola and vessels, run second-SAM locator refinement only after keeping the current selected-piece evidence fixed.</div>
"""

    append_report(section)
    rebuild_zip()

    print(OUT / "runtime_selection_summary_by_class.csv")
    print(OUT / "runtime_selected_piece_audit.csv")
    print(OUT / "runtime_selected_false_positives.csv")
    print(OUT / "runtime_missed_top_true_pieces.csv")
    print(OUT / "runtime_feature_delta_by_error_type.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
