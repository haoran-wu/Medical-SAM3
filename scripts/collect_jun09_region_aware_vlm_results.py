#!/usr/bin/env python3
"""Collect Jun09 region-aware all6 VLM results into the skill-ranker report.

This is deliberately a result-gate script, not only a copy script.  It checks
whether a completed VLM run is interpretable before it is allowed to become a
method result in the report:

1. parsing / JSON validity,
2. score diversity and class collapse,
3. piece-level top1 by class,
4. optional piece-first assembly Dice / Precision / Recall.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
PACK = BASE / "region_aware_piece_context_locator_pack"
RESULTS = BASE / "region_aware_vlm_results"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
LABEL_TO_CLASS = {
    "lung_bronchiola": "bronchiola",
    "lung_alveoli_normal_adjacent": "alveoli",
    "lung_vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_key(value: str) -> str:
    return LABEL_TO_CLASS.get(value, value)


def html_table(rows: Iterable[dict[str, object]], cols: list[str]) -> str:
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


def score_vector(row: dict[str, str]) -> tuple[int, ...]:
    vals = []
    for key in CLASS_KEYS:
        try:
            vals.append(int(float(row.get(key, 0) or 0)))
        except ValueError:
            vals.append(0)
    return tuple(vals)


def copy_result_dir(src: Path, dest: Path) -> Path:
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    keep = {
        "cross_label_scores.csv",
        "per_candidate_predictions.csv",
        "failed_rows.csv",
        "overall_accuracy.csv",
        "bucket_accuracy.csv",
        "per_class_accuracy.csv",
        "same_class_retrieval_scores.csv",
        "same_class_retrieval_metrics.csv",
        "score_distribution_checks.csv",
        "assembly_summary.csv",
        "assembly_policy_comparison.csv",
        "selected_pieces.csv",
        "run_config.json",
        "prompt_system.txt",
        "prompt_user_template.txt",
        "prompt_user_first_row.txt",
        "index.html",
    }
    dest.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if rel.name in keep or rel.parts[0] in {"figures", "selected_union_masks"}:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    return dest


def summarize_vlm(vlm_dir: Path, run_name: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    pred_path = vlm_dir / "per_candidate_predictions.csv"
    failed_path = vlm_dir / "failed_rows.csv"
    if not pred_path.exists():
        return (
            {
                "run": run_name,
                "status": "missing_predictions",
                "parsed_rows": 0,
                "failed_rows": "",
                "unique_score_vectors": "",
                "dominant_prediction": "",
                "collapse_warning": "missing per_candidate_predictions.csv",
            },
            [],
        )
    rows = read_csv(pred_path)
    ok = [row for row in rows if row.get("parse_status", "ok") == "ok"]
    failed = read_csv(failed_path) if failed_path.exists() else []
    vectors = [score_vector(row) for row in ok]
    all_same = sum(1 for vec in vectors if len(set(vec)) == 1)
    all_zero = sum(1 for vec in vectors if all(v == 0 for v in vec))
    pred_counter = Counter(row.get("predicted_label") or row.get("predicted_class") or "" for row in ok)
    dominant_label, dominant_count = pred_counter.most_common(1)[0] if pred_counter else ("", 0)
    dominant_frac = dominant_count / len(ok) if ok else 0.0
    warnings = []
    if failed:
        warnings.append(f"{len(failed)} failed rows")
    if all_zero:
        warnings.append(f"{all_zero} all-zero rows")
    if all_same / max(1, len(ok)) > 0.20:
        warnings.append(f"{all_same} all-same score rows")
    if dominant_frac > 0.70:
        warnings.append(f"class collapse: {dominant_label} {dominant_count}/{len(ok)}")
    if not warnings:
        warnings.append("pass")

    by_class: list[dict[str, object]] = []
    for c in CLASS_KEYS:
        cls_rows = [row for row in ok if class_key(row.get("true_label") or row.get("label") or "") == c]
        correct = [
            row
            for row in cls_rows
            if class_key(row.get("predicted_label") or row.get("predicted_class") or "") == c
            and str(row.get("top_score_tie", "False")) != "True"
        ]
        target_scores = []
        for row in cls_rows:
            try:
                target_scores.append(int(float(row.get(c, 0) or 0)))
            except ValueError:
                pass
        by_class.append(
            {
                "run": run_name,
                "class": c,
                "piece_top1": f"{len(correct)}/{len(cls_rows)}",
                "candidate_count": len(cls_rows),
                "median_target_score": int(sorted(target_scores)[len(target_scores) // 2]) if target_scores else "",
                "unique_target_scores": len(set(target_scores)) if target_scores else 0,
            }
        )
    return (
        {
            "run": run_name,
            "status": "ok",
            "parsed_rows": len(ok),
            "failed_rows": len(failed),
            "unique_score_vectors": f"{len(set(vectors))}/{len(ok)}" if ok else "0/0",
            "all_same_score_rows": all_same,
            "all_zero_score_rows": all_zero,
            "dominant_prediction": f"{dominant_label} {dominant_count}/{len(ok)}" if ok else "",
            "collapse_warning": "; ".join(warnings),
        },
        by_class,
    )


def summarize_assembly(assembly_dir: Path, run_name: str) -> tuple[list[dict[str, object]], list[str]]:
    summary_path = assembly_dir / "assembly_summary.csv"
    if not summary_path.exists():
        return [], []
    rows = []
    figures = []
    for row in read_csv(summary_path):
        figure = row.get("figure_rel", "")
        if figure:
            figures.append(str((assembly_dir / figure).relative_to(BASE)))
        rows.append(
            {
                "run": run_name,
                "class": row.get("class", ""),
                "policy": row.get("chosen_policy", row.get("policy", "")),
                "selected_piece_count": row.get("selected_piece_count", ""),
                "dice": row.get("dice", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "figure_rel": str((assembly_dir / figure).relative_to(BASE)) if figure else "",
            }
        )
    return rows, figures


def replace_section(text: str, title: str, section: str) -> str:
    marker = f"<h2>{title}</h2>"
    if marker in text:
        start = text.index(marker)
        next_heading = re.search(r"<h2[^>]*>", text[start + 1 :])
        end = start + 1 + next_heading.start() if next_heading else text.index("</body>", start)
        return text[:start] + section + text[end:]
    return text.replace("</body>", section + "</body>")


def write_report_section(
    run_rows: list[dict[str, object]],
    class_rows: list[dict[str, object]],
    assembly_rows: list[dict[str, object]],
) -> None:
    if run_rows:
        intro = "The table below contains completed region-aware local VLM runs. A run must pass parse, score-diversity, and class-collapse gates before its assembly metrics are interpreted."
    else:
        intro = "No completed region-aware local VLM run has been collected yet. The executable smoke/full/assembly scripts are already packaged; this section will be filled as soon as Bouchet results are available."
    figures = []
    for row in assembly_rows:
        rel = row.get("figure_rel", "")
        if rel:
            figures.append(
                f"<section class='class-block'><h3>{html.escape(str(row['run']))}: {html.escape(str(row['class']))}</h3>"
                f"<img src='{html.escape(str(rel))}' alt='region-aware assembly {html.escape(str(row['class']))}'>"
                "</section>"
            )
    section = f"""
<h2>17C. Region-Aware all6 VLM Results</h2>
<p>{html.escape(intro)}</p>
<h3>Run-level gates</h3>
{html_table(run_rows, ['run', 'status', 'parsed_rows', 'failed_rows', 'unique_score_vectors', 'all_same_score_rows', 'all_zero_score_rows', 'dominant_prediction', 'collapse_warning']) if run_rows else '<p class="small">Pending Bouchet smoke/full results.</p>'}
<h3>Piece Top1 by class</h3>
{html_table(class_rows, ['run', 'class', 'piece_top1', 'candidate_count', 'median_target_score', 'unique_target_scores']) if class_rows else '<p class="small">Pending.</p>'}
<h3>Assembly Dice / Precision / Recall</h3>
{html_table(assembly_rows, ['run', 'class', 'policy', 'selected_piece_count', 'dice', 'precision', 'recall']) if assembly_rows else '<p class="small">Pending.</p>'}
<h3>Six-panel assembly checks</h3>
{''.join(figures) if figures else '<p class="small">Pending.</p>'}
"""
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        html_path.write_text(replace_section(text, "17C. Region-Aware all6 VLM Results", section))


def update_zip() -> None:
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
        PACK,
        RESULTS,
        ROOT / "scripts/hpc/jun09_region_aware_multiimage_vlm.sbatch",
        ROOT / "scripts/hpc/jun09_region_aware_piece_assembly_cpu.sbatch",
        ROOT / "scripts/collect_jun09_region_aware_vlm_results.py",
    ]:
        if root.is_file():
            include.append(root)
        elif root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for p in include:
            if p in seen:
                continue
            seen.add(p)
            try:
                arcname = p.relative_to(BASE)
            except ValueError:
                arcname = Path("support_scripts") / p.relative_to(ROOT)
            handle.write(p, arcname)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-output-dir", type=Path, action="append", default=[])
    parser.add_argument("--assembly-output-dir", type=Path, action="append", default=[])
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    assembly_rows: list[dict[str, object]] = []

    copied_vlm_dirs: list[Path] = []
    for idx, src in enumerate(args.vlm_output_dir):
        name = args.run_name[idx] if idx < len(args.run_name) else src.name
        dest = src if args.no_copy else copy_result_dir(src, RESULTS / name / "vlm_output")
        copied_vlm_dirs.append(dest)
        run_summary, by_class = summarize_vlm(dest, name)
        run_rows.append(run_summary)
        class_rows.extend(by_class)

    for idx, src in enumerate(args.assembly_output_dir):
        name = args.run_name[idx] if idx < len(args.run_name) else src.name
        dest = src if args.no_copy else copy_result_dir(src, RESULTS / name / "assembly_output")
        rows, _ = summarize_assembly(dest, name)
        assembly_rows.extend(rows)

    write_csv(RESULTS / "region_aware_vlm_run_gates.csv", run_rows)
    write_csv(RESULTS / "region_aware_vlm_piece_top1_by_class.csv", class_rows)
    write_csv(RESULTS / "region_aware_vlm_assembly_summary.csv", assembly_rows)
    write_report_section(run_rows, class_rows, assembly_rows)
    update_zip()
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(RESULTS / "region_aware_vlm_run_gates.csv")
    print(RESULTS / "region_aware_vlm_piece_top1_by_class.csv")
    print(RESULTS / "region_aware_vlm_assembly_summary.csv")


if __name__ == "__main__":
    main()
