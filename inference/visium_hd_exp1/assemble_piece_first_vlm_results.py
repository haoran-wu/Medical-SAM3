#!/usr/bin/env python3
"""Assemble piece-first VLM scores into final masks for VisiumHD Exp1.

This script is intentionally post-hoc: annotations are used only to evaluate
and compare assembly policies after the VLM has scored individual pieces.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_KEYS = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune_infiltration",
]

CLASS_TO_ANNOTATION = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}

CLASS_DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> str:
    if row.get("row_key"):
        return row["row_key"]
    return "__".join(
        [
            row.get("target_label") or row.get("label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            str(row.get("candidate_id", "")),
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
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
        for row in rows:
            writer.writerow(row)


def load_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    precision = tp / pred_area if pred_area else 0.0
    recall = tp / gt_area if gt_area else 0.0
    dice = (2 * tp) / (pred_area + gt_area) if pred_area + gt_area else 0.0
    return {"dice": dice, "precision": precision, "recall": recall}


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
    resized = image.convert("RGB")
    resized.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def render_six_panel(
    out_path: Path,
    tissue_class: str,
    he: Image.Image,
    ficture: Image.Image,
    selected_mask: np.ndarray,
    annotation: np.ndarray,
    title: str,
    subtitle: str,
    note: str,
) -> None:
    panel_w, panel_h = 260, 275
    gap = 18
    margin = 26
    header_h = 112
    title_h = 24
    items = [
        ("Annotation on H&E", overlay_mask(he, annotation, (0, 180, 90))),
        ("Selected union on H&E", overlay_mask(he, selected_mask, (0, 90, 255))),
        ("Selected union mask only", mask_only(selected_mask, (0, 90, 255))),
        ("Annotation mask only", mask_only(annotation, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    width = margin * 2 + panel_w * 6 + gap * 5
    height = margin * 2 + header_h + title_h + panel_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), subtitle, fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), note[:260], fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 72), f"{tissue_class}: six-panel piece-first VLM assembly check", fill=(70, 80, 90), font=font)
    y0 = margin + header_h
    for idx, (label, image) in enumerate(items):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(image, panel_w, panel_h), (x, y0 + title_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except ValueError:
        return 0


def row_scores(row: dict[str, str]) -> dict[str, int]:
    return {key: as_int(row, key) for key in CLASS_KEYS}


def predicted_class(scores: dict[str, int]) -> tuple[str, int, bool]:
    max_score = max(scores.values())
    winners = [key for key, value in scores.items() if value == max_score]
    return (winners[0], max_score, len(winners) > 1)


def score_margin(scores: dict[str, int], target: str) -> int:
    others = [value for key, value in scores.items() if key != target]
    return scores[target] - max(others)


def overlap_fraction(candidate: np.ndarray, selected_union: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 1.0
    return int(np.logical_and(candidate, selected_union).sum()) / area


def policy_grid(tissue_class: str) -> list[dict[str, object]]:
    if tissue_class in {"bronchiola", "vessels", "immune_infiltration"}:
        return [
            {"policy": "precision", "min_score": 70, "min_margin": 20, "max_overlap": 0.55, "max_pieces": 6},
            {"policy": "balanced", "min_score": 55, "min_margin": 10, "max_overlap": 0.65, "max_pieces": 12},
            {"policy": "recall_push", "min_score": 35, "min_margin": 4, "max_overlap": 0.75, "max_pieces": 24},
        ]
    return [
        {"policy": "precision", "min_score": 75, "min_margin": 20, "max_overlap": 0.55, "max_pieces": 3},
        {"policy": "balanced", "min_score": 60, "min_margin": 12, "max_overlap": 0.65, "max_pieces": 6},
        {"policy": "recall_push", "min_score": 45, "min_margin": 5, "max_overlap": 0.75, "max_pieces": 10},
    ]


def choose_best_policy(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No policies to choose from")
    # Prefer Dice, but avoid choosing a precision-collapse variant when another
    # policy is close in Dice and much cleaner.
    ordered = sorted(rows, key=lambda row: (float(row["dice"]), float(row["precision"])), reverse=True)
    best = ordered[0]
    for candidate in ordered[1:]:
        if (
            float(candidate["dice"]) >= float(best["dice"]) - 0.015
            and float(candidate["precision"]) > float(best["precision"]) + 0.10
        ):
            return candidate
    return best


def render_html(output_dir: Path, model_name: str, summary_rows: list[dict[str, object]], policy_rows: list[dict[str, object]], distribution_rows: list[dict[str, object]], warning: str) -> None:
    def table(rows: Iterable[dict[str, object]], cols: list[str]) -> str:
        parts = ["<table><thead><tr>"]
        parts.extend(f"<th>{html.escape(col)}</th>" for col in cols)
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>")
            parts.extend(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in cols)
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    figures = []
    for row in summary_rows:
        rel = row.get("figure_rel", "")
        if rel:
            figures.append(
                f"<section class='class-block'><h3>{html.escape(str(row['class']))}</h3>"
                f"<img src='{html.escape(str(rel))}' alt='{html.escape(str(row['class']))} piece-first assembly'>"
                "</section>"
            )

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Piece-first VLM assembly - {html.escape(model_name)}</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; margin:34px; color:#111827; line-height:1.55; }}
table {{ border-collapse:collapse; width:100%; margin:14px 0 28px; font-size:13px; }}
th,td {{ border-bottom:1px solid #e5e7eb; padding:8px 9px; text-align:left; vertical-align:top; }}
th {{ background:#f3f4f6; }}
.callout {{ background:#f8fafc; border-left:4px solid #2563eb; padding:13px 17px; margin:16px 0 24px; }}
img {{ width:100%; height:auto; border:1px solid #e5e7eb; display:block; }}
.class-block {{ margin:28px 0 42px; }}
code {{ background:#f3f4f6; padding:1px 5px; border-radius:5px; }}
</style></head><body>
<h1>Piece-first VLM assembly: {html.escape(model_name)}</h1>
<div class="callout"><b>How to read this.</b> The VLM saw one small clustered candidate piece at a time. The program then selected high-scoring, low-overlap pieces for each class and unioned them. Annotation is used only here, after scoring, to evaluate Dice / Precision / Recall.</div>
<h2>Score Distribution</h2>
<p>{html.escape(warning)}</p>
{table(distribution_rows, ["metric", "value"])}
<h2>Final selected assembly by class</h2>
{table(summary_rows, ["class", "chosen_policy", "selected_piece_count", "dice", "precision", "recall", "mean_selected_score", "max_selected_score", "note"])}
<h2>Policy comparison</h2>
{table(policy_rows, ["class", "policy", "selected_piece_count", "dice", "precision", "recall", "min_score", "min_margin", "max_overlap", "max_pieces"])}
<h2>Six-panel visual checks</h2>
{''.join(figures)}
</body></html>
"""
    (output_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-output-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--he-roi", type=Path, required=True)
    parser.add_argument("--ficture-roi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="")
    args = parser.parse_args()

    pred_path = args.vlm_output_dir / "per_candidate_predictions.csv"
    hidden_path = args.pool_dir / "hidden_candidate_truth.csv"
    failed_path = args.vlm_output_dir / "failed_rows.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions: {pred_path}")
    if not hidden_path.exists():
        raise SystemExit(f"Missing hidden truth: {hidden_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_out = args.output_dir / "selected_union_masks"
    figure_out = args.output_dir / "figures"

    predictions = [row for row in read_csv(pred_path) if row.get("parse_status") == "ok"]
    hidden_by_key = {row_key(row): row for row in read_csv(hidden_path)}
    failed_count = len(read_csv(failed_path)) if failed_path.exists() else 0
    rows: list[dict[str, object]] = []
    missing_hidden = 0
    for pred in predictions:
        hidden = hidden_by_key.get(row_key(pred))
        if hidden is None:
            missing_hidden += 1
            continue
        scores = row_scores(pred)
        predicted, top_score, tie = predicted_class(scores)
        merged: dict[str, object] = {**hidden, **pred}
        merged["predicted_class_calc"] = predicted
        merged["top_score_calc"] = top_score
        merged["top_score_tie_calc"] = tie
        merged["score_vector"] = tuple(scores[key] for key in CLASS_KEYS)
        for key, value in scores.items():
            merged[key] = value
        rows.append(merged)

    if not rows:
        raise SystemExit("No parseable rows with hidden truth to assemble")

    expected_size = Image.open(args.he_roi).size
    he = Image.open(args.he_roi).convert("RGB")
    ficture = Image.open(args.ficture_roi).convert("RGB")
    if Image.open(args.ficture_roi).size != expected_size:
        raise SystemExit("H&E and FICTURE ROI sizes do not match")

    annotation_masks = {
        key: load_mask(args.annotation_dir / name, expected_size)
        for key, name in CLASS_TO_ANNOTATION.items()
    }
    mask_cache: dict[str, np.ndarray] = {}

    def get_candidate_mask(row: dict[str, object]) -> np.ndarray:
        path = str(row["mask_path"])
        if path not in mask_cache:
            mask_cache[path] = load_mask(Path(path), expected_size)
        return mask_cache[path]

    vectors = [tuple(int(row[key]) for key in CLASS_KEYS) for row in rows]
    pred_counts = Counter(str(row["predicted_class_calc"]) for row in rows)
    all_same = sum(1 for vec in vectors if len(set(vec)) == 1)
    all_zero = sum(1 for vec in vectors if all(value == 0 for value in vec))
    top_pred, top_pred_count = pred_counts.most_common(1)[0]
    collapse_fraction = top_pred_count / len(rows)
    score_distribution_rows: list[dict[str, object]] = [
        {"metric": "parsed_rows", "value": len(rows)},
        {"metric": "failed_rows", "value": failed_count},
        {"metric": "missing_hidden_rows", "value": missing_hidden},
        {"metric": "unique_score_vectors", "value": f"{len(set(vectors))}/{len(rows)}"},
        {"metric": "all_same_score_rows", "value": all_same},
        {"metric": "all_zero_score_rows", "value": all_zero},
        {"metric": "predicted_class_counts", "value": json.dumps(dict(pred_counts), sort_keys=True)},
        {"metric": "dominant_prediction", "value": f"{top_pred} {top_pred_count}/{len(rows)}"},
    ]
    for class_key in CLASS_KEYS:
        values = [int(row[class_key]) for row in rows]
        score_distribution_rows.append(
            {
                "metric": f"{class_key}_score_summary",
                "value": f"unique={len(set(values))}; min={min(values)}; median={int(np.median(values))}; max={max(values)}",
            }
        )

    warnings = []
    if failed_count:
        warnings.append(f"{failed_count} rows failed to parse or generate.")
    if all_same / len(rows) > 0.20:
        warnings.append("Many rows have identical scores across classes.")
    if all_zero / len(rows) > 0.05:
        warnings.append("Some rows are all-zero score vectors.")
    if collapse_fraction > 0.70:
        warnings.append(f"Class-collapse warning: {top_pred} is predicted for {top_pred_count}/{len(rows)} rows.")
    if not warnings:
        warnings.append("No severe parse/tie/collapse warning detected by the automatic checks.")

    policy_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for class_key in CLASS_KEYS:
        class_policy_rows: list[dict[str, object]] = []
        annotation = annotation_masks[class_key]
        ranked = sorted(rows, key=lambda row: int(row[class_key]), reverse=True)
        for policy in policy_grid(class_key):
            selected: list[dict[str, object]] = []
            selected_union = np.zeros((expected_size[1], expected_size[0]), dtype=bool)
            for row in ranked:
                scores = {key: int(row[key]) for key in CLASS_KEYS}
                target_score = scores[class_key]
                margin = score_margin(scores, class_key)
                if target_score < int(policy["min_score"]):
                    continue
                if margin < int(policy["min_margin"]):
                    continue
                mask = get_candidate_mask(row)
                overlap = overlap_fraction(mask, selected_union)
                if overlap > float(policy["max_overlap"]):
                    continue
                selected.append(row)
                selected_union |= mask
                if len(selected) >= int(policy["max_pieces"]):
                    break
            met = metrics(selected_union, annotation)
            class_policy_rows.append(
                {
                    "class": class_key,
                    "policy": policy["policy"],
                    "selected_piece_count": len(selected),
                    "dice": f"{met['dice']:.6f}",
                    "precision": f"{met['precision']:.6f}",
                    "recall": f"{met['recall']:.6f}",
                    "min_score": policy["min_score"],
                    "min_margin": policy["min_margin"],
                    "max_overlap": policy["max_overlap"],
                    "max_pieces": policy["max_pieces"],
                    "_selected": selected,
                    "_mask": selected_union,
                }
            )
        policy_rows.extend([{key: value for key, value in row.items() if not key.startswith("_")} for row in class_policy_rows])
        best = choose_best_policy(class_policy_rows)
        selected = list(best["_selected"])  # type: ignore[index]
        union_mask = best["_mask"]  # type: ignore[index]
        met = metrics(union_mask, annotation)
        mask_path = mask_out / f"{class_key}_piece_first_selected_union.png"
        save_mask(union_mask, mask_path)
        selected_scores = [int(row[class_key]) for row in selected]
        fig_path = figure_out / f"{class_key}_piece_first_selected_union.png"
        note = (
            f"Selected by {best['policy']} policy from VLM-ranked clustered pieces; "
            f"score gate {best['min_score']}, margin gate {best['min_margin']}."
        )
        render_six_panel(
            fig_path,
            class_key,
            he,
            ficture,
            union_mask,
            annotation,
            f"{class_key}: piece-first VLM selected union",
            (
                f"Model: {args.model_name or args.vlm_output_dir.name} | Policy: {best['policy']} | "
                f"Dice {met['dice']:.3f} | Precision {met['precision']:.3f} | Recall {met['recall']:.3f}"
            ),
            note,
        )
        summary_rows.append(
            {
                "class": class_key,
                "chosen_policy": best["policy"],
                "selected_piece_count": len(selected),
                "dice": f"{met['dice']:.6f}",
                "precision": f"{met['precision']:.6f}",
                "recall": f"{met['recall']:.6f}",
                "mean_selected_score": f"{float(np.mean(selected_scores)):.2f}" if selected_scores else "",
                "max_selected_score": max(selected_scores) if selected_scores else "",
                "mask_path": str(mask_path),
                "figure_rel": str(fig_path.relative_to(args.output_dir)),
                "note": note,
            }
        )
        for rank, row in enumerate(selected, start=1):
            scores = {key: int(row[key]) for key in CLASS_KEYS}
            selected_rows.append(
                {
                    "class": class_key,
                    "chosen_policy": best["policy"],
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "row_key": row["row_key"],
                    "target_score": scores[class_key],
                    "score_margin": score_margin(scores, class_key),
                    "predicted_class": row["predicted_class_calc"],
                    "best_component_label": row.get("best_component_label", ""),
                    "best_component_dice": row.get("best_component_dice", ""),
                    "component_dice": row.get("component_dice", ""),
                    "component_precision": row.get("component_precision", ""),
                    "component_recall": row.get("component_recall", ""),
                    "mask_path": row["mask_path"],
                }
            )

    write_csv(args.output_dir / "score_distribution_checks.csv", score_distribution_rows)
    write_csv(args.output_dir / "assembly_policy_comparison.csv", policy_rows)
    write_csv(args.output_dir / "assembly_summary.csv", summary_rows)
    write_csv(args.output_dir / "selected_pieces.csv", selected_rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "vlm_output_dir": str(args.vlm_output_dir),
                "pool_dir": str(args.pool_dir),
                "annotation_dir": str(args.annotation_dir),
                "he_roi": str(args.he_roi),
                "ficture_roi": str(args.ficture_roi),
                "model_name": args.model_name,
                "warning": " ".join(warnings),
            },
            indent=2,
        )
    )
    render_html(
        args.output_dir,
        args.model_name or args.vlm_output_dir.name,
        summary_rows,
        policy_rows,
        score_distribution_rows,
        " ".join(warnings),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
