#!/usr/bin/env python3
"""Diagnose when component-aware mask assembly is worth using.

The goal is to make the "should we assemble multiple candidate masks?"
decision reproducible instead of judging one example by eye.

The diagnostic combines three signals:

1. Multi-piece ground truth: the annotation has more than one meaningful
   connected component.
2. Single-mask recall ceiling: the best single candidate is precise enough
   but misses a meaningful amount of the annotation.
3. Component candidate availability: each important component has at least one
   usable candidate mask in the pool.

It writes a plain-language recommendation table: Yes / Maybe / No.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


MASK_MAP = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune infiltration": "03_immune_infiltration_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}


@dataclass(frozen=True)
class Thresholds:
    min_component_area_px: int
    important_component_area_fraction: float
    min_important_components: int
    single_best_recall_low: float
    single_best_precision_ok: float
    component_candidate_dice_ok: float
    component_candidate_precision_ok: float
    coverage_fraction_yes: float
    coverage_fraction_maybe: float


DEFAULT_THRESHOLDS = Thresholds(
    min_component_area_px=256,
    important_component_area_fraction=0.05,
    min_important_components=2,
    single_best_recall_low=0.75,
    single_best_precision_ok=0.50,
    component_candidate_dice_ok=0.50,
    component_candidate_precision_ok=0.50,
    coverage_fraction_yes=0.70,
    coverage_fraction_maybe=0.40,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
            "cropped_annotation_masks"
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune infiltration"],
    )
    parser.add_argument(
        "--he-best",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/candidate_pool_reports/he_same_roi_pool_48of48/"
            "he_same_roi_candidate_best_by_dice.csv"
        ),
    )
    parser.add_argument(
        "--ficture-best",
        type=Path,
        default=Path(
            "output/visium_hd_exp1/candidate_pool_reports/ficture_official_pool_current/"
            "ficture_official_candidate_best_by_dice.csv"
        ),
    )
    parser.add_argument(
        "--component-candidates",
        type=Path,
        default=None,
        help=(
            "Optional component_top_candidates.csv from componentwise_candidate_oracle.py. "
            "When absent, the diagnostic still reports GT structure and single-best signs, "
            "but component candidate availability is unknown."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-component-area-px", type=int, default=DEFAULT_THRESHOLDS.min_component_area_px)
    parser.add_argument(
        "--important-component-area-fraction",
        type=float,
        default=DEFAULT_THRESHOLDS.important_component_area_fraction,
    )
    parser.add_argument("--single-best-recall-low", type=float, default=DEFAULT_THRESHOLDS.single_best_recall_low)
    parser.add_argument("--single-best-precision-ok", type=float, default=DEFAULT_THRESHOLDS.single_best_precision_ok)
    parser.add_argument(
        "--component-candidate-dice-ok",
        type=float,
        default=DEFAULT_THRESHOLDS.component_candidate_dice_ok,
    )
    parser.add_argument(
        "--component-candidate-precision-ok",
        type=float,
        default=DEFAULT_THRESHOLDS.component_candidate_precision_ok,
    )
    return parser.parse_args()


def normalize_class(label: str) -> str:
    return label.replace("_", " ").strip().lower()


def load_binary_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def component_stats(mask: np.ndarray, min_area: int) -> list[dict[str, object]]:
    labeled, n_components = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndi.find_objects(labeled)
    rows: list[dict[str, object]] = []
    total = int(mask.sum())
    for original_id, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        component = labeled[sl] == original_id
        area = int(component.sum())
        if area < min_area:
            continue
        y0, y1 = int(sl[0].start), int(sl[0].stop)
        x0, x1 = int(sl[1].start), int(sl[1].stop)
        rows.append(
            {
                "original_component": original_id,
                "area_px": area,
                "area_fraction": area / total if total else 0.0,
                "bbox_xyxy": f"{x0},{y0},{x1},{y1}",
                "bbox_size": f"{x1 - x0}x{y1 - y0}",
            }
        )
    rows.sort(key=lambda row: int(row["area_px"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["component"] = f"C{idx}"
    return rows


def find_col(df: pd.DataFrame, options: list[str]) -> str:
    lower = {col.lower(): col for col in df.columns}
    for option in options:
        if option.lower() in lower:
            return lower[option.lower()]
    raise KeyError(f"Could not find any of {options} in columns {list(df.columns)}")


def load_best_table(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    class_col = find_col(df, ["class", "tissue class", "label"])
    dice_col = find_col(df, ["best Dice", "Dice", "best_dice"])
    precision_col = find_col(df, ["Precision", "best Precision", "precision"])
    recall_col = find_col(df, ["Recall", "best Recall", "recall"])
    out = pd.DataFrame(
        {
            "class": df[class_col].map(normalize_class),
            f"{source}_single_best_dice": pd.to_numeric(df[dice_col], errors="coerce"),
            f"{source}_single_best_precision": pd.to_numeric(df[precision_col], errors="coerce"),
            f"{source}_single_best_recall": pd.to_numeric(df[recall_col], errors="coerce"),
        }
    )
    method_cols = [col for col in df.columns if col.lower() in {"method", "best-performing method", "setting"}]
    if method_cols:
        out[f"{source}_single_best_method"] = df[method_cols[0]].astype(str)
    return out


def summarize_component_availability(
    component_candidates: pd.DataFrame | None,
    class_name: str,
    important_components: set[str],
    thresholds: Thresholds,
) -> dict[str, object]:
    if component_candidates is None or not important_components:
        return {
            "component_candidates_known": False,
            "important_components_with_usable_candidate": "",
            "important_component_coverage_fraction": "",
            "usable_component_list": "",
        }
    sub = component_candidates[component_candidates["class"].map(normalize_class) == class_name]
    if sub.empty:
        return {
            "component_candidates_known": False,
            "important_components_with_usable_candidate": "",
            "important_component_coverage_fraction": "",
            "usable_component_list": "",
        }
    usable_components: list[str] = []
    for component in sorted(important_components, key=lambda item: int(item.replace("C", ""))):
        comp = sub[sub["component"] == component]
        if comp.empty:
            continue
        dice = pd.to_numeric(comp["dice_vs_component"], errors="coerce")
        precision = pd.to_numeric(comp["precision_vs_component"], errors="coerce")
        usable = comp[(dice >= thresholds.component_candidate_dice_ok) & (precision >= thresholds.component_candidate_precision_ok)]
        if not usable.empty:
            usable_components.append(component)
    coverage = len(usable_components) / len(important_components) if important_components else 0.0
    return {
        "component_candidates_known": True,
        "important_components_with_usable_candidate": f"{len(usable_components)}/{len(important_components)}",
        "important_component_coverage_fraction": coverage,
        "usable_component_list": ", ".join(usable_components),
    }


def recommend(row: dict[str, object], thresholds: Thresholds) -> tuple[str, str]:
    multi_piece = int(row["important_component_count"]) >= thresholds.min_important_components
    recall_gap = bool(row["single_best_recall_gap"])
    availability_known = bool(row["component_candidates_known"])
    coverage_value = row["important_component_coverage_fraction"]
    coverage = float(coverage_value) if coverage_value != "" else None

    if not multi_piece:
        return "No", "GT does not have multiple meaningful pieces."
    if not recall_gap:
        return "Maybe", "GT has multiple pieces, but the best single mask is not clearly recall-limited."
    if availability_known and coverage is not None:
        if coverage >= thresholds.coverage_fraction_yes:
            return "Yes", "Multiple meaningful pieces, single-best recall is capped, and most pieces have usable candidates."
        if coverage >= thresholds.coverage_fraction_maybe:
            return "Maybe", "Some pieces have usable candidates, but coverage is incomplete."
        return "No", "Candidate pool does not cover enough important pieces reliably."
    return "Maybe", "Multiple meaningful pieces and recall gap exist; run component oracle to confirm candidate availability."


def write_html(out_dir: Path, thresholds: Thresholds, rows: list[dict[str, object]], component_rows: list[dict[str, object]]) -> None:
    def table(items: list[dict[str, object]]) -> str:
        if not items:
            return "<p>No rows.</p>"
        keys = list(items[0].keys())
        head = "".join(f"<th>{html.escape(str(key))}</th>" for key in keys)
        body = []
        for item in items:
            cells = "".join(f"<td>{html.escape(str(item.get(key, '')))}</td>" for key in keys)
            body.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    html_text = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Component-aware assembly diagnostic</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px; color: #1f2933; }}
h1 {{ margin-bottom: 8px; }}
p {{ line-height: 1.55; max-width: 1080px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px 10px; vertical-align: top; }}
th {{ background: #f7f7f8; }}
.note {{ background: #f3f4f6; border-left: 4px solid #2563eb; padding: 12px 16px; }}
</style>
</head>
<body>
<h1>Component-aware assembly diagnostic</h1>
<p class="note">这个报告保存了通用判断流程：不是每个组织类别都需要拼多个 candidate。只有当 annotation 有多个有意义的小块、单个 best mask 主要受 recall 限制、并且 candidate pool 里确实有可用的小块候选时，才建议使用 component-aware assembly。</p>
<p>默认阈值：important component area >= {thresholds.important_component_area_fraction:.0%} of GT；single-best recall &lt; {thresholds.single_best_recall_low:.2f} 且 precision >= {thresholds.single_best_precision_ok:.2f} 视为 recall-limited；component candidate Dice >= {thresholds.component_candidate_dice_ok:.2f} 且 precision >= {thresholds.component_candidate_precision_ok:.2f} 视为可用。</p>
<h2>Recommendation</h2>
{table(rows)}
<h2>GT Component Details</h2>
{table(component_rows)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds(
        min_component_area_px=args.min_component_area_px,
        important_component_area_fraction=args.important_component_area_fraction,
        min_important_components=DEFAULT_THRESHOLDS.min_important_components,
        single_best_recall_low=args.single_best_recall_low,
        single_best_precision_ok=args.single_best_precision_ok,
        component_candidate_dice_ok=args.component_candidate_dice_ok,
        component_candidate_precision_ok=args.component_candidate_precision_ok,
        coverage_fraction_yes=DEFAULT_THRESHOLDS.coverage_fraction_yes,
        coverage_fraction_maybe=DEFAULT_THRESHOLDS.coverage_fraction_maybe,
    )

    he_best = load_best_table(args.he_best, "HE")
    ficture_best = load_best_table(args.ficture_best, "FICTURE")
    best = he_best.merge(ficture_best, on="class", how="outer")

    component_candidates = None
    if args.component_candidates:
        component_candidates = pd.read_csv(args.component_candidates)
        component_candidates["class"] = component_candidates["class"].map(normalize_class)

    summary_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for raw_class in args.classes:
        class_name = normalize_class(raw_class)
        mask_name = MASK_MAP.get(class_name)
        if not mask_name:
            raise KeyError(f"No annotation mask mapping for class: {raw_class}")
        mask_path = args.annotation_dir / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        components = component_stats(load_binary_mask(mask_path), thresholds.min_component_area_px)
        important = {
            str(component["component"])
            for component in components
            if float(component["area_fraction"]) >= thresholds.important_component_area_fraction
        }
        for component in components:
            component_rows.append(
                {
                    "class": class_name,
                    "component": component["component"],
                    "area_px": component["area_px"],
                    "area_%_of_GT": f"{100 * float(component['area_fraction']):.1f}",
                    "important_for_assembly": component["component"] in important,
                    "bbox_xyxy": component["bbox_xyxy"],
                    "bbox_size": component["bbox_size"],
                }
            )

        best_row = best[best["class"] == class_name]
        base: dict[str, object] = {
            "class": class_name,
            "gt_component_count": len(components),
            "important_component_count": len(important),
            "important_components": ", ".join(sorted(important, key=lambda item: int(item.replace("C", "")))),
        }
        if best_row.empty:
            base.update(
                {
                    "best_single_source": "",
                    "best_single_Dice": "",
                    "best_single_Precision": "",
                    "best_single_Recall": "",
                    "single_best_recall_gap": "",
                }
            )
        else:
            row = best_row.iloc[0]
            source_options = []
            for source in ["HE", "FICTURE"]:
                dice = row.get(f"{source}_single_best_dice")
                if pd.notna(dice):
                    source_options.append((float(dice), source))
            source_options.sort(reverse=True)
            best_source = source_options[0][1] if source_options else ""
            precision = row.get(f"{best_source}_single_best_precision", "")
            recall = row.get(f"{best_source}_single_best_recall", "")
            dice = row.get(f"{best_source}_single_best_dice", "")
            recall_gap = (
                pd.notna(precision)
                and pd.notna(recall)
                and float(precision) >= thresholds.single_best_precision_ok
                and float(recall) < thresholds.single_best_recall_low
            )
            base.update(
                {
                    "best_single_source": best_source,
                    "best_single_Dice": f"{float(dice):.3f}" if pd.notna(dice) else "",
                    "best_single_Precision": f"{float(precision):.3f}" if pd.notna(precision) else "",
                    "best_single_Recall": f"{float(recall):.3f}" if pd.notna(recall) else "",
                    "single_best_recall_gap": recall_gap,
                }
            )

        base.update(summarize_component_availability(component_candidates, class_name, important, thresholds))
        rec, reason = recommend(base, thresholds)
        base["recommend_component_assembly"] = rec
        base["reason"] = reason
        summary_rows.append(base)

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "component_assembly_recommendations.csv", index=False)
    pd.DataFrame(component_rows).to_csv(args.out_dir / "gt_component_diagnostics.csv", index=False)
    (args.out_dir / "diagnostic_thresholds.json").write_text(
        json.dumps(thresholds.__dict__, indent=2, ensure_ascii=False)
    )
    write_html(args.out_dir, thresholds, summary_rows, component_rows)
    print(args.out_dir)


if __name__ == "__main__":
    main()
