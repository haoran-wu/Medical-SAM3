#!/usr/bin/env python3
"""Finalize official filtered FICTURE deliverables.

This script refuses deprecated FICTURE outputs, verifies PASS_OFFICIAL and the
official ROI dimensions, then packages:
  - official candidate-oracle best-by-Dice table and eight six-panel figures
  - official retrieve/VLM best-by-Dice aggregate
  - overall best-by-Dice table across candidate oracle and retrieve/VLM
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
    ("erythorocytes", "erythrocytes"),
    ("pigment", "pigment"),
]

REQUIRED_STATUS_CHECKS = [
    "uses_filtered_png_source",
    "ficture_md_says_filtered_feature_bc_matrix",
    "no_manual_shift",
    "full_canvas_matches_he",
    "formula_uses_microns_per_pixel_and_tissue_hires_scalef",
    "background_false_color_fraction_zero",
    "orientation_selected_fliplr",
    "orientation_qc_best_is_fliplr",
]

FORBIDDEN_FRAGMENTS = [
    "ficture_coord_scaled_hires",
    "ficture_corrected",
    "agent_verified_filtered_ficture_he_align",
    "deprecated_ficture_pre_official_20260517",
]


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def fmt(value: Any) -> str:
    return f"{as_float(value):.3f}"


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def resize_bool(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(image.resize((w, h), Image.Resampling.NEAREST)) > 127


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = image.astype(np.float32).copy()
    active = mask.astype(bool)
    out[active] = (1.0 - alpha) * out[active] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def solid_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    out[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return out


def render_six_panel(
    display: str,
    he: np.ndarray,
    ficture: np.ndarray,
    prediction: np.ndarray,
    annotation: np.ndarray,
    metrics: dict[str, float],
    method: str,
    output_path: Path,
    prediction_label: str,
) -> None:
    pred_color = (0, 112, 255)
    ann_color = (0, 185, 95)
    panels = [
        (overlay(he, prediction, pred_color), f"{prediction_label} on H&E ROI", "blue"),
        (solid_mask(prediction, pred_color), f"{prediction_label} only", "blue"),
        (overlay(he, annotation, ann_color), "Annotation on H&E ROI", "green"),
        (solid_mask(annotation, ann_color), "Annotation only", "green"),
        (he, "H&E ROI only", "black"),
        (ficture, "FICTURE map", "black"),
    ]
    title_text = (
        f"{display}: ROI FICTURE retrieve comparison\n"
        f"Dice {metrics['dice']:.3f} | Precision {metrics['precision']:.3f} | Recall {metrics['recall']:.3f}\n"
        f"{method}"
    )

    panel_h = 900
    panel_w = int(round(panel_h * he.shape[1] / he.shape[0]))
    header_h = 145
    title_h = 28
    gap = 16
    margin = 18
    canvas_w = margin * 2 + len(panels) * panel_w + (len(panels) - 1) * gap
    canvas_h = header_h + title_h + panel_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((margin, 12), title_text, fill="black", spacing=4)

    title_colors = {
        "blue": (0, 0, 255),
        "green": (0, 150, 60),
        "black": (0, 0, 0),
    }
    x = margin
    for image, panel_title, color_name in panels:
        draw.text((x, header_h), panel_title, fill=title_colors[color_name])
        panel = Image.fromarray(image.astype(np.uint8)).resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        canvas.paste(panel, (x, header_h + title_h))
        x += panel_w + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def check_official_paths(paths: list[Path]) -> None:
    bad = []
    for path in paths:
        text = str(path)
        bad.extend(fragment for fragment in FORBIDDEN_FRAGMENTS if fragment in text)
    if bad:
        raise SystemExit(f"Refusing deprecated FICTURE path fragments: {sorted(set(bad))}")


def verify_official(root: Path) -> dict[str, Any]:
    summary_path = root / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"
    input_dir = root / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi"
    check_official_paths([summary_path, input_dir])

    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {summary.get('status')}")

    checks = summary.get("status_checks", {})
    failed = [key for key in REQUIRED_STATUS_CHECKS if not checks.get(key)]
    if failed:
        raise SystemExit(f"Official FICTURE checks failed: {failed}")

    he_path = input_dir / "he_roi_matching_official_ficture_coverage.png"
    ficture_path = input_dir / "ficture_official_filtered_roi_rgb.png"
    factor_path = input_dir / "ficture_official_filtered_roi_factor_index.npy"
    region_path = input_dir / "region_summary_official_filtered_roi.json"
    for path in [he_path, ficture_path, factor_path, region_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    he_size = Image.open(he_path).size
    ficture_size = Image.open(ficture_path).size
    factor_shape = tuple(np.load(factor_path, mmap_mode="r").shape)
    if he_size != (3144, 3327):
        raise SystemExit(f"Unexpected H&E ROI size {he_size}; expected (3144, 3327)")
    if ficture_size != he_size:
        raise SystemExit(f"H&E/FICTURE ROI size mismatch: {he_size} vs {ficture_size}")
    if factor_shape != (3327, 3144):
        raise SystemExit(f"Unexpected factor-index shape {factor_shape}; expected (3327, 3144)")

    return {
        "summary_path": summary_path,
        "input_dir": input_dir,
        "summary": summary,
        "roi_size_wh": he_size,
        "factor_shape_hw": factor_shape,
    }


def load_candidate_oracle(root: Path) -> list[dict[str, Any]]:
    source = root / "output/visium_hd_exp1/official_ficture_candidate_best_report/official_ficture_candidate_best_by_dice.csv"
    display_to_label = {display: label for label, display in LABEL_ORDER}
    rows: list[dict[str, Any]] = []
    with source.open(newline="") as fh:
        for row in csv.DictReader(fh):
            display = row["tissue class"]
            label = display_to_label[display]
            rows.append(
                {
                    "label": label,
                    "display": display,
                    "dice": as_float(row["best Dice"]),
                    "precision": as_float(row["Precision"]),
                    "recall": as_float(row["Recall"]),
                    "source_type": "candidate_oracle",
                    "source_run": row.get("job_root", ""),
                    "method": row.get("best-performing method", ""),
                    "ranker": "candidate_by_dice",
                    "top_k": "1",
                    "selected": f"{row.get('setting', '')}/{row.get('best_candidate_index', '')}",
                    "path": row.get("report_path", ""),
                    "figure_path": row.get("figure_path", ""),
                    "candidate_mask_path": row.get("candidate_mask_path", ""),
                }
            )
    return rows


def load_ranking_rows(root: Path) -> list[dict[str, Any]]:
    label_to_display = dict(LABEL_ORDER)
    rows: list[dict[str, Any]] = []
    bases = [
        root / "output/visium_hd_exp1/multimodal_candidate_ranking",
        root / "output/visium_hd_exp1/vlm_candidate_judge",
        root / "output/visium_hd_exp1/vlm_direct_grounding",
    ]
    for base in bases:
        for path in sorted(base.glob("official*/best_by_label.csv")):
            check_official_paths([path])
            family = base.name
            run = path.parent.name
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    label = row.get("label", "")
                    if label not in label_to_display:
                        continue
                    rows.append(
                        {
                            "label": label,
                            "display": row.get("display") or label_to_display[label],
                            "dice": as_float(row.get("dice")),
                            "precision": as_float(row.get("precision")),
                            "recall": as_float(row.get("recall")),
                            "source_type": family,
                            "source_run": run,
                            "method": f"{family}: {run}",
                            "ranker": row.get("ranker", ""),
                            "top_k": row.get("top_k", ""),
                            "selected": row.get("selected", ""),
                            "path": str(path.relative_to(root)),
                        }
                    )
    order = {label: i for i, (label, _) in enumerate(LABEL_ORDER)}
    rows.sort(key=lambda item: (order.get(item["label"], 99), -item["dice"], item["source_run"]))
    return rows


def best_per_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = []
    for label, _display in LABEL_ORDER:
        label_rows = [row for row in rows if row["label"] == label]
        if label_rows:
            best.append(max(label_rows, key=lambda row: row["dice"]))
    return best


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_overall(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "tissue class",
        "best Dice",
        "Precision",
        "Recall",
        "source",
        "best-performing method/run",
        "ranker",
        "top_k",
        "selected",
        "path",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tissue class": row["display"],
                    "best Dice": f"{row['dice']:.6f}",
                    "Precision": f"{row['precision']:.6f}",
                    "Recall": f"{row['recall']:.6f}",
                    "source": row["source_type"],
                    "best-performing method/run": row.get("method") or row.get("source_run"),
                    "ranker": row.get("ranker", ""),
                    "top_k": row.get("top_k", ""),
                    "selected": row.get("selected", ""),
                    "path": row.get("path", ""),
                }
            )


def render_table(rows: list[dict[str, Any]], output_png: Path, title: str) -> None:
    headers = ["tissue class", "best Dice", "Precision", "Recall", "source", "best-performing method/run"]
    cells = []
    for row in rows:
        method = str(row.get("method") or row.get("source_run", ""))
        if len(method) > 58:
            method = method[:55] + "..."
        cells.append(
            [
                row["display"],
                fmt(row["dice"]),
                fmt(row["precision"]),
                fmt(row["recall"]),
                row["source_type"],
                method,
            ]
        )

    fig_height = 0.58 * (len(cells) + 1) + 0.9
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.14, 0.09, 0.09, 0.09, 0.18, 0.41],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row_i, _col_i), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_i == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        elif row_i % 2 == 0:
            cell.set_facecolor("#fafafa")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def copy_candidate_report(root: Path, out: Path) -> None:
    source_dir = root / "output/visium_hd_exp1/official_ficture_candidate_best_report"
    for filename in [
        "official_ficture_candidate_best_by_dice.csv",
        "official_ficture_candidate_best_by_dice.md",
        "official_ficture_candidate_best_by_dice_table.png",
        "manifest.json",
    ]:
        shutil.copy2(source_dir / filename, out / filename)
    for subdir in ["six_panel", "best_candidate_masks"]:
        target = out / subdir
        target.mkdir(exist_ok=True)
        for path in sorted((source_dir / subdir).glob("*.png")):
            shutil.copy2(path, target / path.name)


def resolve_region_mask(root: Path, region_summary_path: Path, value: str) -> Path:
    path = Path(value)
    candidates = [
        path,
        root / path,
        region_summary_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def load_annotation_masks(root: Path, region_summary_path: Path, shape_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    summary = json.loads(region_summary_path.read_text())
    masks = {}
    for item in summary.get("labels", []):
        label = item.get("label") or item.get("slug")
        if not label:
            continue
        mask_path = resolve_region_mask(root, region_summary_path, item["mask_path"])
        masks[label] = resize_bool(load_mask(mask_path), shape_hw)
    return masks


def load_candidate_feature_paths(root: Path, result_path: str) -> dict[tuple[str, str, int], Path]:
    best_path = root / result_path
    feature_path = best_path.parent / "candidate_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    paths: dict[tuple[str, str, int], Path] = {}
    with feature_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["source"], row["setting"], int(row["candidate_id"]))
            paths[key] = Path(row["mask_path"])
    return paths


def selected_union_mask(row: dict[str, Any], feature_paths: dict[tuple[str, str, int], Path], shape_hw: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape_hw, dtype=bool)
    missing = []
    for token in str(row.get("selected", "")).split(";"):
        if not token:
            continue
        parts = token.split("/")
        if len(parts) != 3:
            missing.append(token)
            continue
        source, setting, candidate_id_text = parts
        key = (source, setting, int(candidate_id_text))
        path = feature_paths.get(key)
        if path is None or not path.exists():
            missing.append(token)
            continue
        union |= resize_bool(load_mask(path), shape_hw)
    if missing:
        raise RuntimeError(f"Could not resolve selected retrieve masks for {row['display']}: {missing}")
    return union


def render_retrieve_panels(root: Path, out: Path, ranking_best: list[dict[str, Any]], preflight: dict[str, Any]) -> None:
    input_dir = preflight["input_dir"]
    he = np.array(Image.open(input_dir / "he_roi_matching_official_ficture_coverage.png").convert("RGB"))
    ficture = np.array(Image.open(input_dir / "ficture_official_filtered_roi_rgb.png").convert("RGB"))
    shape_hw = he.shape[:2]
    annotations = load_annotation_masks(root, input_dir / "region_summary_official_filtered_roi.json", shape_hw)

    figure_dir = out / "retrieve_six_panel"
    mask_dir = out / "retrieve_best_masks"
    figure_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    feature_cache: dict[str, dict[tuple[str, str, int], Path]] = {}

    rows = []
    for idx, (label, _display) in enumerate(LABEL_ORDER, start=1):
        row = next((item for item in ranking_best if item["label"] == label), None)
        if row is None:
            continue
        result_path = row["path"]
        if result_path not in feature_cache:
            feature_cache[result_path] = load_candidate_feature_paths(root, result_path)
        prediction = selected_union_mask(row, feature_cache[result_path], shape_hw)
        annotation = annotations[label]
        mask_path = mask_dir / f"{idx:02d}_{label}_retrieve_union_mask.png"
        Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
        figure_path = figure_dir / f"{idx:02d}_{label}_official_ficture_retrieve_best_by_dice_6panel.png"
        method = f"{row['source_run']} | {row['ranker']} top-{row['top_k']}"
        render_six_panel(
            row["display"],
            he,
            ficture,
            prediction,
            annotation,
            {"dice": row["dice"], "precision": row["precision"], "recall": row["recall"]},
            method,
            figure_path,
            "Retrieve union",
        )
        row = dict(row)
        row["retrieve_figure_path"] = str(figure_path.relative_to(root))
        row["retrieve_union_mask_path"] = str(mask_path.relative_to(root))
        rows.append(row)

    fields = [
        "label",
        "display",
        "dice",
        "precision",
        "recall",
        "source_type",
        "source_run",
        "ranker",
        "top_k",
        "selected",
        "method",
        "path",
        "retrieve_figure_path",
        "retrieve_union_mask_path",
    ]
    write_rows(out / "official_retrieve_best_by_dice_with_figures.csv", rows, fields)


def write_summary(out: Path, candidate: list[dict[str, Any]], ranking_best: list[dict[str, Any]], overall: list[dict[str, Any]], preflight: dict[str, Any]) -> None:
    lines = [
        "# Official Filtered FICTURE Final Bests\n\n",
        "Official preflight: PASS_OFFICIAL; ROI 3144 x 3327; factor index 3327 x 3144; no manual shift; formula uses microns_per_pixel and tissue_hires_scalef.\n\n",
        "## Overall Best By Dice\n\n",
        "| tissue class | best Dice | Precision | Recall | source | method/run | ranker | top_k |\n",
        "|---|---:|---:|---:|---|---|---|---:|\n",
    ]
    for row in overall:
        lines.append(
            f"| {row['display']} | {row['dice']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['source_type']} | {row.get('method') or row.get('source_run')} | {row.get('ranker', '')} | {row.get('top_k', '')} |\n"
        )

    lines.append("\n## Candidate Oracle Focus Bests\n\n")
    for display in ["bronchiola", "alveoli", "vessels"]:
        row = next(item for item in candidate if item["display"] == display)
        lines.append(
            f"- {display}: Dice {row['dice']:.3f}, P {row['precision']:.3f}, R {row['recall']:.3f}; "
            f"{row['method']} ({row['selected']})\n"
        )

    lines.append("\n## Ranking/VLM Focus Bests\n\n")
    for display in ["bronchiola", "alveoli", "vessels"]:
        row = next(item for item in ranking_best if item["display"] == display)
        lines.append(
            f"- {display}: Dice {row['dice']:.3f}, P {row['precision']:.3f}, R {row['recall']:.3f}; "
            f"{row['source_run']} / {row['ranker']} top-{row['top_k']}\n"
        )

    lines.append("\n## Files\n\n")
    lines.append("- `official_ficture_candidate_best_by_dice.csv`: single-candidate oracle table.\n")
    lines.append("- `official_ranking_vlm_best_by_dice.csv`: best retrieve/VLM output per class.\n")
    lines.append("- `official_overall_best_by_dice.csv`: best by Dice across candidate oracle and retrieve/VLM.\n")
    lines.append("- `six_panel/`: eight six-panel candidate-oracle figures.\n")
    lines.append(f"- Official summary: `{preflight['summary_path']}`.\n")
    (out / "FINAL_SUMMARY.md").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/visium_hd_exp1/final_official_ficture_clean_20260517"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = args.output_dir
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    preflight = verify_official(root)
    candidate = load_candidate_oracle(root)
    ranking_rows = load_ranking_rows(root)
    ranking_best = best_per_label(ranking_rows)
    overall = best_per_label(candidate + ranking_best)

    fields = [
        "label",
        "display",
        "dice",
        "precision",
        "recall",
        "source_type",
        "source_run",
        "ranker",
        "top_k",
        "selected",
        "method",
        "path",
    ]
    write_rows(out / "official_ranking_vlm_all_rows.csv", ranking_rows, fields)
    write_rows(out / "official_ranking_vlm_best_by_dice.csv", ranking_best, fields)
    write_rows(out / "official_candidate_oracle_best_by_dice_normalized.csv", candidate, fields)
    write_overall(out / "official_overall_best_by_dice.csv", overall)
    copy_candidate_report(root, out)
    render_retrieve_panels(root, out, ranking_best, preflight)
    render_table(overall, out / "official_overall_best_by_dice_table.png", "Official filtered FICTURE: overall best by Dice")
    render_table(ranking_best, out / "official_ranking_vlm_best_by_dice_table.png", "Official filtered FICTURE: retrieve/VLM best by Dice")
    write_summary(out, candidate, ranking_best, overall, preflight)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "official_status": preflight["summary"].get("status"),
        "official_summary": str(preflight["summary_path"].relative_to(root)),
        "official_input_dir": str(preflight["input_dir"].relative_to(root)),
        "roi_size_wh": list(preflight["roi_size_wh"]),
        "factor_shape_hw": list(preflight["factor_shape_hw"]),
        "ranking_vlm_best_files": len({row["path"] for row in ranking_rows}),
        "ranking_vlm_rows": len(ranking_rows),
    }
    (out / "final_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(out)
    print("overall focus:")
    for display in ["bronchiola", "alveoli", "vessels"]:
        row = next(item for item in overall if item["display"] == display)
        print(display, fmt(row["dice"]), fmt(row["precision"]), fmt(row["recall"]), row["source_type"], row["source_run"], row["ranker"])
    print("ranking/VLM focus:")
    for display in ["bronchiola", "alveoli", "vessels"]:
        row = next(item for item in ranking_best if item["display"] == display)
        print(display, fmt(row["dice"]), fmt(row["precision"]), fmt(row["recall"]), row["source_run"], row["ranker"])


if __name__ == "__main__":
    main()
