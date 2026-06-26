#!/usr/bin/env python3
"""Component-fold validation for the Jun09 immune morphology branch.

The H&E density + immune support branch improved immune infiltration, but the
best parameters were selected with annotation. This diagnostic asks whether the
rule is at least stable under component-level folds inside the only annotated
ROI. It is not external validation, but it is stricter than choosing parameters
on the full annotation and reporting that same annotation.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import add_jun09_quality_adjusted_ranker as qa
import add_jun09_immune_he_morphology_branch as immune


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "immune_loco_validation"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
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


def fold_masks(annotation: np.ndarray, n_folds: int = 5) -> list[np.ndarray]:
    labels, n_labels = ndi.label(annotation)
    if n_labels == 0:
        return [np.zeros_like(annotation, dtype=bool) for _ in range(n_folds)]
    sizes = np.bincount(labels.ravel())
    component_ids = [idx for idx in range(1, n_labels + 1) if sizes[idx] > 0]
    component_ids.sort(key=lambda idx: int(sizes[idx]), reverse=True)
    fold_components: list[list[int]] = [[] for _ in range(n_folds)]
    for i, component_id in enumerate(component_ids):
        fold_components[i % n_folds].append(component_id)
    return [np.isin(labels, ids) for ids in fold_components]


def objective(dice: float, precision: float, recall: float) -> float:
    return immune.objective(dice, precision, recall)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_size = Image.open(qa.HE_ROI).size
    small_size = (512, int(round(512 * full_size[1] / full_size[0])))
    full_area = full_size[0] * full_size[1]
    small_area = small_size[0] * small_size[1]
    area_scale = small_area / full_area

    ann_small = qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES["immune_infiltration"], small_size)
    folds = fold_masks(ann_small, n_folds=5)
    all_ann = np.logical_or.reduce(folds)
    small_maps = immune.load_he_maps(small_size)
    small_factors = immune.load_factor_masks(small_size)

    settings: list[dict[str, object]] = []
    for variant in ["he_density_only", "he_density_plus_immune_support", "he_density_minus_tumor_stroma_veto"]:
        for signal in ["hematoxylin", "purple"]:
            for signal_threshold in [0.50, 0.62, 0.74]:
                for density_sigma in [2, 4, 8, 12]:
                    for density_threshold in [0.08, 0.15, 0.25, 0.40]:
                        for close_iter in [0, 2, 5]:
                            for min_area_full in [800, 2500, 8000]:
                                min_area = max(1, int(round(min_area_full * area_scale)))
                                for keep_top in [None, 50, 100]:
                                    settings.append(
                                        {
                                            "variant": variant,
                                            "signal": signal,
                                            "signal_threshold": signal_threshold,
                                            "density_sigma": density_sigma,
                                            "density_threshold": density_threshold,
                                            "close_iter": close_iter,
                                            "min_area_full": min_area_full,
                                            "min_area_small": min_area,
                                            "keep_top": "all" if keep_top is None else keep_top,
                                        }
                                    )

    fold_rows: list[dict[str, object] | None] = [None for _ in folds]
    # Generate each setting once, then evaluate it against every fold. The first
    # implementation looped fold -> setting and rebuilt the same mask five times.
    for setting in settings:
        keep_top = None if setting["keep_top"] == "all" else int(setting["keep_top"])
        mask, components_before, components_after = immune.build_density_mask(
            small_maps[str(setting["signal"])],
            small_maps["tissue"],
            small_factors,
            signal_threshold=float(setting["signal_threshold"]),
            density_sigma=float(setting["density_sigma"]),
            density_threshold=float(setting["density_threshold"]),
            close_iter=int(setting["close_iter"]),
            min_area=int(setting["min_area_small"]),
            keep_top=keep_top,
            variant=str(setting["variant"]),
        )
        d_all, p_all, r_all = qa.metrics(mask, all_ann)
        for fold_zero, heldout in enumerate(folds):
            fold_idx = fold_zero + 1
            train = all_ann & ~heldout
            d_train, p_train, r_train = qa.metrics(mask, train)
            score = objective(d_train, p_train, r_train)
            current_best = fold_rows[fold_zero]
            if current_best is None or score > float(current_best["train_objective"]):
                d_hold, p_hold, r_hold = qa.metrics(mask, heldout)
                best = dict(setting)
                best.update(
                    {
                        "fold": fold_idx,
                        "components_before_filter": components_before,
                        "components_after_filter": components_after,
                        "train_dice": f"{d_train:.4f}",
                        "train_precision": f"{p_train:.4f}",
                        "train_recall": f"{r_train:.4f}",
                        "train_objective": f"{score:.4f}",
                        "heldout_dice": f"{d_hold:.4f}",
                        "heldout_precision": f"{p_hold:.4f}",
                        "heldout_recall": f"{r_hold:.4f}",
                        "all_dice": f"{d_all:.4f}",
                        "all_precision": f"{p_all:.4f}",
                        "all_recall": f"{r_all:.4f}",
                    }
                )
                fold_rows[fold_zero] = best

    assert all(row is not None for row in fold_rows)
    fold_rows = [row for row in fold_rows if row is not None]

    means = {}
    for key in ["heldout_dice", "heldout_precision", "heldout_recall", "all_dice", "all_precision", "all_recall"]:
        vals = [float(row[key]) for row in fold_rows]
        means[key] = float(np.mean(vals))
        means[f"{key}_min"] = float(np.min(vals))
        means[f"{key}_max"] = float(np.max(vals))

    summary_rows = [
        {
            "validation": "5-fold annotation-component validation on low-resolution ROI",
            "mean heldout D/P/R": f"{means['heldout_dice']:.3f} / {means['heldout_precision']:.3f} / {means['heldout_recall']:.3f}",
            "range heldout Dice": f"{means['heldout_dice_min']:.3f} - {means['heldout_dice_max']:.3f}",
            "mean all-component D/P/R": f"{means['all_dice']:.3f} / {means['all_precision']:.3f} / {means['all_recall']:.3f}",
            "interpretation": interpret(means),
        }
    ]

    write_csv(OUT / "immune_component_fold_validation.csv", fold_rows)
    write_csv(OUT / "immune_component_fold_validation_summary.csv", summary_rows)

    section = build_section(summary_rows, fold_rows)
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AJ. Immune Component-Fold Validation</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[K-Z]|<h2>18\\.", text[start + len(marker):])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "immune_component_fold_validation_summary.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


def interpret(means: dict[str, float]) -> str:
    if means["heldout_dice"] >= 0.45 and means["heldout_precision"] >= 0.45:
        return "supports the branch: not perfect, but the rule generalizes across held-out annotation components within this ROI"
    if means["heldout_recall"] >= 0.55 and means["heldout_precision"] < 0.35:
        return "recall generalizes but false positives remain too high"
    return "weak component-fold stability; treat the full-annotation optimum as diagnostic only"


def build_section(summary_rows: list[dict[str, object]], fold_rows: list[dict[str, object]]) -> str:
    return f"""
<h2>17AJ. Immune Component-Fold Validation</h2>
<p><b>Purpose.</b> The immune H&amp;E-density branch improved after recall-oriented full-resolution recompute, but it was still selected using annotation. This gate asks a stricter question: if we tune on four groups of immune annotation components, does the selected rule still hit the held-out component group?</p>
<p><b>Important caveat.</b> This is not external validation because all folds come from the same ROI. It is a sanity check against the worst kind of parameter overfitting inside the only annotated sample.</p>
<h3>Summary</h3>
{table(summary_rows, ['validation', 'mean heldout D/P/R', 'range heldout Dice', 'mean all-component D/P/R', 'interpretation'])}
<h3>Fold details</h3>
{table(fold_rows, ['fold', 'variant', 'signal', 'signal_threshold', 'density_sigma', 'density_threshold', 'close_iter', 'min_area_full', 'keep_top', 'train_dice', 'train_precision', 'train_recall', 'heldout_dice', 'heldout_precision', 'heldout_recall', 'all_dice', 'all_precision', 'all_recall'])}
<div class='callout'><b>Decision rule.</b> If held-out components are unstable, the full-annotation immune result is only a diagnostic direction. If held-out components are reasonably stable, the next step is to freeze a simple annotation-free immune-density rule and test it on another ROI or with manual review.</div>
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
        p = BASE / rel
        if p.exists():
            include.append(p)
    for root in [
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he",
        BASE / "runtime_policy_prototype",
        BASE / "alveoli_broad_box_branch",
        BASE / "semantic_ficture_proposal_generator",
        BASE / "failure_driven_hybrid_controller",
        BASE / "immune_he_morphology_branch",
        BASE / "failure_analysis_protocol_v2",
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


if __name__ == "__main__":
    main()
