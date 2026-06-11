#!/usr/bin/env python3
"""Immune-specific H&E morphology/density branch for Jun09.

The current controller still has one unstable class: immune infiltration.
Raw FICTURE immune colors improve recall but introduce many false positives,
while the RF+H&E candidate ranker is more precise but misses coverage. This
script tests a different, pathology-inspired failure hypothesis:

    immune infiltration may be better represented as local nuclear-density
    regions in H&E, optionally constrained by official FICTURE immune priors.

The parameter sweep is diagnostic: annotations are used only to evaluate and
select the best variant. The output is meant to decide the next iteration, not
to claim a final deployable model.
"""

from __future__ import annotations

import csv
import html
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hed

import add_jun09_quality_adjusted_ranker as qa


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "immune_he_morphology_branch"
FACTOR_INDEX = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_factor_index.npy"

IMMUNE_FACTORS = [4, 6, 8, 10, 11]
TUMOR_FACTORS = [0, 2]
STROMA_VESSEL_FACTORS = [1, 9]


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


def resize_bool(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(size, Image.Resampling.NEAREST)) > 0


def normalize_map(values: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    valid = tissue_mask & np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(values[valid], [1, 99])
    if hi <= lo:
        return out
    out = (values - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def load_he_maps(size: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    image = Image.open(qa.HE_ROI).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.BICUBIC)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    tissue = (gray < 0.94) | (saturation > 0.08)
    hed = rgb2hed(np.clip(rgb, 1e-4, 1.0))
    hematoxylin = normalize_map(hed[..., 0], tissue)
    darkness = normalize_map(1.0 - gray, tissue)
    purple = normalize_map(0.55 * hematoxylin + 0.45 * darkness, tissue)
    return {"tissue": tissue, "hematoxylin": hematoxylin, "darkness": darkness, "purple": purple}


def load_factor_masks(size: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    idx = np.load(FACTOR_INDEX)
    masks = {
        "immune_prior": np.isin(idx, IMMUNE_FACTORS),
        "tumor_prior": np.isin(idx, TUMOR_FACTORS),
        "stroma_vessel_prior": np.isin(idx, STROMA_VESSEL_FACTORS),
        "valid_ficture": idx >= 0,
    }
    if size is not None:
        masks = {k: resize_bool(v, size) for k, v in masks.items()}
    return masks


def remove_small_and_keep(mask: np.ndarray, min_area: int, keep_top: int | None) -> tuple[np.ndarray, int, int]:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0
    sizes = np.bincount(labels.ravel())
    valid = np.flatnonzero(sizes >= min_area)
    valid = valid[valid != 0]
    if keep_top is not None and len(valid) > keep_top:
        valid = valid[np.argsort(sizes[valid])[-keep_top:]]
    if len(valid) == 0:
        return np.zeros_like(mask, dtype=bool), n_labels, 0
    return np.isin(labels, valid), n_labels, len(valid)


def build_density_mask(
    base_signal: np.ndarray,
    tissue: np.ndarray,
    factor_masks: dict[str, np.ndarray],
    signal_threshold: float,
    density_sigma: float,
    density_threshold: float,
    close_iter: int,
    min_area: int,
    keep_top: int | None,
    variant: str,
) -> tuple[np.ndarray, int, int]:
    nuclei = (base_signal >= signal_threshold) & tissue
    density = ndi.gaussian_filter(nuclei.astype(np.float32), sigma=density_sigma)
    if density.max() > 0:
        density = density / float(density.max())
    mask = density >= density_threshold
    if variant == "he_density_plus_immune_support":
        support = ndi.binary_dilation(factor_masks["immune_prior"], iterations=5)
        mask &= support
    elif variant == "he_density_minus_tumor_stroma_veto":
        veto = ndi.binary_dilation(factor_masks["tumor_prior"] | factor_masks["stroma_vessel_prior"], iterations=3)
        mask &= ~veto
    elif variant == "he_density_immune_support_minus_veto":
        support = ndi.binary_dilation(factor_masks["immune_prior"], iterations=5)
        veto = ndi.binary_dilation(factor_masks["tumor_prior"] | factor_masks["stroma_vessel_prior"], iterations=3)
        mask = mask & support & ~veto
    if close_iter:
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iter)
    return remove_small_and_keep(mask.astype(bool), min_area=min_area, keep_top=keep_top)


def objective(dice: float, precision: float, recall: float) -> float:
    # Immune needs recall, but precision below ~0.35 made earlier branches
    # visually unstable. This objective therefore gives a small precision bonus
    # and penalizes very low precision.
    penalty = 0.08 if precision < 0.35 else 0.0
    return dice + 0.08 * precision + 0.12 * recall - penalty


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_size = Image.open(qa.HE_ROI).size
    small_size = (512, int(round(512 * full_size[1] / full_size[0])))
    full_area = full_size[0] * full_size[1]
    small_area = small_size[0] * small_size[1]
    area_scale = small_area / full_area
    ann_full = qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES["immune_infiltration"], full_size)
    ann_small = qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES["immune_infiltration"], small_size)

    small_maps = load_he_maps(small_size)
    small_factors = load_factor_masks(small_size)

    variants = [
        "he_density_only",
        "he_density_plus_immune_support",
        "he_density_minus_tumor_stroma_veto",
    ]
    signals = ["hematoxylin", "purple"]
    lowres_rows: list[dict[str, object]] = []
    for variant in variants:
        for signal in signals:
            for signal_threshold in [0.50, 0.62, 0.74]:
                for density_sigma in [2, 4, 8, 12]:
                    for density_threshold in [0.08, 0.15, 0.25, 0.40]:
                        for close_iter in [0, 2, 5]:
                            for min_area_full in [800, 2500, 8000]:
                                min_area = max(1, int(round(min_area_full * area_scale)))
                                for keep_top in [None, 50, 100]:
                                    mask, components_before, components_after = build_density_mask(
                                        small_maps[signal],
                                        small_maps["tissue"],
                                        small_factors,
                                        signal_threshold=signal_threshold,
                                        density_sigma=density_sigma,
                                        density_threshold=density_threshold,
                                        close_iter=close_iter,
                                        min_area=min_area,
                                        keep_top=keep_top,
                                        variant=variant,
                                    )
                                    d, p, r = qa.metrics(mask, ann_small)
                                    lowres_rows.append(
                                        {
                                            "phase": "lowres_parameter_search",
                                            "variant": variant,
                                            "signal": signal,
                                            "signal_threshold": signal_threshold,
                                            "density_sigma": density_sigma,
                                            "density_threshold": density_threshold,
                                            "close_iter": close_iter,
                                            "min_area_full": min_area_full,
                                            "keep_top": "all" if keep_top is None else keep_top,
                                            "components_before_filter": components_before,
                                            "components_after_filter": components_after,
                                            "area_pixels": int(mask.sum()),
                                            "dice": f"{d:.4f}",
                                            "precision": f"{p:.4f}",
                                            "recall": f"{r:.4f}",
                                            "objective": f"{objective(d, p, r):.4f}",
                                        }
                                    )
    lowres_rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    write_csv(OUT / "immune_he_morphology_lowres_sweep_top200.csv", lowres_rows[:200])

    full_maps = load_he_maps()
    full_factors = load_factor_masks()
    full_rows: list[dict[str, object]] = []
    selected_settings: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    def setting_key(row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["variant"],
            row["signal"],
            row["signal_threshold"],
            row["density_sigma"],
            row["density_threshold"],
            row["close_iter"],
            row["min_area_full"],
            row["keep_top"],
        )

    def add_settings(rows: list[dict[str, object]], reason: str, limit: int) -> None:
        added = 0
        for row in rows:
            key = setting_key(row)
            if key in seen:
                continue
            seen.add(key)
            row = dict(row)
            row["selection_reason"] = reason
            selected_settings.append(row)
            added += 1
            if added >= limit:
                break

    # Full-resolution recompute is deliberately multi-objective. A pure
    # objective-top list can over-favor high precision / low recall settings, so
    # include explicit recall-oriented candidates before ruling out this branch.
    by_objective = sorted(lowres_rows, key=lambda row: float(row["objective"]), reverse=True)
    by_dice = sorted(lowres_rows, key=lambda row: float(row["dice"]), reverse=True)
    by_recall = sorted(lowres_rows, key=lambda row: float(row["recall"]), reverse=True)
    by_recall_p025 = sorted(
        [row for row in lowres_rows if float(row["precision"]) >= 0.25],
        key=lambda row: float(row["recall"]),
        reverse=True,
    )
    by_recall_p035 = sorted(
        [row for row in lowres_rows if float(row["precision"]) >= 0.35],
        key=lambda row: float(row["recall"]),
        reverse=True,
    )
    by_dice_p035 = sorted(
        [row for row in lowres_rows if float(row["precision"]) >= 0.35],
        key=lambda row: float(row["dice"]),
        reverse=True,
    )
    add_settings(by_objective, "lowres_objective_top", 20)
    add_settings(by_dice, "lowres_dice_top", 20)
    add_settings(by_recall, "lowres_recall_top", 20)
    add_settings(by_recall_p025, "lowres_recall_top_precision_ge_0.25", 20)
    add_settings(by_recall_p035, "lowres_recall_top_precision_ge_0.35", 20)
    add_settings(by_dice_p035, "lowres_dice_top_precision_ge_0.35", 20)

    write_csv(OUT / "immune_he_morphology_fullres_recompute_settings.csv", selected_settings)

    for row in selected_settings:
        key = (
            row["variant"],
            row["signal"],
            row["signal_threshold"],
            row["density_sigma"],
            row["density_threshold"],
            row["close_iter"],
            row["min_area_full"],
            row["keep_top"],
        )
    best_row: dict[str, object] | None = None
    best_mask: np.ndarray | None = None
    for row in selected_settings:
        keep_top = None if row["keep_top"] == "all" else int(row["keep_top"])
        mask, components_before, components_after = build_density_mask(
            full_maps[str(row["signal"])],
            full_maps["tissue"],
            full_factors,
            signal_threshold=float(row["signal_threshold"]),
            density_sigma=float(row["density_sigma"]),
            density_threshold=float(row["density_threshold"]),
            close_iter=int(row["close_iter"]),
            min_area=int(row["min_area_full"]),
            keep_top=keep_top,
            variant=str(row["variant"]),
        )
        d, p, r = qa.metrics(mask, ann_full)
        full_row = dict(row)
        full_row.update(
            {
                "phase": "fullres_top_recompute",
                "components_before_filter": components_before,
                "components_after_filter": components_after,
                "area_pixels": int(mask.sum()),
                "dice": f"{d:.4f}",
                "precision": f"{p:.4f}",
                "recall": f"{r:.4f}",
                "objective": f"{objective(d, p, r):.4f}",
            }
        )
        full_rows.append(full_row)
        if best_row is None or float(full_row["objective"]) > float(best_row["objective"]):
            best_row = full_row
            best_mask = mask
    assert best_row is not None and best_mask is not None
    full_rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    write_csv(OUT / "immune_he_morphology_fullres_top_recompute.csv", full_rows)
    write_csv(OUT / "immune_he_morphology_fullres_multiobjective_recompute.csv", full_rows)

    best_by_dice = max(full_rows, key=lambda row: float(row["dice"]))
    candidates_p035 = [row for row in full_rows if float(row["precision"]) >= 0.35]
    candidates_p025 = [row for row in full_rows if float(row["precision"]) >= 0.25]
    best_recall_p035 = max(candidates_p035, key=lambda row: float(row["recall"])) if candidates_p035 else {}
    best_recall_p025 = max(candidates_p025, key=lambda row: float(row["recall"])) if candidates_p025 else {}
    multiobjective_leaders = [
        dict(best_row, leader="best controller objective"),
        dict(best_by_dice, leader="best Dice"),
    ]
    if best_recall_p035:
        multiobjective_leaders.append(dict(best_recall_p035, leader="best Recall with Precision >= 0.35"))
    if best_recall_p025:
        multiobjective_leaders.append(dict(best_recall_p025, leader="best Recall with Precision >= 0.25"))
    write_csv(OUT / "immune_he_morphology_multiobjective_leaders.csv", multiobjective_leaders)

    fig = OUT / "figures" / "immune_infiltration_he_morphology_best.png"
    qa.render_six_panel(
        fig,
        "immune_infiltration",
        best_mask,
        ann_full,
        "immune infiltration: H&E nuclear-density morphology branch",
        (
            f"variant={best_row['variant']}, signal={best_row['signal']}, "
            f"signal_threshold={best_row['signal_threshold']}, density_sigma={best_row['density_sigma']}, "
            f"density_threshold={best_row['density_threshold']}, close={best_row['close_iter']}, "
            f"min_area={best_row['min_area_full']}, keep_top={best_row['keep_top']}."
        ),
    )
    best_row = dict(best_row)
    best_row["figure_rel"] = str(fig.relative_to(BASE))
    write_csv(OUT / "immune_he_morphology_best.csv", [best_row])

    comparison_rows = [
        {"variant": "runtime piece selector", "D/P/R": "0.448 / 0.393 / 0.522", "decision": "baseline from current runtime policy"},
        {"variant": "RF + H&E precision option", "D/P/R": "0.462 / 0.543 / 0.402", "decision": "better precision, lower recall"},
        {"variant": "semantic immune factors", "D/P/R": "0.378 / 0.268 / 0.641", "decision": "high recall but too many false positives"},
        {"variant": "runtime plus supported semantic components", "D/P/R": "0.465 / 0.350 / 0.694", "decision": "higher recall, precision still too low"},
        {"variant": "smoothed immune density map from FICTURE", "D/P/R": "0.431 / 0.332 / 0.614", "decision": "does not beat simpler hybrid"},
        {
            "variant": "new H&E nuclear-density morphology branch",
            "D/P/R": f"{float(best_row['dice']):.3f} / {float(best_row['precision']):.3f} / {float(best_row['recall']):.3f}",
            "decision": branch_decision(float(best_row["dice"]), float(best_row["precision"]), float(best_row["recall"])),
        },
    ]
    write_csv(OUT / "immune_branch_comparison_after_he_morphology.csv", comparison_rows)

    section = build_section(best_row, comparison_rows, full_rows[:20], multiobjective_leaders)
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AH. Immune H&E Morphology Branch</h2>"
        if marker in text:
            start = text.index(marker)
            next_pos = text.find("<h2>17AI.", start + len(marker))
            end = next_pos if next_pos != -1 else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "immune_he_morphology_best.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


def branch_decision(dice: float, precision: float, recall: float) -> str:
    if dice >= 0.50 and precision >= 0.40:
        return "promising: improves immune enough to become candidate controller branch"
    if recall >= 0.65 and precision < 0.35:
        return "recall-only: still too many false positives"
    if dice > 0.465:
        return "small gain only: useful evidence but not stable enough to freeze"
    return "not enough: H&E density alone does not solve immune"


def build_section(
    best_row: dict[str, object],
    comparison_rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
    multiobjective_leaders: list[dict[str, object]],
) -> str:
    rel = html.escape(str(best_row["figure_rel"]))
    return f"""
<h2>17AH. Immune H&amp;E Morphology Branch</h2>
<p><b>Goal.</b> Immune infiltration is the remaining unstable class. This branch tests whether H&amp;E morphology can identify immune-rich areas by local nuclear density, rather than asking the VLM to classify tiny pieces or trusting raw FICTURE immune colors.</p>
<p><b>What was tested.</b> The script computes H&amp;E hematoxylin/dark-purple density maps, thresholds local nuclear density, removes small components, and optionally uses official FICTURE immune support or tumor/stroma veto. Annotation is used only after scoring to evaluate each rule.</p>
<h3>Immune branch comparison</h3>
{table(comparison_rows, ['variant', 'D/P/R', 'decision'])}
<h3>Best H&amp;E morphology setting</h3>
{table([best_row], ['variant', 'signal', 'signal_threshold', 'density_sigma', 'density_threshold', 'close_iter', 'min_area_full', 'keep_top', 'components_after_filter', 'area_pixels', 'dice', 'precision', 'recall', 'objective'])}
<h3>Top full-resolution recomputed settings</h3>
{table(top_rows, ['selection_reason', 'variant', 'signal', 'signal_threshold', 'density_sigma', 'density_threshold', 'close_iter', 'min_area_full', 'keep_top', 'components_after_filter', 'dice', 'precision', 'recall', 'objective'])}
<h3>Recall-oriented guardrail</h3>
<p>This guardrail prevents a false negative. Instead of recomputing only objective-top settings, it also recomputes low-resolution Dice-top and Recall-top settings at full resolution. If a high-recall H&amp;E-density rule existed, it should appear here.</p>
{table(multiobjective_leaders, ['leader', 'selection_reason', 'variant', 'signal', 'signal_threshold', 'density_sigma', 'density_threshold', 'close_iter', 'min_area_full', 'keep_top', 'components_after_filter', 'dice', 'precision', 'recall', 'objective'])}
<div class='callout'><b>Interpretation.</b> If the recall-oriented leaders still do not clearly exceed the RF+H&amp;E or runtime+semantic baselines, the next immune iteration should not be another threshold sweep. It should use a better tissue encoder or explicit cell-density / nuclei segmentation features.</div>
<figure class='wide-figure'><img src='{rel}' alt='immune H&amp;E morphology best'><figcaption>Immune infiltration: best H&amp;E nuclear-density morphology result.</figcaption></figure>
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
    import re

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
