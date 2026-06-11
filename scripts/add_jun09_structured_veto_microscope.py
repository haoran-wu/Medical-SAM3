#!/usr/bin/env python3
"""Structured-veto failure microscope for the Jun09 skill ranker.

This is a diagnostic branch, not a final deployable model. It asks whether
FICTURE composition and H&E morphology should be used as a veto after the
piece-level ranker, instead of being treated as another raw VLM image.
"""

from __future__ import annotations

import csv
import html
import itertools
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "structured_veto_microscope"
ANNOTATION_DIR = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
HE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/he_roi_matching_official_ficture_coverage.png"
FICTURE_ROI = ROOT / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/ficture_official_filtered_roi_rgb.png"

CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
ANNOTATION_FILES = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune_infiltration": "03_immune_infiltration_target_roi.png",
}

# Class-specific target/conflict priors. These are not ground-truth labels; they
# are computed from the official FICTURE RGB/cell-type composition inside each
# candidate mask.
CONFLICT_PRIORS = {
    "bronchiola": ["vessels", "stroma", "immune_infiltration"],
    "alveoli": ["vessels", "tumor", "stroma"],
    "vessels": ["bronchiola", "tumor", "immune_infiltration"],
    "tumor": ["immune_infiltration", "vessels"],
    "stroma": ["immune_infiltration", "tumor", "bronchiola"],
    "immune_infiltration": ["vessels", "tumor", "stroma"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def row_key(row: dict[str, str]) -> str:
    if row.get("row_key"):
        return row["row_key"]
    return "__".join(
        [
            row.get("target_label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            row.get("candidate_id", ""),
        ]
    )


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.logical_and(pred, gt).sum())
    pa = int(pred.sum())
    ga = int(gt.sum())
    precision = tp / pa if pa else 0.0
    recall = tp / ga if ga else 0.0
    dice = 2 * tp / (pa + ga) if pa + ga else 0.0
    return dice, precision, recall


def predicted(scores: dict[str, int]) -> tuple[str, int, int, bool]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ordered[0][1]
    tie = sum(1 for _, value in ordered if value == top_score) > 1
    second = ordered[1][1] if len(ordered) > 1 else 0
    return ordered[0][0], top_score, top_score - second, tie


def overlap_fraction(candidate: np.ndarray, union: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 1.0
    return int(np.logical_and(candidate, union).sum()) / area


def prior(row: dict[str, str], tissue_class: str) -> float:
    return float(row.get(f"ficture_prior_{tissue_class}", 0) or 0)


def he_score(row: dict[str, str], tissue_class: str) -> float:
    return float(row.get(f"he_clip_large_{tissue_class}", 0) or 0)


def guard_pass(row: dict[str, str], tissue_class: str, min_prior: int, min_he: int, min_prior_margin: int) -> tuple[bool, str]:
    target_prior = prior(row, tissue_class)
    target_he = he_score(row, tissue_class)
    conflict = max((prior(row, c) for c in CONFLICT_PRIORS[tissue_class]), default=0.0)
    if target_prior < min_prior:
        return False, f"target_prior {target_prior:.0f} < {min_prior}"
    if target_he < min_he:
        return False, f"he_morphology {target_he:.0f} < {min_he}"
    if target_prior - conflict < min_prior_margin:
        return False, f"prior_margin {target_prior - conflict:.0f} < {min_prior_margin}"
    return True, "pass"


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
    img = image.convert("RGB")
    img.thumbnail((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return canvas


def render_six_panel(path: Path, tissue_class: str, union: np.ndarray, ann: np.ndarray, title: str, subtitle: str) -> None:
    he = Image.open(HE_ROI).convert("RGB")
    ficture = Image.open(FICTURE_ROI).convert("RGB")
    dice, precision, recall = metrics(union, ann)
    panel_w, panel_h = 260, 275
    gap, margin, header_h, label_h = 18, 26, 104, 24
    sheet = Image.new("RGB", (margin * 2 + panel_w * 6 + gap * 5, margin * 2 + header_h + label_h + panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    draw.text((margin, margin + 24), f"{tissue_class} | Dice {dice:.3f} | Precision {precision:.3f} | Recall {recall:.3f}", fill=(70, 80, 90), font=font)
    draw.text((margin, margin + 48), subtitle[:190], fill=(70, 80, 90), font=font)
    panels = [
        ("Annotation on H&E", overlay_mask(he, ann, (0, 180, 90))),
        ("Structured-veto union on H&E", overlay_mask(he, union, (0, 90, 255))),
        ("Structured-veto mask only", mask_only(union, (0, 90, 255))),
        ("Annotation mask only", mask_only(ann, (0, 180, 90))),
        ("H&E ROI", he),
        ("FICTURE ROI", ficture),
    ]
    y0 = margin + header_h
    for idx, (label, image) in enumerate(panels):
        x = margin + idx * (panel_w + gap)
        draw.text((x, y0), label, fill=(35, 45, 60), font=font)
        sheet.paste(fit_image(image, panel_w, panel_h), (x, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


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
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(HE_ROI).size
    small_size = (512, int(round(512 * size[1] / size[0])))
    public_rows = read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    features = {row["row_key"]: row for row in read_csv(BASE / "candidate_skill_features.csv")}
    public = {row_key(row): row for row in public_rows}
    loco_rows = read_csv(BASE / "score_outputs/loco_component_logistic_validation/per_candidate_predictions.csv")
    annotations = {c: load_mask(ANNOTATION_DIR / ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    small_annotations = {c: load_mask(ANNOTATION_DIR / ANNOTATION_FILES[c], small_size) for c in CLASS_KEYS}
    mask_cache: dict[str, np.ndarray] = {}
    small_mask_cache: dict[str, np.ndarray] = {}

    def get_mask(key: str) -> np.ndarray:
        path = public[key]["mask_path"]
        if path not in mask_cache:
            mask_cache[path] = load_mask(Path(path), size)
        return mask_cache[path]

    def get_small_mask(key: str) -> np.ndarray:
        path = public[key]["mask_path"]
        if path not in small_mask_cache:
            small_mask_cache[path] = load_mask(Path(path), small_size)
        return small_mask_cache[path]

    # A small grid exposes which failure is fixable by structured guards. The
    # selected best is explicitly diagnostic because annotation evaluates it.
    base_grid = {
        "min_score": [20, 35, 50, 70],
        "min_margin": [-15, 0, 10],
        "max_overlap": [0.45, 0.75],
        "max_pieces": [2, 4, 8, 12, 24, 36],
        "min_prior": [0, 50, 70],
        "min_he": [0, 50, 75],
        "min_prior_margin": [-50, 0, 15],
    }
    class_piece_caps = {
        "bronchiola": [2, 3, 4, 6],
        "alveoli": [1, 2, 3, 4],
        "vessels": [4, 6, 8, 12],
        "tumor": [4, 8, 12, 16],
        "stroma": [8, 16, 24, 36],
        "immune_infiltration": [12, 24, 36, 48],
    }

    all_grid_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    reject_examples: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        best = None
        for min_score, min_margin, max_overlap, max_pieces, min_prior, min_he, min_prior_margin in itertools.product(
            base_grid["min_score"],
            base_grid["min_margin"],
            base_grid["max_overlap"],
            class_piece_caps[tissue_class],
            base_grid["min_prior"],
            base_grid["min_he"],
            base_grid["min_prior_margin"],
        ):
            ranked = []
            rejects = []
            for row in loco_rows:
                scores = {c: int(float(row[c])) for c in CLASS_KEYS}
                pred, score, margin, tie = predicted(scores)
                if pred != tissue_class or tie:
                    continue
                if score < min_score or margin < min_margin:
                    continue
                feat = features[row["row_key"]]
                ok, reason = guard_pass(feat, tissue_class, min_prior, min_he, min_prior_margin)
                if not ok:
                    rejects.append((score, margin, reason, row))
                    continue
                ranked.append((score, margin, prior(feat, tissue_class), he_score(feat, tissue_class), row))
            ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
            union = np.zeros((small_size[1], small_size[0]), dtype=bool)
            selected = []
            for score, margin, target_prior, target_he, row in ranked:
                mask = get_small_mask(row["row_key"])
                if overlap_fraction(mask, union) > max_overlap:
                    continue
                selected.append((score, margin, target_prior, target_he, row))
                union |= mask
                if len(selected) >= max_pieces:
                    break
            dice, precision, recall = metrics(union, small_annotations[tissue_class])
            if tissue_class in {"bronchiola", "vessels"}:
                objective = dice + 0.20 * precision + 0.10 * recall
            elif tissue_class == "immune_infiltration":
                objective = dice + 0.10 * precision + 0.20 * recall
            else:
                objective = dice + 0.25 * precision + 0.05 * recall
            grid_row = {
                "class": tissue_class,
                "min_score": min_score,
                "min_margin": min_margin,
                "max_overlap": max_overlap,
                "max_pieces": max_pieces,
                "min_prior": min_prior,
                "min_he": min_he,
                "min_prior_margin": min_prior_margin,
                "selected_piece_count": len(selected),
                "dice": f"{dice:.6f}",
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "objective": f"{objective:.6f}",
            }
            all_grid_rows.append(grid_row)
            if best is None or objective > best[0]:
                best = (objective, grid_row, selected, rejects, union)
        assert best is not None
        _, best_grid, _, rejects, _ = best
        # Re-run the winning threshold on full-resolution masks for exact metrics
        # and for the six-panel figure. The low-resolution grid is only used to
        # avoid repeated expensive full-size mask operations.
        ranked_full = []
        for row in loco_rows:
            scores = {c: int(float(row[c])) for c in CLASS_KEYS}
            pred, score, margin, tie = predicted(scores)
            if pred != tissue_class or tie:
                continue
            if score < int(best_grid["min_score"]) or margin < int(best_grid["min_margin"]):
                continue
            feat = features[row["row_key"]]
            ok, _ = guard_pass(
                feat,
                tissue_class,
                int(best_grid["min_prior"]),
                int(best_grid["min_he"]),
                int(best_grid["min_prior_margin"]),
            )
            if not ok:
                continue
            ranked_full.append((score, margin, prior(feat, tissue_class), he_score(feat, tissue_class), row))
        ranked_full.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        union = np.zeros((size[1], size[0]), dtype=bool)
        selected = []
        for score, margin, target_prior, target_he, row in ranked_full:
            mask = get_mask(row["row_key"])
            if overlap_fraction(mask, union) > float(best_grid["max_overlap"]):
                continue
            selected.append((score, margin, target_prior, target_he, row))
            union |= mask
            if len(selected) >= int(best_grid["max_pieces"]):
                break
        dice, precision, recall = metrics(union, annotations[tissue_class])
        fig = OUT / "figures" / f"{tissue_class}_structured_veto_union.png"
        subtitle = (
            f"Diagnostic best: score>={best_grid['min_score']}, margin>={best_grid['min_margin']}, "
            f"FICTURE target-prior>={best_grid['min_prior']}, H&E morphology>={best_grid['min_he']}, "
            f"prior-margin>={best_grid['min_prior_margin']}."
        )
        render_six_panel(fig, tissue_class, union, annotations[tissue_class], f"{tissue_class}: structured-veto diagnostic assembly", subtitle)
        summary_rows.append(
            {
                "class": tissue_class,
                "selected_piece_count": len(selected),
                "dice": f"{dice:.3f}",
                "precision": f"{precision:.3f}",
                "recall": f"{recall:.3f}",
                "best_guard": (
                    f"score>={best_grid['min_score']}, margin>={best_grid['min_margin']}, "
                    f"prior>={best_grid['min_prior']}, he>={best_grid['min_he']}, "
                    f"prior_margin>={best_grid['min_prior_margin']}, overlap<={best_grid['max_overlap']}, max_pieces={best_grid['max_pieces']}"
                ),
                "figure_rel": str(fig.relative_to(BASE)),
            }
        )
        for rank, (score, margin, target_prior, target_he, row) in enumerate(selected, 1):
            feat = features[row["row_key"]]
            selected_rows.append(
                {
                    "class": tissue_class,
                    "rank": rank,
                    "candidate_uid": row["candidate_uid"],
                    "loco_score": score,
                    "loco_margin": margin,
                    "true_class_hidden_for_eval": row["true_label"],
                    "target_ficture_prior": f"{target_prior:.1f}",
                    "target_he_morphology": f"{target_he:.1f}",
                    "target_component_dice_hidden_for_eval": f"{float(feat.get('component_dice', 0) or 0):.3f}",
                }
            )
        for score, margin, reason, row in sorted(rejects, key=lambda item: (item[0], item[1]), reverse=True)[:6]:
            feat = features[row["row_key"]]
            reject_examples.append(
                {
                    "class": tissue_class,
                    "candidate_uid": row["candidate_uid"],
                    "loco_score": score,
                    "loco_margin": margin,
                    "hidden_true_class": row["true_label"],
                    "reject_reason": reason,
                    "target_ficture_prior": f"{prior(feat, tissue_class):.1f}",
                    "target_he_morphology": f"{he_score(feat, tissue_class):.1f}",
                }
            )

    write_csv(OUT / "structured_veto_grid.csv", all_grid_rows)
    write_csv(OUT / "structured_veto_summary.csv", summary_rows)
    write_csv(OUT / "structured_veto_selected_pieces.csv", selected_rows)
    write_csv(OUT / "structured_veto_rejected_examples.csv", reject_examples)

    figures = "".join(
        f"<section class='class-block'><h3>{html.escape(row['class'])}</h3><img src='{html.escape(row['figure_rel'])}' alt='{html.escape(row['class'])} structured veto'></section>"
        for row in summary_rows
    )
    section = f"""
<h2>17H. Structured-Veto Failure Microscope</h2>
<p>This diagnostic branch tests a more precise failure hypothesis: raw FICTURE color images often mislead a VLM, but structured FICTURE composition can still be useful as a veto after H&amp;E/ranker scoring. In other words, the scorer first proposes a tissue class, then the candidate must pass class-specific composition and H&amp;E morphology checks before it can be assembled.</p>
<p><b>Important:</b> the guard grid below is diagnostic. Annotation is hidden from the ranker and guards, but annotation is used afterward to choose which diagnostic threshold worked best on this one ROI. A deployable version must freeze the rule and validate on another ROI.</p>
{html_table(summary_rows, ['class', 'selected_piece_count', 'dice', 'precision', 'recall', 'best_guard'])}
<h3>Selected pieces after structured veto</h3>
{html_table(selected_rows[:80], ['class', 'rank', 'candidate_uid', 'loco_score', 'loco_margin', 'true_class_hidden_for_eval', 'target_ficture_prior', 'target_he_morphology'])}
<h3>High-scoring pieces rejected by the veto</h3>
{html_table(reject_examples, ['class', 'candidate_uid', 'loco_score', 'loco_margin', 'hidden_true_class', 'reject_reason', 'target_ficture_prior', 'target_he_morphology'])}
{figures}
<div class='callout'><b>Interpretation.</b> If this branch improves a class, the problem was not just VLM recognition; the assembly step needed a biologically grounded veto. If it fails or only works with annotation-tuned thresholds, the class should move to a generator/refinement branch rather than another prompt.</div>
"""

    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17H. Structured-Veto Failure Microscope</h2>"
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

    print(OUT / "structured_veto_summary.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
