#!/usr/bin/env python3
"""Collect the Jun09 minimal three-setting component oracle branch.

The remote array writes one output directory per tissue label.  This collector
combines those per-label CSVs, extracts the synthetic
``minimal_three_settings`` source, renders the standard six-panel figures, and
appends the result back into the Jun09 skill-ranker notebook.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT_DIR = BASE / "refined_diagnostics/minimal_three_setting_component_oracle"

ANNOTATION_DIR = (
    ROOT
    / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/cropped_annotation_masks"
)
HE_IMAGE = (
    ROOT
    / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
    "he_roi_matching_official_ficture_coverage.png"
)
FICTURE_IMAGE = (
    ROOT
    / "output/visium_hd_exp1/ficture_official_filtered_candidate_pool_input_roi/"
    "ficture_official_filtered_roi_rgb.png"
)

MASK_BY_CLASS = {
    "bronchiola": "01_lung_bronchiola_target_roi.png",
    "alveoli": "04_lung_alveoli_normal_adjacent_target_roi.png",
    "vessels": "05_lung_vessels_target_roi.png",
    "tumor": "08_tumor_target_roi.png",
    "stroma": "07_stroma_target_roi.png",
    "immune infiltration": "03_immune_infiltration_target_roi.png",
}

CLASS_ORDER = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune infiltration"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


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


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def overlay_mask(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], title: str) -> Image.Image:
    out = base.convert("RGBA")
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    overlay[mask] = (*color, 115)
    out = Image.alpha_composite(out, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 980), 34), fill=(255, 255, 255, 225))
    draw.text((10, 10), title, fill=(0, 0, 0, 255), font=ImageFont.load_default())
    return out.convert("RGB")


def mask_only(mask: np.ndarray, color: tuple[int, int, int], title: str) -> Image.Image:
    arr = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    arr[mask] = color
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, min(img.width, 980), 34), fill=(255, 255, 255))
    draw.text((10, 10), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return img


def titled_image(image: Image.Image, title: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 980), 34), fill=(255, 255, 255))
    draw.text((10, 10), title, fill=(0, 0, 0), font=ImageFont.load_default())
    return out


def make_six_panel(
    class_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    he: Image.Image,
    ficture: Image.Image,
    title: str,
    out_path: Path,
) -> None:
    panels = [
        overlay_mask(he, gt, (0, 170, 90), "Annotation on H&E"),
        overlay_mask(he, pred, (0, 90, 255), "Selected union on H&E"),
        mask_only(pred, (0, 90, 255), "Selected union mask only"),
        mask_only(gt, (0, 170, 90), "Annotation mask only"),
        titled_image(he, "H&E ROI"),
        titled_image(ficture, "FICTURE ROI"),
    ]
    thumb_w = 500
    title_h = 70
    gap = 22
    resized = []
    for panel in panels:
        scale = thumb_w / panel.width
        resized.append(panel.resize((thumb_w, int(panel.height * scale)), Image.Resampling.BICUBIC))
    width = len(resized) * thumb_w + (len(resized) + 1) * gap
    height = title_h + max(img.height for img in resized) + gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 18), f"{class_name}: {title}", fill=(31, 41, 55), font=ImageFont.load_default())
    x = gap
    y = title_h
    for panel in resized:
        sheet.paste(panel, (x, y))
        x += thumb_w + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def parse_dpr(text: str) -> tuple[str, str, str]:
    nums = re.findall(r"\d+\.\d+", text)
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return "", "", ""


def find_union_mask(result_root: Path, class_name: str, source: str) -> Path | None:
    class_dir = class_name.replace(" ", "_")
    patterns = [
        f"**/{class_dir}/*{source}*component_union_mask.png",
        f"**/*{source}*component_union_mask.png",
    ]
    for pattern in patterns:
        hits = sorted(result_root.glob(pattern))
        if hits:
            return hits[0]
    return None


def collect_rows(result_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(result_root.glob("**/component_union_summary.csv")):
        for row in read_csv(csv_path):
            clean = dict(row)
            clean["result_subdir"] = str(csv_path.parent.relative_to(result_root))
            rows.append(clean)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source", default="minimal_three_settings")
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    result_root = args.result_root
    if not result_root.exists():
        raise FileNotFoundError(result_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = collect_rows(result_root)
    if not all_rows:
        raise RuntimeError(f"No component_union_summary.csv files found under {result_root}")
    write_csv(args.out_dir / "minimal_three_setting_component_oracle_all_rows.csv", all_rows)

    source_rows = [r for r in all_rows if r.get("source") == args.source]
    source_by_class = {r["class"]: r for r in source_rows}

    he = Image.open(HE_IMAGE).convert("RGB")
    ficture = Image.open(FICTURE_IMAGE).convert("RGB")
    expected_size = he.size
    if ficture.size != expected_size:
        raise ValueError(f"H&E/FICTURE size mismatch: {he.size} vs {ficture.size}")

    comparison_plan = read_csv(args.base_dir / "refined_diagnostics/class_specific_iteration_plan.csv")
    plan_by_class = {r["class"]: r for r in comparison_plan}
    expanded = read_csv(args.base_dir / "refined_diagnostics/expanded_pool_diagnostic.csv")
    expanded_by_class = {r["class"]: r for r in expanded}

    summary: list[dict[str, object]] = []
    comparison: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []

    for class_name in CLASS_ORDER:
        row = source_by_class.get(class_name)
        if not row:
            summary.append({"class": class_name, "status": "missing minimal_three_settings row"})
            continue

        gt = load_mask(ANNOTATION_DIR / MASK_BY_CLASS[class_name], size=expected_size)
        union_mask_path = find_union_mask(result_root, class_name, args.source)
        figure_rel = ""
        if union_mask_path is not None:
            pred = load_mask(union_mask_path, size=expected_size)
            fig_path = args.out_dir / "figures" / f"{class_name.replace(' ', '_')}_minimal_three_settings_six_panel.jpg"
            make_six_panel(
                class_name,
                gt,
                pred,
                he,
                ficture,
                f"D {row['component_union_Dice']} | P {row['component_union_Precision']} | R {row['component_union_Recall']}",
                fig_path,
            )
            figure_rel = str(fig_path.relative_to(args.base_dir))

        summary.append(
            {
                "class": class_name,
                "single best D/P/R": f"{row['single_best_Dice']} / {row['single_best_Precision']} / {row['single_best_Recall']}",
                "component union D/P/R": f"{row['component_union_Dice']} / {row['component_union_Precision']} / {row['component_union_Recall']}",
                "scored components": row.get("n_scored_components", ""),
                "selected candidates": row.get("selected_candidates", ""),
                "single best candidate": row.get("single_best_candidate", ""),
                "figure": figure_rel,
            }
        )
        plan = plan_by_class.get(class_name, {})
        exp = expanded_by_class.get(class_name, {})
        comparison.append(
            {
                "class": class_name,
                "minimal 3-setting component union D/P/R": f"{row['component_union_Dice']} / {row['component_union_Precision']} / {row['component_union_Recall']}",
                "medpt24 full oracle union D/P/R": exp.get("full oracle union D/P/R", ""),
                "current all-48 best D/P/R": plan.get("current all-48 best D/P/R", ""),
                "diagnosis after this branch": (
                    "minimal setting recovered a usable upper bound"
                    if float(row["component_union_Dice"]) >= 0.9 * float(parse_dpr(plan.get("current all-48 best D/P/R", "0 / 0 / 0"))[0] or 0)
                    else "still below current all-48; inspect selected components/generator"
                ),
            }
        )
        figure_rows.append({"class": class_name, "figure_rel": figure_rel})

    write_csv(args.out_dir / "minimal_three_setting_component_oracle_summary.csv", summary)
    write_csv(args.out_dir / "minimal_three_setting_component_oracle_comparison.csv", comparison)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "result_root": str(result_root),
                "source": args.source,
                "annotation_dir": str(ANNOTATION_DIR),
                "he_image": str(HE_IMAGE),
                "ficture_image": str(FICTURE_IMAGE),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    figures_html = "".join(
        f"<section class='figure-block'><h3>{html.escape(str(r['class']))}</h3>"
        f"<img src='{html.escape(str(r['figure_rel']))}' alt='{html.escape(str(r['class']))} minimal three setting'></section>"
        for r in figure_rows
        if r.get("figure_rel")
    )
    section = f"""
<h2>18. Minimal Three-Setting Component Oracle Result</h2>
<p>This is the first concrete refinement branch after the failure gates. Instead of returning to all 48 SAM settings, it tests a small complementary candidate pool: <code>base_official_points_step24</code>, <code>medical_official_points_step24</code>, and <code>base_box384_s128_m1536</code>. For each annotation component, the oracle chooses the best candidate from this combined pool, then unions those selected pieces.</p>
<p><b>How to read this table:</b> single best means the best one mask for the whole class. Component union means one best candidate per annotation component, then union. If component union is close to all-48, the generator can be simplified; if not, that class still needs another generator or refinement branch.</p>
{html_table(summary, ['class', 'single best D/P/R', 'component union D/P/R', 'scored components', 'single best candidate', 'selected candidates'])}
<h3>Comparison to current baselines</h3>
{html_table(comparison, ['class', 'minimal 3-setting component union D/P/R', 'medpt24 full oracle union D/P/R', 'current all-48 best D/P/R', 'diagnosis after this branch'])}
{figures_html}
<p class='small'>Raw collection files are saved under <code>refined_diagnostics/minimal_three_setting_component_oracle/</code>.</p>
"""

    for html_path in [
        args.base_dir / "Jun09_SkillRanker_ComponentAwareMaskSelection.html",
        args.base_dir / "index.html",
    ]:
        text = html_path.read_text()
        marker = "<h2>18. Minimal Three-Setting Component Oracle Result</h2>"
        if marker in text:
            start = text.index(marker)
            end = text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)

    zip_path = args.base_dir / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
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
        p = args.base_dir / rel
        if p.exists():
            include.append(p)
    for root in [
        args.base_dir / "corrected_pool",
        args.base_dir / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures",
        args.base_dir / "refined_diagnostics",
        args.base_dir / "guarded_loco_variant",
    ]:
        if root.exists():
            include.extend(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for p in include:
            if p in seen:
                continue
            seen.add(p)
            handle.write(p, p.relative_to(args.base_dir))

    print(args.out_dir / "minimal_three_setting_component_oracle_comparison.csv")
    print(args.base_dir / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(zip_path)


if __name__ == "__main__":
    main()
