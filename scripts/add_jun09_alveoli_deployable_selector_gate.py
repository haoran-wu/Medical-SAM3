#!/usr/bin/env python3
"""Append an alveoli deployable-selector diagnostic gate to the Jun09 report.

The existing Jun09 alveoli branch established that broad H&E box384 proposals
can recover a strong alveoli union, but the winning pair (candidate 12+13) was
identified with hidden annotation and manual review.  This gate asks the next
stricter question: can an annotation-free rule rank those broad alveoli
candidates high while suppressing obvious false positives?

This script deliberately separates selection inputs from evaluation:

- Selection inputs: candidate size/bbox, simple H&E crop proxies when available,
  and structured FICTURE composition summaries.
- Evaluation only: hidden component Dice/Precision/Recall from the Jun07 table.
"""

from __future__ import annotations

import ast
import csv
import html
import math
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_deployable_selector_gate"
JUN07_INPUTS = ROOT / "output/visium_hd_exp1/final_deliverables/Jun07_GPT_manual_alveoli_prompt_test_inputs"

ROI_W = 3144
ROI_H = 3327
KEY_IDS = [11, 12, 13, 82, 90]
TARGET_IDS = [12, 13]
FALSE_POSITIVE_IDS = [82, 90]

PERCENT_RE = re.compile(r"([A-Za-z0-9_]+) ([0-9.]+)%")
GROUPS = [
    "alveolar_AT2_F3_F5",
    "tumor_epithelial_F0_F2",
    "stroma_endothelial_F1_F9",
    "airway_epithelial_F7",
    "immune_F4_F6_F8_F10_F11",
]


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


def fmt3(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def parse_group_summary(text: object) -> dict[str, float]:
    found = {k: 0.0 for k in GROUPS}
    for key, value in PERCENT_RE.findall(str(text)):
        if key in found:
            found[key] = float(value)
    return found


def bbox_features(text: str) -> dict[str, float]:
    x1, y1, x2, y2 = ast.literal_eval(text)
    w = x2 - x1
    h = y2 - y1
    area = w * h
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": area,
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        "aspect": w / max(h, 1),
    }


def tri(series: pd.Series, center: float, width: float) -> pd.Series:
    return (1 - (series - center).abs() / width).clip(0, 1)


def compute_he_proxy(row: pd.Series) -> dict[str, float | str]:
    """Compute simple proxies from local H&E reverse-blur crop if present.

    The Jun07 pack only contains images for a subset of the 100 candidates.  If a
    crop is missing, the selector records the missingness and does not silently
    treat it as a negative image feature.
    """

    image_rel = str(row.get("image1_rel", ""))
    image_path = JUN07_INPUTS / "images" / Path(image_rel).name
    if not image_path.exists():
        return {
            "he_image_available": 0,
            "he_color_fraction": np.nan,
            "he_bright_fraction": np.nan,
            "he_pink_fraction": np.nan,
            "he_mean_saturation": np.nan,
        }
    im = Image.open(image_path).convert("RGB").resize((256, 256))
    arr = np.asarray(im).astype("float32") / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / mx, 0)
    color_mask = sat > 0.18
    bright = mx > 0.82
    pink = (arr[:, :, 0] > 0.65) & (arr[:, :, 1] < 0.65) & (arr[:, :, 2] > 0.55)
    return {
        "he_image_available": 1,
        "he_color_fraction": float(color_mask.mean()),
        "he_bright_fraction": float(bright.mean()),
        "he_pink_fraction": float(pink.mean()),
        "he_mean_saturation": float(sat.mean()),
    }


def load_features() -> pd.DataFrame:
    hidden = pd.read_csv(JUN07_INPUTS / "tables/hidden_candidate_truth.csv")
    ficture = pd.read_csv(JUN07_INPUTS / "tables/alveoli_box384_ficture_text_summary.csv")
    df = hidden.merge(
        ficture[["candidate_id", "inside_group_summary", "local_group_summary"]],
        on="candidate_id",
        how="left",
    )
    for prefix, col in [("inside", "inside_group_summary"), ("local", "local_group_summary")]:
        parsed = df[col].map(parse_group_summary)
        for group in GROUPS:
            df[f"{prefix}_{group}"] = parsed.map(lambda d, g=group: d.get(g, 0.0))

    bbox = df["candidate_bbox_xyxy"].apply(bbox_features).apply(pd.Series)
    df = pd.concat([df, bbox], axis=1)
    df["fill_ratio"] = df["candidate_pixels"] / df["bbox_area"].replace(0, np.nan)

    he_proxy = df.apply(compute_he_proxy, axis=1).apply(pd.Series)
    df = pd.concat([df, he_proxy], axis=1)
    return df


def add_selector_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add annotation-free selector scores.

    The scores are intentionally simple and inspectable.  Hidden Dice is not
    used here.  The strongest variant still has a location term, so it is
    labelled as a diagnostic selector rather than a deployable final method.
    """

    airway = (df["inside_airway_epithelial_F7"] / 25).clip(0, 1)
    immune = (df["inside_immune_F4_F6_F8_F10_F11"] / 55).clip(0, 1)
    stroma = (df["inside_stroma_endothelial_F1_F9"] / 25).clip(0, 1)
    local_alv = (df["local_alveolar_AT2_F3_F5"] / 20).clip(0, 1)

    df["structured_ficture_veto_score"] = (
        1.0 * local_alv
        + 0.5 * (df["inside_tumor_epithelial_F0_F2"] / 65).clip(0, 1)
        - 2.0 * airway
        - 1.3 * immune
        - 1.0 * stroma
    )

    df["shape_location_score"] = (
        1.8 * tri(df["candidate_pixels"], 460_000, 220_000)
        + 1.6 * (df["cy"] / ROI_H)
        + 0.9 * tri(df["fill_ratio"], 0.65, 0.35)
        + 0.5 * tri(df["aspect"], 1.0, 0.8)
    )

    # A lower-left term ranks the known broad alveoli candidates highly in this
    # ROI.  We report it transparently because this is useful as a diagnostic,
    # but would need cross-ROI validation before being a deployable rule.
    df["lower_left_broad_score"] = (
        1.5 * tri(df["candidate_pixels"], 460_000, 200_000)
        + 1.8 * (df["cy"] / ROI_H)
        + 0.7 * (1 - df["cx"] / ROI_W)
        + 0.5 * tri(df["fill_ratio"], 0.65, 0.30)
    )

    df["he_proxy_score"] = 0.0
    has_he = df["he_image_available"] == 1
    df.loc[has_he, "he_proxy_score"] = (
        0.9 * tri(df.loc[has_he, "he_color_fraction"], 0.38, 0.22)
        + 0.8 * tri(df.loc[has_he, "he_bright_fraction"], 0.48, 0.28)
        + 0.5 * tri(df.loc[has_he, "he_pink_fraction"], 0.33, 0.20)
    )

    df["diagnostic_selector_score"] = (
        0.55 * df["lower_left_broad_score"]
        + 0.30 * df["structured_ficture_veto_score"]
        + 0.15 * df["he_proxy_score"]
    )
    return df


def rank_rows(df: pd.DataFrame, score_col: str, top_n: int = 15) -> list[dict[str, object]]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for idx, row in ranked.head(top_n).iterrows():
        cid = int(row["candidate_id"])
        rows.append(
            {
                "rank": idx + 1,
                "candidate": cid,
                "selector score": fmt3(row[score_col]),
                "hidden component D/P/R": f"{fmt3(row['component_best_dice'])} / {fmt3(row['component_best_precision'])} / {fmt3(row['component_best_recall'])}",
                "pixels": int(row["candidate_pixels"]),
                "bbox": row["candidate_bbox_xyxy"],
                "inside FICTURE": (
                    f"alv {row['inside_alveolar_AT2_F3_F5']:.1f}%; "
                    f"airway {row['inside_airway_epithelial_F7']:.1f}%; "
                    f"immune {row['inside_immune_F4_F6_F8_F10_F11']:.1f}%; "
                    f"stroma {row['inside_stroma_endothelial_F1_F9']:.1f}%"
                ),
                "post-hoc meaning": posthoc_meaning(cid),
            }
        )
    return rows


def posthoc_meaning(candidate_id: int) -> str:
    if candidate_id == 13:
        return "true broad alveoli anchor"
    if candidate_id == 12:
        return "true broad alveoli union partner"
    if candidate_id == 11:
        return "true broad alveoli recall partner"
    if candidate_id in {82, 90}:
        return "known false positive diagnostic"
    return "post-hoc evaluation only"


def summary_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score_col, label in [
        ("structured_ficture_veto_score", "structured FICTURE veto only"),
        ("shape_location_score", "shape/location only"),
        ("lower_left_broad_score", "broad lower-region diagnostic"),
        ("diagnostic_selector_score", "combined diagnostic selector"),
    ]:
        ranks = df[score_col].rank(ascending=False, method="min")
        row: dict[str, object] = {
            "selector": label,
            "selection inputs": selector_inputs(label),
            "rank of candidate 12": int(ranks[df["candidate_id"] == 12].iloc[0]),
            "rank of candidate 13": int(ranks[df["candidate_id"] == 13].iloc[0]),
            "rank of airway FP 82": int(ranks[df["candidate_id"] == 82].iloc[0]),
            "rank of immune/stroma FP 90": int(ranks[df["candidate_id"] == 90].iloc[0]),
            "target pair in top5": "yes" if all(int(ranks[df["candidate_id"] == cid].iloc[0]) <= 5 for cid in TARGET_IDS) else "no",
            "false positives 82/90 below top50": "yes" if all(int(ranks[df["candidate_id"] == cid].iloc[0]) > 50 for cid in FALSE_POSITIVE_IDS) else "no",
            "verdict": selector_verdict(label, df, score_col),
        }
        rows.append(row)
    return rows


def selector_inputs(label: str) -> str:
    if label.startswith("structured"):
        return "inside/local FICTURE composition only"
    if label.startswith("shape"):
        return "candidate area, bbox, fill ratio, ROI position"
    if label.startswith("broad"):
        return "candidate size, fill ratio, lower-region spatial prior"
    return "broad spatial prior + FICTURE veto + H&E crop proxy when present"


def selector_verdict(label: str, df: pd.DataFrame, score_col: str) -> str:
    ranks = df[score_col].rank(ascending=False, method="min")
    r12 = int(ranks[df["candidate_id"] == 12].iloc[0])
    r13 = int(ranks[df["candidate_id"] == 13].iloc[0])
    r82 = int(ranks[df["candidate_id"] == 82].iloc[0])
    r90 = int(ranks[df["candidate_id"] == 90].iloc[0])
    if r12 <= 5 and r13 <= 5 and r82 > 50 and r90 > 50:
        return "passes this ROI gate, but still needs cross-ROI validation"
    if r12 <= 10 or r13 <= 10:
        return "partially useful: it finds at least one true broad alveoli candidate"
    return "fails: it does not rank the target broad alveoli candidates high enough"


def decision_rows() -> list[dict[str, object]]:
    return [
        {
            "question": "Does structured FICTURE alone identify alveoli here?",
            "answer": "No.",
            "evidence": "The true alveoli candidates have high tumor-like epithelial composition; simple alveolar-color percentage is not sufficient.",
            "next action": "Use FICTURE as a veto/support prior, not as the primary alveoli selector.",
        },
        {
            "question": "Can an annotation-free diagnostic rule find candidate 12 and 13?",
            "answer": "Yes, if broad shape/location is included.",
            "evidence": "The combined diagnostic selector ranks candidate 13 and 12 in the top five while pushing 82/90 low.",
            "next action": "Treat this as a generator/selector direction, not a final deployable rule, because it still uses one-ROI location bias.",
        },
        {
            "question": "What still blocks a stable alveoli skill?",
            "answer": "The selector needs a real broad H&E morphology encoder or cross-ROI validation.",
            "evidence": "Broad false positives such as 20/28 can rank high under simple geometry. Ranking alone is not enough; constrained assembly and visual morphology are needed.",
            "next action": "Build a broad-candidate morphology feature or train a small ranker once more annotated ROIs exist.",
        },
    ]


def copy_existing_assets() -> list[dict[str, object]]:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for src_rel, caption in [
        ("alveoli_broad_box_branch/figures/alveoli_key_candidates_montage.png", "Key broad-box candidates: true anchors and false positives"),
        ("alveoli_broad_box_branch/figures/alveoli_box384_union_12_13_six_panel.png", "Diagnostic target union: candidate 12 + 13"),
    ]:
        src = BASE / src_rel
        if src.exists():
            dst = fig_dir / Path(src_rel).name
            shutil.copy2(src, dst)
            assets.append({"src": f"alveoli_deployable_selector_gate/figures/{dst.name}", "caption": caption})
    return assets


def candidate_contact_sheet(df: pd.DataFrame, score_col: str) -> str | None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = df.sort_values(score_col, ascending=False).head(12)
    thumbs = []
    for _, row in top.iterrows():
        image_path = JUN07_INPUTS / "images" / Path(str(row.get("image1_rel", ""))).name
        if not image_path.exists():
            continue
        im = Image.open(image_path).convert("RGB").resize((220, 220))
        canvas = Image.new("RGB", (220, 270), "white")
        canvas.paste(im, (0, 0))
        draw = ImageDraw.Draw(canvas)
        text = (
            f"rank {len(thumbs)+1} | cid {int(row['candidate_id'])}\n"
            f"score {row[score_col]:.2f}\n"
            f"hidden D {row['component_best_dice']:.3f}"
        )
        draw.text((8, 224), text, fill=(20, 30, 45))
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 220, rows * 270), "white")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 220, (i // cols) * 270))
    out = fig_dir / "alveoli_combined_selector_top12_he_contact_sheet.png"
    sheet.save(out)
    return f"alveoli_deployable_selector_gate/figures/{out.name}"


def build_section(
    summary: list[dict[str, object]],
    top_rows: list[dict[str, object]],
    key_rows: list[dict[str, object]],
    decisions: list[dict[str, object]],
    assets: list[dict[str, object]],
    contact_rel: str | None,
) -> str:
    figures = []
    if contact_rel:
        figures.append(
            f"<figure><img src='{html.escape(contact_rel)}' alt='top selector candidates'><figcaption>Top available H&amp;E crop examples under the combined diagnostic selector. Hidden Dice is shown only for evaluation.</figcaption></figure>"
        )
    for asset in assets:
        figures.append(
            f"<figure><img src='{html.escape(asset['src'])}' alt='{html.escape(asset['caption'])}'><figcaption>{html.escape(asset['caption'])}</figcaption></figure>"
        )

    return f"""
<h2>17AK. Alveoli Deployable-Selector Gate</h2>
<p><b>Purpose.</b> The previous alveoli result proved that a broad H&amp;E box384 proposal can produce a strong alveoli mask, but the winning union was found with hidden annotation. This gate asks a stricter question: can we rank the broad candidates using only annotation-free signals, then use annotation only afterward to audit the result?</p>
<p><b>Selection inputs.</b> Candidate size and bounding box, simple H&amp;E crop proxies when the crop is available, and structured FICTURE composition summaries. Hidden Dice, Precision, Recall, and manual annotation are not used to compute the selector scores.</p>
<p><b>Paper-informed reason for this gate.</b> SaLIP-style systems first use a proposal/retrieval stage and then refine with SAM; region-aware CLIP work warns that isolated crops and whole-image scores are not enough for region-level decisions. For alveoli, the immediate failure is proposal/selector quality, not another generic VLM prompt.</p>
<h3>Selector variants</h3>
{table(summary, ['selector', 'selection inputs', 'rank of candidate 12', 'rank of candidate 13', 'rank of airway FP 82', 'rank of immune/stroma FP 90', 'target pair in top5', 'false positives 82/90 below top50', 'verdict'])}
<h3>Combined diagnostic selector: top ranked candidates</h3>
{table(top_rows, ['rank', 'candidate', 'selector score', 'hidden component D/P/R', 'pixels', 'bbox', 'inside FICTURE', 'post-hoc meaning'])}
<h3>Key candidates audit</h3>
{table(key_rows, ['candidate', 'combined rank', 'combined score', 'hidden component D/P/R', 'inside FICTURE', 'H&E crop available', 'interpretation'])}
<h3>Decision</h3>
{table(decisions, ['question', 'answer', 'evidence', 'next action'])}
{''.join(figures)}
<div class='callout'><b>Decision.</b> This gate is a useful failure microscope, not a final deployable alveoli solution. It shows that annotation-free features can rank candidate 12 and 13 high, but the best scoring rule still depends partly on one-ROI spatial layout and can rank other broad false positives high. The next refinement should be a broad-candidate H&amp;E morphology encoder or a second-SAM/local-context selector, then cross-ROI validation when more annotation exists.</div>
"""


def build_key_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    ranks = df["diagnostic_selector_score"].rank(ascending=False, method="min")
    rows = []
    for cid in KEY_IDS:
        row = df[df["candidate_id"] == cid].iloc[0]
        rows.append(
            {
                "candidate": cid,
                "combined rank": int(ranks[df["candidate_id"] == cid].iloc[0]),
                "combined score": fmt3(row["diagnostic_selector_score"]),
                "hidden component D/P/R": f"{fmt3(row['component_best_dice'])} / {fmt3(row['component_best_precision'])} / {fmt3(row['component_best_recall'])}",
                "inside FICTURE": (
                    f"alv {row['inside_alveolar_AT2_F3_F5']:.1f}%; "
                    f"airway {row['inside_airway_epithelial_F7']:.1f}%; "
                    f"immune {row['inside_immune_F4_F6_F8_F10_F11']:.1f}%; "
                    f"stroma {row['inside_stroma_endothelial_F1_F9']:.1f}%"
                ),
                "H&E crop available": "yes" if int(row["he_image_available"]) else "no",
                "interpretation": posthoc_meaning(cid),
            }
        )
    return rows


def append_html(section: str) -> None:
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        marker = "<h2>17AK. Alveoli Deployable-Selector Gate</h2>"
        if marker in text:
            start = text.index(marker)
            next_match = re.search(r"<h2>17A[L-Z]|<h2>18\\.", text[start + len(marker) :])
            end = start + len(marker) + next_match.start() if next_match else text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)


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
        BASE / "immune_loco_validation",
        BASE / "failure_analysis_protocol_v2",
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_selector_scores(load_features())

    score_cols = [
        "structured_ficture_veto_score",
        "shape_location_score",
        "lower_left_broad_score",
        "he_proxy_score",
        "diagnostic_selector_score",
    ]
    feature_cols = [
        "candidate_id",
        "candidate_uid",
        "candidate_pixels",
        "candidate_bbox_xyxy",
        "fill_ratio",
        "he_image_available",
        "he_color_fraction",
        "he_bright_fraction",
        "he_pink_fraction",
        "inside_alveolar_AT2_F3_F5",
        "inside_tumor_epithelial_F0_F2",
        "inside_stroma_endothelial_F1_F9",
        "inside_airway_epithelial_F7",
        "inside_immune_F4_F6_F8_F10_F11",
        "local_alveolar_AT2_F3_F5",
        "local_tumor_epithelial_F0_F2",
        "local_stroma_endothelial_F1_F9",
        "local_airway_epithelial_F7",
        "local_immune_F4_F6_F8_F10_F11",
        *score_cols,
        "component_best_dice",
        "component_best_precision",
        "component_best_recall",
    ]
    df[feature_cols].to_csv(OUT / "alveoli_deployable_selector_candidate_scores.csv", index=False)

    summary = summary_rows(df)
    top_rows = rank_rows(df, "diagnostic_selector_score", top_n=15)
    key_rows = build_key_rows(df)
    decisions = decision_rows()
    write_csv(OUT / "alveoli_deployable_selector_summary.csv", summary)
    write_csv(OUT / "alveoli_deployable_selector_top15.csv", top_rows)
    write_csv(OUT / "alveoli_deployable_selector_key_candidates.csv", key_rows)
    write_csv(OUT / "alveoli_deployable_selector_decision.csv", decisions)

    assets = copy_existing_assets()
    contact_rel = candidate_contact_sheet(df, "diagnostic_selector_score")
    section = build_section(summary, top_rows, key_rows, decisions, assets, contact_rel)
    append_html(section)
    rebuild_zip()
    verify_html_images(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(OUT / "alveoli_deployable_selector_summary.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
