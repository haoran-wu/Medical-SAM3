#!/usr/bin/env python3
"""Render a direct CLIP-vs-VLM hit-test report on a hand-picked candidate pool.

This report is intentionally not a second-stage reranking report.  It compares
CLIP and VLM as direct selectors on the same small GOOD/MID/BAD candidate set.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


LABEL_ORDER = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune infiltration",
]


def parse_named_path(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected NAME:/path/to/vlm_candidate_scores.csv")
    name, path = value.split(":", 1)
    return name.strip(), Path(path)


def verify_pass_official(path: Path) -> None:
    data = json.loads(path.read_text())
    if data.get("status") != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {path}")


def candidate_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["mask_path"].astype(str)
        .str.replace("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/", "", regex=False)
        .str.lstrip("/")
    )


def load_table(fresh_csv: Path, vlm_scores: List[Tuple[str, Path]]) -> pd.DataFrame:
    table = pd.read_csv(fresh_csv).copy()
    table["mask_key"] = candidate_key(table)
    table["vlm_label"] = table["label_slug"].astype(str)
    table = table.drop(columns=[c for c in ["vlm_score", "vlm_fused_score", "vlm_reason"] if c in table])
    table = table.rename(
        columns={
            "clip_he_score": "CLIP H&E",
            "clip_ficture_semantic_score": "CLIP FICTURE semantic",
            "clip_mean_score": "CLIP mean",
        }
    )

    for name, path in vlm_scores:
        vlm = pd.read_csv(path).copy()
        vlm["mask_key"] = candidate_key(vlm)
        vlm["vlm_label"] = vlm["label"].astype(str)
        keep = vlm[["vlm_label", "mask_key", "vlm_score", "reason"]].rename(
            columns={"vlm_score": f"{name} VLM", "reason": f"{name} reason"}
        )
        table = table.merge(keep, on=["vlm_label", "mask_key"], how="left")
    return table


def score_columns(table: pd.DataFrame) -> List[str]:
    cols = ["CLIP H&E", "CLIP FICTURE semantic", "CLIP mean"]
    cols.extend(c for c in table.columns if c.endswith(" VLM"))
    return [c for c in cols if c in table.columns]


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label in LABEL_ORDER:
        part = table[table["label"] == label].copy()
        if part.empty:
            continue
        for col in score_columns(table):
            scored = part.dropna(subset=[col]).copy()
            if scored.empty:
                rows.append(
                    {
                        "label": label,
                        "selector": col,
                        "top_pick": "no score",
                        "top_true_dice": np.nan,
                        "good_rank": np.nan,
                        "good_score": np.nan,
                        "unique_scores": 0,
                        "interpretation": "missing scores",
                    }
                )
                continue
            scored = scored.sort_values([col, "true_dice"], ascending=[False, False])
            top_score = float(scored.iloc[0][col])
            top_tie = scored[np.isclose(scored[col].astype(float), top_score)]
            good = scored[scored["manual_pick"] == "GOOD"]
            good_score = float(good.iloc[0][col]) if not good.empty else np.nan
            better = scored[scored[col].astype(float) > good_score] if not good.empty else scored
            good_rank = int(len(better) + 1) if not good.empty else np.nan
            unique_scores = int(scored[col].round(8).nunique())
            top_pick = ",".join(top_tie["manual_pick"].astype(str).tolist())
            if unique_scores <= 1:
                interpretation = "uninformative tie"
            elif good_rank == 1 and len(top_tie) == 1:
                interpretation = "GOOD selected top-1"
            elif good_rank == 1:
                interpretation = "GOOD tied for top"
            else:
                interpretation = "missed GOOD"
            rows.append(
                {
                    "label": label,
                    "selector": col,
                    "top_pick": top_pick,
                    "top_true_dice": float(top_tie.iloc[0]["true_dice"]),
                    "good_rank": good_rank,
                    "good_score": good_score,
                    "unique_scores": unique_scores,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def local_mask_path(output_dir: Path, mask_key: str) -> Path | None:
    path = output_dir / mask_key
    if path.exists():
        return path
    return None


def render_thumbnail(he: Image.Image, mask_path: Path, out_path: Path) -> None:
    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask) > 0
    if not arr.any():
        return
    ys, xs = np.where(arr)
    pad = 180
    x0, x1 = max(0, int(xs.min()) - pad), min(mask.width, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(mask.height, int(ys.max()) + pad)
    crop = he.crop((x0, y0, x1, y1)).convert("RGBA")
    crop_mask = mask.crop((x0, y0, x1, y1))
    overlay = Image.new("RGBA", crop.size, (0, 160, 255, 0))
    overlay.putalpha(crop_mask.point(lambda v: 105 if v else 0))
    crop = Image.alpha_composite(crop, overlay)
    crop.thumbnail((360, 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(out_path, quality=92)


def render_html(table: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, title: str) -> None:
    cols = score_columns(table)
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 28px; color: #202124; }
h1 { font-size: 28px; margin-bottom: 4px; }
h2 { margin-top: 30px; border-top: 1px solid #ddd; padding-top: 18px; }
.note { max-width: 1050px; line-height: 1.45; color: #3c4043; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 22px; font-size: 14px; }
th, td { border-bottom: 1px solid #e0e0e0; padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 700; position: sticky; top: 0; }
.good { background: #e8f5e9; }
.mid { background: #fff8e1; }
.bad { background: #ffebee; }
.score { font-variant-numeric: tabular-nums; white-space: nowrap; }
.thumb { width: 180px; max-height: 150px; object-fit: contain; border: 1px solid #ddd; background: #fff; }
.miss { color: #b3261e; font-weight: 700; }
.hit { color: #137333; font-weight: 700; }
.tie { color: #9a6700; font-weight: 700; }
"""
    html_parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{css}</style>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='note'><b>Experiment design.</b> This is a direct retrieval/hit-test on the same hand-picked candidate pool. "
        "For each tissue class we mixed one known GOOD candidate, one MID candidate, and one BAD cross-label candidate. "
        "CLIP and each VLM see the same candidates and are judged only by whether their own score puts the GOOD candidate on top. "
        "No <code>vlm_fused_score</code>, no second-stage reranking, and no annotation mask is shown to the models.</p>",
        "<p class='note'><b>Scores.</b> CLIP scores were freshly recomputed on these exact masks. VLM columns are pure parsed <code>vlm_score</code> values. "
        "Truth Dice/Precision/Recall are hidden labels used only after scoring to evaluate whether retrieval worked.</p>",
        "<h2>Selector Summary</h2>",
        "<table><thead><tr><th>label</th><th>selector</th><th>top pick(s)</th><th>top Dice</th><th>GOOD rank</th><th>GOOD score</th><th>unique scores</th><th>interpretation</th></tr></thead><tbody>",
    ]
    for _, row in summary.iterrows():
        interp = str(row["interpretation"])
        klass = "hit" if "top-1" in interp else "tie" if "tie" in interp else "miss"
        html_parts.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['selector']))}</td>"
            f"<td>{html.escape(str(row['top_pick']))}</td>"
            f"<td class='score'>{row['top_true_dice']:.3f}</td>"
            f"<td class='score'>{row['good_rank']}</td>"
            f"<td class='score'>{row['good_score']:.4f}</td>"
            f"<td class='score'>{row['unique_scores']}</td>"
            f"<td class='{klass}'>{html.escape(interp)}</td>"
            "</tr>"
        )
    html_parts.append("</tbody></table>")

    for label in LABEL_ORDER:
        part = table[table["label"] == label].copy()
        if part.empty:
            continue
        html_parts.append(f"<h2>{html.escape(label)}</h2>")
        headers = ["candidate view", "manual pick", "source candidate", "source class", "true Dice", "Precision", "Recall"] + cols
        html_parts.append("<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>")
        for _, row in part.iterrows():
            pick = str(row["manual_pick"])
            row_class = pick.lower()
            img = ""
            if thumb := row.get("thumbnail"):
                img = f"<img class='thumb' src='{html.escape(str(thumb))}'>"
            cells = [
                img,
                html.escape(pick),
                html.escape(str(row["candidate"])),
                html.escape(str(row["original_best_for"])),
                f"{row['true_dice']:.3f}",
                f"{row['true_precision']:.3f}",
                f"{row['true_recall']:.3f}",
            ]
            for col in cols:
                val = row.get(col)
                cells.append("" if pd.isna(val) else f"{float(val):.4f}")
            html_parts.append("<tr class='%s'>%s</tr>" % (row_class, "".join(f"<td>{c}</td>" for c in cells)))
        html_parts.append("</tbody></table>")

    (output_dir / "index.html").write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-csv", type=Path, required=True)
    parser.add_argument("--vlm-score", action="append", type=parse_named_path, default=[])
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Manual direct retrieval hit-test: CLIP vs VLM")
    args = parser.parse_args()

    verify_pass_official(args.official_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(args.fresh_csv, args.vlm_score)

    he = Image.open(args.he_image).convert("RGB")
    thumb_dir = args.output_dir / "thumbs"
    for idx, row in table.iterrows():
        mask = local_mask_path(args.output_dir, row["mask_key"])
        if mask is None:
            continue
        thumb_name = f"{row['label_slug']}__{row['manual_pick'].lower()}__{Path(row['mask_key']).parent.name}__{Path(row['mask_key']).stem}.jpg"
        thumb_path = thumb_dir / thumb_name
        render_thumbnail(he, mask, thumb_path)
        table.at[idx, "thumbnail"] = thumb_path.relative_to(args.output_dir)

    summary = summarize(table)
    table.to_csv(args.output_dir / "manual_direct_retrieval_scores.csv", index=False)
    summary.to_csv(args.output_dir / "manual_direct_retrieval_summary.csv", index=False)
    shutil.copy2(args.fresh_csv, args.output_dir / "source_fresh_clip_examples.csv")
    render_html(table, summary, args.output_dir, args.title)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
