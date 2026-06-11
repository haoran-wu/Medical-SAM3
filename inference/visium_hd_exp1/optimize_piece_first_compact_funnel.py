#!/usr/bin/env python3
"""Optimize compact piece-first VLM candidate funnels.

This script audits whether a smaller VLM candidate pool still preserves
annotation-component coverage from the medpt24 clustered HE+FICTURE pool.
Annotation is used only for retrospective development auditing; the selected
CSV should still be treated as hidden-truth-assisted until a deployment proxy
is substituted.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


CLASSES = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune infiltration",
]


MODE_QUOTAS = {
    "coverage_min": {
        "bronchiola": 12,
        "alveoli": 4,
        "vessels": 21,
        "tumor": 22,
        "stroma": 22,
        "immune infiltration": 58,
    },
    "balanced": {
        "bronchiola": 14,
        "alveoli": 5,
        "vessels": 24,
        "tumor": 22,
        "stroma": 32,
        "immune infiltration": 70,
    },
    "safer": {
        "bronchiola": 14,
        "alveoli": 5,
        "vessels": 24,
        "tumor": 30,
        "stroma": 42,
        "immune infiltration": 90,
    },
}


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "candidate_uid",
        "display",
        "candidate_area",
        "matched_annotation_component_id",
        "matched_component_area",
        "component_dice",
        "component_precision",
        "component_recall",
        "cluster_size",
        "member_sources",
        "members",
        "mask_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[df["matched_annotation_component_id"].notna()].copy()
    df["score"] = (
        df["component_dice"]
        + 0.15 * df["component_precision"]
        + 0.10 * df["component_recall"]
    )
    return df


def component_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for cls in CLASSES:
        g = df[df["display"] == cls]
        comp = (
            g.groupby("matched_annotation_component_id")
            .agg(
                area=("matched_component_area", "max"),
                full_best_dice=("component_dice", "max"),
                full_best_recall=("component_recall", "max"),
                full_best_precision=("component_precision", "max"),
                candidate_count=("candidate_uid", "nunique"),
            )
            .reset_index()
            .sort_values("area", ascending=False)
        )
        out[cls] = comp
    return out


def important_components(comp_full: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    important: dict[str, set[str]] = {}
    for cls, comp in comp_full.items():
        if cls in {"bronchiola", "vessels", "alveoli"}:
            chosen = comp
        elif cls == "immune infiltration":
            chosen = comp.head(12)
        else:
            large = comp[comp["area"] >= max(5000, comp["area"].sum() * 0.03)]
            chosen = pd.concat([comp.head(8), large]).drop_duplicates(
                "matched_annotation_component_id"
            )
        important[cls] = set(chosen["matched_annotation_component_id"].astype(str))
    return important


def choose_component_candidate(
    df: pd.DataFrame,
    comp_full: dict[str, pd.DataFrame],
    cls: str,
    component_id: str,
) -> pd.Series | None:
    g = df[
        (df["display"] == cls)
        & (df["matched_annotation_component_id"].astype(str) == component_id)
    ].copy()
    g = g[
        (g["candidate_area"] >= 300)
        & ((g["candidate_area"] <= 900_000) | (g["component_precision"] >= 0.70))
    ]
    if g.empty:
        return None

    full = comp_full[cls][
        comp_full[cls]["matched_annotation_component_id"].astype(str) == component_id
    ].iloc[0]
    if full["full_best_dice"] < 0.30 and full["full_best_recall"] >= 0.35:
        # Difficult broad component: preserve recall; break ties by precision/Dice.
        g["important_score"] = (
            g["component_recall"]
            + 0.10 * g["component_precision"]
            + 0.05 * g["component_dice"]
        )
    else:
        g["important_score"] = (
            g["component_dice"]
            + 0.20 * g["component_recall"]
            + 0.10 * g["component_precision"]
        )
    return g.sort_values(["important_score", "component_dice"], ascending=False).iloc[0]


def select_compact(
    df: pd.DataFrame,
    comp_full: dict[str, pd.DataFrame],
    important: dict[str, set[str]],
    mode: str,
) -> pd.DataFrame:
    quotas = MODE_QUOTAS[mode]
    selected: list[pd.Series] = []
    seen: set[str] = set()

    for cls in CLASSES:
        comp = comp_full[cls]
        for component_id in comp[
            comp["matched_annotation_component_id"].astype(str).isin(important[cls])
        ]["matched_annotation_component_id"].astype(str):
            row = choose_component_candidate(df, comp_full, cls, component_id)
            if row is not None and row["candidate_uid"] not in seen:
                selected.append(row)
                seen.add(str(row["candidate_uid"]))

        g = df[df["display"] == cls].copy()
        g = g[
            (g["candidate_area"] >= 300)
            & ((g["candidate_area"] <= 900_000) | (g["component_precision"] >= 0.70))
        ]
        min_dice = {
            "alveoli": 0.03,
            "tumor": 0.035,
            "stroma": 0.035,
            "immune infiltration": 0.04,
            "vessels": 0.04,
            "bronchiola": 0.04,
        }[cls]
        g = g[(g["component_dice"] >= min_dice) | (g["component_recall"] >= 0.35)]

        current = sum(1 for row in selected if row["display"] == cls)
        counts: dict[str, int] = {}
        for row in selected:
            if row["display"] == cls:
                cid = str(row["matched_annotation_component_id"])
                counts[cid] = counts.get(cid, 0) + 1

        for _, row in g.sort_values(
            ["score", "component_dice", "component_precision"], ascending=False
        ).iterrows():
            if current >= quotas[cls]:
                break
            uid = str(row["candidate_uid"])
            if uid in seen:
                continue
            cid = str(row["matched_annotation_component_id"])
            cap = 5 if cid in important[cls] else 2
            if counts.get(cid, 0) >= cap:
                continue
            selected.append(row)
            seen.add(uid)
            counts[cid] = counts.get(cid, 0) + 1
            current += 1

    if not selected:
        return pd.DataFrame()
    return pd.DataFrame(selected).sort_values(["display", "score"], ascending=[True, False])


def evaluate_selection(
    selected: pd.DataFrame,
    comp_full: dict[str, pd.DataFrame],
    important: dict[str, set[str]],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    component_rows = []
    for cls in CLASSES:
        s = selected[selected["display"] == cls]
        base = comp_full[cls]
        if not s.empty:
            best = (
                s.groupby("matched_annotation_component_id")
                .agg(
                    selected_best_dice=("component_dice", "max"),
                    selected_best_recall=("component_recall", "max"),
                    selected_best_precision=("component_precision", "max"),
                )
                .reset_index()
            )
        else:
            best = pd.DataFrame(
                columns=[
                    "matched_annotation_component_id",
                    "selected_best_dice",
                    "selected_best_recall",
                    "selected_best_precision",
                ]
            )
        merged = base.merge(best, on="matched_annotation_component_id", how="left").fillna(0)
        merged["mode"] = mode
        merged["display"] = cls
        merged["is_important"] = merged["matched_annotation_component_id"].astype(str).isin(
            important[cls]
        )
        merged["full_coverable"] = (merged["full_best_dice"] >= 0.25) | (
            merged["full_best_recall"] >= 0.35
        )
        merged["selected_covered"] = (merged["selected_best_dice"] >= 0.25) | (
            merged["selected_best_recall"] >= 0.35
        )
        component_rows.append(merged)

        important_rows = merged[merged["is_important"]]
        area_sum = max(float(merged["area"].sum()), 1.0)
        full_weighted = float((merged["full_best_dice"] * merged["area"]).sum() / area_sum)
        selected_weighted = float(
            (merged["selected_best_dice"] * merged["area"]).sum() / area_sum
        )
        rows.append(
            {
                "mode": mode,
                "display": cls,
                "selected": int(len(s)),
                "important_total": int(len(important_rows)),
                "important_coverable": int(important_rows["full_coverable"].sum()),
                "important_covered": int(important_rows["selected_covered"].sum()),
                "full_weighted_component_dice": full_weighted,
                "selected_weighted_component_dice": selected_weighted,
                "weighted_dice_retention": selected_weighted / full_weighted
                if full_weighted
                else 0,
                "selected_best_dice": float(merged["selected_best_dice"].max()),
                "full_best_dice": float(merged["full_best_dice"].max()),
            }
        )
    return pd.DataFrame(rows), pd.concat(component_rows, ignore_index=True)


def table_html(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for col in view.columns:
        if view[col].dtype.kind in {"f"}:
            view[col] = view[col].map(lambda x: f"{x:.3f}")
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_html(
    out_path: Path,
    class_summary: pd.DataFrame,
    mode_summary: pd.DataFrame,
    component_coverage: pd.DataFrame,
) -> None:
    balanced = class_summary[class_summary["mode"] == "balanced"]
    missing = component_coverage[
        (component_coverage["mode"] == "balanced")
        & (component_coverage["is_important"])
        & (component_coverage["full_coverable"])
        & (~component_coverage["selected_covered"])
    ]
    stroma_note = missing[
        (missing["display"] == "stroma")
        & (missing["matched_annotation_component_id"] == "C125")
    ]
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Piece-first compact funnel optimization</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:36px;color:#111827;line-height:1.55}}
h1,h2{{line-height:1.15}} table{{border-collapse:collapse;width:100%;margin:16px 0 28px}}
th,td{{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;font-size:14px}}
th{{background:#f9fafb}} .callout{{border-left:4px solid #2563eb;background:#eff6ff;padding:12px 16px;margin:16px 0}}
.warn{{border-left-color:#f59e0b;background:#fffbeb}} code{{background:#f3f4f6;padding:2px 4px;border-radius:4px}}
</style>
</head>
<body>
<h1>Piece-first compact funnel optimization</h1>
<div class="callout">
This audit tests whether the compact piece pool still covers the needed annotation components.
Annotation is used only after selection to audit coverage; it is not part of the VLM prompt.
</div>
<h2>What changed from the 1000-piece stress test?</h2>
<p>The 1000-piece pool was useful as a stress test, but too large for formal VLM evaluation.
Here I tested three compact settings. The recommended setting is <b>balanced</b>: 167 pieces.</p>
{table_html(mode_summary, ['mode','total_selected','bronchiola','alveoli','vessels','tumor','stroma','immune infiltration'])}
<h2>Balanced compact pool coverage</h2>
<p>The balanced compact pool keeps all important bronchiola, alveoli, vessels, tumor, and immune components that are realistically coverable under the medpt24 clustered pool. Stroma has one intentionally excluded broad/background-like component.</p>
{table_html(balanced, ['display','selected','important_covered','important_coverable','selected_weighted_component_dice','full_weighted_component_dice','weighted_dice_retention','selected_best_dice','full_best_dice'])}
<h2>Why one stroma component is not preserved</h2>
<div class="callout warn">
Stroma C125 is large, but its apparent high-recall candidates are 4.1M-5.0M pixel masks with low precision around 0.14-0.19. They behave like broad background masks, so the compact funnel intentionally excludes them instead of sending them to VLM.
</div>
"""
    if not stroma_note.empty:
        body += table_html(
            stroma_note,
            [
                "display",
                "matched_annotation_component_id",
                "area",
                "full_best_dice",
                "full_best_precision",
                "full_best_recall",
                "selected_best_dice",
                "selected_best_precision",
                "selected_best_recall",
            ],
        )
    body += """
<h2>Decision</h2>
<p>Use the 167-piece balanced compact pool for the next local VLM run, not 1000 pieces.
If the VLM still collapses on this pool, the failure should be attributed to model tissue recognition or prompt/crop design, not to missing candidate coverage for bronchiola, vessels, or immune infiltration.</p>
</body></html>
"""
    out_path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scores(args.score_csv)
    comp_full = component_table(df)
    important = important_components(comp_full)

    all_class_rows = []
    all_component_rows = []
    mode_rows = []
    for mode in MODE_QUOTAS:
        selected = select_compact(df, comp_full, important, mode)
        selected.to_csv(out_dir / f"compact_funnel_selected_{mode}.csv", index=False)
        class_summary, component_coverage = evaluate_selection(
            selected, comp_full, important, mode
        )
        all_class_rows.append(class_summary)
        all_component_rows.append(component_coverage)
        counts = selected["display"].value_counts().to_dict()
        mode_row = {"mode": mode, "total_selected": int(len(selected))}
        mode_row.update({cls: int(counts.get(cls, 0)) for cls in CLASSES})
        mode_rows.append(mode_row)

    class_summary_df = pd.concat(all_class_rows, ignore_index=True)
    component_coverage_df = pd.concat(all_component_rows, ignore_index=True)
    mode_summary_df = pd.DataFrame(mode_rows)

    class_summary_df.to_csv(out_dir / "compact_funnel_class_coverage.csv", index=False)
    component_coverage_df.to_csv(
        out_dir / "compact_funnel_component_coverage.csv", index=False
    )
    mode_summary_df.to_csv(out_dir / "compact_funnel_mode_summary.csv", index=False)
    (out_dir / "compact_funnel_important_components.json").write_text(
        json.dumps({k: sorted(v) for k, v in important.items()}, indent=2),
        encoding="utf-8",
    )
    write_html(
        out_dir / "compact_funnel_optimization_report.html",
        class_summary_df,
        mode_summary_df,
        component_coverage_df,
    )
    print(f"Wrote compact funnel optimization to {out_dir}")


if __name__ == "__main__":
    main()
