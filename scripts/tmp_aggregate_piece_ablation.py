#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


BASE = Path("/home/hw646/Medical-SAM3_remote_runs/visium_hd_exp1/piece_first_prompt_ablation_20260604")
CLASSES = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


def class_match(series: pd.Series, cls: str) -> pd.Series:
    text = series.astype(str)
    return (text == cls) | text.str.contains(cls.replace("_", " "), case=False, na=False)


def aggregate_assembly() -> None:
    rows = []
    roots = []
    for dirname in ["piece_assembly_outputs", "piece_assembly_outputs_strong", "piece_assembly_outputs_api"]:
        root = BASE / dirname
        if root.exists():
            roots.extend([p for p in root.glob("*") if p.is_dir()])
    for root in sorted(roots):
        path = root / "assembly_summary.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"skip empty predictions: {path}")
            continue
        dice_col = "dice" if "dice" in df.columns else "Dice"
        prec_col = "precision" if "precision" in df.columns else "Precision"
        rec_col = "recall" if "recall" in df.columns else "Recall"
        sel_col = "selected_pieces" if "selected_pieces" in df.columns else (
            "selected_piece_count" if "selected_piece_count" in df.columns else None
        )
        for _, row in df.iterrows():
            cls = row.get("class", row.get("display", ""))
            rows.append(
                {
                    "run": root.name,
                    "class": cls,
                    "selected": row.get(sel_col, "") if sel_col else "",
                    "dice": row.get(dice_col, ""),
                    "precision": row.get(prec_col, ""),
                    "recall": row.get(rec_col, ""),
                }
            )
    out = pd.DataFrame(rows)
    print(f"N assembly rows {len(out)} runs {out['run'].nunique() if len(out) else 0}")
    if out.empty:
        return
    out["dice_num"] = pd.to_numeric(out["dice"], errors="coerce")
    for cls in CLASSES:
        sub = out[class_match(out["class"], cls)]
        if sub.empty:
            continue
        print(f"\n### {cls}")
        best = sub.sort_values("dice_num", ascending=False).head(8)
        print(best[["run", "selected", "dice", "precision", "recall"]].to_string(index=False))


def aggregate_vlm() -> None:
    rows = []
    roots = []
    for dirname in ["local_vlm_outputs", "openrouter_gpt55_outputs"]:
        root = BASE / dirname
        if root.exists():
            roots.extend([p for p in root.glob("*") if p.is_dir()])
    for root in sorted(roots):
        path = root / "per_candidate_predictions.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"skip empty predictions: {path}")
            continue
        pred_col = "predicted_class" if "predicted_class" in df.columns else "predicted_label"
        score_cols = [c for c in CLASSES if c in df.columns]
        unique_vectors = df[score_cols].astype(str).agg("|".join, axis=1).nunique() if score_cols else ""
        correct = 0
        if "is_correct" in df.columns:
            correct = int(sum(str(x).lower() in {"true", "1", "yes"} for x in df["is_correct"]))
        rows.append(
            {
                "run": root.name,
                "n": len(df),
                "correct": correct,
                "unique_vectors": unique_vectors,
                "pred_dist": dict(df[pred_col].fillna("").value_counts()),
            }
        )
    out = pd.DataFrame(rows)
    print(f"\nVLM runs {len(out)}")
    if out.empty:
        return
    out["unique_num"] = pd.to_numeric(out["unique_vectors"], errors="coerce")
    for _, row in out.sort_values(["correct", "unique_num"], ascending=False).head(25).iterrows():
        print(
            row["run"],
            "correct",
            f"{int(row['correct'])}/{int(row['n'])}",
            "unique",
            row["unique_vectors"],
            "dist",
            row["pred_dist"],
        )


def main() -> None:
    aggregate_assembly()
    aggregate_vlm()


if __name__ == "__main__":
    main()
