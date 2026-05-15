#!/usr/bin/env python3
"""Build presentation-ready training-history visualizations for VisiumHD Exp1."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class RunHistory:
    run: str
    group: str
    backbone: str
    label_set: str
    loss_variant: str
    strategy: str
    history_path: Path
    rows: list[dict]


def parse_run(path: Path) -> tuple[str, str, str, str, str]:
    run = path.parent.name
    group = path.parent.parent.name if path.parent.parent.name in {"loss_ablation", "training_strategy"} else "initial"
    backbone = "GigaPath" if "gigapath" in run else "ResNet50" if "resnet" in run else "Other"
    label_set = "3-class" if "3class" in run else "8-class" if "8class" in run else "metric"
    if "infonce_only" in run:
        loss_variant = "InfoNCE-only"
    elif "ce_only" in run:
        loss_variant = "CE-only"
    elif "infonce_l005" in run:
        loss_variant = "CE+InfoNCE 0.05"
    elif "infonce_l010" in run:
        loss_variant = "CE+InfoNCE 0.10"
    elif "infonce_l020" in run:
        loss_variant = "CE+InfoNCE 0.20"
    elif "infonce_l050" in run:
        loss_variant = "CE+InfoNCE 0.50"
    elif "metric" in run:
        loss_variant = "CE+MSE baseline"
    else:
        loss_variant = "other"
    if "last2" in run:
        strategy = "last-2 warmup/cosine"
    elif "last1" in run:
        strategy = "last-1 warmup/cosine"
    elif "frozen_warmcos" in run:
        strategy = "frozen warmup/cosine"
    elif "frozen_constant" in run:
        strategy = "frozen constant"
    else:
        strategy = "frozen/default"
    return group, backbone, label_set, loss_variant, strategy


def flatten_history(run: RunHistory) -> list[dict]:
    out = []
    for item in run.rows:
        epoch = item["epoch"]
        for split in ("train", "val"):
            block = item.get(split, {})
            metrics_expr = block.get("metrics", {}).get("expr", {})
            metrics_img = block.get("metrics", {}).get("img", {})
            out.append(
                {
                    "run": run.run,
                    "group": run.group,
                    "backbone": run.backbone,
                    "label_set": run.label_set,
                    "loss_variant": run.loss_variant,
                    "strategy": run.strategy,
                    "epoch": epoch,
                    "split": split,
                    "loss": block.get("loss"),
                    "ce_img": block.get("ce_img"),
                    "ce_expr": block.get("ce_expr"),
                    "align": block.get("align"),
                    "acc_img": block.get("acc_img"),
                    "acc_expr": block.get("acc_expr"),
                    "macro_f1_img": metrics_img.get("macro_f1"),
                    "macro_f1_expr": metrics_expr.get("macro_f1"),
                }
            )
    return out


def load_histories(root: Path) -> list[RunHistory]:
    histories = []
    for path in sorted((root / "training_runs").glob("**/history.json")):
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        group, backbone, label_set, loss_variant, strategy = parse_run(path)
        histories.append(
            RunHistory(
                run=path.parent.name,
                group=group,
                backbone=backbone,
                label_set=label_set,
                loss_variant=loss_variant,
                strategy=strategy,
                history_path=path,
                rows=rows,
            )
        )
    return histories


def style_axes(ax, title: str, ylabel: str = ""):
    ax.set_title(title, fontsize=11, weight="bold", color="#13213A")
    ax.set_xlabel("Epoch", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def plot_loss_ablation_grid(df: pd.DataFrame, outdir: Path):
    data = df[(df.group == "loss_ablation") & (df.split == "val") & (df.embed512 != True)]  # noqa: E712
    combos = [
        ("GigaPath", "3-class"),
        ("GigaPath", "8-class"),
        ("ResNet50", "3-class"),
        ("ResNet50", "8-class"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
    for ax, (backbone, label_set) in zip(axes.ravel(), combos):
        sub = data[(data.backbone == backbone) & (data.label_set == label_set)]
        for variant, g in sub.groupby("loss_variant", sort=False):
            g = g.sort_values("epoch")
            ax.plot(g.epoch, g.loss, marker="o", linewidth=1.8, label=variant)
        style_axes(ax, f"{backbone} {label_set}: validation total loss", "Val total loss")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Loss ablation training curves across all main runs", fontsize=16, weight="bold", color="#13213A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "all_loss_ablation_val_loss_grid.png", dpi=220)
    plt.close(fig)


def plot_loss_components_by_family(df: pd.DataFrame, outdir: Path):
    selected = df[
        (df.group == "loss_ablation")
        & (df.split == "val")
        & (df.backbone == "GigaPath")
        & (df.label_set.isin(["3-class", "8-class"]))
        & (df.loss_variant.isin(["CE-only", "CE+InfoNCE 0.10", "InfoNCE-only"]))
        & (df.embed512 != True)  # noqa: E712
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for row, label_set in enumerate(["3-class", "8-class"]):
        for col, variant in enumerate(["CE-only", "CE+InfoNCE 0.10", "InfoNCE-only"]):
            ax = axes[row, col]
            sub = selected[(selected.label_set == label_set) & (selected.loss_variant == variant)].sort_values("epoch")
            if sub.empty:
                ax.axis("off")
                continue
            ax.plot(sub["epoch"], sub["ce_img"], marker="o", label="CE image", color="#2F6F9F")
            ax.plot(sub["epoch"], sub["ce_expr"], marker="o", label="CE expression", color="#3F8F75")
            ax.plot(sub["epoch"], sub["align"], marker="o", label="alignment", color="#D97706")
            style_axes(ax, f"{label_set} / {variant}", "Val component")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("GigaPath validation loss components: CE vs alignment", fontsize=16, weight="bold", color="#13213A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "gigapath_loss_components_by_variant.png", dpi=220)
    plt.close(fig)


def plot_training_strategy(df: pd.DataFrame, outdir: Path):
    sub = df[(df.group == "training_strategy") & (df.split == "val") & (df.label_set == "3-class")]
    metrics = [
        ("loss", "Validation total loss"),
        ("acc_expr", "Validation expression accuracy"),
        ("macro_f1_expr", "Validation expression macro-F1"),
        ("acc_img", "Validation image accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for strategy, g in sub.groupby("strategy", sort=False):
            g = g.sort_values("epoch")
            ax.plot(g.epoch, g[metric], marker="o", linewidth=2, label=strategy)
        style_axes(ax, title, metric)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Training-strategy curves: frozen vs partial GigaPath fine-tuning", fontsize=16, weight="bold", color="#13213A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "training_strategy_all_curves.png", dpi=220)
    plt.close(fig)


def plot_best_epoch_summary(df: pd.DataFrame, outdir: Path):
    val = df[(df.split == "val") & (df.group.isin(["loss_ablation", "training_strategy"]))]
    idx = val.groupby("run")["macro_f1_expr"].idxmax()
    best = val.loc[idx].copy()
    best["run_family"] = best["backbone"] + " " + best["label_set"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [
        (axes[0], "macro_f1_expr", "Best validation expression macro-F1"),
        (axes[1], "acc_img", "Best validation image accuracy"),
    ]:
        summary = best.groupby(["run_family", "loss_variant"], as_index=False)[metric].max()
        families = list(summary["run_family"].drop_duplicates())
        x = range(len(families))
        for variant in ["CE-only", "CE+InfoNCE 0.05", "CE+InfoNCE 0.10", "CE+InfoNCE 0.20", "CE+InfoNCE 0.50", "InfoNCE-only"]:
            vals = []
            for fam in families:
                sub = summary[(summary.run_family == fam) & (summary.loss_variant == variant)]
                vals.append(float(sub[metric].iloc[0]) if not sub.empty else None)
            ax.plot(list(x), vals, marker="o", label=variant)
        ax.set_xticks(list(x), families, rotation=20, ha="right")
        style_axes(ax, title, metric)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Best-epoch metric overview across experiment families", fontsize=16, weight="bold", color="#13213A")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / "best_epoch_metric_overview.png", dpi=220)
    plt.close(fig)


def write_coverage(runs: list[RunHistory], outdir: Path):
    rows = []
    for run in runs:
        rows.append(
            {
                "run": run.run,
                "group": run.group,
                "backbone": run.backbone,
                "label_set": run.label_set,
                "loss_variant": run.loss_variant,
                "strategy": run.strategy,
                "epochs_recorded": len(run.rows),
                "history_path": str(run.history_path),
            }
        )
    out = outdir / "training_history_coverage.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/visium_hd_exp1"))
    parser.add_argument("--outdir", type=Path, default=Path("results/visium_hd_exp1/figures/group_meeting"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = load_histories(args.root)
    all_rows = []
    for run in runs:
        all_rows.extend(flatten_history(run))
    df = pd.DataFrame(all_rows)
    df["embed512"] = df["run"].str.contains("embed512")
    df.to_csv(args.outdir / "all_training_history_long.csv", index=False)
    write_coverage(runs, args.outdir)
    plot_loss_ablation_grid(df, args.outdir)
    plot_loss_components_by_family(df, args.outdir)
    plot_training_strategy(df, args.outdir)
    plot_best_epoch_summary(df, args.outdir)
    print(f"Loaded {len(runs)} histories")
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
