#!/usr/bin/env python3
"""Create presentation-ready plots for Visium HD Exp1 training/eval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = Path("/nfs/roberts/project/pi_xy48/hw646/Medical-SAM3/results/visium_hd_exp1")


RUNS = {
    "GigaPath crop64 8-class": "training_runs/cls_gigapath_crop64_metric_gpu_devel_project_v2_devel_11016957",
    "GigaPath crop128 8-class": "training_runs/cls_gigapath_crop128_metric_gpu_rtx6000_project_v2_11016973",
    "GigaPath crop128 3-class": "training_runs/cls_gigapath_crop128_3class_gpu_rtx6000_project_v2_11016974",
}

RETRIEVAL_RUNS = {
    "GigaPath crop128 3-class": "retrieval_eval/g128_3class_devel_11017417_2/metrics.json",
    "ResNet50 contrastive 8-class": "retrieval_eval/resnet_contrastive_crop64_11025621/metrics.json",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_classification_curves(results_dir: Path, out_dir: Path) -> list[str]:
    outputs: list[str] = []
    colors = {
        "GigaPath crop64 8-class": "#0f766e",
        "GigaPath crop128 8-class": "#b45309",
        "GigaPath crop128 3-class": "#1d4ed8",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for label, rel in RUNS.items():
        hist_path = results_dir / rel / "history.json"
        if not hist_path.exists():
            continue
        hist = load_json(hist_path)
        epochs = [x["epoch"] for x in hist]
        train_loss = [x["train"]["loss"] for x in hist]
        val_loss = [x["val"]["loss"] for x in hist]
        val_expr_acc = [x["val"]["acc_expr"] for x in hist]
        val_img_acc = [x["val"]["acc_img"] for x in hist]
        val_macro_f1_expr = [x["val"]["metrics"]["expr"]["macro_f1"] for x in hist]

        c = colors[label]
        axes[0].plot(epochs, train_loss, marker="o", color=c, linestyle="-", label=f"{label} train")
        axes[0].plot(epochs, val_loss, marker="s", color=c, linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, val_expr_acc, marker="o", color=c, label=label)
        axes[2].plot(epochs, val_img_acc, marker="o", color=c, label=label)
        axes[3].plot(epochs, val_macro_f1_expr, marker="o", color=c, label=label)

    titles = [
        "Dual-tower classification loss",
        "Validation expression accuracy",
        "Validation image accuracy",
        "Validation expression macro-F1",
    ]
    ylabels = ["Loss", "Accuracy", "Accuracy", "Macro-F1"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Visium HD Exp1: GigaPath alignment training curves", fontsize=16, weight="bold")
    fig.tight_layout()
    out = out_dir / "gigapath_training_curves.png"
    savefig(out)
    outputs.append(str(out))

    return outputs


def plot_loss_components(results_dir: Path, out_dir: Path) -> list[str]:
    outputs: list[str] = []
    colors = {
        "GigaPath crop64 8-class": "#0f766e",
        "GigaPath crop128 8-class": "#b45309",
        "GigaPath crop128 3-class": "#1d4ed8",
    }
    components = [
        ("ce_img", "Image CE loss"),
        ("ce_expr", "Expression CE loss"),
        ("align", "Embedding alignment MSE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, rel in RUNS.items():
        hist_path = results_dir / rel / "history.json"
        if not hist_path.exists():
            continue
        hist = load_json(hist_path)
        epochs = [x["epoch"] for x in hist]
        c = colors[label]
        for ax, (key, title) in zip(axes, components):
            train_values = [x["train"][key] for x in hist]
            val_values = [x["val"][key] for x in hist]
            ax.plot(epochs, train_values, marker="o", color=c, linestyle="-", label=f"{label} train")
            ax.plot(epochs, val_values, marker="s", color=c, linestyle="--", label=f"{label} val")
            ax.set_title(title, fontsize=12, weight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)

    axes[0].set_ylabel("Loss component value")
    axes[2].legend(fontsize=6, loc="upper right")
    fig.suptitle("Classification loss decomposition", fontsize=15, weight="bold")
    fig.tight_layout()
    out = out_dir / "classification_loss_components.png"
    savefig(out)
    outputs.append(str(out))

    table_lines = [
        "run,epoch,train_ce_img,val_ce_img,train_ce_expr,val_ce_expr,train_align,val_align,train_total,val_total",
    ]
    for label, rel in RUNS.items():
        hist_path = results_dir / rel / "history.json"
        if not hist_path.exists():
            continue
        hist = load_json(hist_path)
        for x in hist:
            table_lines.append(
                ",".join(
                    [
                        label,
                        str(x["epoch"]),
                        f"{x['train']['ce_img']:.6f}",
                        f"{x['val']['ce_img']:.6f}",
                        f"{x['train']['ce_expr']:.6f}",
                        f"{x['val']['ce_expr']:.6f}",
                        f"{x['train']['align']:.6f}",
                        f"{x['val']['align']:.6f}",
                        f"{x['train']['loss']:.6f}",
                        f"{x['val']['loss']:.6f}",
                    ]
                )
            )
    csv_path = out_dir / "classification_loss_components.csv"
    csv_path.write_text("\n".join(table_lines) + "\n")
    outputs.append(str(csv_path))
    return outputs


def plot_story_loss_panels(results_dir: Path, out_dir: Path) -> list[str]:
    """Simpler presentation figures with fewer visual elements per slide."""
    outputs: list[str] = []
    selected = {
        "8-class crop128": RUNS["GigaPath crop128 8-class"],
        "3-class crop128": RUNS["GigaPath crop128 3-class"],
    }
    colors = {"8-class crop128": "#b45309", "3-class crop128": "#1d4ed8"}

    # Slide-friendly comparison of validation metrics.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, rel in selected.items():
        hist = load_json(results_dir / rel / "history.json")
        epochs = [x["epoch"] for x in hist]
        expr_acc = [x["val"]["acc_expr"] for x in hist]
        macro_f1 = [x["val"]["metrics"]["expr"]["macro_f1"] for x in hist]
        axes[0].plot(epochs, expr_acc, marker="o", color=colors[label], label=label)
        axes[1].plot(epochs, macro_f1, marker="o", color=colors[label], label=label)
    for ax, title, ylabel in [
        (axes[0], "Expression branch accuracy", "Val accuracy"),
        (axes[1], "Expression branch macro-F1", "Val macro-F1"),
    ]:
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.25, 0.82)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Coarse 3-class target is much easier than fine 8-class target", fontsize=15, weight="bold")
    fig.tight_layout()
    out = out_dir / "presentation_3class_vs_8class_metrics.png"
    savefig(out)
    outputs.append(str(out))

    # Stacked total loss decomposition for one clear main run.
    run_label = "GigaPath crop128 3-class"
    hist = load_json(results_dir / RUNS[run_label] / "history.json")
    epochs = np.array([x["epoch"] for x in hist])
    ce_img = np.array([x["val"]["ce_img"] for x in hist])
    ce_expr = np.array([x["val"]["ce_expr"] for x in hist])
    align = np.array([0.5 * x["val"]["align"] for x in hist])
    total = np.array([x["val"]["loss"] for x in hist])

    plt.figure(figsize=(8, 5))
    plt.stackplot(
        epochs,
        ce_img,
        ce_expr,
        align,
        labels=["Image CE", "Expression CE", "0.5 x alignment MSE"],
        colors=["#93c5fd", "#fbbf24", "#a7f3d0"],
        alpha=0.9,
    )
    plt.plot(epochs, total, marker="o", color="#111827", linewidth=2.2, label="Total val loss")
    plt.title("Total loss decomposed into image, expression, and alignment terms", fontsize=14, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    out = out_dir / "presentation_3class_total_loss_decomposition.png"
    savefig(out)
    outputs.append(str(out))

    # One diagnostic panel for why 8-class is unstable.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, rel in selected.items():
        hist = load_json(results_dir / rel / "history.json")
        epochs = [x["epoch"] for x in hist]
        train_expr_ce = [x["train"]["ce_expr"] for x in hist]
        val_expr_ce = [x["val"]["ce_expr"] for x in hist]
        axes[0].plot(epochs, train_expr_ce, marker="o", color=colors[label], linestyle="-", label=f"{label} train")
        axes[1].plot(epochs, val_expr_ce, marker="s", color=colors[label], linestyle="--", label=f"{label} val")
    axes[0].set_title("Expression CE on train", fontsize=13, weight="bold")
    axes[1].set_title("Expression CE on validation", fontsize=13, weight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Expression CE loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("8-class issue: training gets easier, validation expression CE stays high", fontsize=15, weight="bold")
    fig.tight_layout()
    out = out_dir / "presentation_expression_ce_generalization_gap.png"
    savefig(out)
    outputs.append(str(out))
    return outputs


def plot_resnet_contrastive(results_dir: Path, out_dir: Path) -> list[str]:
    hist_path = results_dir / "training_runs/contrastive_resnet50_crop64_rtx6000_project_v2_11016975/history.json"
    if not hist_path.exists():
        return []
    hist = load_json(hist_path)
    epochs = [x["epoch"] for x in hist]
    train_loss = [x["train_loss"] for x in hist]
    val_loss = [x["val_loss"] for x in hist]
    best_idx = int(np.argmin(val_loss))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train InfoNCE loss", color="#7c2d12")
    plt.plot(epochs, val_loss, marker="s", label="val InfoNCE loss", color="#1e3a8a")
    plt.scatter([epochs[best_idx]], [val_loss[best_idx]], s=90, color="#dc2626", zorder=5, label=f"best val epoch {epochs[best_idx]}")
    plt.title("ResNet50 contrastive baseline overfits after early epochs", fontsize=14, weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE loss")
    plt.grid(alpha=0.25)
    plt.legend()
    out = out_dir / "resnet_contrastive_loss.png"
    savefig(out)
    return [str(out)]


def plot_retrieval(results_dir: Path, out_dir: Path) -> list[str]:
    metrics = {}
    for label, rel in RETRIEVAL_RUNS.items():
        path = results_dir / rel
        if path.exists():
            metrics[label] = load_json(path)
    if not metrics:
        return []

    labels = list(metrics)
    recalls = ["recall@1", "recall@5", "recall@10"]
    x = np.arange(len(labels))
    width = 0.24

    plt.figure(figsize=(9, 5))
    for i, r in enumerate(recalls):
        values = [metrics[label]["retrieval_overall"][r] for label in labels]
        plt.bar(x + (i - 1) * width, values, width=width, label=r)
    plt.xticks(x, labels, rotation=12, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Expression -> image retrieval recall")
    plt.title("Cross-modal retrieval: GigaPath 3-class vs ResNet contrastive", fontsize=14, weight="bold")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    out1 = out_dir / "retrieval_overall_recall.png"
    savefig(out1)

    # Per-class plot for the strongest completed run.
    gp = metrics.get("GigaPath crop128 3-class")
    outputs = [str(out1)]
    if gp:
        per_label = gp["retrieval_per_label"]
        cls = list(per_label)
        vals = [per_label[c]["recall@1"] for c in cls]
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(cls, vals, color=["#2563eb", "#0f766e", "#b45309"])
        plt.ylim(0, 1.0)
        plt.ylabel("Recall@1")
        plt.title("GigaPath 3-class retrieval recall@1 by tissue label", fontsize=14, weight="bold")
        plt.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)
        out2 = out_dir / "gigapath_3class_per_label_recall1.png"
        savefig(out2)
        outputs.append(str(out2))

    return outputs


def write_summary(results_dir: Path, out_dir: Path, outputs: list[str]) -> str:
    lines = [
        "# Visium HD Exp1 Training Summary Figures",
        "",
        "Generated from saved project results. Key source files are `history.json`, `summary.json`, and `metrics.json` under the project results directory.",
        "",
        "## Figures",
    ]
    for path in outputs:
        lines.append(f"- `{path}`")

    lines.extend(["", "## Key Completed Metrics"])
    for label, rel in RUNS.items():
        summary_path = results_dir / rel / "summary.json"
        if not summary_path.exists():
            continue
        s = load_json(summary_path)
        lines.append(
            f"- {label}: best epoch {s.get('best_epoch')}, "
            f"val expr acc {s.get('best_val_acc_expr'):.4f}, "
            f"val image acc {s.get('best_val_acc_img'):.4f}, "
            f"expr macro-F1 {s.get('best_val_macro_f1_expr'):.4f}"
        )

    for label, rel in RETRIEVAL_RUNS.items():
        path = results_dir / rel
        if not path.exists():
            continue
        m = load_json(path)
        r = m["retrieval_overall"]
        lp = m["linear_probe"]
        lines.append(
            f"- {label} retrieval: recall@1 {r['recall@1']:.4f}, "
            f"recall@5 {r['recall@5']:.4f}, recall@10 {r['recall@10']:.4f}, "
            f"concat linear probe {lp['concatenated']:.4f}"
        )

    out = out_dir / "figure_summary.md"
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_dir / "figures" / "group_meeting"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    outputs.extend(plot_classification_curves(args.results_dir, out_dir))
    outputs.extend(plot_loss_components(args.results_dir, out_dir))
    outputs.extend(plot_story_loss_panels(args.results_dir, out_dir))
    outputs.extend(plot_resnet_contrastive(args.results_dir, out_dir))
    outputs.extend(plot_retrieval(args.results_dir, out_dir))
    outputs.append(write_summary(args.results_dir, out_dir, outputs))

    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
