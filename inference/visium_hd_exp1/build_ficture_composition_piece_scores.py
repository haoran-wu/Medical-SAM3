#!/usr/bin/env python3
"""Build FICTURE composition priors as six-class piece scores.

This creates VLM-shaped score outputs from candidate mask composition in the
official FICTURE factor-label map. The outputs can be passed to
assemble_piece_first_vlm_results.py for the same downstream assembly/evaluation
used by VLM runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


CLASS_KEYS = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune_infiltration",
]

CLASS_TO_SEMANTIC = {
    "bronchiola": "lung_bronchiola",
    "alveoli": "lung_alveoli_normal_adjacent",
    "vessels": "lung_vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune_infiltration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_key(row: dict[str, str]) -> str:
    return "__".join(
        [
            row.get("target_label") or row.get("label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            str(row.get("candidate_id", "")),
        ]
    )


def load_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    expected_size = (expected_hw[1], expected_hw[0])
    if image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def load_semantic_scores(path: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    data = json.loads(path.read_text())
    factors = sorted(data["factors"], key=lambda item: int(item["factor"]))
    n_factors = max(int(item["factor"]) for item in factors) + 1
    vectors: dict[str, np.ndarray] = {
        class_key: np.zeros(n_factors, dtype=np.float64) for class_key in CLASS_KEYS
    }
    for item in factors:
        factor = int(item["factor"])
        target_scores = item.get("target_scores") or {}
        for class_key, semantic_key in CLASS_TO_SEMANTIC.items():
            vectors[class_key][factor] = float(target_scores.get(semantic_key, 0.0))
    return factors, vectors


def load_prompt_legend(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = read_csv(path)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        out[int(row["Factor"])] = {
            "rgb": row.get("RGB", ""),
            "major_compartment": row.get("Major Compartment", ""),
            "cell_type": row.get("cell type", ""),
        }
    return out


def factor_composition_summary(hist: np.ndarray, factors: list[dict], prompt_legend: dict[int, dict[str, str]], top_n: int = 5) -> str:
    parts: list[str] = []
    for factor in np.argsort(hist)[::-1][:top_n]:
        frac = float(hist[factor])
        if frac <= 0:
            continue
        factor_info = next((item for item in factors if int(item["factor"]) == int(factor)), {})
        prompt_info = prompt_legend.get(int(factor), {})
        rgb = prompt_info.get("rgb") or factor_info.get("rgb_text", "")
        major = prompt_info.get("major_compartment", "")
        cell_type = prompt_info.get("cell_type") or factor_info.get("llm_inferred_cell_type", "")
        parts.append(f"F{int(factor)} {frac:.3f} RGB {rgb}; {major}; {cell_type}")
    return " | ".join(parts)


def percentile_scores(abs_rows: list[dict[str, object]]) -> list[dict[str, int]]:
    values_by_class: dict[str, list[float]] = {
        key: [float(row[f"{key}_abs_score"]) for row in abs_rows] for key in CLASS_KEYS
    }
    out: list[dict[str, int]] = []
    for row in abs_rows:
        scores: dict[str, int] = {}
        for key in CLASS_KEYS:
            values = values_by_class[key]
            value = float(row[f"{key}_abs_score"])
            # Mid-rank percentile, scaled to 0-100. This preserves class-wise
            # ranking even when absolute factor affinity has a narrow range.
            below = sum(1 for x in values if x < value)
            equal = sum(1 for x in values if x == value)
            pct = (below + 0.5 * equal) / max(1, len(values))
            scores[key] = int(round(100 * pct))
        out.append(scores)
    return out


def predicted_class(scores: dict[str, int]) -> tuple[str, int, bool]:
    max_score = max(scores.values())
    winners = [key for key, value in scores.items() if value == max_score]
    return winners[0], max_score, len(winners) > 1


def write_score_output(
    output_dir: Path,
    pool_rows: list[dict[str, str]],
    score_rows: list[dict[str, int]],
    truth_by_key: dict[str, dict[str, str]],
    raw_feature_rows: list[dict[str, object]],
    method_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    for pool_row, scores, feature_row in zip(pool_rows, score_rows, raw_feature_rows):
        key = row_key(pool_row)
        truth = truth_by_key.get(key, {})
        pred, top_score, tie = predicted_class(scores)
        true_label = truth.get("classification_true_label") or pool_row.get("target_label") or pool_row.get("label", "")
        class_key_true = {
            "lung_bronchiola": "bronchiola",
            "lung_alveoli_normal_adjacent": "alveoli",
            "lung_vessels": "vessels",
            "tumor": "tumor",
            "stroma": "stroma",
            "immune_infiltration": "immune_infiltration",
        }.get(true_label, true_label)
        base = {
            "row_key": key,
            "parse_status": "ok",
            "label": pool_row.get("label", ""),
            "display": pool_row.get("display", ""),
            "target_label": pool_row.get("target_label", ""),
            "source": pool_row.get("source", ""),
            "run": pool_row.get("run", ""),
            "setting": pool_row.get("setting", ""),
            "candidate_id": pool_row.get("candidate_id", ""),
            "candidate_uid": pool_row.get("candidate_uid", ""),
            "true_label": true_label,
            "predicted_label": pred,
            "top_score": top_score,
            "top_score_tie": str(tie).lower(),
            "is_correct": str((pred == class_key_true) and not tie).lower(),
            "method": method_name,
            "factor_composition_top": feature_row.get("factor_composition_top", ""),
        }
        prediction_rows.append({**base, **scores})
        for class_key, value in scores.items():
            cross_rows.append({**base, "class": class_key, "score": value})
    write_csv(output_dir / "per_candidate_predictions.csv", prediction_rows)
    write_csv(output_dir / "cross_label_scores.csv", cross_rows)
    write_csv(output_dir / "failed_rows.csv", [], ["row_key", "error"])
    correct = [row for row in prediction_rows if row["is_correct"] == "true"]
    write_csv(
        output_dir / "overall_accuracy.csv",
        [{"method": method_name, "correct": len(correct), "total": len(prediction_rows), "accuracy": len(correct) / max(1, len(prediction_rows))}],
    )
    per_class: list[dict[str, object]] = []
    for class_key in CLASS_KEYS:
        subset = [row for row in prediction_rows if row["true_label"] in {class_key, CLASS_TO_SEMANTIC[class_key]}]
        n_correct = sum(1 for row in subset if row["is_correct"] == "true")
        per_class.append({"class": class_key, "correct": n_correct, "total": len(subset), "accuracy": n_correct / max(1, len(subset))})
    write_csv(output_dir / "per_class_accuracy.csv", per_class)


def load_existing_scores(vlm_dir: Path, pool_rows: list[dict[str, str]]) -> list[dict[str, int]]:
    pred_path = vlm_dir / "per_candidate_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    existing = {row.get("row_key") or row_key(row): row for row in read_csv(pred_path) if row.get("parse_status", "ok") == "ok"}
    scores: list[dict[str, int]] = []
    for row in pool_rows:
        key = row_key(row)
        src = existing[key]
        scores.append({class_key: int(float(src.get(class_key, 0) or 0)) for class_key in CLASS_KEYS})
    return scores


def fuse_scores(vlm: list[dict[str, int]], prior: list[dict[str, int]], prior_weight: float) -> list[dict[str, int]]:
    fused: list[dict[str, int]] = []
    for vlm_scores, prior_scores in zip(vlm, prior):
        fused.append(
            {
                key: int(round((1.0 - prior_weight) * vlm_scores[key] + prior_weight * prior_scores[key]))
                for key in CLASS_KEYS
            }
        )
    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--factor-label-image", type=Path, required=True)
    parser.add_argument("--semantic-legend", type=Path, required=True)
    parser.add_argument("--prompt-legend-csv", type=Path)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--fuse-vlm-output", action="append", default=[], help="NAME=/path/to/vlm_output_dir")
    parser.add_argument("--fusion-prior-weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_rows = read_csv(args.pool_dir / "public_vlm_requests.csv")
    truth_rows = read_csv(args.pool_dir / "hidden_candidate_truth.csv")
    truth_by_key = {row_key(row): row for row in truth_rows}
    factor_labels = np.load(args.factor_label_image)
    factors, score_vectors = load_semantic_scores(args.semantic_legend)
    prompt_legend = load_prompt_legend(args.prompt_legend_csv)

    raw_feature_rows: list[dict[str, object]] = []
    direct_scores: list[dict[str, int]] = []
    for row in public_rows:
        mask = load_mask(Path(row["mask_path"]), factor_labels.shape)
        values = factor_labels[mask].astype(int)
        values = values[(values >= 0) & (values < len(factors))]
        if values.size:
            counts = np.bincount(values, minlength=len(factors)).astype(np.float64)
            hist = counts / counts.sum()
        else:
            hist = np.zeros(len(factors), dtype=np.float64)
        abs_scores_float = {
            class_key: float(np.dot(hist, score_vectors[class_key])) for class_key in CLASS_KEYS
        }
        direct_scores.append({class_key: int(round(100 * abs_scores_float[class_key])) for class_key in CLASS_KEYS})
        raw_feature_rows.append(
            {
                "row_key": row_key(row),
                "candidate_uid": row.get("candidate_uid", ""),
                "mask_path": row.get("mask_path", ""),
                "factor_composition_top": factor_composition_summary(hist, factors, prompt_legend),
                **{f"{class_key}_abs_score": abs_scores_float[class_key] for class_key in CLASS_KEYS},
            }
        )

    rank_scores = percentile_scores(raw_feature_rows)

    args.output_base.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_base / "candidate_ficture_composition_features.csv", raw_feature_rows)
    write_score_output(
        args.output_base / "FICTURE_composition_prior_absolute",
        public_rows,
        direct_scores,
        truth_by_key,
        raw_feature_rows,
        "FICTURE_composition_prior_absolute",
    )
    write_score_output(
        args.output_base / "FICTURE_composition_prior_rankcalibrated",
        public_rows,
        rank_scores,
        truth_by_key,
        raw_feature_rows,
        "FICTURE_composition_prior_rankcalibrated",
    )

    manifest: dict[str, object] = {
        "pool_dir": str(args.pool_dir),
        "factor_label_image": str(args.factor_label_image),
        "semantic_legend": str(args.semantic_legend),
        "prompt_legend_csv": str(args.prompt_legend_csv) if args.prompt_legend_csv else "",
        "n_candidates": len(public_rows),
        "outputs": [
            "FICTURE_composition_prior_absolute",
            "FICTURE_composition_prior_rankcalibrated",
        ],
    }

    for item in args.fuse_vlm_output:
        if "=" not in item:
            raise SystemExit(f"Expected NAME=/path/to/vlm_output_dir, got {item}")
        name, path_text = item.split("=", 1)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        vlm_scores = load_existing_scores(Path(path_text), public_rows)
        fused = fuse_scores(vlm_scores, rank_scores, args.fusion_prior_weight)
        out_name = f"{safe_name}_plus_FICTURE_rankprior_w{int(round(args.fusion_prior_weight * 100))}"
        write_score_output(args.output_base / out_name, public_rows, fused, truth_by_key, raw_feature_rows, out_name)
        manifest["outputs"].append(out_name)  # type: ignore[index]

    (args.output_base / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.output_base)


if __name__ == "__main__":
    main()
