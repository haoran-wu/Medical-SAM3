#!/usr/bin/env python3
"""June6 CLIP/FICTURE semantic-latent piece ranker.

This script scores the Jun5/Jun6 piece-first candidate pool with several
CLIP-shaped branches:

1. H&E crop image -> CLIP image embedding -> tissue-class text similarity.
2. FICTURE crop image -> CLIP image embedding -> tissue-class text similarity.
3. FICTURE RGB/cell-type composition -> weighted text embedding -> class text
   similarity.

The outputs are VLM-shaped six-class score folders, so they can be evaluated by
assemble_piece_first_vlm_results.py without changing the downstream assembly
logic.
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

SEMANTIC_TO_CLASS = {v: k for k, v in CLASS_TO_SEMANTIC.items()}

CLASS_PROMPTS = {
    "bronchiola": [
        "bronchiolar airway tissue with airway lumen and bronchiolar epithelial lining",
        "airway epithelial structure, bronchiola, lumen, club goblet and ciliated cells",
    ],
    "alveoli": [
        "alveolar lung parenchyma with open air spaces and thin septa",
        "normal alveoli, alveolar air spaces, delicate septal wall, AT2 cells",
    ],
    "vessels": [
        "blood vessel or vascular wall with endothelial cells and lumen",
        "vascular structure, endothelial lining, smooth muscle vessel wall, lumen",
    ],
    "tumor": [
        "lung adenocarcinoma tumor region with malignant epithelial tumor cells",
        "dense epithelial tumor nests and malignant tumor compartment",
    ],
    "stroma": [
        "stromal mesenchymal tissue with fibroblasts, collagen, and smooth muscle",
        "fibroblast-rich stroma, mesenchymal compartment, collagen-like tissue",
    ],
    "immune_infiltration": [
        "immune cell rich region with lymphocytes, macrophages, and plasma cells",
        "small round cell aggregate, T cells, B cells, macrophages, plasma cells",
    ],
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
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    if row.get("row_key"):
        return row["row_key"]
    return "__".join(
        [
            row.get("target_label") or row.get("label", ""),
            row.get("source", ""),
            row.get("run", ""),
            row.get("setting", ""),
            str(row.get("candidate_id", "")),
        ]
    )


def truth_class(row: dict[str, str], pool_row: dict[str, str]) -> str:
    value = row.get("classification_true_label") or row.get("true_label") or pool_row.get("target_label") or pool_row.get("label", "")
    return SEMANTIC_TO_CLASS.get(value, value)


def load_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    expected_size = (expected_hw[1], expected_hw[0])
    if image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.array(image) > 0


def pooled_features(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    return output


def l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return array / denom


def encode_texts(processor, model, device: str, texts: list[str], batch_size: int = 128) -> np.ndarray:
    import torch

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            features = pooled_features(model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


def encode_images(processor, model, device: str, paths: list[Path], batch_size: int) -> np.ndarray:
    import torch

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = pooled_features(model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.detach().cpu().float().numpy())
            print(f"Encoded images {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(chunks, axis=0)


def class_text_features(processor, model, device: str) -> np.ndarray:
    all_prompts: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for key in CLASS_KEYS:
        start = len(all_prompts)
        all_prompts.extend(CLASS_PROMPTS[key])
        slices[key] = (start, len(all_prompts))
    prompt_features = encode_texts(processor, model, device, all_prompts)
    rows: list[np.ndarray] = []
    for key in CLASS_KEYS:
        start, end = slices[key]
        feat = prompt_features[start:end].mean(axis=0)
        rows.append(l2_normalize(feat[None, :])[0])
    return np.stack(rows, axis=0)


def load_prompt_legend(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "factor": row.get("Factor", ""),
                "rgb": row.get("RGB", ""),
                "major_compartment": row.get("Major Compartment", ""),
                "cell_type": row.get("cell type", ""),
                "text": f"{row.get('cell type', '')}; {row.get('Major Compartment', '')}",
            }
        )
    return out


def factor_composition_features(
    public_rows: list[dict[str, str]],
    factor_labels: np.ndarray,
    legend_rows: list[dict[str, str]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    n_factors = len(legend_rows)
    hists: list[np.ndarray] = []
    feature_rows: list[dict[str, object]] = []
    for row in public_rows:
        mask = load_mask(Path(row["mask_path"]), factor_labels.shape)
        values = factor_labels[mask].astype(int)
        values = values[(values >= 0) & (values < n_factors)]
        if values.size:
            counts = np.bincount(values, minlength=n_factors).astype(np.float64)
            hist = counts / counts.sum()
        else:
            hist = np.zeros(n_factors, dtype=np.float64)
        hists.append(hist)
        top_parts: list[str] = []
        for idx in np.argsort(hist)[::-1][:5]:
            frac = float(hist[idx])
            if frac <= 0:
                continue
            info = legend_rows[int(idx)]
            top_parts.append(
                f"F{idx} {frac:.3f} RGB {info['rgb']}; {info['major_compartment']}; {info['cell_type']}"
            )
        feature_rows.append(
            {
                "row_key": row_key(row),
                "candidate_uid": row.get("candidate_uid", ""),
                "mask_path": row.get("mask_path", ""),
                "factor_composition_top": " | ".join(top_parts),
                **{f"factor_{idx}_fraction": float(hist[idx]) for idx in range(n_factors)},
            }
        )
    return np.stack(hists, axis=0), feature_rows


def logits_to_class_rank_scores(logits: np.ndarray) -> list[dict[str, int]]:
    """Convert raw similarities to class-wise percentile scores.

    Class-wise rank scores are better for downstream per-class assembly because
    each class gets a usable 0-100 ranking scale, even when absolute CLIP
    similarities are narrow.
    """

    scores: list[dict[str, int]] = []
    for i in range(logits.shape[0]):
        row: dict[str, int] = {}
        for j, key in enumerate(CLASS_KEYS):
            values = logits[:, j]
            value = logits[i, j]
            below = int((values < value).sum())
            equal = int((values == value).sum())
            pct = (below + 0.5 * equal) / max(1, len(values))
            row[key] = int(round(100 * pct))
        scores.append(row)
    return scores


def logits_to_softmax_scores(logits: np.ndarray) -> list[dict[str, int]]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return [
        {key: int(round(100 * probs[i, j])) for j, key in enumerate(CLASS_KEYS)}
        for i in range(logits.shape[0])
    ]


def predicted_class(scores: dict[str, int]) -> tuple[str, int, bool]:
    max_score = max(scores.values())
    winners = [key for key, value in scores.items() if value == max_score]
    return winners[0], max_score, len(winners) > 1


def write_score_output(
    output_dir: Path,
    public_rows: list[dict[str, str]],
    truth_by_key: dict[str, dict[str, str]],
    score_rows: list[dict[str, int]],
    method_name: str,
    feature_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    for pool_row, scores, feature_row in zip(public_rows, score_rows, feature_rows):
        key = row_key(pool_row)
        truth = truth_by_key.get(key, {})
        true_label = truth_class(truth, pool_row)
        pred, top_score, tie = predicted_class(scores)
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
            "is_correct": str((pred == true_label) and not tie).lower(),
            "method": method_name,
            "factor_composition_top": feature_row.get("factor_composition_top", ""),
        }
        prediction_rows.append({**base, **scores})
        for class_key, score in scores.items():
            cross_rows.append({**base, "class": class_key, "score": score})
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
        subset = [row for row in prediction_rows if row["true_label"] == class_key]
        n_correct = sum(1 for row in subset if row["is_correct"] == "true")
        per_class.append({"class": class_key, "correct": n_correct, "total": len(subset), "accuracy": n_correct / max(1, len(subset))})
    write_csv(output_dir / "per_class_accuracy.csv", per_class)


def combine_scores(*items: tuple[float, list[dict[str, int]]]) -> list[dict[str, int]]:
    n = len(items[0][1])
    out: list[dict[str, int]] = []
    total_weight = sum(weight for weight, _scores in items)
    for i in range(n):
        row: dict[str, int] = {}
        for key in CLASS_KEYS:
            value = sum(weight * scores[i][key] for weight, scores in items) / total_weight
            row[key] = int(round(value))
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--factor-label-image", type=Path, required=True)
    parser.add_argument("--prompt-legend-csv", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_rows = read_csv(args.pool_dir / "public_vlm_requests.csv")
    truth_rows = read_csv(args.pool_dir / "hidden_candidate_truth.csv")
    truth_by_key = {row_key(row): row for row in truth_rows}
    factor_labels = np.load(args.factor_label_image)
    legend_rows = load_prompt_legend(args.prompt_legend_csv)
    if len(legend_rows) <= int(np.max(factor_labels)):
        raise SystemExit(
            f"Prompt legend has {len(legend_rows)} rows but factor label image uses factor {int(np.max(factor_labels))}"
        )

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    class_features = class_text_features(processor, model, device)
    he_paths = [args.pool_dir / row["he_crop_rel"] for row in public_rows]
    ficture_paths = [args.pool_dir / row["ficture_crop_rel"] for row in public_rows]
    missing = [str(path) for path in he_paths + ficture_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing crop images, first missing: {missing[:3]}")

    print("Encoding H&E crops", flush=True)
    he_features = encode_images(processor, model, device, he_paths, args.batch_size)
    print("Encoding FICTURE crops", flush=True)
    ficture_image_features = encode_images(processor, model, device, ficture_paths, args.batch_size)

    legend_texts = [row["text"] for row in legend_rows]
    factor_text_features = encode_texts(processor, model, device, legend_texts)
    factor_hists, feature_rows = factor_composition_features(public_rows, factor_labels, legend_rows)
    ficture_semantic_features = l2_normalize(factor_hists @ factor_text_features)

    he_logits = he_features @ class_features.T
    ficture_image_logits = ficture_image_features @ class_features.T
    ficture_semantic_logits = ficture_semantic_features @ class_features.T

    methods: dict[str, list[dict[str, int]]] = {
        "HE_CLIP_rank": logits_to_class_rank_scores(he_logits),
        "FICTURE_image_CLIP_rank": logits_to_class_rank_scores(ficture_image_logits),
        "FICTURE_semantic_latent_rank": logits_to_class_rank_scores(ficture_semantic_logits),
        "HE_CLIP_softmax": logits_to_softmax_scores(he_logits),
        "FICTURE_image_CLIP_softmax": logits_to_softmax_scores(ficture_image_logits),
        "FICTURE_semantic_latent_softmax": logits_to_softmax_scores(ficture_semantic_logits),
    }
    methods["HE_plus_FICTURE_semantic_rank_w55_45"] = combine_scores(
        (0.55, methods["HE_CLIP_rank"]),
        (0.45, methods["FICTURE_semantic_latent_rank"]),
    )
    methods["HE_plus_FICTURE_image_plus_semantic_rank_w45_20_35"] = combine_scores(
        (0.45, methods["HE_CLIP_rank"]),
        (0.20, methods["FICTURE_image_CLIP_rank"]),
        (0.35, methods["FICTURE_semantic_latent_rank"]),
    )

    args.output_base.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_base / "candidate_ficture_composition_features.csv", feature_rows)
    logit_rows: list[dict[str, object]] = []
    for i, row in enumerate(public_rows):
        base = {
            "row_key": row_key(row),
            "candidate_uid": row.get("candidate_uid", ""),
            "true_label": truth_class(truth_by_key.get(row_key(row), {}), row),
        }
        for prefix, logits in [
            ("he_clip", he_logits),
            ("ficture_image_clip", ficture_image_logits),
            ("ficture_semantic", ficture_semantic_logits),
        ]:
            for j, class_key in enumerate(CLASS_KEYS):
                base[f"{prefix}_{class_key}_logit"] = float(logits[i, j])
        logit_rows.append(base)
    write_csv(args.output_base / "raw_clip_similarity_logits.csv", logit_rows)

    manifest = {
        "line": "June6 Clip",
        "pool_dir": str(args.pool_dir),
        "factor_label_image": str(args.factor_label_image),
        "prompt_legend_csv": str(args.prompt_legend_csv),
        "clip_model": args.clip_model,
        "device": device,
        "n_candidates": len(public_rows),
        "methods": list(methods),
        "class_prompts": CLASS_PROMPTS,
        "note": "Candidate-level scoring only. Final masks are assembled downstream from selected pieces.",
    }
    for method_name, scores in methods.items():
        write_score_output(args.output_base / method_name, public_rows, truth_by_key, scores, method_name, feature_rows)
    (args.output_base / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(args.output_base)


if __name__ == "__main__":
    main()
