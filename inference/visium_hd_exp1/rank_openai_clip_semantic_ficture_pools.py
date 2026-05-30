#!/usr/bin/env python3
"""OpenAI CLIP semantic retrieval on same-ROI H&E/FICTURE candidate masks.

This is a direct retrieval test, not a fused ranker:

candidate mask -> crop the official FICTURE factor-color image with that mask ->
OpenAI CLIP image/text similarity against feature-aware label prompts ->
top-k selected masks -> Dice/Precision/Recall.

The candidate masks can come from H&E SAM or FICTURE SAM, but every candidate is
scored in the same official FICTURE ROI coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from ficture_factor_semantics import load_semantic_legend, semantic_clip_prompts
from rank_multimodal_sam_candidates import (
    CLIP_TEXT_PROMPTS,
    LABEL_ORDER,
    Candidate,
    compute_clip_scores,
    load_candidates,
    metrics,
    read_mask,
    resize_bool,
    select_union,
    slugify,
)


SIX_LABELS = {
    "lung_bronchiola",
    "lung_alveoli_normal_adjacent",
    "lung_vessels",
    "tumor",
    "stroma",
    "immune_infiltration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI CLIP semantic retrieval on the official FICTURE ROI image."
    )
    parser.add_argument("--candidate-root", action="append", required=True, help="source=/path/to/root")
    parser.add_argument("--ficture-image", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--factor-semantic-legend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--mask-cache-size", type=int, default=64)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20])
    parser.add_argument("--max-overlap", type=float, default=0.86)
    parser.add_argument(
        "--label-slugs",
        nargs="+",
        default=[slug for slug, _ in LABEL_ORDER if slug in SIX_LABELS],
        help="Labels to report. Defaults to the six user-facing labels.",
    )
    parser.add_argument(
        "--official-summary",
        type=Path,
        default=None,
        help="Optional PASS_OFFICIAL summary JSON to verify before reporting official FICTURE results.",
    )
    return parser.parse_args()


def verify_official(summary_path: Path | None) -> None:
    if summary_path is None:
        return
    payload = json.loads(summary_path.read_text())
    status = payload.get("status")
    if status != "PASS_OFFICIAL":
        raise SystemExit(f"Official FICTURE summary is not PASS_OFFICIAL: {summary_path} status={status}")
    failed = [key for key, value in payload.get("status_checks", {}).items() if not value]
    if failed:
        raise SystemExit(f"Official FICTURE status checks failed: {failed}")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_counts(candidates: Sequence[Candidate]) -> List[dict]:
    by_source = Counter(c.source for c in candidates)
    by_source_setting = Counter((c.source, c.run, c.setting) for c in candidates)
    rows: List[dict] = [
        {
            "scope": "all_sources",
            "source": "all",
            "run": "",
            "setting": "",
            "n_candidates": len(candidates),
        }
    ]
    for source, count in sorted(by_source.items()):
        rows.append(
            {
                "scope": "source_total",
                "source": source,
                "run": "",
                "setting": "",
                "n_candidates": count,
            }
        )
    for (source, run, setting), count in sorted(by_source_setting.items()):
        rows.append(
            {
                "scope": "source_setting",
                "source": source,
                "run": run,
                "setting": setting,
                "n_candidates": count,
            }
        )
    return rows


def load_gt_masks(summary_path: Path, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    summary = json.loads(summary_path.read_text())
    masks: Dict[str, np.ndarray] = {}
    fallback_dir = summary_path.parent / "cropped_annotation_masks"
    for item in summary["labels"]:
        slug = item.get("slug") or slugify(item["label"])
        raw = Path(item["mask_path"])
        candidates = [
            raw,
            summary_path.parent / item["mask_path"],
            fallback_dir / raw.name,
        ]
        for path in candidates:
            if path.exists():
                masks[slug] = resize_bool(read_mask(path), shape_hw)
                break
        else:
            raise FileNotFoundError(f"Could not resolve mask for {slug}: tried {candidates}")
    return masks


def parse_candidate_roots(items: Sequence[str]) -> List[str]:
    # load_candidates already understands source=/path. This helper only checks
    # formatting early so errors are clear in Slurm logs.
    roots: List[str] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected source=/path candidate-root mapping, got: {item}")
        roots.append(item)
    return roots


def prompt_rows(prompts: Dict[str, List[str]], label_slugs: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for slug, display in LABEL_ORDER:
        if slug not in set(label_slugs):
            continue
        for idx, prompt in enumerate(prompts.get(slug, []), start=1):
            rows.append(
                {
                    "label": slug,
                    "display": display,
                    "prompt_index": idx,
                    "prompt": prompt,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    verify_official(args.official_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ficture_image = np.array(Image.open(args.ficture_image).convert("RGB"))
    target_shape = ficture_image.shape[:2]
    if target_shape != (3327, 3144):
        raise SystemExit(f"Unexpected official FICTURE ROI shape hw={target_shape}; expected (3327, 3144)")

    gt_masks = load_gt_masks(args.summary_path, target_shape)
    candidate_roots = parse_candidate_roots(args.candidate_root)
    candidates = load_candidates(candidate_roots)
    if not candidates:
        raise SystemExit("No candidates found.")

    semantic_factors = load_semantic_legend(args.factor_semantic_legend, n_factors=12)
    semantic_prompts = semantic_clip_prompts(CLIP_TEXT_PROMPTS, semantic_factors)

    mask_cache: Dict[Path, np.ndarray] = {}
    mask_cache_order: List[Path] = []

    def get_mask(path: Path) -> np.ndarray:
        if args.mask_cache_size <= 0:
            return resize_bool(read_mask(path), target_shape)
        if path in mask_cache:
            return mask_cache[path]
        mask = resize_bool(read_mask(path), target_shape)
        if len(mask_cache_order) >= args.mask_cache_size:
            old = mask_cache_order.pop(0)
            mask_cache.pop(old, None)
        mask_cache[path] = mask
        mask_cache_order.append(path)
        return mask

    print(
        f"Loaded {len(candidates)} total saved candidates for OpenAI CLIP semantic FICTURE retrieval.",
        flush=True,
    )
    write_csv(args.output_dir / "candidate_pool_counts.csv", source_counts(candidates))
    write_csv(args.output_dir / "openai_clip_semantic_text_prompts.csv", prompt_rows(semantic_prompts, args.label_slugs))

    clip_scores = compute_clip_scores(
        candidates=candidates,
        get_mask=get_mask,
        image=ficture_image,
        model_name=args.clip_model,
        batch_size=args.clip_batch_size,
        device=args.clip_device,
        text_prompts_by_label=semantic_prompts,
    )

    label_order = [(slug, display) for slug, display in LABEL_ORDER if slug in set(args.label_slugs)]
    score_rows: List[dict] = []
    for cand in candidates:
        scores = clip_scores.get(cand.mask_path, {})
        for slug, display in label_order:
            score_rows.append(
                {
                    "label": slug,
                    "display": display,
                    "ranker": "openai_clip_ficture_semantic",
                    "score": scores.get(slug, 0.0),
                    "source": cand.source,
                    "run": cand.run,
                    "setting": cand.setting,
                    "candidate_id": cand.candidate_id,
                    "mask_path": str(cand.mask_path),
                    "prompt_type": cand.prompt_type,
                    "source_image": str(args.ficture_image),
                    "text_prompt_family": "base_label_prompts_plus_ficture_factor_celltype_gene_hints",
                }
            )
    write_csv(args.output_dir / "openai_clip_semantic_candidate_scores.csv", score_rows)

    by_source: Dict[str, List[Candidate]] = {}
    for cand in candidates:
        by_source.setdefault(cand.source, []).append(cand)
    scopes: List[Tuple[str, List[Candidate]]] = [("all_sources", list(candidates))]
    scopes.extend((f"{source}_only", cands) for source, cands in sorted(by_source.items()))

    selection_rows: List[dict] = []
    best_rows: Dict[Tuple[str, str], dict] = {}
    for scope, scope_candidates in scopes:
        for slug, display in label_order:
            gt = gt_masks.get(slug)
            if gt is None:
                continue
            scored = [
                (clip_scores.get(c.mask_path, {}).get(slug, 0.0), c)
                for c in scope_candidates
            ]
            for top_k in args.top_ks:
                pred, selected = select_union(
                    scored,
                    get_mask,
                    top_k=top_k,
                    max_overlap=args.max_overlap,
                    shape_hw=target_shape,
                )
                row = {
                    "pool_scope": scope,
                    "label": slug,
                    "display": display,
                    "ranker": "openai_clip_ficture_semantic",
                    "top_k": top_k,
                    "selected": ";".join(f"{c.source}/{c.run}/{c.setting}/{c.candidate_id}" for c in selected),
                    "selected_sources": ";".join(c.source for c in selected),
                    "n_selected": len(selected),
                    **metrics(pred, gt),
                }
                selection_rows.append(row)
                key = (scope, slug)
                if key not in best_rows or float(row["dice"]) > float(best_rows[key]["dice"]):
                    best_rows[key] = row

    fieldnames = [
        "pool_scope",
        "label",
        "display",
        "ranker",
        "top_k",
        "selected",
        "selected_sources",
        "n_selected",
        "dice",
        "iou",
        "precision",
        "recall",
        "pred_pixels",
        "gt_pixels",
    ]
    write_csv(args.output_dir / "openai_clip_semantic_selection_metrics.csv", selection_rows, fieldnames)
    write_csv(args.output_dir / "openai_clip_semantic_best_by_label.csv", list(best_rows.values()), fieldnames)

    summary = {
        "ranker": "openai_clip_ficture_semantic",
        "meaning": (
            "Direct retrieval: each H&E or FICTURE candidate mask is applied to the official FICTURE "
            "factor-color ROI image; OpenAI CLIP compares that masked crop with label prompts augmented "
            "with FICTURE factor/cell-type/gene hints; annotation masks are used only after scoring."
        ),
        "clip_model": args.clip_model,
        "ficture_image": str(args.ficture_image),
        "factor_semantic_legend": str(args.factor_semantic_legend),
        "summary_path": str(args.summary_path),
        "official_summary": str(args.official_summary) if args.official_summary else None,
        "candidate_roots": candidate_roots,
        "n_candidates": len(candidates),
        "labels": args.label_slugs,
        "top_ks": args.top_ks,
        "max_overlap": args.max_overlap,
        "output_files": {
            "candidate_scores": "openai_clip_semantic_candidate_scores.csv",
            "selection_metrics": "openai_clip_semantic_selection_metrics.csv",
            "best_by_label": "openai_clip_semantic_best_by_label.csv",
            "candidate_pool_counts": "candidate_pool_counts.csv",
            "text_prompts": "openai_clip_semantic_text_prompts.csv",
        },
    }
    (args.output_dir / "openai_clip_semantic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
