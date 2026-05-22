#!/usr/bin/env python3
"""Pure CLIP image-text retrieval over large SAM candidate pools.

This is intentionally narrower than rank_multimodal_sam_candidates.py: it only
computes the SaLIP-style CLIP image/text similarity score (`salip_clip`) and
then evaluates top-k unions against the held-out annotation masks.
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
        description="Run pure salip_clip retrieval over one or more candidate pools."
    )
    parser.add_argument("--candidate-root", action="append", required=True, help="source=/path/to/root")
    parser.add_argument("--he-image", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
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
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
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
    """Load annotation masks, allowing archived summaries with stale mask paths."""
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


def main() -> None:
    args = parse_args()
    verify_official(args.official_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    he = np.array(Image.open(args.he_image).convert("RGB"))
    target_shape = he.shape[:2]
    gt_masks = load_gt_masks(args.summary_path, target_shape)

    candidates = load_candidates(args.candidate_root)
    if not candidates:
        raise SystemExit("No candidates found.")

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

    print(f"Loaded {len(candidates)} total saved candidates for salip_clip retrieval.", flush=True)
    write_csv(args.output_dir / "candidate_pool_counts.csv", source_counts(candidates))

    clip_scores = compute_clip_scores(
        candidates=candidates,
        get_mask=get_mask,
        image=he,
        model_name=args.clip_model,
        batch_size=args.clip_batch_size,
        device=args.clip_device,
        text_prompts_by_label=CLIP_TEXT_PROMPTS,
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
                    "ranker": "salip_clip",
                    "score": scores.get(slug, 0.0),
                    "source": cand.source,
                    "run": cand.run,
                    "setting": cand.setting,
                    "candidate_id": cand.candidate_id,
                    "mask_path": str(cand.mask_path),
                    "prompt_type": cand.prompt_type,
                }
            )
    write_csv(args.output_dir / "salip_clip_candidate_scores.csv", score_rows)

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
                    "ranker": "salip_clip",
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
    write_csv(args.output_dir / "salip_clip_selection_metrics.csv", selection_rows, fieldnames)
    write_csv(args.output_dir / "salip_clip_best_by_label.csv", list(best_rows.values()), fieldnames)

    summary = {
        "ranker": "salip_clip",
        "meaning": "pure CLIP image-text similarity between the candidate-masked H&E crop and label text prompts",
        "clip_model": args.clip_model,
        "he_image": str(args.he_image),
        "summary_path": str(args.summary_path),
        "candidate_roots": args.candidate_root,
        "n_candidates_total": len(candidates),
        "candidate_counts": source_counts(candidates),
        "top_ks": args.top_ks,
        "max_overlap": args.max_overlap,
        "mask_cache_size": args.mask_cache_size,
        "labels": [slug for slug, _ in label_order],
    }
    (args.output_dir / "salip_clip_large_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved salip_clip outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
