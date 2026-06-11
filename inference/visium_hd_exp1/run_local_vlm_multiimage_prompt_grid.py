#!/usr/bin/env python3
"""Run multiple context prompt/image-mode variants in one local VLM process."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import List

from PIL import Image

from run_local_vlm_cross_label_top1_accuracy import (
    CLASS_KEYS,
    TRUE_LABEL_TO_CLASS,
    parse_scores,
    predicted_from_scores,
    row_key_string,
    write_accuracy_tables,
    write_same_class_retrieval_tables,
)
from run_local_vlm_multiimage_cross_label import SYSTEM_PROMPT, build_context_prompt, image_columns, prompt_for_row
from run_paired_vlm_hit_test import generate, load_vlm, write_csv
from vlm_prompt_contract import DEFAULT_FACTOR_LEGEND_CSV


def parse_prompt_grid(text: str) -> List[tuple[str, str, str]]:
    configs: List[tuple[str, str, str]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) == 2:
            style, mode = parts
            slug = f"{style}_{mode}"
        elif len(parts) == 3:
            style, mode, slug = parts
        else:
            raise ValueError(f"Bad prompt-grid item: {item!r}")
        configs.append((style, mode, slug))
    if not configs:
        raise ValueError("No prompt-grid configs were provided")
    return configs


def write_outputs(
    output_dir: Path,
    prediction_rows: list[dict],
    failed_rows: list[dict],
) -> None:
    prediction_rows = sorted(prediction_rows, key=lambda item: int(item["row_index"]))
    pred_path = output_dir / "per_candidate_predictions.csv"
    cross_path = output_dir / "cross_label_scores.csv"
    failed_path = output_dir / "failed_rows.csv"
    write_csv(pred_path, prediction_rows)
    cross_rows = [
        {
            "row_index": pred["row_index"],
            "row_key": pred["row_key"],
            "candidate_uid": pred["candidate_uid"],
            "true_class": pred["true_class"],
            "sample_bucket": pred["sample_bucket"],
            **{class_key: pred[class_key] for class_key in CLASS_KEYS},
        }
        for pred in prediction_rows
    ]
    write_csv(cross_path, cross_rows)
    write_csv(
        failed_path,
        failed_rows,
        fieldnames=["row_index", "row_key", "candidate_uid", "true_label", "sample_bucket", "error", "raw_response"],
    )
    write_accuracy_tables(output_dir, prediction_rows)
    write_same_class_retrieval_tables(output_dir, prediction_rows)


def run_one_config(
    *,
    vlm,
    rows: list[dict],
    pool_dir: Path,
    output_dir: Path,
    model: str,
    factor_legend_csv: Path,
    style: str,
    image_mode: str,
    device: str,
    max_new_tokens: int,
    resume: bool,
) -> None:
    os.environ["CONTEXT_PROMPT_STYLE"] = style
    os.environ["CONTEXT_IMAGE_MODE"] = image_mode
    user_prompt = build_context_prompt(factor_legend_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (output_dir / "prompt_user_template.txt").write_text(user_prompt, encoding="utf-8")
    if rows:
        (output_dir / "prompt_user_first_row.txt").write_text(prompt_for_row(user_prompt, rows[0]), encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(pool_dir / "public_vlm_requests.csv"),
                "factor_legend_csv": str(factor_legend_csv),
                "model": model,
                "max_new_tokens": max_new_tokens,
                "input_style": "multiimage_context_aware_piece",
                "context_prompt_style": style,
                "context_image_mode": image_mode,
                "class_keys": CLASS_KEYS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_path = output_dir / "per_candidate_predictions.csv"
    failed_path = output_dir / "failed_rows.csv"
    responses_path = output_dir / "raw_responses.jsonl"
    prediction_rows: list[dict] = []
    done_keys = set()
    if resume and pred_path.exists():
        prediction_rows = list(csv.DictReader(pred_path.open()))
        done_keys = {row["row_key"] for row in prediction_rows if row.get("parse_status") == "ok"}
    failed_rows: list[dict] = []
    if resume and failed_path.exists():
        failed_rows = list(csv.DictReader(failed_path.open()))

    for idx, row in enumerate(rows, start=1):
        key = row_key_string(row)
        if key in done_keys:
            continue
        raw = ""
        try:
            images = [Image.open(pool_dir / rel).convert("RGB") for rel in image_columns(row)]
            raw = generate(vlm, device, images, SYSTEM_PROMPT, prompt_for_row(user_prompt, row), max_new_tokens)
            scores, parsed_json = parse_scores(raw)
            predicted, top_tie, top_score = predicted_from_scores(scores)
            true_class = TRUE_LABEL_TO_CLASS[row["label"]]
            is_correct = bool(predicted and predicted == true_class and not top_tie)
            out = {
                "row_index": idx,
                "row_key": key,
                "candidate_uid": row["candidate_uid"],
                "true_label": row["label"],
                "true_class": true_class,
                "display": row["display"],
                "sample_bucket": row["sample_bucket"],
                "source": row["source"],
                "run": row["run"],
                "setting": row["setting"],
                "candidate_id": row["candidate_id"],
                "predicted_class": predicted,
                "top_score": top_score,
                "top_score_tie": "true" if top_tie else "false",
                "is_correct": "true" if is_correct else "false",
                "parse_status": "ok",
                **{class_key: scores[class_key] for class_key in CLASS_KEYS},
            }
            prediction_rows = [old for old in prediction_rows if old["row_key"] != key] + [out]
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            done_keys.add(key)
            with responses_path.open("a") as handle:
                handle.write(json.dumps({"row_key": key, "parsed_json": parsed_json, "raw_response": raw}) + "\n")
            print(
                f"[{style}/{image_mode}] Scored {idx}/{len(rows)} true={true_class} "
                f"pred={predicted or 'TIE'} correct={is_correct}",
                flush=True,
            )
        except Exception as exc:
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            failed_rows.append(
                {
                    "row_index": idx,
                    "row_key": key,
                    "candidate_uid": row["candidate_uid"],
                    "true_label": row["label"],
                    "sample_bucket": row["sample_bucket"],
                    "error": str(exc)[:2400],
                    "raw_response": raw[:2400],
                }
            )
            print(f"[{style}/{image_mode}] FAILED {idx}/{len(rows)} {key}: {str(exc)[:200]}", flush=True)
        write_outputs(output_dir, prediction_rows, failed_rows)
    print(f"[{style}/{image_mode}] Output: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prompt-grid", required=True)
    parser.add_argument(
        "--factor-legend-csv",
        type=Path,
        default=Path(os.environ.get("FACTOR_LEGEND_CSV", DEFAULT_FACTOR_LEGEND_CSV)),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.limit:
        rows = rows[: args.limit]
    label_counts = Counter(row["label"] for row in rows)
    if not rows:
        raise SystemExit("Input pool has no rows")
    if set(label_counts) - set(TRUE_LABEL_TO_CLASS):
        raise SystemExit(f"Unexpected labels in input pool: {sorted(set(label_counts) - set(TRUE_LABEL_TO_CLASS))}")
    configs = parse_prompt_grid(args.prompt_grid)
    args.output_base.mkdir(parents=True, exist_ok=True)

    vlm = load_vlm(args.model, args.device)
    pool_dir = args.pool_csv.parent
    for style, mode, slug in configs:
        output_dir = args.output_base / f"{args.run_name}_{slug}"
        run_one_config(
            vlm=vlm,
            rows=rows,
            pool_dir=pool_dir,
            output_dir=output_dir,
            model=args.model,
            factor_legend_csv=args.factor_legend_csv,
            style=style,
            image_mode=mode,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
