#!/usr/bin/env python3
"""Run local VLM evidence-first audit for H&E grayscale-white + FICTURE inputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import List

from PIL import Image

from run_local_vlm_cross_label_top1_accuracy import CLASS_KEYS, TRUE_LABEL_TO_CLASS, row_key_string
from run_local_vlm_evidence_first_audit import flatten_reason_fields, parse_response, write_outputs
from run_openrouter_vlm_evidence_first_audit import SYSTEM_PROMPT, build_prompt
from run_paired_vlm_hit_test import generate, load_vlm
from vlm_prompt_contract import DEFAULT_FACTOR_LEGEND_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-legend-csv", type=Path, default=DEFAULT_FACTOR_LEGEND_CSV)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2200)
    parser.add_argument("--candidate-uid", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.candidate_uid:
        wanted = set(args.candidate_uid)
        rows = [row for row in rows if row.get("candidate_uid") in wanted]
    if not rows:
        raise SystemExit("Input pool has no selected rows")
    label_counts = Counter(row["label"] for row in rows)
    unexpected = sorted(set(label_counts) - set(TRUE_LABEL_TO_CLASS))
    if unexpected:
        raise SystemExit(f"Unexpected labels in input pool: {unexpected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    user_prompt = build_prompt(args.factor_legend_csv)
    (args.output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (args.output_dir / "prompt_user_template.txt").write_text(user_prompt, encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(args.pool_csv),
                "factor_legend_csv": str(args.factor_legend_csv),
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "class_keys": CLASS_KEYS,
                "input_style": "H&E grayscale morphology white background + FICTURE gray reverse-blur",
                "candidate_uid_filter": args.candidate_uid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_path = args.output_dir / "per_candidate_evidence_first_predictions.csv"
    failed_path = args.output_dir / "failed_rows.csv"
    responses_path = args.output_dir / "raw_responses.jsonl"
    prediction_rows: List[dict] = []
    done_keys = set()
    if args.resume and pred_path.exists():
        prediction_rows = list(csv.DictReader(pred_path.open()))
        done_keys = {row["row_key"] for row in prediction_rows if row.get("parse_status") == "ok"}
    failed_rows: List[dict] = []
    if args.resume and failed_path.exists():
        failed_rows = list(csv.DictReader(failed_path.open()))

    vlm = load_vlm(args.model, args.device)
    pool_dir = args.pool_csv.parent
    for idx, row in enumerate(rows, start=1):
        key = row_key_string(row)
        if key in done_keys:
            continue
        raw = ""
        try:
            he_image = Image.open(pool_dir / row["he_crop_rel"]).convert("RGB")
            ficture_image = Image.open(pool_dir / row["ficture_crop_rel"]).convert("RGB")
            raw = generate(vlm, args.device, [he_image, ficture_image], SYSTEM_PROMPT, user_prompt, args.max_new_tokens)
            scores, parsed, parsed_json = parse_response(raw)
            predicted = parsed["score_argmax_class"]
            top_tie = bool(parsed["top_score_tie"])
            true_class = TRUE_LABEL_TO_CLASS[row["label"]]
            is_correct = bool(predicted and predicted == true_class and not top_tie)
            out = {
                "row_index": row.get("row_index") or idx,
                "run_order": idx,
                "row_key": key,
                "candidate_uid": row["candidate_uid"],
                "true_label": row["label"],
                "true_class": true_class,
                "display": row.get("display", true_class),
                "sample_bucket": row.get("sample_bucket", ""),
                "source": row.get("source", ""),
                "run": row.get("run", ""),
                "setting": row.get("setting", ""),
                "candidate_id": row.get("candidate_id", ""),
                "predicted_class": predicted,
                "model_predicted_class": parsed["predicted_class"],
                "top_score": parsed["top_score"],
                "top_score_tie": "true" if top_tie else "false",
                "is_correct": "true" if is_correct else "false",
                "parse_status": "ok",
                **{class_key: scores[class_key] for class_key in CLASS_KEYS},
                **flatten_reason_fields(parsed),
            }
            prediction_rows = [old for old in prediction_rows if old["row_key"] != key] + [out]
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            done_keys.add(key)
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"row_key": key, "parsed_json": parsed_json, "raw_response": raw}, ensure_ascii=False) + "\n")
            print(f"Scored {idx}/{len(rows)} uid={row['candidate_uid']} true={true_class} pred={predicted or 'TIE'} correct={is_correct}", flush=True)
        except Exception as exc:
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            failed_rows.append(
                {
                    "row_index": row.get("row_index") or idx,
                    "row_key": key,
                    "candidate_uid": row["candidate_uid"],
                    "true_label": row["label"],
                    "sample_bucket": row.get("sample_bucket", ""),
                    "error": str(exc)[:2400],
                    "raw_response": raw[:2400],
                }
            )
            print(f"FAILED {idx}/{len(rows)} {key}: {str(exc)[:240]}", flush=True)
        write_outputs(args.output_dir, prediction_rows, failed_rows)

    write_outputs(args.output_dir, prediction_rows, failed_rows)
    if failed_rows:
        print(f"{len(failed_rows)} rows failed; rerun with --resume after fixing the issue.", flush=True)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
