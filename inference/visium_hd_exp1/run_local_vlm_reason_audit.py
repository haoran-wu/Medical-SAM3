#!/usr/bin/env python3
"""Run local VLM full-description scoring with visible evidence fields.

This is a debugging/audit runner. It keeps the full tissue-class descriptions
and the RGB+cell-type FICTURE legend, but asks the VLM to return concise,
visible evidence fields so we can inspect why it collapses to a class.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from run_local_vlm_cross_label_top1_accuracy import (
    CLASS_DISPLAY,
    CLASS_KEYS,
    TRUE_LABEL_TO_CLASS,
    predicted_from_scores,
    row_key_string,
    write_accuracy_tables,
    write_same_class_retrieval_tables,
)
from run_paired_vlm_hit_test import generate, load_vlm, write_csv
from vlm_prompt_contract import DEFAULT_FACTOR_LEGEND_CSV, DETAILED_TISSUE_CLASSES, build_ficture_legend_text


SYSTEM_PROMPT = (
    "You are a careful pathology image classifier. "
    "Return only one valid JSON object. "
    "Give concise visible evidence, not hidden chain-of-thought."
)


def build_reason_prompt(factor_legend_csv: Path) -> str:
    legend = build_ficture_legend_text(factor_legend_csv)
    return f"""You are given two aligned crops of the same candidate region:

Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

FICTURE color legend for Image 2:

{legend}

Score how likely this candidate region belongs to each tissue class.

{DETAILED_TISSUE_CLASSES}

Rules:
- Judge only the candidate region.
- Use H&E morphology as the main evidence.
- Use FICTURE RGB/cell-type information only as supporting evidence.
- Do not classify by FICTURE color alone.
- Return exactly one JSON object.
- The JSON must have these keys: scores, predicted_class, he_evidence, ficture_evidence, visible_reason.
- scores must contain exactly these six integer keys from 0 to 100:
  bronchiola, alveoli, vessels, tumor, stroma, immune_infiltration.
- predicted_class must be one of the six class keys.
- he_evidence, ficture_evidence, and visible_reason must each be one short sentence.
- Do not include markdown, code fences, extra text, or hidden chain-of-thought.
"""


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start_positions = [idx for idx, char in enumerate(stripped) if char == "{"]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return ast.literal_eval(candidate)
                        except Exception:
                            break
    raise ValueError("No valid JSON object found")


def parse_reason_response(text: str) -> Tuple[Dict[str, int], dict, str]:
    data = extract_json_object(text)
    scores_raw = data.get("scores")
    if not isinstance(scores_raw, dict):
        scores_raw = {key: data.get(key) for key in CLASS_KEYS if key in data}
    if set(scores_raw) != set(CLASS_KEYS):
        raise ValueError(f"Expected score keys {CLASS_KEYS}, got {sorted(scores_raw)}")
    scores: Dict[str, int] = {}
    for key in CLASS_KEYS:
        value = scores_raw[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} score is boolean")
        if isinstance(value, str):
            value = value.strip()
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            numeric *= 100.0
        integer = int(round(numeric))
        if not 0 <= integer <= 100:
            raise ValueError(f"{key} score out of range: {integer}")
        scores[key] = integer

    predicted, top_tie, top_score = predicted_from_scores(scores)
    model_pred = str(data.get("predicted_class") or "").strip()
    if model_pred not in CLASS_KEYS:
        model_pred = predicted
    parsed = {
        "scores": scores,
        "predicted_class": model_pred,
        "score_argmax_class": predicted,
        "top_score": top_score,
        "top_score_tie": top_tie,
        "he_evidence": str(data.get("he_evidence") or "")[:500],
        "ficture_evidence": str(data.get("ficture_evidence") or "")[:500],
        "visible_reason": str(data.get("visible_reason") or "")[:700],
    }
    return scores, parsed, json.dumps(parsed, sort_keys=True)


def write_outputs(output_dir: Path, prediction_rows: List[dict], failed_rows: List[dict]) -> None:
    prediction_rows = sorted(prediction_rows, key=lambda item: int(item["row_index"]))
    pred_path = output_dir / "per_candidate_reason_predictions.csv"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-legend-csv", type=Path, default=DEFAULT_FACTOR_LEGEND_CSV)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    user_prompt = build_reason_prompt(args.factor_legend_csv)
    (args.output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (args.output_dir / "prompt_user_template.txt").write_text(user_prompt, encoding="utf-8")
    if rows:
        (args.output_dir / "prompt_user_first_row.txt").write_text(user_prompt, encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(args.pool_csv),
                "factor_legend_csv": str(args.factor_legend_csv),
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "class_keys": CLASS_KEYS,
                "output_schema": "scores_plus_visible_reason",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pred_path = args.output_dir / "per_candidate_reason_predictions.csv"
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
            scores, parsed, parsed_json = parse_reason_response(raw)
            predicted = parsed["score_argmax_class"]
            top_tie = bool(parsed["top_score_tie"])
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
                "model_predicted_class": parsed["predicted_class"],
                "top_score": parsed["top_score"],
                "top_score_tie": "true" if top_tie else "false",
                "is_correct": "true" if is_correct else "false",
                "parse_status": "ok",
                "he_evidence": parsed["he_evidence"],
                "ficture_evidence": parsed["ficture_evidence"],
                "visible_reason": parsed["visible_reason"],
                **{class_key: scores[class_key] for class_key in CLASS_KEYS},
            }
            prediction_rows = [old for old in prediction_rows if old["row_key"] != key] + [out]
            failed_rows = [old for old in failed_rows if old.get("row_key") != key]
            done_keys.add(key)
            with responses_path.open("a") as handle:
                handle.write(json.dumps({"row_key": key, "parsed_json": parsed_json, "raw_response": raw}) + "\n")
            print(f"Scored {idx}/{len(rows)} true={true_class} pred={predicted or 'TIE'} correct={is_correct}", flush=True)
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
            print(f"FAILED {idx}/{len(rows)} {key}: {str(exc)[:200]}", flush=True)
        write_outputs(args.output_dir, prediction_rows, failed_rows)

    if failed_rows:
        print(f"{len(failed_rows)} rows failed; rerun with --resume after fixing the issue.", flush=True)
    write_outputs(args.output_dir, prediction_rows, failed_rows)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
