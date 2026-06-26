#!/usr/bin/env python3
"""Run local VLM scoring with evidence-first, per-class score reasons.

This audit is different from the earlier "visible_reason" run: the model must
write concise H&E/FICTURE evidence for each tissue class and then assign that
class score. This lets us inspect why the top class is high and why the hidden
true class may be low without writing post-hoc explanations ourselves.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from run_local_vlm_cross_label_top1_accuracy import (
    CLASS_KEYS,
    TRUE_LABEL_TO_CLASS,
    predicted_from_scores,
    row_key_string,
    write_accuracy_tables,
    write_same_class_retrieval_tables,
)
from run_local_vlm_reason_audit import extract_json_object
from run_paired_vlm_hit_test import generate, load_vlm, write_csv
from vlm_prompt_contract import DEFAULT_FACTOR_LEGEND_CSV, DETAILED_TISSUE_CLASSES, build_ficture_legend_text


SYSTEM_PROMPT = (
    "You are a careful pathology image classifier. "
    "Use only visible image evidence. Return only one valid JSON object."
)

CLASS_REASON_FIELDS = [
    "score",
    "he_support",
    "he_against_or_missing",
    "ficture_support",
    "ficture_against_or_missing",
    "score_reason",
]


def build_prompt(factor_legend_csv: Path) -> str:
    legend = build_ficture_legend_text(factor_legend_csv)
    return f"""You are given two aligned crops of the same candidate region.

Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

Image 2: FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

FICTURE color legend for Image 2:

{legend}

Task:
Score how likely this candidate region belongs to each tissue class.

{DETAILED_TISSUE_CLASSES}

Evidence-first scoring rule:
For each class, first inspect H&E morphology and FICTURE RGB/cell-type evidence.
Then assign that class score from 0 to 100 based on that evidence.
The class score must be explained by the evidence written for that same class.

Important:
- Judge only the highlighted candidate region.
- H&E morphology is the main evidence.
- FICTURE RGB/cell-type information is supporting evidence only.
- Do not classify by FICTURE color alone.
- If a score is low, explicitly say what visual evidence is missing or contradicts that class.
- If a score is high, explicitly say whether the support comes from H&E, FICTURE, or both.

Return exactly one JSON object with this schema:
{{
  "per_class": {{
    "bronchiola": {{
      "score": 0,
      "he_support": "...",
      "he_against_or_missing": "...",
      "ficture_support": "...",
      "ficture_against_or_missing": "...",
      "score_reason": "..."
    }},
    "alveoli": {{ "... same fields ..." }},
    "vessels": {{ "... same fields ..." }},
    "tumor": {{ "... same fields ..." }},
    "stroma": {{ "... same fields ..." }},
    "immune_infiltration": {{ "... same fields ..." }}
  }},
  "scores": {{
    "bronchiola": 0,
    "alveoli": 0,
    "vessels": 0,
    "tumor": 0,
    "stroma": 0,
    "immune_infiltration": 0
  }},
  "predicted_class": "one of the six class keys",
  "top_score_reason": "one short sentence explaining why the top class received the highest score"
}}

Rules:
- Use exactly the six class keys shown above.
- Each score must be an integer from 0 to 100.
- Keep each evidence field concise, one short sentence or phrase.
- Do not include markdown, code fences, hidden chain-of-thought, or extra text.
"""


def _to_int_score(value: object, key: str) -> int:
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
    return integer


def parse_response(text: str) -> Tuple[Dict[str, int], dict, str]:
    data = extract_json_object(text)
    per_class = data.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError("Missing per_class object")
    if set(per_class) != set(CLASS_KEYS):
        raise ValueError(f"Expected per_class keys {CLASS_KEYS}, got {sorted(per_class)}")

    scores_raw = data.get("scores")
    if not isinstance(scores_raw, dict):
        scores_raw = {}
    scores: Dict[str, int] = {}
    cleaned_per_class: dict[str, dict[str, object]] = {}
    for key in CLASS_KEYS:
        entry = per_class.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"per_class.{key} is not an object")
        score_value = entry.get("score", scores_raw.get(key))
        scores[key] = _to_int_score(score_value, key)
        cleaned: dict[str, object] = {"score": scores[key]}
        for field in CLASS_REASON_FIELDS:
            if field == "score":
                continue
            cleaned[field] = str(entry.get(field) or "")[:700]
        cleaned_per_class[key] = cleaned

    predicted, top_tie, top_score = predicted_from_scores(scores)
    model_pred = str(data.get("predicted_class") or "").strip()
    if model_pred not in CLASS_KEYS:
        model_pred = predicted or ""
    parsed = {
        "scores": scores,
        "per_class": cleaned_per_class,
        "predicted_class": model_pred,
        "score_argmax_class": predicted or "",
        "top_score": top_score,
        "top_score_tie": top_tie,
        "top_score_reason": str(data.get("top_score_reason") or "")[:1000],
    }
    return scores, parsed, json.dumps(parsed, sort_keys=True)


def flatten_reason_fields(parsed: dict) -> dict:
    flat: dict[str, object] = {"top_score_reason": parsed.get("top_score_reason", "")}
    per_class = parsed["per_class"]
    for label in CLASS_KEYS:
        entry = per_class[label]
        for field in CLASS_REASON_FIELDS:
            flat[f"{label}_{field}"] = entry.get(field, "")
    return flat


def write_outputs(output_dir: Path, prediction_rows: List[dict], failed_rows: List[dict]) -> None:
    prediction_rows = sorted(prediction_rows, key=lambda item: int(item["row_index"]))
    write_csv(output_dir / "per_candidate_evidence_first_predictions.csv", prediction_rows)
    write_csv(
        output_dir / "cross_label_scores.csv",
        [
            {
                "row_index": pred["row_index"],
                "row_key": pred["row_key"],
                "candidate_uid": pred["candidate_uid"],
                "true_class": pred["true_class"],
                "sample_bucket": pred["sample_bucket"],
                **{class_key: pred[class_key] for class_key in CLASS_KEYS},
            }
            for pred in prediction_rows
        ],
    )
    write_csv(
        output_dir / "failed_rows.csv",
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
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate-uid", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.candidate_uid:
        wanted = set(args.candidate_uid)
        rows = [row for row in rows if row["candidate_uid"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    label_counts = Counter(row["label"] for row in rows)
    if not rows:
        raise SystemExit("Input pool has no rows")
    if set(label_counts) - set(TRUE_LABEL_TO_CLASS):
        raise SystemExit(f"Unexpected labels in input pool: {sorted(set(label_counts) - set(TRUE_LABEL_TO_CLASS))}")

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
                "output_schema": "evidence_first_per_class_reasons",
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
                **{class_key: scores[class_key] for class_key in CLASS_KEYS},
                **flatten_reason_fields(parsed),
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
                    "row_index": row.get("row_index") or idx,
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
