#!/usr/bin/env python3
"""Run OpenRouter VLM scoring with evidence-first per-class reasons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

from run_local_vlm_evidence_first_audit import (
    CLASS_KEYS,
    TRUE_LABEL_TO_CLASS,
    flatten_reason_fields,
    parse_response,
    write_outputs,
)
from run_local_vlm_cross_label_top1_accuracy import row_key_string
from run_openrouter_vlm_hit_test import image_data_url
from vlm_prompt_contract import DEFAULT_FACTOR_LEGEND_CSV, DETAILED_TISSUE_CLASSES, build_ficture_legend_text


SYSTEM_PROMPT = (
    "You are a careful pathology image classifier. "
    "Use only visible image evidence. Return only one valid JSON object."
)


def load_api_key(env_name: str, key_file: Path | None) -> str:
    api_key = os.environ.get(env_name, "").strip()
    if api_key:
        return api_key
    if key_file is not None and key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


def build_prompt(factor_legend_csv: Path, he_style: str = "graywhite") -> str:
    legend = build_ficture_legend_text(factor_legend_csv)
    if he_style == "old_reverse_blur":
        he_description = """Image 1: H&E gray reverse-blur crop.
The candidate region is sharp and full-color H&E staining; the outside region is grayscale and blurred.
H&E colors are ordinary H&E staining colors only. H&E colors are not FICTURE RGB/cell-type colors."""
    else:
        he_description = """Image 1: H&E grayscale morphology crop.
The candidate mask pixels show grayscale H&E morphology only. The outside of the candidate mask is pure white. H&E color was intentionally removed."""
    return f"""You are given two aligned crops of the same candidate region.

{he_description}

Image 2: FICTURE gray reverse-blur crop.
The candidate region is sharp and full color; the outside region is grayscale and blurred.

FICTURE color legend for Image 2 only:

{legend}

Critical image-separation rule:
- The FICTURE RGB legend applies only to Image 2.
- Never interpret H&E color, H&E grayscale intensity, H&E staining, or H&E pixels as FICTURE RGB/cell type.
- H&E pink, purple, blue, or gray staining is not Color 0, Color 1, or any other FICTURE legend color.
- If you cite FICTURE support for a class, cite the Image 2 RGB/color/cell type evidence. If Image 2 does not show that evidence, say it is missing.
- H&E morphology and FICTURE color/cell-type evidence are separate evidence streams.

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


def openrouter_evidence_chat(
    *,
    api_key: str,
    model: str,
    he_data_url: str,
    ficture_data_url: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    response_format: bool,
    he_style: str,
) -> Tuple[str, dict]:
    if he_style == "old_reverse_blur":
        he_lead = (
            "Image 1 follows: H&E gray reverse-blur crop. Candidate is sharp full-color H&E; outside is grayscale/blurred. "
            "Do not interpret H&E staining colors as FICTURE RGB/cell type."
        )
    else:
        he_lead = "Image 1 follows: H&E grayscale morphology crop with white background outside the candidate mask."
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": he_lead},
                    {"type": "image_url", "image_url": {"url": he_data_url}},
                    {"type": "text", "text": "Image 2 follows: FICTURE gray reverse-blur crop. The RGB/cell-type legend applies only to this FICTURE image."},
                    {"type": "image_url", "image_url": {"url": ficture_data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/Medical-SAM3",
        "X-Title": "Medical-SAM3 evidence-first GPT-5.5 audit",
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) for item in content)
    if content is None:
        content = json.dumps(
            {
                "error": "openrouter_returned_no_message_content",
                "finish_reason": choice.get("finish_reason"),
                "native_finish_reason": choice.get("native_finish_reason"),
                "message_keys": sorted(message.keys()),
            }
        )
    return str(content), payload


def safe_request(
    *,
    api_key: str,
    model: str,
    he_data_url: str,
    ficture_data_url: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
    response_format: bool,
    he_style: str,
) -> Tuple[Dict[str, int], dict, str, dict]:
    last_error = ""
    last_raw = ""
    last_payload: dict = {}
    for attempt in range(retries + 1):
        try:
            raw, payload = openrouter_evidence_chat(
                api_key=api_key,
                model=model,
                he_data_url=he_data_url,
                ficture_data_url=ficture_data_url,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
                he_style=he_style,
            )
            last_raw = raw
            last_payload = payload
            scores, parsed, parsed_json = parse_response(raw)
            return scores, parsed, parsed_json, payload
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:1600]
            last_error = f"OpenRouter HTTP {exc.code}: {body_text}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(json.dumps({"error": last_error, "raw_response": last_raw, "payload": last_payload})[:2600])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-legend-csv", type=Path, default=DEFAULT_FACTOR_LEGEND_CSV)
    parser.add_argument("--model", default="openai/gpt-5.5")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-key-file", type=Path, default=Path(".openrouter_api_key"))
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--candidate-uid", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--he-style", choices=["graywhite", "old_reverse_blur"], default="graywhite")
    args = parser.parse_args()

    api_key = load_api_key(args.api_key_env, args.api_key_file)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or provide --api-key-file.")

    rows = list(csv.DictReader(args.pool_csv.open()))
    if args.candidate_uid:
        wanted = set(args.candidate_uid)
        rows = [row for row in rows if row.get("candidate_uid") in wanted]
    if not rows:
        raise SystemExit("No rows selected")
    unexpected = sorted({row["label"] for row in rows} - set(TRUE_LABEL_TO_CLASS))
    if unexpected:
        raise SystemExit(f"Unexpected labels in input pool: {unexpected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    user_prompt = build_prompt(args.factor_legend_csv, he_style=args.he_style)
    (args.output_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (args.output_dir / "prompt_user_template.txt").write_text(user_prompt, encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pool_csv": str(args.pool_csv),
                "factor_legend_csv": str(args.factor_legend_csv),
                "model": args.model,
                "max_side": args.max_side,
                "jpeg_quality": args.jpeg_quality,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "class_keys": CLASS_KEYS,
                "input_style": "H&E grayscale morphology white background + FICTURE gray reverse-blur",
                "he_style": args.he_style,
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

    pool_dir = args.pool_csv.parent
    for idx, row in enumerate(rows, start=1):
        key = row_key_string(row)
        if key in done_keys:
            continue
        raw = ""
        try:
            he_path = pool_dir / row["he_crop_rel"]
            ficture_path = pool_dir / row["ficture_crop_rel"]
            scores, parsed, parsed_json, payload = safe_request(
                api_key=api_key,
                model=args.model,
                he_data_url=image_data_url(he_path, args.max_side, args.jpeg_quality),
                ficture_data_url=image_data_url(ficture_path, args.max_side, args.jpeg_quality),
                user_prompt=user_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                response_format=not args.no_response_format,
                he_style=args.he_style,
            )
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
                handle.write(json.dumps({"row_key": key, "parsed_json": parsed_json, "payload": payload}, ensure_ascii=False) + "\n")
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
