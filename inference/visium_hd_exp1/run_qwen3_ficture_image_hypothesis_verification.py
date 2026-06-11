#!/usr/bin/env python3
"""Verify Step 1 cell-type hypotheses with the FICTURE image only.

This is the FICTURE-image counterpart to H&E morphology verification:

Step 1:
  Use structured FICTURE-derived RGB/cell-type composition text to
  keep plausible / uncertain tissue-class hypotheses.

This script:
  For every retained Step 1 hypothesis, show only the FICTURE reverse-blur crop
  and ask whether the spatial FICTURE image supports that hypothesis.

It intentionally does not use H&E.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Iterable

from PIL import Image

from run_paired_vlm_hit_test import generate, load_vlm


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]


SYSTEM_PROMPT = (
    "You are a careful spatial transcriptomics image verifier. "
    "You judge only the FICTURE image for a highlighted candidate mask. "
    "Return only valid JSON."
)


USER_TEMPLATE_FICTURE_HYPOTHESIS_VERIFY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look at the FICTURE candidate crop.

Image:
FICTURE candidate reverse-blur crop. The candidate region is sharp and full color; the outside region is grayscale and blurred.

Task:
Judge whether the FICTURE image pattern inside the highlighted candidate mask supports this one hypothesis.
Use only the FICTURE image and the Step 1 hypothesis list. Do not use H&E morphology.

Important:
- FICTURE colors are cell-type / molecular-domain signals, not direct tissue labels.
- A single cell type can appear in more than one tissue class, and one tissue class can contain multiple cell types.
- Therefore, support should mean the FICTURE pattern is spatially coherent and compatible with the hypothesis, not merely that one color is present.
- This is a filtering step, not the final tissue decision.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "ficture_support_score": 0,
  "supported": false
}}

ficture_support_score must be an integer from 0 to 100.
supported should be true only if the FICTURE image spatial pattern supports this hypothesis."""

USER_TEMPLATE_FICTURE_SOFT_CONSISTENCY = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the FICTURE candidate crop.

Image:
FICTURE reverse-blur crop. The candidate region is sharp and full color; outside the candidate is grayscale and blurred.

Task:
Give a soft consistency score for whether the FICTURE pattern is compatible with this one hypothesis.
This is not a final classifier and not a hard tissue label.

Important:
- FICTURE colors are cell-type / molecular-domain signals, not direct tissue labels.
- A tissue class can contain several colors, and one color can appear in several tissue classes.
- For bronchiola, alveoli, vessels, and immune infiltration, do not reject only because the local FICTURE colors are mixed.
- Reject only when the highlighted candidate has a coherent FICTURE pattern that clearly contradicts the hypothesis.
- Use this as a weak consistency check after Step 1, not as a hard morphology decision.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "ficture_support_score": 0,
  "supported": false
}}

ficture_support_score must be an integer from 0 to 100.
supported should be true for compatible or uncertain-but-not-contradictory FICTURE patterns."""


USER_TEMPLATE_FICTURE_TUMOR_STROMA_NARROW = """Step 1 used only FICTURE-derived RGB/cell-type composition and kept this candidate class as a possible hypothesis:
{target_hypothesis}

All Step 1 hypotheses:
{step1_hypothesis_json}

Now look only at the FICTURE candidate crop.

Image:
FICTURE reverse-blur crop. The candidate region is sharp and full color; outside the candidate is grayscale and blurred.

Task:
Use FICTURE only as a narrow consistency check.

Rules:
- For tumor and stroma, supported=true only if the highlighted candidate has a coherent FICTURE pattern compatible with that hypothesis.
- For bronchiola, alveoli, vessels, and immune_infiltration, FICTURE image alone is not reliable enough for hard rejection. Keep supported=true unless the FICTURE pattern clearly contradicts the hypothesis.
- Do not treat one color as a direct tissue label.
- Do not reject a structural class just because the candidate contains mixed colors.

Return exactly one JSON object:
{{
  "target_class": "{target_hypothesis}",
  "ficture_support_score": 0,
  "supported": false
}}

ficture_support_score must be an integer from 0 to 100."""


FICTURE_PROMPTS = {
    "strict_spatial": USER_TEMPLATE_FICTURE_HYPOTHESIS_VERIFY,
    "soft_consistency": USER_TEMPLATE_FICTURE_SOFT_CONSISTENCY,
    "tumor_stroma_narrow": USER_TEMPLATE_FICTURE_TUMOR_STROMA_NARROW,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_json(text: str) -> dict:
    candidates = re.findall(r"\{.*\}", text.strip(), flags=re.S)
    if not candidates:
        raise ValueError("No JSON object found")
    last_error: Exception | None = None
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_error = exc
            try:
                return ast.literal_eval(candidate)
            except Exception as exc2:
                last_error = exc2
    raise ValueError(f"JSON parse failed: {last_error}")


def clamp_int(value: object) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = 0
    return max(0, min(100, number))


def parse_response(text: str, target_class: str) -> dict[str, object]:
    data = extract_json(text)
    support_score = clamp_int(data.get("ficture_support_score", data.get("support_score", 0)))
    supported_raw = data.get("supported", False)
    if isinstance(supported_raw, str):
        supported = supported_raw.strip().lower() in {"true", "yes", "1", "supported"}
    else:
        supported = bool(supported_raw)
    return {
        "target_hypothesis_class": target_class,
        "ficture_support_score": support_score,
        "ficture_supported": str(supported).lower(),
    }


def step1_hypotheses_json(row: dict[str, str]) -> str:
    return json.dumps({key: row.get(key, "uncertain") for key in CLASS_KEYS}, ensure_ascii=False)


def step1_retained_hypotheses(row: dict[str, str]) -> list[str]:
    retained = []
    for key in CLASS_KEYS:
        status = str(row.get(key, "")).strip().lower()
        if status in {"plausible", "uncertain"}:
            retained.append(key)
    return retained


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_retention_summary(output_dir: Path, rows: list[dict[str, object]], failed: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("parse_status") == "ok"]
    summary = []
    for cls in CLASS_KEYS:
        cls_rows = [row for row in ok if row.get("true_class") == cls]
        after_step1 = sum(1 for row in cls_rows if row.get("true_retained_after_step1") == "true")
        after_ficture = sum(1 for row in cls_rows if row.get("true_retained_after_ficture") == "true")
        summary.append(
            {
                "true_class": cls,
                "n": len(cls_rows),
                "after_step1_true_retained": after_step1,
                "after_ficture_true_retained": after_ficture,
            }
        )
    write_csv(output_dir / "ficture_retention_by_true_class.csv", summary)
    write_csv(
        output_dir / "ficture_retention_run_gate.csv",
        [
            {
                "parsed_candidates": len(ok),
                "failed_candidates": len(failed),
                "n_candidates": len(rows) + len(failed),
                "parse_ok": "true" if not failed else "false",
            }
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-csv", type=Path, required=True)
    parser.add_argument("--step1-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-style", choices=sorted(FICTURE_PROMPTS), default="strict_spatial")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_rows = read_csv(args.requests_csv)
    step1_rows = {(row["row_index"], row["candidate_uid"]): row for row in read_csv(args.step1_csv)}
    joined = []
    for row in request_rows:
        key = (row["row_index"], row["candidate_uid"])
        if key not in step1_rows:
            raise KeyError(f"Missing Step 1 row for {key}")
        joined.append({**row, "_step1": step1_rows[key]})
    if args.limit:
        joined = joined[: args.limit]

    retention_path = args.output_dir / "ficture_candidate_retention_summary.csv"
    hypothesis_path = args.output_dir / "ficture_hypothesis_verification_scores.csv"
    failed_path = args.output_dir / "failed_rows.csv"

    candidate_rows: list[dict[str, object]] = []
    hypothesis_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    done: set[str] = set()
    if args.resume and retention_path.exists():
        candidate_rows = read_csv(retention_path)
        done = {str(row.get("row_index")) for row in candidate_rows}
    if args.resume and hypothesis_path.exists():
        hypothesis_rows = read_csv(hypothesis_path)

    vlm = load_vlm(args.model, args.device)
    for idx, row in enumerate(joined, start=1):
        if str(row["row_index"]) in done:
            continue
        uid = row["candidate_uid"]
        ficture_crop = args.image_root / row["ficture_crop_rel"]
        try:
            if not ficture_crop.exists():
                raise FileNotFoundError(f"missing FICTURE crop: {ficture_crop}")
            retained = step1_retained_hypotheses(row["_step1"])
            step1_map = step1_hypotheses_json(row["_step1"])
            prompt_template = FICTURE_PROMPTS[args.prompt_style]
            per_candidate = []
            for target_class in retained:
                prompt = prompt_template.format(
                    target_hypothesis=target_class,
                    step1_hypothesis_json=step1_map,
                )
                raw = generate(vlm, args.device, [open_rgb(ficture_crop)], SYSTEM_PROMPT, prompt, args.max_new_tokens)
                parsed = parse_response(raw, target_class)
                hyp_out = {
                    **{k: v for k, v in row.items() if k != "_step1"},
                    **parsed,
                    "parse_status": "ok",
                    "model": args.model,
                    "prompt_style": args.prompt_style,
                    "raw_response": raw,
                }
                hypothesis_rows.append(hyp_out)
                write_csv(hypothesis_path, hypothesis_rows)
                per_candidate.append(hyp_out)

            ficture_retained = [
                str(item["target_hypothesis_class"]) for item in per_candidate if item.get("ficture_supported") == "true"
            ]
            true_class = str(row.get("true_class", ""))
            true_hyp = next((item for item in per_candidate if item.get("target_hypothesis_class") == true_class), None)
            top_hyp = max(per_candidate, key=lambda item: int(item.get("ficture_support_score", 0) or 0)) if per_candidate else {}
            out = {
                **{k: v for k, v in row.items() if k != "_step1"},
                "step1_retained_classes": ";".join(retained),
                "step1_retained_count": len(retained),
                "true_retained_after_step1": str(true_class in retained).lower(),
                "ficture_retained_classes": ";".join(ficture_retained),
                "ficture_retained_count": len(ficture_retained),
                "true_retained_after_ficture": str(true_class in ficture_retained).lower(),
                "true_class_ficture_support_score": true_hyp.get("ficture_support_score", "") if true_hyp else "",
                "true_class_ficture_supported": true_hyp.get("ficture_supported", "false") if true_hyp else "false",
                "top_ficture_support_class": top_hyp.get("target_hypothesis_class", ""),
                "top_ficture_support_score": top_hyp.get("ficture_support_score", ""),
                "parse_status": "ok",
                "model": args.model,
                "prompt_style": args.prompt_style,
            }
            candidate_rows.append(out)
            write_csv(retention_path, candidate_rows)
            print(
                f"FICTURE verify {idx}/{len(joined)} {uid} true={true_class} "
                f"step1_retained={true_class in retained} ficture_retained={true_class in ficture_retained} ok",
                flush=True,
            )
        except Exception as exc:
            fail = {**{k: v for k, v in row.items() if k != "_step1"}, "error": str(exc)[:1000], "parse_status": "failed"}
            failed_rows.append(fail)
            write_csv(failed_path, failed_rows)
            print(f"FAILED FICTURE verify {idx}/{len(joined)} {uid}: {exc}", flush=True)

    write_csv(retention_path, candidate_rows)
    write_csv(hypothesis_path, hypothesis_rows)
    write_csv(failed_path, failed_rows)
    build_retention_summary(args.output_dir, candidate_rows, failed_rows)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "requests_csv": str(args.requests_csv),
                "step1_csv": str(args.step1_csv),
                "image_root": str(args.image_root),
                "limit": args.limit,
                "input": "FICTURE reverse-blur crop only",
                "prompt_style": args.prompt_style,
            },
            indent=2,
        )
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
