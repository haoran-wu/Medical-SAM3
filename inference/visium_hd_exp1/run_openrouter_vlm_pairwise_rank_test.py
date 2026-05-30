#!/usr/bin/env python3
"""Run pairwise paired H&E/FICTURE VLM direct retrieval through OpenRouter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from run_openrouter_vlm_hit_test import image_data_url, openrouter_chat, write_csv
from run_paired_vlm_pairwise_rank_test import (
    LABEL_ORDER,
    build_round_robin_pairs,
    retrieval_metrics,
    row_key,
    write_html,
)


def build_pairwise_prompt(row_a: dict, row_b: dict, label_display: str, label_slug: str) -> str:
    description = row_a.get("target_description") or row_b.get("target_description") or ""
    image_cue = row_a.get("image_cue") or (
        "Each candidate has two aligned crops: one H&E histology crop and one official FICTURE factor-color crop."
    )
    return f"""Target tissue class: {label_display} ({label_slug})
Meaning of the target class: {description}

You are comparing two candidate masks from the same candidate pool.
{image_cue}

Image order:
1. Candidate A H&E crop
2. Candidate A FICTURE crop
3. Candidate B H&E crop
4. Candidate B FICTURE crop

Use only the images and target-class description. The GOOD/MID/BAD labels are hidden from you.
Choose the candidate whose visible region is more likely to be the target tissue class.
You must choose A or B. Do not output a tie.
Your entire answer must be exactly one letter: A or B."""


def parse_choice(text: str) -> Tuple[str, str]:
    stripped = text.strip()
    if re.fullmatch(r"[Aa]", stripped):
        return "A", "ok"
    if re.fullmatch(r"[Bb]", stripped):
        return "B", "ok"
    match = re.search(r"\b([AaBb])\b", stripped)
    if match:
        return match.group(1).upper(), "fallback_letter"
    try:
        data = json.loads(stripped)
        value = str(data.get("choice", "")).strip().upper()
        if value in {"A", "B"}:
            return value, "json_choice"
    except Exception:
        pass
    return "", "no_choice_found"


def openrouter_pairwise_chat(
    api_key: str,
    model: str,
    image_urls: Sequence[str],
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> Tuple[str, dict]:
    import time
    import urllib.error
    import urllib.request

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful pathology image judge. Your task is pairwise direct retrieval: "
                    "choose which candidate better matches the requested target tissue class. "
                    "Return exactly A or B."
                ),
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/Medical-SAM3",
        "X-Title": "Medical-SAM3 pairwise VLM candidate retrieval",
    }
    data = json.dumps(body).encode("utf-8")
    url = "https://openrouter.ai/api/v1/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
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
        except urllib.error.HTTPError as exc:
            last_error = exc
            body_text = exc.read().decode("utf-8", errors="replace")[:1200]
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body_text}") from exc
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--limit-labels", type=int, default=0)
    parser.add_argument("--calibrated", action="store_true", help="Ask each A/B pair again with the order swapped.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing {args.api_key_env}")

    rng = random.Random(args.seed)
    requests = list(csv.DictReader((args.pool_dir / "public_vlm_requests.csv").open()))
    hidden = {row_key(row): row for row in csv.DictReader((args.pool_dir / "hidden_candidate_truth.csv").open())}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[Tuple[str, str, str, str, str], dict] = {}
    rows_by_key: Dict[Tuple[str, str, str, str, str], dict] = {}
    for row in requests:
        key = row_key(row)
        rows_by_key[key] = row
        stats[key] = {"wins": 0.0, "losses": 0.0, "ties": 0.0, "comparisons": 0, "support_sum": 0.0}

    comp_path = args.output_dir / "pairwise_comparisons.csv"
    comparison_rows: List[dict] = []
    existing_keys = set()
    if args.resume and comp_path.exists():
        comparison_rows = list(csv.DictReader(comp_path.open()))
        for row in comparison_rows:
            existing_keys.add((row["label"], row["round"], row["candidate_a"], row["candidate_b"], row["order"]))
            choice = row.get("choice", "")
            a_key = tuple(row[k] for k in ["a_label", "a_source", "a_run", "a_setting", "a_candidate_id"])
            b_key = tuple(row[k] for k in ["b_label", "b_source", "b_run", "b_setting", "b_candidate_id"])
            if row.get("a_support") not in {None, ""} and row.get("b_support") not in {None, ""}:
                a_support = float(row["a_support"])
                b_support = float(row["b_support"])
                stats[a_key]["support_sum"] += a_support
                stats[b_key]["support_sum"] += b_support
                if abs(a_support - b_support) < 1e-9:
                    stats[a_key]["ties"] += 1
                    stats[b_key]["ties"] += 1
                elif a_support > b_support:
                    stats[a_key]["wins"] += 1
                    stats[b_key]["losses"] += 1
                else:
                    stats[b_key]["wins"] += 1
                    stats[a_key]["losses"] += 1
            elif choice == "A":
                stats[a_key]["wins"] += 1
                stats[b_key]["losses"] += 1
                stats[a_key]["support_sum"] += 1.0
            elif choice == "B":
                stats[b_key]["wins"] += 1
                stats[a_key]["losses"] += 1
                stats[b_key]["support_sum"] += 1.0
            else:
                stats[a_key]["ties"] += 1
                stats[b_key]["ties"] += 1
                stats[a_key]["support_sum"] += 0.5
                stats[b_key]["support_sum"] += 0.5
            stats[a_key]["comparisons"] += 1
            stats[b_key]["comparisons"] += 1

    label_order = LABEL_ORDER[: args.limit_labels] if args.limit_labels else LABEL_ORDER
    comparison_idx = len(comparison_rows)
    for label, display in label_order:
        label_rows = [row for row in requests if row["label"] == label]
        label_rows = label_rows[:]
        rng.shuffle(label_rows)
        pairs = build_round_robin_pairs(label_rows, args.rounds)
        for row_a0, row_b0, round_idx in pairs:
            if rng.random() < 0.5:
                row_a, row_b = row_a0, row_b0
                order = "original"
            else:
                row_a, row_b = row_b0, row_a0
                order = "swapped"
            comp_key = (
                label,
                str(round_idx),
                f"{row_a['source']}/{row_a['setting']}/{row_a['candidate_id']}",
                f"{row_b['source']}/{row_b['setting']}/{row_b['candidate_id']}",
                order,
            )
            if comp_key in existing_keys:
                continue
            image_urls = [
                image_data_url(args.pool_dir / row_a["he_crop_rel"], args.max_side, args.jpeg_quality),
                image_data_url(args.pool_dir / row_a["ficture_crop_rel"], args.max_side, args.jpeg_quality),
                image_data_url(args.pool_dir / row_b["he_crop_rel"], args.max_side, args.jpeg_quality),
                image_data_url(args.pool_dir / row_b["ficture_crop_rel"], args.max_side, args.jpeg_quality),
            ]
            prompt = build_pairwise_prompt(row_a, row_b, display, label)
            raw, payload = openrouter_pairwise_chat(
                api_key=api_key,
                model=args.model,
                image_urls=image_urls,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
            )
            choice, parse_status = parse_choice(raw)
            swap_choice = ""
            swap_parse_status = ""
            swap_raw = ""
            swap_usage = {}
            a_key = row_key(row_a)
            b_key = row_key(row_b)
            a_support = 0.0
            b_support = 0.0
            if choice == "A":
                a_support += 0.5 if args.calibrated else 1.0
            elif choice == "B":
                b_support += 0.5 if args.calibrated else 1.0
            else:
                a_support += 0.25 if args.calibrated else 0.5
                b_support += 0.25 if args.calibrated else 0.5

            if args.calibrated:
                swap_image_urls = [
                    image_data_url(args.pool_dir / row_b["he_crop_rel"], args.max_side, args.jpeg_quality),
                    image_data_url(args.pool_dir / row_b["ficture_crop_rel"], args.max_side, args.jpeg_quality),
                    image_data_url(args.pool_dir / row_a["he_crop_rel"], args.max_side, args.jpeg_quality),
                    image_data_url(args.pool_dir / row_a["ficture_crop_rel"], args.max_side, args.jpeg_quality),
                ]
                swap_prompt = build_pairwise_prompt(row_b, row_a, display, label)
                swap_raw, swap_payload = openrouter_pairwise_chat(
                    api_key=api_key,
                    model=args.model,
                    image_urls=swap_image_urls,
                    prompt=swap_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                swap_choice, swap_parse_status = parse_choice(swap_raw)
                swap_usage = swap_payload.get("usage") or {}
                # In swapped order, original A is answer B and original B is answer A.
                if swap_choice == "B":
                    a_support += 0.5
                elif swap_choice == "A":
                    b_support += 0.5
                else:
                    a_support += 0.25
                    b_support += 0.25

            stats[a_key]["support_sum"] += a_support
            stats[b_key]["support_sum"] += b_support
            if abs(a_support - b_support) < 1e-9:
                stats[a_key]["ties"] += 1
                stats[b_key]["ties"] += 1
                winner = ""
            elif a_support > b_support:
                stats[a_key]["wins"] += 1
                stats[b_key]["losses"] += 1
                winner = "A"
            else:
                stats[b_key]["wins"] += 1
                stats[a_key]["losses"] += 1
                winner = "B"
            stats[a_key]["comparisons"] += 1
            stats[b_key]["comparisons"] += 1
            comparison_idx += 1
            usage = payload.get("usage") or {}
            comparison_rows.append(
                {
                    "comparison_idx": comparison_idx,
                    "round": round_idx,
                    "label": label,
                    "display": display,
                    "order": order,
                    "a_label": a_key[0],
                    "a_source": a_key[1],
                    "a_run": a_key[2],
                    "a_setting": a_key[3],
                    "a_candidate_id": a_key[4],
                    "b_label": b_key[0],
                    "b_source": b_key[1],
                    "b_run": b_key[2],
                    "b_setting": b_key[3],
                    "b_candidate_id": b_key[4],
                    "candidate_a": f"{row_a['source']}/{row_a['setting']}/{row_a['candidate_id']}",
                    "candidate_b": f"{row_b['source']}/{row_b['setting']}/{row_b['candidate_id']}",
                    "choice": choice,
                    "swap_choice": swap_choice,
                    "winner": winner,
                    "parse_status": parse_status,
                    "swap_parse_status": swap_parse_status,
                    "a_support": f"{a_support:.6f}",
                    "b_support": f"{b_support:.6f}",
                    "raw_response": raw,
                    "swap_raw_response": swap_raw,
                    "prompt_tokens": usage.get("prompt_tokens", ""),
                    "completion_tokens": usage.get("completion_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                    "swap_prompt_tokens": swap_usage.get("prompt_tokens", ""),
                    "swap_completion_tokens": swap_usage.get("completion_tokens", ""),
                    "swap_total_tokens": swap_usage.get("total_tokens", ""),
                }
            )
            write_csv(comp_path, comparison_rows, list(comparison_rows[0].keys()))
            print(
                f"Pairwise {comparison_idx} {display} round={round_idx} "
                f"choice={choice or 'NA'} swap={swap_choice or 'NA'} "
                f"supportA={a_support:.2f} supportB={b_support:.2f}",
                flush=True,
            )

    score_rows: List[dict] = []
    for key, row in rows_by_key.items():
        truth = hidden[key]
        stat = stats[key]
        comparisons = int(stat["comparisons"])
        win_rate = (stat["wins"] + 0.5 * stat["ties"]) / comparisons if comparisons else 0.0
        mean_support = stat["support_sum"] / comparisons if comparisons else 0.0
        out = dict(row)
        out.update(
            {
                "model": args.model,
                "provider": "openrouter",
                "wins": f"{stat['wins']:.3f}",
                "losses": f"{stat['losses']:.3f}",
                "ties": f"{stat['ties']:.3f}",
                "comparisons": comparisons,
                "pairwise_win_rate": f"{win_rate:.9f}",
                "pairwise_mean_choice_prob": f"{mean_support:.9f}",
                "hidden_dice": truth["hidden_dice"],
                "hidden_precision": truth["hidden_precision"],
                "hidden_recall": truth["hidden_recall"],
                "hidden_iou": truth["hidden_iou"],
            }
        )
        score_rows.append(out)

    write_csv(args.output_dir / "vlm_pairwise_candidate_scores.csv", score_rows)
    score_key = "pairwise_mean_choice_prob" if args.calibrated else "pairwise_win_rate"
    metrics_rows = retrieval_metrics(score_rows, score_key, f"OpenRouter pairwise VLM retrieval: {args.model}")
    write_csv(args.output_dir / "retrieval_comparison_metrics.csv", metrics_rows)
    write_html(args.output_dir, args.model, score_rows, metrics_rows)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
