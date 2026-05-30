#!/usr/bin/env python3
"""Pairwise VLM ranker using answer-choice logits instead of generated choices."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from run_paired_vlm_hit_test import LABEL_ORDER, load_vlm, row_key, write_csv
from run_paired_vlm_pairwise_rank_test import (
    build_prompt,
    build_round_robin_pairs,
    retrieval_metrics,
    write_html,
)


def softmax2(a: float, b: float) -> Tuple[float, float]:
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    denom = ea + eb
    return ea / denom, eb / denom


def token_ids_for_choices(processor) -> Tuple[int, int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has no tokenizer; cannot compute A/B logits.")

    def one_token(texts: Sequence[str]) -> int:
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return int(ids[0])
        # Fall back to the last token of the first encoding; logged by caller as approximate.
        ids = tokenizer(texts[0], add_special_tokens=False).input_ids
        if not ids:
            raise RuntimeError(f"Could not tokenize choice text {texts[0]!r}")
        return int(ids[-1])

    return one_token(["A", " A", "\nA"]), one_token(["B", " B", "\nB"])


def choice_prob(vlm, device: str, images: List[Image.Image], system: str, prompt: str) -> Tuple[float, float, float, float]:
    import torch

    if vlm["kind"] != "image_text_to_text":
        raise RuntimeError(f"Logit scoring requires image_text_to_text model; got {vlm['kind']}")

    processor = vlm["processor"]
    model = vlm["model"]
    a_id, b_id = token_ids_for_choices(processor)
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt})
    message_variants = [
        [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ],
        [{"role": "user", "content": [{"type": "text", "text": system}] + user_content}],
    ]
    last_error: Exception | None = None
    for messages in message_variants:
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt")
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"VLM logit scoring failed: {last_error}")
    logits = outputs.logits[0, -1]
    a_logit = float(logits[a_id].detach().cpu())
    b_logit = float(logits[b_id].detach().cpu())
    p_a, p_b = softmax2(a_logit, b_logit)
    return p_a, p_b, a_logit, b_logit


def image_list(pool_dir: Path, row_a: dict, row_b: dict) -> List[Image.Image]:
    return [
        Image.open(pool_dir / row_a["he_crop_rel"]).convert("RGB"),
        Image.open(pool_dir / row_a["ficture_crop_rel"]).convert("RGB"),
        Image.open(pool_dir / row_b["he_crop_rel"]).convert("RGB"),
        Image.open(pool_dir / row_b["ficture_crop_rel"]).convert("RGB"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-labels", type=int, default=0)
    parser.add_argument("--calibrated", action="store_true", help="Score each pair in both orders and average.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    requests = list(csv.DictReader((args.pool_dir / "public_vlm_requests.csv").open()))
    hidden = {row_key(row): row for row in csv.DictReader((args.pool_dir / "hidden_candidate_truth.csv").open())}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vlm = load_vlm(args.model, args.device)
    stats: Dict[Tuple[str, str, str, str, str], dict] = {}
    rows_by_key: Dict[Tuple[str, str, str, str, str], dict] = {}
    for row in requests:
        key = row_key(row)
        rows_by_key[key] = row
        stats[key] = {"wins": 0.0, "losses": 0.0, "ties": 0.0, "comparisons": 0, "prob_sum": 0.0}

    comparison_rows: List[dict] = []
    label_order = LABEL_ORDER[: args.limit_labels] if args.limit_labels else LABEL_ORDER
    comparison_idx = 0
    for label, display in label_order:
        label_rows = [row for row in requests if row["label"] == label]
        label_rows = label_rows[:]
        rng.shuffle(label_rows)
        pairs = build_round_robin_pairs(label_rows, args.rounds)
        for row_left, row_right, round_idx in pairs:
            comparison_idx += 1
            # Randomize first order, but keep identities for calibrated preference.
            if rng.random() < 0.5:
                row_a, row_b = row_left, row_right
                order = "left_as_A"
            else:
                row_a, row_b = row_right, row_left
                order = "right_as_A"
            system, prompt = build_prompt(row_a, row_b, display, label)
            p_a, p_b, a_logit, b_logit = choice_prob(
                vlm, args.device, image_list(args.pool_dir, row_a, row_b), system, prompt
            )
            prob_a_identity = p_a
            prob_b_identity = p_b
            swap_p_b_for_a = ""
            swap_p_a_for_b = ""
            if args.calibrated:
                swap_system, swap_prompt = build_prompt(row_b, row_a, display, label)
                p_a_swapped, p_b_swapped, a_logit_swapped, b_logit_swapped = choice_prob(
                    vlm, args.device, image_list(args.pool_dir, row_b, row_a), swap_system, swap_prompt
                )
                # In swapped order, original A is now answer B and original B is answer A.
                prob_a_identity = 0.5 * (p_a + p_b_swapped)
                prob_b_identity = 0.5 * (p_b + p_a_swapped)
                swap_p_b_for_a = f"{p_b_swapped:.9f}"
                swap_p_a_for_b = f"{p_a_swapped:.9f}"
            key_a = row_key(row_a)
            key_b = row_key(row_b)
            stats[key_a]["prob_sum"] += prob_a_identity
            stats[key_b]["prob_sum"] += prob_b_identity
            stats[key_a]["comparisons"] += 1
            stats[key_b]["comparisons"] += 1
            if abs(prob_a_identity - prob_b_identity) < 1e-6:
                stats[key_a]["ties"] += 1
                stats[key_b]["ties"] += 1
                winner = ""
            elif prob_a_identity > prob_b_identity:
                stats[key_a]["wins"] += 1
                stats[key_b]["losses"] += 1
                winner = "A"
            else:
                stats[key_b]["wins"] += 1
                stats[key_a]["losses"] += 1
                winner = "B"
            comparison_rows.append(
                {
                    "comparison_idx": comparison_idx,
                    "round": round_idx,
                    "label": label,
                    "display": display,
                    "order": order,
                    "candidate_a": f"{row_a['source']}/{row_a['setting']}/{row_a['candidate_id']}",
                    "candidate_b": f"{row_b['source']}/{row_b['setting']}/{row_b['candidate_id']}",
                    "p_a": f"{p_a:.9f}",
                    "p_b": f"{p_b:.9f}",
                    "p_a_identity": f"{prob_a_identity:.9f}",
                    "p_b_identity": f"{prob_b_identity:.9f}",
                    "swap_p_b_for_a": swap_p_b_for_a,
                    "swap_p_a_for_b": swap_p_a_for_b,
                    "a_logit": f"{a_logit:.9f}",
                    "b_logit": f"{b_logit:.9f}",
                    "winner": winner,
                }
            )
            print(
                f"Logprob pair {comparison_idx} {display} pA={prob_a_identity:.3f} pB={prob_b_identity:.3f} winner={winner or 'tie'}",
                flush=True,
            )

    score_rows: List[dict] = []
    for key, row in rows_by_key.items():
        truth = hidden[key]
        stat = stats[key]
        comparisons = int(stat["comparisons"])
        mean_prob = stat["prob_sum"] / comparisons if comparisons else 0.0
        win_rate = (stat["wins"] + 0.5 * stat["ties"]) / comparisons if comparisons else 0.0
        out = dict(row)
        out.update(
            {
                "model": args.model,
                "wins": f"{stat['wins']:.3f}",
                "losses": f"{stat['losses']:.3f}",
                "ties": f"{stat['ties']:.3f}",
                "comparisons": comparisons,
                "pairwise_win_rate": f"{win_rate:.9f}",
                "pairwise_mean_choice_prob": f"{mean_prob:.9f}",
                "hidden_dice": truth["hidden_dice"],
                "hidden_precision": truth["hidden_precision"],
                "hidden_recall": truth["hidden_recall"],
                "hidden_iou": truth["hidden_iou"],
            }
        )
        score_rows.append(out)

    write_csv(args.output_dir / "pairwise_logprob_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "vlm_pairwise_candidate_scores.csv", score_rows)
    metrics_rows = retrieval_metrics(score_rows, "pairwise_mean_choice_prob", f"Pairwise logprob VLM retrieval: {args.model}")
    write_csv(args.output_dir / "retrieval_comparison_metrics.csv", metrics_rows)
    write_html(args.output_dir, args.model, score_rows, metrics_rows)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
