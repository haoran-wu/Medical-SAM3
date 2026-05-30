#!/usr/bin/env python3
"""Pairwise VLM direct-retrieval test for paired H&E/FICTURE candidate crops."""

from __future__ import annotations

import argparse
import csv
import html
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from run_paired_vlm_hit_test import LABEL_ORDER, load_vlm, generate, row_key, write_csv


def build_round_robin_pairs(rows: Sequence[dict], rounds: int) -> List[Tuple[dict, dict, int]]:
    candidates: List[dict | None] = list(rows)
    if len(candidates) % 2:
        candidates.append(None)
    n = len(candidates)
    pairs: List[Tuple[dict, dict, int]] = []
    schedule = candidates[:]
    for round_idx in range(min(rounds, n - 1)):
        for i in range(n // 2):
            a = schedule[i]
            b = schedule[n - 1 - i]
            if a is not None and b is not None:
                pairs.append((a, b, round_idx + 1))
        schedule = [schedule[0]] + [schedule[-1]] + schedule[1:-1]
    return pairs


def build_prompt(row_a: dict, row_b: dict, label_display: str, label_slug: str) -> Tuple[str, str]:
    system = (
        "You are a careful pathology image judge. Your task is pairwise direct retrieval: "
        "choose which candidate better matches the requested target tissue class."
    )
    description = row_a.get("target_description") or row_b.get("target_description") or ""
    image_cue = row_a.get("image_cue") or (
        "Each candidate has two aligned crops: one H&E histology crop and one official FICTURE factor-color crop."
    )
    prompt = f"""
Target tissue class: {label_display} ({label_slug})
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
Your entire answer must be exactly one letter: A or B.
"""
    return system, prompt.strip()


def parse_choice(text: str) -> Tuple[str, str]:
    stripped = text.strip()
    if re.fullmatch(r"[Aa]", stripped):
        return "A", "ok"
    if re.fullmatch(r"[Bb]", stripped):
        return "B", "ok"
    match = re.search(r"\b([AaBb])\b", stripped)
    if match:
        return match.group(1).upper(), "fallback_letter"
    return "", "no_choice_found"


def score_warning(scores: List[float]) -> Tuple[int, int, str]:
    rounded = [round(score, 6) for score in scores]
    unique_count = len(set(rounded))
    top_score = max(rounded) if rounded else 0.0
    top1_tie_size = sum(1 for score in rounded if score == top_score)
    score_range = (max(scores) - min(scores)) if scores else 0.0
    if unique_count <= 1:
        return unique_count, top1_tie_size, "uninformative tie: all scores are identical"
    if score_range < 0.01:
        return unique_count, top1_tie_size, "near-constant scores: differences are numerical noise"
    if unique_count <= 2 or top1_tie_size >= 5:
        return unique_count, top1_tie_size, "coarse/tied scores: ranking may be unstable"
    return unique_count, top1_tie_size, ""


def retrieval_metrics(rows: List[dict], score_key: str, method: str) -> List[dict]:
    out: List[dict] = []
    for label, display in LABEL_ORDER:
        label_rows = [row for row in rows if row["label"] == label]
        ranked = sorted(label_rows, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)
        if not ranked:
            continue
        top1 = ranked[0]
        top5 = ranked[:5]
        good_ranks = [i + 1 for i, row in enumerate(ranked) if row["sample_bucket"] == "GOOD"]
        score_unique_count, top1_tie_size, warning = score_warning(
            [float(row.get(score_key) or 0.0) for row in ranked]
        )
        out.append(
            {
                "method": method,
                "label": label,
                "display": display,
                "n_candidates": len(ranked),
                "top1_is_good": "yes" if top1["sample_bucket"] == "GOOD" else "no",
                "top1_bucket": top1["sample_bucket"],
                "top1_source": top1["source"],
                "top1_hidden_dice": top1["hidden_dice"],
                "top1_hidden_precision": top1["hidden_precision"],
                "top1_hidden_recall": top1["hidden_recall"],
                "top1_score": f"{float(top1.get(score_key) or 0.0):.6f}",
                "top1_candidate": f"{top1['source']}/{top1['setting']}/{top1['candidate_id']}",
                "good_count_top5": sum(1 for row in top5 if row["sample_bucket"] == "GOOD"),
                "mean_hidden_dice_top5": f"{sum(float(row['hidden_dice']) for row in top5) / max(len(top5), 1):.6f}",
                "best_good_rank": min(good_ranks) if good_ranks else "",
                "score_unique_count": score_unique_count,
                "top1_tie_size": top1_tie_size,
                "tie_warning": warning,
            }
        )
    return out


def write_html(output_dir: Path, model_name: str, rows: List[dict], metrics_rows: List[dict]) -> None:
    score_key = "pairwise_mean_choice_prob" if rows and "pairwise_mean_choice_prob" in rows[0] else "pairwise_win_rate"
    score_label = "mean A/B probability" if score_key == "pairwise_mean_choice_prob" else "win rate"
    metric_table = [
        "<table><thead><tr><th>class</th><th>top1</th><th>top1 source</th><th>top1 Dice</th>"
        "<th>GOOD in top5</th><th>best GOOD rank</th><th>score types</th><th>tie warning</th></tr></thead><tbody>"
    ]
    for row in metrics_rows:
        metric_table.append(
            "<tr>"
            f"<td>{html.escape(row['display'])}</td>"
            f"<td>{html.escape(row['top1_bucket'])}</td>"
            f"<td>{html.escape(row.get('top1_source') or '')}</td>"
            f"<td>{float(row['top1_hidden_dice']):.3f}</td>"
            f"<td>{html.escape(str(row['good_count_top5']))} / 5</td>"
            f"<td>{html.escape(str(row['best_good_rank']))}</td>"
            f"<td>{html.escape(str(row['score_unique_count']))}</td>"
            f"<td>{html.escape(row.get('tie_warning') or '')}</td>"
            "</tr>"
        )
    metric_table.append("</tbody></table>")

    cards: List[str] = []
    for label, display in LABEL_ORDER:
        cards.append(f"<h2>{html.escape(display)}</h2><div class='grid'>")
        label_rows = sorted(
            [row for row in rows if row["label"] == label],
            key=lambda r: float(r.get(score_key) or 0.0),
            reverse=True,
        )
        for row in label_rows:
            cards.append(
                "<article>"
                f"<h3>{html.escape(row['sample_bucket'])} | {score_label} {float(row.get(score_key) or 0.0):.3f} | "
                f"Dice {float(row['hidden_dice']):.3f}</h3>"
                "<div class='imgs'>"
                f"<img src=\"{html.escape(row['he_crop_rel'])}\">"
                f"<img src=\"{html.escape(row['ficture_crop_rel'])}\">"
                "</div>"
                f"<p>{html.escape(row['source'])} / {html.escape(row['setting'])} / {html.escape(str(row['candidate_id']))}</p>"
                f"<p>wins/losses/ties/comparisons: {row['wins']} / {row['losses']} / {row['ties']} / {row['comparisons']}</p>"
                "</article>"
            )
        cards.append("</div>")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Pairwise VLM direct retrieval</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:28px; color:#20242a; }}
.note {{ max-width:1050px; line-height:1.55; }}
table {{ border-collapse:collapse; margin:18px 0; width:100%; }}
th,td {{ border-bottom:1px solid #ddd; padding:7px 10px; text-align:left; vertical-align:top; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
article {{ border:1px solid #ddd; border-radius:8px; padding:12px; background:#fff; }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
img {{ width:100%; border:1px solid #eee; }}
</style>
</head>
<body>
<h1>Pairwise VLM Direct Retrieval</h1>
<div class="note">
<p><b>模型：</b>{html.escape(model_name)}</p>
<p><b>实验问题：</b>不是让 VLM 自己给绝对分，而是在同一类别内每次比较两个 candidate，选择更像目标组织的那一个。</p>
<p><b>输入：</b>每个候选有两张图，H&E crop 和 FICTURE crop。GOOD/MID/BAD 与 Dice 对模型隐藏，只用于最后评价。</p>
<p><b>输出：</b>每次比较只允许回答 A 或 B。生成式版本用胜率作为检索分数；logit/probability 版本用模型下一步选择 A/B 的概率作为连续检索分数。</p>
</div>
{''.join(metric_table)}
{''.join(cards)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-labels", type=int, default=0)
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
        stats[key] = {"wins": 0.0, "losses": 0.0, "ties": 0.0, "comparisons": 0}

    comparison_rows: List[dict] = []
    label_order = LABEL_ORDER[: args.limit_labels] if args.limit_labels else LABEL_ORDER
    comparison_idx = 0
    for label, display in label_order:
        label_rows = [row for row in requests if row["label"] == label]
        label_rows = label_rows[:]
        rng.shuffle(label_rows)
        pairs = build_round_robin_pairs(label_rows, args.rounds)
        for row_a0, row_b0, round_idx in pairs:
            comparison_idx += 1
            if rng.random() < 0.5:
                row_a, row_b = row_a0, row_b0
                order = "original"
            else:
                row_a, row_b = row_b0, row_a0
                order = "swapped"
            images = [
                Image.open(args.pool_dir / row_a["he_crop_rel"]).convert("RGB"),
                Image.open(args.pool_dir / row_a["ficture_crop_rel"]).convert("RGB"),
                Image.open(args.pool_dir / row_b["he_crop_rel"]).convert("RGB"),
                Image.open(args.pool_dir / row_b["ficture_crop_rel"]).convert("RGB"),
            ]
            system, prompt = build_prompt(row_a, row_b, display, label)
            raw = generate(vlm, args.device, images, system, prompt, args.max_new_tokens)
            choice, parse_status = parse_choice(raw)

            key_a = row_key(row_a)
            key_b = row_key(row_b)
            if choice == "A":
                winner_key, loser_key = key_a, key_b
                stats[winner_key]["wins"] += 1.0
                stats[loser_key]["losses"] += 1.0
            elif choice == "B":
                winner_key, loser_key = key_b, key_a
                stats[winner_key]["wins"] += 1.0
                stats[loser_key]["losses"] += 1.0
            else:
                winner_key = loser_key = None
                stats[key_a]["ties"] += 1.0
                stats[key_b]["ties"] += 1.0
            stats[key_a]["comparisons"] += 1
            stats[key_b]["comparisons"] += 1
            comparison_rows.append(
                {
                    "comparison_idx": comparison_idx,
                    "round": round_idx,
                    "label": label,
                    "display": display,
                    "order": order,
                    "candidate_a": f"{row_a['source']}/{row_a['setting']}/{row_a['candidate_id']}",
                    "candidate_b": f"{row_b['source']}/{row_b['setting']}/{row_b['candidate_id']}",
                    "choice": choice,
                    "parse_status": parse_status,
                    "winner": "" if winner_key is None else "/".join(map(str, winner_key[1:])),
                    "raw_response": raw,
                }
            )
            print(
                f"Pairwise {comparison_idx} {display} round={round_idx} choice={choice or 'NA'} raw={raw[:80]!r}",
                flush=True,
            )

    score_rows: List[dict] = []
    for key, row in rows_by_key.items():
        truth = hidden[key]
        stat = stats[key]
        comparisons = int(stat["comparisons"])
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
                "hidden_dice": truth["hidden_dice"],
                "hidden_precision": truth["hidden_precision"],
                "hidden_recall": truth["hidden_recall"],
                "hidden_iou": truth["hidden_iou"],
            }
        )
        score_rows.append(out)

    write_csv(args.output_dir / "pairwise_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "vlm_pairwise_candidate_scores.csv", score_rows)
    metrics_rows = retrieval_metrics(score_rows, "pairwise_win_rate", f"Pairwise VLM retrieval: {args.model}")
    write_csv(args.output_dir / "retrieval_comparison_metrics.csv", metrics_rows)
    write_html(args.output_dir, args.model, score_rows, metrics_rows)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
