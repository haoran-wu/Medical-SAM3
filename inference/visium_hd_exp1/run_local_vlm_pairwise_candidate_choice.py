#!/usr/bin/env python3
"""Pairwise local-VLM reranking for one target class candidate pool.

This is a second-stage ranker: instead of asking for an independent score for
each mask, it shows two candidate masks at a time and asks which better matches
the target tissue class. Each pair is evaluated in both A/B orders to reduce
position bias.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Iterable

from PIL import Image

from run_paired_vlm_hit_test import generate, load_vlm, write_csv


SYSTEM_PROMPT = (
    "You are a careful pathology image reranker. Compare two candidate masks for "
    "one target tissue class. Return only valid JSON."
)


def read_rows(pool_csv: Path) -> list[dict]:
    with pool_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_candidate_id(rows: Iterable[dict]) -> dict[str, dict]:
    return {str(row["candidate_id"]): row for row in rows}


def parse_candidate_ids(text: str) -> list[str]:
    ids = [part.strip() for part in text.split(",") if part.strip()]
    if not ids:
        raise ValueError("--candidate-ids did not contain any ids")
    return ids


def top_ids_from_score_csv(path: Path, score_key: str, n: int) -> list[str]:
    rows = list(csv.DictReader(path.open(newline="")))
    rows.sort(key=lambda row: float(row.get(score_key) or 0.0), reverse=True)
    ids: list[str] = []
    for row in rows:
        cid = str(row["candidate_id"])
        if cid not in ids:
            ids.append(cid)
        if len(ids) >= n:
            break
    return ids


def image_rels(row: dict, mode: str) -> list[str]:
    if mode == "he_context":
        return [row["image1_rel"], row["image3_rel"]]
    if mode == "he_ficture_context":
        return [row["image1_rel"], row["image2_rel"], row["image3_rel"], row["image4_rel"]]
    if mode == "he_closeup":
        return [row["image1_rel"]]
    raise ValueError(f"Unsupported image mode: {mode}")


def load_images(pool_dir: Path, row: dict, mode: str) -> list[Image.Image]:
    return [Image.open(pool_dir / rel).convert("RGB") for rel in image_rels(row, mode)]


def build_prompt(
    row_a: dict,
    row_b: dict,
    target_display: str,
    target_description: str,
    mode: str,
    decision_profile: str,
    summary_a: dict | None = None,
    summary_b: dict | None = None,
) -> str:
    if mode == "he_context":
        image_list = """Images are ordered as:
1. Candidate A H&E close-up
2. Candidate A H&E local context
3. Candidate B H&E close-up
4. Candidate B H&E local context"""
    elif mode == "he_ficture_context":
        image_list = """Images are ordered as:
1. Candidate A H&E close-up
2. Candidate A FICTURE close-up
3. Candidate A H&E local context
4. Candidate A FICTURE local context
5. Candidate B H&E close-up
6. Candidate B FICTURE close-up
7. Candidate B H&E local context
8. Candidate B FICTURE local context"""
    else:
        image_list = """Images are ordered as:
1. Candidate A H&E close-up
2. Candidate B H&E close-up"""

    profile_notes = ""
    if decision_profile == "alveoli_broad":
        profile_notes = """
Alveoli-specific rule:
- A strong alveoli mask should cover a broad patch of lung parenchyma with many open air spaces and thin septa.
- A thin, branching, ribbon-like, or tubular wall fragment is usually not the best alveoli mask, even if it borders empty space.
- If one candidate is a broad sponge-like parenchymal patch and the other is a thin wall fragment, choose the broad parenchymal patch.
- Do not reward a candidate just for being near a lumen; reward it for covering alveolar parenchymal tissue.
"""
    elif decision_profile == "alveoli_ficture_text":
        profile_notes = """
Alveoli-specific rule:
- H&E morphology is the primary evidence.
- A strong alveoli mask should cover a broad patch of lung parenchyma with many open air spaces and thin septa.
- A thin, branching, ribbon-like, or tubular wall fragment is usually not the best alveoli mask, even if it borders empty space.
- Use the FICTURE text as supporting evidence, not as a direct label map.
- High airway epithelial signal supports airway/bronchiola rather than alveoli.
- High immune/macrophage/plasma-cell signal argues against alveoli.
- Epithelial/tumor-like FICTURE signal is non-specific in this lung cancer tissue and should not override H&E alveolar morphology.
"""
    elif decision_profile == "alveoli_ficture_reject_only":
        profile_notes = """
Alveoli-specific rule:
- H&E morphology is the primary evidence for choosing the better alveoli mask.
- A strong alveoli mask should cover broad lung parenchyma with many open air spaces and thin septa.
- A thin, branching, ribbon-like, or tubular wall fragment is usually not the best alveoli mask, even if it borders empty space.
- Use FICTURE text only as a rejection filter for obvious false positives.
- Strong immune/macrophage/plasma-cell signal inside the candidate argues against alveoli.
- Strong airway epithelial signal inside the candidate argues for airway/bronchiola, not alveoli.
- Do not use higher AT2/alveolar percentage alone to choose between two H&E-plausible alveoli masks; among plausible masks, choose by H&E morphology and coverage.
- Epithelial/tumor-like FICTURE signal is common and non-specific in this lung cancer tissue; do not use it alone to call tumor or reject alveoli.
"""

    summary_block = ""
    if summary_a is not None and summary_b is not None:
        summary_block = f"""
Structured FICTURE evidence for Candidate A:
- Inside candidate mask groups: {summary_a.get('inside_group_summary', 'not available')}
- Inside candidate top factors: {summary_a.get('inside_top_factors', 'not available')}
- Local context groups: {summary_a.get('local_group_summary', 'not available')}
- Local context top factors: {summary_a.get('local_top_factors', 'not available')}

Structured FICTURE evidence for Candidate B:
- Inside candidate mask groups: {summary_b.get('inside_group_summary', 'not available')}
- Inside candidate top factors: {summary_b.get('inside_top_factors', 'not available')}
- Local context groups: {summary_b.get('local_group_summary', 'not available')}
- Local context top factors: {summary_b.get('local_top_factors', 'not available')}
"""

    return f"""Target tissue class: {target_display}
Meaning: {target_description}

{image_list}

Question: which candidate is the better mask for the target tissue class?

Important decision rules:
- Judge the candidate mask, not the whole image.
- Use the local context to decide whether a small fragment belongs to the target structure.
- Prefer normal alveolar parenchyma only when the context shows open air spaces with thin septa.
- Do not choose an isolated airway/vessel wall fragment just because it borders open space.
- If both are poor, still choose the less wrong candidate.
{profile_notes}
{summary_block}

Candidate A id: {row_a['candidate_id']}
Candidate B id: {row_b['candidate_id']}

Return exactly one JSON object:
{{"winner": "A", "confidence": 0-100}}
or
{{"winner": "B", "confidence": 0-100}}
"""


def parse_winner(raw: str) -> tuple[str, float, str]:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        text = raw.strip().upper()
        if "A" in text and "B" not in text:
            return "A", 0.0, "fallback_letter"
        if "B" in text and "A" not in text:
            return "B", 0.0, "fallback_letter"
        return "", 0.0, "no_json"
    try:
        data = json.loads(match.group(0))
    except Exception:
        return "", 0.0, "bad_json"
    winner = str(data.get("winner", "")).strip().upper()
    if winner not in {"A", "B"}:
        return "", 0.0, "bad_winner"
    try:
        conf = float(data.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return winner, max(0.0, min(100.0, conf)), "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-display", default="alveoli")
    parser.add_argument("--target-description", default="alveolar lung parenchyma, open air spaces, thin septa")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--score-csv", type=Path)
    parser.add_argument("--score-key", default="alveoli")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--image-mode", choices=["he_context", "he_ficture_context", "he_closeup"], default="he_context")
    parser.add_argument(
        "--decision-profile",
        choices=["standard", "alveoli_broad", "alveoli_ficture_text", "alveoli_ficture_reject_only"],
        default="standard",
    )
    parser.add_argument("--summary-csv", type=Path, help="Optional candidate-level structured FICTURE text summaries.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    rows = read_rows(args.pool_csv)
    row_map = by_candidate_id(rows)
    if args.candidate_ids:
        ids = parse_candidate_ids(args.candidate_ids)
    elif args.score_csv:
        ids = top_ids_from_score_csv(args.score_csv, args.score_key, args.top_k)
    else:
        raise SystemExit("Provide --candidate-ids or --score-csv")
    missing = [cid for cid in ids if cid not in row_map]
    if missing:
        raise SystemExit(f"Candidate ids not in pool: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = args.pool_csv.parent
    selected = [row_map[cid] for cid in ids]
    summaries: dict[str, dict] = {}
    if args.summary_csv:
        summaries = {
            str(row["candidate_id"]): row
            for row in csv.DictReader(args.summary_csv.open(newline=""))
        }
    (args.output_dir / "pairwise_candidate_ids.txt").write_text(",".join(ids) + "\n", encoding="utf-8")
    (args.output_dir / "prompt_example.txt").write_text(
        build_prompt(
            selected[0],
            selected[1],
            args.target_display,
            args.target_description,
            args.image_mode,
            args.decision_profile,
            summaries.get(str(selected[0]["candidate_id"])),
            summaries.get(str(selected[1]["candidate_id"])),
        ),
        encoding="utf-8",
    )

    vlm = load_vlm(args.model, args.device)
    vote_rows: list[dict] = []
    win_counts = {cid: 0 for cid in ids}
    valid_counts = {cid: 0 for cid in ids}

    ordered_pairs: list[tuple[dict, dict]] = []
    for a, b in itertools.combinations(selected, 2):
        ordered_pairs.append((a, b))
        ordered_pairs.append((b, a))

    for idx, (row_a, row_b) in enumerate(ordered_pairs, start=1):
        images = load_images(pool_dir, row_a, args.image_mode) + load_images(pool_dir, row_b, args.image_mode)
        prompt = build_prompt(
            row_a,
            row_b,
            args.target_display,
            args.target_description,
            args.image_mode,
            args.decision_profile,
            summaries.get(str(row_a["candidate_id"])),
            summaries.get(str(row_b["candidate_id"])),
        )
        raw = generate(vlm, args.device, images, SYSTEM_PROMPT, prompt, args.max_new_tokens)
        winner, conf, status = parse_winner(raw)
        winner_id = ""
        if winner == "A":
            winner_id = str(row_a["candidate_id"])
        elif winner == "B":
            winner_id = str(row_b["candidate_id"])
        if winner_id:
            win_counts[winner_id] += 1
            valid_counts[str(row_a["candidate_id"])] += 1
            valid_counts[str(row_b["candidate_id"])] += 1
        out = {
            "pair_index": idx,
            "candidate_a": row_a["candidate_id"],
            "candidate_b": row_b["candidate_id"],
            "winner": winner,
            "winner_candidate_id": winner_id,
            "confidence": f"{conf:.1f}",
            "parse_status": status,
            "raw_response": raw,
        }
        vote_rows.append(out)
        print(
            f"Pair {idx}/{len(ordered_pairs)} A={row_a['candidate_id']} B={row_b['candidate_id']} "
            f"winner={winner_id or 'NA'} status={status}",
            flush=True,
        )

    score_rows: list[dict] = []
    for cid in ids:
        valid = valid_counts[cid]
        wins = win_counts[cid]
        score_rows.append(
            {
                "candidate_id": cid,
                "wins": wins,
                "valid_comparisons": valid,
                "win_rate": f"{wins / valid if valid else 0.0:.6f}",
            }
        )
    score_rows.sort(key=lambda row: (float(row["win_rate"]), int(row["wins"])), reverse=True)
    write_csv(args.output_dir / "pairwise_votes.csv", vote_rows)
    write_csv(args.output_dir / "candidate_pairwise_scores.csv", score_rows)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
