#!/usr/bin/env python3
"""Run a VLM direct-retrieval hit-test on paired H&E/FICTURE candidate crops."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


LABEL_ORDER = [
    ("lung_bronchiola", "bronchiola"),
    ("lung_alveoli_normal_adjacent", "alveoli"),
    ("lung_vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]

SCORE_RUBRIC_SCORE_ONLY = """Give one integer similarity score from 0 to 100.
Use the full scale when the visual evidence differs across candidates:
- near 0 means clearly not the target tissue class
- around 25 means weak or mostly unrelated match
- around 50 means mixed or partial match
- around 75 means likely target, but imperfect boundary or coverage
- near 100 means near-perfect target candidate
Judge this candidate independently from the images. Do not use a default score.
Your entire answer must be only the integer score. Do not return JSON, markdown, words, precision, recall, or a reason."""

SCORE_RUBRIC_WITH_REASON = """Give one integer similarity score from 0 to 100.
Use the full scale when the visual evidence differs across candidates. Also provide one short reason.
Return one valid JSON object with exactly two keys: score and reason. Do not include precision or recall."""


def score_rubric(include_reason: bool) -> str:
    return SCORE_RUBRIC_WITH_REASON if include_reason else SCORE_RUBRIC_SCORE_ONLY


def load_factor_legend(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["Factor"]))
    return rows


def format_factor_legend(rows: Sequence[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            "Factor {factor}: RGB {rgb}; Major Compartment: {major}; Celltype2: {celltype}".format(
                factor=row["Factor"],
                rgb=row["RGB"],
                major=row["Major Compartment"],
                celltype=row["Celltype2"],
            )
        )
    return "\n".join(lines)


def row_key(row: dict) -> Tuple[str, str, str, str, str]:
    return (row["label"], row["source"], row["run"], row["setting"], str(row["candidate_id"]))


def parse_json_score(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
        raw_score = float(nums[0]) if nums else 0.0
        score = raw_score / 100.0 if raw_score > 1.0 else raw_score
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "raw_score": raw_score,
            "reason": text[:240],
            "parse_status": "fallback_number" if nums else "no_score_found",
        }
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"score": 0.0, "raw_score": 0.0, "reason": text[:240], "parse_status": "malformed_json"}
    data["reason"] = str(data.get("reason", ""))[:360]
    try:
        raw_score = float(data.get("score", 0.0))
        data["raw_score"] = raw_score
        data["score"] = max(0.0, min(1.0, raw_score / 100.0 if raw_score > 1.0 else raw_score))
        data["parse_status"] = "ok"
    except Exception:
        data["score"] = 0.0
        data["raw_score"] = 0.0
        data["parse_status"] = "bad_score"
    return data


def load_vlm(model_name: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = None
    processor_error: Exception | None = None
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        processor_error = exc
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    device_map = os.environ.get("VLM_DEVICE_MAP", "").strip()
    if device_map:
        load_kwargs["device_map"] = device_map
        max_memory = {}
        if cuda_mem := os.environ.get("VLM_MAX_MEMORY_CUDA"):
            max_memory[0] = cuda_mem
        if cpu_mem := os.environ.get("VLM_MAX_MEMORY_CPU"):
            max_memory["cpu"] = cpu_mem
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        if offload_dir := os.environ.get("VLM_OFFLOAD_FOLDER"):
            Path(offload_dir).mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = offload_dir
    try:
        if processor is None:
            raise RuntimeError(f"AutoProcessor load failed: {processor_error}")
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        tokenizer = None
        kind = "image_text_to_text"
    except Exception:
        if os.environ.get("VLM_ALLOW_AUTOMODEL_FALLBACK", "1") != "1":
            raise
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        kind = "automodel_chat" if hasattr(model, "chat") else "automodel"
    if not device_map:
        model.to(device)
    model.eval()
    return {"kind": kind, "processor": processor, "tokenizer": tokenizer, "model": model}


def build_messages(
    row: dict,
    prompt_mode: str,
    include_reason: bool = False,
    factor_legend_text: str = "",
) -> Tuple[str, str]:
    system = (
        "You are a careful pathology image judge. Your task is direct retrieval: "
        "score whether the candidate mask/image crop is a good visual match for the requested target class. "
        "You must output one continuous ranking score, not precision/recall estimates."
    )
    image_cue = row.get("image_cue") or (
        "Image 1: H&E histology crop for the candidate mask.\n"
        "Image 2: official FICTURE factor-color crop for the same candidate mask."
    )
    if prompt_mode == "target_only":
        prompt = f"""
You are given two aligned image crops for the same candidate mask:
{image_cue}

Target tissue class: {row['display']} ({row['label']})

Do not assume the candidate is correct. Score how well the candidate matches only this target class.

{score_rubric(include_reason)}
"""
        return system, prompt.strip()

    if prompt_mode == "description_only":
        prompt = f"""
You are given two aligned image crops for the same candidate mask:
{image_cue}

Target tissue class: {row['display']} ({row['label']})
Meaning of the target class: {row['target_description']}

Use only the two images, the candidate mask/crop, and the target-class description above.
Do not use any FICTURE color/cell-type/gene legend. Do not assume the candidate is correct.
Score how well the candidate matches the target class.

{score_rubric(include_reason)}
"""
        return system, prompt.strip()

    if prompt_mode == "full_legend":
        prompt = f"""
You are given two aligned image crops for the same candidate mask:
{image_cue}

The official FICTURE image uses this factor legend:
{factor_legend_text}

Target tissue class: {row['display']} ({row['label']})
Meaning of the target class: {row['target_description']}

Use both H&E morphology and FICTURE factor colors.
Do not assume the candidate is correct. Score how well the candidate matches the target class.

{score_rubric(include_reason)}
"""
        return system, prompt.strip()

    prompt = f"""
You are given two aligned image crops for the same candidate mask:
{image_cue}

Target tissue class: {row['display']} ({row['label']})
Meaning of the target class: {row['target_description']}

FICTURE color/cell-type/gene hints for this target:
{row.get('target_factor_hints') or 'not available'}

Candidate FICTURE color composition:
{row.get('candidate_factor_composition') or 'not available'}

Use the H&E morphology, the FICTURE color/cell-type/gene context, and the candidate mask shape.
Do not assume the candidate is correct. Score how well the candidate matches the target class.

{score_rubric(include_reason)}
"""
    return system, prompt.strip()


def generate(vlm, device: str, images: List[Image.Image], system: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    processor = vlm["processor"]
    model = vlm["model"]
    if vlm["kind"] == "automodel_chat":
        messages = [{"role": "user", "content": images + [f"{system}\n\n{prompt}"]}]
        answer = model.chat(
            image=None,
            msgs=messages,
            tokenizer=vlm["tokenizer"],
            sampling=False,
            max_new_tokens=max_new_tokens,
        )
        return str(answer).strip()
    if vlm["kind"] == "automodel":
        raise RuntimeError(
            "Model loaded with AutoModel fallback but does not expose a chat() API; "
            "this repository likely needs a custom runner."
        )

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
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            input_len = inputs["input_ids"].shape[-1]
            return processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"VLM generation failed: {last_error}")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_warning(scores: List[float]) -> Tuple[int, int, str]:
    rounded = [round(score, 6) for score in scores]
    unique_count = len(set(rounded))
    top_score = max(rounded) if rounded else 0.0
    top1_tie_size = sum(1 for score in rounded if score == top_score)
    if unique_count <= 1:
        return unique_count, top1_tie_size, "uninformative tie: all scores are identical"
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
        top3 = ranked[:3]
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
                "good_count_top3": sum(1 for row in top3 if row["sample_bucket"] == "GOOD"),
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
    prompt_mode = rows[0].get("prompt_mode", "feature_aware") if rows else "feature_aware"
    per_class_counts = sorted({int(row.get("n_candidates") or 0) for row in metrics_rows if row.get("n_candidates")})
    if len(per_class_counts) == 1:
        pool_size_text = f"每个类别 {per_class_counts[0]} 个候选 mask"
    elif per_class_counts:
        pool_size_text = f"每个类别 {per_class_counts[0]} 到 {per_class_counts[-1]} 个候选 mask"
    else:
        pool_size_text = "每个类别若干候选 mask"
    metric_table = [
        "<table><thead><tr><th>method</th><th>class</th><th>top1</th><th>top1 source</th><th>top1 Dice</th><th>GOOD in top5</th><th>best GOOD rank</th><th>tie warning</th></tr></thead><tbody>"
    ]
    for row in metrics_rows:
        metric_table.append(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['display'])}</td>"
            f"<td>{html.escape(row['top1_bucket'])}</td>"
            f"<td>{html.escape(row.get('top1_source') or '')}</td>"
            f"<td>{html.escape(row['top1_hidden_dice'])}</td>"
            f"<td>{html.escape(str(row['good_count_top5']))}</td>"
            f"<td>{html.escape(str(row['best_good_rank']))}</td>"
            f"<td>{html.escape(row.get('tie_warning') or '')}</td>"
            "</tr>"
        )
    metric_table.append("</tbody></table>")

    cards = []
    for label, display in LABEL_ORDER:
        cards.append(f"<h2>{html.escape(display)}</h2><div class='grid'>")
        for row in sorted([r for r in rows if r["label"] == label], key=lambda r: float(r["vlm_score"]), reverse=True):
            cards.append(
                "<article>"
                f"<h3>{html.escape(row['sample_bucket'])} | VLM score {float(row['vlm_score']):.3f} | Dice {float(row['hidden_dice']):.3f}</h3>"
                "<div class='imgs'>"
                f"<img src=\"{html.escape(row['he_crop_rel'])}\">"
                f"<img src=\"{html.escape(row['ficture_crop_rel'])}\">"
                "</div>"
                f"<p>{html.escape(row['source'])} / {html.escape(row['setting'])} / {html.escape(str(row['candidate_id']))}</p>"
                f"<p><b>VLM reason:</b> {html.escape(row.get('vlm_reason') or '')}</p>"
                "</article>"
            )
        cards.append("</div>")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Paired VLM hit-test result</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #20242a; }}
.note {{ max-width: 1040px; line-height: 1.55; }}
table {{ border-collapse: collapse; margin: 18px 0; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px 10px; text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
article {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: white; }}
.imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
img {{ width: 100%; border: 1px solid #eee; }}
</style>
</head>
<body>
<h1>VLM direct retrieval hit-test</h1>
<div class="note">
<p><b>模型：</b>{html.escape(model_name)}</p>
<p><b>Prompt 模式：</b>{html.escape(prompt_mode)}</p>
<p><b>实验设计：</b>{html.escape(pool_size_text)}，包含 GOOD/MID/BAD。模型每次只看同一个候选的两张图：H&E crop 和官方 FICTURE crop。模型输出一个 0-1 分数。Dice/Precision/Recall 是隐藏答案，只用于最后评价它有没有把 GOOD 排到前面。</p>
<p><b>注意：</b>这个页面只报告 VLM 自己的打分结果，不混入 CLIP 表格。</p>
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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-reason", action="store_true", help="Ask the VLM to include a short reason for debugging.")
    parser.add_argument(
        "--prompt-mode",
        choices=["feature_aware", "target_only", "description_only", "full_legend"],
        default="feature_aware",
        help=(
            "feature_aware includes FICTURE color/cell-type/gene context; "
            "full_legend includes the complete FICTURE RGB/Major Compartment/Celltype2 legend; "
            "target_only only names the target class; "
            "description_only includes target class plus plain target morphology/meaning only."
        ),
    )
    parser.add_argument(
        "--factor-legend-csv",
        type=Path,
        help="CSV with complete FICTURE factor legend. Required when --prompt-mode full_legend.",
    )
    args = parser.parse_args()

    factor_legend_text = ""
    if args.prompt_mode == "full_legend":
        if not args.factor_legend_csv:
            raise SystemExit("--factor-legend-csv is required for --prompt-mode full_legend")
        factor_legend_text = format_factor_legend(load_factor_legend(args.factor_legend_csv))

    requests = list(csv.DictReader((args.pool_dir / "public_vlm_requests.csv").open()))
    hidden = {row_key(row): row for row in csv.DictReader((args.pool_dir / "hidden_candidate_truth.csv").open())}
    if args.limit:
        requests = requests[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vlm = load_vlm(args.model, args.device)
    rows: List[dict] = []
    for idx, row in enumerate(requests, start=1):
        he_image = Image.open(args.pool_dir / row["he_crop_rel"]).convert("RGB")
        ficture_image = Image.open(args.pool_dir / row["ficture_crop_rel"]).convert("RGB")
        system, prompt = build_messages(row, args.prompt_mode, args.include_reason, factor_legend_text)
        raw = generate(vlm, args.device, [he_image, ficture_image], system, prompt, args.max_new_tokens)
        parsed = parse_json_score(raw)
        truth = hidden[row_key(row)]
        out = dict(row)
        out.update(
            {
                "model": args.model,
                "prompt_mode": args.prompt_mode,
                "actual_system_text": system,
                "actual_prompt_text": prompt,
                "vlm_score": f"{float(parsed['score']):.9f}",
                "vlm_score_raw": f"{float(parsed.get('raw_score', parsed['score'])):.9f}",
                "vlm_parse_status": parsed.get("parse_status", ""),
                "raw_response": raw,
                "hidden_dice": truth["hidden_dice"],
                "hidden_precision": truth["hidden_precision"],
                "hidden_recall": truth["hidden_recall"],
                "hidden_iou": truth["hidden_iou"],
            }
        )
        if args.include_reason:
            out["vlm_reason"] = parsed.get("reason", "")
        rows.append(out)
        print(f"VLM scored {idx}/{len(requests)} {row['display']} {row['sample_bucket']} score={out['vlm_score']}", flush=True)

    score_fields = list(rows[0].keys())
    write_csv(args.output_dir / "vlm_candidate_scores.csv", rows, score_fields)

    metrics_rows: List[dict] = []
    metrics_rows.extend(retrieval_metrics(rows, "vlm_score", f"VLM direct score ({args.prompt_mode}): {args.model}"))
    write_csv(args.output_dir / "retrieval_comparison_metrics.csv", metrics_rows)

    write_html(args.output_dir, args.model, rows, metrics_rows)
    (args.output_dir / "README_中文.md").write_text(
        "# Paired VLM Hit-Test Result\n\n"
        f"Model: `{args.model}`\n\n"
        f"Prompt mode: `{args.prompt_mode}`\n\n"
        "This tests direct retrieval: can the model rank GOOD candidate masks above MID/BAD masks in a small controlled pool.\n"
        "The model did not receive hidden Dice/Precision/Recall.\n"
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
