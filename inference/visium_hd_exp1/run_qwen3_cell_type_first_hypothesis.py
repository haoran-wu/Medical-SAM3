#!/usr/bin/env python3
"""Run Qwen3/Qwen3-VL on Step 1 cell-type-first text prompts.

This runner intentionally uses no images.  It scores whether a pure
FICTURE-derived cell-type / marker-gene profile can generate useful tissue
hypotheses before H&E and FICTURE images are shown.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
ALLOWED = {"plausible", "uncertain", "unlikely"}


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


def normalize_status(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"possible", "likely", "supported", "yes"}:
        return "plausible"
    if text in {"maybe", "ambiguous", "mixed", "unclear"}:
        return "uncertain"
    if text in {"not_likely", "no", "unsupported"}:
        return "unlikely"
    return text if text in ALLOWED else "uncertain"


def extract_json(text: str) -> dict:
    stripped = text.strip()
    candidates = re.findall(r"\{.*\}", stripped, flags=re.S)
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


def parse_response(text: str) -> dict[str, object]:
    data = extract_json(text)
    hypotheses = data.get("hypotheses", data)
    if not isinstance(hypotheses, dict):
        raise ValueError("hypotheses is not an object")
    out: dict[str, object] = {}
    missing = []
    for key in CLASS_KEYS:
        if key not in hypotheses:
            missing.append(key)
        out[key] = normalize_status(hypotheses.get(key, "uncertain"))
    if missing:
        raise ValueError(f"Missing hypothesis keys: {missing}")
    out["main_ambiguity"] = str(data.get("main_ambiguity", ""))[:500]
    out["composition_warning"] = str(data.get("composition_warning", ""))[:500]
    return out


def load_text_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    common = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    device_map = os.environ.get("VLM_DEVICE_MAP", "").strip()
    if device_map:
        common["device_map"] = device_map
        max_memory = {}
        if cuda_mem := os.environ.get("VLM_MAX_MEMORY_CUDA"):
            max_memory[0] = cuda_mem
        if cpu_mem := os.environ.get("VLM_MAX_MEMORY_CPU"):
            max_memory["cpu"] = cpu_mem
        if max_memory:
            common["max_memory"] = max_memory
        if offload_dir := os.environ.get("VLM_OFFLOAD_FOLDER"):
            Path(offload_dir).mkdir(parents=True, exist_ok=True)
            common["offload_folder"] = offload_dir

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **common)
        if not device_map:
            model.to(device)
        model.eval()
        return {"kind": "causal_lm", "tokenizer": tokenizer, "model": model}
    except Exception as causal_error:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype, **common)
        if not device_map:
            model.to(device)
        model.eval()
        return {
            "kind": "image_text_to_text_text_only",
            "processor": processor,
            "model": model,
            "causal_error": str(causal_error)[:500],
        }


def generate_text(vlm: dict, device: str, system: str, user: str, max_new_tokens: int) -> str:
    import torch

    if vlm["kind"] == "causal_lm":
        tokenizer = vlm["tokenizer"]
        model = vlm["model"]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system}\n\n{user}\n\nAnswer:"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[-1]
        return tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()

    processor = vlm["processor"]
    model = vlm["model"]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        inputs = processor(text=[text], images=None, return_tensors="pt")
    except TypeError:
        inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()


def status_to_score(status: str) -> int:
    return {"plausible": 100, "uncertain": 50, "unlikely": 0}.get(status, 50)


def write_summary(output_dir: Path, rows: list[dict[str, object]], failed: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("parse_status") == "ok"]
    summary: list[dict[str, object]] = []
    for cls in CLASS_KEYS:
        cls_rows = [row for row in ok if row.get("true_class") == cls]
        plausible = sum(1 for row in cls_rows if row.get(cls) == "plausible")
        uncertain = sum(1 for row in cls_rows if row.get(cls) == "uncertain")
        unlikely = sum(1 for row in cls_rows if row.get(cls) == "unlikely")
        summary.append(
            {
                "true_class": cls,
                "n": len(cls_rows),
                "target_plausible": plausible,
                "target_uncertain": uncertain,
                "target_unlikely": unlikely,
                "target_plausible_or_uncertain": plausible + uncertain,
            }
        )
    write_csv(output_dir / "step1_hypothesis_by_true_class.csv", summary)

    collapse_rows = []
    for cls in CLASS_KEYS:
        counter = Counter(row.get(cls, "") for row in ok)
        collapse_rows.append({"class_key": cls, **{key: counter.get(key, 0) for key in sorted(ALLOWED)}})
    write_csv(output_dir / "step1_status_distribution.csv", collapse_rows)

    gates = [
        {
            "parsed_rows": len(ok),
            "failed_rows": len(failed),
            "n_rows": len(rows) + len(failed),
            "parse_ok": "true" if not failed else "false",
        }
    ]
    write_csv(output_dir / "step1_run_gate.csv", gates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_csv(args.requests_csv)
    if args.limit:
        all_rows = all_rows[: args.limit]

    done_keys: set[str] = set()
    existing_rows: list[dict[str, object]] = []
    out_path = args.output_dir / "step1_hypotheses.csv"
    failed_path = args.output_dir / "failed_rows.csv"
    if args.resume and out_path.exists():
        existing_rows = read_csv(out_path)
        done_keys = {str(row.get("row_index")) for row in existing_rows}

    vlm = load_text_model(args.model, args.device)
    results = list(existing_rows)
    failed: list[dict[str, object]] = read_csv(failed_path) if args.resume and failed_path.exists() else []

    for row in all_rows:
        row_index = str(row["row_index"])
        if row_index in done_keys:
            continue
        try:
            raw = generate_text(vlm, args.device, row["system_prompt"], row["user_prompt"], args.max_new_tokens)
            parsed = parse_response(raw)
            out = dict(row)
            out.update(parsed)
            out["model"] = args.model
            out["parse_status"] = "ok"
            out["raw_response"] = raw
            for cls in CLASS_KEYS:
                out[f"{cls}_hypothesis_score"] = status_to_score(str(out[cls]))
            results.append(out)
            print(f"Step1 {int(row_index)+1}/{len(all_rows)} {row['candidate_uid']} true={row.get('true_class')} ok", flush=True)
            write_csv(out_path, results)
        except Exception as exc:
            failed_row = dict(row)
            failed_row.update({"model": args.model, "error": str(exc), "parse_status": "failed"})
            failed.append(failed_row)
            print(f"FAILED Step1 {int(row_index)+1}/{len(all_rows)} {row['candidate_uid']}: {exc}", flush=True)
            write_csv(failed_path, failed)

    write_csv(out_path, results)
    write_csv(failed_path, failed)
    write_summary(args.output_dir, results, failed)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "requests_csv": str(args.requests_csv),
                "limit": args.limit,
                "max_new_tokens": args.max_new_tokens,
                "input_mode": "text_only_cell_type_gene_profile",
                "no_images": True,
            },
            indent=2,
        )
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
