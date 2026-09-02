#!/usr/bin/env python3
"""Run Gemma on every candidate in the corrected frozen TMA30 bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline


CLASSES = ("airway", "arteriole", "venule", "alveoli")
IMAGE_NAMES = (
    "01_close_he.jpg",
    "02_close_ficture.png",
    "03_medium_he.jpg",
    "04_medium_ficture.png",
    "05_full_he.jpg",
    "06_full_ficture.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fixed_ficture_reference(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    lines = []
    for row in sorted(rows, key=lambda item: int(item["Factor"])):
        marker_source = row.get("TopGene_specific") or row.get("TopGene_pval") or ""
        genes = [value.strip() for value in marker_source.split(",") if value.strip()][:6]
        lines.append(
            f"- Factor {int(row['Factor'])}: {row['hex'].upper()} / RGB {row['RGB']}; "
            f"marker-inferred cell type: {row['Celltype2']}; marker genes: {', '.join(genes) or 'not available'}."
        )
    if len(lines) != 12:
        raise RuntimeError(f"Expected 12 FICTURE factors, found {len(lines)}")
    return "\n".join(lines)


def extract_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```.*$", "", stripped, flags=re.DOTALL)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("Response is not a JSON object")
    return value


def validate(value: dict[str, Any], candidate_id: str, prompt_version: str) -> dict[str, Any]:
    expected = {"prompt_version", "candidate_id", "class_scores", "predicted_class", "reason"}
    if set(value) != expected:
        raise ValueError(f"Unexpected keys: {sorted(value)}")
    if value["prompt_version"] != prompt_version or value["candidate_id"] != candidate_id:
        raise ValueError("Candidate or prompt version mismatch")
    if not isinstance(value["class_scores"], dict) or set(value["class_scores"]) != set(CLASSES):
        raise ValueError("class_scores must contain exactly the four target classes")
    scores = {label: float(value["class_scores"][label]) for label in CLASSES}
    if any(score < 0 or score > 100 for score in scores.values()):
        raise ValueError("Class score outside 0-100")
    highest = max(scores.values())
    winners = [label for label, score in scores.items() if score == highest]
    predicted = str(value["predicted_class"])
    if predicted not in winners:
        raise ValueError("predicted_class must be one of the highest-scoring classes")
    reason = str(value["reason"]).strip()
    if not reason:
        raise ValueError("Missing reason")
    return {
        "class_scores": scores,
        "predicted_class": predicted,
        "top_score_tie_count": len(winners),
        "reason": reason,
    }


def generated_text(result: Any) -> str:
    if not isinstance(result, list) or not result:
        raise ValueError("Unexpected pipeline result")
    text = result[0].get("generated_text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and text:
        return str(text[-1].get("content", ""))
    raise ValueError("No generated text")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "order", "candidate_id", "candidate_uid", "source", "gene_module",
        "predicted_class", "top_score_tie_count", *[f"{label}_score" for label in CLASSES],
        "reason", "evaluation_status", "official_annotation",
        "official_class_for_evaluation", "class_correct", "posthoc_precision",
        "posthoc_recall", "posthoc_dice", "attempt_count", "inference_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat.update({f"{label}_score": row["class_scores"][label] for label in CLASSES})
            writer.writerow({field: flat.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest = json.loads((args.bundle_root / "run_manifest.json").read_text())
    source_rows = json.loads((args.bundle_root / "vlm_source_rows.json").read_text())
    expected_count = int(bundle_manifest["candidate_count"])
    if len(source_rows) != expected_count:
        raise RuntimeError("Bundle candidate count mismatch")
    if len({row["candidate_uid"] for row in source_rows}) != expected_count:
        raise RuntimeError("Bundle candidate IDs are not unique")
    system_prompt = (args.bundle_root / "system_prompt.txt").read_text(encoding="utf-8")
    user_template = (args.bundle_root / "user_prompt_template.txt").read_text(encoding="utf-8")
    prompt_version = bundle_manifest["prompt_version"]
    reference = fixed_ficture_reference(args.bundle_root / "assets/factor_info.csv")

    checkpoint = args.output_dir / "checkpoint.json"
    completed = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    cached = {row["candidate_uid"]: row for row in completed}
    generator = pipeline(
        "image-text-to-text",
        model=str(args.model_path),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    for index, source in enumerate(source_rows, start=1):
        uid = source["candidate_uid"]
        if uid in cached:
            print(json.dumps({"index": index, "candidate": source["candidate_id"], "status": "reused"}), flush=True)
            continue
        folder = args.bundle_root / "input_images" / source["folder"]
        image_paths = [folder / name for name in IMAGE_NAMES]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing input images: {missing}")
        user_prompt = user_template.format(
            candidate_id=source["candidate_id"],
            ficture_rgb_reference=reference,
            prompt_version=prompt_version,
        )
        content = [{"type": "image", "url": str(path.resolve())} for path in image_paths]
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        for attempt in range(1, args.max_attempts + 1):
            started = time.time()
            try:
                result = generator(
                    text=messages,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                )
                raw = generated_text(result)
                parsed = validate(extract_json(raw), source["candidate_id"], prompt_version)
                record = {
                    **source,
                    "model": args.model_name,
                    "model_path": str(args.model_path),
                    "prompt_version": prompt_version,
                    "system_prompt_sha256": sha256_text(system_prompt),
                    "user_prompt_sha256": sha256_text(user_prompt),
                    "image_paths": [str(path.relative_to(args.bundle_root)) for path in image_paths],
                    "image_sha256": [sha256_file(path) for path in image_paths],
                    **parsed,
                    "class_correct": (
                        parsed["predicted_class"] == source["official_class_for_evaluation"]
                        if source["evaluation_status"] == "evaluable"
                        else None
                    ),
                    "attempt_count": attempt,
                    "inference_seconds": time.time() - started,
                    "raw_text": raw,
                }
                cached[uid] = record
                completed = [cached[row["candidate_uid"]] for row in source_rows if row["candidate_uid"] in cached]
                atomic_json(checkpoint, completed)
                write_csv(args.output_dir / "vlm_results.csv", completed)
                print(
                    json.dumps(
                        {
                            "index": index,
                            "candidate": source["candidate_id"],
                            "prediction": record["predicted_class"],
                            "evaluable": source["evaluation_status"] == "evaluable",
                        }
                    ),
                    flush=True,
                )
                break
            except Exception as error:  # noqa: BLE001
                print(
                    json.dumps(
                        {
                            "index": index,
                            "candidate": source["candidate_id"],
                            "attempt": attempt,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    ),
                    flush=True,
                )
                if attempt == args.max_attempts:
                    raise

    completed = [cached[row["candidate_uid"]] for row in source_rows]
    atomic_json(args.output_dir / "vlm_results.json", completed)
    write_csv(args.output_dir / "vlm_results.csv", completed)
    evaluable = [row for row in completed if row["evaluation_status"] == "evaluable"]
    correct = [row for row in evaluable if row["class_correct"]]
    manifest = {
        "status": "complete",
        "model": args.model_name,
        "model_path": str(args.model_path),
        "tma": "TMA30",
        "candidate_count": len(completed),
        "unique_candidate_count": len({row["candidate_uid"] for row in completed}),
        "six_image_hashes_per_candidate": all(len(row["image_sha256"]) == 6 for row in completed),
        "prompt_version": prompt_version,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_template_sha256": sha256_text(user_template),
        "predicted_class_counts": dict(Counter(row["predicted_class"] for row in completed)),
        "top_score_tie_candidate_count": sum(row["top_score_tie_count"] > 1 for row in completed),
        "evaluable_candidate_count": len(evaluable),
        "correct_evaluable_candidate_count": len(correct),
        "accuracy_among_evaluable_candidates": len(correct) / len(evaluable) if evaluable else None,
        "total_inference_seconds": sum(row["inference_seconds"] for row in completed),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    atomic_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
