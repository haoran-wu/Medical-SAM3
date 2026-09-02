#!/usr/bin/env python3
"""Validate a run against the locked TMA39 SAM3 workflow contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/tma39_fixed_workflow_v1.json"),
    )
    parser.add_argument("--run-root", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract["workflow_id"] == "tma39-fixed-sam3-v1", "Wrong workflow ID")
    require(contract["inputs"]["ficture_smoothing"] is False, "FICTURE smoothing is forbidden")
    require(contract["inputs"]["ficture_recoloring"] is False, "FICTURE recoloring is forbidden")
    require(contract["genemap"]["fixed_gene_count"] == 120, "Expected 120 fixed genes")
    require(contract["genemap"]["fixed_module_count"] == 9, "Expected nine GeneMap modules")
    require(contract["main_prompt"]["box_expansion_fraction"] == 0.0, "Main boxes must be tight")
    require(
        contract["same_prompt_he_supplement"]["minimum_fraction_outside_complete_ficture_union"] == 0.4,
        "Same-prompt H&E threshold must be 40%",
    )
    independent = contract["independent_he_supplement"]
    require(
        (independent["cell_size_pixels"], independent["minimum_he_tissue_fraction"],
         independent["maximum_ficture_signal_fraction"], independent["minimum_connected_cells"],
         independent["box_context_fraction"]) == (64, 0.35, 0.08, 6, 0.1),
        "Independent H&E detector parameters changed",
    )
    require(contract["pool"]["dedup_iou_threshold"] == 0.9, "Deduplication IoU must be 0.90")
    require(contract["pool"]["union_masks"] is False, "Mask union is forbidden")
    require(contract["vlm"]["views_per_candidate"] == 6, "VLM must receive six views")


def validate_run(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    gene = read_json(root / "genemap_scores/summary.json")
    prompts = read_json(root / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json")
    prompt_csv = root / "prompts/final_sustained_edge3/adaptive_box_peak_sustained_edge3_context0.csv"
    prompt_rows = read_csv(prompt_csv)
    ficture = read_json(root / "sam3_main/ficture/run_manifest.json")
    he = read_json(root / "sam3_main/he/run_manifest.json")
    independent = read_json(root / "sam3_independent_he_gap/run_manifest.json")
    pool = read_json(root / "frozen_pool/run_manifest.json")
    bundle = read_json(root / "vlm_bundle/run_manifest.json")

    expected_prompts = int(prompts["prompt_count"])
    require(gene["fixed_gene_count"] == 120 and gene["active_gene_count"] == 120, "Gene count changed")
    require(gene["gene_module_count"] == 9, "Module count changed")
    require(
        gene["fixed_module_source_sha256"] == contract["genemap"]["module_membership_sha256"],
        "Gene-to-module membership changed",
    )
    require(gene["selection_used_annotation"] is False, "Annotation leaked into GeneMap generation")
    require(prompts["minimum_shared_edge_bins"] == 3, "GeneMap region merge rule changed")
    require(prompts["selection_used_annotation"] is False, "Annotation leaked into prompt generation")
    require(len(prompt_rows) == expected_prompts, "Prompt CSV count mismatch")
    require(all(row["selection_used_annotation"].lower() == "false" for row in prompt_rows), "Prompt row used annotation")

    for source, manifest in (("FICTURE", ficture), ("H&E", he)):
        require(manifest["prompt_count"] == expected_prompts, f"{source} prompt count mismatch")
        require(manifest["image_encoding_count"] == 1, f"{source} image must be encoded once")
        require(manifest["inference_calls_per_prompt"] == 1, f"{source} must have one SAM call per prompt")
        require(manifest["total_inference_calls"] == expected_prompts, f"{source} total call count mismatch")
        require(manifest["positive_point_required"] is True, f"{source} must require the point")
        require(manifest["selection_used_annotation"] is False, f"Annotation leaked into {source} selection")

    require(independent["expansion_percent_per_side"] == 10, "Independent H&E context must be 10%")
    require(independent["inference_calls_per_prompt"] == 1, "Independent H&E must have one call per prompt")
    require(independent["selection_used_annotation"] is False, "Annotation leaked into independent H&E selection")
    require(pool["main_prompt_count"] == expected_prompts, "Pool prompt count mismatch")
    require(pool["ficture_primary_count"] == expected_prompts, "Every FICTURE primary must be retained")
    require(pool["same_prompt_he_threshold"] == 0.4, "Same-prompt H&E threshold changed")
    require(pool["dedup_iou_threshold"] == 0.9, "Pool deduplication threshold changed")
    require(pool["selection_used_annotation"] is False, "Annotation leaked into pool selection")
    require(bundle["candidate_count"] == pool["final_candidate_count"], "VLM bundle count differs from pool")
    require(bundle["image_count_per_candidate"] == 6, "VLM input count changed")
    require(
        bundle["input_hashes"]["factor_info"] == contract["inputs"]["factor_legend_sha256"],
        "FICTURE RGB legend changed",
    )

    final_rows = read_csv(root / "frozen_pool/final_pool.csv")
    require(len(final_rows) == pool["final_candidate_count"], "Final pool CSV count mismatch")
    require(len({row["candidate_id"] for row in final_rows}) == len(final_rows), "Duplicate candidate IDs")
    require(all(row["selection_used_annotation"].lower() == "false" for row in final_rows), "Final pool used annotation")
    for row in final_rows:
        mask = root / "frozen_pool" / row["mask_path"]
        require(mask.is_file(), f"Missing candidate mask: {mask}")
        require(sha256(mask) == row["mask_sha256"], f"Candidate mask hash mismatch: {row['candidate_id']}")

    return {
        "workflow_id": contract["workflow_id"],
        "tma": pool["tma"],
        "main_prompts": expected_prompts,
        "final_candidates": pool["final_candidate_count"],
        "source_counts": pool["final_source_counts"],
        "selection_used_annotation": False,
        "status": "valid",
    }


def main() -> None:
    args = parse_args()
    contract = read_json(args.contract)
    validate_contract(contract)
    result: dict[str, Any] = {"workflow_id": contract["workflow_id"], "contract": "valid"}
    if args.run_root:
        result["run"] = validate_run(args.run_root.resolve(), contract)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
