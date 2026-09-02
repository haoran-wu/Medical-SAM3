#!/usr/bin/env python3
"""Validate a run against the locked TMA39 SAM3 workflow contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/tma39_fixed_workflow_v1.json"),
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--stage",
        choices=("pre-vlm", "vlm"),
        default="vlm",
        help="Validate through the frozen candidate pool or through the six-view VLM bundle.",
    )
    parser.add_argument(
        "--input-audit",
        type=Path,
        help="Optional ten-TMA input audit used to verify registered input hashes.",
    )
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


def binary_pixel_sha256(path: Path) -> tuple[int, str]:
    pixels = (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    return int(pixels.sum()), hashlib.sha256(pixels.tobytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract["workflow_id"] == "tma39-fixed-sam3-v1", "Wrong workflow ID")
    require(contract["inputs"]["ficture_smoothing"] is False, "FICTURE smoothing is forbidden")
    require(contract["inputs"]["ficture_recoloring"] is False, "FICTURE recoloring is forbidden")
    support = contract["foreground_support"]
    require(
        (
            support["method"],
            support["work_max_side"],
            support["minimum_component_fraction"],
            support["closing_disk_radius"],
            support["dilation_disk_radius"],
            support["full_resolution_resampling"],
        )
        == (
            "registered_he_color_threshold_components_v1",
            2048,
            0.0002,
            2,
            4,
            "nearest",
        ),
        "H&E foreground-support algorithm changed",
    )
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


def validate_run(
    root: Path,
    contract: dict[str, Any],
    *,
    stage: str,
    input_audit_path: Path | None,
    contract_dir: Path,
) -> dict[str, Any]:
    gene = read_json(root / "genemap_scores/summary.json")
    prompts = read_json(root / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json")
    prompt_csv = root / "prompts/final_sustained_edge3/adaptive_box_peak_sustained_edge3_context0.csv"
    prompt_rows = read_csv(prompt_csv)
    ficture = read_json(root / "sam3_main/ficture/run_manifest.json")
    he = read_json(root / "sam3_main/he/run_manifest.json")
    independent = read_json(root / "sam3_independent_he_gap/run_manifest.json")
    gap_prompts = read_json(root / "independent_he_gap_detection/prompt_manifest.json")
    pool = read_json(root / "frozen_pool/run_manifest.json")
    support_path = root / "staging" / f"{pool['tma']}_blind_he_tissue_support.png"
    support_manifest = read_json(support_path.with_suffix(".json"))

    expected_prompts = int(prompts["prompt_count"])
    support_contract = contract["foreground_support"]
    require(support_manifest["method"] == support_contract["method"], "Foreground-support method changed")
    require(support_manifest["work_max_side"] == support_contract["work_max_side"], "Foreground-support work size changed")
    require(
        support_manifest["minimum_component_fraction"] == support_contract["minimum_component_fraction"],
        "Foreground-support component threshold changed",
    )
    require(support_manifest["closing_disk_radius"] == support_contract["closing_disk_radius"], "Foreground-support closing changed")
    require(support_manifest["dilation_disk_radius"] == support_contract["dilation_disk_radius"], "Foreground-support dilation changed")
    require(
        support_manifest["full_resolution_resampling"] == support_contract["full_resolution_resampling"],
        "Foreground-support resampling changed",
    )
    if pool["tma"] == "TMA39":
        support_pixels, support_hash = binary_pixel_sha256(support_path)
        require(support_pixels == support_contract["tma39_binary_pixel_count"], "TMA39 support pixel count changed")
        require(support_hash == support_contract["tma39_binary_pixel_sha256"], "TMA39 support pixels changed")
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

    ficture_selected = [
        row
        for row in read_csv(root / "sam3_main/ficture/selected_candidates.csv")
        if row["selection_method"] == "genemap_support_f1"
    ]
    he_selected = [
        row
        for row in read_csv(root / "sam3_main/he/selected_candidates.csv")
        if row["selection_method"] == "genemap_support_f1"
    ]
    require(len(ficture_selected) <= expected_prompts, "FICTURE selected-mask count exceeds prompt count")
    require(len(he_selected) <= expected_prompts, "H&E selected-mask count exceeds prompt count")
    require(
        ficture.get("prompts_without_eligible_mask", 0) == expected_prompts - len(ficture_selected),
        "FICTURE missing-mask count is not fully audited",
    )
    require(
        he.get("prompts_without_eligible_mask", 0) == expected_prompts - len(he_selected),
        "H&E missing-mask count is not fully audited",
    )

    require(independent["expansion_percent_per_side"] == 10, "Independent H&E context must be 10%")
    require(independent["inference_calls_per_prompt"] == 1, "Independent H&E must have one call per prompt")
    require(independent["selection_used_annotation"] is False, "Annotation leaked into independent H&E selection")
    require(gap_prompts["cell_size_px"] == 64, "Independent H&E cell size changed")
    require(gap_prompts["min_he_tissue_fraction"] == 0.35, "Independent H&E tissue threshold changed")
    require(gap_prompts["max_ficture_signal_fraction"] == 0.08, "Independent H&E FICTURE threshold changed")
    require(gap_prompts["minimum_connected_cells"] == 6, "Independent H&E connection rule changed")
    require(gap_prompts["box_expansion_fraction"] == 0.1, "Independent H&E context changed")
    require(gap_prompts["detected_region_count"] == independent["prompt_count"], "Independent H&E prompt count mismatch")
    require(pool["main_prompt_count"] == expected_prompts, "Pool prompt count mismatch")
    require(pool["ficture_primary_count"] == len(ficture_selected), "Pool FICTURE count mismatch")
    require(
        pool.get("ficture_missing_count", 0) == expected_prompts - len(ficture_selected),
        "Pool did not preserve the audited FICTURE no-output count",
    )
    require(
        pool.get("same_prompt_he_missing_count", 0) == expected_prompts - len(he_selected),
        "Pool did not preserve the audited H&E no-output count",
    )
    require(pool["same_prompt_he_threshold"] == 0.4, "Same-prompt H&E threshold changed")
    require(pool["dedup_iou_threshold"] == 0.9, "Pool deduplication threshold changed")
    require(pool["selection_used_annotation"] is False, "Annotation leaked into pool selection")
    factor_info = root / "staging/factor_info.csv"
    if not factor_info.is_file() and stage == "vlm":
        factor_info = root / "vlm_bundle/assets/factor_info.csv"
    require(
        sha256(factor_info) == contract["inputs"]["factor_legend_sha256"],
        "FICTURE RGB legend changed",
    )

    if input_audit_path:
        input_audit = read_json(input_audit_path)
        matches = [row for row in input_audit["tmas"] if row["tma"] == pool["tma"]]
        require(len(matches) == 1, "TMA missing or duplicated in input audit")
        expected_input = matches[0]
        he_path = root / "staging" / f"{pool['tma']}_he.png"
        ficture_path = root / "staging" / f"{pool['tma']}_ficture.png"
        require(sha256(he_path) == expected_input["he_sha256"], "Registered H&E hash changed")
        require(sha256(ficture_path) == expected_input["ficture_sha256"], "Raw FICTURE hash changed")

    if stage == "vlm":
        bundle = read_json(root / "vlm_bundle/run_manifest.json")
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

    tma39_reference_match: bool | None = None
    if pool["tma"] == "TMA39":
        reference_path = contract_dir / contract["foreground_support"]["tma39_candidate_reference_manifest"]
        reference = read_json(reference_path)
        require(sha256(prompt_csv) == reference["prompt_csv_sha256"], "TMA39 prompt CSV differs from reference")
        observed_records = []
        for row in final_rows:
            mask = root / "frozen_pool" / row["mask_path"]
            area, pixel_hash = binary_pixel_sha256(mask)
            observed_records.append(
                {
                    "source": row["source"],
                    "area_pixels": area,
                    "binary_pixel_sha256": pixel_hash,
                }
            )
        observed_records.sort(key=lambda item: (item["source"], item["binary_pixel_sha256"]))
        require(len(observed_records) == reference["reference_candidate_count"], "TMA39 candidate count differs from reference")
        require(observed_records == reference["candidates"], "TMA39 candidate pixels differ from reference")
        tma39_reference_match = True

    return {
        "workflow_id": contract["workflow_id"],
        "tma": pool["tma"],
        "main_prompts": expected_prompts,
        "final_candidates": pool["final_candidate_count"],
        "source_counts": pool["final_source_counts"],
        "selection_used_annotation": False,
        "validated_stage": stage,
        "tma39_reference_match": tma39_reference_match,
        "status": "valid",
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    result: dict[str, Any] = {"workflow_id": contract["workflow_id"], "contract": "valid"}
    if args.run_root:
        result["run"] = validate_run(
            args.run_root.resolve(),
            contract,
            stage=args.stage,
            input_audit_path=args.input_audit.resolve() if args.input_audit else None,
            contract_dir=contract_path.parent,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
