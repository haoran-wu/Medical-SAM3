#!/usr/bin/env python3
"""Build GeneMap scores for one TMA using a fixed cross-TMA gene-module definition."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tma", required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--overview-manifest", type=Path, required=True)
    parser.add_argument("--fixed-gene-modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gene-bin-size", type=int, default=8)
    parser.add_argument("--minimum-total-transcripts", type=int, default=1)
    parser.add_argument("--minimum-detected-bins", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_size(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", 1)
    return int(float(left)), int(float(right))


def load_overview(path: Path, tma: str) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if row["tma"] == tma:
            return row
    raise RuntimeError(f"{tma} missing from overview manifest")


def load_fixed_modules(path: Path) -> dict[int, list[str]]:
    modules: dict[int, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            module = int(row["module"])
            genes = [gene.strip() for gene in row["all_genes"].split(",") if gene.strip()]
            if not genes:
                raise RuntimeError(f"Module {module} has no genes")
            modules[module] = genes
    expected = list(range(1, max(modules) + 1))
    if sorted(modules) != expected:
        raise RuntimeError(f"Module identifiers must be consecutive from 1: {sorted(modules)}")
    return modules


def aggregate_fixed_gene_bins(
    transcripts: Path,
    overview: dict[str, Any],
    bin_size: int,
    fixed_genes: set[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[int, int], int, int]:
    low_w, low_h = parse_size(overview["expanded_lowres_size"])
    grid_w = math.ceil(low_w / bin_size)
    grid_h = math.ceil(low_h / bin_size)
    affine = np.asarray(
        overview["native_to_old_he_affine_meta"]["affine_native_xy_to_old_he"],
        dtype=np.float64,
    )
    expanded = np.asarray(json.loads(overview["expanded_old_he_bbox_xyxy"]), dtype=np.float64)
    gene_bins = {gene: np.zeros(grid_h * grid_w, dtype=np.uint32) for gene in fixed_genes}
    total = np.zeros(grid_h * grid_w, dtype=np.uint32)
    total_rows = 0
    mapped_rows = 0

    with gzip.open(transcripts, "rt", encoding="utf-8") as handle:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                handle,
                sep="\t",
                chunksize=350_000,
                dtype={"#x": "float32", "y": "float32", "gene": "string", "count": "uint16"},
            ),
            start=1,
        ):
            total_rows += len(chunk)
            x = chunk["#x"].to_numpy(dtype=np.float64, copy=False)
            y = chunk["y"].to_numpy(dtype=np.float64, copy=False)
            old_x = affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]
            old_y = affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]
            px = np.floor(old_x - expanded[0]).astype(np.int32)
            py = np.floor(old_y - expanded[1]).astype(np.int32)
            valid = (px >= 0) & (px < low_w) & (py >= 0) & (py < low_h)
            if not valid.any():
                continue
            mapped_rows += int(valid.sum())
            genes = chunk.loc[valid, "gene"].astype(str).to_numpy()
            counts = chunk.loc[valid, "count"].to_numpy(dtype=np.uint32, copy=False)
            flat = (py[valid] // bin_size) * grid_w + px[valid] // bin_size
            np.add.at(total, flat, counts)
            keep = np.fromiter((gene in fixed_genes for gene in genes), dtype=bool, count=len(genes))
            if keep.any():
                genes, flat, counts = genes[keep], flat[keep], counts[keep]
                order = np.argsort(genes, kind="stable")
                genes, flat, counts = genes[order], flat[order], counts[order]
                starts = np.r_[0, np.flatnonzero(genes[1:] != genes[:-1]) + 1]
                ends = np.r_[starts[1:], len(genes)]
                for start, end in zip(starts, ends):
                    np.add.at(gene_bins[str(genes[start])], flat[start:end], counts[start:end])
            print(f"Transcript chunk {chunk_number}: total rows {total_rows:,}", flush=True)
    return gene_bins, total.reshape(grid_h, grid_w), (grid_h, grid_w), total_rows, mapped_rows


def build_fixed_modules(
    gene_bins: dict[str, np.ndarray],
    total: np.ndarray,
    shape: tuple[int, int],
    fixed_modules: dict[int, list[str]],
    minimum_total: int,
    minimum_detected: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    tissue = total >= 20
    if not tissue.any():
        raise RuntimeError("No tissue bins found with total transcript count >= 20")
    maps: dict[str, np.ndarray] = {}
    gene_rows: list[dict[str, Any]] = []
    gene_to_module = {gene: module for module, genes in fixed_modules.items() for gene in genes}
    for gene in gene_to_module:
        plane = gene_bins[gene].reshape(shape)
        total_count = int(plane.sum())
        detected = int((plane > 0).sum())
        included = total_count >= minimum_total and detected >= minimum_detected
        reason = "included"
        if included:
            smooth = ndi.gaussian_filter(np.log1p(plane.astype(np.float32)), sigma=1.0)
            values = smooth[tissue]
            std = float(values.std())
            if std <= 1e-8:
                included = False
                reason = "no spatial variation"
            else:
                maps[gene] = ((smooth - float(values.mean())) / std).astype(np.float32)
        elif total_count < minimum_total:
            reason = "below minimum total transcripts"
        else:
            reason = "below minimum detected bins"
        gene_rows.append(
            {
                "module": gene_to_module[gene],
                "gene": gene,
                "total_transcripts_in_tma": total_count,
                "detected_spatial_bins": detected,
                "included_in_tma_score": included,
                "exclusion_reason": reason,
            }
        )

    raw_modules: list[np.ndarray] = []
    display_modules: list[np.ndarray] = []
    module_rows: list[dict[str, Any]] = []
    for module, fixed_genes in fixed_modules.items():
        active = [gene for gene in fixed_genes if gene in maps]
        if not active:
            raise RuntimeError(f"Module {module} has no expressed genes in this TMA")
        raw = np.mean(np.stack([maps[gene] for gene in active]), axis=0).astype(np.float32)
        raw[~tissue] = np.nan
        values = raw[tissue]
        low, high = np.quantile(values, [0.20, 0.98])
        display = np.clip((raw - low) / max(high - low, 1e-8), 0.0, 1.0)
        display[~tissue] = 0.0
        raw_modules.append(raw)
        display_modules.append(display.astype(np.float32))
        module_rows.append(
            {
                "module": module,
                "fixed_gene_count": len(fixed_genes),
                "active_gene_count": len(active),
                "fixed_genes": ", ".join(fixed_genes),
                "active_genes": ", ".join(active),
                "missing_or_unusable_genes": ", ".join(gene for gene in fixed_genes if gene not in maps),
                "raw_score_min": float(np.min(values)),
                "raw_score_median": float(np.median(values)),
                "raw_score_max": float(np.max(values)),
            }
        )
    return np.stack(raw_modules), np.stack(display_modules), tissue, gene_rows, module_rows


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    overview = load_overview(args.overview_manifest, args.tma)
    fixed_modules = load_fixed_modules(args.fixed_gene_modules)
    fixed_genes = {gene for genes in fixed_modules.values() for gene in genes}
    gene_bins, total, shape, row_count, mapped_rows = aggregate_fixed_gene_bins(
        args.transcripts, overview, args.gene_bin_size, fixed_genes
    )
    raw, display, tissue, gene_rows, module_rows = build_fixed_modules(
        gene_bins,
        total,
        shape,
        fixed_modules,
        args.minimum_total_transcripts,
        args.minimum_detected_bins,
    )
    np.savez_compressed(
        args.output / "gene_module_scores_unclipped.npz",
        raw_modules=raw,
        modules=display,
        tissue=tissue,
        total_gene_counts=total,
    )
    write_csv(args.output / "fixed_gene_audit.csv", gene_rows)
    write_csv(args.output / "gene_modules.csv", module_rows)
    summary = {
        "tma": args.tma,
        "transcript_rows": row_count,
        "transcript_rows_mapped_to_tma_canvas": mapped_rows,
        "gene_bin_size_low_resolution_pixels": args.gene_bin_size,
        "gene_grid_shape": list(shape),
        "fixed_gene_count": len(fixed_genes),
        "active_gene_count": sum(str(row["included_in_tma_score"]).lower() == "true" for row in gene_rows),
        "gene_module_count": len(raw),
        "fixed_module_source": str(args.fixed_gene_modules),
        "fixed_module_source_sha256": sha256(args.fixed_gene_modules),
        "module_scores_clipped_for_prompt_thresholding": False,
        "annotation_files_read": [],
        "selection_used_annotation": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
