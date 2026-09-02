#!/usr/bin/env python3
"""Freeze binary-pixel hashes for the accepted 55-candidate TMA39 pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_NAMES = {
    "FICTURE-SAM": "FICTURE primary",
    "H&E-SAM rescue": "Same-prompt H&E supplement",
    "H&E-SAM independent gap": "Independent H&E supplement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--prompt-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_record(path: Path, source: str) -> dict[str, object]:
    pixels = (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    return {
        "source": SOURCE_NAMES[source],
        "area_pixels": int(pixels.sum()),
        "binary_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    with args.reference_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        source = row["source"]
        if source not in SOURCE_NAMES:
            raise RuntimeError(f"Unexpected reference source: {source}")
        records.append(mask_record(Path(row["mask_path"]), source))
    records.sort(key=lambda item: (str(item["source"]), str(item["binary_pixel_sha256"])))
    counts = Counter(str(item["source"]) for item in records)
    manifest = {
        "workflow_id": "tma39-fixed-sam3-v1",
        "tma": "TMA39",
        "reference_candidate_count": len(records),
        "reference_source_counts": dict(sorted(counts.items())),
        "prompt_csv_sha256": file_sha256(args.prompt_csv),
        "comparison": "unordered source plus full-resolution binary-pixel hash",
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
