#!/usr/bin/env python3
"""Audit all self-contained TMA reports against the fixed presentation schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


TMAS = ("TMA07", "TMA24", "TMA29", "TMA30", "TMA31", "TMA34", "TMA36", "TMA39", "TMA41", "TMA42")
TEMPLATE = "tma39-sam3-presentation-v1"
SECTION_IDS = ("selection", "framework", "pool", "candidates", "audit")
REQUIRED_TEXT = (
    "K=12 strict-official raw FICTURE",
    "GeneMap prompt regions",
    "Actual boxes and points",
    "All nine GeneMap modules",
    "module score distributions",
    "Same-prompt H&amp;E supplements",
    "Independent H&amp;E supplements",
    "All final candidates on registered H&amp;E",
    "candidate inputs ready for VLM",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    audits = []
    for tma in TMAS:
        report = args.reports / tma / f"{tma}_Pre_VLM.html"
        text = report.read_text(encoding="utf-8")
        errors = []
        template_match = re.search(r"<meta name='report-template' content='([^']+)'>", text)
        template = template_match.group(1) if template_match else ""
        if template != TEMPLATE:
            errors.append(f"template is {template!r}")
        section_ids = tuple(re.findall(r"<section id='([^']+)'", text))
        if section_ids != SECTION_IDS:
            errors.append(f"section order is {section_ids}")
        if f"<p class='eyebrow'>{tma} · SAM3 · pre-VLM report</p>" not in text:
            errors.append("TMA identity missing from header")
        for required in REQUIRED_TEXT:
            if required not in text:
                errors.append(f"missing text: {required}")
        if re.search(r"<(?:img|script|link)[^>]+(?:src|href)='https?://", text):
            errors.append("report contains an external asset")
        candidate_match = re.search(r"const candidates=(\[.*?\]);\nconst list=", text)
        if not candidate_match:
            errors.append("candidate payload missing")
            candidate_count = -1
            candidates = []
        else:
            candidates = json.loads(candidate_match.group(1))
            candidate_count = len(candidates)
            for candidate in candidates:
                module = candidate["module"]
                expected_module = "" if module == "none" else f", Module {module}"
                if not candidate["source_label"].endswith(expected_module):
                    errors.append(f"candidate module label mismatch: {candidate['id']}")
                    break
        expected_count = csv_count(args.root / tma / "frozen_pool/final_pool.csv")
        if candidate_count != expected_count:
            errors.append(f"candidate payload {candidate_count} != frozen pool {expected_count}")
        module_cards = text.count("class='module-card'")
        if module_cards != 9:
            errors.append(f"module card count is {module_cards}")
        data_image_count = text.count("data:image/")
        minimum_images = 15 + 2 * expected_count
        if data_image_count < minimum_images:
            errors.append(f"only {data_image_count} embedded images; expected at least {minimum_images}")
        audits.append(
            {
                "tma": tma,
                "report": str(report),
                "report_sha256": sha256(report),
                "report_bytes": report.stat().st_size,
                "template": template,
                "section_ids": list(section_ids),
                "candidate_count": candidate_count,
                "module_card_count": module_cards,
                "embedded_image_references": data_image_count,
                "errors": errors,
                "status": "passed" if not errors else "failed",
            }
        )
    result = {
        "template": TEMPLATE,
        "tma_count": len(audits),
        "passed": sum(row["status"] == "passed" for row in audits),
        "audits": audits,
        "status": "passed" if all(row["status"] == "passed" for row in audits) else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
