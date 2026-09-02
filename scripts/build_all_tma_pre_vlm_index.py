#!/usr/bin/env python3
"""Build the presentation entry page for all fixed-workflow pre-VLM TMA reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from tma39_report_style import REPORT_TEMPLATE_ID, STYLE as REFERENCE_STYLE


TMAS = ("TMA07", "TMA24", "TMA29", "TMA30", "TMA31", "TMA34", "TMA36", "TMA39", "TMA41", "TMA42")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = []
    for tma in TMAS:
        run = args.root / tma
        prompt = read_json(run / "prompts/final_sustained_edge3/adaptive_prompt_manifest.json")
        pool = read_json(run / "frozen_pool/run_manifest.json")
        validation = read_json(run / "pre_vlm_validation.json")["run"]
        report = args.reports / tma / f"{tma}_Pre_VLM.html"
        if validation["status"] != "valid" or validation["validated_stage"] != "pre-vlm":
            raise RuntimeError(f"{tma} has not passed pre-VLM validation")
        if not report.is_file() or report.stat().st_size == 0:
            raise FileNotFoundError(report)
        counts = pool["final_source_counts"]
        rows.append(
            {
                "tma": tma,
                "prompt_count": int(prompt["prompt_count"]),
                "ficture": int(counts["FICTURE primary"]),
                "same_he": int(counts["Same-prompt H&E supplement"]),
                "independent_he": int(counts["Independent H&E supplement"]),
                "final": int(pool["final_candidate_count"]),
                "dedup_removed": int(pool["removed_as_near_duplicate"]),
                "report_bytes": report.stat().st_size,
                "status": "Valid",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        f"""<a class='card' href='{row['tma']}/{row['tma']}_Pre_VLM.html'>
        <div class='top'><h2>{row['tma']}</h2><span>VALID</span></div>
        <p><b>{row['final']}</b> frozen candidates before VLM</p>
        <dl><dt>GeneMap prompts</dt><dd>{row['prompt_count']}</dd><dt>FICTURE primary</dt><dd>{row['ficture']}</dd><dt>Same-prompt H&amp;E</dt><dd>{row['same_he']}</dd><dt>Independent H&amp;E</dt><dd>{row['independent_he']}</dd></dl>
        <div class='open'>Open the complete {row['tma']} report</div></a>"""
        for row in rows
    )
    table_rows = "".join(
        f"<tr><td><a href='{row['tma']}/{row['tma']}_Pre_VLM.html'>{row['tma']}</a></td><td>{row['prompt_count']}</td><td>{row['ficture']}</td><td>{row['same_he']}</td><td>{row['independent_he']}</td><td><b>{row['final']}</b></td><td>{row['dedup_removed']}</td><td>{row['status']}</td></tr>"
        for row in rows
    )
    total_candidates = sum(row["final"] for row in rows)
    extra_style = r"""
.report-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:20px 0}.card{border:1px solid var(--line);color:inherit;text-decoration:none;padding:16px;min-height:235px;display:flex;flex-direction:column;background:#fff;border-radius:5px}.card:hover{border-color:var(--teal);box-shadow:0 3px 12px rgba(0,0,0,.08)}.top{display:flex;align-items:center;justify-content:space-between}.top h2{margin:0;font-size:25px}.top span{font-size:11px;font-weight:bold;color:#0b6b43;background:#e9f6ef;padding:5px 7px}.card p b{font-size:25px}dl{display:grid;grid-template-columns:1fr auto;gap:5px 12px;margin:4px 0 18px}dt{color:var(--muted)}dd{margin:0;font-weight:bold}.open{margin-top:auto;border-top:1px solid var(--line);padding-top:10px;color:var(--teal);font-weight:bold}td a{color:var(--teal);font-weight:bold}@media(max-width:1050px){.report-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:700px){.report-grid{grid-template-columns:1fr}}
"""
    content = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='report-template' content='{REPORT_TEMPLATE_ID}'><link rel='icon' href='data:,'><title>All TMA Pre-VLM Results</title><style>{REFERENCE_STYLE}{extra_style}</style></head><body><header><p class='eyebrow'>SAM3 · fixed TMA39 workflow</p><h1>All TMA Pre-VLM Results</h1><p class='lede'>Each sample uses its registered H&amp;E and strict-official raw FICTURE with the same fixed GeneMap modules, box-plus-point rule, SAM3 selection, H&amp;E supplements, and candidate cleanup.</p><div class='metrics'><div class='metric'><b>{len(rows)}</b><span>TMAs completed</span></div><div class='metric'><b>{total_candidates}</b><span>frozen candidates</span></div><div class='metric'><b>120</b><span>fixed genes</span></div><div class='metric'><b>9</b><span>fixed modules</span></div><div class='metric'><b>1</b><span>shared parameter set</span></div><div class='metric'><b>0</b><span>VLM calls here</span></div></div><nav class='top-nav'><a href='#reports'>TMA reports</a><a href='#summary'>Summary table</a></nav></header><section id='reports'><h2>Open one TMA</h2><div class='report-grid'>{cards}</div></section><section id='summary'><h2>One table for all samples</h2><div class='table-wrap'><table><thead><tr><th>TMA</th><th>GeneMap prompts</th><th>FICTURE primary</th><th>Same-prompt H&amp;E</th><th>Independent H&amp;E</th><th>Final candidates</th><th>Duplicates removed</th><th>Audit</th></tr></thead><tbody>{table_rows}</tbody></table></div><div class='note'><b>Fixed across all samples:</b> exact TMA39 H&amp;E foreground support; fixed gene-to-module membership; tight 0% box plus peak point; SAM3-base; 40% new-area rule for paired H&amp;E; fixed independent H&amp;E detector; IoU 0.90 duplicate removal; no mask union.</div></section></body></html>"""
    args.output.write_text(content, encoding="utf-8")
    args.summary_json.write_text(json.dumps({"workflow_id": "tma39-fixed-sam3-v1", "report_template": REPORT_TEMPLATE_ID, "tma_count": len(rows), "total_final_candidates": total_candidates, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tmas": len(rows), "total_candidates": total_candidates}, indent=2))


if __name__ == "__main__":
    main()
