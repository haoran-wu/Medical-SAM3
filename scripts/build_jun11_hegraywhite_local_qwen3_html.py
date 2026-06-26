#!/usr/bin/env python3
"""Build shareable HTML for Jun11 H&E grayscale-white local Qwen audit."""

from __future__ import annotations

import base64
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_GPT55_HEGrayWhite_EvidenceFirst_3/corrected_pool"
OUT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_HEGrayWhite_LocalQwen_EvidenceFirst_3"
RESULT_32B = OUT_DIR / "Jun11_hegraywhite_evidence3_qwen3vl32b_14806356"
RESULT_8B = OUT_DIR / "Jun11_hegraywhite_evidence3_qwen3vl8b_14806748"
OUT_HTML = OUT_DIR / "Jun11_HEGrayWhite_LocalQwen_EvidenceFirst_3_SHAREABLE.html"
ROOT_COPY = ROOT / "Jun11_HEGrayWhite_LocalQwen_EvidenceFirst_3_SHAREABLE.html"

LABELS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def score_vector(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    body = []
    for label in LABELS:
        classes = []
        if label == pred:
            classes.append("pred")
        if label == true:
            classes.append("truth")
        body.append(
            f"<tr class=\"{' '.join(classes)}\"><td>{esc(DISPLAY[label])}</td><td><strong>{esc(row.get(label, ''))}</strong></td></tr>"
        )
    return (
        "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def reason_table(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    body = []
    for label in LABELS:
        classes = []
        if label == pred:
            classes.append("pred")
        if label == true:
            classes.append("truth")
        body.append(
            "<tr class=\"" + " ".join(classes) + "\">"
            f"<td><strong>{esc(DISPLAY[label])}</strong><br><span class=\"score\">score {esc(row.get(label, ''))}</span></td>"
            f"<td><strong>H&amp;E support:</strong> {esc(row.get(label + '_he_support', ''))}<br>"
            f"<strong>H&amp;E missing/against:</strong> {esc(row.get(label + '_he_against_or_missing', ''))}</td>"
            f"<td><strong>FICTURE support:</strong> {esc(row.get(label + '_ficture_support', ''))}<br>"
            f"<strong>FICTURE missing/against:</strong> {esc(row.get(label + '_ficture_against_or_missing', ''))}</td>"
            f"<td>{esc(row.get(label + '_score_reason', ''))}</td>"
            "</tr>"
        )
    return (
        "<table class=\"reason-table\"><thead><tr><th>class</th><th>H&amp;E evidence</th><th>FICTURE evidence</th><th>score reason</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def comparison_table(rows32: list[dict[str, str]], rows8: list[dict[str, str]]) -> str:
    by8 = {row["candidate_uid"]: row for row in rows8}
    body = []
    for row in rows32:
        uid = row["candidate_uid"]
        r8 = by8.get(uid, {})
        body.append(
            "<tr>"
            f"<td><code>{esc(uid)}</code></td>"
            f"<td>{esc(row.get('true_class'))}</td>"
            f"<td><strong>{esc(row.get('predicted_class'))}</strong></td>"
            f"<td>{' / '.join(esc(row.get(label, '')) for label in LABELS)}</td>"
            f"<td><strong>{esc(r8.get('predicted_class', ''))}</strong></td>"
            f"<td>{' / '.join(esc(r8.get(label, '')) for label in LABELS) if r8 else ''}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>candidate</th><th>true</th><th>Qwen3-VL-32B pred</th><th>32B scores<br>bron/alv/ves/tum/stro/imm</th><th>Qwen3-VL-8B pred</th><th>8B scores<br>bron/alv/ves/tum/stro/imm</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def candidate_card(row: dict[str, str], request: dict[str, str]) -> str:
    uid = row["candidate_uid"]
    true_class = row["true_class"]
    pred = row.get("predicted_class", "")
    correct_cls = "ok" if boolish(row.get("is_correct", "")) else "bad"
    he_path = POOL_ROOT / request["he_crop_rel"]
    fic_path = POOL_ROOT / request["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}: {he_path}, {fic_path}")
    return (
        "<details class=\"candidate\" open>"
        "<summary>"
        f"<span class=\"uid\">{esc(uid)}</span>"
        f"<span>true: <strong>{esc(DISPLAY[true_class])}</strong></span>"
        f"<span>pred: <strong class=\"{correct_cls}\">{esc(DISPLAY.get(pred, pred))}</strong> ({esc(row.get('top_score', ''))})</span>"
        f"<span>correct: <strong class=\"{correct_cls}\">{esc(row.get('is_correct', ''))}</strong></span>"
        "</summary>"
        "<div class=\"body\">"
        "<div class=\"leftcol\">"
        "<table class=\"meta\"><tbody>"
        f"<tr><th>hidden true class</th><td>{esc(DISPLAY[true_class])}</td></tr>"
        f"<tr><th>predicted class</th><td class=\"{correct_cls}\">{esc(DISPLAY.get(pred, pred))}</td></tr>"
        f"<tr><th>top score / tie</th><td>{esc(row.get('top_score', ''))} / {esc(row.get('top_score_tie', ''))}</td></tr>"
        f"<tr><th>setting / candidate</th><td>{esc(row.get('setting', ''))} / {esc(row.get('candidate_id', ''))}</td></tr>"
        "</tbody></table>"
        f"{score_vector(row)}"
        "<div class=\"reason\"><h4>Top score reason</h4>"
        f"<p>{esc(row.get('top_score_reason', ''))}</p></div>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E\"><figcaption>H&amp;E grayscale morphology, white outside candidate</figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE\"><figcaption>FICTURE gray reverse-blur candidate crop</figcaption></figure>"
        "</div>"
        "</div>"
        "<div class=\"reason-block\">"
        "<h4>Evidence-first per-class scoring from Qwen3-VL-32B</h4>"
        f"{reason_table(row)}"
        "</div>"
        "</details>"
    )


def build_html() -> str:
    requests = {row["candidate_uid"]: row for row in read_csv(POOL_ROOT / "public_vlm_requests.csv")}
    rows32 = read_csv(RESULT_32B / "per_candidate_evidence_first_predictions.csv")
    rows8 = read_csv(RESULT_8B / "per_candidate_evidence_first_predictions.csv")
    prompt = (RESULT_32B / "prompt_user_template.txt").read_text(errors="replace").strip()
    system = (RESULT_32B / "prompt_system.txt").read_text(errors="replace").strip()
    cards = "\n".join(candidate_card(row, requests[row["candidate_uid"]]) for row in rows32)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.5}
header{background:#111827;color:white;padding:30px 42px} main{max-width:1500px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px} h2{margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
.sub{color:#cbd5e1}.warn{background:#fff7ed;border-left:5px solid #f97316;padding:13px 15px;border-radius:7px;color:#431407}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}
th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f1f5f9}
code,pre,.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
details.prompt{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}
details.prompt summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}
pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:14px;margin:0;border-radius:0 0 8px 8px;max-height:440px;overflow:auto;font-size:12px}
.candidate{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:18px 0;overflow:hidden}
.candidate summary{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.2fr .7fr 1fr .6fr;gap:10px;align-items:center;background:#fff}
.uid{font-weight:700;color:#1d4ed8}.body{display:grid;grid-template-columns:440px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.meta,.score-vector{font-size:13px;margin:0 0 12px}.meta th,.meta td,.score-vector th,.score-vector td{padding:5px 7px}.meta th{width:150px}
.score-vector tr.pred,.reason-table tr.pred{background:#fff7ed}.score-vector tr.truth,.reason-table tr.truth{box-shadow:inset 4px 0 0 #2563eb}
.ok{color:#047857;font-weight:700}.bad{color:#b91c1c;font-weight:700}.score{color:#475569;font-size:12px}
.reason{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:9px 10px;margin:10px 0}.reason h4{margin-top:0}.reason p{margin:0;color:#334155}
.reason-block{padding:0 14px 14px}.reason-table{font-size:13px}.reason-table th:nth-child(1){width:145px}.reason-table th:nth-child(4){width:25%}
.figgrid{display:grid;gap:16px}.figgrid.mode-2{grid-template-columns:repeat(2,minmax(320px,1fr))}
figure{margin:0} img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white} figcaption{font-size:13px;color:#475569;margin-top:6px}
@media(max-width:1100px){.body,.candidate summary,.figgrid.mode-2{grid-template-columns:1fr}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun11 H&E Grayscale-White Local Qwen Evidence-First Audit</title><style>{css}</style></head>
<body>
<header>
  <h1>Jun11 H&amp;E Grayscale-White Local Qwen Evidence-First Audit</h1>
  <div class="sub">3 bronchiola candidates | local Qwen3-VL-32B main result + Qwen3-VL-8B check | self-contained HTML</div>
</header>
<main>
<section>
  <h2>What This Tests</h2>
  <p><strong>Question:</strong> if H&amp;E color is removed and candidate background is pure white, does the local VLM stop confusing H&amp;E staining with FICTURE tumor colors?</p>
  <p><strong>Input:</strong> Image 1 is H&amp;E grayscale morphology with white background outside the candidate mask. Image 2 is the original FICTURE gray reverse-blur crop.</p>
  <p><strong>Prompt constraint:</strong> the FICTURE RGB legend applies only to Image 2. The model is explicitly forbidden from using H&amp;E grayscale/staining as FICTURE RGB/cell-type evidence.</p>
  <p class="warn"><strong>Observed result:</strong> both Qwen3-VL-8B and Qwen3-VL-32B still predicted all three hidden bronchiola candidates as tumor. The generated reasons still cite FICTURE tumor-cell dominance, so the failure is not fully solved by removing H&amp;E color.</p>
</section>
<section>
  <h2>Result Summary</h2>
  {comparison_table(rows32, rows8)}
</section>
<section>
  <h2>Prompt</h2>
  <details class="prompt"><summary>System prompt</summary><pre>{esc(system)}</pre></details>
  <details class="prompt" open><summary>User prompt template</summary><pre>{esc(prompt)}</pre></details>
</section>
<section>
  <h2>Candidate-Level Results</h2>
  {cards}
</section>
</main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_text = build_html()
    OUT_HTML.write_text(html_text, encoding="utf-8")
    ROOT_COPY.write_text(html_text, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(f"candidate cards: {html_text.count('<details class=\"candidate\"')}")
    print(f"embedded images: {html_text.count('data:image/')}")
    print(f"size_mb: {OUT_HTML.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
