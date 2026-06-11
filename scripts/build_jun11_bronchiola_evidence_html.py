#!/usr/bin/env python3
"""Build bronchiola-only evidence-first VLM audit HTML."""

from __future__ import annotations

import base64
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool"
REQUEST_CSV = POOL_ROOT / "public_vlm_requests.csv"
RESULT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_Bronchiola_VLM_EvidenceFirst_PerClassReasons"
OUT_HTML = RESULT_DIR / "Jun11_Bronchiola_VLM_EvidenceFirst_PerClassReasons_SHAREABLE.html"
ROOT_COPY = ROOT / "Jun11_Bronchiola_VLM_EvidenceFirst_PerClassReasons_SHAREABLE.html"

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
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def score_table(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    body = []
    for label in LABELS:
        cls = []
        if label == pred:
            cls.append("pred")
        if label == true:
            cls.append("truth")
        body.append(
            f"<tr class=\"{' '.join(cls)}\"><td>{esc(DISPLAY[label])}</td><td><strong>{esc(row.get(label, ''))}</strong></td></tr>"
        )
    return "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def evidence_table(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    body = []
    for label in LABELS:
        cls = []
        if label == pred:
            cls.append("pred")
        if label == true:
            cls.append("truth")
        body.append(
            "<tr class=\"" + " ".join(cls) + "\">"
            f"<td><strong>{esc(DISPLAY[label])}</strong><br><span class=\"score\">score {esc(row.get(label, ''))}</span></td>"
            f"<td><strong>support:</strong> {esc(row.get(label + '_he_support', ''))}<br>"
            f"<strong>against/missing:</strong> {esc(row.get(label + '_he_against_or_missing', ''))}</td>"
            f"<td><strong>support:</strong> {esc(row.get(label + '_ficture_support', ''))}<br>"
            f"<strong>against/missing:</strong> {esc(row.get(label + '_ficture_against_or_missing', ''))}</td>"
            f"<td>{esc(row.get(label + '_score_reason', ''))}</td>"
            "</tr>"
        )
    return (
        "<table class=\"evidence\"><thead><tr><th>class</th><th>H&amp;E evidence</th><th>FICTURE evidence</th><th>score reason</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def candidate_card(row: dict[str, str], request: dict[str, str]) -> str:
    uid = row["candidate_uid"]
    pred = row.get("predicted_class", "")
    correct_cls = "ok" if boolish(row.get("is_correct", "")) else "bad"
    he_path = POOL_ROOT / request["he_crop_rel"]
    fic_path = POOL_ROOT / request["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(uid)
    return f"""
<details class="candidate" open>
<summary>
  <span class="uid">{esc(uid)}</span>
  <span>true: <strong>bronchiola</strong></span>
  <span>pred: <strong class="{correct_cls}">{esc(DISPLAY.get(pred, pred))}</strong> ({esc(row.get('top_score', ''))})</span>
  <span>correct: <strong class="{correct_cls}">{esc(row.get('is_correct', ''))}</strong></span>
</summary>
<div class="body">
  <div class="leftcol">
    <table class="meta"><tbody>
      <tr><th>hidden true class</th><td>bronchiola</td></tr>
      <tr><th>predicted class</th><td class="{correct_cls}">{esc(DISPLAY.get(pred, pred))}</td></tr>
      <tr><th>top score / tie</th><td>{esc(row.get('top_score', ''))} / {esc(row.get('top_score_tie', ''))}</td></tr>
      <tr><th>source</th><td>{esc(row.get('source', ''))}</td></tr>
      <tr><th>setting / candidate</th><td>{esc(row.get('setting', ''))} / {esc(row.get('candidate_id', ''))}</td></tr>
    </tbody></table>
    {score_table(row)}
    <div class="reason"><h4>Top score reason</h4><p>{esc(row.get('top_score_reason', ''))}</p></div>
  </div>
  <div class="figgrid">
    <figure><img src="{data_uri(he_path)}" alt="{esc(uid)} H&amp;E"><figcaption>H&amp;E gray reverse-blur crop</figcaption></figure>
    <figure><img src="{data_uri(fic_path)}" alt="{esc(uid)} FICTURE"><figcaption>FICTURE gray reverse-blur crop</figcaption></figure>
  </div>
</div>
<div class="evidence-wrap">
  <h4>Per-class scoring reasons</h4>
  {evidence_table(row)}
</div>
</details>
"""


def prompt_block() -> str:
    system = (RESULT_DIR / "prompt_system.txt").read_text(errors="replace").strip()
    user = (RESULT_DIR / "prompt_user_template.txt").read_text(errors="replace").strip()
    return (
        "<details class=\"prompt\" open><summary>System prompt</summary>"
        f"<pre>{esc(system)}</pre></details>"
        "<details class=\"prompt\"><summary>User prompt template</summary>"
        f"<pre>{esc(user)}</pre></details>"
    )


def build_html() -> str:
    rows = [row for row in read_csv(RESULT_DIR / "per_candidate_evidence_first_predictions.csv") if row.get("true_class") == "bronchiola"]
    rows.sort(key=lambda row: int(row.get("run_order") or row.get("row_index") or 0))
    requests = {row["candidate_uid"]: row for row in read_csv(REQUEST_CSV)}
    if len(rows) != 14:
        raise RuntimeError(f"Expected 14 bronchiola rows, got {len(rows)}")
    correct = sum(1 for row in rows if boolish(row.get("is_correct", "")))
    pred_counts: dict[str, int] = {}
    for row in rows:
        pred_counts[row.get("predicted_class", "") or "PARSE_EMPTY"] = pred_counts.get(row.get("predicted_class", "") or "PARSE_EMPTY", 0) + 1
    cards = "\n".join(candidate_card(row, requests[row["candidate_uid"]]) for row in rows)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.5}
header{background:#111827;color:white;padding:30px 42px}
main{max-width:1600px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px}.sub{color:#cbd5e1}.warn{background:#fff7ed;border-left:5px solid #f97316;padding:13px 15px;border-radius:7px;color:#431407}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}
th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f1f5f9}
details.prompt{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}
details.prompt summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}
pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:14px;margin:0;border-radius:0 0 8px 8px;max-height:440px;overflow:auto;font-size:12px}
.candidate{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0;overflow:hidden}
.candidate summary{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.15fr .65fr 1fr .55fr;gap:10px;align-items:center;background:#fff}
.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-weight:700;color:#1d4ed8}
.body{display:grid;grid-template-columns:470px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.meta,.score-vector{font-size:13px;margin:0 0 12px}.meta th,.meta td,.score-vector th,.score-vector td{padding:5px 7px}.meta th{width:160px}
.score-vector tr.pred,.evidence tr.pred{background:#fff7ed}.score-vector tr.truth,.evidence tr.truth{box-shadow:inset 4px 0 0 #2563eb}
.ok{color:#047857;font-weight:700}.bad{color:#b91c1c;font-weight:700}.score{color:#475569;font-size:12px}
.reason{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:9px 10px;margin:10px 0}.reason h4{margin-top:0}.reason p{margin:0;color:#334155}
.figgrid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:16px}figure{margin:0}img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white}figcaption{font-size:13px;color:#475569;margin-top:6px}
.evidence-wrap{padding:0 14px 14px}.evidence{font-size:13px}.evidence th:nth-child(1){width:145px}.evidence th:nth-child(4){width:25%}
@media(max-width:1100px){.body,.candidate summary,.figgrid{grid-template-columns:1fr}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun11 Bronchiola VLM Evidence-First Per-Class Reasons</title><style>{css}</style></head>
<body><header><h1>Jun11 Bronchiola VLM Evidence-First Per-Class Reasons</h1>
<div class="sub">Qwen3-VL-32B | 14 bronchiola candidates | H&amp;E + FICTURE gray reverse-blur | RGB + cell type only | self-contained HTML</div></header>
<main>
<section><h2>Summary</h2>
<table><tbody>
<tr><th>Bronchiola Piece Top1</th><td><strong>{correct}/14</strong></td></tr>
<tr><th>Predicted distribution</th><td>{esc('; '.join(f'{DISPLAY.get(k,k)}:{v}' for k,v in sorted(pred_counts.items(), key=lambda x:-x[1])))}</td></tr>
</tbody></table>
<p class="warn"><strong>How to read this:</strong> each candidate is truly bronchiola by the hidden annotation-derived label. The orange row is the VLM predicted class, and the blue-left-border row is the hidden true bronchiola class. The per-class evidence table explains why each score was assigned during scoring.</p>
</section>
<section><h2>Prompt</h2>{prompt_block()}</section>
<section><h2>Bronchiola Candidates</h2>{cards}</section>
</main></body></html>"""


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    text = build_html()
    OUT_HTML.write_text(text, encoding="utf-8")
    ROOT_COPY.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(f"cards {text.count('<details class=\"candidate\"')}")
    print(f"images {text.count('data:image/')}")
    print(f"size_mb {OUT_HTML.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
