#!/usr/bin/env python3
"""Build shareable HTML for Jun11 evidence-first per-class VLM audit."""

from __future__ import annotations

import base64
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool"
REQUEST_CSV = POOL_ROOT / "public_vlm_requests.csv"
RESULT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_VLM_EvidenceFirst_PerClassReasons_167/qwen3vl32b_evidence_first_perclass_167"
OUT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_VLM_EvidenceFirst_PerClassReasons_167"
OUT_HTML = OUT_DIR / "Jun11_VLM_EvidenceFirst_PerClassReasons_167_SHAREABLE.html"
ROOT_COPY = ROOT / "Jun11_VLM_EvidenceFirst_PerClassReasons_167_SHAREABLE.html"

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


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def result_summary(rows: list[dict[str, str]]) -> str:
    correct = sum(1 for row in rows if boolish(row.get("is_correct", "")))
    pred_counts = Counter(row.get("predicted_class", "") or "PARSE_EMPTY" for row in rows)
    score_vectors = {tuple(row.get(label, "") for label in LABELS) for row in rows}
    return (
        "<table><tbody>"
        f"<tr><th>overall Piece Top1</th><td><strong>{correct}/{len(rows)}</strong></td></tr>"
        f"<tr><th>unique score vectors</th><td>{len(score_vectors)}/{len(rows)}</td></tr>"
        f"<tr><th>largest predicted class</th><td>{esc(DISPLAY.get(pred_counts.most_common(1)[0][0], pred_counts.most_common(1)[0][0]))}: {pred_counts.most_common(1)[0][1]}</td></tr>"
        f"<tr><th>predicted class distribution</th><td>{esc('; '.join(f'{DISPLAY.get(k,k)}:{v}' for k,v in pred_counts.most_common()))}</td></tr>"
        "</tbody></table>"
    )


def class_summary(rows: list[dict[str, str]]) -> str:
    body = []
    for label in LABELS:
        cls_rows = [row for row in rows if row.get("true_class") == label]
        correct = sum(1 for row in cls_rows if boolish(row.get("is_correct", "")))
        pred_counts = Counter(row.get("predicted_class", "") or "PARSE_EMPTY" for row in cls_rows)
        body.append(
            "<tr>"
            f"<td><strong>{esc(DISPLAY[label])}</strong></td>"
            f"<td>{correct}/{len(cls_rows)}</td>"
            f"<td>{esc('; '.join(f'{DISPLAY.get(k,k)}:{v}' for k,v in pred_counts.most_common()))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>hidden true class</th><th>Piece Top1</th><th>predicted distribution</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def prompt_block() -> str:
    system = (RESULT_DIR / "prompt_system.txt").read_text(errors="replace").strip()
    user = (RESULT_DIR / "prompt_user_template.txt").read_text(errors="replace").strip()
    return (
        "<details class=\"prompt\" open><summary>System prompt</summary>"
        f"<pre>{esc(system)}</pre></details>"
        "<details class=\"prompt\"><summary>User prompt template: evidence-first per-class scoring</summary>"
        f"<pre>{esc(user)}</pre></details>"
    )


def score_vector(row: dict[str, str]) -> str:
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
    return (
        "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def reason_table(row: dict[str, str]) -> str:
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


def candidate_card(row: dict[str, str], request: dict[str, str]) -> str:
    uid = row["candidate_uid"]
    true_class = row["true_class"]
    pred = row.get("predicted_class", "")
    correct_cls = "ok" if boolish(row.get("is_correct", "")) else "bad"
    he_path = POOL_ROOT / request["he_crop_rel"]
    fic_path = POOL_ROOT / request["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}")
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
        f"<tr><th>source</th><td>{esc(row.get('source', ''))}</td></tr>"
        f"<tr><th>setting / candidate</th><td>{esc(row.get('setting', ''))} / {esc(row.get('candidate_id', ''))}</td></tr>"
        "</tbody></table>"
        f"{score_vector(row)}"
        "<div class=\"reason\"><h4>Top score reason</h4>"
        f"<p>{esc(row.get('top_score_reason', ''))}</p></div>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E\"><figcaption>H&amp;E gray reverse-blur candidate crop</figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE\"><figcaption>FICTURE gray reverse-blur candidate crop</figcaption></figure>"
        "</div>"
        "</div>"
        "<div class=\"reason-block\">"
        "<h4>Evidence-first per-class scoring</h4>"
        f"{reason_table(row)}"
        "</div>"
        "</details>"
    )


def build_html() -> str:
    rows = read_csv(RESULT_DIR / "per_candidate_evidence_first_predictions.csv")
    requests = {row["candidate_uid"]: row for row in read_csv(REQUEST_CSV)}
    if len(rows) != 167:
        raise RuntimeError(f"Expected 167 result rows, got {len(rows)}")
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["true_class"]].append(row)
    sections = []
    for label in LABELS:
        class_rows = sorted(by_class[label], key=lambda item: int(item.get("run_order") or item.get("row_index") or 0))
        cards = "\n".join(candidate_card(row, requests[row["candidate_uid"]]) for row in class_rows)
        sections.append(f"<section id=\"{label}\"><h2>{esc(DISPLAY[label])} <span>{len(class_rows)} candidates</span></h2>{cards}</section>")
    nav = " ".join(f"<a href=\"#{label}\">{esc(DISPLAY[label])}</a>" for label in LABELS)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.5}
header{background:#111827;color:white;padding:30px 42px}
main{max-width:1600px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px} h2{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
h2 span{font-size:15px;color:#64748b;font-weight:500;margin-left:8px} h4{margin:12px 0 8px}
code,pre,.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sub{color:#cbd5e1}.note{color:#334155;background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 14px;border-radius:6px}
.warn{background:#fff7ed;border-left:5px solid #f97316;padding:13px 15px;border-radius:7px;color:#431407}
nav{position:sticky;top:0;background:rgba(248,250,252,.96);backdrop-filter:blur(8px);padding:12px 0;z-index:10;border-bottom:1px solid #e5e7eb}
nav a{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;border:1px solid #cbd5e1;border-radius:999px;color:#0f172a;text-decoration:none;background:white;font-size:14px}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}
th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f1f5f9}
.summary-grid{display:grid;grid-template-columns:minmax(300px,1fr) minmax(300px,1fr);gap:22px}
details.prompt{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}
details.prompt summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}
pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:14px;margin:0;border-radius:0 0 8px 8px;max-height:440px;overflow:auto;font-size:12px}
.candidate{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0;overflow:hidden}
.candidate summary{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.15fr .65fr 1fr .55fr;gap:10px;align-items:center;background:#fff}
.candidate summary:hover{background:#f8fafc}.uid{font-weight:700;color:#1d4ed8}
.body{display:grid;grid-template-columns:470px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.meta,.score-vector{font-size:13px;margin:0 0 12px}.meta th,.meta td,.score-vector th,.score-vector td{padding:5px 7px}.meta th{width:160px}
.score-vector tr.pred,.reason-table tr.pred{background:#fff7ed}.score-vector tr.truth,.reason-table tr.truth{box-shadow:inset 4px 0 0 #2563eb}
.ok{color:#047857;font-weight:700}.bad{color:#b91c1c;font-weight:700}.score{color:#475569;font-size:12px}
.reason{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:9px 10px;margin:10px 0}.reason h4{margin-top:0}.reason p{margin:0;color:#334155}
.reason-block{padding:0 14px 14px}.reason-table{font-size:13px}.reason-table th:nth-child(1){width:145px}.reason-table th:nth-child(4){width:25%}
.figgrid{display:grid;gap:16px}.figgrid.mode-2{grid-template-columns:repeat(2,minmax(300px,1fr))}
figure{margin:0} img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white} figcaption{font-size:13px;color:#475569;margin-top:6px}
@media(max-width:1100px){.summary-grid,.body,.candidate summary,.figgrid.mode-2{grid-template-columns:1fr}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun11 VLM Evidence-First Per-Class Reasons</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Jun11 VLM Evidence-First Per-Class Reasons</h1>
  <div class="sub">Qwen3-VL-32B | H&amp;E + FICTURE gray reverse-blur crops | RGB + cell type legend only | 167 candidate pieces | self-contained HTML</div>
</header>
<main>
<nav>{nav}</nav>
<section>
  <h2>Experiment Design</h2>
  <p><strong>Goal:</strong> force the VLM to bind each class score to visible evidence, so errors can be audited class by class.</p>
  <p><strong>What changed from the previous visible-reason run:</strong> the previous run had one global reason for the top prediction. This run has H&amp;E evidence, FICTURE evidence, missing/against evidence, and a score reason for every tissue class.</p>
  <p><strong>How to read a wrong case:</strong> compare the orange predicted-class row with the blue hidden-true-class row. For example, if a bronchiola piece is predicted as tumor, the tumor row explains why tumor is high and the bronchiola row explains why bronchiola is low.</p>
  <p><strong>FICTURE prompt:</strong> RGB + cell type only. Major Compartment is not included.</p>
  <p class="warn"><strong>Important:</strong> these are visible evidence statements produced during scoring. They are not hidden chain-of-thought and are not written after the score by a separate summarizer.</p>
</section>
<section>
  <h2>Result Summary</h2>
  <div class="summary-grid"><div>{result_summary(rows)}</div><div>{class_summary(rows)}</div></div>
</section>
<section>
  <h2>Prompt</h2>
  {prompt_block()}
</section>
{''.join(sections)}
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
