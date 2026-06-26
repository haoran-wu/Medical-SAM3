#!/usr/bin/env python3
"""Build shareable HTML for Jun10 full-description VLM reason audit."""

from __future__ import annotations

import base64
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool"
REQUEST_CSV = POOL_ROOT / "public_vlm_requests.csv"
RESULT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_VLM_FullDescriptions_VisibleReasons_167/qwen3vl32b_full_descriptions_visible_reason"
OUT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_VLM_FullDescriptions_VisibleReasons_167"
OUT_HTML = OUT_DIR / "Jun10_VLM_FullDescriptions_VisibleReasons_167_CandidateScores_SHAREABLE.html"
ROOT_COPY = ROOT / "Jun10_VLM_FullDescriptions_VisibleReasons_167_CandidateScores_SHAREABLE.html"

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


def score_table(row: dict[str, str]) -> str:
    rows = []
    for label in LABELS:
        score = row.get(label, "")
        marker = " top" if label == row.get("predicted_class") else ""
        rows.append(
            f"<tr class=\"{marker.strip()}\"><td>{esc(DISPLAY[label])}</td><td><strong>{esc(score)}</strong></td></tr>"
        )
    return (
        "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def overall_table(rows: list[dict[str, str]]) -> str:
    correct = sum(1 for row in rows if boolish(row.get("is_correct", "")))
    pred_counts = Counter(row.get("predicted_class", "") for row in rows)
    score_vectors = {tuple(row.get(label, "") for label in LABELS) for row in rows}
    return (
        "<table><tbody>"
        f"<tr><th>overall Piece Top1</th><td><strong>{correct}/{len(rows)}</strong></td></tr>"
        f"<tr><th>unique score vectors</th><td>{len(score_vectors)}/{len(rows)}</td></tr>"
        f"<tr><th>largest predicted class</th><td>{esc(pred_counts.most_common(1)[0][0])}: {pred_counts.most_common(1)[0][1]}</td></tr>"
        f"<tr><th>predicted class distribution</th><td>{esc('; '.join(f'{DISPLAY.get(k,k)}:{v}' for k,v in pred_counts.most_common()))}</td></tr>"
        "</tbody></table>"
    )


def class_table(rows: list[dict[str, str]]) -> str:
    body = []
    for label in LABELS:
        cls_rows = [row for row in rows if row.get("true_class") == label]
        correct = sum(1 for row in cls_rows if boolish(row.get("is_correct", "")))
        pred_counts = Counter(row.get("predicted_class", "") for row in cls_rows)
        body.append(
            "<tr>"
            f"<td><strong>{esc(DISPLAY[label])}</strong></td>"
            f"<td>{correct}/{len(cls_rows)}</td>"
            f"<td>{esc('; '.join(f'{DISPLAY.get(k,k)}:{v}' for k,v in pred_counts.most_common()))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>true class</th><th>Piece Top1</th><th>predicted distribution</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def prompt_block() -> str:
    system = (RESULT_DIR / "prompt_system.txt").read_text(errors="replace").strip()
    user = (RESULT_DIR / "prompt_user_template.txt").read_text(errors="replace").strip()
    return (
        "<details class=\"prompt\" open><summary>System prompt</summary>"
        f"<pre>{esc(system)}</pre></details>"
        "<details class=\"prompt\"><summary>User prompt template: full descriptions + visible reasons</summary>"
        f"<pre>{esc(user)}</pre></details>"
    )


def candidate_card(row: dict[str, str], request: dict[str, str]) -> str:
    uid = row["candidate_uid"]
    true_class = row["true_class"]
    he_path = POOL_ROOT / request["he_crop_rel"]
    fic_path = POOL_ROOT / request["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}")
    correct_cls = "ok" if boolish(row.get("is_correct", "")) else "bad"
    pred = row.get("predicted_class", "")
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
        f"{score_table(row)}"
        "<div class=\"reason\"><h4>H&amp;E evidence</h4>"
        f"<p>{esc(row.get('he_evidence', ''))}</p></div>"
        "<div class=\"reason\"><h4>FICTURE evidence</h4>"
        f"<p>{esc(row.get('ficture_evidence', ''))}</p></div>"
        "<div class=\"reason\"><h4>Visible reason for top prediction</h4>"
        f"<p>{esc(row.get('visible_reason', ''))}</p></div>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E\"><figcaption>H&amp;E gray reverse-blur candidate crop</figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE\"><figcaption>FICTURE gray reverse-blur candidate crop</figcaption></figure>"
        "</div>"
        "</div>"
        "</details>"
    )


def build_html() -> str:
    result_rows = read_csv(RESULT_DIR / "per_candidate_reason_predictions.csv")
    requests = {row["candidate_uid"]: row for row in read_csv(REQUEST_CSV)}
    if len(result_rows) != 167:
        raise RuntimeError(f"Expected 167 result rows, got {len(result_rows)}")
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        by_class[row["true_class"]].append(row)
    sections = []
    for label in LABELS:
        rows = by_class[label]
        rows.sort(key=lambda r: int(r.get("row_index", "0") or 0))
        cards = "\n".join(candidate_card(row, requests[row["candidate_uid"]]) for row in rows)
        sections.append(f"<section id=\"{label}\"><h2>{esc(DISPLAY[label])} <span>{len(rows)} candidates</span></h2>{cards}</section>")

    nav = " ".join(f"<a href=\"#{label}\">{esc(DISPLAY[label])}</a>" for label in LABELS)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.48}
header{background:#111827;color:white;padding:30px 42px}
main{max-width:1500px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px} h2{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
h2 span{font-size:15px;color:#64748b;font-weight:500;margin-left:8px}
h3{margin-top:22px} h4{margin:12px 0 8px} code,pre,.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sub{color:#cbd5e1}.muted{color:#64748b}.note{color:#334155;background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 14px;border-radius:6px}
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
.score-vector tr.top{background:#ecfeff}.ok{color:#047857;font-weight:700}.bad{color:#b91c1c;font-weight:700}
.reason{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:9px 10px;margin:10px 0}.reason h4{margin-top:0}.reason p{margin:0;color:#334155}
.figgrid{display:grid;gap:16px}.figgrid.mode-2{grid-template-columns:repeat(2,minmax(300px,1fr))}
figure{margin:0} img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white} figcaption{font-size:13px;color:#475569;margin-top:6px}
@media(max-width:1050px){.summary-grid,.body,.candidate summary,.figgrid.mode-2{grid-template-columns:1fr}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun10 VLM Full Descriptions Visible Reasons</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Jun10 VLM Full Descriptions + Visible Reasons</h1>
  <div class="sub">Qwen3-VL-32B | H&amp;E + FICTURE gray reverse-blur crops | RGB + cell type legend | 167 candidate pieces | self-contained HTML</div>
</header>
<main>
<nav>{nav}</nav>
<section>
  <h2>Experiment Design</h2>
  <p><strong>Goal:</strong> inspect why direct VLM scoring collapses or misclassifies candidates. This run keeps the full tissue-class descriptions and asks for concise visible evidence for every candidate.</p>
  <p><strong>Input:</strong> one H&amp;E crop and one official FICTURE crop per candidate. The FICTURE prompt uses RGB + cell type only.</p>
  <p><strong>Output:</strong> six tissue scores plus <code>he_evidence</code>, <code>ficture_evidence</code>, and <code>visible_reason</code>. These are short audit explanations, not hidden chain-of-thought.</p>
  <p><strong>Piece Top1:</strong> the class with the highest score is the predicted class. It is correct only if it equals the hidden annotation-derived true class.</p>
  <p class="warn"><strong>Important:</strong> this is a debugging audit, not a new final method. The reason fields help diagnose whether the model is overusing tumor-like FICTURE colors, missing H&amp;E structure, or treating small pieces without context as tumor/stroma.</p>
</section>
<section>
  <h2>Result Summary</h2>
  <div class="summary-grid"><div>{overall_table(result_rows)}</div><div>{class_table(result_rows)}</div></div>
</section>
<section>
  <h2>Prompt Used</h2>
  {prompt_block()}
</section>
<section>
  <h2>Candidate-Level Scores And Reasons</h2>
  <p class="note">Cards are grouped by hidden true class. Each card shows the two images, six scores, top prediction, and the model's short visible evidence fields.</p>
</section>
{''.join(sections)}
</main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = build_html()
    OUT_HTML.write_text(text, encoding="utf-8")
    ROOT_COPY.write_text(text, encoding="utf-8")
    for token in ['src="/Users/', 'href="/Users/', 'src="output/']:
        if token in text:
            raise RuntimeError(f"HTML contains local ref token {token}")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(f"candidate cards: {text.count('<details class=\"candidate\"')}")
    print(f"embedded images: {text.count('data:image/')}")
    print(f"size_mb: {OUT_HTML.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
