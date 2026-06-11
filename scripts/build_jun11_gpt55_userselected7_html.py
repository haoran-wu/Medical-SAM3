#!/usr/bin/env python3
"""Build shareable HTML for Jun11 GPT-5.5 user-selected evidence audit."""

from __future__ import annotations

import base64
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "Jun11_GPT55_OldHE_EvidenceFirst_UserSelected20"
OUT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables" / RUN_NAME
POOL_ROOT = OUT_DIR / "corrected_pool"
RESULT_DIR = OUT_DIR / "openrouter_gpt55_evidence"
OUT_HTML = OUT_DIR / f"{RUN_NAME}_SHAREABLE.html"
ROOT_COPY = ROOT / f"{RUN_NAME}_SHAREABLE.html"

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


def truth_class_from_label(label: str) -> str:
    if label == "lung_bronchiola":
        return "bronchiola"
    if label == "lung_alveoli_normal_adjacent":
        return "alveoli"
    if label == "lung_vessels":
        return "vessels"
    return label


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def score_vector(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    rows = []
    for label in LABELS:
        classes = []
        if label == pred:
            classes.append("pred")
        if label == true:
            classes.append("truth")
        rows.append(
            f"<tr class=\"{' '.join(classes)}\"><td>{esc(DISPLAY[label])}</td>"
            f"<td><strong>{esc(row.get(label, ''))}</strong></td></tr>"
        )
    return (
        "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def reason_table(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    rows = []
    for label in LABELS:
        classes = []
        if label == pred:
            classes.append("pred")
        if label == true:
            classes.append("truth")
        rows.append(
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
        "<table class=\"reason-table\"><thead><tr><th>class</th><th>H&amp;E evidence</th>"
        "<th>FICTURE evidence</th><th>score reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def summary_table(rows: list[dict[str, str]], hidden: dict[str, dict[str, str]]) -> str:
    body = []
    for row in rows:
        uid = row["candidate_uid"]
        h = hidden.get(uid, {})
        body.append(
            "<tr>"
            f"<td><code>{esc(uid)}</code><br><span class=\"muted\">{esc(h.get('selection_note', ''))}</span></td>"
            f"<td>{esc(DISPLAY.get(row.get('true_class', ''), row.get('true_class', '')))}</td>"
            f"<td><strong>{esc(DISPLAY.get(row.get('predicted_class', ''), row.get('predicted_class', '')))}</strong></td>"
            f"<td class=\"{'ok' if boolish(row.get('is_correct')) else 'bad'}\">{esc(row.get('is_correct'))}</td>"
            f"<td>{esc(h.get('component_dice', ''))} / {esc(h.get('component_precision', ''))} / {esc(h.get('component_recall', ''))}</td>"
            f"<td>{' / '.join(esc(row.get(label, '')) for label in LABELS)}</td>"
            f"<td>{esc(row.get('top_score_reason', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>candidate</th><th>true</th><th>GPT-5.5 pred</th><th>correct</th>"
        "<th>hidden component<br>Dice / P / R</th><th>scores<br>bron/alv/ves/tum/stro/imm</th>"
        "<th>top reason</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def per_class_table(rows: list[dict[str, str]]) -> str:
    body = []
    present = [label for label in LABELS if any(row.get("true_class") == label for row in rows)]
    for cls in present:
        sub = [row for row in rows if row.get("true_class") == cls]
        correct = sum(boolish(row.get("is_correct")) for row in sub)
        preds: dict[str, int] = {}
        for row in sub:
            preds[row.get("predicted_class", "")] = preds.get(row.get("predicted_class", ""), 0) + 1
        pred_text = ", ".join(f"{DISPLAY.get(k, k)}={v}" for k, v in sorted(preds.items()))
        body.append(
            f"<tr><td>{DISPLAY[cls]}</td><td>{len(sub)}</td><td>{correct}/{len(sub)}</td><td>{esc(pred_text)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>true class</th><th>n</th><th>Piece Top1</th><th>predicted-class distribution</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def candidate_card(row: dict[str, str], request: dict[str, str], hidden: dict[str, dict[str, str]]) -> str:
    uid = row["candidate_uid"]
    h = hidden.get(uid, {})
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    correct_cls = "ok" if boolish(row.get("is_correct", "")) else "bad"
    he_path = POOL_ROOT / request["he_crop_rel"]
    fic_path = POOL_ROOT / request["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}: {he_path}, {fic_path}")
    return (
        "<details class=\"candidate\" open>"
        "<summary>"
        f"<span class=\"uid\">{esc(uid)}</span>"
        f"<span>true: <strong>{esc(DISPLAY.get(true, true))}</strong></span>"
        f"<span>pred: <strong class=\"{correct_cls}\">{esc(DISPLAY.get(pred, pred))}</strong> ({esc(row.get('top_score', ''))})</span>"
        f"<span>correct: <strong class=\"{correct_cls}\">{esc(row.get('is_correct', ''))}</strong></span>"
        "</summary>"
        "<div class=\"body\">"
        "<div class=\"leftcol\">"
        "<table class=\"meta\"><tbody>"
        f"<tr><th>selection note</th><td>{esc(h.get('selection_note', ''))}</td></tr>"
        f"<tr><th>component Dice / P / R</th><td>{esc(h.get('component_dice', ''))} / {esc(h.get('component_precision', ''))} / {esc(h.get('component_recall', ''))}</td></tr>"
        f"<tr><th>component id / area</th><td>{esc(h.get('matched_annotation_component_id', ''))} / {esc(h.get('piece_area', ''))}</td></tr>"
        f"<tr><th>setting / candidate</th><td>{esc(row.get('setting', ''))} / {esc(row.get('candidate_id', ''))}</td></tr>"
        "</tbody></table>"
        f"{score_vector(row)}"
        "<div class=\"reason\"><h4>Top score reason</h4>"
        f"<p>{esc(row.get('top_score_reason', ''))}</p></div>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E\"><figcaption>old H&amp;E gray reverse-blur crop</figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE\"><figcaption>FICTURE gray reverse-blur crop</figcaption></figure>"
        "</div>"
        "</div>"
        "<div class=\"reason-block\">"
        "<h4>Evidence-first per-class scoring from GPT-5.5</h4>"
        f"{reason_table(row)}"
        "</div>"
        "</details>"
    )


def build_html() -> str:
    requests = {row["candidate_uid"]: row for row in read_csv(POOL_ROOT / "public_vlm_requests.csv")}
    hidden_rows = read_csv(POOL_ROOT / "hidden_candidate_truth.csv")
    hidden = {row["candidate_uid"]: row for row in hidden_rows}
    rows = read_csv(RESULT_DIR / "per_candidate_evidence_first_predictions.csv")
    rows.sort(key=lambda row: int(requests[row["candidate_uid"]].get("selection_order", "0")))
    prompt = (RESULT_DIR / "prompt_user_template.txt").read_text(errors="replace").strip()
    system = (RESULT_DIR / "prompt_system.txt").read_text(errors="replace").strip()
    correct = sum(boolish(row.get("is_correct", "")) for row in rows)
    cards = "\n".join(candidate_card(row, requests[row["candidate_uid"]], hidden) for row in rows)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.5}
header{background:#111827;color:white;padding:30px 42px} main{max-width:1540px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px} h2{margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
.sub{color:#cbd5e1}.note{background:#ecfdf5;border-left:5px solid #10b981;padding:13px 15px;border-radius:7px;color:#064e3b}
.warn{background:#fff7ed;border-left:5px solid #f97316;padding:13px 15px;border-radius:7px;color:#7c2d12}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}
th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f1f5f9}
code,pre,.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.muted{color:#64748b;font-size:12px}
details.prompt{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}
details.prompt summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}
pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:14px;margin:0;border-radius:0 0 8px 8px;max-height:460px;overflow:auto;font-size:12px}
.candidate{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:18px 0;overflow:hidden}
.candidate summary{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.2fr .7fr 1fr .6fr;gap:10px;align-items:center;background:#fff}
.uid{font-weight:700;color:#1d4ed8}.body{display:grid;grid-template-columns:440px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.meta,.score-vector{font-size:13px;margin:0 0 12px}.meta th,.meta td,.score-vector th,.score-vector td{padding:5px 7px}.meta th{width:160px}
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
<title>Jun11 GPT-5.5 User-Selected 20-Candidate Evidence Audit</title><style>{css}</style></head>
<body>
<header>
  <h1>Jun11 GPT-5.5 User-Selected 20-Candidate Evidence Audit</h1>
  <div class="sub">20 selected candidates | old H&amp;E gray reverse-blur + FICTURE gray reverse-blur | RGB + cell type only | self-contained HTML</div>
</header>
<main>
<section>
  <h2>What This Tests</h2>
  <p>This small audit tests the exact candidates requested by the user: three alveoli candidates, four vessel candidates, six tumor candidates, four stroma candidates, and three immune-infiltration candidates. GPT-5.5 receives two aligned images per candidate: old H&amp;E gray reverse-blur and FICTURE gray reverse-blur.</p>
  <p class="warn"><strong>Critical prompt constraint:</strong> the FICTURE RGB legend applies only to the FICTURE image. H&amp;E staining colors must not be treated as FICTURE RGB / cell-type colors.</p>
  <p class="note"><strong>Observed result:</strong> GPT-5.5 predicted {correct}/{len(rows)} correctly. It missed all 3 alveoli candidates, but correctly recognized the selected vessel, tumor, stroma, and immune-infiltration candidates.</p>
</section>
<section>
  <h2>Class-Level Summary</h2>
  {per_class_table(rows)}
</section>
<section>
  <h2>Candidate Summary</h2>
  {summary_table(rows, hidden)}
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
    html_text = build_html()
    OUT_HTML.write_text(html_text, encoding="utf-8")
    ROOT_COPY.write_text(html_text, encoding="utf-8")
    missing_local = "/Users/haoranwu" in html_text or "file://" in html_text
    has_major = "Major Compartment" in html_text or "compartment:" in html_text
    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(f"candidate cards: {html_text.count('<details class=\"candidate\"')}")
    print(f"embedded images: {html_text.count('data:image/')}")
    print(f"local refs present: {missing_local}")
    print(f"major compartment text present: {has_major}")
    print(f"size_mb: {OUT_HTML.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
