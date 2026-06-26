#!/usr/bin/env python3
"""Build one shareable HTML for all completed Jun11 GPT-5.5 evidence audits."""

from __future__ import annotations

import base64
import csv
import html
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_GPT55_Today_EvidenceFirst_All"
OUT_HTML = OUT_DIR / "Jun11_GPT55_Today_EvidenceFirst_All_SHAREABLE.html"
ROOT_COPY = ROOT / "Jun11_GPT55_Today_EvidenceFirst_All_SHAREABLE.html"
GLOBAL_HIDDEN = (
    ROOT
    / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool/hidden_candidate_truth.csv"
)

RUNS = [
    {
        "name": "Bronchiola 3-candidate GPT-5.5 audit",
        "root": ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_GPT55_OldHE_EvidenceFirst_3",
        "order": 0,
    },
    {
        "name": "User-selected 20-candidate GPT-5.5 audit",
        "root": ROOT / "output/visium_hd_exp1/final_deliverables/Jun11_GPT55_OldHE_EvidenceFirst_UserSelected20",
        "order": 100,
    },
]

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


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def metric_text(hidden: dict[str, str]) -> str:
    return (
        f"Dice {esc(hidden.get('component_dice', 'NA'))} | "
        f"Precision {esc(hidden.get('component_precision', 'NA'))} | "
        f"Recall {esc(hidden.get('component_recall', 'NA'))}"
    )


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
            f"<tr class=\"{' '.join(classes)}\"><td>{esc(DISPLAY[label])}</td>"
            f"<td><strong>{esc(row.get(label, ''))}</strong></td></tr>"
        )
    return (
        "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
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


def collect_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    hidden = {row["candidate_uid"]: row for row in read_csv(GLOBAL_HIDDEN)}
    request_by_uid: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []
    seen = set()
    for run in RUNS:
        root = run["root"]
        request_rows = read_csv(root / "corrected_pool/public_vlm_requests.csv")
        request_map = {row["candidate_uid"]: row for row in request_rows}
        pred_path = root / "openrouter_gpt55_evidence/per_candidate_evidence_first_predictions.csv"
        if not pred_path.exists():
            continue
        for idx, row in enumerate(read_csv(pred_path)):
            if not row.get("candidate_uid") or row["candidate_uid"] in seen:
                continue
            if row["candidate_uid"] not in request_map:
                raise KeyError(f"Missing request row for {row['candidate_uid']} in {root}")
            seen.add(row["candidate_uid"])
            row = dict(row)
            row["run_name"] = run["name"]
            row["_sort_order"] = str(run["order"] + idx)
            row["_pool_root"] = str(root / "corrected_pool")
            rows.append(row)
            req = dict(request_map[row["candidate_uid"]])
            req["_pool_root"] = str(root / "corrected_pool")
            request_by_uid[row["candidate_uid"]] = req
    rows.sort(key=lambda item: int(item["_sort_order"]))
    return rows, request_by_uid, hidden


def class_summary_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("true_class", "")].append(row)
    body = []
    for label in LABELS:
        sub = grouped.get(label, [])
        if not sub:
            continue
        correct = sum(boolish(row.get("is_correct")) for row in sub)
        pred_counts: dict[str, int] = defaultdict(int)
        for row in sub:
            pred_counts[row.get("predicted_class", "")] += 1
        pred_text = ", ".join(f"{DISPLAY.get(k, k)}={v}" for k, v in sorted(pred_counts.items()))
        body.append(
            f"<tr><td>{DISPLAY[label]}</td><td>{len(sub)}</td><td>{correct}/{len(sub)}</td>"
            f"<td>{esc(pred_text)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>true class</th><th>n</th><th>Piece Top1</th>"
        "<th>predicted distribution</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def summary_table(rows: list[dict[str, str]], hidden: dict[str, dict[str, str]]) -> str:
    body = []
    for row in rows:
        uid = row["candidate_uid"]
        h = hidden.get(uid, {})
        body.append(
            "<tr>"
            f"<td><code>{esc(uid)}</code><br><span class=\"muted\">{esc(row.get('run_name', ''))}</span></td>"
            f"<td>{esc(DISPLAY.get(row.get('true_class', ''), row.get('true_class', '')))}</td>"
            f"<td><strong>{esc(DISPLAY.get(row.get('predicted_class', ''), row.get('predicted_class', '')))}</strong></td>"
            f"<td class=\"{'ok' if boolish(row.get('is_correct')) else 'bad'}\">{esc(row.get('is_correct'))}</td>"
            f"<td>{metric_text(h)}</td>"
            f"<td>{' / '.join(esc(row.get(label, '')) for label in LABELS)}</td>"
            f"<td>{esc(row.get('top_score_reason', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>candidate</th><th>true</th><th>GPT-5.5 pred</th><th>correct</th>"
        "<th>component metric</th><th>scores<br>bron/alv/ves/tum/stro/imm</th><th>top reason</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def candidate_card(row: dict[str, str], request: dict[str, str], hidden: dict[str, str]) -> str:
    uid = row["candidate_uid"]
    pred = row.get("predicted_class", "")
    true = row.get("true_class", "")
    correct_cls = "ok" if boolish(row.get("is_correct", "")) else "bad"
    pool_root = Path(request["_pool_root"])
    he_path = pool_root / request["he_crop_rel"]
    fic_path = pool_root / request["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}: {he_path}, {fic_path}")
    metrics = metric_text(hidden)
    return (
        "<details class=\"candidate\" open>"
        "<summary>"
        f"<span class=\"uid\">{esc(uid)}</span>"
        f"<span>true: <strong>{esc(DISPLAY.get(true, true))}</strong></span>"
        f"<span>pred: <strong class=\"{correct_cls}\">{esc(DISPLAY.get(pred, pred))}</strong> ({esc(row.get('top_score', ''))})</span>"
        f"<span>correct: <strong class=\"{correct_cls}\">{esc(row.get('is_correct', ''))}</strong></span>"
        "</summary>"
        "<div class=\"metric-banner\">"
        f"<strong>{metrics}</strong>"
        f"<span>matched component: {esc(hidden.get('matched_annotation_component_id', 'NA'))}</span>"
        f"<span>piece area: {esc(hidden.get('piece_area', 'NA'))}</span>"
        "</div>"
        "<div class=\"body\">"
        "<div class=\"leftcol\">"
        "<table class=\"meta\"><tbody>"
        f"<tr><th>source run</th><td>{esc(row.get('run_name', ''))}</td></tr>"
        f"<tr><th>hidden label</th><td>{esc(DISPLAY.get(true, true))}</td></tr>"
        f"<tr><th>prediction</th><td class=\"{correct_cls}\">{esc(DISPLAY.get(pred, pred))}</td></tr>"
        f"<tr><th>component metric</th><td>{metrics}</td></tr>"
        f"<tr><th>setting / candidate</th><td>{esc(row.get('setting', ''))} / {esc(row.get('candidate_id', ''))}</td></tr>"
        "</tbody></table>"
        f"{score_vector(row)}"
        "<div class=\"reason\"><h4>Top score reason</h4>"
        f"<p>{esc(row.get('top_score_reason', ''))}</p></div>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E\"><figcaption>H&amp;E crop<br><strong>{metrics}</strong></figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE\"><figcaption>FICTURE crop<br><strong>{metrics}</strong></figcaption></figure>"
        "</div>"
        "</div>"
        "<div class=\"reason-block\">"
        "<h4>Evidence-first per-class scoring from GPT-5.5</h4>"
        f"{reason_table(row)}"
        "</div>"
        "</details>"
    )


def prompt_section() -> str:
    prompt_path = RUNS[-1]["root"] / "openrouter_gpt55_evidence/prompt_user_template.txt"
    system_path = RUNS[-1]["root"] / "openrouter_gpt55_evidence/prompt_system.txt"
    prompt = prompt_path.read_text(errors="replace").strip()
    system = system_path.read_text(errors="replace").strip()
    return (
        "<details class=\"prompt\"><summary>System prompt</summary>"
        f"<pre>{esc(system)}</pre></details>"
        "<details class=\"prompt\" open><summary>User prompt template</summary>"
        f"<pre>{esc(prompt)}</pre></details>"
    )


def build_html() -> str:
    rows, requests, hidden = collect_rows()
    correct = sum(boolish(row.get("is_correct")) for row in rows)
    cards = "\n".join(candidate_card(row, requests[row["candidate_uid"]], hidden.get(row["candidate_uid"], {})) for row in rows)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.5}
header{background:#111827;color:white;padding:30px 42px} main{max-width:1560px;margin:0 auto;padding:28px 34px 70px}
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
.metric-banner{display:flex;gap:20px;flex-wrap:wrap;padding:10px 14px;background:#eff6ff;border-top:1px solid #bfdbfe;border-bottom:1px solid #bfdbfe;color:#1e3a8a}
.uid{font-weight:700;color:#1d4ed8}.body{display:grid;grid-template-columns:450px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.meta,.score-vector{font-size:13px;margin:0 0 12px}.meta th,.meta td,.score-vector th,.score-vector td{padding:5px 7px}.meta th{width:160px}
.score-vector tr.pred,.reason-table tr.pred{background:#fff7ed}.score-vector tr.truth,.reason-table tr.truth{box-shadow:inset 4px 0 0 #2563eb}
.ok{color:#047857;font-weight:700}.bad{color:#b91c1c;font-weight:700}.score{color:#475569;font-size:12px}
.reason{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:9px 10px;margin:10px 0}.reason h4{margin-top:0}.reason p{margin:0;color:#334155}
.reason-block{padding:0 14px 14px}.reason-table{font-size:13px}.reason-table th:nth-child(1){width:145px}.reason-table th:nth-child(4){width:25%}
.figgrid{display:grid;gap:16px}.figgrid.mode-2{grid-template-columns:repeat(2,minmax(330px,1fr))}
figure{margin:0} img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white} figcaption{font-size:13px;color:#475569;margin-top:6px}
@media(max-width:1100px){.body,.candidate summary,.figgrid.mode-2{grid-template-columns:1fr}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun11 GPT-5.5 Evidence-First Candidate Audit - All Completed Runs</title><style>{css}</style></head>
<body>
<header>
  <h1>Jun11 GPT-5.5 Evidence-First Candidate Audit</h1>
  <div class="sub">All completed GPT-5.5 runs from today | 23 unique candidates | self-contained HTML</div>
</header>
<main>
<section>
  <h2>What This Page Contains</h2>
  <p>This page merges today's completed GPT-5.5 evidence-first audits: the 3-candidate bronchiola check and the 20-candidate user-selected alveoli/vessels/tumor/stroma/immune check.</p>
  <p class="warn"><strong>Prompt constraint:</strong> the FICTURE RGB legend applies only to the FICTURE image. H&amp;E staining colors must not be treated as FICTURE RGB or cell-type colors.</p>
  <p class="note"><strong>Overall result:</strong> GPT-5.5 predicted {correct}/{len(rows)} candidates correctly. Dice / Precision / Recall shown here are hidden annotation-based component metrics, not inputs to the model.</p>
</section>
<section>
  <h2>Class-Level Summary</h2>
  {class_summary_table(rows)}
</section>
<section>
  <h2>Candidate Summary</h2>
  {summary_table(rows, hidden)}
</section>
<section>
  <h2>Prompt</h2>
  {prompt_section()}
</section>
<section>
  <h2>Candidate-Level Images, Scores, And Evidence</h2>
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
    checks = {
        "candidate_cards": html_text.count('<details class="candidate"'),
        "embedded_images": html_text.count("data:image/"),
        "has_local_refs": "/Users/haoranwu" in html_text or "file://" in html_text,
        "has_major_compartment": "Major Compartment" in html_text or "compartment:" in html_text,
    }
    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(checks)
    if checks["candidate_cards"] != 23:
        raise SystemExit("Expected 23 candidate cards")
    if checks["embedded_images"] != 46:
        raise SystemExit("Expected 46 embedded images")
    if checks["has_local_refs"] or checks["has_major_compartment"]:
        raise SystemExit("HTML self-check failed")


if __name__ == "__main__":
    main()
