#!/usr/bin/env python3
"""Build a self-contained candidate-level HTML for Jun10 direct VLM results."""

from __future__ import annotations

import base64
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool"
REQUEST_CSV = POOL_ROOT / "public_vlm_requests.csv"
TRUTH_CSV = POOL_ROOT / "hidden_candidate_truth.csv"
RESULT_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_NoCompartment_RGBCellType_VLM_Qwen3_167"
OUT_DIR = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_DirectVLM_RGBCellType_NoCompartment_167_CandidateScores"
OUT_HTML = OUT_DIR / "Jun10_DirectVLM_RGBCellType_NoCompartment_167_CandidateScores_SHAREABLE.html"
ROOT_COPY = ROOT / "Jun10_DirectVLM_RGBCellType_NoCompartment_167_CandidateScores_SHAREABLE.html"

LABELS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
DISPLAY = {
    "bronchiola": "bronchiola",
    "alveoli": "alveoli",
    "vessels": "vessels",
    "tumor": "tumor",
    "stroma": "stroma",
    "immune_infiltration": "immune infiltration",
}
CLASS_ORDER = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]

RUNS = [
    {
        "key": "full_desc",
        "name": "Full class descriptions + RGB/cell type legend",
        "short": "full descriptions",
        "dir": RESULT_ROOT / "qwen3vl32b_rgb_celltype_only_full167_retry_HE_FICTURE_RGB_celltype_only",
    },
    {
        "key": "class_names",
        "name": "Class-name-only + RGB/cell type legend",
        "short": "class names only",
        "dir": RESULT_ROOT / "qwen3vl32b_rgb_celltype_nodesc_full167_HE_FICTURE_RGB_celltype_nodesc",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def fmt_float(value: str, ndigits: int = 3) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return ""


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def prediction_summary(row: dict[str, str]) -> tuple[str, str, bool]:
    scores = {label: int(float(row.get(label, "0") or 0)) for label in LABELS}
    top = max(scores.values())
    winners = [label for label, score in scores.items() if score == top]
    pred = winners[0] if len(winners) == 1 else "tie"
    correct = pred == row.get("true_class")
    return pred, str(top), correct


def score_table(row: dict[str, str]) -> str:
    cells = []
    for label in LABELS:
        cells.append(f"<tr><td>{esc(DISPLAY[label])}</td><td><strong>{esc(row.get(label, ''))}</strong></td></tr>")
    return (
        "<table class=\"score-vector\"><thead><tr><th>class</th><th>score</th></tr></thead>"
        f"<tbody>{''.join(cells)}</tbody></table>"
    )


def load_runs() -> dict[str, dict[str, dict[str, str]]]:
    loaded: dict[str, dict[str, dict[str, str]]] = {}
    for run in RUNS:
        rows = read_csv(run["dir"] / "per_candidate_predictions.csv")
        by_uid = {row["candidate_uid"]: row for row in rows}
        if len(by_uid) != 167:
            raise RuntimeError(f"{run['name']} has {len(by_uid)} unique candidates, expected 167")
        loaded[run["key"]] = by_uid
    return loaded


def build_overall_table(run_rows: dict[str, dict[str, dict[str, str]]]) -> str:
    body = []
    for run in RUNS:
        rows = list(run_rows[run["key"]].values())
        correct = sum(1 for row in rows if boolish(row.get("is_correct", "")))
        ties = sum(1 for row in rows if boolish(row.get("top_score_tie", "")))
        score_vectors = {
            tuple(row.get(label, "") for label in LABELS)
            for row in rows
        }
        pred_counts = Counter(row.get("predicted_class", "") for row in rows)
        body.append(
            "<tr>"
            f"<td>{esc(run['name'])}</td>"
            f"<td><strong>{correct}/{len(rows)}</strong></td>"
            f"<td>{ties}</td>"
            f"<td>{len(score_vectors)}/{len(rows)}</td>"
            f"<td>{esc(pred_counts.most_common(1)[0][0])}: {pred_counts.most_common(1)[0][1]}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Direct VLM setting</th><th>Overall Piece Top1</th>"
        "<th>top-score ties</th><th>unique score vectors</th><th>largest predicted-class collapse</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_class_table(run_rows: dict[str, dict[str, dict[str, str]]]) -> str:
    body = []
    for cls in CLASS_ORDER:
        row_cells = [f"<td><strong>{esc(DISPLAY[cls])}</strong></td>"]
        for run in RUNS:
            rows = [row for row in run_rows[run["key"]].values() if row.get("true_class") == cls]
            correct = sum(1 for row in rows if boolish(row.get("is_correct", "")))
            pred_counts = Counter(row.get("predicted_class", "") for row in rows)
            dist = "; ".join(f"{DISPLAY.get(k, k)}:{v}" for k, v in pred_counts.most_common())
            row_cells.append(f"<td><strong>{correct}/{len(rows)}</strong><br><span class=\"muted\">{esc(dist)}</span></td>")
        body.append(f"<tr>{''.join(row_cells)}</tr>")
    headers = "".join(f"<th>{esc(run['short'])}</th>" for run in RUNS)
    return (
        "<table><thead><tr><th>true class</th>"
        f"{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def build_prompt_block() -> str:
    parts = []
    for run in RUNS:
        system = (run["dir"] / "prompt_system.txt").read_text(errors="replace").strip()
        user = (run["dir"] / "prompt_user_first_row.txt").read_text(errors="replace").strip()
        parts.append(
            f"<details class=\"prompt\"><summary>{esc(run['name'])}</summary>"
            f"<h4>System prompt</h4><pre>{esc(system)}</pre>"
            f"<h4>User prompt example from first row</h4><pre>{esc(user)}</pre>"
            "</details>"
        )
    return "".join(parts)


def candidate_card(
    row: dict[str, str],
    truth: dict[str, str],
    run_rows: dict[str, dict[str, dict[str, str]]],
    request_root: Path,
) -> str:
    uid = row["candidate_uid"]
    true_class = row["display"].replace(" ", "_") if row["display"] == "immune infiltration" else row["display"]
    he_path = request_root / row["he_crop_rel"]
    fic_path = request_root / row["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}: {he_path.exists()} {fic_path.exists()}")

    metric = " / ".join(
        value for value in [
            fmt_float(truth.get("component_dice", "")),
            fmt_float(truth.get("component_precision", "")),
            fmt_float(truth.get("component_recall", "")),
        ] if value
    )
    meta = [
        ("hidden true class", DISPLAY.get(true_class, true_class)),
        ("matched annotation component", truth.get("matched_annotation_component_id", "")),
        ("component Dice / Precision / Recall", metric),
        ("piece area", truth.get("piece_area", "")),
        ("source pieces", truth.get("parent_member_sources", "")),
        ("cluster size", truth.get("parent_cluster_size", "")),
    ]
    meta_rows = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in meta if v)

    run_blocks = []
    status_bits = []
    for run in RUNS:
        pred_row = run_rows[run["key"]][uid]
        pred = pred_row.get("predicted_class", "")
        correct = boolish(pred_row.get("is_correct", ""))
        cls = "ok" if correct else "bad"
        status_bits.append(f"{esc(run['short'])}: <span class=\"{cls}\">{esc(DISPLAY.get(pred, pred))}</span>")
        run_blocks.append(
            "<section class=\"run-block\">"
            f"<h4>{esc(run['name'])}</h4>"
            "<table class=\"mini\"><tbody>"
            f"<tr><th>predicted class</th><td class=\"{cls}\">{esc(DISPLAY.get(pred, pred))}</td></tr>"
            f"<tr><th>Piece Top1 correct?</th><td class=\"{cls}\">{esc(pred_row.get('is_correct', ''))}</td></tr>"
            f"<tr><th>top score / tie</th><td>{esc(pred_row.get('top_score', ''))} / {esc(pred_row.get('top_score_tie', ''))}</td></tr>"
            f"<tr><th>parse status</th><td>{esc(pred_row.get('parse_status', ''))}</td></tr>"
            "</tbody></table>"
            f"{score_table(pred_row)}"
            "</section>"
        )

    return (
        "<details class=\"candidate\" open>"
        "<summary>"
        f"<span class=\"uid\">{esc(uid)}</span>"
        f"<span>true class: <strong>{esc(DISPLAY.get(true_class, true_class))}</strong></span>"
        f"<span>{' | '.join(status_bits)}</span>"
        f"<span>Dice / P / R: {esc(metric)}</span>"
        "</summary>"
        "<div class=\"body\">"
        "<div class=\"scores\">"
        f"<table class=\"score meta\"><tbody>{meta_rows}</tbody></table>"
        f"<div class=\"run-grid\">{''.join(run_blocks)}</div>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E gray reverse-blur\"><figcaption>H&amp;E gray reverse-blur candidate crop</figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE gray reverse-blur\"><figcaption>FICTURE gray reverse-blur candidate crop</figcaption></figure>"
        "</div>"
        "</div>"
        "</details>"
    )


def build_html() -> str:
    requests = read_csv(REQUEST_CSV)
    truth_rows = {row["candidate_uid"]: row for row in read_csv(TRUTH_CSV)}
    run_rows = load_runs()
    if len(requests) != 167 or len(truth_rows) != 167:
        raise RuntimeError(f"Expected 167 requests/truth rows, got {len(requests)} and {len(truth_rows)}")

    missing = []
    for row in requests:
        uid = row["candidate_uid"]
        if uid not in truth_rows:
            missing.append(uid)
        for run in RUNS:
            if uid not in run_rows[run["key"]]:
                missing.append(f"{run['key']}:{uid}")
    if missing:
        raise RuntimeError(f"Missing rows: {missing[:10]}")

    by_class = defaultdict(list)
    for row in requests:
        cls = row["display"].replace(" ", "_") if row["display"] == "immune infiltration" else row["display"]
        by_class[cls].append(row)

    sections = []
    for cls in CLASS_ORDER:
        rows = by_class[cls]
        rows.sort(key=lambda r: float(truth_rows[r["candidate_uid"]].get("component_dice", "0") or 0), reverse=True)
        cards = "\n".join(candidate_card(row, truth_rows[row["candidate_uid"]], run_rows, POOL_ROOT) for row in rows)
        sections.append(
            f"<section id=\"{esc(cls)}\"><h2>{esc(DISPLAY[cls])} <span>{len(rows)} candidates</span></h2>{cards}</section>"
        )

    nav = " ".join(f"<a href=\"#{cls}\">{esc(DISPLAY[cls])}</a>" for cls in CLASS_ORDER)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.48}
header{background:#111827;color:white;padding:30px 42px}
main{max-width:1480px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px} h2{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
h2 span{font-size:15px;color:#64748b;font-weight:500;margin-left:8px}
h3{margin-top:22px} h4{margin:12px 0 8px} code,pre,.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sub{color:#cbd5e1}.muted{color:#64748b}.callout{background:#fff7ed;border-left:5px solid #f97316;padding:13px 15px;border-radius:7px;color:#431407}
.note{color:#334155;background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 14px;border-radius:6px}
nav{position:sticky;top:0;background:rgba(248,250,252,.96);backdrop-filter:blur(8px);padding:12px 0;z-index:10;border-bottom:1px solid #e5e7eb}
nav a{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;border:1px solid #cbd5e1;border-radius:999px;color:#0f172a;text-decoration:none;background:white;font-size:14px}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}
th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f1f5f9}
.summary-grid{display:grid;grid-template-columns:minmax(280px,680px) minmax(280px,680px);gap:22px}
details.prompt{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}
details.prompt summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}
pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:16px;margin:0;border-radius:0 0 8px 8px;max-height:460px;overflow:auto;font-size:12px}
.candidate{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0;overflow:hidden}
.candidate summary{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.2fr .75fr 1.2fr .9fr;gap:10px;align-items:center;background:#fff}
.candidate summary:hover{background:#f8fafc}
.uid{font-weight:700;color:#1d4ed8}
.body{display:grid;grid-template-columns:430px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.score{font-size:13px;margin:0 0 12px}
.score th,.score td{padding:5px 7px}
.mini,.score-vector{font-size:13px;margin:0 0 12px}.mini th,.mini td,.score-vector th,.score-vector td{padding:5px 7px}
.ok{color:#047857;font-weight:700} .bad{color:#b91c1c;font-weight:700}
.figgrid{display:grid;gap:16px} .figgrid.mode-1{grid-template-columns:minmax(320px,620px)} .figgrid.mode-2{grid-template-columns:repeat(2,minmax(300px,1fr))}
figure{margin:0} img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white} figcaption{font-size:13px;color:#475569;margin-top:6px}
.run-grid{display:grid;grid-template-columns:1fr;gap:12px}.run-block{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px}
.meta th{width:180px}
@media(max-width:980px){.summary-grid,.body,.candidate summary,.figgrid.mode-2{grid-template-columns:1fr}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun10 Direct VLM RGB/Cell-Type 167 Candidate Scores</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Jun10 Direct VLM RGB/Cell-Type 167 Candidate Scores</h1>
  <div class="sub">Qwen3-VL-32B | H&amp;E + FICTURE gray reverse-blur crops | RGB + cell type legend only | self-contained HTML</div>
</header>
<main>
<nav>{nav}</nav>
<section>
  <h2>Experiment Design</h2>
  <p><strong>Goal:</strong> test whether direct VLM scoring can recognize the tissue type of each compact candidate piece.</p>
  <p><strong>Candidate pool:</strong> 167 compact piece-level candidates from the <code>medical_official_points_step24</code> HE+FICTURE pool. The model sees one piece at a time, not a final union mask.</p>
  <p><strong>Input images:</strong> two aligned gray reverse-blur crops per candidate: H&amp;E and official FICTURE. The candidate piece is sharp; surrounding tissue is gray/blurred.</p>
  <p><strong>FICTURE legend:</strong> the prompt uses <strong>RGB + cell type only</strong>. The older major-compartment field is not used in these two runs.</p>
  <p><strong>Model output:</strong> six scores from 0 to 100: bronchiola, alveoli, vessels, tumor, stroma, and immune infiltration.</p>
  <p><strong>Piece Top1:</strong> the class with the highest score is the prediction. The candidate is correct only if that predicted class equals the hidden annotation-derived true class.</p>
  <p><strong>Ground truth:</strong> annotation/component labels and Dice/Precision/Recall are hidden from the model and shown here only for evaluation.</p>
  <p><strong>Official-data guardrail:</strong> official FICTURE alignment status is <strong>PASS_OFFICIAL</strong>; deprecated/debug FICTURE roots are not used.</p>
  <p class="callout"><strong>Main finding:</strong> direct VLM still collapses toward tumor. Both prompt variants score <strong>26/167</strong> overall and predict <strong>160/167</strong> candidates as tumor, so direct VLM is not reliable as the main tissue-piece classifier in this setting.</p>
</section>
<section>
  <h2>Result Summary</h2>
  <div class="summary-grid">
    <div>
      <h3>Overall</h3>
      {build_overall_table(run_rows)}
    </div>
    <div>
      <h3>By true class</h3>
      {build_class_table(run_rows)}
    </div>
  </div>
</section>
<section>
  <h2>Prompt Used</h2>
  <p class="note">Both prompts use the same two input images and the same RGB/cell-type legend. The difference is whether the six tissue classes are described in prose or listed only by name.</p>
  {build_prompt_block()}
</section>
<section>
  <h2>Candidate-Level Scores</h2>
  <p class="note">Open each card to inspect the two input crops and the exact six scores from each direct VLM setting. Cards are grouped by hidden true class and sorted by component Dice.</p>
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

    missing_local_paths = [
        token for token in ["src=\"output/", "src=\"/Users/", "href=\"/Users/"]
        if token in text
    ]
    if missing_local_paths:
        raise RuntimeError(f"HTML contains local references: {missing_local_paths}")

    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(f"candidate cards: {text.count('<details class=\"candidate\"')}")
    print(f"embedded images: {text.count('data:image/')}")
    print(f"size_mb: {OUT_HTML.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
