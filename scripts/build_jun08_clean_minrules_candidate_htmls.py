#!/usr/bin/env python3
"""Build the clean Jun08 three-input-mode candidate HTML set.

This rebuilds exactly three shareable HTML files from the later minrules run:
H&E only, FICTURE only, and H&E + FICTURE. Each file embeds its images and the
actual prompt files from the matching run directory, so metrics and prompt text
cannot drift apart.
"""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path


REPO = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
POOL_DIR = Path(
    "/Users/haoranwu/Desktop/Yan_Lab_Research/"
    "VisiumHD-Segmentation-MaskSelection/data/visium_hd_exp1/"
    "current_ficture_vlm_inputs/compact167_reverse_blur_pool"
)
SCORE_DIR = (
    REPO
    / "output/visium_hd_exp1/final_deliverables/"
    / "Jun08_minrules_no_locator_results/raw_remote_scores"
)
OUT_DIR = (
    REPO
    / "output/visium_hd_exp1/final_deliverables/"
    / "Jun08_CLEAN_minrules_three_input_mode_candidate_htmls"
)
OFFICIAL_SUMMARY = REPO / "output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"

CLASSES = [
    ("bronchiola", "bronchiola"),
    ("alveoli", "alveoli"),
    ("vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]

MODES = {
    "HE only": {
        "display": "H&E only",
        "filename": "Jun08_CLEAN_HE_only_167_candidates.html",
        "score_dir": "qwen3vl32b_minrules_no_locator_14068193_HE_only_no_locator_minrules",
        "images": [("H&E gray reverse-blur candidate crop", "he_crop_rel")],
        "purpose": "This mode tests whether H&E morphology alone lets Qwen3-VL-32B identify the tissue class of each candidate piece.",
    },
    "FICTURE only": {
        "display": "FICTURE only",
        "filename": "Jun08_CLEAN_FICTURE_only_167_candidates.html",
        "score_dir": "qwen3vl32b_minrules_no_locator_14068193_FICTURE_only_no_locator_minrules",
        "images": [("FICTURE gray reverse-blur candidate crop", "ficture_crop_rel")],
        "purpose": "This mode tests whether the official FICTURE color map alone lets Qwen3-VL-32B identify the tissue class of each candidate piece.",
    },
    "HE + FICTURE": {
        "display": "H&E + FICTURE",
        "filename": "Jun08_CLEAN_HE_plus_FICTURE_167_candidates.html",
        "score_dir": "qwen3vl32b_minrules_no_locator_14068193_HE_FICTURE_no_locator_minrules",
        "images": [
            ("H&E gray reverse-blur candidate crop", "he_crop_rel"),
            ("FICTURE gray reverse-blur candidate crop", "ficture_crop_rel"),
        ],
        "purpose": "This mode tests whether adding the official FICTURE crop to the H&E crop improves tissue recognition.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


def fmt_float(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return ""


def join_pool_rows() -> dict[str, dict[str, str]]:
    public_rows = read_csv(POOL_DIR / "public_vlm_requests.csv")
    hidden_rows = read_csv(POOL_DIR / "hidden_candidate_truth.csv")
    hidden_by_uid = {row["candidate_uid"]: row for row in hidden_rows}
    out: dict[str, dict[str, str]] = {}
    for row in public_rows:
        uid = row["candidate_uid"]
        merged = dict(row)
        merged.update(hidden_by_uid.get(uid, {}))
        merged["he_crop_rel"] = row["he_crop_rel"]
        merged["ficture_crop_rel"] = row["ficture_crop_rel"]
        out[uid] = merged
    if len(out) != 167:
        raise RuntimeError(f"Expected 167 pool candidates, got {len(out)}")
    return out


def load_mode_predictions(mode_key: str) -> list[dict[str, str]]:
    mode = MODES[mode_key]
    path = SCORE_DIR / mode["score_dir"] / "per_candidate_predictions.csv"
    rows = read_csv(path)
    if len(rows) != 167:
        raise RuntimeError(f"Expected 167 predictions for {mode_key}, got {len(rows)}")
    return rows


def prompt_text(mode_key: str) -> str:
    mode = MODES[mode_key]
    root = SCORE_DIR / mode["score_dir"]
    system = (root / "prompt_system.txt").read_text(encoding="utf-8")
    user_path = root / "prompt_user_template.txt"
    if not user_path.exists():
        user_path = root / "prompt_user.txt"
    user = user_path.read_text(encoding="utf-8")
    return f"System prompt:\n{system.strip()}\n\nUser prompt template:\n{user.strip()}"


def mode_summary(rows: list[dict[str, str]]) -> tuple[str, dict[str, str]]:
    total = len(rows)
    correct = sum(row.get("is_correct", "").lower() == "true" for row in rows)
    by_class: dict[str, str] = {}
    for key, label in CLASSES:
        cls_rows = [row for row in rows if row.get("true_class") == key]
        cls_correct = sum(row.get("is_correct", "").lower() == "true" for row in cls_rows)
        by_class[label] = f"{cls_correct}/{len(cls_rows)}"
    return f"{correct}/{total}", by_class


def all_mode_summary_table(mode_predictions: dict[str, list[dict[str, str]]]) -> str:
    lines = ["<table><thead><tr><th>input mode</th><th>Overall Piece Top1</th></tr></thead><tbody>"]
    for mode_key, rows in mode_predictions.items():
        overall, _ = mode_summary(rows)
        lines.append(f"<tr><td>{html.escape(MODES[mode_key]['display'])}</td><td><strong>{overall}</strong></td></tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def class_summary_table(rows: list[dict[str, str]]) -> str:
    _, by_class = mode_summary(rows)
    lines = ["<table><thead><tr><th>true class</th><th>Piece Top1</th></tr></thead><tbody>"]
    for _, label in CLASSES:
        lines.append(f"<tr><td>{html.escape(label)}</td><td><strong>{html.escape(by_class[label])}</strong></td></tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def score_table(row: dict[str, str]) -> str:
    pred = row.get("predicted_class", "").replace("_", " ")
    ok = row.get("is_correct", "").lower() == "true"
    lines = [
        "<table class=\"score\"><tbody>",
        f"<tr><th>predicted class</th><td class=\"{'ok' if ok else 'bad'}\">{html.escape(pred)}</td></tr>",
        f"<tr><th>Piece Top1 correct?</th><td class=\"{'ok' if ok else 'bad'}\">{html.escape(row.get('is_correct',''))}</td></tr>",
        f"<tr><th>top score / tie</th><td>{html.escape(row.get('top_score',''))} / {html.escape(row.get('top_score_tie',''))}</td></tr>",
        f"<tr><th>parse status</th><td>{html.escape(row.get('parse_status',''))}</td></tr>",
        "</tbody></table>",
        "<table class=\"score\"><thead><tr><th>class</th><th>score</th></tr></thead><tbody>",
    ]
    for key, label in CLASSES:
        lines.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(row.get(key, ''))}</td></tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def candidate_card(mode_key: str, pred: dict[str, str], pool: dict[str, str]) -> str:
    uid = pred["candidate_uid"]
    truth = pool[uid]
    true_class = pred.get("true_class", "").replace("_", " ")
    figures = []
    for label, col in MODES[mode_key]["images"]:
        rel = truth[col]
        path = POOL_DIR / rel
        if not path.exists():
            raise FileNotFoundError(path)
        figures.append(
            "<figure>"
            f"<img src=\"{data_uri(path)}\" alt=\"{html.escape(uid)} {html.escape(label)}\">"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    d = fmt_float(truth.get("component_dice", ""))
    p = fmt_float(truth.get("component_precision", ""))
    r = fmt_float(truth.get("component_recall", ""))
    return f"""
<details class="candidate" open>
  <summary>
    <span class="uid">{html.escape(uid)}</span>
    <span>true class: <strong>{html.escape(true_class)}</strong></span>
    <span>component {html.escape(truth.get('matched_annotation_component_id',''))}</span>
    <span>Dice / P / R: {d} / {p} / {r}</span>
  </summary>
  <div class="body">
    <div class="scores">{score_table(pred)}</div>
    <div class="figgrid mode-{len(MODES[mode_key]['images'])}">{''.join(figures)}</div>
  </div>
</details>
"""


def official_status() -> str:
    try:
        return json.loads(OFFICIAL_SUMMARY.read_text()).get("status", "unknown")
    except Exception:
        return "unknown"


def build_html(mode_key: str, pool_rows: dict[str, dict[str, str]], mode_predictions: dict[str, list[dict[str, str]]]) -> str:
    mode = MODES[mode_key]
    rows = mode_predictions[mode_key]
    overall, _ = mode_summary(rows)
    nav = " ".join(f"<a href=\"#{key}\">{html.escape(label)}</a>" for key, label in CLASSES)
    class_blocks = []
    for key, label in CLASSES:
        cls_rows = [row for row in rows if row.get("true_class") == key]
        cards = "\n".join(candidate_card(mode_key, row, pool_rows) for row in cls_rows)
        class_blocks.append(
            f"<section id=\"{html.escape(key)}\"><h2>{html.escape(label)} <span>{len(cls_rows)} candidates</span></h2>{cards}</section>"
        )

    prompt = prompt_text(mode_key)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun08 CLEAN {html.escape(mode['display'])} 167 Candidate Audit</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.48}}
header{{background:#111827;color:white;padding:30px 42px}}
main{{max-width:1480px;margin:0 auto;padding:28px 34px 70px}}
h1{{margin:0 0 8px;font-size:30px}} h2{{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}}
h2 span{{font-size:15px;color:#64748b;font-weight:500;margin-left:8px}}
h3{{margin-top:22px}} code,pre,.uid{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.sub{{color:#cbd5e1}} .note{{color:#334155;background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 14px;border-radius:6px}}
nav{{position:sticky;top:0;background:rgba(248,250,252,.96);backdrop-filter:blur(8px);padding:12px 0;z-index:10;border-bottom:1px solid #e5e7eb}}
nav a{{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;border:1px solid #cbd5e1;border-radius:999px;color:#0f172a;text-decoration:none;background:white;font-size:14px}}
table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}}
th,td{{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top}} th{{background:#f1f5f9}}
.summary-grid{{display:grid;grid-template-columns:minmax(280px,520px) minmax(280px,520px);gap:22px}}
details.prompt{{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}}
details.prompt summary{{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:16px;margin:0;border-radius:0 0 8px 8px;max-height:460px;overflow:auto;font-size:12px}}
.candidate{{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0;overflow:hidden}}
.candidate summary{{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.2fr .75fr .65fr .9fr;gap:10px;align-items:center;background:#fff}}
.candidate summary:hover{{background:#f8fafc}}
.uid{{font-weight:700;color:#1d4ed8}}
.body{{display:grid;grid-template-columns:360px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}}
.score{{font-size:13px;margin:0 0 12px}}
.score th,.score td{{padding:5px 7px}}
.ok{{color:#047857;font-weight:700}} .bad{{color:#b91c1c;font-weight:700}}
.figgrid{{display:grid;gap:16px}} .figgrid.mode-1{{grid-template-columns:minmax(320px,620px)}} .figgrid.mode-2{{grid-template-columns:repeat(2,minmax(300px,1fr))}}
figure{{margin:0}} img{{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white}} figcaption{{font-size:13px;color:#475569;margin-top:6px}}
@media(max-width:980px){{.summary-grid,.body,.candidate summary,.figgrid.mode-2{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style>
</head>
<body>
<header>
  <h1>Jun08 CLEAN {html.escape(mode['display'])} 167 Candidate Audit</h1>
  <div class="sub">Qwen3-VL-32B | minrules prompt | compact 167 piece-first pool | self-contained HTML</div>
</header>
<main>
<nav>{nav}</nav>
<section>
  <h2>Experiment Design</h2>
  <p><strong>Goal:</strong> compare whether Qwen3-VL-32B recognizes each candidate tissue piece from {html.escape(mode['display'])} input.</p>
  <p><strong>Candidate pool:</strong> the same 167 compact piece candidates are used in all three files. Hidden annotation labels are used only for evaluation and are not shown to the model.</p>
  <p><strong>Model input:</strong> {html.escape(mode['purpose'])}</p>
  <p><strong>Model output:</strong> the model returns six numeric scores, one for each tissue class: bronchiola, alveoli, vessels, tumor, stroma, and immune infiltration.</p>
  <p><strong>Piece Top1:</strong> for each candidate, the tissue class with the highest model score is the prediction. If that predicted class equals the hidden true class, the candidate counts as correct.</p>
  <p><strong>Official-data guardrail:</strong> official FICTURE alignment status is <strong>{html.escape(official_status())}</strong>. Deprecated/debug FICTURE roots are not used.</p>
</section>
<section>
  <h2>Result Summary</h2>
  <div class="summary-grid">
    <div>
      <h3>This file</h3>
      <table><tbody><tr><th>input mode</th><td>{html.escape(mode['display'])}</td></tr><tr><th>Overall Piece Top1</th><td><strong>{html.escape(overall)}</strong></td></tr></tbody></table>
      {class_summary_table(rows)}
    </div>
    <div>
      <h3>All three clean files</h3>
      {all_mode_summary_table(mode_predictions)}
    </div>
  </div>
</section>
<section>
  <h2>Prompt Used For This File</h2>
  <p class="note">This is the exact prompt text saved beside the model output for this run. The three clean files are not mixing prompt versions.</p>
  <details class="prompt" open><summary>Show system and user prompt template</summary><pre>{html.escape(prompt)}</pre></details>
</section>
<section>
  <h2>How To Read Candidate Cards</h2>
  <p class="note">Each card shows one candidate piece. The image(s) are the model input. The score table shows the six scores returned by the model. Dice / Precision / Recall are hidden annotation-based audit values and were not shown to the model.</p>
</section>
{''.join(class_blocks)}
</main>
</body>
</html>
"""


def main() -> None:
    pool_rows = join_pool_rows()
    mode_predictions = {mode_key: load_mode_predictions(mode_key) for mode_key in MODES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode_key, mode in MODES.items():
        html_text = build_html(mode_key, pool_rows, mode_predictions)
        out = OUT_DIR / mode["filename"]
        out.write_text(html_text, encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
