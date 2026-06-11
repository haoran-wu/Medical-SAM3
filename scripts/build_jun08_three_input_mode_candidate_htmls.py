#!/usr/bin/env python3
"""Build self-contained 167-candidate visual audit HTMLs for three input modes."""

from __future__ import annotations

import base64
import csv
import html
from pathlib import Path


POOL_DIR = Path(
    "/Users/haoranwu/Desktop/Yan_Lab_Research/"
    "VisiumHD-Segmentation-MaskSelection/data/visium_hd_exp1/"
    "current_ficture_vlm_inputs/compact167_reverse_blur_pool"
)
OUT_DIR = Path(
    "/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/"
    "output/visium_hd_exp1/final_deliverables/"
    "Jun08_three_input_mode_candidate_htmls"
)
SCORE_DIR = Path(
    "/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/"
    "output/visium_hd_exp1/final_deliverables/"
    "Jun08_three_input_mode_candidate_scores/raw_remote_scores"
)

CLASSES = [
    "bronchiola",
    "alveoli",
    "vessels",
    "tumor",
    "stroma",
    "immune infiltration",
]

SUMMARY = {
    "H&E only": {
        "overall": "71/167",
        "bronchiola": "8/14",
        "alveoli": "0/5",
        "vessels": "17/24",
        "tumor": "10/22",
        "stroma": "0/32",
        "immune infiltration": "36/70",
    },
    "FICTURE only": {
        "overall": "22/167",
        "bronchiola": "0/14",
        "alveoli": "0/5",
        "vessels": "0/24",
        "tumor": "21/22",
        "stroma": "1/32",
        "immune infiltration": "0/70",
    },
    "H&E + FICTURE": {
        "overall": "33/167",
        "bronchiola": "0/14",
        "alveoli": "0/5",
        "vessels": "11/24",
        "tumor": "22/22",
        "stroma": "0/32",
        "immune infiltration": "0/70",
    },
}

MODE_IMAGES = {
    "H&E only": [("H&E gray reverse-blur crop", "he_crop_rel")],
    "FICTURE only": [("FICTURE gray reverse-blur crop", "ficture_crop_rel")],
    "H&E + FICTURE": [
        ("H&E gray reverse-blur crop", "he_crop_rel"),
        ("FICTURE gray reverse-blur crop", "ficture_crop_rel"),
    ],
}

MODE_FILENAMES = {
    "H&E only": "Jun08_HE_only_167_candidates.html",
    "FICTURE only": "Jun08_FICTURE_only_167_candidates.html",
    "H&E + FICTURE": "Jun08_HE_plus_FICTURE_167_candidates.html",
}

MODE_SCORE_DIRS = {
    "H&E only": "qwen3vl32b_HE_only_no_locator_13857226_0",
    "FICTURE only": "qwen3vl32b_FICTURE_only_no_locator_13857226_1",
    "H&E + FICTURE": "qwen3vl32b_HE_FICTURE_no_locator_13857226_2",
}

SCORE_CLASSES = [
    ("bronchiola", "bronchiola"),
    ("alveoli", "alveoli"),
    ("vessels", "vessels"),
    ("tumor", "tumor"),
    ("stroma", "stroma"),
    ("immune_infiltration", "immune infiltration"),
]


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


def join_rows() -> list[dict[str, str]]:
    public_rows = read_csv(POOL_DIR / "public_vlm_requests.csv")
    hidden_rows = read_csv(POOL_DIR / "hidden_candidate_truth.csv")
    hidden_by_uid = {row["candidate_uid"]: row for row in hidden_rows}
    rows: list[dict[str, str]] = []
    for row in public_rows:
        merged = dict(row)
        merged.update(hidden_by_uid.get(row["candidate_uid"], {}))
        # Keep public crop paths, because hidden truth intentionally blanks them.
        merged["he_crop_rel"] = row["he_crop_rel"]
        merged["ficture_crop_rel"] = row["ficture_crop_rel"]
        rows.append(merged)
    return rows


def read_scores() -> dict[str, dict[str, dict[str, str]]]:
    scores: dict[str, dict[str, dict[str, str]]] = {}
    for mode, dirname in MODE_SCORE_DIRS.items():
        path = SCORE_DIR / dirname / "per_candidate_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing score CSV for {mode}: {path}")
        rows = read_csv(path)
        if len(rows) != 167:
            raise ValueError(f"Expected 167 score rows for {mode}, got {len(rows)}")
        scores[mode] = {row["candidate_uid"]: row for row in rows}
    return scores


def summary_table(mode: str) -> str:
    cells = ["<tr><th>Class</th><th>Piece Top1</th></tr>"]
    for cls in CLASSES:
        cells.append(
            f"<tr><td>{html.escape(cls)}</td><td><b>{SUMMARY[mode][cls]}</b></td></tr>"
        )
    return (
        f"<table class=\"summary\"><tr><th>Input mode</th>"
        f"<td>{html.escape(mode)}</td></tr><tr><th>Overall Piece Top1</th>"
        f"<td><b>{SUMMARY[mode]['overall']}</b></td></tr></table>"
        f"<table>{''.join(cells)}</table>"
    )


def score_note() -> str:
    return (
        "<p class=\"note\">Piece Top1 means: for each candidate piece, the model outputs "
        "six tissue-class scores; the class with the highest score is treated as the "
        "prediction. If that predicted class equals the hidden true tissue class, this "
        "candidate counts as one Piece Top1 correct case. The cards below show every "
        "candidate image, the model's six predicted scores, and hidden evaluation "
        "metadata used only for audit."
        "</p>"
    )


def score_box(score: dict[str, str] | None) -> str:
    if score is None:
        return "<div class=\"scorebox missing\">Missing score row for this candidate.</div>"
    ok = str(score.get("is_correct", "")).strip().lower() == "true"
    ok_class = "ok" if ok else "bad"
    rows = []
    for key, label in SCORE_CLASSES:
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(score.get(key, ''))}</td></tr>"
        )
    pred = score.get("predicted_class", "").replace("_", " ")
    return f"""
  <div class="scorebox">
    <h3>Qwen3-VL-32B predicted scores</h3>
    <table class="scoretable compact">
      <tbody>
        <tr><th>predicted class</th><td class="{ok_class}">{html.escape(pred or 'tie / empty')}</td></tr>
        <tr><th>Piece Top1 correct?</th><td class="{ok_class}">{html.escape(score.get('is_correct',''))}</td></tr>
        <tr><th>top score / tie</th><td>{html.escape(score.get('top_score',''))} / {html.escape(score.get('top_score_tie',''))}</td></tr>
        <tr><th>parse status</th><td>{html.escape(score.get('parse_status',''))}</td></tr>
      </tbody>
    </table>
    <table class="scoretable">
      <thead><tr><th>class</th><th>score</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
"""


def candidate_card(row: dict[str, str], mode: str, score: dict[str, str] | None) -> str:
    uid = row["candidate_uid"]
    display = row.get("display", "")
    comp = row.get("matched_annotation_component_id", "")
    dice = fmt_float(row.get("component_dice", ""))
    precision = fmt_float(row.get("component_precision", ""))
    recall = fmt_float(row.get("component_recall", ""))
    area = row.get("piece_area") or row.get("candidate_area") or ""
    source = row.get("parent_member_sources", "")
    cluster_size = row.get("parent_cluster_size", "")

    figures = []
    for label, col in MODE_IMAGES[mode]:
        rel = row[col]
        path = POOL_DIR / rel
        if path.exists():
            figures.append(
                "<figure>"
                f"<img src=\"{data_uri(path)}\" alt=\"{html.escape(uid)} {html.escape(label)}\">"
                f"<figcaption>{html.escape(label)}</figcaption>"
                "</figure>"
            )
        else:
            figures.append(
                f"<div class=\"missing\">Missing image: {html.escape(str(path))}</div>"
            )
    return f"""
<details class="candidate" open>
  <summary>
    <span class="uid">{html.escape(uid)}</span>
    <span>{html.escape(display)}</span>
    <span>component {html.escape(comp)}</span>
    <span>Dice {dice} / P {precision} / R {recall}</span>
  </summary>
  <div class="meta">
    <b>candidate_id:</b> {html.escape(row.get('candidate_id',''))}
    <b>cluster_size:</b> {html.escape(cluster_size)}
    <b>source:</b> {html.escape(source)}
    <b>area:</b> {html.escape(area)}
  </div>
  {score_box(score)}
  <div class="figgrid mode-{len(MODE_IMAGES[mode])}">
    {''.join(figures)}
  </div>
</details>
"""


def build_html(mode: str, rows: list[dict[str, str]], scores: dict[str, dict[str, dict[str, str]]]) -> str:
    class_blocks = []
    for cls in CLASSES:
        cls_rows = [row for row in rows if row.get("display") == cls]
        cards = "\n".join(
            candidate_card(row, mode, scores[mode].get(row["candidate_uid"]))
            for row in cls_rows
        )
        class_blocks.append(
            f"<section class=\"class-block\" id=\"{html.escape(cls.replace(' ', '_'))}\">"
            f"<h2>{html.escape(cls)} <span>{len(cls_rows)} candidates</span></h2>"
            f"{cards}</section>"
        )
    nav = " ".join(
        f"<a href=\"#{html.escape(cls.replace(' ', '_'))}\">{html.escape(cls)}</a>"
        for cls in CLASSES
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Jun08 {html.escape(mode)} 167 Candidate Visual Audit</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.5}}
header{{background:#0f172a;color:white;padding:28px 36px}}
main{{max-width:1400px;margin:0 auto;padding:28px 32px 60px}}
h1{{margin:0 0 8px;font-size:30px}} h2{{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}}
h2 span{{font-size:15px;color:#64748b;font-weight:500;margin-left:8px}}
.kicker{{color:#cbd5e1}} .note{{color:#475569;background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 14px;border-radius:6px}}
nav{{position:sticky;top:0;background:rgba(248,250,252,.96);backdrop-filter:blur(8px);padding:12px 0;z-index:10;border-bottom:1px solid #e5e7eb}}
nav a{{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;border:1px solid #cbd5e1;border-radius:999px;color:#0f172a;text-decoration:none;background:white;font-size:14px}}
table{{border-collapse:collapse;width:100%;background:white;margin:16px 0 22px;border:1px solid #e5e7eb}}
th,td{{border-bottom:1px solid #e5e7eb;padding:10px 12px;text-align:left}} th{{background:#f1f5f9}}
.summary{{max-width:560px}}
.candidate{{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0;overflow:hidden}}
summary{{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.3fr .9fr .9fr 1.2fr;gap:10px;align-items:center;background:#fff}}
summary:hover{{background:#f8fafc}} .uid{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-weight:700;color:#1d4ed8}}
.meta{{padding:10px 14px;color:#475569;border-top:1px solid #e5e7eb;font-size:14px;display:flex;gap:14px;flex-wrap:wrap}}
.scorebox{{margin:12px 14px 0;padding:12px;border:1px solid #e5e7eb;border-radius:8px;background:#fcfcfd;max-width:560px}}
.scorebox h3{{margin:0 0 8px;font-size:16px}}
.scoretable{{margin:8px 0 0;font-size:13px}}
.scoretable th,.scoretable td{{padding:5px 7px}}
.scoretable.compact th{{width:180px}}
.ok{{color:#047857;font-weight:700}}
.bad{{color:#b91c1c;font-weight:700}}
.figgrid{{display:grid;gap:16px;padding:14px;border-top:1px solid #e5e7eb}}
.figgrid.mode-1{{grid-template-columns:minmax(320px,620px)}}
.figgrid.mode-2{{grid-template-columns:repeat(2,minmax(300px,1fr))}}
figure{{margin:0}} img{{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white}} figcaption{{font-size:13px;color:#475569;margin-top:6px}}
.missing{{color:#b91c1c;background:#fee2e2;padding:10px;border-radius:6px}}
@media(max-width:900px){{summary{{grid-template-columns:1fr}}.figgrid.mode-2{{grid-template-columns:1fr}}main{{padding:18px}}}}
</style>
</head>
<body>
<header>
  <h1>Jun08 {html.escape(mode)} 167 Candidate Visual Audit</h1>
  <div class="kicker">VisiumHD Exp1 | compact 167 piece-first pool | medical_official_points_step24 | self-contained HTML</div>
</header>
<main>
<nav>{nav}</nav>
<section>
<h2>Result Summary</h2>
{summary_table(mode)}
{score_note()}
<p class="note">Candidate source: <code>{html.escape(str(POOL_DIR))}</code>. Images are gray reverse-blur crops. Scores are from <code>{html.escape(str(SCORE_DIR / MODE_SCORE_DIRS[mode] / 'per_candidate_predictions.csv'))}</code>. Hidden Dice / Precision / Recall below are for audit only and were not shown to the model.</p>
</section>
{''.join(class_blocks)}
</main>
</body>
</html>
"""


def main() -> None:
    rows = join_rows()
    if len(rows) != 167:
        raise SystemExit(f"Expected 167 rows, got {len(rows)}")
    scores = read_scores()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode, filename in MODE_FILENAMES.items():
        out = OUT_DIR / filename
        out.write_text(build_html(mode, rows, scores), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
