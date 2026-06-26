#!/usr/bin/env python3
"""Build a clean, self-contained Jun10 skill-line candidate review HTML.

This report is intentionally separate from the older Jun10 skill HTML. It uses
the no-compartment / RGB+cell-type-only Step1 line and presents the real
three-step flow:

1. Step1 text-only cell-type composition hypotheses.
2. Step2 H&E morphology verification.
3. Step3 FICTURE image consistency check.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_NoCompartment_AlveoliRescue_Qwen3_Skill"
POOL_ROOT = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool"

REQUEST_CSV = BASE / "cell_type_first_requests.csv"
STEP1_CSV = BASE / "remote_outputs/step1_qwen3vl32b_rgbcelltype_alveoli_rescue_limit167_14616732/step1_hypotheses.csv"
STEP1_BY_CLASS_CSV = BASE / "remote_outputs/step1_qwen3vl32b_rgbcelltype_alveoli_rescue_limit167_14616732/step1_hypothesis_by_true_class.csv"
STEP1_STATUS_CSV = BASE / "remote_outputs/step1_qwen3vl32b_rgbcelltype_alveoli_rescue_limit167_14616732/step1_status_distribution.csv"

STEP2_DIR = BASE / "remote_outputs/step2_alveoli_rescue_sweep/step2_qwen3vl32b_alvRescue_comparative_alveoli_septa_rescue_heonly_limit167_14618430"
STEP2_SUMMARY_CSV = STEP2_DIR / "step2_candidate_retention_summary.csv"
STEP2_DETAIL_CSV = STEP2_DIR / "step2_hypothesis_verification_scores.csv"
STEP2_BY_CLASS_CSV = STEP2_DIR / "step2_retention_by_true_class.csv"

STEP3_DIR = BASE / "remote_outputs/step3_ficture_sweep/ficture_filter_qwen3vl32b_soft_consistency_full167_14620410"
STEP3_SUMMARY_CSV = STEP3_DIR / "ficture_candidate_retention_summary.csv"
STEP3_DETAIL_CSV = STEP3_DIR / "ficture_hypothesis_verification_scores.csv"
STEP3_BY_CLASS_CSV = STEP3_DIR / "ficture_retention_by_true_class.csv"

OUT_DIR = BASE / "Jun10_ThreeStep_CellType_HE_FICTURE_CandidateReview"
OUT_HTML = OUT_DIR / "skill_Jun10_ThreeStep_CellType_HE_FICTURE_CandidateReview_SHAREABLE.html"
ROOT_COPY = ROOT / "skill_Jun10_ThreeStep_CellType_HE_FICTURE_CandidateReview_SHAREABLE.html"

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


def fmt_float(value: object, ndigits: int = 3) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return ""


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def split_classes(value: str) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def ordered_intersection(left: list[str], right: list[str]) -> list[str]:
    left_set = set(left)
    right_set = set(right)
    return [label for label in LABELS if label in left_set and label in right_set]


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def parse_json_object(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}


def status_score(status: str) -> int:
    status = str(status).strip().lower()
    return {"plausible": 2, "uncertain": 1, "unlikely": 0}.get(status, -1)


def retained_from_step1(row: dict[str, str]) -> list[str]:
    return [label for label in LABELS if status_score(row.get(label, "")) >= 1]


def class_badges(classes: list[str], true_class: str) -> str:
    if not classes:
        return '<span class="empty">none retained</span>'
    parts = []
    for cls in classes:
        tag = " true" if cls == true_class else ""
        parts.append(f'<span class="chip{tag}">{esc(DISPLAY.get(cls, cls))}</span>')
    return " ".join(parts)


def status_badge(status: str) -> str:
    key = str(status or "missing").strip().lower().replace(" ", "-")
    return f'<span class="status {esc(key)}">{esc(status or "missing")}</span>'


def ok_badge(value: bool) -> str:
    return f'<span class="retain {str(value).lower()}">{"kept" if value else "lost"}</span>'


def load_step2_scores() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(STEP2_DETAIL_CSV):
        uid = row["candidate_uid"]
        cls = row["target_hypothesis_class"]
        out[uid][cls] = {
            "score": row.get("he_support_score", ""),
            "supported": row.get("he_supported", ""),
        }
    return out


def load_step3_scores() -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(STEP3_DETAIL_CSV):
        uid = row["candidate_uid"]
        cls = row["target_hypothesis_class"]
        out[uid][cls] = {
            "score": row.get("ficture_support_score", ""),
            "supported": row.get("ficture_supported", ""),
        }
    return out


def score_table(
    step1: dict[str, str],
    step2_scores: dict[str, dict[str, str]],
    step3_scores: dict[str, dict[str, str]],
) -> str:
    rows = []
    for label in LABELS:
        step1_status = step1.get(label, "")
        step2 = step2_scores.get(label, {})
        step3 = step3_scores.get(label, {})
        rows.append(
            "<tr>"
            f"<td>{esc(DISPLAY[label])}</td>"
            f"<td>{status_badge(step1_status)}</td>"
            f"<td>{esc(step1.get(label + '_hypothesis_score', ''))}</td>"
            f"<td>{esc(step2.get('score', ''))}</td>"
            f"<td>{esc(step2.get('supported', ''))}</td>"
            f"<td>{esc(step3.get('score', ''))}</td>"
            f"<td>{esc(step3.get('supported', ''))}</td>"
            "</tr>"
        )
    return (
        "<table class=\"score-vector\"><thead><tr>"
        "<th>class</th><th>Step1 status</th><th>Step1 score</th>"
        "<th>Step2 H&E score</th><th>Step2 keep?</th>"
        "<th>Step3 FICTURE score</th><th>Step3 keep?</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
    )


def summary_table(
    step1_rows: list[dict[str, str]],
    step2_rows: dict[str, dict[str, str]],
    step3_rows: dict[str, dict[str, str]],
) -> str:
    body = []
    totals = {"n": 0, "s1": 0, "s2": 0, "final": 0, "single": 0, "classes_left": 0}
    for cls in LABELS:
        rows = [row for row in step1_rows if row["true_class"] == cls]
        n = len(rows)
        s1 = 0
        s2 = 0
        final = 0
        single = 0
        classes_left = 0
        for row in rows:
            true_class = row["true_class"]
            step1_retained = retained_from_step1(row)
            step2_retained = split_classes(step2_rows[row["candidate_uid"]].get("he_retained_classes", ""))
            step3_supported = split_classes(step3_rows[row["candidate_uid"]].get("ficture_retained_classes", ""))
            final_retained = ordered_intersection(step2_retained, step3_supported)
            s1 += int(true_class in step1_retained)
            s2 += int(true_class in step2_retained)
            final += int(true_class in final_retained)
            single += int(len(final_retained) == 1)
            classes_left += len(final_retained)
        totals["n"] += n
        totals["s1"] += s1
        totals["s2"] += s2
        totals["final"] += final
        totals["single"] += single
        totals["classes_left"] += classes_left
        mean_left = classes_left / n if n else 0
        body.append(
            "<tr>"
            f"<td><strong>{esc(DISPLAY[cls])}</strong></td>"
            f"<td>{n}</td>"
            f"<td>{s1}/{n}</td>"
            f"<td>{s2}/{n}</td>"
            f"<td>{final}/{n}</td>"
            f"<td>{single}/{n}</td>"
            f"<td>{mean_left:.2f}</td>"
            "</tr>"
        )
    overall_mean = totals["classes_left"] / totals["n"] if totals["n"] else 0
    body.insert(
        0,
        "<tr>"
        "<td><strong>overall</strong></td>"
        f"<td>{totals['n']}</td>"
        f"<td>{totals['s1']}/{totals['n']}</td>"
        f"<td>{totals['s2']}/{totals['n']}</td>"
        f"<td>{totals['final']}/{totals['n']}</td>"
        f"<td>{totals['single']}/{totals['n']}</td>"
        f"<td>{overall_mean:.2f}</td>"
        "</tr>",
    )
    return (
        "<table><thead><tr><th>true class</th><th>n</th>"
        "<th>Step1 target kept?</th><th>Step2 target kept?</th>"
        "<th>After 3-step funnel target kept?</th><th>After 3-step funnel single-class</th>"
        "<th>Mean classes left after 3 steps</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def status_distribution_table() -> str:
    body = []
    for row in read_csv(STEP1_STATUS_CSV):
        body.append(
            "<tr>"
            f"<td>{esc(DISPLAY.get(row['class_key'], row['class_key']))}</td>"
            f"<td>{esc(row['plausible'])}</td>"
            f"<td>{esc(row['uncertain'])}</td>"
            f"<td>{esc(row['unlikely'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>class hypothesis</th><th>plausible</th>"
        "<th>uncertain</th><th>unlikely</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def prompt_text(path: Path, *, cleanup: bool = False) -> str:
    text = path.read_text(errors="replace").strip()
    if cleanup:
        text = text.replace("multiple unrelated FICTURE compartments", "multiple unrelated FICTURE color regions")
    return text


def prompt_block() -> str:
    blocks = [
        (
            "Step1 text-only prompt example",
            prompt_text(BASE / "prompt_step1_first_row.txt"),
            "Step1 receives no image. It sees only an ordered list of RGB colors, proportions, and cell-type names inside this candidate mask.",
        ),
        (
            "Step2 H&E verification prompt template",
            prompt_text(BASE / "prompt_step2_he_verification_template.txt"),
            "Step2 checks whether H&E morphology supports or rejects the Step1 hypotheses.",
        ),
        (
            "Step3 FICTURE image consistency prompt template",
            prompt_text(BASE / "prompt_step3_ficture_image_check_template.txt", cleanup=True),
            "Step3 uses the FICTURE crop as a spatial consistency check, not as the primary class decision.",
        ),
    ]
    html_blocks = []
    for title, text, note in blocks:
        html_blocks.append(
            f"<details class=\"prompt\" open><summary>{esc(title)}</summary>"
            f"<p class=\"note\">{esc(note)}</p><pre>{esc(text)}</pre></details>"
        )
    return "".join(html_blocks)


def candidate_card(
    row: dict[str, str],
    step2: dict[str, str],
    step3: dict[str, str],
    step2_scores: dict[str, dict[str, str]],
    step3_scores: dict[str, dict[str, str]],
) -> str:
    uid = row["candidate_uid"]
    true_class = row["true_class"]
    step1_retained = retained_from_step1(row)
    step2_retained = split_classes(step2.get("he_retained_classes", ""))
    step3_supported = split_classes(step3.get("ficture_retained_classes", ""))
    final_retained = ordered_intersection(step2_retained, step3_supported)

    he_path = POOL_ROOT / row["he_crop_rel"]
    fic_path = POOL_ROOT / row["ficture_crop_rel"]
    if not he_path.exists() or not fic_path.exists():
        raise FileNotFoundError(f"Missing image for {uid}: {he_path.exists()} {fic_path.exists()}")

    metric = (
        f"{fmt_float(row.get('component_dice'))} / "
        f"{fmt_float(row.get('component_precision'))} / "
        f"{fmt_float(row.get('component_recall'))}"
    )
    top_step2 = step2.get("top_he_support_class", "")
    top_step3 = step3.get("top_ficture_support_class", "")

    return (
        "<details class=\"candidate\" open>"
        "<summary>"
        f"<span class=\"uid\">{esc(uid)}</span>"
        f"<span>true class: <strong>{esc(DISPLAY[true_class])}</strong></span>"
        f"<span>funnel: {len(step1_retained)} -> {len(step2_retained)} -> {len(final_retained)} classes | target {ok_badge(true_class in final_retained)}</span>"
        f"<span>Dice / P / R: {esc(metric)}</span>"
        "</summary>"
        "<div class=\"body\">"
        "<div class=\"leftcol\">"
        "<div class=\"funnel\">"
        "<div class=\"funnel-step\"><span>Step1 text</span>"
        f"<strong>{len(step1_retained)}</strong><em>classes left</em>{ok_badge(true_class in step1_retained)}</div>"
        "<div class=\"funnel-arrow\">→</div>"
        "<div class=\"funnel-step\"><span>Step2 H&amp;E</span>"
        f"<strong>{len(step2_retained)}</strong><em>classes left</em>{ok_badge(true_class in step2_retained)}</div>"
        "<div class=\"funnel-arrow\">→</div>"
        "<div class=\"funnel-step final\"><span>After 3 steps</span>"
        f"<strong>{len(final_retained)}</strong><em>classes left</em>{ok_badge(true_class in final_retained)}</div>"
        "</div>"
        "<table class=\"meta\"><tbody>"
        f"<tr><th>hidden true class</th><td>{esc(DISPLAY[true_class])}</td></tr>"
        f"<tr><th>classes left after 3 steps</th><td><strong>{len(final_retained)}</strong></td></tr>"
        f"<tr><th>target in final set?</th><td>{'yes' if true_class in final_retained else 'no'}</td></tr>"
        f"<tr><th>component Dice / Precision / Recall</th><td>{esc(metric)}</td></tr>"
        f"<tr><th>source</th><td>{esc(row.get('source', ''))}</td></tr>"
        f"<tr><th>setting</th><td>{esc(row.get('setting', ''))}</td></tr>"
        f"<tr><th>candidate id</th><td>{esc(row.get('candidate_id', ''))}</td></tr>"
        f"<tr><th>Step2 top H&E support</th><td>{esc(DISPLAY.get(top_step2, top_step2))} ({esc(step2.get('top_he_support_score', ''))})</td></tr>"
        f"<tr><th>Step3 top FICTURE support</th><td>{esc(DISPLAY.get(top_step3, top_step3))} ({esc(step3.get('top_ficture_support_score', ''))})</td></tr>"
        "</tbody></table>"
        "<div class=\"stepbox\"><h4>Step1 possible classes</h4>"
        f"<p>{class_badges(step1_retained, true_class)}</p>"
        "<p class=\"tiny\">From text-only RGB/cell-type composition. Plausible or uncertain classes are kept.</p></div>"
        "<div class=\"stepbox\"><h4>Step2 H&E-retained classes</h4>"
        f"<p>{class_badges(step2_retained, true_class)}</p>"
        "<p class=\"tiny\">From H&E morphology verification of the Step1 hypotheses.</p></div>"
        "<div class=\"stepbox\"><h4>After 3-step funnel</h4>"
        f"<p>{class_badges(final_retained, true_class)}</p>"
        "<p class=\"tiny\">Final retained set = Step2 H&E-retained classes intersected with Step3 FICTURE-supported classes.</p>"
        f"<p class=\"tiny\">Step3 FICTURE-supported before intersection: {class_badges(step3_supported, true_class)}</p></div>"
        f"{score_table(row, step2_scores, step3_scores)}"
        "<details class=\"inner\"><summary>Step1 ordered RGB/cell-type composition list</summary>"
        f"<pre>{esc(row.get('cell_type_list', ''))}</pre></details>"
        "<details class=\"inner\"><summary>Raw model outputs used for audit</summary>"
        f"<h4>Step1 raw JSON</h4><pre>{esc(row.get('raw_response', ''))}</pre>"
        f"<h4>Step2 raw JSON</h4><pre>{esc(step2.get('raw_response', ''))}</pre>"
        "</details>"
        "</div>"
        "<div class=\"figgrid mode-2\">"
        f"<figure><img src=\"{data_uri(he_path)}\" alt=\"{esc(uid)} H&E crop\"><figcaption>H&amp;E gray reverse-blur candidate crop</figcaption></figure>"
        f"<figure><img src=\"{data_uri(fic_path)}\" alt=\"{esc(uid)} FICTURE crop\"><figcaption>FICTURE gray reverse-blur candidate crop</figcaption></figure>"
        "</div>"
        "</div>"
        "</details>"
    )


def build_html() -> str:
    request_rows = {r["candidate_uid"]: r for r in read_csv(REQUEST_CSV)}
    step1_rows = read_csv(STEP1_CSV)
    step2_rows = {r["candidate_uid"]: r for r in read_csv(STEP2_SUMMARY_CSV)}
    step3_rows = {r["candidate_uid"]: r for r in read_csv(STEP3_SUMMARY_CSV)}
    step2_scores = load_step2_scores()
    step3_scores = load_step3_scores()

    if len(step1_rows) != 167:
        raise RuntimeError(f"Expected 167 Step1 rows, got {len(step1_rows)}")
    for row in step1_rows:
        uid = row["candidate_uid"]
        if uid not in request_rows or uid not in step2_rows or uid not in step3_rows:
            raise RuntimeError(f"Missing joined row for {uid}")

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in step1_rows:
        by_class[row["true_class"]].append(row)
    for rows in by_class.values():
        rows.sort(key=lambda r: float(r.get("component_dice", "0") or 0), reverse=True)

    sections = []
    for cls in LABELS:
        rows = by_class[cls]
        cards = "\n".join(
            candidate_card(
                row,
                step2_rows[row["candidate_uid"]],
                step3_rows[row["candidate_uid"]],
                step2_scores.get(row["candidate_uid"], {}),
                step3_scores.get(row["candidate_uid"], {}),
            )
            for row in rows
        )
        sections.append(
            f"<section id=\"{esc(cls)}\"><h2>{esc(DISPLAY[cls])} <span>{len(rows)} candidates</span></h2>{cards}</section>"
        )

    nav = " ".join(f"<a href=\"#{label}\">{esc(DISPLAY[label])}</a>" for label in LABELS)
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f8fafc;color:#111827;line-height:1.48}
header{background:#111827;color:white;padding:30px 42px}
main{max-width:1500px;margin:0 auto;padding:28px 34px 70px}
h1{margin:0 0 8px;font-size:30px} h2{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
h2 span{font-size:15px;color:#64748b;font-weight:500;margin-left:8px}
h3{margin-top:22px} h4{margin:12px 0 8px} code,pre,.uid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sub{color:#cbd5e1}.muted{color:#64748b}.tiny{font-size:12px;color:#64748b;margin:4px 0 0}
.callout{background:#ecfeff;border-left:5px solid #0891b2;padding:13px 15px;border-radius:7px;color:#164e63}
.warn{background:#fff7ed;border-left:5px solid #f97316;padding:13px 15px;border-radius:7px;color:#431407}
.note{color:#334155;background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 14px;border-radius:6px}
nav{position:sticky;top:0;background:rgba(248,250,252,.96);backdrop-filter:blur(8px);padding:12px 0;z-index:10;border-bottom:1px solid #e5e7eb}
nav a{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;border:1px solid #cbd5e1;border-radius:999px;color:#0f172a;text-decoration:none;background:white;font-size:14px}
table{border-collapse:collapse;width:100%;background:white;margin:12px 0 18px;border:1px solid #e5e7eb}
th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f1f5f9}
.summary-grid{display:grid;grid-template-columns:minmax(300px,1fr) minmax(300px,1fr);gap:22px}
details.prompt{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin-top:14px;padding:0}
details.prompt summary{padding:12px 14px;font-weight:700;cursor:pointer;background:#f8fafc}
details.inner{background:#fff;border:1px solid #dbe2ea;border-radius:8px;margin:10px 0;padding:0}
details.inner summary{padding:9px 11px;font-weight:650;cursor:pointer;background:#f8fafc;color:#1d4ed8}
pre{white-space:pre-wrap;background:#0f172a;color:#e5e7eb;padding:14px;margin:0;border-radius:0 0 8px 8px;max-height:420px;overflow:auto;font-size:12px}
.candidate{background:white;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0;overflow:hidden}
.candidate summary{cursor:pointer;padding:12px 14px;display:grid;grid-template-columns:1.15fr .7fr 1.15fr .8fr;gap:10px;align-items:center;background:#fff}
.candidate summary:hover{background:#f8fafc}
.uid{font-weight:700;color:#1d4ed8}
.body{display:grid;grid-template-columns:520px 1fr;gap:16px;padding:14px;border-top:1px solid #e5e7eb}
.meta,.score-vector{font-size:13px;margin:0 0 12px}.meta th,.meta td,.score-vector th,.score-vector td{padding:5px 7px}
.meta th{width:185px}.score-vector th,.score-vector td{font-size:12px}
.retain{display:inline-block;padding:3px 7px;border-radius:999px;font-size:12px;font-weight:700}.retain.true{background:#dcfce7;color:#166534}.retain.false{background:#fee2e2;color:#991b1b}
.chip{display:inline-block;padding:4px 8px;border-radius:999px;background:#f1f5f9;border:1px solid #cbd5e1;margin:3px 3px 3px 0;font-size:12px}.chip.true{background:#dcfce7;border-color:#22c55e;color:#166534;font-weight:700}
.status{display:inline-block;padding:3px 7px;border-radius:999px;font-size:12px;border:1px solid #cbd5e1}.status.plausible{background:#dcfce7;color:#166534}.status.uncertain{background:#fef9c3;color:#854d0e}.status.unlikely{background:#fee2e2;color:#991b1b}.status.missing{background:#f1f5f9;color:#64748b}
.empty{color:#64748b;font-style:italic}.stepbox{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px;margin:10px 0}.stepbox h4{margin-top:0}
.funnel{display:grid;grid-template-columns:1fr 26px 1fr 26px 1fr;gap:6px;align-items:stretch;margin:0 0 12px}
.funnel-step{border:1px solid #cbd5e1;border-radius:8px;background:#fff;padding:9px;text-align:center}.funnel-step.final{background:#ecfeff;border-color:#0891b2}.funnel-step span{display:block;font-size:12px;color:#475569}.funnel-step strong{display:block;font-size:24px;line-height:1.1}.funnel-step em{display:block;font-size:11px;color:#64748b;font-style:normal;margin-bottom:4px}.funnel-arrow{text-align:center;font-size:22px;color:#64748b;align-self:center}
.figgrid{display:grid;gap:16px}.figgrid.mode-2{grid-template-columns:repeat(2,minmax(300px,1fr))}
figure{margin:0} img{width:100%;height:auto;display:block;border:1px solid #e5e7eb;background:white} figcaption{font-size:13px;color:#475569;margin-top:6px}
@media(max-width:1050px){.summary-grid,.body,.candidate summary,.figgrid.mode-2,.funnel{grid-template-columns:1fr}.funnel-arrow{display:none}main{padding:18px}}
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>skill Jun10 Three-Step Cell-Type H&E FICTURE Candidate Review</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>skill Jun10 Three-Step Cell-Type + H&amp;E + FICTURE Candidate Review</h1>
  <div class="sub">Qwen3-VL-32B | 167 candidate pieces | Step1 text-only cell composition -> Step2 H&amp;E morphology -> Step3 FICTURE image consistency | self-contained shareable HTML</div>
</header>
<main>
<nav>{nav}</nav>

<section>
  <h2>Experiment Design</h2>
  <p><strong>Goal:</strong> test a skill-style candidate judge. Instead of asking the VLM to directly choose one tissue class from two images, the task is split into three simpler checks.</p>
  <p><strong>Candidate pool:</strong> 167 compact piece-level candidates from the official same-ROI H&amp;E + FICTURE pool. Each card below is one candidate piece, not a final union mask.</p>
  <p><strong>Step1:</strong> use only the candidate's ordered <strong>RGB + cell type + proportion</strong> list. No images are shown. The output is a hypothesis set: plausible, uncertain, or unlikely for each tissue class.</p>
  <p><strong>Step2:</strong> use the H&amp;E candidate crop to verify morphology. The output is the subset of Step1 classes that still look visually supported in H&amp;E.</p>
  <p><strong>Step3:</strong> use the FICTURE crop as a spatial consistency check. The final retained set after all three steps is computed as <strong>Step2 H&amp;E-retained classes ∩ Step3 FICTURE-supported classes</strong>.</p>
  <p><strong>Ground truth use:</strong> hidden annotation-derived labels and component Dice / Precision / Recall are not given to the model. They are shown here only so we can audit which step kept or lost the correct class.</p>
  <p><strong>How to read retained classes:</strong> retained means “kept as a possible class after this step.” If a class is missing from a step, that step filtered it out for this candidate. It is not a biological impossibility claim.</p>
  <p class="callout"><strong>Main point:</strong> this HTML is now a candidate-level funnel audit. For every candidate, the card shows how many classes remain after Step1, after Step2, and after the final three-step intersection, plus whether the hidden target class is still inside the remaining set.</p>
</section>

<section>
  <h2>Result Summary</h2>
  <div class="summary-grid">
    <div>
      <h3>Funnel retention by hidden true class</h3>
      {summary_table(step1_rows, step2_rows, step3_rows)}
    </div>
    <div>
      <h3>Step1 status distribution</h3>
      {status_distribution_table()}
    </div>
  </div>
  <p class="warn"><strong>Important:</strong> Step1 is intentionally broad, especially for vessels, stroma, and immune infiltration. The key question is whether the final three-step funnel can narrow the hypotheses while keeping the hidden target class.</p>
</section>

<section>
  <h2>Prompt Examples</h2>
  <p class="note">The displayed prompt uses RGB and cell type only. The older major-class field is not included in this skill line.</p>
  {prompt_block()}
</section>

<section>
  <h2>Candidate-Level Review</h2>
  <p class="note">Cards are grouped by hidden true class and sorted by component Dice. Open a card to see the two input crops, the funnel count, whether the hidden target survived the funnel, the final classes left after all three steps, the ordered cell-type list, and six-class support scores.</p>
</section>

{''.join(sections)}

<section>
  <h2>Validation</h2>
  <p>Candidate cards: 167. Embedded images: 334. Image references are embedded as base64 data URIs, so this file can be shared without an assets folder.</p>
</section>
</main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = build_html()
    OUT_HTML.write_text(text, encoding="utf-8")
    ROOT_COPY.write_text(text, encoding="utf-8")

    local_tokens = ["src=\"output/", "src=\"/Users/", "href=\"/Users/"]
    present = [token for token in local_tokens if token in text]
    if present:
        raise RuntimeError(f"HTML contains local references: {present}")
    if "major compartment" in text.lower() or "compartment:" in text.lower():
        raise RuntimeError("HTML still contains major-compartment wording")

    print(f"wrote {OUT_HTML}")
    print(f"wrote {ROOT_COPY}")
    print(f"candidate cards: {text.count('<details class=\"candidate\"')}")
    print(f"embedded images: {text.count('data:image/')}")
    print(f"size_mb: {OUT_HTML.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
