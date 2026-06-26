#!/usr/bin/env python3
"""Build a candidate-by-candidate Jun10 three-step review HTML.

The report joins:
1. Step1 cell-type-text hypotheses.
2. Step2 H&E morphology retained hypotheses.
3. Step3 FICTURE image checks and the current safe final policy.

It intentionally shows all candidate images and hidden true labels so the user
can inspect where the three-step skill keeps or loses each tissue class.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path
from typing import Iterable


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
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


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_classes(value: str) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def class_list_html(classes: list[str], true_class: str) -> str:
    if not classes:
        return '<span class="empty">none</span>'
    parts = []
    for cls in classes:
        mark = " true" if cls == true_class else ""
        parts.append(f'<span class="chip{mark}">{html.escape(DISPLAY.get(cls, cls))}</span>')
    return " ".join(parts)


def ordered_intersection(left: Iterable[str], right: Iterable[str]) -> list[str]:
    left_set = set(left)
    right_set = set(right)
    return [cls for cls in CLASS_KEYS if cls in left_set and cls in right_set]


def status_chip(status: str) -> str:
    safe = html.escape(str(status))
    cls = str(status).lower().replace(" ", "-")
    return f'<span class="status {cls}">{safe}</span>'


def find_image(image_roots: list[Path], rel_path: str, candidate_uid: str, suffix: str) -> Path | None:
    rel = Path(rel_path)
    candidates = []
    for root in image_roots:
        candidates.append(root / rel.name)
        candidates.append(root / rel)
        candidates.append(root / "candidate_pair_crops" / rel.name)
    candidates.extend(Path(".").glob(f"output/visium_hd_exp1/final_deliverables/**/{candidate_uid}*{suffix}*.png"))
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def copy_image(src: Path | None, out_assets: Path, filename: str) -> str:
    if src is None or not src.exists():
        return ""
    out_assets.mkdir(parents=True, exist_ok=True)
    dst = out_assets / filename
    if not dst.exists():
        shutil.copy2(src, dst)
    return f"assets/candidate_pair_crops/{filename}"


def load_step3_rows(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for row in read_csv(path):
        uid = row["candidate_uid"]
        policy = row["policy"]
        out.setdefault(uid, {})[policy] = {
            "kept_classes": split_classes(row.get("kept_classes", "")),
            "n_classes_left": int(row.get("n_classes_left") or 0),
            "true_retained": str(row.get("true_retained", "")).lower() == "true",
        }
    return out


def load_step2_policy(path: Path, policy_name: str) -> dict[str, dict[str, object]]:
    out = {}
    for row in read_csv(path):
        if row.get("policy") != policy_name:
            continue
        out[row["candidate_uid"]] = {
            "kept_classes": split_classes(row.get("kept_classes", "")),
            "n_classes_left": int(row.get("n_classes_left") or 0),
            "true_retained": str(row.get("true_retained", "")).lower() == "true",
        }
    return out


def load_hypothesis_scores(path: Path, score_col: str, target_col: str) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    if not path.exists():
        return scores
    for row in read_csv(path):
        uid = row["candidate_uid"]
        target = row[target_col]
        scores.setdefault(uid, {})[target] = str(row.get(score_col, ""))
    return scores


def classes_from_step1(row: dict[str, str]) -> list[str]:
    kept = []
    for cls in CLASS_KEYS:
        if str(row.get(cls, "")).lower() in {"plausible", "uncertain"}:
            kept.append(cls)
    return kept


def class_status_table(row: dict[str, str], he_scores: dict[str, str], fic_scores: dict[str, str]) -> str:
    lines = [
        "<table class=\"mini\"><thead><tr><th>class</th><th>Step1</th><th>Step1 score</th><th>H&E score</th><th>FICTURE score</th></tr></thead><tbody>"
    ]
    for cls in CLASS_KEYS:
        lines.append(
            "<tr>"
            f"<td>{html.escape(DISPLAY[cls])}</td>"
            f"<td>{status_chip(row.get(cls, ''))}</td>"
            f"<td>{html.escape(str(row.get(cls + '_hypothesis_score', '')))}</td>"
            f"<td>{html.escape(str(he_scores.get(cls, '')))}</td>"
            f"<td>{html.escape(str(fic_scores.get(cls, '')))}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def summary_table(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return ""
    fieldnames = list(rows[0])
    lines = ["<table><thead><tr>"]
    for field in fieldnames:
        lines.append(f"<th>{html.escape(field.replace('_', ' '))}</th>")
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>")
        for field in fieldnames:
            lines.append(f"<td>{html.escape(str(row.get(field, '')))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def prompt_block(title: str, path: Path, note: str) -> str:
    if not path.exists():
        return (
            f"<details open><summary>{html.escape(title)}</summary>"
            f"<p class=\"note\">Prompt file not found: <code>{html.escape(str(path))}</code></p>"
            "</details>"
        )
    text = path.read_text()
    return (
        f"<details open><summary>{html.escape(title)}</summary>"
        f"<p class=\"note\">{html.escape(note)}</p>"
        f"<pre>{html.escape(text)}</pre>"
        "</details>"
    )


def pipeline_summary_tables(summary_rows: list[dict[str, object]]) -> tuple[str, str]:
    step_defs = [
        ("Step1 cell composition text", "step1_possible", "step1_true_retained"),
        ("Step2 H&E morphology check", "step2_he_possible", "step2_true_retained"),
        ("Step3 FICTURE check after H&E", "step3_ficture_possible", "step3_true_retained"),
    ]
    lines = [
        "<table><thead><tr><th>pipeline step</th><th>true class retained</th><th>mean classes left</th><th>what it means</th></tr></thead><tbody>"
    ]
    meanings = {
        "Step1 cell composition text": "Uses only the ordered RGB / cell-type / marker-gene composition text to keep broad biological hypotheses.",
        "Step2 H&E morphology check": "Uses the H&E candidate crop to remove Step1 hypotheses without visible tissue-structure support.",
        "Step3 FICTURE check after H&E": "Uses the FICTURE crop to check the remaining H&E-supported hypotheses; displayed list is Step2 retained classes intersected with FICTURE-supported classes.",
    }
    n = len(summary_rows)
    for label, list_col, retain_col in step_defs:
        retained = sum(1 for row in summary_rows if row[retain_col])
        mean_left = sum(len(split_classes(str(row[list_col]))) for row in summary_rows) / n if n else 0
        lines.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{retained}/{n}</td>"
            f"<td>{mean_left:.2f}</td>"
            f"<td>{html.escape(meanings[label])}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    overall_table = "\n".join(lines)

    class_lines = [
        "<table><thead><tr><th>true class</th><th>n</th><th>Step1 retained</th><th>Step2 retained</th><th>Step3 retained</th><th>Step3 mean classes left</th></tr></thead><tbody>"
    ]
    for cls in CLASS_KEYS:
        rows = [row for row in summary_rows if row["true_class"] == cls]
        denom = len(rows)
        step1 = sum(1 for row in rows if row["step1_true_retained"])
        step2 = sum(1 for row in rows if row["step2_true_retained"])
        step3 = sum(1 for row in rows if row["step3_true_retained"])
        mean_step3 = sum(len(split_classes(str(row["step3_ficture_possible"]))) for row in rows) / denom if denom else 0
        class_lines.append(
            "<tr>"
            f"<td>{html.escape(DISPLAY[cls])}</td>"
            f"<td>{denom}</td>"
            f"<td>{step1}/{denom}</td>"
            f"<td>{step2}/{denom}</td>"
            f"<td>{step3}/{denom}</td>"
            f"<td>{mean_step3:.2f}</td>"
            "</tr>"
        )
    class_lines.append("</tbody></table>")
    return overall_table, "\n".join(class_lines)


def build(args: argparse.Namespace) -> Path:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets" / "candidate_pair_crops"

    step1_rows = read_csv(args.step1_csv)
    step2_policy = load_step2_policy(args.step2_policy_csv, args.step2_policy)
    step3_rows = load_step3_rows(args.step3_rows_csv)
    he_scores = load_hypothesis_scores(args.step2_scores_csv, "he_support_score", "target_hypothesis_class")
    fic_scores = load_hypothesis_scores(args.step3_scores_csv, "ficture_support_score", "target_hypothesis_class")
    image_roots = [Path(p) for p in args.image_root]

    summary_rows: list[dict[str, object]] = []
    cards = []
    missing_images: list[str] = []
    for index, row in enumerate(step1_rows, start=1):
        uid = row["candidate_uid"]
        true_class = row["true_class"]
        he_src = find_image(image_roots, row.get("he_crop_rel", ""), uid, "he")
        fic_src = find_image(image_roots, row.get("ficture_crop_rel", ""), uid, "ficture")
        if he_src is None:
            missing_images.append(f"{uid}: H&E")
        if fic_src is None:
            missing_images.append(f"{uid}: FICTURE")
        he_rel = copy_image(he_src, assets_dir, f"{uid}_he_reverse_blur_gray.png")
        fic_rel = copy_image(fic_src, assets_dir, f"{uid}_ficture_reverse_blur_gray.png")

        step1_classes = classes_from_step1(row)
        step2_classes = list(step2_policy.get(uid, {}).get("kept_classes", []))
        ficture_supported_classes = list(step3_rows.get(uid, {}).get("step3_FICTURE_only", {}).get("kept_classes", []))
        step3_classes = ordered_intersection(step2_classes, ficture_supported_classes)

        summary_rows.append(
            {
                "candidate_uid": uid,
                "true_class": true_class,
                "component_dice": row.get("component_dice", ""),
                "component_precision": row.get("component_precision", ""),
                "component_recall": row.get("component_recall", ""),
                "step1_possible": ";".join(step1_classes),
                "step2_he_possible": ";".join(step2_classes),
                "ficture_image_supported_before_intersection": ";".join(ficture_supported_classes),
                "step3_ficture_possible": ";".join(step3_classes),
                "step1_true_retained": true_class in step1_classes,
                "step2_true_retained": true_class in step2_classes,
                "step3_true_retained": true_class in step3_classes,
            }
        )

        retain_badges = []
        for label, classes in [
            ("Step1", step1_classes),
            ("Step2", step2_classes),
            ("Step3", step3_classes),
        ]:
            ok = true_class in classes
            retain_badges.append(f'<span class="retain {str(ok).lower()}">{html.escape(label)} true retained: {ok}</span>')

        cards.append(
            f"""
<section class="card" id="{html.escape(uid)}" data-class="{html.escape(true_class)}">
  <div class="card-head">
    <div>
      <h3>{index}. {html.escape(uid)}</h3>
      <p class="sub">hidden true class: <strong>{html.escape(DISPLAY[true_class])}</strong> | component Dice / Precision / Recall:
      {float(row.get('component_dice') or 0):.3f} / {float(row.get('component_precision') or 0):.3f} / {float(row.get('component_recall') or 0):.3f}</p>
    </div>
    <div class="badges">{" ".join(retain_badges)}</div>
  </div>
  <div class="images">
    <figure><img src="{html.escape(he_rel)}" alt="H&E crop for {html.escape(uid)}"><figcaption>H&E gray reverse-blur candidate crop</figcaption></figure>
    <figure><img src="{html.escape(fic_rel)}" alt="FICTURE crop for {html.escape(uid)}"><figcaption>FICTURE gray reverse-blur candidate crop</figcaption></figure>
  </div>
  <div class="steps">
    <div class="step"><h4>Step1 text-only cell composition hypotheses</h4><p>{class_list_html(step1_classes, true_class)}</p></div>
    <div class="step"><h4>Step2 H&E morphology retained hypotheses</h4><p>{class_list_html(step2_classes, true_class)}</p></div>
    <div class="step final"><h4>Step3 FICTURE check retained hypotheses</h4><p>{class_list_html(step3_classes, true_class)}</p></div>
  </div>
  <details>
    <summary>Show cell-type list and six-class scores</summary>
    <div class="detail-grid">
      <div><h4>Ordered cell-type / marker-gene list used in Step1</h4><pre>{html.escape(row.get('cell_type_list', ''))}</pre></div>
      <div><h4>Six-class status and scores</h4>{class_status_table(row, he_scores.get(uid, {}), fic_scores.get(uid, {}))}</div>
    </div>
    <p><strong>Step1 ambiguity:</strong> {html.escape(row.get('main_ambiguity', ''))}</p>
    <p><strong>Step1 warning:</strong> {html.escape(row.get('composition_warning', ''))}</p>
  </details>
</section>
"""
        )

    write_csv(out_dir / "candidate_three_step_summary.csv", summary_rows)
    (out_dir / "missing_images.txt").write_text("\n".join(missing_images) + ("\n" if missing_images else ""))

    class_nav = " ".join(
        f'<a href="#class-{cls}">{DISPLAY[cls]} ({sum(1 for r in summary_rows if r["true_class"] == cls)})</a>' for cls in CLASS_KEYS
    )
    cards_by_class = []
    for cls in CLASS_KEYS:
        cards_by_class.append(f'<h2 id="class-{cls}">{html.escape(DISPLAY[cls])}</h2>')
        cards_by_class.extend(card for card in cards if f'data-class="{cls}"' in card)

    official_status = "not checked"
    if args.official_summary.exists():
        try:
            official_status = json.loads(args.official_summary.read_text()).get("status", "unknown")
        except Exception:
            official_status = "unreadable"
    pipeline_overall_table, pipeline_by_class_table = pipeline_summary_tables(summary_rows)

    prompt_examples = "\n".join(
        [
            prompt_block(
                "Step1 example input prompt: cell-type / marker-gene text only",
                args.step1_prompt_example,
                "This is one real candidate prompt. Step1 sees no images; it only receives the ordered RGB / compartment / cell-type / marker-gene composition inside the candidate mask.",
            ),
            prompt_block(
                "Step2 prompt template: H&E morphology verification",
                args.step2_prompt_template,
                "Step2 receives the Step1 hypothesis JSON plus H&E candidate/context images, then keeps or rejects tissue hypotheses using morphology.",
            ),
            prompt_block(
                "Step3 prompt template: FICTURE image consistency check",
                args.step3_prompt_template,
                "Step3 receives Step1 and Step2 outputs plus the official FICTURE crop. In the current safe policy, FICTURE is a consistency check, not the main classifier.",
            ),
        ]
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jun10 Three-Step Candidate Review</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2328; background: #f8fafc; }}
header {{ padding: 36px 44px; background: #fff; border-bottom: 1px solid #d8dee4; }}
h1 {{ margin: 0 0 10px; font-size: 32px; }}
h2 {{ margin: 38px 44px 14px; font-size: 24px; }}
h3 {{ margin: 0; font-size: 19px; }}
h4 {{ margin: 0 0 8px; font-size: 14px; }}
p {{ line-height: 1.45; }}
.sub {{ color: #57606a; margin: 5px 0 0; }}
.nav {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }}
.nav a {{ color: #0969da; background: #ddf4ff; text-decoration: none; padding: 7px 10px; border-radius: 6px; }}
.section {{ margin: 22px 44px; background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 22px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ text-align: left; border-bottom: 1px solid #d8dee4; padding: 8px 10px; vertical-align: top; }}
th {{ background: #f6f8fa; }}
.card {{ margin: 18px 44px; padding: 18px; background: #fff; border: 1px solid #d8dee4; border-radius: 8px; }}
.card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
.badges {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; max-width: 520px; }}
.retain {{ padding: 5px 8px; border-radius: 999px; font-size: 12px; }}
.retain.true {{ background: #dafbe1; color: #116329; }}
.retain.false {{ background: #ffebe9; color: #82071e; }}
.images {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 18px; margin: 16px 0; }}
figure {{ margin: 0; border: 1px solid #d8dee4; background: #fff; }}
figure img {{ width: 100%; display: block; object-fit: contain; background: #fff; }}
figcaption {{ padding: 8px 10px; font-size: 13px; color: #57606a; border-top: 1px solid #d8dee4; }}
.steps {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
.step {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; background: #fbfdff; }}
.step.final {{ background: #fff8c5; }}
.chip {{ display: inline-block; padding: 4px 7px; border-radius: 999px; background: #f6f8fa; border: 1px solid #d8dee4; margin: 3px 3px 3px 0; font-size: 12px; }}
.chip.true {{ background: #dafbe1; border-color: #4ac26b; font-weight: 650; }}
.status {{ display: inline-block; padding: 3px 6px; border-radius: 6px; font-size: 12px; border: 1px solid #d8dee4; }}
.status.plausible {{ background: #dafbe1; }}
.status.uncertain {{ background: #fff8c5; }}
.status.unlikely {{ background: #ffebe9; }}
.empty {{ color: #6e7781; font-style: italic; }}
details {{ margin-top: 14px; }}
summary {{ cursor: pointer; font-weight: 650; color: #0969da; }}
.detail-grid {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(360px, 1fr); gap: 18px; margin-top: 14px; }}
pre {{ white-space: pre-wrap; font-size: 12px; background: #f6f8fa; border: 1px solid #d8dee4; padding: 12px; border-radius: 6px; max-height: 340px; overflow: auto; }}
.mini th, .mini td {{ font-size: 12px; padding: 6px 7px; }}
.note {{ color: #57606a; }}
@media (max-width: 1100px) {{ .steps, .images, .detail-grid {{ grid-template-columns: 1fr; }} .card-head {{ flex-direction: column; }} .badges {{ justify-content: flex-start; }} }}
</style>
</head>
<body>
<header>
  <h1>Jun10 Three-Step Cell-Type-First Candidate Review</h1>
  <p class="sub">All 167 funnel candidates are shown with the hidden true class, H&E crop, FICTURE crop, and the remaining possible tissue classes after Step1, Step2, and Step3.</p>
  <div class="nav">{class_nav}</div>
</header>

<section class="section">
  <h2 style="margin:0 0 12px">Experiment Design</h2>
  <p><strong>Goal:</strong> test a three-step tissue-filtering skill for candidate mask pieces. The goal is not to force one final class immediately; the goal is to keep the correct class while gradually removing unsupported alternatives.</p>
  <p><strong>Candidate pool:</strong> 167 compact pieces from the medical_official_points_step24 H&E+FICTURE candidate pool. The hidden true class is annotation-derived and is used only for evaluation.</p>
  <p><strong>Step1:</strong> uses only structured FICTURE-derived cell-type / marker-gene composition text, ordered by proportion inside the candidate mask. It outputs a broad list of possible classes. It does not see H&E or FICTURE images.</p>
  <p><strong>Step2:</strong> uses the H&E gray reverse-blur candidate crop to check tissue morphology. The Step2 list means: among Step1 possibilities, these classes still have H&E visual support.</p>
  <p><strong>Step3:</strong> uses the FICTURE gray reverse-blur crop to check spatial/cellular consistency. The Step3 list means: classes that survived Step2 and also had FICTURE-image support. In implementation terms, this is <strong>Step2 retained classes intersected with FICTURE-supported classes</strong>.</p>
  <p><strong>How to read missing classes:</strong> if a class is not listed under a step, that step filtered it out for this candidate. It does not mean the class is biologically impossible in general; it only means this candidate did not keep that class under the current rule.</p>
  <p><strong>Green label:</strong> the green chip marks the hidden annotation-derived true class, shown only for evaluation so we can see whether each step kept or lost the correct answer.</p>
  <p><strong>Official-data guardrail:</strong> official FICTURE status = <strong>{html.escape(official_status)}</strong>; deprecated/debug FICTURE roots are not used for this report.</p>
</section>

<section class="section">
  <h2 style="margin:0 0 12px">Overall Summary</h2>
  <h3>Three-step retained-class summary</h3>
  {pipeline_overall_table}
  <h3 style="margin-top:22px">Three-step summary by hidden true class</h3>
  {pipeline_by_class_table}
</section>

<section class="section">
  <h2 style="margin:0 0 12px">Example Input Prompt</h2>
  <p>These are the exact prompt examples/templates used by the three-step skill. Step1 is text-only; Step2 adds H&E morphology; Step3 adds the FICTURE image only as a consistency check.</p>
  {prompt_examples}
</section>

<section class="section">
  <h2 style="margin:0 0 12px">How To Read Each Candidate Card</h2>
  <p><strong>Green chip</strong> marks the hidden true class when it is still retained in a step. A red badge means that step removed the hidden true class for this candidate.</p>
  <p><strong>Step1 possible</strong> means classes marked plausible or uncertain from cell composition text. <strong>Step2 possible</strong> means classes still retained after H&E morphology. <strong>Step3 possible</strong> means classes still retained after the FICTURE image check. Labels that are not shown were filtered out at that step.</p>
</section>

{"".join(cards_by_class)}

<section class="section">
  <h2 style="margin:0 0 12px">Validation</h2>
  <p>Candidate rows: {len(step1_rows)}. Missing copied images: {len(missing_images)}.</p>
  <p>Companion CSV: <code>candidate_three_step_summary.csv</code>. Missing-image list: <code>missing_images.txt</code>.</p>
</section>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)
    return out_dir / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill")
    parser.add_argument("--step1-csv", type=Path, default=base / "step1_prompt_ensemble_baseline_plus_targeted_bronvessel/step1_hypotheses.csv")
    parser.add_argument("--step2-policy-csv", type=Path, default=base / "step2_ensemble_score_threshold_sweep/selected_threshold_policy_by_candidate.csv")
    parser.add_argument("--step2-policy", default="best_class_specific_threshold")
    parser.add_argument("--step2-scores-csv", type=Path, default=base / "step2_ensemble_baseline_plus_targeted_bronvessel_output/step2_all_hypothesis_graded_stroma_lenient_qwen3vl32b_14483529/step2_hypothesis_verification_scores.csv")
    parser.add_argument("--step3-rows-csv", type=Path, default=base / "step3_route_collector_smoke/step3_route_comparison_rows.csv")
    parser.add_argument("--step3-scores-csv", type=Path, default=base / "ficture_image_filter_qwen3_32b_full167/ficture_filter_qwen3vl32b_full167_14379887/ficture_hypothesis_verification_scores.csv")
    parser.add_argument("--step3-policy-overall-csv", type=Path, default=base / "step3_two_route_final_answer/step3_final_policy_overall.csv")
    parser.add_argument("--step3-policy-by-class-csv", type=Path, default=base / "step3_two_route_final_answer/step3_final_policy_by_class.csv")
    parser.add_argument("--official-summary", type=Path, default=Path("output/visium_hd_exp1/ficture_official_filtered_he_aligned/summary_official.json"))
    parser.add_argument("--step1-prompt-example", type=Path, default=base / "step1_targeted_wall_rescue_input/prompt_step1_first_row.txt")
    parser.add_argument("--step2-prompt-template", type=Path, default=base / "prompt_step2_he_verification_template.txt")
    parser.add_argument("--step3-prompt-template", type=Path, default=base / "prompt_step3_ficture_image_check_template.txt")
    parser.add_argument("--image-root", action="append", default=[
        "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool/candidate_pair_crops",
        "output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/he_view_ablation_inputs_clean/gray_reverse_blur/corrected_pool/candidate_pair_crops",
    ])
    parser.add_argument("--output-dir", type=Path, default=base / "Jun10_ThreeStep_CellType_HE_FICTURE_CandidateReview")
    args = parser.parse_args()
    print(build(args))


if __name__ == "__main__":
    main()
