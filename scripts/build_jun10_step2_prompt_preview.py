#!/usr/bin/env python3
"""Build a local audit HTML for Jun10 Step2 prompt ablations.

This does not run a VLM. It expands the exact Step2 prompts that would be sent
to the model, next to each weak candidate's H&E crop and Step1 hypotheses.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import os
from pathlib import Path
from typing import Iterable


CLASS_KEYS = ["bronchiola", "alveoli", "vessels", "tumor", "stroma", "immune_infiltration"]
DEFAULT_STYLES = [
    "all_hypothesis_broad_class_veto",
    "all_hypothesis_direct_evidence_minrules",
    "comparative_hypothesis_veto",
    "comparative_soft_keep",
    "comparative_ranked_retention",
    "comparative_pattern_first",
    "comparative_primary_pattern_budget",
    "comparative_structural_priority",
    "comparative_false_positive_guard",
    "comparative_soft_broad_veto_retention",
    "comparative_fewshot_broad_veto",
    "comparative_structured_checklist",
    "comparative_minimal_retention",
    "comparative_structural_keep_broad_prune",
    "comparative_brief_class_definitions",
    "comparative_alveoli_septa_rescue",
    "comparative_stroma_immune_strict_veto",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_prompt_maps(runner_path: Path) -> tuple[dict[str, str], set[str]]:
    """Return prompt style -> template string and comparative style names."""
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    strings: dict[str, str] = {}
    maps: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            strings[name] = node.value.value
        elif isinstance(node.value, ast.Dict):
            out: dict[str, str] = {}
            for key, val in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(val, ast.Name):
                    out[key.value] = val.id
            if out:
                maps[name] = out

    style_to_template: dict[str, str] = {}
    for map_name in ("ALL_HYPOTHESIS_PROMPTS", "COMPARATIVE_PROMPTS"):
        for style, var_name in maps.get(map_name, {}).items():
            if var_name in strings:
                style_to_template[style] = strings[var_name]
    comparative = set(maps.get("COMPARATIVE_PROMPTS", {}))
    return style_to_template, comparative


def step1_map(row: dict[str, str]) -> dict[str, str]:
    return {cls: str(row.get(cls, "")).strip() or "missing" for cls in CLASS_KEYS}


def retained_hypotheses(row: dict[str, str]) -> list[str]:
    m = step1_map(row)
    return [cls for cls in CLASS_KEYS if m.get(cls) in {"plausible", "uncertain"}]


def rel_link(from_html: Path, target: Path) -> str:
    return os.path.relpath(target, from_html.parent)


def render_prompt(style: str, template: str, comparative: bool, row: dict[str, str], target: str | None = None) -> str:
    m = step1_map(row)
    step1_json = json.dumps(m, ensure_ascii=False, indent=2)
    if comparative:
        return template.format(
            retained_hypotheses=", ".join(retained_hypotheses(row)),
            step1_hypothesis_json=step1_json,
        )
    if target is None:
        raise ValueError(f"target required for non-comparative style {style}")
    return template.format(
        target_hypothesis=target,
        target_step1_status=m[target],
        step1_hypothesis_json=step1_json,
    )


def leakage_flags(prompt: str) -> list[str]:
    forbidden = [
        "true_class",
        "component_dice",
        "component_precision",
        "component_recall",
        "Dice",
        "Precision",
        "Recall",
        "candidate_uid",
    ]
    return [item for item in forbidden if item in prompt]


def html_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in columns)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in columns:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-csv", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_next_prompt_ablation_weak58_input/weak58_requests.csv"))
    parser.add_argument("--step1-csv", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_next_prompt_ablation_weak58_input/step1_ensemble_weak58_hypotheses.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/corrected_pool"))
    parser.add_argument("--context-root", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection/region_aware_piece_context_locator_pack"))
    parser.add_argument("--runner", type=Path, default=Path("inference/visium_hd_exp1/run_qwen3_he_morphology_verification.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/visium_hd_exp1/final_deliverables/Jun10_CellTypeFirst_Qwen3_Skill/step2_prompt_preview_audit"))
    parser.add_argument("--styles", nargs="*", default=DEFAULT_STYLES)
    args = parser.parse_args()

    requests = read_csv(args.requests_csv)
    step1_rows = {(row["row_index"], row["candidate_uid"]): row for row in read_csv(args.step1_csv)}
    prompts, comparative_styles = load_prompt_maps(args.runner)
    html_path = args.output_dir / "index.html"
    wants_context = any("context" in style for style in args.styles)

    summary = []
    missing_images = []
    leakage_rows = []
    candidate_blocks = []
    for row in requests:
        key = (row["row_index"], row["candidate_uid"])
        if key not in step1_rows:
            raise KeyError(f"Missing Step1 row for {key}")
        s1 = step1_rows[key]
        he_path = args.image_root / row["he_crop_rel"]
        if not he_path.exists():
            missing_images.append(str(he_path))
        context_path = args.context_root / "views" / f"{row['candidate_uid']}_he_local_context.jpg"
        if wants_context and not context_path.exists():
            missing_images.append(str(context_path))
        prompt_sections = []
        for style in args.styles:
            if style not in prompts:
                raise KeyError(f"Prompt style {style} not found in {args.runner}")
            comparative = style in comparative_styles
            if comparative:
                prompt = render_prompt(style, prompts[style], True, s1)
                flags = leakage_flags(prompt)
                if flags:
                    leakage_rows.append({"candidate_uid": row["candidate_uid"], "style": style, "flags": ",".join(flags)})
                prompt_sections.append(
                    f"<details><summary>{html.escape(style)} · comparative</summary><pre>{html.escape(prompt)}</pre></details>"
                )
            else:
                target_sections = []
                for target in retained_hypotheses(s1):
                    prompt = render_prompt(style, prompts[style], False, s1, target=target)
                    flags = leakage_flags(prompt)
                    if flags:
                        leakage_rows.append({"candidate_uid": row["candidate_uid"], "style": f"{style}:{target}", "flags": ",".join(flags)})
                    target_sections.append(f"<h5>{html.escape(target)}</h5><pre>{html.escape(prompt)}</pre>")
                prompt_sections.append(
                    f"<details><summary>{html.escape(style)} · per-hypothesis ({len(target_sections)})</summary>{''.join(target_sections)}</details>"
                )

        s1_hyp = step1_map(s1)
        summary.append(
            {
                "true_class_for_audit_only": row.get("true_class", ""),
                "n": 1,
                "step1_retained_count": len(retained_hypotheses(s1)),
            }
        )
        context_figure = ""
        if wants_context:
            context_figure = f"""
                <figure>
                  <img src="{html.escape(rel_link(html_path, context_path))}" alt="H&E local context for {html.escape(row['candidate_uid'])}">
                  <figcaption>H&E local context crop used only by context-aware Step2 prompts</figcaption>
                </figure>
            """
        candidate_blocks.append(
            f"""
            <section class="candidate">
              <div class="candidate-head">
                <div>
                  <h3>{html.escape(row['candidate_uid'])}</h3>
                  <p><b>Audit-only true class:</b> {html.escape(row.get('true_class',''))}
                  · <b>component Dice/P/R hidden from model:</b> {html.escape(row.get('component_dice',''))} / {html.escape(row.get('component_precision',''))} / {html.escape(row.get('component_recall',''))}</p>
                  <p><b>Step1 kept:</b> {html.escape(', '.join(retained_hypotheses(s1)))}</p>
                  <pre class="small">{html.escape(json.dumps(s1_hyp, ensure_ascii=False, indent=2))}</pre>
                </div>
                <figure>
                  <img src="{html.escape(rel_link(html_path, he_path))}" alt="H&E crop for {html.escape(row['candidate_uid'])}">
                  <figcaption>H&E gray reverse-blur crop used by Step2</figcaption>
                </figure>
                {context_figure}
              </div>
              <div class="prompt-list">
                {''.join(prompt_sections)}
              </div>
            </section>
            """
        )

    counts: dict[str, dict[str, int]] = {}
    for row in requests:
        cls = row.get("true_class", "")
        counts.setdefault(cls, {"n": 0})
        counts[cls]["n"] += 1
    count_rows = [{"class": cls, "n": item["n"]} for cls, item in sorted(counts.items())]

    style_rows = [
        {
            "prompt_style": style,
            "mode": "comparative all retained hypotheses at once" if style in comparative_styles else "one prompt per retained hypothesis",
        }
        for style in args.styles
    ]

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }
    h1 { font-size: 30px; margin-bottom: 8px; }
    h2 { margin-top: 32px; border-top: 1px solid #dde3ea; padding-top: 24px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
    th, td { border-bottom: 1px solid #e6ebf0; text-align: left; padding: 8px 10px; vertical-align: top; }
    th { background: #f7f9fb; }
    .candidate { border: 1px solid #d9e1ea; border-radius: 8px; padding: 18px; margin: 22px 0; }
    .candidate-head { display: grid; grid-template-columns: minmax(360px, 1fr) repeat(2, minmax(280px, 360px)); gap: 24px; align-items: start; }
    img { max-width: 360px; border: 1px solid #d9e1ea; background: #fff; }
    figcaption { color: #596673; font-size: 13px; margin-top: 6px; }
    pre { background: #f5f7f9; padding: 12px; border-radius: 6px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.45; }
    pre.small { max-height: 180px; overflow: auto; }
    details { margin: 10px 0; }
    summary { cursor: pointer; font-weight: 700; }
    .ok { color: #137333; font-weight: 700; }
    .warn { color: #a15c00; font-weight: 700; }
    """
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Jun10 Step2 Prompt Preview Audit</title><style>{css}</style></head>
<body>
<h1>Jun10 Step2 Prompt Preview Audit</h1>
<p>This page is a pre-run audit for the weak-candidate Step2 prompt search. It does not contain model outputs. It shows exactly what the H&E verifier will see before submitting the jobs.</p>

<h2>Experiment Design</h2>
<p><b>Goal:</b> test whether alternative H&E-only verification prompts can narrow broad Step1 hypotheses without dropping the true tissue class.</p>
<p><b>Step1 input:</b> FICTURE-derived cell-type / marker-gene composition text only. Step1 produces possible tissue-class hypotheses.</p>
<p><b>Step2 input:</b> H&E gray reverse-blur candidate crop. Context-aware prompt styles also receive a local H&E context crop, but the decision is still only about the highlighted candidate mask. FICTURE image, Dice, Precision, Recall, and annotation are hidden from the model.</p>
<p><b>Weak set:</b> all alveoli, bronchiola, and vessels candidates plus a small sample of tumor, stroma, and immune infiltration candidates. This focuses the prompt search on classes that previous runs either missed or over-kept.</p>

<h2>Prompt Styles To Test</h2>
{html_table(style_rows, ["prompt_style", "mode"])}

<h2>Weak Set Composition</h2>
{html_table(count_rows, ["class", "n"])}

<h2>Gate Checks</h2>
<p class="{ 'ok' if not missing_images else 'warn' }"><b>Missing H&E/context images:</b> {len(missing_images)}</p>
<p class="{ 'ok' if not leakage_rows else 'warn' }"><b>Prompt leakage flags:</b> {len(leakage_rows)}</p>
{html_table(leakage_rows[:50], ["candidate_uid", "style", "flags"]) if leakage_rows else ""}

<h2>Candidate-by-Candidate Prompt Audit</h2>
{''.join(candidate_blocks)}
</body></html>"""
    write_text(html_path, body)
    write_text(args.output_dir / "missing_images.txt", "\n".join(missing_images))
    write_text(args.output_dir / "prompt_leakage_flags.csv", "\n".join([",".join(row.values()) for row in leakage_rows]))
    print(html_path)
    print(f"candidates={len(requests)} styles={len(args.styles)} missing_images={len(missing_images)} leakage_flags={len(leakage_rows)}")


if __name__ == "__main__":
    main()
