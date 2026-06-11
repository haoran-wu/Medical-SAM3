#!/usr/bin/env python3
"""Oracle coverage upper bound for the Jun09 167-piece pool.

This is not a deployable selector.  It uses annotation only to answer a
necessary diagnostic question: if a perfect selector existed, could this
candidate pool assemble a good mask?  If the oracle is weak, the proposal pool
is the bottleneck.  If the oracle is strong but the runtime policy is weak, the
ranker/quality/refinement layer is the bottleneck.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "oracle_coverage_upper_bound"
QA_SCRIPT = ROOT / "scripts/add_jun09_quality_adjusted_ranker.py"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qa = import_module(QA_SCRIPT, "qa_helpers_oracle_coverage")
CLASS_KEYS = qa.CLASS_KEYS
DISPLAY = {"immune_infiltration": "immune infiltration"}

MAX_PIECES = {
    "bronchiola": 8,
    "alveoli": 8,
    "vessels": 12,
    "tumor": 48,
    "stroma": 48,
    "immune_infiltration": 48,
}


def display_class(name: str) -> str:
    return DISPLAY.get(name, name)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(col)}</th>" for col in cols)
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for col in cols:
            out.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def dpr(d: float, p: float, r: float) -> str:
    return f"{d:.3f}/{p:.3f}/{r:.3f}"


def parse_dpr(value: str) -> tuple[float, float, float]:
    a, b, c = str(value).split("/")
    return float(a), float(b), float(c)


def class_key(label: str) -> str:
    return str(label).replace("immune infiltration", "immune_infiltration")


def greedy_oracle(
    candidates: pd.DataFrame,
    annotations: dict[str, np.ndarray],
    tissue_class: str,
    get_small_mask,
    canvas_shape: tuple[int, int],
    max_pieces: int,
) -> tuple[list[pd.Series], np.ndarray, list[dict[str, object]]]:
    gt = annotations[tissue_class]
    union = np.zeros(canvas_shape, dtype=bool)
    selected: list[pd.Series] = []
    unused = list(candidates.index)
    trace: list[dict[str, object]] = []
    current_d, current_p, current_r = qa.metrics(union, gt)
    for step in range(max_pieces):
        best_idx = None
        best_union = None
        best_metrics = (current_d, current_p, current_r)
        for idx in unused:
            row = candidates.loc[idx]
            mask = get_small_mask(str(row["row_key"]))
            proposal = np.logical_or(union, mask)
            d, p, r = qa.metrics(proposal, gt)
            if (d, r, p) > (best_metrics[0], best_metrics[2], best_metrics[1]):
                best_idx = idx
                best_union = proposal
                best_metrics = (d, p, r)
        if best_idx is None:
            break
        if best_metrics[0] <= current_d + 0.002 and best_metrics[2] <= current_r + 0.005:
            break
        row = candidates.loc[best_idx]
        selected.append(row)
        union = best_union
        unused.remove(best_idx)
        current_d, current_p, current_r = best_metrics
        trace.append(
            {
                "class": display_class(tissue_class),
                "step": step + 1,
                "candidate_uid": row["candidate_uid"],
                "true_class_hidden": row["true_class"],
                "smallres_union_D/P/R": dpr(current_d, current_p, current_r),
                "candidate_component_dice_hidden": f"{float(row.get('component_dice', 0)):.3f}",
                "candidate_component_precision_hidden": f"{float(row.get('component_precision', 0)):.3f}",
                "candidate_component_recall_hidden": f"{float(row.get('component_recall', 0)):.3f}",
            }
        )
    return selected, union, trace


def rebuild_zip() -> None:
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    include: list[Path] = []
    for rel in [
        "Jun09_SkillRanker_ComponentAwareMaskSelection.html",
        "index.html",
        "candidate_skill_features.csv",
        "piece_top1_all_methods.csv",
        "assembly_summary_all_methods.csv",
        "selected_pieces_all_methods.csv",
        "failure_attribution_by_class.csv",
        "run_config.json",
    ]:
        p = BASE / rel
        if p.exists():
            include.append(p)
    for root in [
        BASE / "corrected_pool",
        BASE / "assembly_outputs/combined_skill_ranker_rf_plus_he/figures",
        BASE / "refined_diagnostics",
        BASE / "guarded_loco_variant",
        BASE / "second_sam_refinement",
        BASE / "structured_veto_microscope",
        BASE / "quality_skill_diagnostic",
        BASE / "quality_adjusted_ranker",
        BASE / "proxy_quality_ranker",
        BASE / "failure_driven_proxy_refinement",
        BASE / "alveoli_generator_gate",
        BASE / "paper_informed_failure_engine",
        BASE / "remote_gate_preflight",
        BASE / "failure_gate_matrix",
        BASE / "skill_iteration_controller",
        BASE / "feature_space_upper_bound",
        BASE / "runtime_policy_prototype",
        BASE / "runtime_policy_second_sam_locator_pack",
        BASE / "alveoli_broad_box_branch",
        BASE / "failure_debugger_v2",
        BASE / "runtime_candidate_error_auditor",
        BASE / "one_class_assignment_test",
        BASE / "soft_assignment_veto_test",
        BASE / "class_specific_assembly_grid",
        BASE / "iteration_state_after_grid",
        BASE / "structured_quality_v2",
        BASE / "failure_taxonomy_v3_after_quality_v2",
        OUT,
    ]:
        if root.exists():
            include.extend(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        seen: set[Path] = set()
        for path in include:
            if path in seen:
                continue
            seen.add(path)
            handle.write(path, path.relative_to(BASE))


def diagnosis(tissue_class: str, runtime_d: float, all_d: float, hidden_d: float) -> str:
    if all_d < 0.55:
        return "pool/proposal bottleneck: even a perfect selector cannot assemble a strong mask from this 167-piece pool"
    if runtime_d < all_d - 0.18:
        return "selector/quality bottleneck: the pool contains enough pieces, but deployable ranking misses the right combination"
    if hidden_d < all_d - 0.12:
        return "cross-class proposal useful: correct pieces are present but hidden matched class labels alone are too restrictive"
    if runtime_d >= all_d - 0.08:
        return "selector close to oracle; next layer is boundary/refinement or external validation"
    return "mixed bottleneck: both selector quality and proposal/refinement need work"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    size = Image.open(qa.HE_ROI).size
    small_size = (768, int(round(768 * size[1] / size[0])))
    annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], size) for c in CLASS_KEYS}
    small_annotations = {c: qa.load_mask(qa.ANNOTATION_DIR / qa.ANNOTATION_FILES[c], small_size) for c in CLASS_KEYS}
    public = pd.read_csv(BASE / "corrected_pool/public_vlm_requests.csv")
    public["row_key"] = public.apply(qa.row_key, axis=1)
    public_by_key = public.set_index("row_key").to_dict("index")
    features = pd.read_csv(BASE / "candidate_skill_features.csv")
    runtime = pd.read_csv(BASE / "runtime_policy_prototype/runtime_policy_summary.csv")
    runtime_by = {class_key(r["class"]): r for _, r in runtime.iterrows()}

    small_mask_cache: dict[str, np.ndarray] = {}
    full_mask_cache: dict[str, np.ndarray] = {}

    def get_small_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        path_s = str(path)
        if path_s not in small_mask_cache:
            small_mask_cache[path_s] = qa.load_mask(path, small_size)
        return small_mask_cache[path_s]

    def get_full_mask(key: str) -> np.ndarray:
        path = Path(public_by_key[key]["mask_path"])
        path_s = str(path)
        if path_s not in full_mask_cache:
            full_mask_cache[path_s] = qa.load_mask(path, size)
        return full_mask_cache[path_s]

    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for tissue_class in CLASS_KEYS:
        all_candidates = features.copy()
        hidden_true_candidates = features[features["true_class"] == tissue_class].copy()
        selected_hidden, _, hidden_trace = greedy_oracle(
            hidden_true_candidates,
            small_annotations,
            tissue_class,
            get_small_mask,
            (small_size[1], small_size[0]),
            MAX_PIECES[tissue_class],
        )
        selected_all, _, all_trace = greedy_oracle(
            all_candidates,
            small_annotations,
            tissue_class,
            get_small_mask,
            (small_size[1], small_size[0]),
            MAX_PIECES[tissue_class],
        )

        hidden_union = np.zeros((size[1], size[0]), dtype=bool)
        for row in selected_hidden:
            hidden_union |= get_full_mask(str(row["row_key"]))
        all_union = np.zeros((size[1], size[0]), dtype=bool)
        for row in selected_all:
            all_union |= get_full_mask(str(row["row_key"]))

        hd, hp, hr = qa.metrics(hidden_union, annotations[tissue_class])
        ad, ap, ar = qa.metrics(all_union, annotations[tissue_class])
        rt = runtime_by[tissue_class]
        rd, rp, rr = float(rt["dice"]), float(rt["precision"]), float(rt["recall"])
        hidden_fig = OUT / "figures" / f"{tissue_class}_hidden_true_class_oracle_union.png"
        all_fig = OUT / "figures" / f"{tissue_class}_all_candidate_oracle_union.png"
        qa.render_six_panel(
            hidden_fig,
            tissue_class,
            hidden_union,
            annotations[tissue_class],
            f"{display_class(tissue_class)}: hidden true-class oracle",
            "Diagnostic only: selects pieces using hidden class and annotation-driven greedy Dice; not deployable.",
        )
        qa.render_six_panel(
            all_fig,
            tissue_class,
            all_union,
            annotations[tissue_class],
            f"{display_class(tissue_class)}: all-candidate oracle",
            "Diagnostic only: selects any piece that improves target annotation Dice; not deployable.",
        )
        summary_rows.append(
            {
                "class": display_class(tissue_class),
                "runtime D/P/R": dpr(rd, rp, rr),
                "hidden true-class oracle D/P/R": dpr(hd, hp, hr),
                "all-candidate oracle D/P/R": dpr(ad, ap, ar),
                "hidden selected pieces": len(selected_hidden),
                "all-candidate selected pieces": len(selected_all),
                "oracle-runtime Dice gap": f"{ad - rd:+.3f}",
                "diagnosis": diagnosis(tissue_class, rd, ad, hd),
                "hidden_oracle_figure_rel": str(hidden_fig.relative_to(BASE)),
                "all_oracle_figure_rel": str(all_fig.relative_to(BASE)),
            }
        )
        for row in hidden_trace:
            row["oracle type"] = "hidden true-class only"
            trace_rows.append(row)
        for row in all_trace:
            row["oracle type"] = "all candidates"
            trace_rows.append(row)

    write_csv(OUT / "oracle_coverage_summary.csv", summary_rows)
    write_csv(OUT / "oracle_coverage_selection_trace.csv", trace_rows)

    section = f"""
<h2>17AD. Oracle Coverage Upper Bound: Is The Candidate Pool Itself Enough?</h2>
<p><b>Goal.</b> After 17AB showed that another hand-crafted quality proxy does not solve the problem, this section asks a sharper question: if an impossible perfect selector could use the annotation during selection, could the current 167-piece pool assemble a good mask? This separates proposal failure from ranker/quality failure.</p>
<p><b>Two diagnostic oracles.</b> The <b>hidden true-class oracle</b> may only select pieces whose hidden matched class is the target class. The <b>all-candidate oracle</b> may select any piece if it improves the target annotation Dice. Neither oracle is deployable; they are upper-bound tests.</p>
<h3>Oracle coverage summary</h3>
{table(summary_rows, ['class', 'runtime D/P/R', 'hidden true-class oracle D/P/R', 'all-candidate oracle D/P/R', 'hidden selected pieces', 'all-candidate selected pieces', 'oracle-runtime Dice gap', 'diagnosis'])}
<h3>Greedy oracle selected pieces</h3>
{table(trace_rows[:220], ['class', 'oracle type', 'step', 'candidate_uid', 'true_class_hidden', 'smallres_union_D/P/R', 'candidate_component_dice_hidden', 'candidate_component_precision_hidden', 'candidate_component_recall_hidden'])}
<div class='grid'>
{''.join(f"<figure><img src='{html.escape(row['hidden_oracle_figure_rel'])}'><figcaption>{html.escape(row['class'])}: hidden true-class oracle</figcaption></figure><figure><img src='{html.escape(row['all_oracle_figure_rel'])}'><figcaption>{html.escape(row['class'])}: all-candidate oracle</figcaption></figure>" for row in summary_rows)}
</div>
<div class='callout'><b>17AD decision rule.</b> If the all-candidate oracle is weak, fix the proposal generator first. If the oracle is strong but runtime is weak, fix the ranker/quality/refinement layer. This prevents spending more time on a model or prompt when the mask simply is not present in the pool.</div>
"""
    marker = "<h2>17AD. Oracle Coverage Upper Bound: Is The Candidate Pool Itself Enough?</h2>"
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        if marker in text:
            start = text.index(marker)
            end = text.index("</body>", start)
            text = text[:start] + section + text[end:]
        else:
            text = text.replace("</body>", section + "</body>")
        html_path.write_text(text)
    rebuild_zip()
    print(OUT / "oracle_coverage_summary.csv")
    print(OUT / "oracle_coverage_selection_trace.csv")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip")


if __name__ == "__main__":
    main()
