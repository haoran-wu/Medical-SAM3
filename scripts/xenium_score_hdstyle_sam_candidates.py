#!/usr/bin/env python3
"""Score Xenium HD-style SAM candidates against per-class annotation masks.

This is the generic Xenium companion to the older VisiumHD component-aware
scoring scripts. It reads one TMA, one SAM setting, and one or more source
families such as H&E plus a FICTURE-density map, clusters near-duplicate masks,
then scores every cluster-union candidate against every annotation component.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


BITCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


@dataclass(frozen=True)
class RawCandidate:
    idx: int
    source_key: str
    candidate_id: int
    path: Path
    area: int
    bbox: tuple[int, int, int, int]
    packed: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class ClusterCandidate:
    uid: str
    source_keys: str
    member_ids: str
    cluster_size: int
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    mask_path: Path


@dataclass(frozen=True)
class Policy:
    min_component_dice: float
    min_component_precision: float
    min_incremental_precision: float
    max_overlap_with_selected: float
    max_precision_drop: float
    max_selected: int
    min_component_area: int


POLICY_PRESETS = {
    "bronchiola": Policy(0.45, 0.50, 0.45, 0.75, 0.06, 8, 200),
    "airway": Policy(0.45, 0.50, 0.45, 0.75, 0.06, 8, 200),
    "alveoli": Policy(0.15, 0.08, 0.05, 0.85, 0.12, 5, 500),
    "vessels": Policy(0.20, 0.08, 0.05, 0.85, 0.12, 10, 200),
    "tumor": Policy(0.25, 0.25, 0.20, 0.90, 0.05, 5, 500),
    "stroma": Policy(0.25, 0.25, 0.20, 0.90, 0.05, 5, 500),
    "immune_infiltration": Policy(0.15, 0.08, 0.05, 0.90, 0.10, 12, 100),
    "generic": Policy(0.15, 0.08, 0.05, 0.90, 0.10, 8, 100),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--tma", type=int, required=True)
    parser.add_argument("--setting-name", required=True)
    parser.add_argument("--source-keys", nargs="+", default=["he", "ficture_sparse"])
    parser.add_argument("--annotation-policy", choices=["dense", "points"], default="dense")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cluster-threshold", type=float, default=0.90)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--largest-components-for-policy-union", type=int, default=40)
    parser.add_argument(
        "--union-min-component-dice",
        type=float,
        default=0.0,
        help=(
            "For piece-wise GT assembly, include a component top-1 candidate only "
            "when its Dice is strictly greater than this value. Default 0 keeps "
            "only components with positive candidate overlap."
        ),
    )
    return parser.parse_args()


def slugify(text: str, max_len: int = 72) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "label"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_mask(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if expected_size is not None and image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return (0, 0, -1, -1)
    return (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()))


def bbox_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1:
        return 0
    return int((x2 - x1 + 1) * (y2 - y1 + 1))


def packed_iou(a: RawCandidate, b: RawCandidate) -> float:
    inter = int(BITCOUNT[np.bitwise_and(a.packed, b.packed)].sum())
    if inter == 0:
        return 0.0
    union = int(a.area + b.area - inter)
    return inter / union if union else 0.0


def load_candidates(candidate_root: Path, tma: int, source_keys: list[str], setting_name: str, min_area: int) -> list[RawCandidate]:
    out: list[RawCandidate] = []
    for source_key in source_keys:
        mask_dir = candidate_root / f"TMA{tma:02d}" / source_key / setting_name / "candidate_masks"
        if not mask_dir.exists():
            print(f"Warning: missing candidate mask dir: {mask_dir}")
            continue
        for path in sorted(mask_dir.glob("candidate_*.png")):
            mask = read_mask(path)
            area = int(mask.sum())
            if area < min_area:
                continue
            match = re.search(r"candidate_(\d+)", path.stem)
            candidate_id = int(match.group(1)) if match else len(out)
            out.append(
                RawCandidate(
                    idx=len(out),
                    source_key=source_key,
                    candidate_id=candidate_id,
                    path=path,
                    area=area,
                    bbox=bbox(mask),
                    packed=np.packbits(mask.reshape(-1)),
                    mask=mask,
                )
            )
    return out


def complete_link_clusters(raw: list[RawCandidate], threshold: float) -> tuple[list[list[int]], list[dict]]:
    scores: dict[tuple[int, int], float] = {}
    edges: list[dict] = []
    for i in range(len(raw)):
        ai = raw[i]
        for j in range(i + 1, len(raw)):
            aj = raw[j]
            area_upper = min(ai.area, aj.area) / max(ai.area, aj.area)
            if area_upper < threshold:
                continue
            bbox_upper = bbox_intersection_area(ai.bbox, aj.bbox) / max(ai.area, aj.area)
            if bbox_upper < threshold:
                continue
            score = packed_iou(ai, aj)
            if score >= threshold:
                scores[(i, j)] = score
                scores[(j, i)] = score
                edges.append(
                    {
                        "source_a": ai.source_key,
                        "candidate_a": ai.candidate_id,
                        "source_b": aj.source_key,
                        "candidate_b": aj.candidate_id,
                        "iou": f"{score:.6f}",
                    }
                )
    clusters: list[list[int]] = []
    order = sorted(range(len(raw)), key=lambda idx: (raw[idx].area, -raw[idx].candidate_id), reverse=True)
    for idx in order:
        for cluster in clusters:
            if all(scores.get((idx, member), 0.0) >= threshold for member in cluster):
                cluster.append(idx)
                break
        else:
            clusters.append([idx])
    return [sorted(cluster, key=lambda idx: (raw[idx].source_key, raw[idx].candidate_id)) for cluster in clusters], edges


def build_cluster_candidates(raw: list[RawCandidate], out_dir: Path, threshold: float) -> tuple[list[ClusterCandidate], list[dict], list[dict]]:
    clusters, edges = complete_link_clusters(raw, threshold)
    mask_dir = out_dir / "cluster_masks"
    candidates: list[ClusterCandidate] = []
    inventory: list[dict] = []
    for cluster_id, members in enumerate(clusters):
        member_items = [raw[idx] for idx in members]
        union = np.logical_or.reduce([item.mask for item in member_items])
        uid = f"iou{threshold:.2f}_cluster_{cluster_id:05d}"
        mask_path = mask_dir / f"{uid}.png"
        save_mask(union, mask_path)
        source_keys = sorted(set(item.source_key for item in member_items))
        member_ids = ";".join(f"{item.source_key}:{item.candidate_id:03d}" for item in member_items)
        candidate = ClusterCandidate(
            uid=uid,
            source_keys=";".join(source_keys),
            member_ids=member_ids,
            cluster_size=len(member_items),
            mask=union,
            area=int(union.sum()),
            bbox=bbox(union),
            mask_path=mask_path,
        )
        candidates.append(candidate)
        inventory.append(
            {
                "cluster_uid": uid,
                "cluster_id": cluster_id,
                "cluster_size": candidate.cluster_size,
                "source_keys": candidate.source_keys,
                "members": candidate.member_ids,
                "area": candidate.area,
                "bbox_xyxy": ",".join(map(str, candidate.bbox)),
                "mask_path": str(mask_path),
            }
        )
    return candidates, inventory, edges


def load_annotation_masks(prepared_root: Path, tma: int, policy: str, expected_size: tuple[int, int]) -> list[dict]:
    label_rows = [row for row in read_csv(prepared_root / "label_manifest.csv") if int(row["tma"]) == int(tma)]
    mask_key = "dense_mask_rel" if policy == "dense" else "point_mask_rel"
    masks = []
    for row in label_rows:
        mask_path = prepared_root / row[mask_key]
        full = read_mask(mask_path, expected_size=expected_size)
        labeled, n_components = ndimage.label(full, structure=np.ones((3, 3), dtype=np.uint8))
        component_ids = list(range(1, n_components + 1))
        component_areas = ndimage.sum(full, labeled, index=component_ids) if component_ids else []
        masks.append(
            {
                "label": row["label"],
                "label_slug": row["label_slug"],
                "mask_path": mask_path,
                "full_mask": full,
                "label_map": labeled,
                "component_ids": component_ids,
                "component_areas": [int(v) for v in component_areas],
            }
        )
    return masks


def metrics(overlap: int, pred_area: int, gt_area: int) -> tuple[float, float, float]:
    dice = (2 * overlap / (pred_area + gt_area)) if pred_area + gt_area else 0.0
    precision = overlap / pred_area if pred_area else 0.0
    recall = overlap / gt_area if gt_area else 0.0
    return dice, precision, recall


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    overlap = int((pred & gt).sum())
    dice, precision, recall = metrics(overlap, int(pred.sum()), int(gt.sum()))
    union_area = int((pred | gt).sum())
    return {
        "dice": dice,
        "iou": overlap / union_area if union_area else 0.0,
        "precision": precision,
        "recall": recall,
        "tp": overlap,
        "pred_area": int(pred.sum()),
        "gt_area": int(gt.sum()),
    }


def overlap_with_selected(candidate: np.ndarray, selected: np.ndarray) -> float:
    area = int(candidate.sum())
    if area == 0:
        return 0.0
    return float((candidate & selected).sum() / area)


def incremental_precision(candidate: np.ndarray, selected: np.ndarray, gt: np.ndarray) -> float:
    new_pixels = candidate & ~selected
    new_area = int(new_pixels.sum())
    if new_area == 0:
        return 0.0
    return float((new_pixels & gt).sum() / new_area)


def policy_for_label(label_slug: str, label: str) -> tuple[str, Policy]:
    text = f"{label_slug} {label}".lower()
    if "alveoli" in text:
        return "alveoli", POLICY_PRESETS["alveoli"]
    if any(term in text for term in ["arteriole", "venule", "vessel", "vascular"]):
        return "vessels", POLICY_PRESETS["vessels"]
    if any(term in text for term in ["airway", "bronchi", "mucous_plug", "mucous plug"]):
        return "airway", POLICY_PRESETS["airway"]
    if "tumor" in text:
        return "tumor", POLICY_PRESETS["tumor"]
    if any(term in text for term in ["lymphoid", "macrophage", "immune", "cellularized_granuloma"]):
        return "immune_infiltration", POLICY_PRESETS["immune_infiltration"]
    if any(term in text for term in ["fibroblast", "stroma", "muscle", "border", "hyalinized"]):
        return "stroma", POLICY_PRESETS["stroma"]
    return "generic", POLICY_PRESETS["generic"]


def push_top_piece_row(store: dict[tuple[str, int], list[dict]], key: tuple[str, int], row: dict, top_n: int) -> None:
    rows = store.setdefault(key, [])
    rows.append(row)
    rows.sort(
        key=lambda item: (
            float(item["component_dice"]),
            float(item["component_precision"]),
            float(item["component_recall"]),
            float(item["full_dice"]),
        ),
        reverse=True,
    )
    del rows[top_n:]


def score_candidates(
    candidates: list[ClusterCandidate],
    annotations: list[dict],
    top_n: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    top_piece_store: dict[tuple[str, int], list[dict]] = {}
    component_inventory: list[dict] = []
    for ann in annotations:
        for component_id, component_area in zip(ann["component_ids"], ann["component_areas"]):
            comp_mask = ann["label_map"] == component_id
            component_inventory.append(
                {
                    "label": ann["label"],
                    "label_slug": ann["label_slug"],
                    "component_id": component_id,
                    "component_area": int(component_area),
                    "component_bbox_xyxy": ",".join(map(str, bbox(comp_mask))),
                }
            )
    for cand in candidates:
        pred = cand.mask
        pred_area = cand.area
        for ann in annotations:
            full = ann["full_mask"]
            full_overlap = int((pred & full).sum())
            full_dice, full_precision, full_recall = metrics(full_overlap, pred_area, int(full.sum()))
            best = {
                "component_id": "",
                "component_area": 0,
                "component_overlap": 0,
                "component_dice": 0.0,
                "component_precision": 0.0,
                "component_recall": 0.0,
            }
            for component_id, component_area in zip(ann["component_ids"], ann["component_areas"]):
                comp = ann["label_map"] == component_id
                overlap = int((pred & comp).sum())
                dice, precision, recall = metrics(overlap, pred_area, int(component_area))
                piece_row = {
                    "label": ann["label"],
                    "label_slug": ann["label_slug"],
                    "component_id": component_id,
                    "component_area": int(component_area),
                    "component_overlap": overlap,
                    "component_dice": f"{dice:.6f}",
                    "component_precision": f"{precision:.6f}",
                    "component_recall": f"{recall:.6f}",
                    "candidate_uid": cand.uid,
                    "source_keys": cand.source_keys,
                    "cluster_size": cand.cluster_size,
                    "members": cand.member_ids,
                    "candidate_area": cand.area,
                    "candidate_bbox_xyxy": ",".join(map(str, cand.bbox)),
                    "candidate_mask_path": str(cand.mask_path),
                    "full_dice": f"{full_dice:.6f}",
                    "full_precision": f"{full_precision:.6f}",
                    "full_recall": f"{full_recall:.6f}",
                }
                push_top_piece_row(top_piece_store, (ann["label_slug"], int(component_id)), piece_row, top_n)
                if dice > best["component_dice"]:
                    best = {
                        "component_id": component_id,
                        "component_area": int(component_area),
                        "component_overlap": overlap,
                        "component_dice": dice,
                        "component_precision": precision,
                        "component_recall": recall,
                    }
            rows.append(
                {
                    "label": ann["label"],
                    "label_slug": ann["label_slug"],
                    "candidate_uid": cand.uid,
                    "source_keys": cand.source_keys,
                    "cluster_size": cand.cluster_size,
                    "members": cand.member_ids,
                    "candidate_area": cand.area,
                    "candidate_bbox_xyxy": ",".join(map(str, cand.bbox)),
                    "candidate_mask_path": str(cand.mask_path),
                    "full_dice": f"{full_dice:.6f}",
                    "full_precision": f"{full_precision:.6f}",
                    "full_recall": f"{full_recall:.6f}",
                    "component_id": best["component_id"],
                    "component_area": best["component_area"],
                    "component_overlap": best["component_overlap"],
                    "component_dice": f"{best['component_dice']:.6f}",
                    "component_precision": f"{best['component_precision']:.6f}",
                    "component_recall": f"{best['component_recall']:.6f}",
                }
            )
    top_piece_rows: list[dict] = []
    for (_label_slug, _component_id), ranked in sorted(top_piece_store.items()):
        for rank, row in enumerate(ranked, start=1):
            top_piece_rows.append({"rank": rank, **row})
    return rows, top_piece_rows, component_inventory


def piecewise_union_rows(
    candidates: list[ClusterCandidate],
    annotations: list[dict],
    top_piece_rows: list[dict],
    out_dir: Path,
    min_component_dice: float,
) -> list[dict]:
    by_uid = {cand.uid: cand for cand in candidates}
    top1: dict[tuple[str, int], dict] = {}
    for row in top_piece_rows:
        if int(row["rank"]) == 1:
            top1[(row["label_slug"], int(row["component_id"]))] = row

    mask_dir = out_dir / "piecewise_union_masks"
    union_rows: list[dict] = []
    for ann in annotations:
        full = ann["full_mask"]
        full_area = int(full.sum())
        selected: list[dict] = []
        union = np.zeros_like(full, dtype=bool)
        for component_id, _component_area in zip(ann["component_ids"], ann["component_areas"]):
            row = top1.get((ann["label_slug"], int(component_id)))
            if row is None:
                continue
            if float(row["component_dice"]) <= min_component_dice:
                continue
            cand = by_uid.get(row["candidate_uid"])
            if cand is None:
                continue
            union |= cand.mask
            selected.append(row)

        overlap = int((union & full).sum())
        union_dice, union_precision, union_recall = metrics(overlap, int(union.sum()), full_area)
        best_full = max(
            (row for row in top_piece_rows if row["label_slug"] == ann["label_slug"]),
            key=lambda row: float(row["full_dice"]),
            default=None,
        )
        best_piece = max(
            (row for row in top_piece_rows if row["label_slug"] == ann["label_slug"]),
            key=lambda row: float(row["component_dice"]),
            default=None,
        )
        mask_path = mask_dir / f"{ann['label_slug']}_piecewise_union.png"
        save_mask(union, mask_path)
        union_rows.append(
            {
                "label": ann["label"],
                "label_slug": ann["label_slug"],
                "n_gt_components": len(ann["component_ids"]),
                "n_selected_components": len(selected),
                "selected_component_ids": ";".join(str(row["component_id"]) for row in selected),
                "selected_candidate_uids": ";".join(row["candidate_uid"] for row in selected),
                "selected_candidate_mask_paths": ";".join(row["candidate_mask_path"] for row in selected),
                "union_mask_path": str(mask_path),
                "union_area": int(union.sum()),
                "full_gt_area": full_area,
                "full_overlap": overlap,
                "piecewise_union_dice": f"{union_dice:.6f}",
                "piecewise_union_precision": f"{union_precision:.6f}",
                "piecewise_union_recall": f"{union_recall:.6f}",
                "best_single_full_dice": "" if best_full is None else best_full["full_dice"],
                "best_single_full_precision": "" if best_full is None else best_full["full_precision"],
                "best_single_full_recall": "" if best_full is None else best_full["full_recall"],
                "best_single_component_dice": "" if best_piece is None else best_piece["component_dice"],
                "best_single_component_uid": "" if best_piece is None else best_piece["candidate_uid"],
                "union_min_component_dice": f"{min_component_dice:.6f}",
            }
        )
    return union_rows


def policy_union_rows(
    candidates: list[ClusterCandidate],
    annotations: list[dict],
    top_piece_rows: list[dict],
    out_dir: Path,
    largest_components_for_union: int,
) -> tuple[list[dict], list[dict]]:
    by_uid = {cand.uid: cand for cand in candidates}
    by_label_component: dict[tuple[str, int], list[dict]] = {}
    for row in top_piece_rows:
        key = (row["label_slug"], int(row["component_id"]))
        by_label_component.setdefault(key, []).append(row)
    for rows in by_label_component.values():
        rows.sort(
            key=lambda row: (
                float(row["component_dice"]),
                float(row["component_precision"]),
                -int(row["cluster_size"]),
                float(row["full_dice"]),
            ),
            reverse=True,
        )

    mask_dir = out_dir / "policy_union_masks"
    summary_rows: list[dict] = []
    selected_all: list[dict] = []
    for ann in annotations:
        policy_name, policy = policy_for_label(ann["label_slug"], ann["label"])
        full = ann["full_mask"]
        components = sorted(
            [
                {"component_id": int(component_id), "component_area": int(component_area)}
                for component_id, component_area in zip(ann["component_ids"], ann["component_areas"])
                if int(component_area) >= policy.min_component_area
            ],
            key=lambda item: item["component_area"],
            reverse=True,
        )[:largest_components_for_union]
        selected_union = np.zeros_like(full, dtype=bool)
        selected_rows: list[dict] = []
        selected_uids: set[str] = set()

        for component in components:
            rows = by_label_component.get((ann["label_slug"], component["component_id"]), [])
            for row in rows:
                if len(selected_rows) >= policy.max_selected:
                    break
                if float(row["component_dice"]) < policy.min_component_dice:
                    continue
                if float(row["component_precision"]) < policy.min_component_precision:
                    continue
                cand = by_uid.get(row["candidate_uid"])
                if cand is None or cand.uid in selected_uids:
                    continue
                overlap_selected = overlap_with_selected(cand.mask, selected_union)
                if overlap_selected > policy.max_overlap_with_selected:
                    continue
                inc_precision = incremental_precision(cand.mask, selected_union, full)
                if inc_precision < policy.min_incremental_precision:
                    continue
                before = mask_metrics(selected_union, full)
                proposed = selected_union | cand.mask
                after = mask_metrics(proposed, full)
                if before["pred_area"] > 0 and after["precision"] < before["precision"] - policy.max_precision_drop:
                    continue
                selected_union = proposed
                selected_uids.add(cand.uid)
                selected_rows.append(
                    {
                        "label": ann["label"],
                        "label_slug": ann["label_slug"],
                        "policy_name": policy_name,
                        "component_id": component["component_id"],
                        "component_area": component["component_area"],
                        "candidate_uid": cand.uid,
                        "source_keys": cand.source_keys,
                        "cluster_size": cand.cluster_size,
                        "members": cand.member_ids,
                        "candidate_area": cand.area,
                        "candidate_mask_path": str(cand.mask_path),
                        "component_dice": row["component_dice"],
                        "component_precision": row["component_precision"],
                        "component_recall": row["component_recall"],
                        "full_dice": row["full_dice"],
                        "full_precision": row["full_precision"],
                        "full_recall": row["full_recall"],
                        "incremental_precision": f"{inc_precision:.6f}",
                        "overlap_with_selected": f"{overlap_selected:.6f}",
                        "union_dice_after": f"{float(after['dice']):.6f}",
                        "union_precision_after": f"{float(after['precision']):.6f}",
                        "union_recall_after": f"{float(after['recall']):.6f}",
                        "selection_reason": "policy_gate_pass",
                    }
                )
                break
            if len(selected_rows) >= policy.max_selected:
                break

        fallback_reason = ""
        if not selected_rows:
            label_rows = [row for row in top_piece_rows if row["label_slug"] == ann["label_slug"]]
            best_full = max(label_rows, key=lambda row: float(row["full_dice"]), default=None)
            if best_full is not None:
                cand = by_uid.get(best_full["candidate_uid"])
                if cand is not None:
                    selected_union = cand.mask.copy()
                    fallback_reason = "best_full_dice_candidate"
                    selected_rows.append(
                        {
                            "label": ann["label"],
                            "label_slug": ann["label_slug"],
                            "policy_name": policy_name,
                            "component_id": best_full["component_id"],
                            "component_area": best_full["component_area"],
                            "candidate_uid": cand.uid,
                            "source_keys": cand.source_keys,
                            "cluster_size": cand.cluster_size,
                            "members": cand.member_ids,
                            "candidate_area": cand.area,
                            "candidate_mask_path": str(cand.mask_path),
                            "component_dice": best_full["component_dice"],
                            "component_precision": best_full["component_precision"],
                            "component_recall": best_full["component_recall"],
                            "full_dice": best_full["full_dice"],
                            "full_precision": best_full["full_precision"],
                            "full_recall": best_full["full_recall"],
                            "incremental_precision": "",
                            "overlap_with_selected": "",
                            "union_dice_after": best_full["full_dice"],
                            "union_precision_after": best_full["full_precision"],
                            "union_recall_after": best_full["full_recall"],
                            "selection_reason": fallback_reason,
                        }
                    )

        selected_all.extend(selected_rows)
        final_metrics = mask_metrics(selected_union, full)
        best_full = max(
            (row for row in top_piece_rows if row["label_slug"] == ann["label_slug"]),
            key=lambda row: float(row["full_dice"]),
            default=None,
        )
        best_piece = max(
            (row for row in top_piece_rows if row["label_slug"] == ann["label_slug"]),
            key=lambda row: float(row["component_dice"]),
            default=None,
        )
        mask_path = mask_dir / f"{ann['label_slug']}_policy_union.png"
        save_mask(selected_union, mask_path)
        summary_rows.append(
            {
                "label": ann["label"],
                "label_slug": ann["label_slug"],
                "policy_name": policy_name,
                "n_gt_components": len(ann["component_ids"]),
                "n_policy_components_considered": len(components),
                "n_selected_components": len(selected_rows),
                "selected_component_ids": ";".join(str(row["component_id"]) for row in selected_rows),
                "selected_candidate_uids": ";".join(row["candidate_uid"] for row in selected_rows),
                "selected_candidate_mask_paths": ";".join(row["candidate_mask_path"] for row in selected_rows),
                "union_mask_path": str(mask_path),
                "union_area": int(selected_union.sum()),
                "full_gt_area": int(full.sum()),
                "full_overlap": int((selected_union & full).sum()),
                "policy_union_dice": f"{float(final_metrics['dice']):.6f}",
                "policy_union_precision": f"{float(final_metrics['precision']):.6f}",
                "policy_union_recall": f"{float(final_metrics['recall']):.6f}",
                "best_single_full_dice": "" if best_full is None else best_full["full_dice"],
                "best_single_full_precision": "" if best_full is None else best_full["full_precision"],
                "best_single_full_recall": "" if best_full is None else best_full["full_recall"],
                "best_single_component_dice": "" if best_piece is None else best_piece["component_dice"],
                "best_single_component_uid": "" if best_piece is None else best_piece["candidate_uid"],
                "fallback_reason": fallback_reason,
                "min_component_dice": policy.min_component_dice,
                "min_component_precision": policy.min_component_precision,
                "min_incremental_precision": policy.min_incremental_precision,
                "max_overlap_with_selected": policy.max_overlap_with_selected,
                "max_precision_drop": policy.max_precision_drop,
                "max_selected": policy.max_selected,
                "min_component_area": policy.min_component_area,
            }
        )
    return summary_rows, selected_all


def top_rows(score_rows: list[dict], top_n: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in score_rows:
        grouped.setdefault(row["label"], []).append(row)
    out = []
    for label, rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda row: (
                float(row["component_dice"]),
                float(row["component_precision"]),
                float(row["component_recall"]),
                float(row["full_dice"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows[:top_n], start=1):
            out.append({"rank": rank, **row})
    return out


def write_html(
    out_dir: Path,
    args: argparse.Namespace,
    n_raw: int,
    n_clusters: int,
    best_rows: list[dict],
    top_piece_rows: list[dict],
    union_rows: list[dict],
    policy_union_summary: list[dict],
) -> None:
    rows_html = []
    for row in best_rows:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['rank']}</td>"
            f"<td>{html.escape(row['candidate_uid'])}</td>"
            f"<td>{html.escape(row['source_keys'])}</td>"
            f"<td>{html.escape(row['component_dice'])}</td>"
            f"<td>{html.escape(row['component_precision'])}</td>"
            f"<td>{html.escape(row['component_recall'])}</td>"
            f"<td>{html.escape(row['full_dice'])}</td>"
            "</tr>"
        )
    piece_rows_html = []
    for row in top_piece_rows:
        if int(row["rank"]) > 3:
            continue
        piece_rows_html.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(str(row['component_id']))}</td>"
            f"<td>{row['rank']}</td>"
            f"<td>{html.escape(row['candidate_uid'])}</td>"
            f"<td>{html.escape(row['source_keys'])}</td>"
            f"<td>{html.escape(row['component_dice'])}</td>"
            f"<td>{html.escape(row['component_precision'])}</td>"
            f"<td>{html.escape(row['component_recall'])}</td>"
            "</tr>"
        )
    union_rows_html = []
    for row in union_rows:
        union_rows_html.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(str(row['n_selected_components']))}/{html.escape(str(row['n_gt_components']))}</td>"
            f"<td>{html.escape(row['piecewise_union_dice'])}</td>"
            f"<td>{html.escape(row['piecewise_union_precision'])}</td>"
            f"<td>{html.escape(row['piecewise_union_recall'])}</td>"
            f"<td>{html.escape(row['best_single_full_dice'])}</td>"
            f"<td>{html.escape(row['best_single_component_dice'])}</td>"
            "</tr>"
        )
    policy_rows_html = []
    for row in policy_union_summary:
        policy_rows_html.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(row['policy_name'])}</td>"
            f"<td>{html.escape(str(row['n_selected_components']))}/{html.escape(str(row['n_policy_components_considered']))}</td>"
            f"<td>{html.escape(row['policy_union_dice'])}</td>"
            f"<td>{html.escape(row['policy_union_precision'])}</td>"
            f"<td>{html.escape(row['policy_union_recall'])}</td>"
            f"<td>{html.escape(row['best_single_full_dice'])}</td>"
            f"<td>{html.escape(row['fallback_reason'])}</td>"
            "</tr>"
        )
    (out_dir / "index.html").write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Xenium HD-style SAM score</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;font-size:13px}}th{{background:#edf1f6}}</style>
</head><body>
<h1>Xenium HD-style SAM candidate scoring</h1>
	<p>TMA{args.tma}; setting <code>{html.escape(args.setting_name)}</code>; sources <code>{html.escape(','.join(args.source_keys))}</code>; annotation policy <code>{args.annotation_policy}</code>.</p>
	<p>Raw masks: {n_raw:,}. Complete-link IoU clusters: {n_clusters:,} at threshold {args.cluster_threshold:.2f}.</p>
	<p><b>Policy union:</b> the old VisiumHD-style pre-VLM selection. For each tissue class, large disconnected GT pieces are visited separately; candidates must pass gates on piece Dice, piece Precision, overlap with already selected masks, incremental Precision, and allowed Precision drop before being unioned.</p>
	<p><b>Piece-wise assembly:</b> each disconnected GT component is scored separately, the top candidate for each covered component is unioned, and the union is then evaluated against the full tissue-class mask.</p>
	<h2>Policy union by tissue class</h2>
	<table><thead><tr><th>Label</th><th>Policy</th><th>Selected / considered pieces</th><th>Union Dice</th><th>Union Precision</th><th>Union Recall</th><th>Best single full Dice</th><th>Fallback</th></tr></thead>
	<tbody>{''.join(policy_rows_html)}</tbody></table>
	<h2>Piece-wise union by tissue class</h2>
	<table><thead><tr><th>Label</th><th>Selected components</th><th>Union Dice</th><th>Union Precision</th><th>Union Recall</th><th>Best single full Dice</th><th>Best single component Dice</th></tr></thead>
	<tbody>{''.join(union_rows_html)}</tbody></table>
<h2>Top candidates per GT piece</h2>
<table><thead><tr><th>Label</th><th>GT piece</th><th>Rank</th><th>Candidate</th><th>Sources</th><th>Piece Dice</th><th>Precision</th><th>Recall</th></tr></thead>
<tbody>{''.join(piece_rows_html)}</tbody></table>
<h2>Legacy best single candidates by label</h2>
<table><thead><tr><th>Label</th><th>Rank</th><th>Candidate</th><th>Sources</th><th>Component Dice</th><th>Precision</th><th>Recall</th><th>Full Dice</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_candidates(args.candidate_root, args.tma, args.source_keys, args.setting_name, args.min_area)
    if not raw:
        raise SystemExit("No candidate masks found")
    expected_shape = raw[0].mask.shape
    expected_size = (expected_shape[1], expected_shape[0])
    annotations = load_annotation_masks(args.prepared_root, args.tma, args.annotation_policy, expected_size)
    clusters, inventory, edges = build_cluster_candidates(raw, args.out_dir, args.cluster_threshold)
    score_rows, top_piece_rows, component_inventory = score_candidates(clusters, annotations, args.top_n)
    union_rows = piecewise_union_rows(
        clusters,
        annotations,
        top_piece_rows,
        args.out_dir,
        args.union_min_component_dice,
    )
    policy_union_summary, policy_selected_rows = policy_union_rows(
        clusters,
        annotations,
        top_piece_rows,
        args.out_dir,
        args.largest_components_for_policy_union,
    )
    best = top_rows(score_rows, args.top_n)

    write_csv(args.out_dir / "tables/cluster_inventory.csv", inventory)
    write_csv(args.out_dir / "tables/cluster_edges.csv", edges)
    write_csv(args.out_dir / "tables/gt_component_inventory.csv", component_inventory)
    write_csv(args.out_dir / "tables/candidate_component_scores.csv", score_rows)
    write_csv(args.out_dir / "tables/top_candidates_by_component_dice.csv", best)
    write_csv(args.out_dir / "tables/top_candidates_by_gt_piece.csv", top_piece_rows)
    write_csv(args.out_dir / "tables/component_union_summary.csv", union_rows)
    write_csv(args.out_dir / "tables/policy_union_summary.csv", policy_union_summary)
    write_csv(args.out_dir / "tables/selected_policy_union_candidates.csv", policy_selected_rows)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "prepared_root": str(args.prepared_root),
                "candidate_root": str(args.candidate_root),
                "tma": args.tma,
                "setting_name": args.setting_name,
                "source_keys": args.source_keys,
                "annotation_policy": args.annotation_policy,
                "cluster_threshold": args.cluster_threshold,
                "union_min_component_dice": args.union_min_component_dice,
                "largest_components_for_policy_union": args.largest_components_for_policy_union,
                "n_raw_candidates": len(raw),
                "n_cluster_candidates": len(clusters),
                "n_score_rows": len(score_rows),
                "n_top_piece_rows": len(top_piece_rows),
                "n_component_union_rows": len(union_rows),
                "n_policy_union_rows": len(policy_union_summary),
                "n_policy_selected_rows": len(policy_selected_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(args.out_dir, args, len(raw), len(clusters), best, top_piece_rows, union_rows, policy_union_summary)
    print(
        f"raw={len(raw)} clusters={len(clusters)} score_rows={len(score_rows)} "
        f"piece_rows={len(top_piece_rows)} unions={len(union_rows)} "
        f"policy_unions={len(policy_union_summary)} out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
