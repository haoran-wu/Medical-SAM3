#!/usr/bin/env python3
"""Serve an interactive manual alignment tool for Xenium morphology-to-H&E QC."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xenium_wholeslide_tif_he_translation_registration as ws_align


DEFAULT_ROOT = Path("output/xenium_silica_wholeslide_tif_he_translation_registration_20260624")
DEFAULT_SLIDE = "771-1"
DEFAULT_TMA: int | None = None


def read_manifest(root: Path, slide: str) -> dict:
    path = root / f"{slide}_wholeslide_tif_to_he_transform.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tma_options_for_root(root: Path) -> list[dict]:
    options: list[dict] = []
    for path in sorted(
        (ws_align.base.XENIUM_ROOT / "metadata").glob("TMA*.csv"),
        key=ws_align.base.tma_number_from_path,
    ):
        tma = ws_align.base.tma_number_from_path(path)
        try:
            rows = ws_align.base.read_tma_frame(tma)
        except Exception:
            continue
        if not rows:
            continue
        slide = ws_align.base.clean_histo(rows[0].get("HistoSlide", ""))
        manifest_path = root / f"{slide}_wholeslide_tif_to_he_transform.json"
        options.append(
            {
                "tma": int(tma),
                "slide": slide,
                "n_cells": int(len(rows)),
                "manifest_ready": bool(manifest_path.exists()),
            }
        )
    return options


def make_cyan_overlay(gray: np.ndarray) -> Image.Image:
    arr = gray.astype(np.float32)
    finite = arr[arr > 0]
    if finite.size:
        lo, hi = np.percentile(finite, [25, 99.5])
    else:
        lo, hi = 0.0, 1.0
    alpha = np.clip((arr - lo) / max(hi - lo, 1e-3), 0, 1)
    alpha = (alpha ** 0.7 * 210).astype(np.uint8)
    rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = 0
    rgba[:, :, 1] = 215
    rgba[:, :, 2] = 255
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def sanitize_token(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def crop_box_from_points(xy: np.ndarray, size: tuple[int, int], pad_px: int) -> list[int]:
    finite = np.isfinite(xy).all(axis=1)
    pts = xy[finite]
    if len(pts) == 0:
        raise ValueError("No finite TMA points available for crop")
    width, height = size
    x0 = max(0, int(math.floor(float(pts[:, 0].min()) - pad_px)))
    y0 = max(0, int(math.floor(float(pts[:, 1].min()) - pad_px)))
    x1 = min(width, int(math.ceil(float(pts[:, 0].max()) + pad_px)))
    y1 = min(height, int(math.ceil(float(pts[:, 1].max()) + pad_px)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop bbox from TMA points: {[x0, y0, x1, y1]}")
    return [x0, y0, x1, y1]


def crop_image(path: Path, bbox_xyxy: list[int], out_path: Path) -> None:
    x0, y0, x1, y1 = bbox_xyxy
    Image.open(path).crop((x0, y0, x1, y1)).save(out_path)


def ensure_assets(
    root: Path,
    slide: str,
    tma: int | None = None,
    pad_px: int = 0,
    options: list[dict] | None = None,
) -> dict:
    manifest = read_manifest(root, slide)
    he_path = root / f"{slide}_wholeslide_rotated_he_grid.png"
    morph_path = root / f"{slide}_wholeslide_morphology_grid.png"
    warped_path = root / f"{slide}_wholeslide_morphology_affine_warped.png"
    overlay_path = root / f"{slide}_wholeslide_morphology_affine_warped_cyan.png"
    if not warped_path.exists() or not overlay_path.exists():
        morph = np.asarray(Image.open(morph_path).convert("L"))
        affine = np.array(manifest["selected_transform"]["affine_xy_from_morphology_to_oriented_he"], dtype=float)
        warped = ws_align.warp_affine_image_u8(morph, affine)
        Image.fromarray(warped, "L").save(warped_path)
        make_cyan_overlay(warped).save(overlay_path)
    width, height = Image.open(he_path).size
    unit_id = slide
    mode = "whole_slide"
    crop_bbox_xyxy = [0, 0, int(width), int(height)]
    tma_summary = None
    if tma is not None:
        mode = "single_tma"
        unit_id = f"{slide}_TMA{tma}"
        rows = ws_align.base.read_tma_frame(int(tma))
        morph_meta = manifest["morphology"]
        affine = np.array(manifest["selected_transform"]["affine_xy_from_morphology_to_oriented_he"], dtype=float)
        morph_xy = ws_align.point_coordinates(rows, morph_meta)
        he_xy = ws_align.apply_affine_to_points(morph_xy, affine)
        crop_bbox_xyxy = crop_box_from_points(he_xy, (width, height), pad_px)
        safe_unit = sanitize_token(unit_id)
        he_crop_path = root / f"{safe_unit}_manual_he_crop.png"
        warped_crop_path = root / f"{safe_unit}_manual_morphology_affine_warped.png"
        overlay_crop_path = root / f"{safe_unit}_manual_morphology_affine_warped_cyan.png"
        crop_image(he_path, crop_bbox_xyxy, he_crop_path)
        crop_image(warped_path, crop_bbox_xyxy, warped_crop_path)
        crop_image(overlay_path, crop_bbox_xyxy, overlay_crop_path)
        he_path = he_crop_path
        warped_path = warped_crop_path
        overlay_path = overlay_crop_path
        morph_path = warped_crop_path
        width, height = Image.open(he_path).size
        tma_summary = {
            "tma": int(tma),
            "n_cells": int(len(rows)),
            "pad_px": int(pad_px),
            "cell_bbox_xyxy_in_oriented_he_grid": [
                float(np.nanmin(he_xy[:, 0])),
                float(np.nanmin(he_xy[:, 1])),
                float(np.nanmax(he_xy[:, 0])),
                float(np.nanmax(he_xy[:, 1])),
            ],
        }
    live_params_path = root / "manual_alignment_params" / f"{sanitize_token(unit_id)}_manual_alignment_live.json"
    existing_manual_params = None
    if live_params_path.exists():
        try:
            existing_manual_params = json.loads(live_params_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manual_params = None
    if existing_manual_params is not None:
        old_bbox = existing_manual_params.get("crop_bbox_xyxy_in_oriented_he_grid")
        old_size = existing_manual_params.get("image_size_px") or {}
        same_bbox = old_bbox == crop_bbox_xyxy
        same_size = old_size.get("width") == width and old_size.get("height") == height
        if not (same_bbox and same_size):
            existing_manual_params = None
    return {
        "unit_id": unit_id,
        "slide": slide,
        "mode": mode,
        "tma": tma_summary,
        "width": width,
        "height": height,
        "he": he_path.name,
        "morphology_raw": morph_path.name,
        "morphology_affine_warped": warped_path.name,
        "morphology_affine_warped_cyan": overlay_path.name,
        "manifest": f"{slide}_wholeslide_tif_to_he_transform.json",
        "crop_bbox_xyxy_in_oriented_he_grid": crop_bbox_xyxy,
        "options": options or tma_options_for_root(root),
        "existing_manual_params": existing_manual_params,
        "selected_transform": manifest["selected_transform"],
    }


def html_page(config: dict) -> str:
    cfg = json.dumps(config, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Xenium H&E Morphology Manual Alignment</title>
  <style>
    :root {{
      color-scheme: light;
      --panel: #f5f7fb;
      --line: #d8dee9;
      --text: #1f2937;
      --muted: #667085;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #e9edf3;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 10px 14px;
      background: rgba(245, 247, 251, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.3;
    }}
    .subtitle {{
      margin-top: 2px;
      font-size: 12px;
      color: var(--muted);
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      justify-content: flex-end;
      gap: 8px;
      max-width: 980px;
    }}
    label {{
      display: grid;
      gap: 2px;
      font-size: 11px;
      color: var(--muted);
    }}
    input, select, button {{
      height: 30px;
      border: 1px solid #c7cfdb;
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
    }}
    input[type="number"] {{ width: 86px; padding: 0 6px; }}
    input[type="range"] {{ width: 110px; }}
    select {{ padding: 0 6px; }}
    button {{
      padding: 0 10px;
      cursor: pointer;
      font-weight: 600;
    }}
    button.primary {{
      background: #1463ff;
      color: white;
      border-color: #1463ff;
    }}
    .readout {{
      min-width: 290px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.35;
      white-space: pre;
    }}
    main {{
      height: calc(100vh - 86px);
      overflow: auto;
      padding: 16px;
    }}
    #stage {{
      position: relative;
      margin: 0 auto;
      background: white;
      box-shadow: 0 3px 18px rgba(15, 23, 42, 0.18);
      transform-origin: top left;
      cursor: grab;
      touch-action: none;
    }}
    #stage.dragging {{ cursor: grabbing; }}
    .layer {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      image-rendering: auto;
      user-select: none;
      -webkit-user-drag: none;
      pointer-events: none;
      transform-origin: 0 0;
    }}
    #heLayer {{ position: absolute; inset: 0; }}
    #morphLayer {{ will-change: transform, opacity; }}
    #crosshair {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      background-image: linear-gradient(rgba(20,99,255,.18) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(20,99,255,.18) 1px, transparent 1px);
      background-size: 100px 100px;
      display: none;
    }}
    #toast {{
      position: fixed;
      right: 14px;
      bottom: 14px;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(17, 24, 39, .92);
      color: white;
      font-size: 13px;
      opacity: 0;
      transform: translateY(8px);
      transition: opacity .16s ease, transform .16s ease;
      pointer-events: none;
    }}
    #toast.show {{ opacity: 1; transform: translateY(0); }}
  </style>
</head>
<body>
<header>
  <div>
    <h1>Xenium H&E ↔ morphology manual alignment</h1>
    <div class="subtitle" id="subtitle">Bottom: rotated/mirrored H&E. Top: affine-warped morphology. Drag the image, adjust scale, then Save.</div>
  </div>
  <div class="controls">
    <label>View
      <select id="tmaSelect"></select>
    </label>
    <label>Top layer
      <select id="layerMode">
        <option value="cyan">affine morphology cyan</option>
        <option value="gray">affine morphology gray</option>
        <option value="raw">raw morphology gray</option>
      </select>
    </label>
    <label>dx px<input id="dxInput" type="number" step="0.1" value="0"></label>
    <label>dy px<input id="dyInput" type="number" step="0.1" value="0"></label>
    <label>scale both<input id="scaleInput" type="number" min="0.50" max="1.80" step="0.001" value="1"></label>
    <label>stretch X<input id="scaleXInput" type="number" min="0.50" max="1.80" step="0.001" value="1"></label>
    <label>stretch Y<input id="scaleYInput" type="number" min="0.50" max="1.80" step="0.001" value="1"></label>
    <label>step px<input id="stepInput" type="number" step="0.1" value="1"></label>
    <label>both slider<input id="scaleRange" type="range" min="0.50" max="1.80" step="0.001" value="1"></label>
    <label>X slider<input id="scaleXRange" type="range" min="0.50" max="1.80" step="0.001" value="1"></label>
    <label>Y slider<input id="scaleYRange" type="range" min="0.50" max="1.80" step="0.001" value="1"></label>
    <label>opacity<input id="opacityInput" type="range" min="0" max="1" step="0.02" value="0.72"></label>
    <label>zoom<input id="zoomInput" type="range" min="0.10" max="1.60" step="0.02" value="0.22"></label>
    <button id="resetBtn">Reset</button>
    <button id="gridBtn">Grid</button>
    <button id="saveBtn" class="primary">Save</button>
    <div id="readout" class="readout"></div>
  </div>
</header>
<main id="viewport">
  <div id="stage">
    <img id="heLayer" class="layer" alt="H&E" />
    <img id="morphLayer" class="layer" alt="morphology" />
    <div id="crosshair"></div>
  </div>
</main>
<div id="toast"></div>
<script>
const CONFIG = {cfg};
const state = {{
  dx: 0,
  dy: 0,
  scale: 1,
  scaleX: 1,
  scaleY: 1,
  opacity: 0.72,
  zoom: 0.22,
  mode: 'cyan',
  grid: false,
  dragging: false,
  dragStart: null,
  autosaveTimer: null,
}};
const stage = document.getElementById('stage');
const heLayer = document.getElementById('heLayer');
const morphLayer = document.getElementById('morphLayer');
const crosshair = document.getElementById('crosshair');
const readout = document.getElementById('readout');
const dxInput = document.getElementById('dxInput');
const dyInput = document.getElementById('dyInput');
const scaleInput = document.getElementById('scaleInput');
const scaleXInput = document.getElementById('scaleXInput');
const scaleYInput = document.getElementById('scaleYInput');
const scaleRange = document.getElementById('scaleRange');
const scaleXRange = document.getElementById('scaleXRange');
const scaleYRange = document.getElementById('scaleYRange');
const stepInput = document.getElementById('stepInput');
const opacityInput = document.getElementById('opacityInput');
const zoomInput = document.getElementById('zoomInput');
const layerMode = document.getElementById('layerMode');
const tmaSelect = document.getElementById('tmaSelect');
const toast = document.getElementById('toast');

heLayer.src = CONFIG.he;
const sources = {{
  cyan: CONFIG.morphology_affine_warped_cyan,
  gray: CONFIG.morphology_affine_warped,
  raw: CONFIG.morphology_raw,
}};
morphLayer.src = sources[state.mode];
function addViewOption(value, text, selected=false, disabled=false) {{
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = text;
  opt.selected = selected;
  opt.disabled = disabled;
  tmaSelect.appendChild(opt);
}}
const slides = [];
for (const option of CONFIG.options || []) {{
  if (!slides.includes(option.slide)) slides.push(option.slide);
}}
if (!slides.includes(CONFIG.slide)) slides.unshift(CONFIG.slide);
for (const slide of slides) {{
  addViewOption(
    `${{slide}}:`,
    `Whole slide | ${{slide}}`,
    CONFIG.mode === 'whole_slide' && CONFIG.slide === slide,
    false
  );
}}
for (const option of CONFIG.options || []) {{
  addViewOption(
    `${{option.slide}}:${{option.tma}}`,
    `TMA${{option.tma}} | ${{option.slide}} | ${{option.n_cells.toLocaleString()}} cells${{option.manifest_ready ? '' : ' | missing slide assets'}}`,
    CONFIG.tma && option.tma === CONFIG.tma.tma && option.slide === CONFIG.slide,
    !option.manifest_ready
  );
}}
tmaSelect.addEventListener('change', () => {{
  const [slide, tma] = tmaSelect.value.split(':');
  window.location.href = `/?slide=${{encodeURIComponent(slide)}}&tma=${{encodeURIComponent(tma)}}`;
}});
if (CONFIG.mode === 'single_tma') {{
  document.title = `Xenium ${{CONFIG.unit_id}} manual alignment`;
  document.querySelector('h1').textContent = `Xenium ${{CONFIG.unit_id}} manual alignment`;
  document.getElementById('subtitle').textContent =
    'Single-core crop. Bottom: local rotated/mirrored H&E crop. Top: local affine-warped morphology crop. Adjust dx/dy/scale for this TMA only.';
  const rawOption = [...layerMode.options].find(option => option.value === 'raw');
  if (rawOption) rawOption.textContent = 'affine morphology gray copy';
}} else {{
  document.title = `Xenium ${{CONFIG.slide}} whole-slide manual alignment`;
  document.querySelector('h1').textContent = `Xenium ${{CONFIG.slide}} whole-slide manual alignment`;
  document.getElementById('subtitle').textContent =
    'Whole-slide view. Bottom: rotated/mirrored H&E grid. Top: whole-slide affine-warped morphology. Adjust dx/dy/stretch for this whole slide.';
}}

function fmt(x) {{ return Number(x).toFixed(2); }}
function fmt4(x) {{ return Number(x).toFixed(4); }}
function payload(kind='autosave') {{
  return {{
    kind,
    unit_id: CONFIG.unit_id,
    slide: CONFIG.slide,
    mode: CONFIG.mode,
    tma: CONFIG.tma,
    timestamp: new Date().toISOString(),
    coordinate_space: CONFIG.mode === 'single_tma'
      ? 'single-TMA crop pixels within rotated_mirrored_H&E level-4 display grid'
      : 'rotated_mirrored_H&E level-4 display grid pixels',
    crop_bbox_xyxy_in_oriented_he_grid: CONFIG.crop_bbox_xyxy_in_oriented_he_grid,
    base_transform: CONFIG.selected_transform.transform_type,
    base_affine_xy_from_morphology_to_oriented_he: CONFIG.selected_transform.affine_xy_from_morphology_to_oriented_he,
    manual_translation_after_base_affine_px: {{ dx: state.dx, dy: state.dy }},
    manual_scale_after_base_affine: state.scale,
    manual_scale_xy_after_base_affine: {{ sx: state.scaleX, sy: state.scaleY }},
    manual_stretch_after_base_affine: {{
      x_over_y: state.scaleY === 0 ? null : state.scaleX / state.scaleY,
      sx_minus_sy: state.scaleX - state.scaleY
    }},
    manual_scale_origin: CONFIG.mode === 'single_tma' ? 'center of this single-TMA crop' : 'center of image grid',
    layer_mode: state.mode,
    opacity: state.opacity,
    image_size_px: {{ width: CONFIG.width, height: CONFIG.height }},
    note: CONFIG.mode === 'single_tma'
      ? 'Apply this residual dx/dy/scaleX/scaleY after the saved morphology-to-rotated-H&E affine, but only inside this TMA crop. Positive dx moves morphology right; positive dy moves morphology down.'
      : 'Apply manual dx/dy/scaleX/scaleY after the saved morphology-to-rotated-H&E affine for this whole-slide grid. Positive dx moves morphology right; positive dy moves morphology down.'
  }};
}}
function showToast(text) {{
  toast.textContent = text;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 1400);
}}
function updateView(autosave=true) {{
  stage.style.width = `${{CONFIG.width * state.zoom}}px`;
  stage.style.height = `${{CONFIG.height * state.zoom}}px`;
  const tx = (state.dx + (1 - state.scaleX) * CONFIG.width / 2) * state.zoom;
  const ty = (state.dy + (1 - state.scaleY) * CONFIG.height / 2) * state.zoom;
  morphLayer.style.transform = `matrix(${{state.scaleX}}, 0, 0, ${{state.scaleY}}, ${{tx}}, ${{ty}})`;
  morphLayer.style.opacity = String(state.opacity);
  morphLayer.style.mixBlendMode = state.mode === 'cyan' ? 'normal' : 'multiply';
  morphLayer.src = sources[state.mode];
  crosshair.style.display = state.grid ? 'block' : 'none';
  dxInput.value = fmt(state.dx);
  dyInput.value = fmt(state.dy);
  scaleInput.value = fmt4(state.scale);
  scaleXInput.value = fmt4(state.scaleX);
  scaleYInput.value = fmt4(state.scaleY);
  scaleRange.value = state.scale;
  scaleXRange.value = state.scaleX;
  scaleYRange.value = state.scaleY;
  opacityInput.value = state.opacity;
  zoomInput.value = state.zoom;
  layerMode.value = state.mode;
  const p = payload();
  readout.textContent = `dx=${{fmt(state.dx)}} px\\ndy=${{fmt(state.dy)}} px\\nscaleX=${{fmt4(state.scaleX)}}\\nscaleY=${{fmt4(state.scaleY)}}\\nmode=${{state.mode}}`;
  if (autosave) scheduleAutosave(p);
}}
function scheduleAutosave(p) {{
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(() => saveParams('autosave', false), 350);
}}
async function saveParams(kind='save', notify=true) {{
  const res = await fetch('/save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(payload(kind)),
  }});
  if (!res.ok) {{
    showToast('Save failed');
    return;
  }}
  const data = await res.json();
  if (notify) showToast(`Saved: ${{data.path}}`);
}}
function setDxDy(dx, dy) {{
  state.dx = Number(dx);
  state.dy = Number(dy);
  updateView();
}}
function setScale(scale) {{
  state.scale = Math.max(0.2, Math.min(5, Number(scale)));
  state.scaleX = state.scale;
  state.scaleY = state.scale;
  updateView();
}}
function setScaleX(scale) {{
  state.scaleX = Math.max(0.2, Math.min(5, Number(scale)));
  state.scale = (state.scaleX + state.scaleY) / 2;
  updateView();
}}
function setScaleY(scale) {{
  state.scaleY = Math.max(0.2, Math.min(5, Number(scale)));
  state.scale = (state.scaleX + state.scaleY) / 2;
  updateView();
}}
function applyExistingParams(params) {{
  if (!params || typeof params !== 'object') return;
  const shift = params.manual_translation_after_base_affine_px || {{}};
  if (Number.isFinite(Number(shift.dx))) state.dx = Number(shift.dx);
  if (Number.isFinite(Number(shift.dy))) state.dy = Number(shift.dy);
  const scaleXY = params.manual_scale_xy_after_base_affine || {{}};
  if (Number.isFinite(Number(scaleXY.sx))) state.scaleX = Number(scaleXY.sx);
  if (Number.isFinite(Number(scaleXY.sy))) state.scaleY = Number(scaleXY.sy);
  if (!Number.isFinite(Number(scaleXY.sx)) || !Number.isFinite(Number(scaleXY.sy))) {{
    const oldScale = Number(params.manual_scale_after_base_affine);
    if (Number.isFinite(oldScale) && oldScale > 0) {{
      state.scaleX = oldScale;
      state.scaleY = oldScale;
    }}
  }}
  state.scale = (state.scaleX + state.scaleY) / 2;
  if (Number.isFinite(Number(params.opacity))) state.opacity = Number(params.opacity);
  if (params.layer_mode && sources[params.layer_mode]) state.mode = params.layer_mode;
}}
stage.addEventListener('pointerdown', (event) => {{
  state.dragging = true;
  stage.classList.add('dragging');
  stage.setPointerCapture(event.pointerId);
  state.dragStart = {{ x: event.clientX, y: event.clientY, dx: state.dx, dy: state.dy }};
}});
stage.addEventListener('pointermove', (event) => {{
  if (!state.dragging || !state.dragStart) return;
  const ddx = (event.clientX - state.dragStart.x) / state.zoom;
  const ddy = (event.clientY - state.dragStart.y) / state.zoom;
  state.dx = state.dragStart.dx + ddx;
  state.dy = state.dragStart.dy + ddy;
  updateView();
}});
stage.addEventListener('pointerup', (event) => {{
  state.dragging = false;
  stage.classList.remove('dragging');
  try {{ stage.releasePointerCapture(event.pointerId); }} catch {{}}
  saveParams('autosave', false);
}});
stage.addEventListener('pointercancel', () => {{
  state.dragging = false;
  stage.classList.remove('dragging');
}});
dxInput.addEventListener('change', () => setDxDy(dxInput.value, state.dy));
dyInput.addEventListener('change', () => setDxDy(state.dx, dyInput.value));
scaleInput.addEventListener('change', () => setScale(scaleInput.value));
scaleRange.addEventListener('input', () => setScale(scaleRange.value));
scaleXInput.addEventListener('change', () => setScaleX(scaleXInput.value));
scaleYInput.addEventListener('change', () => setScaleY(scaleYInput.value));
scaleXRange.addEventListener('input', () => setScaleX(scaleXRange.value));
scaleYRange.addEventListener('input', () => setScaleY(scaleYRange.value));
opacityInput.addEventListener('input', () => {{
  state.opacity = Number(opacityInput.value);
  updateView(false);
}});
zoomInput.addEventListener('input', () => {{
  state.zoom = Number(zoomInput.value);
  updateView(false);
}});
layerMode.addEventListener('change', () => {{
  state.mode = layerMode.value;
  updateView();
}});
document.getElementById('resetBtn').addEventListener('click', () => setDxDy(0, 0));
document.getElementById('resetBtn').addEventListener('dblclick', () => {{
  state.dx = 0;
  state.dy = 0;
  state.scale = 1;
  state.scaleX = 1;
  state.scaleY = 1;
  updateView();
}});
document.getElementById('gridBtn').addEventListener('click', () => {{
  state.grid = !state.grid;
  updateView(false);
}});
document.getElementById('saveBtn').addEventListener('click', () => saveParams('save', true));
window.addEventListener('keydown', (event) => {{
  if (['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
  const base = Number(stepInput.value || 1);
  const step = event.shiftKey ? base * 10 : (event.altKey ? base * 0.2 : base);
  if (event.key === 'ArrowLeft') {{ state.dx -= step; event.preventDefault(); updateView(); }}
  if (event.key === 'ArrowRight') {{ state.dx += step; event.preventDefault(); updateView(); }}
  if (event.key === 'ArrowUp') {{ state.dy -= step; event.preventDefault(); updateView(); }}
  if (event.key === 'ArrowDown') {{ state.dy += step; event.preventDefault(); updateView(); }}
  if (event.key === '[' || event.key === '{{') {{
    const delta = 0.001 * (event.shiftKey ? 10 : 1);
    state.scale -= delta; state.scaleX -= delta; state.scaleY -= delta;
    event.preventDefault(); updateView();
  }}
  if (event.key === ']' || event.key === '}}') {{
    const delta = 0.001 * (event.shiftKey ? 10 : 1);
    state.scale += delta; state.scaleX += delta; state.scaleY += delta;
    event.preventDefault(); updateView();
  }}
  if (event.key === ',') {{ state.scaleX -= 0.001 * (event.shiftKey ? 10 : 1); state.scale = (state.scaleX + state.scaleY) / 2; event.preventDefault(); updateView(); }}
  if (event.key === '.') {{ state.scaleX += 0.001 * (event.shiftKey ? 10 : 1); state.scale = (state.scaleX + state.scaleY) / 2; event.preventDefault(); updateView(); }}
  if (event.key === ';') {{ state.scaleY -= 0.001 * (event.shiftKey ? 10 : 1); state.scale = (state.scaleX + state.scaleY) / 2; event.preventDefault(); updateView(); }}
  if (event.key === "'") {{ state.scaleY += 0.001 * (event.shiftKey ? 10 : 1); state.scale = (state.scaleX + state.scaleY) / 2; event.preventDefault(); updateView(); }}
  if (event.key.toLowerCase() === 's') {{ event.preventDefault(); saveParams('save', true); }}
}});
applyExistingParams(CONFIG.existing_manual_params);
updateView(false);
</script>
</body>
</html>
"""


class AlignmentHandler(SimpleHTTPRequestHandler):
    server_version = "XeniumManualAlignment/1.0"

    def __init__(self, *args, directory: str | None = None, **kwargs):
        self.root = Path(directory or ".").resolve()
        super().__init__(*args, directory=str(self.root), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            query = parse_qs(parsed.query, keep_blank_values=True)
            default_slide = self.server.default_slide  # type: ignore[attr-defined]
            default_tma = self.server.default_tma  # type: ignore[attr-defined]
            slide = query.get("slide", [default_slide])[0]
            default_tma_text = "" if default_tma is None else str(default_tma)
            tma_text = query.get("tma", [default_tma_text])[0]
            tma = int(tma_text) if str(tma_text).strip() else None
            try:
                config = ensure_assets(
                    self.root,
                    slide,
                    tma=tma,
                    pad_px=self.server.pad_px,  # type: ignore[attr-defined]
                    options=self.server.options,  # type: ignore[attr-defined]
                )
            except Exception as exc:
                body = (
                    "<!doctype html><html><body>"
                    "<h1>Failed to build alignment page</h1>"
                    f"<pre>{str(exc)}</pre>"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            html = html_page(config).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        self.path = unquote(parsed.path)
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return
        out_dir = self.root / "manual_alignment_params"
        out_dir.mkdir(parents=True, exist_ok=True)
        unit_id = sanitize_token(payload.get("unit_id", payload.get("slide", "slide")))
        live_path = out_dir / f"{unit_id}_manual_alignment_live.json"
        live_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved_path = live_path
        if payload.get("kind") == "save":
            stamp = time.strftime("%Y%m%d_%H%M%S")
            saved_path = out_dir / f"{unit_id}_manual_alignment_saved_{stamp}.json"
            saved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        response = json.dumps({"ok": True, "path": str(saved_path)}, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--slide", default=DEFAULT_SLIDE)
    parser.add_argument("--tma", type=int, default=DEFAULT_TMA, help="Default TMA number, e.g. 39")
    parser.add_argument("--pad-px", type=int, default=0, help="Single-TMA crop padding in display-grid pixels")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    root = args.root.resolve()
    options = tma_options_for_root(root)
    config = ensure_assets(root, args.slide, tma=args.tma, pad_px=args.pad_px, options=options)
    handler = lambda *a, **kw: AlignmentHandler(*a, directory=str(root), **kw)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.default_slide = args.slide  # type: ignore[attr-defined]
    httpd.default_tma = args.tma  # type: ignore[attr-defined]
    httpd.pad_px = args.pad_px  # type: ignore[attr-defined]
    httpd.options = options  # type: ignore[attr-defined]
    print(json.dumps({"url": f"http://{args.host}:{args.port}/", "root": str(root), "config": config}, indent=2))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
