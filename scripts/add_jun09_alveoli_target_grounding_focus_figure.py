#!/usr/bin/env python3
"""Add a visual focus figure for the alveoli target-grounding failure.

The table in 17AO is useful, but this figure makes the key failure visible:
candidate 48 has alveoli-like H&E morphology, yet it sits away from the target
annotation region used by the current benchmark.
"""

from __future__ import annotations

import ast
import re
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3")
BASE = ROOT / "output/visium_hd_exp1/final_deliverables/Jun09_SkillRanker_ComponentAwareMaskSelection"
OUT = BASE / "alveoli_target_grounding_audit"
HE_FULL = ROOT / "output/visium_hd_exp1/assets/tissue_hires_image.png"
ROI_BBOX = (75, 40, 3219, 3367)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    vals = ast.literal_eval(value)
    return tuple(int(v) for v in vals)  # type: ignore[return-value]


def draw_labeled_box(img: Image.Image, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    for off in range(4):
        draw.rectangle([x0 - off, y0 - off, x1 + off, y1 + off], outline=color)
    draw.rectangle([x0, max(0, y0 - 28), min(img.width, x0 + 320), y0], fill=(255, 255, 255))
    draw.text((x0 + 4, max(0, y0 - 25)), label, fill=color, font=font(18))


def make_focus_figure() -> Path:
    df = pd.read_csv(OUT / "alveoli_target_grounding_focus_candidates.csv")
    focus_ids = [48, 13, 12, 11, 22, 24]
    rows = df[df["candidate"].isin(focus_ids)].copy()
    he = Image.open(HE_FULL).convert("RGB")
    roi = he.crop(ROI_BBOX)

    locator = roi.copy()
    locator.thumbnail((620, 660), Image.Resampling.LANCZOS)
    sx = locator.width / roi.width
    sy = locator.height / roi.height
    colors = {
        48: (220, 38, 38),
        13: (22, 163, 74),
        12: (22, 163, 74),
        11: (59, 130, 246),
        22: (245, 158, 11),
        24: (245, 158, 11),
    }
    labels = {
        48: "48 FP: alveoli-like",
        13: "13 true anchor",
        12: "12 true anchor",
        11: "11 recall partner",
        22: "22 FP/neighbor",
        24: "24 partial",
    }
    for _, row in rows.iterrows():
        cid = int(row["candidate"])
        x0, y0, x1, y1 = parse_bbox(row["bbox"])
        scaled = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
        draw_labeled_box(locator, scaled, labels.get(cid, str(cid)), colors.get(cid, (0, 0, 0)))

    crop_cards: list[Image.Image] = []
    for cid in focus_ids:
        row = rows[rows["candidate"] == cid].iloc[0]
        x0, y0, x1, y1 = parse_bbox(row["bbox"])
        pad = 36
        crop = roi.crop((max(0, x0 - pad), max(0, y0 - pad), min(roi.width, x1 + pad), min(roi.height, y1 + pad)))
        crop.thumbnail((270, 230), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (300, 305), "white")
        card.paste(crop, ((300 - crop.width) // 2, 44))
        draw = ImageDraw.Draw(card)
        color = colors.get(cid, (0, 0, 0))
        draw.text((12, 10), f"candidate {cid}", fill=color, font=font(18))
        draw.text((12, 250), f"D/P/R {row['hidden D/P/R']}", fill=(31, 41, 55), font=font(14))
        draw.text((12, 272), f"morph {float(row['H&E morphology score']):.3f}", fill=(31, 41, 55), font=font(14))
        crop_cards.append(card)

    w = 620 + 40 + 3 * 300 + 2 * 22
    h = 120 + max(locator.height, 2 * 305 + 22)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 22), "Alveoli target-grounding failure: morphology-like is not always target-region-correct", fill=(17, 24, 39), font=font(24))
    draw.text((24, 58), "Red candidate 48 looks alveoli-like but has hidden Dice 0; green candidates 12/13 are target-region anchors.", fill=(75, 85, 99), font=font(16))
    canvas.paste(locator, (24, 100))
    xbase = 24 + 620 + 40
    ybase = 100
    for i, card in enumerate(crop_cards):
        x = xbase + (i % 3) * (300 + 22)
        y = ybase + (i // 3) * (305 + 22)
        canvas.paste(card, (x, y))
    out = OUT / "figures" / "alveoli_target_grounding_focus_candidates.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def update_html(fig: Path) -> None:
    rel = fig.relative_to(BASE)
    figure_html = (
        "<h3>Visual focus example</h3>"
        f"<figure><img src=\"{rel}\" alt=\"Alveoli target-grounding focus candidates\" />"
        "<figcaption>Candidate 48 is a morphology-like false positive; 12/13 are target-region anchors. This is why the next skill iteration needs target grounding, not only stronger alveoli texture scoring.</figcaption></figure>"
    )
    for html_path in [BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html", BASE / "index.html"]:
        text = html_path.read_text()
        anchor = "<h3>Failure subtypes</h3>"
        if "alveoli_target_grounding_focus_candidates.png" not in text:
            text = text.replace(anchor, figure_html + anchor)
            html_path.write_text(text)


def verify_and_zip() -> None:
    html_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html"
    text = html_path.read_text(errors="ignore")
    missing = []
    for src in re.findall(r"<img[^>]+src=\"([^\"]+)\"", text):
        if src.startswith(("data:", "http://", "https://")):
            continue
        if not (html_path.parent / src).exists():
            missing.append(src)
    if missing:
        raise RuntimeError(f"missing image assets: {missing[:5]}")
    zip_path = BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection_shareable.zip"
    # Reuse the broader packer so the new image is included.
    import importlib.util
    script = ROOT / "scripts/add_jun09_precise_failure_framework_v5.py"
    spec = importlib.util.spec_from_file_location("pack", script)
    pack = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pack)
    pack.rebuild_zip()
    with zipfile.ZipFile(zip_path) as handle:
        bad = handle.testzip()
        if bad:
            raise RuntimeError(f"bad zip entry: {bad}")


def main() -> None:
    fig = make_focus_figure()
    update_html(fig)
    verify_and_zip()
    print(fig)
    print(BASE / "Jun09_SkillRanker_ComponentAwareMaskSelection.html")


if __name__ == "__main__":
    main()
