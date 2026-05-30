#!/usr/bin/env python3
"""
Deprecated debug renderer for FICTURE results.tsv spot maps.

Do not use this script for the official VisiumHD Exp1 FICTURE candidate-pool
map. The official filtered, H&E-aligned pipeline is:

    inference/visium_hd_exp1/render_official_filtered_ficture_he_map.py

That official pipeline uses the filtered FICTURE PNG from
``data/visium_hd_exp1/pixel_cell_type_image/visiumhd_exp1_hex12_k12/hex_12.k12.pixel.png``,
applies ``np.fliplr``, and maps with
``he_x = y_um / microns_per_pixel * tissue_hires_scalef`` and
``he_y = x_um / microns_per_pixel * tissue_hires_scalef``.
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import argparse
from pathlib import Path

# Color palette for factors (from hex_12.k12.pixel.info.tsv)
FACTOR_COLORS = {
    0: (255, 204, 255),    # pink
    1: (0, 255, 255),      # cyan
    2: (255, 255, 0),      # yellow
    3: (255, 84, 0),       # orange
    4: (0, 255, 84),       # lime
    5: (84, 0, 255),       # blue
    6: (170, 0, 170),      # purple
    7: (0, 170, 170),      # teal
    8: (170, 170, 0),      # olive
    9: (255, 0, 127),      # magenta
    10: (153, 76, 0),      # brown
    11: (0, 115, 0),       # dark green
}

def generate_ficture_map(
    ficture_tsv_path,
    output_dir,
    metadata,
    target_size=(3144, 3327),
    spot_radius_pixels=4,
    sample_every_n=1,
):
    """
    Generate FICTURE factor map aligned with H&E.

    Args:
        ficture_tsv_path: Path to FICTURE results TSV
        output_dir: Output directory for maps
        metadata: Dict with microns_per_pixel and tissue_hires_scalef
        target_size: (width, height) of output image
        spot_radius_pixels: Radius in pixels for rendering each spot
        sample_every_n: Sample every nth spot (for preview)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FICTURE results
    print(f"Loading FICTURE results from {ficture_tsv_path}")
    df = pd.read_csv(ficture_tsv_path, sep='\t')

    # Extract coordinates and factors
    x_coords = df['x'].values
    y_coords = df['y'].values

    # Apply coordinate transformation
    microns_per_pixel = metadata['microns_per_pixel']
    tissue_hires_scalef = metadata['tissue_hires_scalef']

    x_transformed = (x_coords / microns_per_pixel) * tissue_hires_scalef
    y_transformed = (y_coords / microns_per_pixel) * tissue_hires_scalef

    print(f"Original coordinate range: X [{x_coords.min():.1f}, {x_coords.max():.1f}], "
          f"Y [{y_coords.min():.1f}, {y_coords.max():.1f}]")
    print(f"Transformed coordinate range: X [{x_transformed.min():.1f}, {x_transformed.max():.1f}], "
          f"Y [{y_transformed.min():.1f}, {y_transformed.max():.1f}]")
    print(f"Target canvas size: {target_size}")

    # Get dominant factor for each spot
    factor_cols = [str(i) for i in range(12)]
    factors = df[factor_cols].values
    dominant_factors = np.argmax(factors, axis=1)

    # Create full resolution map
    print("\nGenerating full resolution FICTURE map...")
    img_full = Image.new('RGB', target_size, color=(255, 255, 255))
    draw_full = ImageDraw.Draw(img_full, 'RGBA')

    for i in range(0, len(df), sample_every_n):
        x = int(round(x_transformed[i]))
        y = int(round(y_transformed[i]))
        factor = dominant_factors[i]
        color = FACTOR_COLORS[factor]

        if 0 <= x < target_size[0] and 0 <= y < target_size[1]:
            # Draw circle
            bbox = [x - spot_radius_pixels, y - spot_radius_pixels,
                   x + spot_radius_pixels, y + spot_radius_pixels]
            draw_full.ellipse(bbox, fill=color)

    full_path = output_dir / 'ficture_factor_map_full.png'
    img_full.save(full_path)
    print(f"Saved full map: {full_path}")

    # Create preview (lower resolution for quick viewing)
    print("\nGenerating preview FICTURE map (every 5th spot)...")
    img_preview = Image.new('RGB', target_size, color=(255, 255, 255))
    draw_preview = ImageDraw.Draw(img_preview, 'RGBA')

    for i in range(0, len(df), max(5, sample_every_n)):
        x = int(round(x_transformed[i]))
        y = int(round(y_transformed[i]))
        factor = dominant_factors[i]
        color = FACTOR_COLORS[factor]

        if 0 <= x < target_size[0] and 0 <= y < target_size[1]:
            bbox = [x - spot_radius_pixels, y - spot_radius_pixels,
                   x + spot_radius_pixels, y + spot_radius_pixels]
            draw_preview.ellipse(bbox, fill=color)

    preview_path = output_dir / 'ficture_factor_map_preview.png'
    img_preview.save(preview_path)
    print(f"Saved preview map: {preview_path}")

    # Create legend
    print("\nGenerating legend...")
    factor_names = [
        "Factor 0", "Factor 1", "Factor 2", "Factor 3",
        "Factor 4", "Factor 5", "Factor 6", "Factor 7",
        "Factor 8", "Factor 9", "Factor 10", "Factor 11"
    ]

    legend_img = Image.new('RGB', (400, 500), color=(255, 255, 255))
    draw_legend = ImageDraw.Draw(legend_img)

    y_pos = 20
    for factor, color in FACTOR_COLORS.items():
        draw_legend.rectangle([20, y_pos, 40, y_pos + 20], fill=color)
        draw_legend.text((50, y_pos + 2), factor_names[factor], fill=(0, 0, 0))
        y_pos += 35

    legend_path = output_dir / 'ficture_factor_legend.png'
    legend_img.save(legend_path)
    print(f"Saved legend: {legend_path}")

    # Create overlay on H&E if available
    he_path = Path("/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/"
                   "agent_verified_filtered_ficture_he_align/strict_results_coordinates_no_shift/"
                   "filtered_ficture_results_coords_aligned_full_he_canvas.png")

    if he_path.exists():
        print(f"\nCreating overlay on H&E from {he_path}...")
        he_img = Image.open(he_path).convert('RGB')

        # Resize FICTURE map to match H&E if needed
        if he_img.size != target_size:
            print(f"Resizing FICTURE map from {target_size} to match H&E size {he_img.size}")
            img_full = img_full.resize(he_img.size, Image.Resampling.LANCZOS)

        # Create overlay
        overlay = Image.new('RGBA', he_img.size)
        overlay_draw = ImageDraw.Draw(overlay, 'RGBA')

        # Draw semi-transparent spots on overlay
        for i in range(0, len(df), sample_every_n):
            x = int(round(x_transformed[i]))
            y = int(round(y_transformed[i]))

            # Resize coordinates if canvas was resized
            if he_img.size != target_size:
                x = int(x * he_img.size[0] / target_size[0])
                y = int(y * he_img.size[1] / target_size[1])

            factor = dominant_factors[i]
            color = FACTOR_COLORS[factor]
            color_rgba = color + (180,)  # Add alpha channel

            if 0 <= x < he_img.size[0] and 0 <= y < he_img.size[1]:
                bbox = [x - spot_radius_pixels, y - spot_radius_pixels,
                       x + spot_radius_pixels, y + spot_radius_pixels]
                overlay_draw.ellipse(bbox, fill=color_rgba)

        # Composite overlay on H&E
        result = Image.new('RGB', he_img.size)
        result.paste(he_img)
        result.paste(overlay, (0, 0), overlay)

        overlay_path = output_dir / 'ficture_factor_overlay_on_he.png'
        result.save(overlay_path)
        print(f"Saved overlay: {overlay_path}")
    else:
        print(f"H&E reference not found at {he_path}")

    print(f"\n✓ Complete! Maps saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate FICTURE map aligned with H&E')
    parser.add_argument('--ficture-tsv',
                       default='/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/ficture_hw568_filtered_source/hex_12.k12.results.tsv',
                       help='Path to FICTURE results TSV')
    parser.add_argument('--output-dir',
                       default='/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/ficture_factor_map_new',
                       help='Output directory')
    parser.add_argument('--spot-radius', type=int, default=4,
                       help='Spot rendering radius in pixels')

    args = parser.parse_args()

    metadata = {
        'microns_per_pixel': 0.2737554241192739,
        'tissue_hires_scalef': 0.13752006,
    }

    generate_ficture_map(
        args.ficture_tsv,
        args.output_dir,
        metadata,
        spot_radius_pixels=args.spot_radius,
    )
