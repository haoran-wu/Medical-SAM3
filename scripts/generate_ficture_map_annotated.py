#!/usr/bin/env python3
"""
Generate annotated FICTURE map with gene information for each factor.
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path

# Factor gene information from hex_12.k12.pixel.info.tsv
FACTOR_INFO = {
    0: {
        'name': 'AT0',
        'top_genes': ['CXCL14', 'SPINK1', 'CEACAM5', 'CEACAM6', 'NAPSA'],
        'color': (255, 204, 255),
    },
    1: {
        'name': 'Fibroblast-like',
        'top_genes': ['TAGLN', 'MYL9', 'MYH11', 'A2M', 'ACTA2'],
        'color': (0, 255, 255),
    },
    2: {
        'name': 'Type II Pneumocyte',
        'top_genes': ['WSB1', 'CEACAM5', 'SLC22A31', 'RAB11FIP1', 'LAMA5'],
        'color': (255, 255, 0),
    },
    3: {
        'name': 'Epithelial (SFTPB+)',
        'top_genes': ['SFTPB', 'LPCAT1', 'NPC2', 'NUPR1', 'CD9'],
        'color': (255, 84, 0),
    },
    4: {
        'name': 'Lymphocyte',
        'top_genes': ['SFTPC', 'CXCR4', 'CCL19', 'IL7R', 'TRAC'],
        'color': (0, 255, 84),
    },
    5: {
        'name': 'Type I Pneumocyte',
        'top_genes': ['LPCAT1', 'PGGHG', 'ATP13A4', 'BTBD9', 'ABCC3'],
        'color': (84, 0, 255),
    },
    6: {
        'name': 'Macrophage',
        'top_genes': ['SPP1', 'FTL', 'APOE', 'CTSB', 'CHIT1'],
        'color': (170, 0, 170),
    },
    7: {
        'name': 'Secretory Cell',
        'top_genes': ['SCGB1A1', 'SCGB3A1', 'BPIFB1', 'MSMB', 'SLPI'],
        'color': (0, 170, 170),
    },
    8: {
        'name': 'Plasma Cell',
        'top_genes': ['IGKC', 'IGHG1', 'IGLC1', 'IGHM', 'IGHA1'],
        'color': (170, 170, 0),
    },
    9: {
        'name': 'Neuroendocrine',
        'top_genes': ['GRP', 'CDK5R1', 'NRXN1', 'LRRD1', 'TPH1'],
        'color': (255, 0, 127),
    },
    10: {
        'name': 'Plasma Cell (IgA+)',
        'top_genes': ['IGHA1', 'IGKC', 'JCHAIN', 'IGHG3', 'IGLC1'],
        'color': (153, 76, 0),
    },
    11: {
        'name': 'Plasma Cell (IgM+)',
        'top_genes': ['IGHM', 'IGKC', 'JCHAIN', 'IGHA1', 'IGLC1'],
        'color': (0, 115, 0),
    },
}


def create_factor_legend_detailed(output_path, img_width=800, img_height=1200):
    """Create detailed legend with gene information."""
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_genes = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font_title = ImageFont.load_default()
        font_genes = ImageFont.load_default()

    y_pos = 20
    for factor, info in sorted(FACTOR_INFO.items()):
        # Color box
        color = info['color']
        draw.rectangle([20, y_pos, 40, y_pos + 20], fill=color)

        # Factor name
        draw.text((50, y_pos - 2), f"Factor {factor}: {info['name']}", fill=(0, 0, 0), font=font_title)
        y_pos += 25

        # Top genes
        genes_str = ", ".join(info['top_genes'][:3])
        draw.text((50, y_pos), f"Top genes: {genes_str}", fill=(80, 80, 80), font=font_genes)
        y_pos += 20

        y_pos += 10

    img.save(output_path)
    print(f"Saved detailed legend: {output_path}")


def create_statistics_image(df, output_path):
    """Create image showing factor distribution statistics."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    factor_cols = [str(i) for i in range(12)]
    factors = df[factor_cols].values
    dominant_factors = np.argmax(factors, axis=1)

    # Count occurrences
    unique, counts = np.unique(dominant_factors, return_counts=True)
    factor_counts = dict(zip(unique, counts))

    # Create matplotlib figure
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)

    factors_list = list(range(12))
    counts_list = [factor_counts.get(i, 0) for i in factors_list]
    colors = [FACTOR_INFO[i]['color'] for i in factors_list]
    colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors]

    ax.bar(factors_list, counts_list, color=colors_normalized)
    ax.set_xlabel('Factor', fontsize=12)
    ax.set_ylabel('Number of spots', fontsize=12)
    ax.set_title('FICTURE Factor Distribution', fontsize=14)
    ax.set_xticks(factors_list)

    # Save figure
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    size = canvas.get_width_height()
    img = Image.frombytes("RGB", size, raw_data, "raw", "RGB", 0, 1)
    img.save(output_path)
    print(f"Saved distribution statistics: {output_path}")


def main():
    output_dir = Path('/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/ficture_factor_map_new')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FICTURE results for statistics
    ficture_tsv = '/Users/haoranwu/Desktop/Yan_Lab_Research/Medical-SAM3/output/visium_hd_exp1/ficture_hw568_filtered_source/hex_12.k12.results.tsv'
    df = pd.read_csv(ficture_tsv, sep='\t')

    # Create detailed legend
    create_factor_legend_detailed(output_dir / 'ficture_factor_legend_detailed.png')

    # Create statistics
    try:
        create_statistics_image(df, output_dir / 'ficture_factor_distribution.png')
    except ImportError:
        print("matplotlib not available for statistics visualization")

    print(f"✓ Annotation complete!")


if __name__ == '__main__':
    main()
