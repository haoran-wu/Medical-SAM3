#!/usr/bin/env python3
"""Prepare a clean run root from validated inputs, GeneMaps, and prompts only."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


TMAS = ("TMA07", "TMA24", "TMA29", "TMA30", "TMA31", "TMA34", "TMA36", "TMA39", "TMA41", "TMA42")
REUSED_TMA_DIRS = ("genemap_scores", "prompts")
REUSED_STAGING_FILES = ("{tma}_he.png", "{tma}_ficture.png", "factor_info.csv")
REUSED_SHARED_FILES = ("input_audit.json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Existing target differs from source: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target, copy_function=lambda src, dst: link_or_copy(Path(src), Path(dst)))


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for name in REUSED_SHARED_FILES:
        source = source_root / "shared" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, target_root / "shared" / name)

    for tma in TMAS:
        for directory in REUSED_TMA_DIRS:
            source = source_root / tma / directory
            if not source.is_dir():
                raise FileNotFoundError(source)
            copy_tree(source, target_root / tma / directory)
        for pattern in REUSED_STAGING_FILES:
            name = pattern.format(tma=tma)
            source = source_root / tma / "staging" / name
            if not source.is_file():
                raise FileNotFoundError(source)
            link_or_copy(source, target_root / tma / "staging" / name)

    (target_root / "logs").mkdir(exist_ok=True)
    print(f"Prepared {len(TMAS)} clean TMA inputs in {target_root}")
    print("Reused only registered inputs, factor legend, GeneMap scores, and prompts.")


if __name__ == "__main__":
    main()
