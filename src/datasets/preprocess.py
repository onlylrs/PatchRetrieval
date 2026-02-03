"""
Preprocess CCS dataset for patch retrieval training.

This script:
1. Parses train/val/test split files from CCS-Cell-Cls
2. Filters to only target categories (ASC-US, LSIL, ASC-H, HSIL, SCC)
3. Maps cells to patches and filters patches that exist in PATCH_DATA
4. Re-splits data to 7:1:2 ratio
5. Generates positive/negative sample lists for each category
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# Target categories we care about (in severity order)
TARGET_CATEGORIES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]

# Severity mapping (higher = more severe)
SEVERITY = {
    "ASC-US": 1,
    "LSIL": 2,
    "ASC-H": 3,
    "HSIL": 4,
    "SCC": 5,
}


@dataclass
class CellInfo:
    """Information about a single cell."""
    cell_path: str       # Full path to cell image
    cell_filename: str   # Just the filename
    patch_name: str      # Patch name without extension
    cell_index: int      # Cell index within patch
    category: str        # Cell category


@dataclass
class PatchInfo:
    """Information about a patch and its cells."""
    patch_name: str
    patch_path: str
    cells: list[CellInfo]
    categories_present: set[str]


def extract_patch_name(cell_filename: str) -> tuple[str, int]:
    """
    Extract patch name and cell index from cell filename.
    
    Cell filename format: {patch_name}_{cell_index}.png
    Example: T-194730_30631_-0645_2.png -> (T-194730_30631_-0645, 2)
    
    Returns:
        (patch_name, cell_index)
    """
    # Remove extension
    name = cell_filename.rsplit('.', 1)[0]
    
    # Find the last underscore followed by a number
    match = re.match(r'^(.+)_(\d+)$', name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        # Fallback: treat entire name as patch name
        return name, 0


def parse_split_file(file_path: Path, cell_data_root: Path) -> list[CellInfo]:
    """
    Parse a train/val/test split file.
    
    Format: ./CELL_DATA/{category}/{filename} {category}
    """
    cells = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            
            rel_path, category = parts
            
            # Skip non-target categories
            if category not in TARGET_CATEGORIES:
                continue
            
            # Parse path
            # ./CELL_DATA/ASC-US/T-194730_30631_-0645_2.png
            filename = os.path.basename(rel_path)
            patch_name, cell_index = extract_patch_name(filename)
            
            # Build full path
            full_path = str(cell_data_root / rel_path.lstrip('./'))
            
            cells.append(CellInfo(
                cell_path=full_path,
                cell_filename=filename,
                patch_name=patch_name,
                cell_index=cell_index,
                category=category,
            ))
    
    return cells


def build_patch_mapping(
    cells: list[CellInfo],
    patch_data_dir: Path,
) -> dict[str, PatchInfo]:
    """
    Build mapping from patch names to patch info.
    Only includes patches that exist in patch_data_dir.
    """
    # Group cells by patch
    print("  Grouping cells by patch...")
    patch_to_cells = defaultdict(list)
    for cell in tqdm(cells, desc="  Grouping cells"):
        patch_to_cells[cell.patch_name].append(cell)
    
    print(f"  Unique patches from cells: {len(patch_to_cells)}")
    
    # Filter to patches that exist
    print("  Checking patch files exist...")
    patches = {}
    for patch_name, cells_list in tqdm(patch_to_cells.items(), desc="  Filtering patches"):
        patch_path = patch_data_dir / f"{patch_name}.png"
        if patch_path.exists():
            categories = set(c.category for c in cells_list)
            patches[patch_name] = PatchInfo(
                patch_name=patch_name,
                patch_path=str(patch_path),
                cells=cells_list,
                categories_present=categories,
            )
    
    return patches


def split_patches(
    patches: dict[str, PatchInfo],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split patches into train/val/test sets.
    
    Returns dict mapping split name to list of patch names.
    """
    np.random.seed(seed)
    
    patch_names = list(patches.keys())
    np.random.shuffle(patch_names)
    
    n = len(patch_names)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        "train": patch_names[:train_end],
        "val": patch_names[train_end:val_end],
        "test": patch_names[val_end:],
    }


def generate_category_samples(
    patch_names: list[str],
    patches: dict[str, PatchInfo],
    category: str,
) -> tuple[list[str], list[str]]:
    """
    Generate positive and negative sample lists for a category.
    
    Positive: patch contains at least one cell of this category
    Negative: patch does NOT contain any cell of this category
    """
    positives = []
    negatives = []
    
    for name in patch_names:
        patch = patches[name]
        if category in patch.categories_present:
            positives.append(name)
        else:
            negatives.append(name)
    
    return positives, negatives


class CCSPreprocessor:
    """Preprocessor for CCS dataset."""
    
    def __init__(
        self,
        cell_cls_root: str,
        cell_det_root: str,
    ):
        """
        Initialize preprocessor.
        
        Args:
            cell_cls_root: Path to CCS-Cell-Cls directory
            cell_det_root: Path to CCS-Cell-Det directory
        """
        self.cell_cls_root = Path(cell_cls_root)
        self.cell_det_root = Path(cell_det_root)
        self.cell_data_dir = self.cell_cls_root / "CELL_DATA"
        self.patch_data_dir = self.cell_det_root / "PATCH_DATA"
    
    def process_and_save(
        self,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        """
        Process dataset and save to output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Parse all split files
        print("Parsing split files...")
        all_cells = []
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            file_path = self.cell_cls_root / split_file
            if file_path.exists():
                cells = parse_split_file(file_path, self.cell_cls_root)
                all_cells.extend(cells)
                print(f"  {split_file}: {len(cells)} target cells")
        
        print(f"Total target cells: {len(all_cells)}")
        
        # Build patch mapping
        print("\nBuilding patch mapping...")
        patches = build_patch_mapping(all_cells, self.patch_data_dir)
        print(f"Patches with existing images: {len(patches)}")
        
        # Count cells in valid patches
        valid_cells = sum(len(p.cells) for p in patches.values())
        print(f"Cells in valid patches: {valid_cells}")
        
        # Split patches
        print("\nSplitting patches (7:1:2)...")
        splits = split_patches(patches, train_ratio, val_ratio, test_ratio, seed)
        
        for split_name, patch_names in splits.items():
            print(f"  {split_name}: {len(patch_names)} patches")
        
        # Save splits
        with open(output_path / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)
        
        # Save patch info
        patches_data = []
        for patch_name, patch_info in patches.items():
            cells_data = [
                {
                    "cell_path": c.cell_path,
                    "cell_filename": c.cell_filename,
                    "category": c.category,
                    "cell_index": c.cell_index,
                }
                for c in patch_info.cells
            ]
            patches_data.append({
                "patch_name": patch_name,
                "patch_path": patch_info.patch_path,
                "cells": cells_data,
                "categories_present": list(patch_info.categories_present),
            })
        
        with open(output_path / "patches.json", "w") as f:
            json.dump(patches_data, f, indent=2)
        
        # Generate category-specific data
        print("\nGenerating category-specific samples...")
        stats = {"categories": {}}
        
        for category in TARGET_CATEGORIES:
            cat_dir = output_path / category
            cat_dir.mkdir(exist_ok=True)
            
            print(f"\n{category}:")
            cat_stats = {}
            
            for split_name, patch_names in splits.items():
                positives, negatives = generate_category_samples(
                    patch_names, patches, category
                )
                
                with open(cat_dir / f"{split_name}_positives.json", "w") as f:
                    json.dump(positives, f, indent=2)
                
                with open(cat_dir / f"{split_name}_negatives.json", "w") as f:
                    json.dump(negatives, f, indent=2)
                
                print(f"  {split_name}: {len(positives)} pos, {len(negatives)} neg")
                cat_stats[split_name] = {
                    "positives": len(positives),
                    "negatives": len(negatives),
                }
            
            stats["categories"][category] = cat_stats
        
        # Save overall stats
        stats["total_patches"] = len(patches)
        stats["total_cells"] = valid_cells
        stats["splits"] = {name: len(names) for name, names in splits.items()}
        
        with open(output_path / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Dataset saved to {output_path}")
        return stats


def main():
    """Main entry point for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CCS dataset")
    parser.add_argument(
        "--cell_cls_root",
        type=str,
        default="/homes/rliuar/fyp/COMP4901B/mnt/nas6/Public/CCS-Cell-Cls",
        help="Path to CCS-Cell-Cls directory",
    )
    parser.add_argument(
        "--cell_det_root",
        type=str,
        default="/homes/rliuar/fyp/COMP4901B/mnt/nas6/Public/CCS-Cell-Det",
        help="Path to CCS-Cell-Det directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    preprocessor = CCSPreprocessor(
        cell_cls_root=args.cell_cls_root,
        cell_det_root=args.cell_det_root,
    )
    
    preprocessor.process_and_save(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
