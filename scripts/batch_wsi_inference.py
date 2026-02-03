#!/usr/bin/env python3
"""
Batch WSI inference script.
Processes the dataset on a single GPU.

Usage:
    python batch_wsi_inference.py --gpu_id 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Parse gpu_id from command line BEFORE importing torch
# This ensures CUDA_VISIBLE_DEVICES is set before PyTorch initializes CUDA
if "--gpu_id" in sys.argv:
    gpu_idx = sys.argv.index("--gpu_id")
    if gpu_idx + 1 < len(sys.argv):
        gpu_id = sys.argv[gpu_idx + 1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.models.retrieval_model import PatchRetrievalModel

CATEGORIES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]

CHECKPOINT_PATHS = {
    "ASC-US": "checkpoints/ASC-US/20251226_223843_bestAUC/best.pt",
    "LSIL": "checkpoints/LSIL/20251226_103805_bestAUC/best.pt",
    "ASC-H": "checkpoints/ASC-H/20251227_092502_bestAUC/best.pt",
    "HSIL": "checkpoints/HSIL/20251224_210120_bestAUC/best.pt",
    "SCC": "checkpoints/SCC/20251229_102629_bestAUC/best.pt",
}

QUERIES_DIR = "queries"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch WSI inference")
    
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/ssddata/hjiangaz/rliuar/1_Research/RetrieverVerifierAgent/1500wsi_dataset.csv",
        help="Path to CSV file with WSI data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssddata/hjiangaz/rliuar/1_Research/retrieval_result",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top results per retriever",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Aggregation method for multi-query",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    
    return parser.parse_args()


class PatchFolderDataset(Dataset):
    """Dataset for loading patches from a folder."""
    
    def __init__(self, patch_dir: str, image_size: int = 224):
        self.patch_dir = Path(patch_dir)
        self.image_size = image_size
        
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.patch_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.patch_dir.glob(f"*{ext.upper()}"))
        
        self.image_paths = sorted(self.image_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return {
            "image": self.transform(img),
            "filename": path.name,
            "path": str(path),
        }


def load_query_images(category, base_dir, image_size=224):
    """Load all query images for a category."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    
    query_dir = base_dir / QUERIES_DIR / category
    extensions = {".png", ".jpg", ".jpeg"}
    query_paths = []
    for ext in extensions:
        query_paths.extend(query_dir.glob(f"*{ext}"))
    query_paths = sorted(query_paths)
    
    images = []
    for path in query_paths:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    
    return torch.stack(images)


def load_config(checkpoint_path):
    """Load config from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    config_file = checkpoint_dir / "config.yaml"
    
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    
    return {"model": {"backbone": "openai/clip-vit-base-patch16", "cache_dir": None}}


@torch.no_grad()
def run_inference(model, query_images, dataloader, device, aggregation="max"):
    """Run inference for one retriever."""
    model.eval()
    
    all_similarities = []
    all_filenames = []
    all_paths = []
    
    query_embeddings = model.encoder.encode_query(query_images.to(device))
    
    for batch in dataloader:
        images = batch["image"].to(device)
        filenames = batch["filename"]
        paths = batch["path"]
        
        patch_embeddings = model.encoder.encode_patches(images)
        
        batch_similarities = []
        for q_idx in range(query_embeddings.shape[0]):
            q_emb = query_embeddings[q_idx:q_idx+1].expand(images.shape[0], -1)
            sims = model.compute_similarity(q_emb, patch_embeddings)
            batch_similarities.append(sims)
        
        batch_similarities = torch.stack(batch_similarities, dim=0)
        
        if aggregation == "max":
            final_similarities = batch_similarities.max(dim=0).values
        else:
            final_similarities = batch_similarities.mean(dim=0)
        
        all_similarities.append(final_similarities.cpu().float().numpy())
        all_filenames.extend(filenames)
        all_paths.extend(paths)
    
    similarities = np.concatenate(all_similarities)
    return similarities, all_filenames, all_paths


def get_topk_results(similarities, filenames, paths, topk):
    """Get top-k results with paths."""
    sorted_indices = np.argsort(-similarities)[:topk]
    
    results = []
    for rank, idx in enumerate(sorted_indices):
        results.append({
            "rank": rank + 1,
            "filename": filenames[idx],
            "path": paths[idx],
            "similarity": float(similarities[idx]),
        })
    
    return results


def create_topk_grid(topk_results, output_path, category, cols=5):
    """Create a grid visualization of top-k patches for a category."""
    n = len(topk_results)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(topk_results):
        row, col = i // cols, i % cols
        try:
            img = Image.open(result["path"])
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"#{result['rank']}: {result['similarity']:.3f}", fontsize=8)
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f"Error", ha='center', va='center')
        axes[row, col].axis('off')
    
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Top {n} Patches - {category}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_single_wsi(wsi_info, args, base_dir, device, models_cache):
    """Process a single WSI."""
    wsi_name = wsi_info["file_name"]
    patch_dir = wsi_info["patch_path"]
    gt_category = wsi_info["category"]
    
    output_dir = Path(args.output_dir) / f"{wsi_name}-gt{gt_category}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    results_file = output_dir / "results.json"
    if results_file.exists() and not getattr(args, 'overwrite', False):
        print(f"  [{wsi_name}] Already processed, skipping...")
        return wsi_name, "skipped"
    
    # Check patch directory exists
    if not Path(patch_dir).exists():
        print(f"  [{wsi_name}] ERROR: Patch directory not found: {patch_dir}")
        return wsi_name, "no_patches"
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing WSI: {wsi_name}")
        print(f"GT Category: {gt_category}")
        print(f"Patch Directory: {patch_dir}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = PatchFolderDataset(patch_dir)
        if len(dataset) == 0:
            print(f"  [{wsi_name}] ERROR: No patches found in directory")
            return wsi_name, "empty"
        
        print(f"Found {len(dataset)} patches")
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        all_results = {}
        all_statistics = {}
        
        # Run each retriever
        print(f"\nRunning inference with {len(CATEGORIES)} retrievers...")
        for idx, category in enumerate(CATEGORIES, 1):
            print(f"\n  [{idx}/{len(CATEGORIES)}] Processing category: {category}")
            
            model = models_cache[category]
            query_images = models_cache[f"{category}_queries"]
            print(f"    Using {len(query_images)} query image(s)")
            
            print(f"    Running inference...", end=" ", flush=True)
            similarities, filenames, paths = run_inference(
                model, query_images, dataloader, device, args.aggregation
            )
            print(f"Done!")
            
            topk_results = get_topk_results(similarities, filenames, paths, args.topk)
            all_results[category] = topk_results
            
            all_statistics[category] = {
                "total_patches": len(similarities),
                "mean_similarity": float(similarities.mean()),
                "std_similarity": float(similarities.std()),
                "min_similarity": float(similarities.min()),
                "max_similarity": float(similarities.max()),
                "top1_similarity": float(topk_results[0]["similarity"]) if topk_results else 0,
            }
            
            # Print statistics
            stats = all_statistics[category]
            print(f"    Statistics:")
            print(f"      Total patches: {stats['total_patches']}")
            print(f"      Mean similarity: {stats['mean_similarity']:.4f}")
            print(f"      Max similarity: {stats['max_similarity']:.4f}")
            print(f"      Top-1 similarity: {stats['top1_similarity']:.4f}")
            
            # Print top 5 results
            print(f"    Top 5 patches:")
            for r in topk_results[:5]:
                print(f"      #{r['rank']}: {r['filename']} (sim={r['similarity']:.4f})")
            
            # Create topk grid for this category
            print(f"    Creating top-k grid...", end=" ", flush=True)
            grid_path = output_dir / f"topk_grid_{category}.png"
            create_topk_grid(topk_results, grid_path, category)
            print(f"Saved to {grid_path.name}")
            
            # Save topk filenames for this category
            topk_filenames_path = output_dir / f"topk_filenames_{category}.txt"
            with open(topk_filenames_path, "w") as f:
                for r in topk_results:
                    f.write(f"{r['filename']}\t{r['similarity']:.6f}\n")
            print(f"    Saved top-k filenames to {topk_filenames_path.name}")
        
        # Save combined results
        print(f"\n  Saving results...", end=" ", flush=True)
        output_data = {
            "wsi_name": wsi_name,
            "gt_category": gt_category,
            "patch_dir": patch_dir,
            "config": {
                "topk": args.topk,
                "aggregation": args.aggregation,
            },
            "statistics": all_statistics,
            "results": all_results,
        }
        
        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {results_file.name}")
        
        # Save combined topk filenames
        combined_path = output_dir / "topk_filenames_all.txt"
        with open(combined_path, "w") as f:
            for category in CATEGORIES:
                f.write(f"\n=== {category} ===\n")
                for r in all_results[category]:
                    f.write(f"{r['rank']}\t{r['filename']}\t{r['similarity']:.6f}\n")
        
        print(f"\n  ✓ Completed {wsi_name} successfully!")
        print(f"  Output directory: {output_dir}")
        
        return wsi_name, "success"
        
    except Exception as e:
        print(f"\n  ✗ ERROR processing {wsi_name}: {e}")
        import traceback
        traceback.print_exc()
        return wsi_name, f"error: {e}"


def main():
    args = parse_args()
    
    # CUDA_VISIBLE_DEVICES was already set before torch import
    # After setting CUDA_VISIBLE_DEVICES, the specified GPU becomes device 0
    if torch.cuda.is_available():
        # Verify we're using the correct GPU
        actual_gpu_id = torch.cuda.current_device()
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {actual_gpu_id}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = Path(__file__).parent
    
    print("=" * 60)
    print(f"Batch WSI Inference")
    print("=" * 60)
    print(f"GPU: {args.gpu_id}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Load CSV
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} WSIs to process")
    print()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Pre-load all models and queries (do this once)
    print("Loading models...")
    models_cache = {}
    for category in tqdm(CATEGORIES, desc="Loading models"):
        checkpoint_path = base_dir / CHECKPOINT_PATHS[category]
        config = load_config(checkpoint_path)
        
        pooling_method = config["model"].get("pooling_method", "softmax_attn")
        model = PatchRetrievalModel(
            model_name=config["model"]["backbone"],
            pooling_method=pooling_method,
            cache_dir=config["model"].get("cache_dir"),
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        models_cache[category] = model
        models_cache[f"{category}_queries"] = load_query_images(category, base_dir)
    
    print(f"Models loaded!\n")
    
    # Process each WSI
    success = 0
    skipped = 0
    failed = 0
    
    # Process with progress bar
    pbar = tqdm(total=len(df), 
                desc="WSIs", 
                ncols=100,
                leave=True)
    
    for idx, row in df.iterrows():
        wsi_info = row.to_dict()
        wsi_name = wsi_info["file_name"]
        
        # Update progress bar description
        pbar.set_description(f"Processing {wsi_name[:20]}")
        
        wsi_name, status = process_single_wsi(wsi_info, args, base_dir, device, models_cache)
        
        if status == "success":
            success += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
        
        # Update progress bar
        pbar.set_postfix({
            "✓": success, 
            "⊘": skipped, 
            "✗": failed
        })
        pbar.update(1)
    
    print()
    print("=" * 60)
    print(f"Inference Complete!")
    print("=" * 60)
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print(f"Total:   {len(df)}")


if __name__ == "__main__":
    main()
