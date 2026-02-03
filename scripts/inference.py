"""
Inference script for external patch retrieval.
Given a folder of patches, compute similarities and output results.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

from src.models.retrieval_model import PatchRetrievalModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on external patches")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        required=True,
        help="Directory containing patch images",
    )
    parser.add_argument(
        "--query_images",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to query cell image(s), or 'all' to load from queries_dir",
    )
    parser.add_argument(
        "--queries_dir",
        type=str,
        default=None,
        help="Directory containing query images (used with --query_images all)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category subdirectory in queries_dir",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to checkpoint dir's config.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "experiments/inference"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top results to output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for filtering",
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
        help="How to aggregate similarities across multiple queries",
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


def resolve_query_images(query_images, queries_dir, category):
    """Resolve query image paths."""
    if len(query_images) == 1 and query_images[0].lower() == "all":
        if queries_dir is None or category is None:
            raise ValueError("--queries_dir and --category required when using 'all'")
        
        category_dir = Path(queries_dir) / category
        extensions = {".png", ".jpg", ".jpeg"}
        paths = []
        for ext in extensions:
            paths.extend(category_dir.glob(f"*{ext}"))
        return sorted([str(p) for p in paths])
    return query_images


def load_query_images(query_paths, image_size=224):
    """Load and preprocess query images."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    
    images = []
    for path in query_paths:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    
    return torch.stack(images)


def load_config(checkpoint_path, config_path=None):
    """Load config from checkpoint directory or specified path."""
    if config_path is not None:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    checkpoint_dir = Path(checkpoint_path).parent
    config_file = checkpoint_dir / "config.yaml"
    
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    
    return {"model": {"backbone": "openai/clip-vit-base-patch16", "cache_dir": None}}


@torch.no_grad()
def inference(model, query_images, dataloader, device, aggregation="max"):
    """Run inference on patches."""
    model.eval()
    
    all_similarities = []
    all_filenames = []
    all_paths = []
    
    # Encode queries once
    query_embeddings = model.encoder.encode_query(query_images.to(device))
    
    for batch in tqdm(dataloader, desc="Processing patches"):
        images = batch["image"].to(device)
        filenames = batch["filename"]
        paths = batch["path"]
        
        # Encode patches
        patch_embeddings = model.encoder.encode_patches(images)
        
        # Compute similarities for each query
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


def create_topk_grid(topk_results, output_path, cols=5, img_size=256):
    """Create a grid visualization of top-k patches."""
    n = len(topk_results)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(topk_results):
        row, col = i // cols, i % cols
        img = Image.open(result["path"])
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"#{result['rank']}: {result['similarity']:.3f}", fontsize=8)
        axes[row, col].axis('off')
    
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Top {n} Retrieved Patches", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_distribution(similarities, output_path, threshold=0.5):
    """Plot similarity distribution."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Similarity Distribution', fontsize=14)
    plt.legend()
    
    # Add statistics
    stats_text = f"Mean: {similarities.mean():.4f}\n"
    stats_text += f"Std: {similarities.std():.4f}\n"
    stats_text += f"Min: {similarities.min():.4f}\n"
    stats_text += f"Max: {similarities.max():.4f}\n"
    stats_text += f"Above {threshold}: {(similarities >= threshold).sum()}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.checkpoint, args.config)
    
    # Resolve paths relative to ROOT_DIR if not absolute
    if args.queries_dir and not Path(args.queries_dir).is_absolute():
        args.queries_dir = str(ROOT_DIR / args.queries_dir)

    # Resolve query images
    query_paths = resolve_query_images(
        args.query_images, args.queries_dir, args.category
    )
    print(f"Using {len(query_paths)} query image(s)")
    
    # Load query images
    query_images = load_query_images(query_paths)
    
    # Create dataset
    dataset = PatchFolderDataset(args.patch_dir)
    print(f"Found {len(dataset)} patches in {args.patch_dir}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Load model
    print("Loading model...")
    pooling_method = config["model"].get("pooling_method", "softmax_attn")
    model = PatchRetrievalModel(
        model_name=config["model"]["backbone"],
        pooling_method=pooling_method,
        cache_dir=config["model"].get("cache_dir"),
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Run inference
    print("\nRunning inference...")
    similarities, filenames, paths = inference(
        model, query_images, dataloader, device, args.aggregation
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot distribution
    plot_distribution(similarities, output_dir / "distribution.png", args.threshold)
    print(f"Saved distribution plot to {output_dir / 'distribution.png'}")
    
    # Get results
    sorted_indices = np.argsort(-similarities)
    
    # Top K results
    topk_results = []
    print(f"\nTop {args.topk} patches:")
    for i in range(min(args.topk, len(sorted_indices))):
        idx = sorted_indices[i]
        topk_results.append({
            "rank": i + 1,
            "filename": filenames[idx],
            "similarity": float(similarities[idx]),
            "path": paths[idx],
        })
        if i < 10:
            print(f"  {i+1}. {filenames[idx]}: {similarities[idx]:.4f}")
    
    if args.topk > 10:
        print(f"  ... (see output file for full list)")
    
    # Create top-k grid image
    create_topk_grid(topk_results, output_dir / "topk_grid.png", cols=5)
    print(f"Saved top-k grid to {output_dir / 'topk_grid.png'}")
    
    # Above threshold results
    above_threshold = similarities >= args.threshold
    above_threshold_results = []
    for idx in np.where(above_threshold)[0]:
        above_threshold_results.append({
            "filename": filenames[idx],
            "similarity": float(similarities[idx]),
            "path": paths[idx],
        })
    above_threshold_results.sort(key=lambda x: -x["similarity"])
    
    print(f"\nPatches above threshold {args.threshold}: {len(above_threshold_results)}")
    
    # Save results
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "patch_dir": args.patch_dir,
            "query_images": query_paths,
            "threshold": args.threshold,
            "topk": args.topk,
            "aggregation": args.aggregation,
        },
        "statistics": {
            "total_patches": len(similarities),
            "mean_similarity": float(similarities.mean()),
            "std_similarity": float(similarities.std()),
            "min_similarity": float(similarities.min()),
            "max_similarity": float(similarities.max()),
            "above_threshold": len(above_threshold_results),
        },
        "top_k": topk_results,
        "above_threshold": above_threshold_results,
    }
    
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_file}")
    
    # Save filenames only
    topk_filenames = [r["filename"] for r in topk_results]
    above_threshold_filenames = [r["filename"] for r in above_threshold_results]
    
    with open(output_dir / "topk_filenames.txt", "w") as f:
        f.write("\n".join(topk_filenames))
    
    with open(output_dir / "above_threshold_filenames.txt", "w") as f:
        f.write("\n".join(above_threshold_filenames))
    
    print(f"\nSaved filename lists to {output_dir}")


if __name__ == "__main__":
    main()

