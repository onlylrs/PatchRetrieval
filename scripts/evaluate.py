"""
Evaluation script for Patch Retrieval model on CCS dataset.
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.datasets.dataset import CCSEvalDataset
from src.models.retrieval_model import PatchRetrievalModel
from src.utils.metrics import compute_retrieval_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Patch Retrieval Model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs/default.yaml"),
        help="Path to config file (if not in checkpoint directory)",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Target category (e.g., HSIL)",
    )
    parser.add_argument(
        "--query_images",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to query cell image(s), or 'all' to load all from queries_dir/{category}/",
    )
    parser.add_argument(
        "--queries_dir",
        type=str,
        default=None,
        help="Root directory for query images (used when --query_images is 'all')",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Number of query images to randomly sample (used with --query_images all)",
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=None,
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    # parser.add_argument(
    #     "--share_weights",
    #     action="store_true",
    #     help="Use shared weights between query and patch encoders",
    # )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="How to aggregate similarities across multiple queries (default: max)",
    )
    
    return parser.parse_args()


def resolve_query_images(
    query_images: list[str], 
    queries_dir: str, 
    category: str,
    num_queries: int = None,
    seed: int = 42,
) -> list[str]:
    """
    Resolve query image paths.
    
    If query_images is ['all'], load images from queries_dir/{category}/.
    If num_queries is specified, randomly sample that many images.
    """
    if len(query_images) == 1 and query_images[0].lower() == "all":
        category_dir = Path(queries_dir) / category
        if not category_dir.exists():
            raise ValueError(f"Query directory not found: {category_dir}")
        
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(category_dir.glob(f"*{ext}"))
            image_paths.extend(category_dir.glob(f"*{ext.upper()}"))
        
        if not image_paths:
            raise ValueError(f"No images found in {category_dir}")
        
        image_paths = sorted([str(p) for p in image_paths])
        print(f"Found {len(image_paths)} images in {category_dir}")
        
        # Random sample if num_queries specified
        if num_queries is not None and num_queries < len(image_paths):
            import random
            random.seed(seed)
            image_paths = random.sample(image_paths, num_queries)
            print(f"Randomly sampled {num_queries} query images")
        
        return image_paths
    else:
        return query_images


def load_config(checkpoint_path: str, config_path: str = None) -> dict:
    """Load config from checkpoint directory or specified path."""
    if config_path is not None:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    checkpoint_dir = Path(checkpoint_path).parent
    config_file = checkpoint_dir / "config.yaml"
    
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    
    return yaml.safe_load(open("configs/default.yaml", "r"))


@torch.no_grad()
def evaluate(
    model: PatchRetrievalModel,
    dataloader: DataLoader,
    device: torch.device,
    aggregation: str = "max",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Evaluate model on dataset with multiple query support.
    """
    model.eval()
    
    all_similarities = []
    all_labels = []
    all_patch_ids = []
    
    query_images = dataloader.dataset.get_all_queries().to(device)
    num_queries = query_images.shape[0]
    print(f"Using {num_queries} query image(s) with {aggregation} aggregation")
    
    query_embeddings = model.encoder.encode_query(query_images)
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        patches = batch["patch"].to(device)
        labels = batch["label"]
        patch_ids = batch["patch_id"]
        
        B = patches.shape[0]
        patch_embeddings = model.encoder.encode_patches(patches)
        
        batch_similarities = []
        for q_idx in range(num_queries):
            query_emb = query_embeddings[q_idx:q_idx+1]
            query_expanded = query_emb.expand(B, -1)
            sims = model.compute_similarity(query_expanded, patch_embeddings)
            batch_similarities.append(sims)
        
        batch_similarities = torch.stack(batch_similarities, dim=0)
        if aggregation == "max":
            final_similarities = batch_similarities.max(dim=0).values
        else:
            final_similarities = batch_similarities.mean(dim=0)
        
        all_similarities.append(final_similarities.cpu().float().numpy())
        all_labels.append(labels.numpy())
        all_patch_ids.extend(patch_ids)
    
    similarities = np.concatenate(all_similarities)
    labels = np.concatenate(all_labels)
    
    return similarities, labels, all_patch_ids


def main():
    args = parse_args()
    
    config = load_config(args.checkpoint, args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use config defaults if not specified
    processed_data_dir = args.processed_data_dir or config["data"]["processed_data_dir"]
    queries_dir = args.queries_dir or config["data"]["queries_dir"]
    
    # Resolve paths relative to ROOT_DIR if not absolute
    if processed_data_dir and not Path(processed_data_dir).is_absolute():
        processed_data_dir = str(ROOT_DIR / processed_data_dir)
    if queries_dir and not Path(queries_dir).is_absolute():
        queries_dir = str(ROOT_DIR / queries_dir)
    
    # Resolve query images
    query_images = resolve_query_images(
        args.query_images, queries_dir, args.category,
        num_queries=args.num_queries, seed=42
    )
    
    # Create dataset and dataloader
    print(f"\nLoading {args.split} data...")
    print(f"Query images ({len(query_images)}): {query_images}")
    dataset = CCSEvalDataset(
        processed_data_dir=processed_data_dir,
        category=args.category,
        split=args.split,
        query_image_paths=query_images,
        image_size=config["data"]["image_size"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    num_positives = len(dataset.positive_patches)
    num_total = len(dataset)
    print(f"Total patches: {num_total}")
    print(f"Positive patches: {num_positives} ({100*num_positives/num_total:.1f}%)")
    
    # Create model
    print("\nLoading model...")
    pooling_method = config["model"].get("pooling_method", "softmax_attn")

    model = PatchRetrievalModel(
        model_name=config["model"]["backbone"],
        pooling_method=pooling_method,
        cache_dir=config["model"]["cache_dir"],
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    print("\nEvaluating...")
    similarities, labels, patch_ids = evaluate(
        model, dataloader, device, aggregation=args.aggregation
    )
    
    # Compute metrics
    metrics = compute_retrieval_metrics(
        similarities, labels,
        k_values=[1, 5, 10, 20, 50, 100],
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results on {args.split} set")
    print("=" * 60)
    
    print("\nRetrieval Metrics:")
    for k in [1, 5, 10, 20, 50]:
        recall_key = f"recall@{k}"
        precision_key = f"precision@{k}"
        if recall_key in metrics:
            print(f"  Recall@{k}: {metrics[recall_key]:.4f}")
            print(f"  Precision@{k}: {metrics[precision_key]:.4f}")
    
    print(f"\n  mAP: {metrics['mAP']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    # Threshold-based metrics
    print("\nThreshold-based Metrics:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        above_thresh = similarities >= threshold
        num_above = above_thresh.sum()
        if num_above > 0:
            num_correct = (above_thresh & (labels == 1)).sum()
            precision = num_correct / num_above
            recall = num_correct / labels.sum()
            print(f"  Threshold={threshold:.1f}: {num_correct}/{num_above} correct "
                  f"(Precision={precision:.4f}, Recall={recall:.4f})")
        else:
            print(f"  Threshold={threshold:.1f}: 0 samples above threshold")
    
    # Get top retrieved patches
    print("\nTop 10 Retrieved Patches:")
    sorted_indices = np.argsort(-similarities)
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        patch_id = patch_ids[idx]
        sim = similarities[idx]
        label = "✓ Positive" if labels[idx] == 1 else "✗ Negative"
        print(f"  {i+1}. {patch_id}: {sim:.4f} ({label})")
    
    # Save results
    if args.output_file:
        results = {
            "category": args.category,
            "split": args.split,
            "query_images": query_images,
            "checkpoint": args.checkpoint,
            "num_total": num_total,
            "num_positives": num_positives,
            "metrics": metrics,
            "rankings": [
                {
                    "rank": i + 1,
                    "patch_id": patch_ids[sorted_indices[i]],
                    "similarity": float(similarities[sorted_indices[i]]),
                    "is_positive": bool(labels[sorted_indices[i]] == 1),
                }
                for i in range(len(sorted_indices))
            ],
        }
        
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
