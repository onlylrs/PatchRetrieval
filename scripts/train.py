"""
Training script for Patch Retrieval model on CCS dataset.
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.datasets.dataset import CCSRetrievalDataset
from src.models.retrieval_model import PatchRetrievalModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train Patch Retrieval Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Target category (e.g., HSIL)",
    )
    
    parser.add_argument(
        "--queries_root",
        type=str,
        default=None,
        help="Override config: Root directory for query images",
    )
    
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=None,
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "experiments"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Override config options
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--share_weights", action="store_true") # Moved to config
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_query_images(
    queries_root: str,
    category: str,
) -> list[str]:
    """
    Resolve query image paths from queries_root/category.
    """
    if not queries_root:
        raise ValueError("queries_root is not specified")
        
    queries_root = Path(queries_root)
    
    # Check absolute or relative to ROOT_DIR
    if not queries_root.is_absolute():
        queries_root = ROOT_DIR / queries_root
        
    category_dir = queries_root / category
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
    print(f"Found {len(image_paths)} query images in {category_dir}")
    
    return image_paths


def create_dataloaders(
    processed_data_dir: str,
    category: str,
    query_images: list[str],
    batch_size: int,
    num_negatives: int,
    image_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    
    train_dataset = CCSRetrievalDataset(
        processed_data_dir=processed_data_dir,
        category=category,
        split="train",
        query_image_paths=query_images,
        image_size=image_size,
        num_negatives=num_negatives,
    )
    
    val_dataset = CCSRetrievalDataset(
        processed_data_dir=processed_data_dir,
        category=category,
        split="val",
        query_image_paths=query_images,
        image_size=image_size,
        num_negatives=num_negatives,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
    num_training_steps: int,
) -> tuple:
    """Create optimizer and learning rate scheduler."""
    
    training_config = config["training"]
    
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        betas=tuple(training_config["betas"]),
    )
    
    warmup_epochs = training_config["warmup_epochs"]
    num_epochs = training_config["num_epochs"]
    steps_per_epoch = num_training_steps // num_epochs
    warmup_steps = warmup_epochs * steps_per_epoch
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=float(training_config["learning_rate"]) * 0.01,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_every: int = 10,
) -> dict:
    """Train for one epoch."""
    
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        query = batch["query"].to(device)
        positive = batch["positive"].to(device)
        negatives = batch["negatives"].to(device)
        query_rank = batch.get("query_rank").to(device) if "query_rank" in batch else None
        negative_ranks = batch.get("negative_ranks").to(device) if "negative_ranks" in batch else None
        
        optimizer.zero_grad()
        outputs = model.compute_contrastive_loss(
            query, positive, negatives, 
            query_ranks=query_rank, 
            negative_ranks=negative_ranks
        )
        loss = outputs["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_accuracy += outputs["accuracy"].item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{outputs['accuracy'].item():.4f}",
            "temp": f"{outputs['temperature'].item():.4f}",
        })
        
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % log_every == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/accuracy", outputs["accuracy"].item(), global_step)
            writer.add_scalar("train/temperature", outputs["temperature"].item(), global_step)
            writer.add_scalar("train/pos_similarity", outputs["pos_similarity"].item(), global_step)
            writer.add_scalar("train/neg_similarity", outputs["neg_similarity"].item(), global_step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
    
    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    compute_map: bool = True,
) -> dict:
    """Validate the model."""
    
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    # For mAP computation: collect all similarities and labels
    all_similarities = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        query = batch["query"].to(device)
        positive = batch["positive"].to(device)
        negatives = batch["negatives"].to(device)
        query_rank = batch.get("query_rank").to(device) if "query_rank" in batch else None
        negative_ranks = batch.get("negative_ranks").to(device) if "negative_ranks" in batch else None
        
        outputs = model.compute_contrastive_loss(
            query, positive, negatives,
            query_ranks=query_rank,
            negative_ranks=negative_ranks
        )
        
        total_loss += outputs["loss"].item()
        total_accuracy += outputs["accuracy"].item()
        num_batches += 1
        
        # Collect similarities and labels for mAP
        if compute_map:
            # logits shape: [B, 1 + num_neg], first column is positive
            # Convert to float32 first since numpy doesn't support bfloat16
            logits = outputs["logits"].float().cpu()
            batch_size = logits.shape[0]
            num_candidates = logits.shape[1]
            
            for i in range(batch_size):
                # Similarities for this sample
                sims = logits[i].numpy()
                # Labels: first is positive (1), rest are negative (0)
                labels = np.zeros(num_candidates)
                labels[0] = 1
                
                all_similarities.append(sims)
                all_labels.append(labels)
        
        pbar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "acc": f"{outputs['accuracy'].item():.4f}",
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", avg_accuracy, epoch)
    
    result = {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
    }
    
    # Compute mAP
    if compute_map and all_similarities:
        from src.utils.metrics import compute_retrieval_metrics
        
        # Concatenate all similarities and labels
        all_sims = np.concatenate(all_similarities)
        all_labs = np.concatenate(all_labels)
        
        retrieval_metrics = compute_retrieval_metrics(all_sims, all_labs)
        result["mAP"] = retrieval_metrics["mAP"]
        
        writer.add_scalar("val/mAP", result["mAP"], epoch)
    
    return result


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    output_dir: Path,
    is_best: bool = False,
    save_every: int = 1,
):
    """Save model checkpoint."""
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    
    torch.save(checkpoint, output_dir / "latest.pt")
    
    if (epoch + 1) % save_every == 0:
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_negatives is not None:
        config["training"]["num_negatives"] = args.num_negatives
    
    # Use config defaults if not specified
    processed_data_dir = args.processed_data_dir or config["data"]["processed_data_dir"]
    queries_root = args.queries_root or config["data"].get("queries_root", "data/queries")

    # Resolve paths relative to ROOT_DIR if not absolute
    if processed_data_dir and not Path(processed_data_dir).is_absolute():
        processed_data_dir = str(ROOT_DIR / processed_data_dir)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / args.category / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Resolve query images
    query_images = resolve_query_images(queries_root, args.category)
    
    
    # Create dataloaders
    print("Creating dataloaders...")
    query_dir = str(Path(query_images[0]).parent)
    print(f"Query images ({len(query_images)}) from: {query_dir}")
    train_loader, val_loader = create_dataloaders(
        processed_data_dir=processed_data_dir,
        category=args.category,
        query_images=query_images,
        batch_size=config["training"]["batch_size"],
        num_negatives=config["training"]["num_negatives"],
        image_size=config["data"]["image_size"],
        num_workers=args.num_workers,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    pooling_method = config["model"].get("pooling_method", "softmax_attn")
    share_weights = config["model"].get("share_weights", False)
    print(f"Using backbone: {config['model']['backbone']}")
    print(f"Pooling method: {pooling_method}")
    print(f"Share weights: {share_weights}")
    
    model = PatchRetrievalModel(
        model_name=config["model"]["backbone"],
        share_weights=share_weights,
        pooling_method=pooling_method,
        init_temperature=config["training"]["init_temperature"],
        min_temperature=config["training"]["min_temperature"],
        max_temperature=config["training"]["max_temperature"],
        ordinal_margin_base=config["training"].get("ordinal_margin_base", 0.0),
        cache_dir=config["model"]["cache_dir"],
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    num_epochs = config["training"]["num_epochs"]
    num_training_steps = num_epochs * len(train_loader)
    
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, num_training_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    
    # Determine which metric to use for best model
    save_best_metric = config["training"].get("save_best_metric", "loss")
    if save_best_metric == "mAP":
        best_metric_value = 0.0  # Higher is better for mAP
        metric_improved = lambda new, old: new > old
    else:
        best_metric_value = float("inf")  # Lower is better for loss
        metric_improved = lambda new, old: new < old
    
    print(f"Saving best model based on: {save_best_metric}")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # Restore best metric value
        if save_best_metric == "mAP":
            best_metric_value = checkpoint["metrics"].get("val_mAP", 0.0)
        else:
            best_metric_value = checkpoint["metrics"].get("val_loss", float("inf"))
    
    # TensorBoard writer
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer,
            log_every=config["logging"]["log_every"],
        )
        
        val_metrics = validate(model, val_loader, device, epoch, writer, compute_map=True)
        
        # Determine if this is the best model
        if save_best_metric == "mAP":
            current_metric = val_metrics.get("mAP", 0.0)
        else:
            current_metric = val_metrics["loss"]
        
        is_best = metric_improved(current_metric, best_metric_value)
        if is_best:
            best_metric_value = current_metric
        
        metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_mAP": val_metrics.get("mAP", 0.0),
        }
        
        save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir, is_best,
                        save_every=config["training"].get("save_every", 1))
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, mAP: {val_metrics.get('mAP', 0.0):.4f}")
        if is_best:
            print(f"  *** New best model! ({save_best_metric}: {current_metric:.4f}) ***")
        print()
    
    writer.close()
    print(f"\nTraining complete! Best {save_best_metric}: {best_metric_value:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
