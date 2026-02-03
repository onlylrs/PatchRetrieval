"""
WSI-level multi-retriever inference and visualization.
Run 5 retrievers (ASC-US, LSIL, ASC-H, HSIL, SCC) on a WSI's patches
and visualize results on thumbnail.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from src.models.retrieval_model import PatchRetrievalModel



CATEGORIES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]

# Updated paths for legacy checkpoints
# Note: These paths are relative to ROOT_DIR if you use str(ROOT_DIR / path) later,
# but currently they are strings. The script needs to handle them.
# The original script apparently expected CWD based paths.
# I will use ROOT_DIR / "experiments/legacy/checkpoints/..."
CHECKPOINT_PATHS = {
    "ASC-US": str(ROOT_DIR / "experiments/legacy/checkpoints/ASC-US/20251226_223843_bestAUC/best.pt"),
    "LSIL": str(ROOT_DIR / "experiments/legacy/checkpoints/LSIL/20251226_103805_bestAUC/best.pt"),
    "ASC-H": str(ROOT_DIR / "experiments/legacy/checkpoints/ASC-H/20251227_092502_bestAUC/best.pt"),
    "HSIL": str(ROOT_DIR / "experiments/legacy/checkpoints/HSIL/20251224_210120_bestAUC/best.pt"),
    "SCC": str(ROOT_DIR / "experiments/legacy/checkpoints/SCC/20251229_102629_bestAUC/best.pt"),
}

CATEGORY_COLORS = {
    "ASC-US": (65, 105, 225),    # Royal Blue
    "LSIL": (50, 205, 50),       # Lime Green
    "ASC-H": (255, 165, 0),      # Orange
    "HSIL": (220, 20, 60),       # Crimson
    "SCC": (148, 0, 211),        # Dark Violet
}

CATEGORY_SHORT = {
    "ASC-US": "AU",
    "LSIL": "L",
    "ASC-H": "AH",
    "HSIL": "H",
    "SCC": "S",
}

QUERIES_DIR = "queries"


def parse_args():
    parser = argparse.ArgumentParser(description="WSI multi-retriever inference")
    
    parser.add_argument(
        "--patch_dir",
        type=str,
        required=True,
        help="Directory containing patch images",
    )
    parser.add_argument(
        "--wsi_path",
        type=str,
        default=None,
        help="Path to original WSI file for overlay visualization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
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
        "--thumbnail_scale",
        type=float,
        default=0.1,
        help="Scale factor for thumbnail (relative to original WSI)",
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


def parse_patch_position(filename):
    """Parse patch position from filename like '9760_9856_1200.jpg'."""
    stem = Path(filename).stem
    parts = stem.split("_")
    x = int(parts[0])
    y = int(parts[1])
    size = int(parts[2])
    return x, y, size


def load_query_images(category, image_size=224):
    """Load all query images for a category."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    
    query_dir = Path(QUERIES_DIR) / category
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
    
    query_embeddings = model.encoder.encode_query(query_images.to(device))
    
    for batch in dataloader:
        images = batch["image"].to(device)
        filenames = batch["filename"]
        
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
    
    similarities = np.concatenate(all_similarities)
    return similarities, all_filenames


def get_topk_patches(similarities, filenames, topk):
    """Get top-k patches with their positions and tier."""
    sorted_indices = np.argsort(-similarities)[:topk]
    
    tier_size = topk // 3
    results = []
    
    for rank, idx in enumerate(sorted_indices):
        x, y, size = parse_patch_position(filenames[idx])
        
        if rank < tier_size:
            tier = 1
        elif rank < tier_size * 2:
            tier = 2
        else:
            tier = 3
        
        results.append({
            "rank": rank + 1,
            "tier": tier,
            "filename": filenames[idx],
            "similarity": float(similarities[idx]),
            "x": x,
            "y": y,
            "size": size,
        })
    
    return results


def blend_colors(colors, alpha=0.6):
    """Blend multiple RGBA colors together."""
    if len(colors) == 1:
        r, g, b = colors[0]
        return (r, g, b, int(255 * alpha))
    
    r = min(255, sum(c[0] for c in colors))
    g = min(255, sum(c[1] for c in colors))
    b = min(255, sum(c[2] for c in colors))
    return (r, g, b, int(255 * alpha))


def get_tier_alpha(tier):
    """Get alpha value based on tier (1=highest, 3=lowest)."""
    alphas = {1: 0.8, 2: 0.5, 3: 0.3}
    return alphas.get(tier, 0.3)


def get_wsi_dimensions(patch_dir):
    """Get WSI dimensions from patch filenames."""
    all_patches = list(Path(patch_dir).glob("*.jpg")) + list(Path(patch_dir).glob("*.png"))
    
    max_x, max_y = 0, 0
    patch_size = 1200
    
    for p in all_patches:
        x, y, size = parse_patch_position(p.name)
        max_x = max(max_x, x + size)
        max_y = max(max_y, y + size)
        patch_size = size
    
    return max_x, max_y, patch_size


def create_wsi_visualization(all_results, patch_dir, output_path, scale=0.1):
    """Create WSI thumbnail with colored overlay for top-k patches."""
    
    max_x, max_y, patch_size = get_wsi_dimensions(patch_dir)
    
    thumb_w = int(max_x * scale)
    thumb_h = int(max_y * scale)
    patch_thumb_size = int(patch_size * scale)
    
    print(f"WSI size: {max_x} x {max_y}")
    print(f"Thumbnail size: {thumb_w} x {thumb_h}")
    
    thumbnail = Image.new("RGBA", (thumb_w, thumb_h), (240, 240, 240, 255))
    overlay = Image.new("RGBA", (thumb_w, thumb_h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    position_to_categories = defaultdict(list)
    
    for category, results in all_results.items():
        for patch_info in results:
            x, y = patch_info["x"], patch_info["y"]
            tier = patch_info["tier"]
            position_to_categories[(x, y)].append((category, tier, patch_info["rank"]))
    
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", max(8, patch_thumb_size // 3))
    except:
        font_label = ImageFont.load_default()
    
    for (x, y), cat_list in position_to_categories.items():
        thumb_x = int(x * scale)
        thumb_y = int(y * scale)
        
        min_tier = min(t for _, t, _ in cat_list)
        alpha = get_tier_alpha(min_tier)
        
        colors = [CATEGORY_COLORS[cat] for cat, _, _ in cat_list]
        blended = blend_colors(colors, alpha)
        
        is_overlap = len(cat_list) > 1
        
        if is_overlap:
            draw_overlay.rectangle(
                [thumb_x, thumb_y, thumb_x + patch_thumb_size, thumb_y + patch_thumb_size],
                fill=blended,
                outline=(255, 255, 255, 255),
                width=3,
            )
            draw_overlay.rectangle(
                [thumb_x + 2, thumb_y + 2, thumb_x + patch_thumb_size - 2, thumb_y + patch_thumb_size - 2],
                outline=(0, 0, 0, 255),
                width=2,
            )
            label = "".join([CATEGORY_SHORT[cat] for cat, _, _ in sorted(cat_list, key=lambda x: x[2])])
            draw_overlay.text(
                (thumb_x + patch_thumb_size // 2, thumb_y + patch_thumb_size // 2),
                label,
                fill=(255, 255, 255, 255),
                font=font_label,
                anchor="mm",
            )
        else:
            draw_overlay.rectangle(
                [thumb_x, thumb_y, thumb_x + patch_thumb_size, thumb_y + patch_thumb_size],
                fill=blended,
                outline=(0, 0, 0, 100),
            )
    
    result = Image.alpha_composite(thumbnail, overlay)
    
    legend_height = 140
    final_img = Image.new("RGBA", (thumb_w, thumb_h + legend_height), (255, 255, 255, 255))
    final_img.paste(result, (0, 0))
    
    draw = ImageDraw.Draw(final_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    y_pos = thumb_h + 10
    x_pos = 10
    
    draw.text((x_pos, y_pos), "Legend:", fill=(0, 0, 0, 255), font=font)
    y_pos += 22
    
    for category in CATEGORIES:
        color = CATEGORY_COLORS[category]
        draw.rectangle([x_pos, y_pos, x_pos + 20, y_pos + 15], fill=color + (255,), outline=(0, 0, 0, 255))
        draw.text((x_pos + 25, y_pos), f"{category} ({CATEGORY_SHORT[category]})", fill=(0, 0, 0, 255), font=font_small)
        x_pos += 120
    
    y_pos += 25
    x_pos = 10
    draw.text((x_pos, y_pos), "Tier intensity: Tier1(dark)=Top1/3 | Tier2(medium)=Middle1/3 | Tier3(light)=Bottom1/3", 
              fill=(0, 0, 0, 255), font=font_small)
    
    overlapping_positions = [(pos, cats) for pos, cats in position_to_categories.items() if len(cats) > 1]
    
    y_pos += 20
    draw.text((x_pos, y_pos), f"Overlapping patches: {len(overlapping_positions)} (marked with white+black border and category labels)", 
              fill=(200, 0, 0, 255), font=font_small)
    
    if overlapping_positions:
        y_pos += 18
        overlap_examples = []
        for pos, cats in list(overlapping_positions)[:5]:
            cat_names = "+".join([cat for cat, _, _ in cats])
            overlap_examples.append(f"({pos[0]},{pos[1]}): {cat_names}")
        draw.text((x_pos, y_pos), "Examples: " + " | ".join(overlap_examples), 
                  fill=(100, 100, 100, 255), font=font_small)
    
    final_img.save(output_path, quality=95)
    print(f"Saved visualization to {output_path}")
    
    return position_to_categories


def create_wsi_overlay_visualization(all_results, wsi_path, patch_dir, output_path, scale=0.5):
    """Create visualization with overlay on actual WSI thumbnail."""
    from Aslide import Slide
    
    slide = Slide(wsi_path)
    wsi_w, wsi_h = slide.dimensions
    
    thumb_w = int(wsi_w * scale)
    thumb_h = int(wsi_h * scale)
    
    print(f"WSI actual size: {wsi_w} x {wsi_h}")
    print(f"Overlay thumbnail size: {thumb_w} x {thumb_h}")
    
    best_level = slide.get_best_level_for_downsample(1.0 / scale)
    level_w, level_h = slide.level_dimensions[best_level]
    wsi_thumbnail = slide.read_region((0, 0), best_level, (level_w, level_h)).convert("RGB")
    wsi_thumbnail = wsi_thumbnail.resize((thumb_w, thumb_h), Image.LANCZOS)
    wsi_thumbnail = wsi_thumbnail.convert("RGBA")
    slide.close()
    
    _, _, patch_size = get_wsi_dimensions(patch_dir)
    patch_thumb_size = int(patch_size * scale)
    
    overlay = Image.new("RGBA", (thumb_w, thumb_h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    position_to_categories = defaultdict(list)
    
    for category, results in all_results.items():
        for patch_info in results:
            x, y = patch_info["x"], patch_info["y"]
            tier = patch_info["tier"]
            position_to_categories[(x, y)].append((category, tier, patch_info["rank"]))
    
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", max(8, patch_thumb_size // 3))
    except:
        font_label = ImageFont.load_default()
    
    overlay_alpha = 0.5
    
    for (x, y), cat_list in position_to_categories.items():
        thumb_x = int(x * scale)
        thumb_y = int(y * scale)
        
        min_tier = min(t for _, t, _ in cat_list)
        tier_factor = {1: 1.0, 2: 0.7, 3: 0.4}[min_tier]
        alpha = overlay_alpha * tier_factor
        
        colors = [CATEGORY_COLORS[cat] for cat, _, _ in cat_list]
        blended = blend_colors(colors, alpha)
        
        is_overlap = len(cat_list) > 1
        
        if is_overlap:
            draw_overlay.rectangle(
                [thumb_x, thumb_y, thumb_x + patch_thumb_size, thumb_y + patch_thumb_size],
                fill=blended,
                outline=(255, 255, 0, 255),
                width=3,
            )
            label = "".join([CATEGORY_SHORT[cat] for cat, _, _ in sorted(cat_list, key=lambda x: x[2])])
            draw_overlay.text(
                (thumb_x + patch_thumb_size // 2, thumb_y + patch_thumb_size // 2),
                label,
                fill=(255, 255, 0, 255),
                font=font_label,
                anchor="mm",
            )
        else:
            category = cat_list[0][0]
            draw_overlay.rectangle(
                [thumb_x, thumb_y, thumb_x + patch_thumb_size, thumb_y + patch_thumb_size],
                fill=blended,
                outline=CATEGORY_COLORS[category] + (200,),
                width=2,
            )
    
    result = Image.alpha_composite(wsi_thumbnail, overlay)
    
    legend_height = 140
    final_img = Image.new("RGBA", (thumb_w, thumb_h + legend_height), (255, 255, 255, 255))
    final_img.paste(result, (0, 0))
    
    draw = ImageDraw.Draw(final_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    y_pos = thumb_h + 10
    x_pos = 10
    
    draw.text((x_pos, y_pos), "Legend:", fill=(0, 0, 0, 255), font=font)
    y_pos += 22
    
    for category in CATEGORIES:
        color = CATEGORY_COLORS[category]
        draw.rectangle([x_pos, y_pos, x_pos + 20, y_pos + 15], fill=color + (255,), outline=(0, 0, 0, 255))
        draw.text((x_pos + 25, y_pos), f"{category} ({CATEGORY_SHORT[category]})", fill=(0, 0, 0, 255), font=font_small)
        x_pos += 120
    
    y_pos += 25
    x_pos = 10
    draw.text((x_pos, y_pos), "Tier intensity: Tier1(dark)=Top1/3 | Tier2(medium)=Middle1/3 | Tier3(light)=Bottom1/3", 
              fill=(0, 0, 0, 255), font=font_small)
    
    overlapping_positions = [(pos, cats) for pos, cats in position_to_categories.items() if len(cats) > 1]
    
    y_pos += 20
    draw.text((x_pos, y_pos), f"Overlapping patches: {len(overlapping_positions)} (marked with YELLOW border and category labels)", 
              fill=(200, 150, 0, 255), font=font_small)
    
    if overlapping_positions:
        y_pos += 18
        overlap_examples = []
        for pos, cats in list(overlapping_positions)[:5]:
            cat_names = "+".join([cat for cat, _, _ in cats])
            overlap_examples.append(f"({pos[0]},{pos[1]}): {cat_names}")
        draw.text((x_pos, y_pos), "Examples: " + " | ".join(overlap_examples), 
                  fill=(100, 100, 100, 255), font=font_small)
    
    final_img.save(output_path, quality=95)
    print(f"Saved WSI overlay visualization to {output_path}")
    
    return position_to_categories


def main():
    args = parse_args()
    
    base_dir = Path(__file__).parent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = PatchFolderDataset(args.patch_dir)
    print(f"Found {len(dataset)} patches in {args.patch_dir}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    all_results = {}
    all_statistics = {}
    
    for category in tqdm(CATEGORIES, desc="Running retrievers"):
        print(f"\n{'='*50}")
        print(f"Processing retriever: {category}")
        print(f"{'='*50}")
        
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
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        query_images = load_query_images(category)
        print(f"Using {len(query_images)} query images")
        
        similarities, filenames = run_inference(
            model, query_images, dataloader, device, args.aggregation
        )
        
        topk_results = get_topk_patches(similarities, filenames, args.topk)
        all_results[category] = topk_results
        
        all_statistics[category] = {
            "total_patches": len(similarities),
            "mean_similarity": float(similarities.mean()),
            "std_similarity": float(similarities.std()),
            "min_similarity": float(similarities.min()),
            "max_similarity": float(similarities.max()),
            "top1_similarity": float(topk_results[0]["similarity"]),
            "top10_mean_similarity": float(np.mean([r["similarity"] for r in topk_results[:10]])),
        }
        
        print(f"Top 5 patches for {category}:")
        for r in topk_results[:5]:
            print(f"  #{r['rank']}: {r['filename']} (sim={r['similarity']:.4f})")
        
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("Creating WSI visualization...")
    print(f"{'='*50}")
    
    position_to_categories = create_wsi_visualization(
        all_results, 
        args.patch_dir, 
        output_dir / "wsi_visualization.png",
        scale=args.thumbnail_scale,
    )
    
    if args.wsi_path:
        print(f"\n{'='*50}")
        print("Creating WSI overlay visualization...")
        print(f"{'='*50}")
        
        create_wsi_overlay_visualization(
            all_results,
            args.wsi_path,
            args.patch_dir,
            output_dir / "wsi_overlay.png",
            scale=args.thumbnail_scale,
        )
    
    overlapping_patches = []
    for (x, y), cat_list in position_to_categories.items():
        if len(cat_list) > 1:
            overlapping_patches.append({
                "position": {"x": x, "y": y},
                "categories": [{"category": cat, "tier": tier, "rank": rank} 
                              for cat, tier, rank in cat_list],
            })
    
    wsi_name = Path(args.patch_dir).name
    
    output_data = {
        "wsi_name": wsi_name,
        "patch_dir": args.patch_dir,
        "wsi_path": args.wsi_path,
        "config": {
            "topk": args.topk,
            "aggregation": args.aggregation,
            "thumbnail_scale": args.thumbnail_scale,
        },
        "statistics": all_statistics,
        "results": {cat: results for cat, results in all_results.items()},
        "overlapping_patches": overlapping_patches,
        "summary": {
            "total_unique_topk_patches": len(position_to_categories),
            "overlapping_count": len(overlapping_patches),
            "per_category_topk": {cat: args.topk for cat in CATEGORIES},
        },
    }
    
    output_json = output_dir / "results.json"
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {output_json}")
    
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Total unique top-{args.topk} patches across all retrievers: {len(position_to_categories)}")
    print(f"Overlapping patches (detected by multiple retrievers): {len(overlapping_patches)}")
    
    if overlapping_patches:
        print(f"\nOverlapping patch details:")
        for i, op in enumerate(overlapping_patches[:10]):
            cats = ", ".join([f"{c['category']}(rank{c['rank']})" for c in op["categories"]])
            print(f"  {i+1}. Position ({op['position']['x']}, {op['position']['y']}): {cats}")
        if len(overlapping_patches) > 10:
            print(f"  ... and {len(overlapping_patches) - 10} more")
    
    for category in CATEGORIES:
        stats = all_statistics[category]
        print(f"\n{category}:")
        print(f"  Mean sim: {stats['mean_similarity']:.4f}, Top1: {stats['top1_similarity']:.4f}")


if __name__ == "__main__":
    main()
