"""
PyTorch Dataset for CCS Patch Retrieval training.
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CCSRetrievalDataset(Dataset):
    """
    Dataset for contrastive learning of patch retrieval on CCS dataset.
    
    Each sample returns:
    - query_image: A cell query image (e.g., HSIL cell)
    - positive_patch: A patch containing the target cell type
    - negative_patches: Patches NOT containing the target cell type
    
    Supports multiple query images for capturing different cell morphologies.
    """
    
    def __init__(
        self,
        processed_data_dir: str,
        category: str,
        split: str = "train",
        query_image_paths: Optional[list[str]] = None,
        image_size: int = 224,
        num_negatives: int = 7,
        use_augmentation: bool = True,
        augmentation_config: Optional[dict] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            processed_data_dir: Path to preprocessed data directory
            category: Target category (e.g., "HSIL")
            split: One of "train", "val", "test"
            query_image_paths: List of paths to query cell images
            image_size: Size to resize images to
            num_negatives: Number of negative samples per positive
            use_augmentation: Whether to use augmentation for training queries
            augmentation_config: Dict with augmentation parameters (rotation_degrees, color_jitter, etc.)
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.num_negatives = num_negatives
        self.is_train = (split == "train")
        
        # Load positive and negative patch lists
        cat_dir = self.processed_data_dir / category
        
        with open(cat_dir / f"{split}_positives.json", "r") as f:
            self.positive_patches = json.load(f)
            
        with open(cat_dir / f"{split}_negatives.json", "r") as f:
            self.negative_patches = json.load(f)
        
        # Load patches data for image paths and severity ranks
        with open(self.processed_data_dir / "patches.json", "r") as f:
            patches_data = json.load(f)
            self.patch_name_to_path = {}
            self.patch_name_to_rank = {}
            
            # Severity mapping from default.yaml
            severity_map = {
                "ASC-US": 1,
                "LSIL": 2,
                "ASC-H": 3,
                "HSIL": 4,
                "SCC": 5
            }
            
            for p in patches_data:
                name = p["patch_name"]
                self.patch_name_to_path[name] = p["patch_path"]
                
                # Get max severity rank present in the patch
                ranks = [severity_map.get(cat, 0) for cat in p.get("categories_present", [])]
                self.patch_name_to_rank[name] = max(ranks) if ranks else 0
        
        # Query image paths (lazy loading, not cached in memory)
        self.query_image_paths = query_image_paths or []
        
        # Default augmentation config
        if augmentation_config is None:
            augmentation_config = {
                "rotation_degrees": 10,
                "color_jitter": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.05},
                "horizontal_flip_prob": 0.3,
                "vertical_flip_prob": 0.3,
            }
        
        # Image transforms
        # For training: optional strong augmentation on query images to prevent overfitting
        # For val/test: no augmentation
        if self.is_train and use_augmentation:
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(augmentation_config.get("rotation_degrees", 10), fill=255),
                transforms.ColorJitter(**augmentation_config.get("color_jitter", {})),
                transforms.RandomHorizontalFlip(p=augmentation_config.get("horizontal_flip_prob", 0.3)),
                transforms.RandomVerticalFlip(p=augmentation_config.get("vertical_flip_prob", 0.3)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
            self.query_transform = transforms.Compose(transform_list)
        else:
            self.query_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        
        # Patch transform (no augmentation for patches)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        
        # For training, randomly sample queries and negatives
        self.is_train = split == "train"
    
    def _load_query(self, path: str) -> torch.Tensor:
        """Load and transform a query image."""
        img = Image.open(path).convert("RGB")
        return self.query_transform(img)
    
    def get_random_query(self) -> torch.Tensor:
        """Get a random query image (dynamically loaded, for training)."""
        path = random.choice(self.query_image_paths)
        return self._load_query(path)
    
    def __len__(self) -> int:
        return len(self.positive_patches)
    
    def _load_patch(self, patch_name: str) -> torch.Tensor:
        """Load and transform a patch image."""
        path = self.patch_name_to_path[patch_name]
        img = Image.open(path).convert("RGB")
        return self.transform(img)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a training sample.
        
        Returns dict with:
        - query: Query image tensor [3, H, W]
        - query_rank: Severity rank of query
        - positive: Positive patch tensor [3, H, W]
        - positive_rank: Severity rank of positive patch
        - negatives: Negative patches tensor [num_neg, 3, H, W]
        - negative_ranks: Severity ranks of negative patches [num_neg]
        - positive_id: Patch name of positive sample
        - negative_ids: Patch names of negative samples
        """
        # Get positive sample
        positive_id = self.positive_patches[idx]
        positive = self._load_patch(positive_id)
        positive_rank = self.patch_name_to_rank.get(positive_id, 0)
        
        # Sample negatives
        if self.is_train:
            # Random sampling during training
            if len(self.negative_patches) >= self.num_negatives:
                negative_ids = random.sample(self.negative_patches, self.num_negatives)
            else:
                # If not enough negatives, sample with replacement
                negative_ids = random.choices(self.negative_patches, k=self.num_negatives)
        else:
            # Deterministic for val/test: use modulo indexing
            start_idx = (idx * self.num_negatives) % len(self.negative_patches)
            negative_ids = []
            for i in range(self.num_negatives):
                neg_idx = (start_idx + i) % len(self.negative_patches)
                negative_ids.append(self.negative_patches[neg_idx])
        
        negatives = torch.stack([self._load_patch(nid) for nid in negative_ids])
        negative_ranks = torch.tensor([self.patch_name_to_rank.get(nid, 0) for nid in negative_ids], dtype=torch.long)
        
        # Severity map for query
        severity_map = {
            "ASC-US": 1,
            "LSIL": 2,
            "ASC-H": 3,
            "HSIL": 4,
            "SCC": 5
        }
        query_rank = severity_map.get(self.category, 0)
        
        # Get query: random during training, deterministic but diverse during val/test
        if self.is_train:
            query = self.get_random_query()
        else:
            query_idx = idx % len(self.query_image_paths)
            query = self._load_query(self.query_image_paths[query_idx])
        
        return {
            "query": query,
            "query_rank": query_rank,
            "positive": positive,
            "positive_rank": positive_rank,
            "negatives": negatives,
            "negative_ranks": negative_ranks,
            "positive_id": positive_id,
            "negative_ids": negative_ids,
        }


class CCSEvalDataset(Dataset):
    """
    Dataset for evaluation: retrieves all patches for ranking.
    Supports multiple query images.
    """
    
    def __init__(
        self,
        processed_data_dir: str,
        category: str,
        split: str = "test",
        query_image_paths: Optional[list[str]] = None,
        image_size: int = 224,
    ):
        """
        Initialize evaluation dataset.
        
        Args:
            processed_data_dir: Path to preprocessed data directory
            category: Target category (e.g., "HSIL")
            split: One of "train", "val", "test"
            query_image_paths: List of paths to query cell images
            image_size: Size to resize images to
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        
        # Load patch names for this split
        cat_dir = self.processed_data_dir / category
        pos_file = cat_dir / f"{split}_positives.json"
        neg_file = cat_dir / f"{split}_negatives.json"
        if pos_file.exists() and neg_file.exists():
            with open(pos_file, "r") as f:
                positives = json.load(f)
            with open(neg_file, "r") as f:
                negatives = json.load(f)
            self.all_patch_names = positives + negatives
            self.positive_patches = set(positives)
        else:
            with open(self.processed_data_dir / "splits.json", "r") as f:
                splits = json.load(f)
                self.all_patch_names = splits[split]
            with open(cat_dir / f"{split}_positives.json", "r") as f:
                self.positive_patches = set(json.load(f))
        
        # Load patches data for image paths
        with open(self.processed_data_dir / "patches.json", "r") as f:
            patches_data = json.load(f)
            self.patch_name_to_path = {
                p["patch_name"]: p["patch_path"] for p in patches_data
            }
        
        # Query images (support multiple)
        self.query_image_paths = query_image_paths or []
        self._query_images = None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    
    @property
    def query_images(self) -> list[torch.Tensor]:
        """Load and cache all query images."""
        if self._query_images is None:
            if not self.query_image_paths:
                raise ValueError("No query image paths set")
            self._query_images = []
            for path in self.query_image_paths:
                img = Image.open(path).convert("RGB")
                self._query_images.append(self.transform(img))
        return self._query_images
    
    def get_all_queries(self) -> torch.Tensor:
        """Get all query images stacked [num_queries, 3, H, W]."""
        return torch.stack(self.query_images)
    
    @property
    def num_queries(self) -> int:
        """Number of query images."""
        return len(self.query_image_paths)
    
    def __len__(self) -> int:
        return len(self.all_patch_names)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a patch for evaluation.
        
        Returns dict with:
        - patch: Patch image tensor [3, H, W]
        - patch_id: Patch name
        - label: 1 if positive (contains target cell), 0 otherwise
        """
        patch_name = self.all_patch_names[idx]
        path = self.patch_name_to_path[patch_name]
        
        img = Image.open(path).convert("RGB")
        
        patch = self.transform(img)
        
        label = 1 if patch_name in self.positive_patches else 0
        
        return {
            "patch": patch,
            "patch_id": patch_name,
            "label": label,
        }
