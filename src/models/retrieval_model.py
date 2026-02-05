"""
Patch Retrieval Model with configurable pooling methods.
Supports CLIP, SigLIP2, and DINOv3 backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import DualViTEncoder, DEFAULT_CACHE_DIR

# Supported pooling methods
POOLING_METHODS = ["softmax_attn", "mean", "max"]


class PatchRetrievalModel(nn.Module):
    """
    Patch Retrieval Model.
    
    Query: ViT -> CLS token embedding (or pooled output for SigLIP2)
    Patch: ViT -> Patch tokens -> Pooling with query
    
    Pooling methods:
    - softmax_attn: Softmax attention weighted sum of similarities (default)
    - mean: Simple mean of all patch-query similarities
    - max: Maximum similarity across patches
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        pooling_method: str = "softmax_attn",
        init_temperature: float = 0.07,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
        ordinal_margin_base: float = 0.0,
        dropout: float = 0.0,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize retrieval model.
        
        Args:
            model_name: HuggingFace model name for ViT (CLIP, SigLIP2, or DINOv3)
            pooling_method: Method to pool patch similarities ("softmax_attn", "mean", "max")
            init_temperature: Initial temperature for softmax
            min_temperature: Minimum temperature (for clamping)
            max_temperature: Maximum temperature (for clamping)
            ordinal_margin_base: Base margin for ordinal contrastive loss (0.0 to disable)
            dropout: Dropout rate for regularization (0.0 to disable)
            cache_dir: Directory to cache downloaded model weights
        """
        super().__init__()
        
        if pooling_method not in POOLING_METHODS:
            raise ValueError(f"pooling_method must be one of {POOLING_METHODS}, got {pooling_method}")
        
        self.pooling_method = pooling_method
        self.encoder = DualViTEncoder(model_name, cache_dir=cache_dir)
        self.hidden_size = self.encoder.hidden_size
        self.ordinal_margin_base = ordinal_margin_base
        
        # Learnable temperature (logit_scale like CLIP)
        # temperature = 1 / exp(logit_scale)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature))
        )
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # L2 normalization for embeddings
        self.normalize = True
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return 1.0 / self.logit_scale.exp()
    
    def clamp_temperature(self):
        """Clamp logit_scale to keep temperature in valid range."""
        with torch.no_grad():
            device = self.logit_scale.device
            min_logit_scale = torch.log(torch.tensor(1.0 / self.max_temperature, device=device))
            max_logit_scale = torch.log(torch.tensor(1.0 / self.min_temperature, device=device))
            self.logit_scale.clamp_(min_logit_scale, max_logit_scale)
    
    def compute_similarity(
        self,
        query_embedding: torch.Tensor,
        patch_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between query and patch using configurable pooling method.
        
        Args:
            query_embedding: Query CLS embedding [B, D] or [D]
            patch_embeddings: Patch token embeddings [B, N, D] or [N, D]
            
        Returns:
            Similarity scores [B] or scalar
        """
        # Handle single query case
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if patch_embeddings.dim() == 2:
            patch_embeddings = patch_embeddings.unsqueeze(0)
        
        # L2 normalize
        if self.normalize:
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            patch_embeddings = F.normalize(patch_embeddings, p=2, dim=-1)
        
        # Compute per-patch similarities: [B, N]
        # query: [B, D], patches: [B, N, D]
        per_patch_similarities = torch.einsum("bd,bnd->bn", query_embedding, patch_embeddings)
        
        # Apply pooling method
        if self.pooling_method == "mean":
            # Simple mean of all patch similarities
            similarity = per_patch_similarities.mean(dim=-1)  # [B]
            
        elif self.pooling_method == "max":
            # Maximum similarity across patches
            similarity = per_patch_similarities.max(dim=-1).values  # [B]
            
        elif self.pooling_method == "softmax_attn":
            # Softmax attention weighted sum (original method)
            # Scale by temperature
            attention_logits = per_patch_similarities * self.logit_scale.exp()
            
            # Softmax over patches
            attention_weights = F.softmax(attention_logits, dim=-1)  # [B, N]
            
            # Weighted sum of similarities
            similarity = (attention_weights * per_patch_similarities).sum(dim=-1)  # [B]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        return similarity
    
    def forward(
        self,
        query_images: torch.Tensor,
        patch_images: torch.Tensor,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            query_images: Query images [B, 3, H, W]
            patch_images: Patch images [B, 3, H, W] or [B, N, 3, H, W] for multiple patches
            
        Returns:
            dict with:
            - similarities: Similarity scores
            - query_embeddings: Query CLS embeddings
            - patch_embeddings: Patch token embeddings
        """
        # Encode
        encoded = self.encoder(query_images, patch_images)
        query_embeddings = encoded["query_embeddings"]  # [B, D]
        patch_embeddings = encoded["patch_embeddings"]  # [B, N_patches, D] or [B, M, N_patches, D]
        
        # Compute similarities
        if patch_embeddings.dim() == 4:
            # Multiple patches per query: [B, M, N_patches, D]
            B, M, N_patches, D = patch_embeddings.shape
            
            # Expand query for each patch
            query_expanded = query_embeddings.unsqueeze(1).expand(B, M, D)  # [B, M, D]
            query_flat = query_expanded.reshape(B * M, D)
            patch_flat = patch_embeddings.reshape(B * M, N_patches, D)
            
            similarities_flat = self.compute_similarity(query_flat, patch_flat)
            similarities = similarities_flat.view(B, M)  # [B, M]
        else:
            similarities = self.compute_similarity(query_embeddings, patch_embeddings)
        
        return {
            "similarities": similarities,
            "query_embeddings": query_embeddings,
            "patch_embeddings": patch_embeddings,
        }
    
    def compute_contrastive_loss(
        self,
        query_images: torch.Tensor,
        positive_patches: torch.Tensor,
        negative_patches: torch.Tensor,
        query_ranks: Optional[torch.Tensor] = None,
        negative_ranks: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute InfoNCE contrastive loss with optional ordinal margin.
        
        Args:
            query_images: Query images [B, 3, H, W]
            positive_patches: Positive patches [B, 3, H, W]
            negative_patches: Negative patches [B, num_neg, 3, H, W]
            query_ranks: Severity ranks of queries [B]
            negative_ranks: Severity ranks of negative patches [B, num_neg]
            
        Returns:
            dict with loss and other metrics
        """
        B = query_images.shape[0]
        num_neg = negative_patches.shape[1]
        
        # Encode query
        query_embeddings = self.encoder.encode_query(query_images)  # [B, D]
        query_embeddings = self.dropout(query_embeddings)  # Apply dropout
        
        # Encode positive patches
        pos_patch_embeddings = self.encoder.encode_patches(positive_patches)  # [B, N, D]
        pos_patch_embeddings = self.dropout(pos_patch_embeddings)  # Apply dropout
        
        # Encode negative patches
        neg_flat = negative_patches.view(B * num_neg, *negative_patches.shape[2:])
        neg_patch_embeddings_flat = self.encoder.encode_patches(neg_flat)  # [B*num_neg, N, D]
        neg_patch_embeddings_flat = self.dropout(neg_patch_embeddings_flat)  # Apply dropout
        N_patches, D = neg_patch_embeddings_flat.shape[1], neg_patch_embeddings_flat.shape[2]
        neg_patch_embeddings = neg_patch_embeddings_flat.view(B, num_neg, N_patches, D)
        
        # Compute positive similarities
        pos_similarities = self.compute_similarity(query_embeddings, pos_patch_embeddings)  # [B]
        
        # Compute negative similarities
        query_expanded = query_embeddings.unsqueeze(1).expand(B, num_neg, D)  # [B, num_neg, D]
        query_flat = query_expanded.reshape(B * num_neg, D)
        neg_flat = neg_patch_embeddings.reshape(B * num_neg, N_patches, D)
        neg_similarities_flat = self.compute_similarity(query_flat, neg_flat)
        neg_similarities = neg_similarities_flat.view(B, num_neg)  # [B, num_neg]
        
        # Apply Ordinal Margin if enabled
        if self.ordinal_margin_base > 0 and query_ranks is not None and negative_ranks is not None:
            # Distance in pathology severity: [B, num_neg]
            # query_ranks: [B] -> [B, 1]
            q_ranks = query_ranks.unsqueeze(1).to(neg_similarities.device)
            n_ranks = negative_ranks.to(neg_similarities.device)
            
            # Severity distance: small distance means "hard negative" (looks like query but isn't)
            severity_dist = torch.abs(q_ranks - n_ranks)
            
            # Inverse distance as margin: small dist -> large margin
            # We add 1 to avoid division by zero and moderate the margin
            ordinal_margin = self.ordinal_margin_base / (severity_dist + 1.0)
            
            # Add margin to negative similarities to make them harder to distinguish from positive
            neg_similarities = neg_similarities + ordinal_margin
        
        # InfoNCE loss
        # logits: [B, 1 + num_neg] where first column is positive
        logits = torch.cat([pos_similarities.unsqueeze(1), neg_similarities], dim=1)
        
        # Scale logits
        logits = logits * self.logit_scale.exp()
        
        # Labels: positive is always at index 0
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Clamp temperature after forward pass
        self.clamp_temperature()
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
        
        return {
            "loss": loss,
            "contrastive_loss": loss,
            "accuracy": accuracy,
            "pos_similarity": pos_similarities.mean(),
            "neg_similarity": neg_similarities.mean(),
            "temperature": self.temperature,
            "logits": logits,
        }
    
    @torch.no_grad()
    def retrieve(
        self,
        query_image: torch.Tensor,
        patch_images: torch.Tensor,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Retrieve patches by computing similarities with query.
        
        Args:
            query_image: Query image [1, 3, H, W] or [3, H, W]
            patch_images: All patch images [N, 3, H, W]
            batch_size: Batch size for processing patches
            
        Returns:
            Similarity scores [N]
        """
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)
        
        # Encode query once
        query_embedding = self.encoder.encode_query(query_image)  # [1, D]
        
        # Encode patches in batches
        all_similarities = []
        num_patches = patch_images.shape[0]
        
        for i in range(0, num_patches, batch_size):
            batch = patch_images[i:i+batch_size]
            patch_embeddings = self.encoder.encode_patches(batch)  # [B, N, D]
            
            # Expand query for batch
            query_expanded = query_embedding.expand(batch.shape[0], -1)  # [B, D]
            similarities = self.compute_similarity(query_expanded, patch_embeddings)
            all_similarities.append(similarities)
        
        return torch.cat(all_similarities, dim=0)

