"""
Loss functions for contrastive learning.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute InfoNCE loss.
    
    Args:
        positive_logits: Similarity scores for positive pairs [B]
        negative_logits: Similarity scores for negative pairs [B, num_neg]
        temperature: Temperature scaling factor
        
    Returns:
        Scalar loss value
    """
    # Combine logits: [B, 1 + num_neg]
    logits = torch.cat([
        positive_logits.unsqueeze(1),
        negative_logits
    ], dim=1)
    
    # Scale by temperature
    logits = logits / temperature
    
    # Labels: positive is always at index 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    
    return F.cross_entropy(logits, labels)


def symmetric_info_nce_loss(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss (like CLIP).
    Uses in-batch negatives.
    
    Args:
        query_embeddings: Query embeddings [B, D]
        positive_embeddings: Positive patch embeddings [B, D]
        temperature: Temperature scaling factor
        
    Returns:
        Scalar loss value
    """
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(query_embeddings, positive_embeddings.T) / temperature
    
    # Labels: diagonal elements are positives
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Symmetric loss
    loss_q2p = F.cross_entropy(logits, labels)
    loss_p2q = F.cross_entropy(logits.T, labels)
    
    return (loss_q2p + loss_p2q) / 2

