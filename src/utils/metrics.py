"""
Evaluation metrics for retrieval.
"""

import numpy as np
import torch
from typing import Optional


def compute_retrieval_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Compute retrieval metrics.
    
    Args:
        similarities: Similarity scores [N]
        labels: Binary labels (1 for positive, 0 for negative) [N]
        k_values: List of k values for Recall@k and Precision@k
        
    Returns:
        dict with:
        - recall@k for each k
        - precision@k for each k
        - mAP (mean average precision)
        - auc (area under ROC curve)
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(-similarities)
    sorted_labels = labels[sorted_indices]
    
    # Total positives
    num_positives = labels.sum()
    
    if num_positives == 0:
        return {f"recall@{k}": 0.0 for k in k_values}
    
    metrics = {}
    
    # Recall@k and Precision@k
    for k in k_values:
        top_k_labels = sorted_labels[:k]
        hits = top_k_labels.sum()
        
        metrics[f"recall@{k}"] = float(hits / num_positives)
        metrics[f"precision@{k}"] = float(hits / k)
    
    # Mean Average Precision (mAP)
    precisions = []
    num_hits = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            precisions.append(precision_at_i)
    
    metrics["mAP"] = float(np.mean(precisions)) if precisions else 0.0
    
    # AUC (Area Under ROC Curve)
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auc"] = float(roc_auc_score(labels, similarities))
    except Exception:
        metrics["auc"] = 0.0
    
    return metrics


def compute_batch_accuracy(
    positive_similarities: torch.Tensor,
    negative_similarities: torch.Tensor,
) -> float:
    """
    Compute accuracy for a batch of samples.
    
    Accuracy = fraction of samples where positive has highest similarity.
    
    Args:
        positive_similarities: [B]
        negative_similarities: [B, num_neg]
        
    Returns:
        Accuracy as float
    """
    # Positive should have higher similarity than all negatives
    max_neg = negative_similarities.max(dim=1).values  # [B]
    correct = (positive_similarities > max_neg).float()
    return correct.mean().item()

