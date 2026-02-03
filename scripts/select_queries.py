"""
Select representative query images using CLIP features and K-means clustering.
Visualizes clusters and selects more samples from larger clusters.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPProcessor
from tqdm import tqdm
from dotenv import load_dotenv

# 配置
ROOT_DIR = Path(__file__).resolve().parent.parent
CELL_DATA_DIR = "/ssddata/hjiangaz/rliuar/1_Research/CCS-Cell-Cls/CELL_DATA"
OUTPUT_DIR = str(ROOT_DIR / "data/queries")
CACHE_DIR = "/ssddata/hjiangaz/rliuar/0_Official/huggingface/hub"
CATEGORY = "LSIL"
NUM_CLUSTERS = 10
TOTAL_SAMPLES = 30
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def load_clip_model():
    print("Loading CLIP model...")
    model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16",
        cache_dir=CACHE_DIR,
        use_safetensors=True,
    )
    model = model.to(DEVICE).eval()
    
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch16",
        cache_dir=CACHE_DIR,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    return model, processor


def get_image_paths(category):
    category_dir = Path(CELL_DATA_DIR) / category
    extensions = {".png", ".jpg", ".jpeg"}
    paths = []
    for ext in extensions:
        paths.extend(category_dir.glob(f"*{ext}"))
        paths.extend(category_dir.glob(f"*{ext.upper()}"))
    return sorted([str(p) for p in paths])


@torch.no_grad()
def extract_features(model, processor, image_paths):
    print(f"Extracting features from {len(image_paths)} images...")
    features = []
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        cls_features = outputs.pooler_output  # [B, 768]
        features.append(cls_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def cluster_and_select(features, image_paths, num_clusters, total_samples):
    print(f"Clustering into {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # 统计每个 cluster 的大小
    cluster_sizes = np.bincount(labels)
    print(f"Cluster sizes: {cluster_sizes}")
    
    # 按 cluster 大小比例分配样本数
    proportions = cluster_sizes / cluster_sizes.sum()
    samples_per_cluster = np.round(proportions * total_samples).astype(int)
    
    # 确保每个 cluster 至少选 1 个（如果有样本的话）
    samples_per_cluster = np.maximum(samples_per_cluster, 1)
    
    # 调整总数
    while samples_per_cluster.sum() > total_samples:
        max_idx = samples_per_cluster.argmax()
        samples_per_cluster[max_idx] -= 1
    while samples_per_cluster.sum() < total_samples:
        # 给最大的 cluster 加一个
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        for idx in sorted_indices:
            if samples_per_cluster[idx] < cluster_sizes[idx]:
                samples_per_cluster[idx] += 1
                break
    
    print(f"Samples per cluster: {samples_per_cluster}")
    
    # 从每个 cluster 选最接近中心的样本
    selected_indices = []
    for cluster_id in range(num_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features[cluster_mask]
        
        # 计算到中心的距离
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        
        # 选最近的 n 个
        n_select = samples_per_cluster[cluster_id]
        nearest_indices = np.argsort(distances)[:n_select]
        selected_indices.extend(cluster_indices[nearest_indices])
    
    selected_paths = [image_paths[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]
    
    return labels, selected_indices, selected_paths, selected_labels, kmeans


def visualize_clusters(features, labels, selected_indices, output_path):
    print("Running t-SNE for visualization...")
    
    # 降采样以加速 t-SNE，但确保 selected_indices 都包含在内
    max_samples = 5000
    if len(features) > max_samples:
        selected_set = set(selected_indices)
        other_indices = [i for i in range(len(features)) if i not in selected_set]
        n_other = max_samples - len(selected_indices)
        sampled_other = np.random.choice(other_indices, n_other, replace=False)
        sample_idx = np.concatenate([np.array(selected_indices), sampled_other])
        sample_idx = np.sort(sample_idx)
        features_sampled = features[sample_idx]
        labels_sampled = labels[sample_idx]
        # 映射 selected_indices 到采样后的索引
        idx_map = {orig: new for new, orig in enumerate(sample_idx)}
        selected_in_sample = [idx_map[i] for i in selected_indices]
    else:
        features_sampled = features
        labels_sampled = labels
        selected_in_sample = selected_indices
        sample_idx = np.arange(len(features))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sampled)
    
    # 绘图
    plt.figure(figsize=(14, 10))
    
    # 绘制所有点
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels_sampled, cmap='tab10', alpha=0.5, s=10
    )
    
    # 高亮选中的点
    if selected_in_sample:
        plt.scatter(
            features_2d[selected_in_sample, 0],
            features_2d[selected_in_sample, 1],
            c='red', s=200, marker='*', edgecolors='black', linewidths=1,
            label='Selected samples'
        )
    
    plt.colorbar(scatter, label='Cluster')
    plt.legend(fontsize=12)
    plt.title(f'{CATEGORY} Cell Clustering (t-SNE)\nRed stars = selected representative samples', fontsize=14)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")


def visualize_selected_images(selected_paths, selected_labels, output_path):
    """Create a grid visualization of selected images."""
    n = len(selected_paths)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_2d(axes)
    
    for i, (path, label) in enumerate(zip(selected_paths, selected_labels)):
        row, col = i // cols, i % cols
        img = Image.open(path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Cluster {label}\n{Path(path).name[:20]}...", fontsize=8)
        axes[row, col].axis('off')
    
    # 隐藏空白子图
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Selected {CATEGORY} Query Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved selected images to {output_path}")


def main():
    # 加载模型
    model, processor = load_clip_model()
    
    # 获取图片路径
    image_paths = get_image_paths(CATEGORY)
    print(f"Found {len(image_paths)} {CATEGORY} images")
    
    # 提取特征
    features = extract_features(model, processor, image_paths)
    print(f"Features shape: {features.shape}")
    
    # 聚类并选择
    labels, selected_indices, selected_paths, selected_labels, kmeans = cluster_and_select(
        features, image_paths, NUM_CLUSTERS, TOTAL_SAMPLES
    )
    
    # 创建输出目录
    output_category_dir = Path(OUTPUT_DIR) / CATEGORY
    output_category_dir.mkdir(parents=True, exist_ok=True)
    
    vis_category_dir = Path(OUTPUT_DIR) / "visualization" / CATEGORY
    vis_category_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化（保存到 visualization 目录）
    visualize_clusters(
        features, labels, selected_indices,
        vis_category_dir / "cluster_visualization.png"
    )
    
    visualize_selected_images(
        selected_paths, selected_labels,
        vis_category_dir / "selected_images.png"
    )
    
    # 复制选中的图片（保存到 query 目录）
    print(f"\nCopying selected images to {output_category_dir}")
    for i, path in enumerate(selected_paths):
        src = Path(path)
        dst = output_category_dir / f"query_{i:02d}_{src.name}"
        shutil.copy2(src, dst)
        print(f"  [{i+1}/{len(selected_paths)}] Cluster {selected_labels[i]}: {src.name}")
    
    print(f"\nDone! Selected {len(selected_paths)} representative images.")
    print(f"Use them with: --query_images all --queries_dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

