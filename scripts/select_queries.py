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
CELL_DATA_DIR = "/home/rliuar/2_Official/CCS-Cell-Cls/CELL_DATA"
OUTPUT_DIR = str(ROOT_DIR / "data/queries/split")
CACHE_DIR = "/home/rliuar/0_Storage/hf_home/hub"
CATEGORY = "HSIL"

# Model selection: "clip" or "dinov3"
MODEL_TYPE = "dinov3"  # Options: "clip", "dinov3"
MODEL_NAME = {
    "clip": "openai/clip-vit-base-patch16",
    "dinov3": "/home/rliuar/2_Official/dinov3-vitb16-pretrain-lvd1689m",
}

# Query split configuration
NUM_CLUSTERS = 8
TRAIN_SAMPLES = 2000  # Number of train query images
VAL_SAMPLES = 20     # Number of val query images
TEST_SAMPLES = 20    # Number of test query images
TOTAL_SAMPLES = TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES

# Selection strategy: "nearest" (all from center), "random" (all random), "hybrid" (half-half)
SELECTION_STRATEGY = "hybrid"  # Options: "nearest", "random", "hybrid"

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def load_model():
    """Load feature extraction model (CLIP or DINOv3)."""
    model_name = MODEL_NAME[MODEL_TYPE]
    print(f"Loading {MODEL_TYPE.upper()} model from {model_name}...")
    
    if MODEL_TYPE == "clip":
        model = CLIPVisionModel.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            use_safetensors=True,
        )
        model = model.to(DEVICE).eval()
        
        processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            use_safetensors=True,
            token=HF_TOKEN,
        )
        return model, processor
    
    elif MODEL_TYPE == "dinov3":
        from transformers import AutoModel, AutoImageProcessor
        
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
        )
        model = model.to(DEVICE).eval()
        
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
        )
        return model, processor
    
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Use 'clip' or 'dinov3'.")


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
    print(f"Extracting features from {len(image_paths)} images using {MODEL_TYPE.upper()}...")
    features = []
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        
        if MODEL_TYPE == "clip":
            cls_features = outputs.pooler_output  # [B, 768]
        elif MODEL_TYPE == "dinov3":
            cls_features = outputs.last_hidden_state[:, 0, :]  # CLS token [B, 768]
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
        
        features.append(cls_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def cluster_and_select(features, image_paths, num_clusters, train_samples, val_samples, test_samples):
    """
    Cluster images and select train/val/test samples.
    Returns separate sets for train, val, test.
    """
    total_samples = train_samples + val_samples + test_samples
    print(f"Clustering into {num_clusters} clusters...")
    print(f"Selecting {train_samples} train + {val_samples} val + {test_samples} test = {total_samples} total")
    
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
    print(f"Selection strategy: {SELECTION_STRATEGY}")
    
    # 从每个 cluster 选样本
    all_selected_indices = []
    for cluster_id in range(num_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features[cluster_mask]
        
        # 计算到中心的距离
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        
        n_select = samples_per_cluster[cluster_id]
        
        if SELECTION_STRATEGY == "nearest":
            # 全部选离中心最近的
            nearest_local_indices = np.argsort(distances)[:n_select]
            selected_local_indices = nearest_local_indices
        
        elif SELECTION_STRATEGY == "random":
            # 全部随机选择
            selected_local_indices = np.random.choice(len(cluster_indices), n_select, replace=False)
        
        elif SELECTION_STRATEGY == "hybrid":
            # 一半选最近的，一半随机选
            n_nearest = n_select // 2
            n_random = n_select - n_nearest
            
            # 选最近的一半
            nearest_local_indices = np.argsort(distances)[:n_nearest]
            
            # 从剩余样本中随机选另一半
            remaining_mask = np.ones(len(cluster_indices), dtype=bool)
            remaining_mask[nearest_local_indices] = False
            remaining_local_indices = np.where(remaining_mask)[0]
            
            if len(remaining_local_indices) >= n_random:
                random_local_indices = np.random.choice(remaining_local_indices, n_random, replace=False)
            else:
                # 如果剩余样本不够，全选
                random_local_indices = remaining_local_indices
            
            selected_local_indices = np.concatenate([nearest_local_indices, random_local_indices])
        
        else:
            raise ValueError(f"Unknown SELECTION_STRATEGY: {SELECTION_STRATEGY}")
        
        # 转换为全局索引
        selected_global_indices = cluster_indices[selected_local_indices]
        all_selected_indices.extend(selected_global_indices)
    
    # 随机打乱并划分为train/val/test
    np.random.seed(42)
    np.random.shuffle(all_selected_indices)
    
    train_indices = all_selected_indices[:train_samples]
    val_indices = all_selected_indices[train_samples:train_samples+val_samples]
    test_indices = all_selected_indices[train_samples+val_samples:]
    
    def get_split_data(indices):
        paths = [image_paths[i] for i in indices]
        split_labels = [labels[i] for i in indices]
        return indices, paths, split_labels
    
    train_data = get_split_data(train_indices)
    val_data = get_split_data(val_indices)
    test_data = get_split_data(test_indices)
    
    return labels, train_data, val_data, test_data, kmeans, all_selected_indices


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
    print("="*60)
    print(f"Query Selection Configuration:")
    print(f"  Category: {CATEGORY}")
    print(f"  Model: {MODEL_TYPE.upper()} ({MODEL_NAME[MODEL_TYPE]})")
    print(f"  Clusters: {NUM_CLUSTERS}")
    print(f"  Selection: {SELECTION_STRATEGY}")
    print(f"  Train samples: {TRAIN_SAMPLES}")
    print(f"  Val samples: {VAL_SAMPLES}")
    print(f"  Test samples: {TEST_SAMPLES}")
    print(f"  Total: {TOTAL_SAMPLES}")
    print("="*60 + "\n")
    
    # 加载模型
    model, processor = load_model()
    
    # 获取图片路径
    image_paths = get_image_paths(CATEGORY)
    print(f"Found {len(image_paths)} {CATEGORY} images")
    
    # 提取特征
    features = extract_features(model, processor, image_paths)
    print(f"Features shape: {features.shape}")
    
    # 聚类并选择train/val/test
    labels, train_data, val_data, test_data, kmeans, all_selected_indices = cluster_and_select(
        features, image_paths, NUM_CLUSTERS, TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES
    )
    
    train_indices, train_paths, train_labels = train_data
    val_indices, val_paths, val_labels = val_data
    test_indices, test_paths, test_labels = test_data
    
    # 创建输出目录结构
    output_category_dir = Path(OUTPUT_DIR) / CATEGORY
    for split in ["train", "val", "test"]:
        (output_category_dir / split).mkdir(parents=True, exist_ok=True)
    
    vis_category_dir = Path(OUTPUT_DIR) / "visualization" / CATEGORY
    vis_category_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化（保存到 visualization 目录）
    print("\nGenerating visualizations...")
    visualize_clusters(
        features, labels, all_selected_indices,
        vis_category_dir / f"cluster_visualization_{MODEL_TYPE}.png"
    )
    
    # 分别可视化train/val/test
    for split_name, split_paths, split_labels in [("train", train_paths, train_labels),
                                                     ("val", val_paths, val_labels),
                                                     ("test", test_paths, test_labels)]:
        visualize_selected_images(
            split_paths, split_labels,
            vis_category_dir / f"selected_{split_name}_{MODEL_TYPE}.png"
        )
    
    # 复制选中的图片到对应的train/val/test子目录
    print(f"\nCopying selected images to {output_category_dir}")
    
    for split_name, split_paths, split_labels in [("train", train_paths, train_labels),
                                                     ("val", val_paths, val_labels),
                                                     ("test", test_paths, test_labels)]:
        split_dir = output_category_dir / split_name
        print(f"\n{split_name.upper()} set ({len(split_paths)} images):")
        for i, (path, label) in enumerate(zip(split_paths, split_labels)):
            src = Path(path)
            dst = split_dir / f"query_{i:02d}_cluster{label}_{src.name}"
            shutil.copy2(src, dst)
            print(f"  [{i+1}/{len(split_paths)}] Cluster {label}: {src.name}")
    
    print(f"\n{'='*60}")
    print(f"Done! Selected {TOTAL_SAMPLES} representative images:")
    print(f"  Train: {len(train_paths)} images in {output_category_dir}/train/")
    print(f"  Val: {len(val_paths)} images in {output_category_dir}/val/")
    print(f"  Test: {len(test_paths)} images in {output_category_dir}/test/")
    print(f"\nVisualizations saved to {vis_category_dir}/")
    print(f"Model used: {MODEL_TYPE.upper()}")
    print(f"Selection strategy: {SELECTION_STRATEGY}")
    print("="*60)


if __name__ == "__main__":
    main()

