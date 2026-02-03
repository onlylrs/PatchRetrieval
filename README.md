# PatchRetrieval (AI generated Readme, will update later)

PatchRetrieval 是一个用于病理图像（特别是宫颈细胞学 CCS 数据集）的图像块检索（Patch Retrieval）项目。它利用深度学习模型（如 DINOv3, CLIP, SigLIP2）作为特征提取器，通过对比学习或相似度计算，从大规模切片或图像库中检索与查询图像（Query Images）相似的图像块。

## 目录结构

```
PatchRetrieval/
├── configs/          # 配置文件目录
│   └── default.yaml  # 默认配置文件（数据路径、模型参数、训练参数）
├── data/             # 数据目录（包含处理后的数据和 split 信息）
├── scripts/          # 运行脚本
│   ├── train.py              # 训练检索模型
│   ├── inference.py          # 单文件夹 Patch 推理
│   ├── wsi_inference.py      # 全切片图像 (WSI) 推理
│   ├── batch_wsi_inference.py # 批量 WSI 推理
│   └── evaluate.py           # 模型评估
└── src/              # 源代码
    ├── datasets/     # 数据集加载与预处理
    ├── models/       # 模型定义 (Encoder, RetrievalModel)
    └── utils/        # 工具函数 (Loss, Metrics)
```

## 环境依赖

项目依赖以下主要 Python 库：
- python >= 3.8
- pytorch
- torchvision
- numpy
- pyyaml
- tqdm
- tensorboard
- pillow

请确保您的环境已正确配置。

## 配置 (Configuration)

主要配置文件位于 `configs/default.yaml`。在运行脚本前，请根据您的环境修改以下关键参数：

- `data.cell_cls_root`: 细胞分类数据根目录
- `data.cell_det_root`: 细胞检测数据根目录
- `model.backbone`: 使用的预训练模型骨干 (例如 "facebook/dinov3-vitb16-pretrain-lvd1689m" 或本地路径)
- `training`: 训练超参数 (batch_size, learning_rate 等)

## 使用方法 (Usage)

### 1. 训练 (Training)

使用 `scripts/train.py` 训练模型。需要指定目标类别 (`--category`) 和查询图像 (`--query_images`)。

**示例：**

```bash
python scripts/train.py \
    --category HSIL \
    --query_images all \
    --queries_dir /path/to/queries \
    --batch_size 48 \
    --num_epochs 10
```

**参数说明：**
- `--category`: 目标类别 (e.g., ASC-US, LSIL, ASC-H, HSIL, SCC)
- `--query_images`: 查询图像路径列表，或者使用 'all' 加载目录下所有图像
- `--queries_dir`: 当 `--query_images` 为 'all' 时，指定查询图像的根目录
- `--config`: 配置文件路径 (默认: `configs/default.yaml`)

### 2. 推理 (Inference)

#### 针对 Patch 文件夹推理

使用 `scripts/inference.py` 对包含 Patch 的文件夹进行检索推理。

```bash
python scripts/inference.py \
    --checkpoint experiments/best_model.pth \
    --patch_dir /path/to/patches \
    --query_images /path/to/query.jpg \
    --output_dir results/inference
```

#### 针对 WSI (全切片) 推理

使用 `scripts/wsi_inference.py` 或 `scripts/batch_wsi_inference.py`。

```bash
python scripts/wsi_inference.py \
    --wsi_path /path/to/slide.svs \
    --checkpoint experiments/best_model.pth \
    ...
```

### 3. 评估 (Evaluation)

使用 `scripts/evaluate.py` 评估训练好的模型性能。

```bash
python scripts/evaluate.py \
    --checkpoint experiments/best_model.pth \
    --test_data /path/to/test_data
```

## 模型说明

项目支持多种 Vision Transformer 骨干网络作为特征提取器，可以在 `config.yaml` 中配置 `model.backbone`。支持的模型包括：
- **DINOv3**: Facebook 的自监督 ViT 模型
- **CLIP**: OpenAI 的图文预训练模型
- **SigLIP**: Google 的 Sigmoid Loss 用于语言图像预训练的模型

模型采用对比学习或相似度加权的方式来拉近 Query 和正样本 Patch 的距离，推远负样本。
