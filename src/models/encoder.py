"""
ViT Encoder based on CLIP's Vision Transformer, SigLIP2, or DINOv3.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig
from dotenv import load_dotenv

# Default HuggingFace cache directory (should be overridden via config)
DEFAULT_CACHE_DIR = None
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Supported backbones
CLIP_BACKBONES = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
]

SIGLIP2_BACKBONES = [
    "google/siglip2-base-patch16-224",
    "google/siglip2-base-patch16-256",
    "google/siglip2-base-patch16-384",
    "google/siglip2-base-patch16-512",
    "google/siglip2-large-patch16-256",
    "google/siglip2-large-patch16-384",
    "google/siglip2-so400m-patch14-384",
]

DINOV3_BACKBONES = [
    # HuggingFace Hub models
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vitg16-pretrain-lvd1689m",
    "facebook/dinov3-vitsplus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    # Local paths are also supported - just specify the full path containing config.json and model.safetensors
]


def is_siglip_model(model_name: str) -> bool:
    """Check if model name is a SigLIP or SigLIP2 model."""
    return "siglip" in model_name.lower()


def is_dinov3_model(model_name: str) -> bool:
    """Check if model name is a DINOv3 model (either HuggingFace ID or local path)."""
    return "dinov3" in model_name.lower()


def is_local_path(model_name: str) -> bool:
    """Check if model_name is a local path (absolute or relative)."""
    return os.path.isdir(model_name)


class SigLIPViTEncoder(nn.Module):
    """
    SigLIP/SigLIP2 Vision encoder that outputs both CLS token and patch tokens.
    
    Uses AutoModel for automatic model type detection.
    SigLIP models don't have a CLS token by default, so we use pooler_output
    or mean pooling of patch tokens as the global embedding.
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        freeze: bool = False,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize SigLIP/SigLIP2 ViT encoder.
        
        Args:
            model_name: HuggingFace model name (e.g., google/siglip2-large-patch16-256)
            freeze: Whether to freeze encoder weights
            cache_dir: Directory to cache downloaded model weights
        """
        super().__init__()
        
        from transformers import AutoModel
        
        # Ensure cache directory exists
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Use AutoModel which automatically detects the correct model class
        full_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            token=HF_TOKEN,
        )
        
        # Extract vision model from the full model
        if hasattr(full_model, 'vision_model'):
            self.vision_model = full_model.vision_model
            self.config = full_model.vision_model.config
        else:
            # If it's already a vision-only model
            self.vision_model = full_model
            self.config = full_model.config
        
        # Get dimensions
        self.hidden_size = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        
        # Calculate number of patches (SigLIP doesn't have CLS token)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            return_patch_tokens: Whether to return patch tokens
            
        Returns:
            dict with:
            - cls_token: Pooled embedding (mean of patch tokens) [B, hidden_size]
            - patch_tokens: (optional) Patch token embeddings [B, num_patches, hidden_size]
            - last_hidden_state: Full hidden states [B, num_patches, hidden_size]
        """
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # SigLIP: last_hidden_state is [B, num_patches, hidden_size] (no CLS token)
        last_hidden_state = outputs.last_hidden_state
        
        # Use pooler_output if available, otherwise mean pool patch tokens
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_token = outputs.pooler_output
        else:
            # Mean pooling over patch tokens
            cls_token = last_hidden_state.mean(dim=1)  # [B, hidden_size]
        
        result = {
            "cls_token": cls_token,
            "last_hidden_state": last_hidden_state,
        }
        
        if return_patch_tokens:
            # All tokens are patch tokens in SigLIP
            result["patch_tokens"] = last_hidden_state
        
        return result
    
    def get_cls_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get only CLS token embedding (pooled output)."""
        return self.forward(pixel_values)["cls_token"]
    
    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get patch token embeddings."""
        return self.forward(pixel_values, return_patch_tokens=True)["patch_tokens"]


class DINOv3ViTEncoder(nn.Module):
    """
    DINOv3 Vision encoder that outputs both CLS token and patch tokens.
    
    DINOv3 produces high-quality dense features and includes:
    - CLS token: global image embedding at position 0
    - Register tokens: learnable tokens that reduce artifacts (positions 1 to num_register_tokens)
    - Patch tokens: local patch embeddings (remaining positions)
    
    Supports loading from:
    - HuggingFace Hub: "facebook/dinov3-vitb16-pretrain-lvd1689m"
    - Local path: "/path/to/dinov3-vitb16-pretrain-lvd1689m"
    
    Reference: https://huggingface.co/docs/transformers/model_doc/dinov3
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze: bool = False,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize DINOv3 ViT encoder.
        
        Args:
            model_name: HuggingFace model name or local path to model directory
                        (e.g., "facebook/dinov3-vitb16-pretrain-lvd1689m" or 
                         "/path/to/dinov3-vitb16-pretrain-lvd1689m")
            freeze: Whether to freeze encoder weights
            cache_dir: Directory to cache downloaded model weights (ignored for local paths)
        """
        super().__init__()
        
        from transformers import AutoModel
        
        # Check if loading from local path or HuggingFace Hub
        loading_from_local = is_local_path(model_name)
        
        if loading_from_local:
            print(f"Loading DINOv3 from local path: {model_name}")
            # Load from local path - no token or cache_dir needed
            self.vision_model = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True,
                attn_implementation="sdpa",  # DINOv3 uses SDPA attention
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            # Ensure cache directory exists for HuggingFace downloads
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Load from HuggingFace Hub
            self.vision_model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_safetensors=True,
                attn_implementation="sdpa",  # DINOv3 uses SDPA attention
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN,
            )
        
        self.config = self.vision_model.config
        
        # Get dimensions
        self.hidden_size = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size
        
        # Get number of register tokens (DINOv3 specific)
        self.num_register_tokens = getattr(self.config, 'num_register_tokens', 0)
        
        # Calculate number of patches (excluding CLS and register tokens)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            return_patch_tokens: Whether to return patch tokens
            
        Returns:
            dict with:
            - cls_token: CLS token embeddings [B, hidden_size]
            - patch_tokens: (optional) Patch token embeddings [B, num_patches, hidden_size]
            - last_hidden_state: Full hidden states [B, 1 + num_register_tokens + num_patches, hidden_size]
        """
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # DINOv3 output structure:
        # last_hidden_state: [B, 1 + num_register_tokens + num_patches, hidden_size]
        # Position 0: CLS token
        # Positions 1 to num_register_tokens: register tokens
        # Remaining positions: patch tokens
        last_hidden_state = outputs.last_hidden_state
        
        # Extract CLS token (first position)
        cls_token = last_hidden_state[:, 0, :]  # [B, hidden_size]
        
        # Alternatively, use pooler_output if available
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_token = outputs.pooler_output
        
        result = {
            "cls_token": cls_token,
            "last_hidden_state": last_hidden_state,
        }
        
        if return_patch_tokens:
            # Extract patch tokens (skip CLS and register tokens)
            # Patch tokens start at position 1 + num_register_tokens
            patch_start_idx = 1 + self.num_register_tokens
            patch_tokens = last_hidden_state[:, patch_start_idx:, :]  # [B, num_patches, hidden_size]
            result["patch_tokens"] = patch_tokens
        
        return result
    
    def get_cls_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get only CLS token embedding."""
        return self.forward(pixel_values)["cls_token"]
    
    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get patch token embeddings (excluding CLS and register tokens)."""
        return self.forward(pixel_values, return_patch_tokens=True)["patch_tokens"]


class CLIPViTEncoder(nn.Module):
    """
    CLIP ViT-B/16 encoder that outputs both CLS token and patch tokens.
    
    For query images: use CLS token as the global embedding
    For patch images: use patch tokens for spatial attention
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = False,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize CLIP ViT encoder.
        
        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze encoder weights
            cache_dir: Directory to cache downloaded model weights
        """
        super().__init__()
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
        )
        self.config = self.vision_model.config
        
        # Get dimensions
        self.hidden_size = self.config.hidden_size  # 768 for ViT-B
        self.patch_size = self.config.patch_size    # 16
        self.image_size = self.config.image_size    # 224
        
        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 196 for 224x224
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            return_patch_tokens: Whether to return patch tokens
            
        Returns:
            dict with:
            - cls_token: CLS token embeddings [B, hidden_size]
            - patch_tokens: (optional) Patch token embeddings [B, num_patches, hidden_size]
            - last_hidden_state: Full hidden states [B, 1+num_patches, hidden_size]
        """
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get last hidden state: [B, 1 + num_patches, hidden_size]
        # First token is CLS, rest are patch tokens
        last_hidden_state = outputs.last_hidden_state
        
        cls_token = last_hidden_state[:, 0, :]  # [B, hidden_size]
        
        result = {
            "cls_token": cls_token,
            "last_hidden_state": last_hidden_state,
        }
        
        if return_patch_tokens:
            patch_tokens = last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_size]
            result["patch_tokens"] = patch_tokens
        
        return result
    
    def get_cls_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get only CLS token embedding."""
        return self.forward(pixel_values)["cls_token"]
    
    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get patch token embeddings."""
        return self.forward(pixel_values, return_patch_tokens=True)["patch_tokens"]


def create_encoder(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR, freeze: bool = False):
    """
    Factory function to create the appropriate encoder based on model name.
    
    Args:
        model_name: HuggingFace model name (CLIP, SigLIP/SigLIP2, or DINOv3)
        cache_dir: Directory to cache downloaded model weights
        freeze: Whether to freeze encoder weights
        
    Returns:
        Encoder module (CLIPViTEncoder, SigLIPViTEncoder, or DINOv3ViTEncoder)
    """
    if is_dinov3_model(model_name):
        return DINOv3ViTEncoder(model_name, freeze=freeze, cache_dir=cache_dir)
    elif is_siglip_model(model_name):
        return SigLIPViTEncoder(model_name, freeze=freeze, cache_dir=cache_dir)
    else:
        return CLIPViTEncoder(model_name, freeze=freeze, cache_dir=cache_dir)


class DualViTEncoder(nn.Module):
    """
    Dual encoder with separate ViTs for query and patch.
    Supports CLIP, SigLIP/SigLIP2, and DINOv3 backbones.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize dual encoder.
        
        Args:
            model_name: HuggingFace model name (CLIP, SigLIP/SigLIP2, or DINOv3)
            cache_dir: Directory to cache downloaded model weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.is_siglip = is_siglip_model(model_name)
        self.is_dinov3 = is_dinov3_model(model_name)
        
        self.query_encoder = create_encoder(model_name, cache_dir=cache_dir)
        self.patch_encoder = create_encoder(model_name, cache_dir=cache_dir)
        
        self.hidden_size = self.query_encoder.hidden_size
    
    def encode_query(self, query_images: torch.Tensor) -> torch.Tensor:
        """
        Encode query images to CLS embeddings.
        
        Args:
            query_images: Query images [B, 3, H, W]
            
        Returns:
            CLS embeddings [B, hidden_size]
        """
        return self.query_encoder.get_cls_embedding(query_images)
    
    def encode_patches(self, patch_images: torch.Tensor) -> torch.Tensor:
        """
        Encode patch images to patch token embeddings.
        
        Args:
            patch_images: Patch images [B, 3, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, hidden_size]
        """
        return self.patch_encoder.get_patch_embeddings(patch_images)
    
    def forward(
        self,
        query_images: torch.Tensor,
        patch_images: torch.Tensor,
    ) -> dict:
        """
        Forward pass for both query and patch images.
        
        Args:
            query_images: Query images [B, 3, H, W]
            patch_images: Patch images [B, 3, H, W] or [B, N, 3, H, W]
            
        Returns:
            dict with query_embeddings and patch_embeddings
        """
        query_embeddings = self.encode_query(query_images)
        
        # Handle batched patches (for multiple negatives)
        if patch_images.dim() == 5:
            B, N, C, H, W = patch_images.shape
            patch_images_flat = patch_images.view(B * N, C, H, W)
            patch_embeddings_flat = self.encode_patches(patch_images_flat)
            # [B*N, num_patches, hidden_size] -> [B, N, num_patches, hidden_size]
            num_patches = patch_embeddings_flat.shape[1]
            hidden_size = patch_embeddings_flat.shape[2]
            patch_embeddings = patch_embeddings_flat.view(B, N, num_patches, hidden_size)
        else:
            patch_embeddings = self.encode_patches(patch_images)
        
        return {
            "query_embeddings": query_embeddings,
            "patch_embeddings": patch_embeddings,
        }


# Alias for backward compatibility
DualCLIPViTEncoder = DualViTEncoder

