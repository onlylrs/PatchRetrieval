"""
Stain Normalization for pathology images.
Supports Macenko and Reinhard methods.
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional, Union


def _is_torch_available():
    """Check if torchstain is available."""
    try:
        import torchstain
        return True
    except ImportError:
        return False


def _is_staintools_available():
    """Check if staintools is available."""
    try:
        import staintools
        return True
    except ImportError:
        return False


class StainNormalizer:
    """
    Stain normalization wrapper supporting multiple backends.
    
    Supports:
    - torchstain (PyTorch, GPU-accelerated, preferred)
    - staintools (NumPy, CPU-only, fallback)
    """
    
    def __init__(
        self,
        method: str = "macenko",
        target_image_path: Optional[str] = None,
        backend: str = "auto",
    ):
        """
        Initialize stain normalizer.
        
        Args:
            method: Normalization method ("macenko" or "reinhard")
            target_image_path: Path to target/reference image (if None, uses first image)
            backend: "torchstain", "staintools", or "auto"
        """
        self.method = method.lower()
        self.target_image_path = target_image_path
        
        # Select backend
        if backend == "auto":
            if _is_torch_available():
                self.backend = "torchstain"
            elif _is_staintools_available():
                self.backend = "staintools"
            else:
                raise ImportError(
                    "No stain normalization library found. "
                    "Install torchstain (pip install torchstain) or "
                    "staintools (pip install staintools)"
                )
        else:
            self.backend = backend
        
        print(f"Using stain normalization backend: {self.backend}")
        
        # Initialize normalizer
        self.normalizer = None
        self._initialize_normalizer()
        
        # Fit to target if provided
        if target_image_path:
            self.fit_target(target_image_path)
    
    def _initialize_normalizer(self):
        """Initialize the normalizer based on backend."""
        if self.backend == "torchstain":
            import torchstain
            if self.method == "macenko":
                self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
            elif self.method == "reinhard":
                self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        elif self.backend == "staintools":
            import staintools
            if self.method == "macenko":
                self.normalizer = staintools.StainNormalizer(method='macenko')
            elif self.method == "reinhard":
                self.normalizer = staintools.StainNormalizer(method='reinhard')
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def fit_target(self, image_path: str):
        """Fit normalizer to target image."""
        target = Image.open(image_path).convert("RGB")
        target_np = np.array(target)
        
        if self.backend == "torchstain":
            target_tensor = torch.from_numpy(target_np).float()
            self.normalizer.fit(target_tensor)
        else:
            self.normalizer.fit(target_np)
    
    def normalize(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """
        Normalize an image.
        
        Args:
            image: PIL Image, numpy array, or torch tensor
            
        Returns:
            Normalized PIL Image
        """
        if self.normalizer is None:
            # If not fitted, fit on first image
            if isinstance(image, Image.Image):
                img_np = np.array(image)
            elif isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
            else:
                img_np = image
            
            if self.backend == "torchstain":
                img_tensor = torch.from_numpy(img_np).float()
                self.normalizer.fit(img_tensor)
            else:
                self.normalizer.fit(img_np)
        
        # Convert input to appropriate format
        if isinstance(image, Image.Image):
            img_input = np.array(image)
        elif isinstance(image, torch.Tensor):
            img_input = image.cpu().numpy()
        else:
            img_input = image
        
        # Normalize
        try:
            if self.backend == "torchstain":
                img_tensor = torch.from_numpy(img_input).float()
                normalized_tensor, _, _ = self.normalizer.normalize(img_tensor)
                normalized_np = normalized_tensor.cpu().numpy().astype(np.uint8)
            else:
                normalized_np = self.normalizer.transform(img_input)
            
            return Image.fromarray(normalized_np)
        
        except Exception as e:
            # If normalization fails (e.g., too dark image), return original
            print(f"Warning: Stain normalization failed: {e}. Returning original image.")
            return Image.fromarray(img_input)


class StainAugmentation:
    """
    Stain augmentation: randomly perturb stain concentrations.
    Useful for training robustness.
    """
    
    def __init__(self, sigma1: float = 0.2, sigma2: float = 0.2):
        """
        Args:
            sigma1: Std of Gaussian noise for H stain
            sigma2: Std of Gaussian noise for E stain
        """
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        if not _is_staintools_available():
            raise ImportError("StainAugmentation requires staintools")
        
        import staintools
        self.augmentor = staintools.StainAugmentor(
            method='vahadane',
            sigma1=sigma1,
            sigma2=sigma2
        )
    
    def augment(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Augment image with random stain perturbation."""
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        try:
            augmented = self.augmentor.augment(img_np)
            return Image.fromarray(augmented)
        except Exception as e:
            print(f"Warning: Stain augmentation failed: {e}")
            return Image.fromarray(img_np)
