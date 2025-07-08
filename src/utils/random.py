"""
Random seed utilities for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)            # Python random
    np.random.seed(seed)         # NumPy
    torch.manual_seed(seed)      # PyTorch (CPU)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False