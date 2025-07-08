"""
Models package for AIMNet-X2D.

This package contains all model-related components organized by functionality.
Maintains backward compatibility with the original model.py imports.
"""

# Import all classes for backward compatibility
from .gnn import GNN
from .layers import ShellConvolutionLayer
from .pooling import (
    MeanPoolingLayer,
    MaxPoolingLayer, 
    SumPoolingLayer,
    MultiHeadAttentionPoolingLayer
)
from .losses import (
    WeightedL1Loss, 
    WeightedMSELoss, 
    EvidentialLoss, 
    WeightedEvidentialLoss
)

# Export everything that's commonly used
__all__ = [
    # Main model
    "GNN",
    
    # Layers
    "ShellConvolutionLayer",
    
    # Pooling
    "MeanPoolingLayer",
    "MaxPoolingLayer", 
    "SumPoolingLayer",
    "MultiHeadAttentionPoolingLayer",
    
    # Loss functions
    "WeightedL1Loss",
    "WeightedMSELoss",
    "EvidentialLoss",
    "WeightedEvidentialLoss",
]