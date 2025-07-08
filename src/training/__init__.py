"""
Training package for AIMNet-X2D.

This package contains all training-related functionality organized by purpose.
Maintains backward compatibility with the original training.py imports.
"""

# Main training interface - import everything for backward compatibility
from .trainer import train_gnn
from .evaluator import evaluate
from .predictor import predict_gnn, predict_with_mc_dropout, predict_gnn_with_smiles
from .extractors import (
    extract_all_embeddings, 
    save_embeddings_to_hdf5, 
    extract_partial_charges, 
    extract_embeddings_main
)

__all__ = [
    # Training
    "train_gnn",
    
    # Evaluation
    "evaluate", 
    
    # Prediction
    "predict_gnn",
    "predict_with_mc_dropout",
    "predict_gnn_with_smiles",
    
    # Extraction
    "extract_all_embeddings",
    "save_embeddings_to_hdf5", 
    "extract_partial_charges",
    "extract_embeddings_main",
]