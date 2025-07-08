"""
Uncertainty estimation for inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Callable
from tqdm import tqdm

from utils.distributed import safe_get_rank


class UncertaintyEstimator:
    """Base class for uncertainty estimation methods."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def predict_with_uncertainty(self, data_loader, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        raise NotImplementedError


class MCDropoutPredictor(UncertaintyEstimator):
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, model: nn.Module, device: torch.device, num_samples: int = 30):
        super().__init__(model, device)
        self.num_samples = num_samples
    
    def predict_with_uncertainty(
        self, 
        data_loader, 
        preprocessing_pipeline=None,
        show_progress: bool = True,
        embedding_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with Monte Carlo Dropout uncertainty estimation.
        
        Args:
            data_loader: DataLoader with input data
            preprocessing_pipeline: Optional preprocessing pipeline
            show_progress: Whether to show progress bars
            embedding_callback: Optional callback for embedding extraction
            
        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        self.model.eval()
        
        # Enable dropout during inference
        self._enable_dropout()
        
        all_sample_predictions = []
        
        # Progress bar for MC samples
        sample_iterator = range(self.num_samples)
        if show_progress and (safe_get_rank() == 0):
            sample_iterator = tqdm(sample_iterator, desc="MC Dropout Samples")
        
        for sample_idx in sample_iterator:
            sample_predictions = self._single_forward_pass(
                data_loader, 
                embedding_callback if sample_idx == 0 else None  # Only extract embeddings on first pass
            )
            
            if len(sample_predictions) > 0:
                sample_array = np.concatenate(sample_predictions, axis=0)
                
                # Apply inverse preprocessing if needed
                if preprocessing_pipeline is not None:
                    sample_array = preprocessing_pipeline.inverse_transform(sample_array)
                
                all_sample_predictions.append(sample_array)
        
        if len(all_sample_predictions) > 0:
            # Stack predictions: [num_samples, num_molecules, num_tasks]
            stacked = np.stack(all_sample_predictions, axis=0)
            mean_preds = np.mean(stacked, axis=0)
            uncertainties = np.std(stacked, axis=0)
            return mean_preds, uncertainties
        
        return np.array([]), np.array([])
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        def enable_dropout_fn(module):
            if isinstance(module, nn.Dropout):
                module.train()
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.apply(enable_dropout_fn)
        else:
            self.model.apply(enable_dropout_fn)
    
    def _single_forward_pass(self, data_loader, embedding_callback: Optional[Callable] = None, show_progress: bool = False) -> List[np.ndarray]:
        """Perform a single forward pass through the data."""
        predictions = []
        
        # Add progress bar for MC dropout batches
        batch_iterator = data_loader
        if show_progress:
            batch_iterator = tqdm(
                data_loader, 
                desc="MC dropout batches", 
                unit="batch",
                leave=False
            )
        
        with torch.no_grad():
            for batch in batch_iterator:
                if batch is None:
                    continue
                
                # Extract embeddings if callback provided
                if embedding_callback is not None:
                    embedding_callback(batch)
                
                # Prepare batch
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(self.device)
                batch_indices = batch.batch_indices.to(self.device)
                batch_atom_features = {k: v.to(self.device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(self.device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(self.device)
                cis_indices = batch.final_cis_tensor.to(self.device)
                trans_indices = batch.final_trans_tensor.to(self.device)
                
                # Forward pass
                outputs, _, _ = self.model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                
                predictions.append(outputs.detach().cpu().numpy())
        
        return predictions


class DeterministicPredictor:
    """Standard deterministic prediction without uncertainty."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def predict(
        self, 
        data_loader, 
        preprocessing_pipeline=None,
        embedding_callback: Optional[Callable] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Make deterministic predictions.
        
        Args:
            data_loader: DataLoader with input data
            preprocessing_pipeline: Optional preprocessing pipeline
            embedding_callback: Optional callback for embedding extraction
            show_progress: Whether to show progress bars
            
        Returns:
            Array of predictions
        """
        self.model.eval()
        predictions = []
        
        # Add progress bar for batch processing
        batch_iterator = data_loader
        if show_progress:
            batch_iterator = tqdm(
                data_loader, 
                desc="Inference batches", 
                unit="batch",
                leave=False
            )
        
        with torch.no_grad():
            for batch in batch_iterator:
                if batch is None:
                    continue
                
                # Extract embeddings if callback provided
                if embedding_callback is not None:
                    embedding_callback(batch)
                
                # Prepare batch
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(self.device)
                batch_indices = batch.batch_indices.to(self.device)
                batch_atom_features = {k: v.to(self.device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(self.device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(self.device)
                cis_indices = batch.final_cis_tensor.to(self.device)
                trans_indices = batch.final_trans_tensor.to(self.device)
                
                # Forward pass
                outputs, _, _ = self.model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                
                predictions.append(outputs.detach().cpu().numpy())
        
        if predictions:
            all_preds = np.concatenate(predictions, axis=0)
            # Apply inverse preprocessing
            if preprocessing_pipeline is not None:
                all_preds = preprocessing_pipeline.inverse_transform(all_preds)
            return all_preds
        
        return np.array([])