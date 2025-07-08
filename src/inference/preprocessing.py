"""
Preprocessing pipeline reconstruction for inference.
"""

import numpy as np
from typing import Optional, Dict, Any

from data.preprocessing import PreprocessingPipeline, PreprocessingConfig, SAENormalizer, StandardScaler


class PreprocessingReconstructor:
    """Reconstructs preprocessing pipeline from saved model artifacts."""
    
    @staticmethod
    def load_preprocessing_pipeline(model_artifact: Dict[str, Any]) -> Optional[PreprocessingPipeline]:
        """
        Reconstruct preprocessing pipeline from saved model artifact.
        
        Args:
            model_artifact: Dictionary containing model state and hyperparameters
            
        Returns:
            PreprocessingPipeline or None if no preprocessing was used
        """
        hyperparams = model_artifact["hyperparams"]
        
        # Check if preprocessing config exists
        preprocessing_info = hyperparams.get("preprocessing_config")
        if not preprocessing_info:
            return PreprocessingReconstructor._load_legacy_format(hyperparams)
        
        # Reconstruct full preprocessing config
        config = PreprocessingConfig(
            apply_sae=preprocessing_info.get("apply_sae", False),
            sae_subtasks=preprocessing_info.get("sae_subtasks"),
            apply_standard_scaling=preprocessing_info.get("apply_standard_scaling", True),
            task_type=preprocessing_info.get("task_type", "regression"),
            sae_percentile_cutoff=preprocessing_info.get("sae_percentile_cutoff", 2.0)
        )
        
        # Create pipeline
        pipeline = PreprocessingPipeline(config)
        
        # Restore SAE normalizer if it was used
        if config.apply_sae and "sae_statistics" in hyperparams and hyperparams["sae_statistics"]:
            pipeline.sae_normalizer = SAENormalizer(
                task_type=config.task_type,
                percentile_cutoff=config.sae_percentile_cutoff
            )
            pipeline.sae_normalizer.sae_statistics = hyperparams["sae_statistics"]
            pipeline.sae_normalizer.is_fitted = True
            print(f"[Preprocessing] Restored SAE normalizer with {len(hyperparams['sae_statistics'])} task(s)")
        
        # Restore standard scaler if it was used
        if config.apply_standard_scaling and "scaler_means" in hyperparams and hyperparams["scaler_means"]:
            pipeline.standard_scaler = StandardScaler()
            pipeline.standard_scaler.means = np.array(hyperparams["scaler_means"])
            pipeline.standard_scaler.stds = np.array(hyperparams["scaler_stds"])
            pipeline.standard_scaler.is_fitted = True
            print(f"[Preprocessing] Restored standard scaler: means={pipeline.standard_scaler.means}, stds={pipeline.standard_scaler.stds}")
        
        pipeline.is_fitted = True
        return pipeline
    
    @staticmethod
    def _load_legacy_format(hyperparams: Dict[str, Any]) -> Optional[PreprocessingPipeline]:
        """Load legacy preprocessing format."""
        # Check for legacy format (old scaler format)
        if "scaler_means" in hyperparams and hyperparams["scaler_means"] is not None:
            print("[Preprocessing] Detected legacy model format, reconstructing standard scaler only")
            # Create minimal config for standard scaling only
            config = PreprocessingConfig(
                apply_sae=False,
                sae_subtasks=None,
                apply_standard_scaling=True,
                task_type=hyperparams.get("task_type", "regression")
            )
            
            pipeline = PreprocessingPipeline(config)
            
            # Restore standard scaler
            pipeline.standard_scaler = StandardScaler()
            pipeline.standard_scaler.means = np.array(hyperparams["scaler_means"])
            pipeline.standard_scaler.stds = np.array(hyperparams["scaler_stds"])
            pipeline.standard_scaler.is_fitted = True
            pipeline.is_fitted = True
            
            return pipeline
        
        return None