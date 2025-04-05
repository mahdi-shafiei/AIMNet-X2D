#config.py

# Standard libraries
import argparse
import os

# For YAML configuration files
import yaml
from datetime import datetime


def save_experiment_config(args, filepath=None):
    """Save experiment configuration to a YAML file."""
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = args.experiment_name if hasattr(args, 'experiment_name') else "experiment"
        filepath = f"{exp_name}_{timestamp}_config.yaml"
    
    # Convert args to dictionary
    config = vars(args)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Experiment configuration saved to {filepath}")
    return filepath

def load_experiment_config(filepath):
    """Load experiment configuration from a YAML file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert dictionary to argparse Namespace
    args = argparse.Namespace(**config)
    return args


# Argument validation

def validate_args(args):
    """
    Validates command-line arguments and performs consistency checks.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    # Initialize validation status
    valid = True
    errors = []
    warnings = []
    
    # Check data loading paths
    if args.data_path:
        if args.train_data or args.val_data or args.test_data:
            errors.append("Cannot specify both --data_path and individual train/val/test dataset paths.")
            valid = False
        
        # Check split ratios
        total_split = args.train_split + args.val_split + args.test_split
        if not (0.999 <= total_split <= 1.001):  # Allow for floating point imprecision
            errors.append(f"Split ratios must sum to 1.0, got {total_split}")
            valid = False
    elif not (args.train_data and args.val_data and args.test_data) and not (args.inference_csv or args.inference_hdf5):
        errors.append("Must specify either --data_path or all of --train_data, --val_data, and --test_data when not in inference mode.")
        valid = False
    
    # Check task-specific requirements
    if args.task_type == 'multitask':
        if not args.multi_target_columns:
            errors.append("For --task_type=multitask, must specify --multi_target_columns.")
            valid = False
        
        if args.multitask_weights:
            weights = [float(w.strip()) for w in args.multitask_weights.split(',')]
            multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]
            if len(weights) != len(multi_cols):
                errors.append(f"Number of multitask weights ({len(weights)}) must match number of target columns ({len(multi_cols)}).")
                valid = False
    
    # Check SAE configuration
    if args.calculate_sae and args.task_type == 'multitask' and not args.sae_subtasks:
        warnings.append("SAE calculation enabled for multitask, but no --sae_subtasks specified. SAE will be skipped.")
    
    # Check for HDF5 paths when using iterable dataset
    if args.iterable_dataset:
        if not (args.train_hdf5 and args.val_hdf5 and args.test_hdf5):
            errors.append("When using --iterable_dataset, must specify --train_hdf5, --val_hdf5, and --test_hdf5.")
            valid = False
    
    # Check inference configuration
    if args.inference_csv or args.inference_hdf5:
        if not args.model_save_path:
            errors.append("Must specify --model_save_path when running in inference mode.")
            valid = False
        
        if not args.inference_output:
            warnings.append("No --inference_output specified. Results will be saved to 'predictions.csv'.")
    
    # Check DDP configuration
    if args.num_gpu_devices > 1:
        if args.num_workers > 1:
            warnings.append("Using multiple GPUs with --num_workers > 1 may cause issues with some PyTorch versions.")
    
    # Check hyperparameter tuning configuration
    if args.hyperparameter_file and args.num_trials <= 1:
        warnings.append("Hyperparameter file specified but --num_trials is 1. Only one set will be used.")
    
    # Check wandb configuration
    if args.enable_wandb and not args.wandb_project:
        warnings.append("Wandb enabled but no --wandb_project specified. Using default project name.")
    
    # Check transfer learning configuration
    if args.transfer_learning:
        if args.freeze_layers and args.unfreeze_layers:
            warnings.append("Both --freeze_layers and --unfreeze_layers specified. --unfreeze_layers will take precedence for overlapping patterns.")
    
    # Check embedding extraction configuration
    if args.save_embeddings and not args.embeddings_output_path:
        warnings.append("No --embeddings_output_path specified. Embeddings will be saved to 'molecular_embeddings.h5'.")
    
    # Print warnings and errors
    for warning in warnings:
        print(f"WARNING: {warning}")
    
    for error in errors:
        print(f"ERROR: {error}")
    
    return valid


