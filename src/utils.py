# utils.py


# Standard libraries
import math
import random
import itertools
from typing import List, Dict, Optional

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

# RDKit for chemistry
from rdkit import Chem


# Random and Environment Setup

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)            # Python random
    np.random.seed(seed)         # NumPy
    torch.manual_seed(seed)      # PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)




# Activation and Learning Rate Helpers

def get_activation_function(activation_type: str):
    """
    Returns the appropriate activation function based on the provided type.
    
    Args:
        activation_type: A string specifying the activation function type. 
                        Should be one of "relu", "leakyrelu", "elu", "gelu", or "silu".
    
    Returns:
        An activation function from torch.nn
    """
    activation_map = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU()
    }
    
    if activation_type in activation_map:
        return activation_map[activation_type]
    else:
        raise ValueError(f"Invalid activation type: {activation_type}")

def get_layer_wise_learning_rates(model, base_lr, decay_factor=0.8):
    """Apply lower learning rates to earlier layers in the model."""
    parameter_groups = []
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Group parameters by layer depth
    layers = {}
    for name, param in named_params:
        if param.requires_grad:
            depth = name.count('.')
            if depth not in layers:
                layers[depth] = []
            layers[depth].append((name, param))
    
    # Sort layers by depth (deeper layers come last)
    sorted_depths = sorted(layers.keys())
    
    # Calculate learning rate for each depth
    for i, depth in enumerate(sorted_depths):
        # Calculate decay based on relative position
        position_factor = i / max(1, len(sorted_depths) - 1)  # 0 for first, 1 for last
        # Apply decay - deeper layers get higher learning rates
        lr = base_lr * (decay_factor ** (1 - position_factor))
        
        params = [param for _, param in layers[depth]]
        parameter_groups.append({'params': params, 'lr': lr})
        
    return parameter_groups




# Gathering Functions (for DDP)


def gather_ndarray_to_rank0(arr, device="cpu"):
    """
    Gather a NumPy array from all ranks onto rank 0.
    
    Args:
        arr: NumPy array to gather
        device: Device to perform gathering on
        
    Returns:
        On rank 0: Concatenated array from all ranks
        On other ranks: Empty array
    """
    # Convert local NumPy to torch tensor
    local_tensor = torch.from_numpy(arr).float().to(device)
    local_size = torch.tensor([local_tensor.size(0)], dtype=torch.long, device=device)

    # Gather all local sizes
    world_size = dist.get_world_size()
    sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes_list, local_size)

    # Pad local tensor to max size
    max_size = max(s.item() for s in sizes_list)
    if local_size < max_size:
        pad_amount = max_size - local_size.item()
        pad_shape = (pad_amount,) + tuple(local_tensor.shape[1:])
        local_tensor = torch.cat([local_tensor, torch.zeros(pad_shape, device=device)], dim=0)

    # All-gather
    gathered_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_tensor)

    # Unpad each chunk and convert to CPU
    rank_idx = dist.get_rank()
    cat_list = []
    for i in range(world_size):
        valid_n = sizes_list[i].item()
        chunk = gathered_list[i][:valid_n]
        cat_list.append(chunk.cpu().numpy())

    if rank_idx == 0:
        return np.concatenate(cat_list, axis=0)
    else:
        return np.array([], dtype=arr.dtype)

def gather_strings_to_rank0(local_list, device="cpu"):
    """
    Gather a list of strings from all ranks onto rank 0.
    
    Args:
        local_list: List of strings on each rank
        device: Device to perform gathering on
        
    Returns:
        On rank 0: Combined list of strings from all ranks
        On other ranks: Empty list
    """
    import pickle
    
    # Convert local_list into pickled bytes
    local_bytes = pickle.dumps(local_list)
    local_size = torch.tensor([len(local_bytes)], dtype=torch.long, device=device)

    # Gather all local sizes
    world_size = dist.get_world_size()
    sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes_list, local_size)

    max_size = max(s.item() for s in sizes_list)
    # Pad local_bytes to max_size
    if local_size.item() < max_size:
        local_bytes += b'\x00' * (max_size - local_size.item())

    # Convert to tensor
    local_tensor = torch.ByteTensor(list(local_bytes)).to(device)

    # All-gather
    gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)

    rank_idx = dist.get_rank()
    if rank_idx == 0:
        all_strs = []
        for i in range(world_size):
            valid_n = sizes_list[i].item()
            chunk_bytes = gathered_tensors[i][:valid_n].cpu().numpy().tobytes()
            chunk_list = pickle.loads(chunk_bytes)
            all_strs.extend(chunk_list)
        return all_strs
    else:
        return []

def is_main_process():
    """
    Returns True if the current process is the main process (rank 0) or if 
    distributed processing is not being used.
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

def safe_get_rank():
    """
    Returns the current process rank. If not running in a distributed environment, returns 0.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0