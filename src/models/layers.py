"""
Neural network layers for molecular GNN.

This module contains the core layers used in the GNN architecture,
particularly the shell-based convolution layer for message passing.
"""

import torch
import torch.nn as nn
from typing import List

from utils.activation import get_activation_function


class ShellConvolutionLayer(nn.Module):
    """
    Shell-based graph convolution layer.
    
    Performs message passing between atoms connected within a certain number of hops.
    
    Args:
        atom_input_dim: Input dimension of atom features
        output_dim: Output dimension
        num_hops: Number of hops for message passing
        dropout: Dropout probability
        activation_type: Type of activation function
        num_mlp_layers: Number of MLP layers
    """
    
    def __init__(self, 
                 atom_input_dim: int, 
                 output_dim: int, 
                 num_hops: int = 3, 
                 dropout: float = 0.00, 
                 activation_type: str = "silu", 
                 num_mlp_layers: int = 2):
        super(ShellConvolutionLayer, self).__init__()
        
        self.num_hops = num_hops
        input_dim = atom_input_dim * (num_hops + 1)

        self.activation = get_activation_function(activation_type)
        
        # Main input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Create MLP blocks with per-layer skip connections
        self.mlp_blocks = nn.ModuleList()
        for _ in range(num_mlp_layers):
            block = nn.ModuleDict({
                'linear_1': nn.Linear(output_dim, output_dim),
                'activation': get_activation_function(activation_type),
                'dropout': nn.Dropout(dropout),
                'linear_2': nn.Linear(output_dim, output_dim)
            })
            self.mlp_blocks.append(block)
        
        # Skip connection projection (if input and output dimensions differ)
        self.global_skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
    def forward(self, x: torch.Tensor, target: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shell convolution layer.
        
        Args:
            x: Input atom features
            target: Target indices for message passing
            src: Source indices for message passing
            
        Returns:
            Updated atom features after message passing
        """
        # Collect multi-hop features
        all_features = [x]
        hop_features = self.message_passing(x, target, src)
        all_features.extend(hop_features)
        input_features = torch.cat(all_features, dim=-1)
        
        # Project input to expected dimension
        x = self.input_proj(input_features)
        x = self.activation(x)  # Initial activation
        
        # Store for global skip connection
        if self.global_skip_proj is not None:
            global_skip = self.global_skip_proj(input_features)
        else:
            global_skip = x.clone()  # Already projected to output_dim
        
        # Apply MLP blocks with per-layer skip connections
        for block in self.mlp_blocks:
            # Store for layer-wise skip connection
            layer_skip = x
            
            # Forward through this block
            x = block['linear_1'](x)
            x = block['activation'](x)
            x = block['dropout'](x)
            x = block['linear_2'](x)
            
            # Add layer-wise skip connection
            x = x + layer_skip
        
        # Add global skip connection at the end
        x = x + global_skip
        
        return x
        
    def message_passing(self, atom_features: torch.Tensor, target: torch.Tensor, src: torch.Tensor) -> List[torch.Tensor]:
        """
        Perform multi-hop message passing.
        
        Args:
            atom_features: Input atom features
            target: Target atom indices
            src: Source atom indices
            
        Returns:
            List of aggregated features for each hop
        """
        expanded_atom_features = atom_features.repeat(self.num_hops, 1)
        aggregated = torch.zeros_like(expanded_atom_features)
        
        if target.numel() != 0:
            source_features = expanded_atom_features[src]
            aggregated.index_add_(0, target, source_features)
            
        chunks = torch.split(aggregated, atom_features.shape[0], dim=0)
        return chunks


class LinearBlock(nn.Module):
    """
    Basic linear block with activation and dropout.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        activation_type: Type of activation function
        dropout: Dropout probability
        use_skip: Whether to use skip connection
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 activation_type: str = "silu",
                 dropout: float = 0.0,
                 use_skip: bool = True):
        super(LinearBlock, self).__init__()
        
        self.use_skip = use_skip and (input_dim == output_dim)
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = get_activation_function(activation_type)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(output_dim, output_dim)
        
        # Skip connection projection if needed
        if self.use_skip and input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear block."""
        identity = x
        
        # Main path
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        # Skip connection
        if self.use_skip:
            if self.skip_proj is not None:
                identity = self.skip_proj(identity)
            out = out + identity
            
        return out


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron with configurable depth and skip connections.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        activation_type: Type of activation function
        dropout: Dropout probability
        use_skip: Whether to use skip connections
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers: int = 2,
                 activation_type: str = "silu",
                 dropout: float = 0.0,
                 use_skip: bool = True):
        super(MultiLayerPerceptron, self).__init__()
        
        layers = []
        
        # Input layer
        if num_layers == 1:
            layers.append(LinearBlock(input_dim, output_dim, activation_type, dropout, False))
        else:
            layers.append(LinearBlock(input_dim, hidden_dim, activation_type, dropout, False))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(LinearBlock(hidden_dim, hidden_dim, activation_type, dropout, use_skip))
            
            # Output layer
            layers.append(LinearBlock(hidden_dim, output_dim, activation_type, dropout, False))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        for layer in self.layers:
            x = layer(x)
        return x