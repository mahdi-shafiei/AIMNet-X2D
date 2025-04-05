# model.py

# Standard libraries
import math
from typing import List, Dict, Optional

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# For pooling operations
import torch_scatter

# For local imports
from utils import get_activation_function

# NumPy for array operations
import numpy as np

from torch_geometric.data import InMemoryDataset, Data, Batch


## GNN Layer and Pooling Classes
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
    def __init__(self, atom_input_dim, output_dim, num_hops=3, dropout=0.00, activation_type="silu", num_mlp_layers=2):
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
        
    def forward(self, x, target, src):
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
        
    def message_passing(self, atom_features, target, src):
        expanded_atom_features = atom_features.repeat(self.num_hops, 1)
        aggregated = torch.zeros_like(expanded_atom_features)
        if target.numel() != 0:
            source_features = expanded_atom_features[src]
            aggregated.index_add_(0, target, source_features)
        chunks = torch.split(aggregated, atom_features.shape[0], dim=0)
        return chunks

class MeanPoolingLayer(nn.Module):
    """Mean pooling for graph-level representations."""
    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, x, batch_indices):
        x = x.to(batch_indices.device)
        x_pooled = torch_scatter.scatter_mean(x, batch_indices, dim=0)
        return x_pooled, None

class MaxPoolingLayer(nn.Module):
    """Max pooling for graph-level representations."""
    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, x, batch_indices):
        """
        Performs max pooling on node features within each graph in the batch.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features].
            batch_indices (Tensor): Batch index for each node of shape [num_nodes].

        Returns:
            Tensor: Pooled graph features of shape [num_graphs, num_features].
            None: Placeholder to maintain consistency with other pooling layers.
        """
        x = x.to(batch_indices.device)
        # scatter_max returns a tuple (values, argmax_indices)
        x_pooled, _ = torch_scatter.scatter_max(x, batch_indices, dim=0)
        return x_pooled, None

class SumPoolingLayer(nn.Module):
    """Sum pooling for graph-level representations."""
    def __init__(self):
        super(SumPoolingLayer, self).__init__()

    def forward(self, x, batch_indices):
        """
        Performs sum pooling on node features within each graph in the batch.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features].
            batch_indices (Tensor): Batch index for each node of shape [num_nodes].

        Returns:
            Tensor: Pooled graph features of shape [num_graphs, num_features].
            None: Placeholder to maintain consistency with other pooling layers.
        """
        x = x.to(batch_indices.device)
        # scatter_add sums the features for each graph in the batch
        x_pooled = torch_scatter.scatter_add(x, batch_indices, dim=0)
        return x_pooled, None

class MultiHeadAttentionPoolingLayer(nn.Module):
    """
    Multi-head attention pooling for graph-level representations.
    
    Args:
        input_dim: Input dimension
        num_heads: Number of attention heads
        initial_temperature: Initial temperature for attention scaling
        learnable_temperature: Whether temperature is learnable
        dropout_prob: Dropout probability
    """
    def __init__(self, input_dim, num_heads=4, initial_temperature=1.0,
                 learnable_temperature=True, dropout_prob=0.0):
        super(MultiHeadAttentionPoolingLayer, self).__init__()
        self.num_heads = num_heads
        self.attention_weights = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(self.num_heads)])
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, batch_indices):
        attention_scores_list = []
        for i in range(self.num_heads):
            attention_scores = self.attention_weights[i](x).squeeze(-1) / self.temperature
            attention_scores_list.append(attention_scores)

        attention_scores = torch.stack(attention_scores_list, dim=0)
        if batch_indices is not None:
            expanded_batch_indices = batch_indices.unsqueeze(0).expand(self.num_heads, -1)
            attention_weights = torch_scatter.scatter_softmax(attention_scores, expanded_batch_indices, dim=1)
        else:
            attention_weights = F.softmax(attention_scores, dim=1)

        x_expanded = x.unsqueeze(0).expand(self.num_heads, -1, -1)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        weighted_x = x_expanded * attention_weights_expanded

        if batch_indices is not None:
            x_pooled = torch_scatter.scatter_sum(weighted_x, batch_indices, dim=1)
            x_pooled = x_pooled.mean(dim=0)
        else:
            x_pooled = torch.sum(weighted_x, dim=1, keepdim=True)
            x_pooled = x_pooled.mean(dim=0)

        if self.dropout.p > 0:
            x_pooled = self.dropout(x_pooled)

        return x_pooled, attention_weights


## Main

class GNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    
    Args:
        feature_sizes: Dictionary of feature dimensions
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_shells: Number of shells/hops for message passing
        num_message_passing_layers: Number of message passing layers
        dropout: Dropout probability
        ffn_hidden_dim: Feed-forward network hidden dimension
        ffn_num_layers: Number of feed-forward layers
        pooling_type: Type of graph pooling ('attention', 'mean', 'max', 'sum')
        task_type: Type of task ('regression', 'multitask')
        embedding_dim: Embedding dimension for atom features
        use_partial_charges: Whether to use partial charges
        use_stereochemistry: Whether to use stereochemistry features
        ffn_dropout: Dropout rate for feed-forward layers
        activation_type: Type of activation function
        shell_conv_num_mlp_layers: Number of MLP layers in shell convolution
        shell_conv_dropout: Dropout rate for shell convolution
        attention_num_heads: Number of attention heads
        attention_temperature: Initial temperature for attention
    """
    def __init__(self, 
                 feature_sizes, 
                 hidden_dim, 
                 output_dim, 
                 num_shells=3,
                 num_message_passing_layers=3, 
                 dropout=0.05, 
                 ffn_hidden_dim=None,
                 ffn_num_layers=3, 
                 pooling_type='attention',
                 task_type='regression',
                 embedding_dim=64,
                 use_partial_charges=False,
                 use_stereochemistry=False,
                 ffn_dropout=0.05,
                 activation_type="silu",
                 shell_conv_num_mlp_layers=2,
                 shell_conv_dropout=0.05,
                 attention_num_heads=4,
                 attention_temperature=1.0):
        super(GNN, self).__init__()

        self.use_partial_charges = use_partial_charges
        self.use_stereochemistry = use_stereochemistry

        self.shell_conv_num_mlp_layers = shell_conv_num_mlp_layers
        self.shell_conv_dropout = shell_conv_dropout
        self.attention_num_heads = attention_num_heads 
        self.attention_temperature = attention_temperature

        # Print feature activation status
        print(f"[GNN Initialization] Partial Charges Enabled: {self.use_partial_charges}")
        print(f"[GNN Initialization] Stereochemistry Features Enabled: {self.use_stereochemistry}")

        self.task_type = task_type
        self.embedding_dim = embedding_dim
        if ffn_hidden_dim is None:
            ffn_hidden_dim = hidden_dim

        self.embedding_dim = embedding_dim

        # Define embedding layers for our simplified feature set with equal weighting
        self.atom_type_embedding = nn.Embedding(
            num_embeddings=feature_sizes['atom_type'],
            embedding_dim=embedding_dim  # All features get same embedding dimension
        )
        
        self.hydrogen_count_embedding = nn.Embedding(
            num_embeddings=feature_sizes['hydrogen_count'],
            embedding_dim=embedding_dim  # All features get same embedding dimension
        )
        
        self.degree_embedding = nn.Embedding(
            num_embeddings=feature_sizes['degree'],
            embedding_dim=embedding_dim  # All features get same embedding dimension
        )
        
        self.hybridization_embedding = nn.Embedding(
            num_embeddings=feature_sizes['hybridization'],
            embedding_dim=embedding_dim  # All features get same embedding dimension
        )

        total_atom_embedding_dim = embedding_dim * 4
        self.embedding_projection = nn.Linear(total_atom_embedding_dim, hidden_dim)
        self.activation = get_activation_function(activation_type)

        self.x_other_dim = int(0.3 * hidden_dim)
        self.x_self_dim = hidden_dim - self.x_other_dim

        self.forward_pass_count = 0

        self.message_passing_layers = nn.ModuleList()
        for _ in range(num_message_passing_layers):
            self.message_passing_layers.append(
                ShellConvolutionLayer(
                    self.x_other_dim, 
                    self.x_other_dim, 
                    num_hops=num_shells, 
                    activation_type=activation_type, 
                    dropout=self.shell_conv_dropout, 
                    num_mlp_layers=self.shell_conv_num_mlp_layers
                )
            )

        if pooling_type == 'attention':
            self.pooling = MultiHeadAttentionPoolingLayer(
                input_dim=hidden_dim,
                num_heads=self.attention_num_heads,
                initial_temperature=self.attention_temperature,
                learnable_temperature=True,
                dropout_prob=0.0
            )
        elif pooling_type == 'mean':
            self.pooling = MeanPoolingLayer()
        elif pooling_type == 'max':
            self.pooling = MaxPoolingLayer()
        elif pooling_type == 'sum':
            self.pooling = SumPoolingLayer()
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}")

        self.concat_self_other = nn.Linear(hidden_dim, hidden_dim)
        self.stereochemical_embedding = nn.Linear(hidden_dim * 3, hidden_dim)
        self.stereochemical_embedding_2 = nn.Linear(self.x_other_dim * 3, self.x_other_dim)
        self.post_pooling_projection = nn.Linear(hidden_dim, ffn_hidden_dim)

        self.ffn_layers = nn.ModuleList()
        input_dim = ffn_hidden_dim
        for _ in range(ffn_num_layers):
            layer_block = nn.ModuleDict({
                'linear_1': nn.Linear(input_dim, ffn_hidden_dim),
                'activation': get_activation_function(activation_type),
                'dropout': nn.Dropout(p=ffn_dropout),
                'linear_2': nn.Linear(ffn_hidden_dim, ffn_hidden_dim),
            })
            self.ffn_layers.append(layer_block)

        self.skip_transform = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        self.output_layer = nn.Linear(ffn_hidden_dim * 2, output_dim)

        self.long_range_projection = nn.Linear(hidden_dim, ffn_hidden_dim)

    def forward(self, atom_features, multi_hop_edge_indices, batch_indices,
                total_charges, tetrahedral_indices, cis_indices, trans_indices):
        self.forward_pass_count += 1

        # Get embeddings for our simplified feature set - all equally weighted
        atom_type_emb = self.atom_type_embedding(atom_features['atom_type'])
        hydrogen_count_emb = self.hydrogen_count_embedding(atom_features['hydrogen_count'])
        degree_emb = self.degree_embedding(atom_features['degree'])
        hybridization_emb = self.hybridization_embedding(atom_features['hybridization'])

        # Concatenate all embeddings - all have equal embedding dimension
        atom_embeddings = torch.cat([
            atom_type_emb,           # All features get same embedding_dim 
            hydrogen_count_emb,
            degree_emb,
            hybridization_emb,
        ], dim=-1)

        atom_embeddings = self.embedding_projection(atom_embeddings)
        atom_embeddings = self.activation(atom_embeddings)

        x_self, x_other = torch.split(atom_embeddings, [self.x_self_dim, self.x_other_dim], dim=-1)

        # Message passing
        x_other_updated = x_other
        if multi_hop_edge_indices.numel() > 0:
            for _, layer in enumerate(self.message_passing_layers):
                if self.use_partial_charges:
                    x_other_updated = self.partial_charge_calculation(x_other_updated, batch_indices, total_charges)

                partial_charges_all_atoms = None
                if self.use_partial_charges:
                    partial_charges_all_atoms = x_other_updated[:, 0].clone()  # shape [num_atoms]

                if self.use_stereochemistry:
                    cis_trans_features = self.cis_trans_calculation(x_other_updated, cis_indices, trans_indices)
                    tetrahedral_features = self.tetrahedral_feature_calculation_additive_rolled(x_other_updated, tetrahedral_indices)
                    x_concat_stereochemistry = torch.cat([x_other_updated, cis_trans_features, tetrahedral_features], dim=-1)
                    x_other_updated = self.stereochemical_embedding_2(x_concat_stereochemistry)

                x_other_updated = layer(
                    x_other_updated,
                    multi_hop_edge_indices[:, 0],
                    multi_hop_edge_indices[:, 1]
                ) + x_other_updated

        # Combine
        x_concat = torch.cat([x_self, x_other_updated], dim=-1)
        x = self.concat_self_other(x_concat)

        # Pooling
        x_pooled, attention_weights = self.pooling(x, batch_indices)
        pooled_for_skip = x_pooled.clone()
        x = self.post_pooling_projection(x_pooled)

        # Feed-forward with residual
        for layer_block in self.ffn_layers:
            residual = x
            x = layer_block['linear_1'](x)
            x = layer_block['activation'](x)
            x = layer_block['dropout'](x)
            x = layer_block['linear_2'](x)
            x = x + residual

        skip_connection = self.skip_transform(x)
        final_output = torch.cat([x, skip_connection], dim=-1)
        output = self.output_layer(final_output)

        return output, attention_weights, partial_charges_all_atoms

    def tetrahedral_feature_calculation_additive_rolled(self, atom_features, tetrahedral_indices):
        """
        Vectorized approach that:
        1) Gathers each row's four neighbors => (M,4,D)
        2) Computes chirality in one go using torch.roll (no loops).
        3) index_add_ them back into 'updated' so repeated atoms accumulate all contributions.

        This matches your old code's final result for each neighbor index, but now in a single pass.
        """
        if tetrahedral_indices.numel() == 0:
            return atom_features

        # Start with a clone (if you want "original + chirality" effect).
        # If you ONLY want chirality sums, you can do zeros_like(atom_features).
        updated = atom_features.clone()

        # (1) Gather => shape (M,4,D)
        emb = updated[tetrahedral_indices]

        # (2) squares => shape (M,4,D). Then roll them to line up neighbors
        squares = emb**2
        squares_1 = torch.roll(squares, shifts=-1, dims=1)
        squares_2 = torch.roll(squares, shifts=-2, dims=1)
        squares_3 = torch.roll(squares, shifts=-3, dims=1)

        emb_1 = torch.roll(emb, shifts=-1, dims=1)
        emb_2 = torch.roll(emb, shifts=-2, dims=1)
        emb_3 = torch.roll(emb, shifts=-3, dims=1)

        # (3) Compute chirality => shape (M,4,D)
        #     For each row i in [0..M-1], columns j=0..3 correspond to
        #     the old "chirality_1..4" calculations, but done in bulk.
        C = (
            squares_1 * (emb_2 - emb_3)
        + squares_2 * (emb_3 - emb_1)
        + squares_3 * (emb_1 - emb_2)
        )

        # (4) Flatten indices + chirality for a single index_add_ call
        #     tetrahedral_indices => (M,4) => flatten => (M*4,)
        idx = tetrahedral_indices.reshape(-1)
        # C => (M,4,D) => flatten => (M*4, D)
        C_flat = C.reshape(-1, updated.shape[-1])

        # (5) Accumulate => each row in idx gets an addition of the corresponding row in C_flat
        updated.index_add_(0, idx, C_flat)

        return updated

    def cis_trans_calculation(self, atom_features, cis_indices, trans_indices):
        """
        Performs the same final transformation as the old code:
        - Subtract source_cis_features from the corresponding cis target rows,
        - Add source_trans_features to the corresponding trans target rows,
        but in a single scatter_add_ call.

        Args:
        atom_features: (N, D) Tensor of node features
        cis_indices: (2, K1) Indices for cis edges => row0 = source, row1 = target
        trans_indices: (2, K2) Indices for trans edges => same shape usage

        Returns:
        updated_features: (N, D), same shape as atom_features
        """
        # If there are no cis edges, just return as-is
        if cis_indices.numel() == 0 and trans_indices.numel() == 0:
            return atom_features

        # 1) Gather 'source' features for cis
        source_cis_nodes = cis_indices[0]  # shape (K1,)
        target_cis_nodes = cis_indices[1]  # shape (K1,)
        source_cis_features = atom_features[source_cis_nodes]  # (K1, D)

        # 2) Gather 'source' features for trans
        source_trans_nodes = trans_indices[0]  # shape (K2,)
        target_trans_nodes = trans_indices[1]  # shape (K2,)
        source_trans_features = atom_features[source_trans_nodes]  # (K2, D)

        # 3) Combine target indices:
        #    cis requires a negative contribution => we store -source_cis_features
        #    trans requires a positive contribution => source_trans_features
        all_targets = torch.cat([target_cis_nodes, target_trans_nodes], dim=0)           # shape (K1 + K2,)
        all_sources = torch.cat([-source_cis_features, source_trans_features], dim=0)   # shape (K1 + K2, D)

        # 4) Single scatter_add => shape (N, D)
        #    This is the same as doing two separate scatter_add calls, but faster.
        updated_features = atom_features.scatter_add(
            dim=0,
            index=all_targets.unsqueeze(1).expand(-1, atom_features.shape[1]),
            src=all_sources
        )

        return updated_features

    def partial_charge_calculation(self, atom_features, batch_indices, total_charges):
        _q, _f, delta_a = atom_features.split([1, 1, atom_features.shape[-1] - 2], dim=-1)
        _f = torch.clamp(_f, min=1e-6)

        target = torch.zeros((total_charges.shape[0], _q.shape[1]), device=_q.device)
        Q_u = target.scatter_add(
            0, batch_indices.unsqueeze(1), _q
        )

        target = torch.zeros((total_charges.shape[0], _f.shape[1]), device=_f.device)
        F_u = target.scatter_add(
            0, batch_indices.unsqueeze(1), _f
        ) + 1e-6

        F_u = torch.clamp(F_u, min=1e-6)
        dQ = total_charges.unsqueeze(-1) - Q_u

        F_u_expanded = F_u[batch_indices]
        dQ_expanded = dQ[batch_indices]

        f_new = _f / F_u_expanded
        q_new = _q + f_new * dQ_expanded

        return torch.cat([q_new, f_new, delta_a], dim=-1)

    def init_weights(self):
        """Initialize model weights."""
        layers_to_init = [
            self.atom_type_embedding,
            self.degree_embedding,
            self.hybridization_embedding,
            self.hydrogen_count_embedding,
            self.embedding_projection,
            self.post_pooling_projection,
            self.skip_transform,
            self.output_layer,
            self.concat_self_other,
            self.stereochemical_embedding,
            self.stereochemical_embedding_2,
            self.long_range_projection,
            *self.message_passing_layers
        ]
        
        # Initialize all the linear layers
        for layer in layers_to_init:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        # Initialize FFN layers
        for layer_block in self.ffn_layers:
            nn.init.xavier_uniform_(layer_block['linear_1'].weight)
            if layer_block['linear_1'].bias is not None:
                nn.init.zeros_(layer_block['linear_1'].bias)

        for layer_block in self.ffn_layers:
            nn.init.xavier_uniform_(layer_block['linear_2'].weight)
            if layer_block['linear_2'].bias is not None:
                nn.init.zeros_(layer_block['linear_2'].bias)
        
        # Initialize MultiHeadAttention weights if that's the pooling type used
        if isinstance(self.pooling, MultiHeadAttentionPoolingLayer):
            for attention_weight in self.pooling.attention_weights:
                nn.init.xavier_uniform_(attention_weight.weight)
                if attention_weight.bias is not None:
                    nn.init.zeros_(attention_weight.bias)



###############################################################################
# Normalization Classes
###############################################################################

class SizeExtensiveNormalizer:
    """
    For single-task regression only.
    Uses 'atomic_numbers' from the dataset objects.
    """
    def __init__(self):
        self.sae_dict = None

    def calc_sae_from_dataset(self, dataset: InMemoryDataset, percentile_cutoff: float = 2.0) -> Dict[int, float]:
        """
        Gather atomic_numbers and targets from dataset,
        then solve for atomic contributions in a least-squares sense.
        """
        print("Collecting atomic numbers and targets for SAE...")

        all_numbers = []
        all_targets = []
        for data in dataset.data_list:
            nums = data.atomic_numbers.cpu().numpy()
            tval = data.target.item()
            all_numbers.append(nums)
            all_targets.append(tval)

        all_targets = np.array(all_targets, dtype=np.float64)

        max_atomic_num = 119
        N = len(all_numbers)
        A = np.zeros((N, max_atomic_num), dtype=np.float64)

        for i, nums in enumerate(all_numbers):
            unique, counts = np.unique(nums, return_counts=True)
            for u, c in zip(unique, counts):
                if 1 <= u < max_atomic_num:
                    A[i, u] = c

        pct_low, pct_high = np.percentile(all_targets, [percentile_cutoff, 100 - percentile_cutoff])
        mask = (all_targets >= pct_low) & (all_targets <= pct_high)
        A_filt = A[mask]
        b_filt = all_targets[mask]

        print(f"Fitting atomic contributions using {len(b_filt)} molecules (after percentile filtering).")
        sae_values, residuals, rank, s = np.linalg.lstsq(A_filt, b_filt, rcond=None)
        self.sae_dict = {}
        for atomic_num in range(max_atomic_num):
            val = sae_values[atomic_num]
            if not np.isnan(val):
                self.sae_dict[atomic_num] = val

        return self.sae_dict

    def normalize_dataset(self, dataset: InMemoryDataset):
        """
        Subtract sum of atomic contributions from each molecule's target.
        """
        if self.sae_dict is None:
            print("SAE values not found. Calculating on the dataset provided...")
            self.calc_sae_from_dataset(dataset)

        for data in dataset.data_list:
            nums = data.atomic_numbers.cpu().numpy()
            shift = 0.0
            for n in nums:
                if n in self.sae_dict:
                    shift += self.sae_dict[n]
            data.target = data.target - shift
        return dataset


class MultiTaskSAENormalizer:
    """
    Applies single-task SAE logic per subtask index for multi-task.
    """
    def __init__(self, subtasks: List[int]):
        self.subtasks = subtasks
        self.sae_normalizers = {st: SizeExtensiveNormalizer() for st in subtasks}

    def calc_sae_from_dataset(self, dataset: InMemoryDataset, percentile_cutoff: float = 2.0):
        print(f"Calculating SAE for subtasks: {self.subtasks}")
        for st in self.subtasks:
            print(f"  => Subtask {st}")
            single_task_copy = InMemoryDataset(None)
            single_task_copy.data_list = []
            for data in dataset.data_list:
                if data.target.shape[0] <= st:
                    continue
                dcopy = Data()
                dcopy.atomic_numbers = data.atomic_numbers
                st_val = data.target[st].item()
                dcopy.target = torch.tensor([st_val], dtype=torch.float)
                single_task_copy.data_list.append(dcopy)

            self.sae_normalizers[st].calc_sae_from_dataset(single_task_copy, percentile_cutoff=percentile_cutoff)

    def normalize_dataset(self, dataset: InMemoryDataset):
        for data in dataset.data_list:
            nums = data.atomic_numbers.cpu().numpy()
            for st in self.subtasks:
                if data.target.shape[0] <= st:
                    continue
                shift = 0.0
                for n in nums:
                    if n in self.sae_normalizers[st].sae_dict:
                        shift += self.sae_normalizers[st].sae_dict[n]
                data.target[st] -= shift
        return dataset


class MultiTargetStandardScaler:
    """
    Standard scaling (subtract mean, divide by std) per dimension (multi-task).
    """
    def __init__(self):
        self.means = None
        self.stds = None
        self.fitted = False

    def fit(self, Y: np.ndarray):
        """Compute means and standard deviations from data."""
        self.means = Y.mean(axis=0)
        self.stds = Y.std(axis=0, ddof=1)
        self.stds[self.stds < 1e-12] = 1.0  # Avoid division by near-zero values
        self.fitted = True

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Apply scaling to data."""
        return (Y - self.means) / self.stds

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Reverse scaling."""
        return Y * self.stds + self.means

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(Y)
        return self.transform(Y)

###############################################################################
# Loss Functions
###############################################################################

class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss for multi-task.
    """
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = torch.abs(y_pred - y_true)
        weighted_error = error * self.weights.to(y_pred.device)
        loss_per_sample = weighted_error.sum(dim=1)
        loss = loss_per_sample.mean()
        return loss

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for multi-task.
    """
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = (y_pred - y_true) ** 2
        weighted_error = error * self.weights.to(y_pred.device)
        loss_per_sample = weighted_error.sum(dim=1)
        loss = loss_per_sample.mean()
        return loss

