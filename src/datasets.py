# datasets.py

# Standard libraries
import os
import math
import random
import time
import pickle
from typing import List, Dict, Tuple, Any
from functools import partial
from multiprocessing import Pool

# Third-party libraries
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch_geometric.data import InMemoryDataset, Data, Batch
import tqdm

# RDKit imports for molecule parsing
from rdkit import Chem
from rdkit.Chem import rdBase
from rdkit.Chem.rdchem import HybridizationType

# Numba for BFS
from numba import njit, boolean
from numba.typed import List as NumbaList

# For data splitting
from sklearn.model_selection import train_test_split


# Atom feature constants
ATOM_TYPES = list(range(1, 119))  # Atomic numbers from 1 to 118
DEGREES = list(range(6))          # Degrees from 0 to 5
HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2
]



# SAE Helpers and Atomic Parsing

def partial_parse_atomic_numbers(smiles: str) -> np.ndarray or None:
    """Quick parse of SMILES to get atomic numbers only."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
    except:
        return None
    nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return np.array(nums, dtype=np.int32)

def compute_sae_dict_from_atomic_numbers_list(
    atomic_numbers_list: List[np.ndarray],
    target_values: List[float],
    percentile_cutoff: float = 2.0
) -> Dict[int, float]:
    """
    Compute Size-Extensive Additive (SAE) contribution for each atom type.
    
    Args:
        atomic_numbers_list: List of arrays containing atomic numbers for each molecule
        target_values: Target property values for each molecule
        percentile_cutoff: Percentile cutoff for filtering outliers
        
    Returns:
        Dictionary mapping atomic numbers to their SAE contributions
    """
    all_targets = np.array(target_values, dtype=np.float64)
    max_atomic_num = 119
    N = len(atomic_numbers_list)
    A = np.zeros((N, max_atomic_num), dtype=np.float64)

    for i, nums in enumerate(atomic_numbers_list):
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

    sae_dict = {}
    for atomic_num in range(max_atomic_num):
        val = sae_values[atomic_num]
        if not np.isnan(val):
            sae_dict[atomic_num] = val

    return sae_dict


def precompute_all_and_filter(
    smiles_list: List[str],
    target_values: List[Any],  # float or list[float]
    max_hops: int,
    num_workers: int = 4
) -> Tuple[List[str], List[Any], List[Dict[str, Any]]]:
    """
    In-memory BFS + feature precomputation with multiprocessing.
    
    Args:
        smiles_list: List of SMILES strings
        target_values: List of target values (single values or lists for multi-task)
        max_hops: Maximum number of hops for BFS
        num_workers: Number of parallel workers
        
    Returns:
        Tuple of (valid_smiles, valid_targets, precomputed_data)
    """
    print(f"Precomputing multi-hop edge + features for {len(smiles_list)} SMILES using {num_workers} workers...")
    start_time = time.time()

    from functools import partial
    compute_partial = partial(compute_all, max_hops=max_hops)

    valid_smiles = []
    valid_targets = []
    precomputed_data = []

    with Pool(num_workers) as pool:
        for smi, tgt, res in tqdm.tqdm(
            zip(smiles_list, target_values, pool.imap(compute_partial, smiles_list, chunksize=1000)),
            total=len(smiles_list)
        ):
            if res is not None:
                valid_smiles.append(smi)
                valid_targets.append(tgt)
                precomputed_data.append(res)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    discarded = len(smiles_list) - len(valid_smiles)
    print(f"Kept {len(valid_smiles)} valid SMILES; discarded {discarded} invalid or unparseable.")

    return valid_smiles, valid_targets, precomputed_data

def precompute_and_write_hdf5_parallel_chunked(
    smiles_list: List[str],
    target_values: List[Any],
    max_hops: int,
    hdf5_path: str,
    num_workers: int = 4,
    chunk_size: int = 1000,
    sae_subtasks: List[int] = None,  # For SAE subtasks
    task_type: str = "regression",
    multi_target_columns: List[str] = None,
):
    """
    Parallel BFS + chunked writes to HDF5.
    
    This function computes molecular features and BFS results in parallel,
    optionally applies SAE normalization, and writes results to an HDF5 file
    in chunks for memory efficiency.
    
    Args:
        smiles_list: List of SMILES strings
        target_values: List of target values (floats or lists for multi-task)
        max_hops: Maximum number of hops for BFS
        hdf5_path: Path to write HDF5 file
        num_workers: Number of parallel workers
        chunk_size: Size of processing chunks
        sae_subtasks: List of subtask indices for SAE normalization (multi-task only)
        task_type: Task type ('regression' or 'multitask')
        multi_target_columns: Column names for multi-task targets
    """
    print(f"Using {num_workers} workers")
    from functools import partial
    from multiprocessing import Pool
    import h5py

    with h5py.File(hdf5_path, "w") as f:
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dset = f.create_dataset("data", (len(smiles_list),), dtype=dt)

        # Create index map dataset
        index_map_dset = f.create_dataset("index_map", (len(smiles_list),), dtype=np.int32)
        index_map_dset[:] = np.arange(len(smiles_list))  # Initialize with identity mapping

        # Add metadata group
        metadata = f.create_group("metadata")
        metadata.attrs["num_samples"] = len(smiles_list)
        metadata.attrs["task_type"] = task_type
        metadata.attrs["max_hops"] = max_hops
        if multi_target_columns is not None:
            dt_str = h5py.special_dtype(vlen=str)
            target_cols = metadata.create_dataset("target_columns", (len(multi_target_columns),), dtype=dt_str)
            for i, col in enumerate(multi_target_columns):
                target_cols[i] = col

        print(f"Writing BFS data to HDF5 (parallel + chunked) => {hdf5_path}")

        # SAE calculations for multi-task
        if task_type == "multitask" and sae_subtasks is not None:
            print(f"Performing SAE calculations for subtasks: {sae_subtasks} in multitask mode.")
            target_array = np.array(target_values, dtype=np.float64)

            # Partial parse for atomic numbers
            atomic_nums_list = []
            for smi in tqdm.tqdm(smiles_list, desc="Partial Parse for Atomic Numbers"):
                nums = partial_parse_atomic_numbers(smi)
                atomic_nums_list.append(nums)

            # Process each subtask
            for st in sae_subtasks:
                print(f"Computing SAE for subtask {st} ({multi_target_columns[st] if multi_target_columns else ''})...")
                subtask_targets = []
                subtask_nums = []
                for nums, tvals in zip(atomic_nums_list, target_array):
                    if nums is not None:
                        subtask_targets.append(tvals[st])
                        subtask_nums.append(nums)

                # Compute SAE dictionary for this subtask
                sae_dict_st = compute_sae_dict_from_atomic_numbers_list(
                    subtask_nums, subtask_targets
                )

                # print(f"  SAE Dict for subtask {st} ({multi_target_columns[st] if multi_target_columns else ''}): {sae_dict_st}")

                # Apply shift to target_array (in place)
                for i, nums in enumerate(atomic_nums_list):
                    if nums is not None:
                        shift_val = sum(sae_dict_st.get(n, 0.0) for n in nums)
                        target_array[i, st] -= shift_val
                print(f"  SAE shift applied to subtask {st} targets.")

            # Update target_values with shifted values
            target_values = target_array.tolist()
            
            # Store SAE information in metadata
            sae_group = metadata.create_group("sae")
            sae_group.attrs["applied"] = True
            sae_group.attrs["subtasks"] = np.array(sae_subtasks, dtype=np.int32)
            
        elif task_type == "regression" and sae_subtasks is not None:
            print(f"SAE calculation for regression task.")
            
            # Partial parse for atomic numbers
            atomic_nums_list = []
            for smi in tqdm.tqdm(smiles_list, desc="Partial Parse for Atomic Numbers"):
                nums = partial_parse_atomic_numbers(smi)
                atomic_nums_list.append(nums)
                
            # Calculate SAE dict
            subtask_targets = []
            subtask_nums = []
            for nums, tval in zip(atomic_nums_list, target_values):
                if nums is not None:
                    subtask_targets.append(tval)
                    subtask_nums.append(nums)
                    
            sae_dict = compute_sae_dict_from_atomic_numbers_list(
                subtask_nums, subtask_targets
            )
            
            print(f"  SAE Dict: {sae_dict}")
            
            # Apply shift to targets
            shifted_targets = []
            for nums, tval in zip(atomic_nums_list, target_values):
                shift = 0.0
                if nums is not None:
                    for n in nums:
                        shift += sae_dict.get(n, 0.0)
                shifted_targets.append(tval - shift)
                
            # Update target_values with shifted values
            target_values = shifted_targets
            
            # Store SAE information in metadata
            sae_group = metadata.create_group("sae")
            sae_group.attrs["applied"] = True
        else:
            # No SAE normalization
            sae_group = metadata.create_group("sae")
            sae_group.attrs["applied"] = False

        # Setup worker function and parallel pool
        func_partial = partial(_worker_bfs, max_hops=max_hops)

        with Pool(num_workers) as pool:
            # Process SMILES in parallel
            results_iter = pool.imap(
                func_partial, 
                zip(smiles_list, target_values), 
                chunksize=chunk_size
            )

            # Process and write in chunks
            buffer = []
            buffer_indices = []
            
            for i, res in enumerate(
                tqdm.tqdm(results_iter, total=len(smiles_list), desc="Processing molecules")
            ):
                if res is None:
                    # Encode None result for invalid SMILES
                    encoded = pickle.dumps(None)
                else:
                    # Encode valid result
                    to_store = {
                        'smiles': res['smiles'],
                        'target': res['target'],
                        'precomputed': res['precomputed']
                    }
                    encoded = pickle.dumps(to_store)
                
                # Add to buffer
                buffer.append(np.frombuffer(encoded, dtype=np.uint8))
                buffer_indices.append(i)

                # Once we have chunk_size items, or end of iteration => bulk write
                if len(buffer) >= chunk_size or i == len(smiles_list) - 1:
                    if buffer_indices:  # Make sure there's something to write
                        dset[buffer_indices[0] : buffer_indices[-1] + 1] = buffer
                        #print(f"Wrote chunk of {len(buffer)} molecules (indices {buffer_indices[0]}-{buffer_indices[-1]})")
                        buffer = []
                        buffer_indices = []

        # Calculate and store statistics
        valid_count = 0
        invalid_count = 0
        
        # Sample a few entries to determine validity
        sample_size = min(1000, len(smiles_list))
        sample_indices = random.sample(range(len(smiles_list)), sample_size)
        
        for idx in sample_indices:
            raw = dset[idx]
            decoded = pickle.loads(raw.tobytes())
            if decoded is not None and decoded.get('precomputed') is not None:
                valid_count += 1
            else:
                invalid_count += 1
        
        # Extrapolate statistics to full dataset
        estimated_valid_pct = (valid_count / sample_size) * 100
        metadata.attrs["estimated_valid_pct"] = estimated_valid_pct
        
        print(f"HDF5 file created successfully at {hdf5_path}")
        print(f"Estimated valid molecules: {estimated_valid_pct:.1f}% (based on sample of {sample_size})")

###############################################################################
# Dataset Classes
###############################################################################

class PyGSMILESDataset(InMemoryDataset):
    """
    InMemoryDataset that stores each molecule as a Data object.
    
    Args:
        smiles_list: List of SMILES strings
        targets: List of target values (single values or lists for multi-task)
        precomputed_data: List of dictionaries with precomputed molecular features
        transform: PyG transform to apply
        pre_transform: PyG pre-transform to apply
    """
    def __init__(
        self,
        smiles_list: List[str],
        targets: List[Any],  # float or list
        precomputed_data: List[Dict[str, Any]],
        transform=None,
        pre_transform=None,
        **kwargs
    ):
        self.smiles_list = smiles_list
        self.targets = targets
        self.precomputed_data = precomputed_data
        super().__init__(None, transform, pre_transform)
        self.process()
        self.data = None
        self.slices = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        self.data_list = []
        for i, smiles in enumerate(self.smiles_list):
            precomp = self.precomputed_data[i]

            num_atoms = precomp['atom_features']['atom_type'].shape[0]
            x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)

            data = Data()
            data.x = x_dummy
            data.smiles = precomp["processed_smiles"]

            t = self.targets[i]
            if isinstance(t, list) or isinstance(t, np.ndarray):
                data.target = torch.tensor(t, dtype=torch.float)
            else:
                data.target = torch.tensor([t], dtype=torch.float)

            # Multi_hop edges
            data.multi_hop_edges = [torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]]

            # Atom features map
            atom_feats_map = {}
            for k, arr in precomp["atom_features"].items():
                atom_feats_map[k] = torch.from_numpy(arr).long()
            data.atom_features_map = atom_feats_map

            # Chirality / cis / trans
            data.chiral_tensors = [torch.from_numpy(x).long() for x in precomp["chiral_tensors"]]
            data.cis_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]]
            data.trans_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]]

            data.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
            data.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()

            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class MyBatch(Batch):
    """
    Custom Batch class for molecular graphs.
    
    Handles proper batching of molecular graphs with BFS-based features
    and stereo/chemical information.
    """
    @staticmethod
    def from_data_list(data_list):
        """
        Collates multiple Data objects into a single batch
        with BFS offset/padding logic + offset chirality/cis/trans indices.
        """
        batch_size = len(data_list)
        if batch_size == 0:
            return Batch()

        num_hops = len(data_list[0].multi_hop_edges)

        # 1) Number of atoms per sample + offsets
        num_atoms_per_sample = [d.x.size(0) for d in data_list]
        atom_offsets = torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.cumsum(torch.tensor(num_atoms_per_sample[:-1], dtype=torch.long), dim=0)
        ])

        # 2) Collect + offset chirality/cis/trans
        shifted_chiral_centers = []
        shifted_cis_bonds = []
        shifted_trans_bonds = []
        for i, d in enumerate(data_list):
            offset = atom_offsets[i]
            # Chirality
            shifted_chiral_centers.extend(ch + offset for ch in d.chiral_tensors if ch.size(0) == 4)
            # Cis
            shifted_cis_bonds.extend(bond + offset for bond in d.cis_bonds_tensors)
            # Trans
            shifted_trans_bonds.extend(bond + offset for bond in d.trans_bonds_tensors)

        final_tetrahedral_chiral_tensor = (
            torch.stack(shifted_chiral_centers, dim=0)
            if shifted_chiral_centers
            else torch.empty((0, 4), dtype=torch.long)
        )
        cis_bonds_tensor = (
            torch.stack(shifted_cis_bonds, dim=0)
            if shifted_cis_bonds
            else torch.empty((0, 2), dtype=torch.long)
        )
        trans_bonds_tensor = (
            torch.stack(shifted_trans_bonds, dim=0)
            if shifted_trans_bonds
            else torch.empty((0, 2), dtype=torch.long)
        )

        # Add reversed direction
        final_cis_tensor = (
            torch.cat([cis_bonds_tensor, cis_bonds_tensor[:, [1, 0]]], dim=0)
            if cis_bonds_tensor.numel() > 0
            else torch.empty((0, 2), dtype=torch.long)
        )
        final_trans_tensor = (
            torch.cat([trans_bonds_tensor, trans_bonds_tensor[:, [1, 0]]], dim=0)
            if trans_bonds_tensor.numel() > 0
            else torch.empty((0, 2), dtype=torch.long)
        )

        # 4) Concatenate atom features
        feature_keys = list(data_list[0].atom_features_map.keys())
        all_atom_features_map = {
            key: torch.cat([d.atom_features_map[key] for d in data_list], dim=0)
            for key in feature_keys
        }

        # 5) Create batch_indices
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long),
            torch.tensor(num_atoms_per_sample, dtype=torch.long)
        )

        # 6) Concatenate targets and total_charges
        first_target_dim = data_list[0].target.shape[0]
        is_all_same_dim = all(d.target.shape[0] == first_target_dim for d in data_list)
        if is_all_same_dim:
            targets = torch.stack([d.target for d in data_list], dim=0)
        else:
            targets = [d.target for d in data_list]

        total_charges = torch.cat([d.total_charge for d in data_list], dim=0)

        # 7) Collect smiles and atomic_numbers
        smiles_list = [d.smiles for d in data_list]
        batched_atomic_numbers = torch.cat([d.atomic_numbers for d in data_list], dim=0)

        # 8) Build final BFS edge indices
        shell_edge_indices_list = []
        for i, d in enumerate(data_list):
            offset_val = atom_offsets[i]
            for h_idx in range(num_hops):
                e = d.multi_hop_edges[h_idx]
                if e.numel() > 0:
                    shell_edge_indices_list.append(e + offset_val)

        if len(shell_edge_indices_list) > 0:
            shell_edge_indices = torch.cat(shell_edge_indices_list, dim=1).t()
        else:
            shell_edge_indices = torch.empty((0, 2), dtype=torch.long)

        # 9) Construct new Batch
        batch = Batch()
        batch.x = torch.cat([d.x for d in data_list], dim=0)
        batch.multi_hop_edge_indices = shell_edge_indices
        batch.batch_indices = batch_indices
        batch.atom_features_map = all_atom_features_map
        batch.targets = targets
        batch.total_charges = total_charges

        batch.final_tetrahedral_chiral_tensor = final_tetrahedral_chiral_tensor
        batch.final_cis_tensor = final_cis_tensor
        batch.final_trans_tensor = final_trans_tensor
        batch.smiles_list = smiles_list

        # For PyG usage
        batch.batch = batch_indices
        batch.atomic_numbers = batched_atomic_numbers

        return batch


class HDF5MolecularIterableDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for large-scale molecular data stored in HDF5 format.
    
    Allows efficient streaming of molecular data without loading everything into memory.
    
    Args:
        hdf5_path: Path to HDF5 file
        shuffle: Whether to shuffle data during iteration
        buffer_size: Size of shuffle buffer
        ddp_enabled: Whether distributed data parallel is enabled
        rank: Process rank for DDP
        world_size: Total number of processes for DDP
        fold_indices: Indices to use for cross-validation
        cv_fold: Current cross-validation fold
        seed: Random seed for shuffling (default: 42)
    """
    def __init__(self,
                 hdf5_path: str,
                 shuffle: bool = False,
                 buffer_size: int = 1000,
                 ddp_enabled: bool = False,
                 rank: int = 0,
                 world_size: int = 1,
                 fold_indices: List[int] = None,
                 cv_fold: int = None,
                 seed: int = 42):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.ddp_enabled = ddp_enabled
        self.rank = rank
        self.world_size = world_size
        self.fold_indices = fold_indices
        self.cv_fold = cv_fold
        self.seed = seed
        self._length = None
        self.index_map = None
        self.cv_splits = None
        
        # Pre-load metadata outside the iterator
        self._load_metadata()
        
    def _load_metadata(self):
        """Load HDF5 metadata only once to avoid repeated file access."""
        with h5py.File(self.hdf5_path, "r") as f:
            self._length = f["data"].shape[0]
            
            # Load index_map if available
            if "index_map" in f:
                self.index_map = f["index_map"][:]
            else:
                self.index_map = np.arange(self._length)
            
            # Load cv_splits if they exist
            if "cv_splits" in f:
                self.cv_splits = {}
                cv_splits_group = f["cv_splits"]
                for fold_name in cv_splits_group:
                    fold_group = cv_splits_group[fold_name]
                    self.cv_splits[fold_name] = {
                        "train_indices": fold_group["train_indices"][:],
                        "val_indices": fold_group["val_indices"][:]
                    }

        # If cv_fold is specified, use indices from that fold
        if self.cv_fold is not None and self.cv_splits is not None:
            fold_key = f"fold_{self.cv_fold}"
            if fold_key in self.cv_splits:
                # Use train or val indices based on whether fold_indices is None
                # If fold_indices is None, we're in training mode
                # If fold_indices is not None, it should be set to train or val indices
                self.fold_indices = self.cv_splits[fold_key]["train_indices"]

    def __len__(self):
        # Calculate length based on fold_indices if available
        if self.fold_indices is not None:
            return len(self.fold_indices)
        elif self.ddp_enabled:
            # Ensure even distribution of samples across processes
            return self._length // self.world_size + (1 if self._length % self.world_size > self.rank else 0)
        else:
            return self._length

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        
        # Set deterministic seed for this rank/worker combination
        if self.shuffle:
            # Create a unique seed for this process and epoch
            # This ensures consistent but different shuffling across processes and epochs
            epoch_seed = torch.initial_seed()  # This changes each epoch
            process_seed = self.seed + self.rank * 10000  # Different for each process
            combined_seed = (epoch_seed + process_seed) % (2**32 - 1)  # Combine seeds
            rng = random.Random(combined_seed)
        
        # Determine indices to process
        if self.fold_indices is not None:
            total_indices = self.fold_indices
        else:
            total_indices = list(range(len(self.index_map)))

        # Shuffle indices once at the dataset level (if needed)
        # This ensures the same shuffling across all workers within the same process
        if self.shuffle:
            # Use the seeded RNG instance to shuffle
            indices_copy = total_indices.copy()  # Make a copy to avoid modifying the original
            rng.shuffle(indices_copy)
            total_indices = indices_copy

        # Distribute indices across processes (DDP)
        if self.ddp_enabled:
            # Ensure consistent partitioning across processes
            num_samples = len(total_indices)
            chunk_size = int(math.ceil(num_samples / float(self.world_size)))
            
            # Get indices for this process
            start_idx = self.rank * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            process_indices = total_indices[start_idx:end_idx]
        else:
            process_indices = total_indices

        # Further distribute across workers within this process
        if worker_info is not None:
            # Divide process indices among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            per_worker = int(math.ceil(len(process_indices) / float(num_workers)))
            worker_start = worker_id * per_worker
            worker_end = min(worker_start + per_worker, len(process_indices))
            
            # Final indices for this specific worker
            final_indices = process_indices[worker_start:worker_end]
        else:
            final_indices = process_indices

        # Open the HDF5 file for reading
        f = h5py.File(self.hdf5_path, "r")
        dset = f["data"]
        
        try:
            # Generate samples
            for idx in final_indices:
                mapped_idx = self.index_map[idx]  # Use the index_map
                raw = dset[mapped_idx]
                try:
                    decoded = pickle.loads(raw.tobytes())
                    data_obj = self._build_data_object(decoded)
                    if data_obj is not None:
                        yield data_obj
                except Exception as e:
                    print(f"Error processing index {idx} (mapped to {mapped_idx}): {str(e)}")
                    continue
        finally:
            # Ensure file is properly closed
            f.close()

    def _build_data_object(self, item):
        """
        Convert raw BFS result into a PyG Data object.
        """
        if item is None or (item.get('precomputed') is None):
            return None

        smi = item['smiles']
        tgt = item['target']
        precomp = item['precomputed']

        num_atoms = precomp['atom_features']['atom_type'].shape[0]
        x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)

        data_obj = Data()
        data_obj.x = x_dummy
        data_obj.smiles = smi

        if isinstance(tgt, list):
            data_obj.target = torch.tensor(tgt, dtype=torch.float)
        else:
            data_obj.target = torch.tensor([tgt], dtype=torch.float)

        data_obj.multi_hop_edges = [
            torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]
        ]

        atom_feats_map = {}
        for k, arr in precomp["atom_features"].items():
            atom_feats_map[k] = torch.from_numpy(arr).long()
        data_obj.atom_features_map = atom_feats_map

        data_obj.chiral_tensors = [
            torch.from_numpy(x).long() for x in precomp["chiral_tensors"]
        ]
        data_obj.cis_bonds_tensors = [
            torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]
        ]
        data_obj.trans_bonds_tensors = [
            torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]
        ]

        data_obj.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
        data_obj.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()

        return data_obj

## Data loading functions:
def load_dataset_simple(
    file_path: str,
    smiles_column: str,
    target_column: str
) -> Tuple[List[str], List[float]]:
    """
    Load a simple dataset from CSV with one target.
    
    Args:
        file_path: Path to CSV file
        smiles_column: Column name for SMILES strings
        target_column: Column name for target values
        
    Returns:
        Tuple of (smiles_list, target_values)
    """
    df = pd.read_csv(file_path)
    smiles_list = df[smiles_column].tolist()
    target_values = df[target_column].tolist()
    return smiles_list, target_values

def load_dataset_multitask(
    file_path: str,
    smiles_column: str,
    multi_target_columns: List[str]
) -> Tuple[List[str], List[List[float]]]:
    """
    Load a multi-task dataset from CSV.
    
    Args:
        file_path: Path to CSV file
        smiles_column: Column name for SMILES strings
        multi_target_columns: List of column names for multiple targets
        
    Returns:
        Tuple of (smiles_list, target_values) where target_values is a list of lists
    """
    df = pd.read_csv(file_path)
    smiles_list = df[smiles_column].tolist()
    target_values = df[multi_target_columns].values.tolist()
    return smiles_list, target_values

def split_dataset(
    smiles_list: List[str],
    target_values: List[Any],
    train_split: float,
    val_split: float,
    test_split: float,
    task_type='regression'
):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        smiles_list: List of SMILES strings
        target_values: List of target values
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        task_type: Type of task ('regression' or 'multitask')
        
    Returns:
        Tuple of (smiles_train, target_train, smiles_val, target_val, smiles_test, target_test)
    """
    train_val_split = train_split + val_split
    smiles_train_val, smiles_test, target_train_val, target_test = train_test_split(
        smiles_list, target_values, test_size=test_split, random_state=42
    )
    smiles_train, smiles_val, target_train, target_val = train_test_split(
        smiles_train_val, target_train_val,
        test_size=val_split / train_val_split, random_state=42
    )
    return smiles_train, target_train, smiles_val, target_val, smiles_test, target_test


# Dataloader Creation and Collate Functions

def subset_in_memory_dataset(dataset: PyGSMILESDataset, indices: List[int]) -> PyGSMILESDataset:
    """
    Creates a new PyGSMILESDataset containing only the data objects at the specified indices.
    
    Args:
        dataset: Original dataset
        indices: Indices to include in subset
        
    Returns:
        New dataset containing only the specified indices
    """
    subset_smiles = [dataset.smiles_list[i] for i in indices]
    subset_targets = [dataset.targets[i] for i in indices]
    subset_precomputed = [dataset.precomputed_data[i] for i in indices]
    return PyGSMILESDataset(subset_smiles, subset_targets, subset_precomputed)

def iterable_collate_fn(batch_list):
    """Collate function for iterable datasets, filtering out None values."""
    filtered = [b for b in batch_list if b is not None]
    if len(filtered) == 0:
        return None
    return MyBatch.from_data_list(filtered)

def create_pyg_dataloader(
    dataset: PyGSMILESDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None
):
    """
    Creates a PyTorch DataLoader for an InMemoryDataset.
    
    Args:
        dataset: PyG dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        sampler: Optional sampler (e.g., for distributed training)
        
    Returns:
        DataLoader for the dataset
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        collate_fn=MyBatch.from_data_list,
        sampler=sampler
    )

def create_iterable_pyg_dataloader(
    hdf5_path: str, 
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    shuffle_buffer_size: int,
    ddp_enabled: bool = False,
    rank: int = 0,
    world_size: int = 1
):
    """
    Creates a DataLoader for an HDF5MolecularIterableDataset.
    
    Args:
        hdf5_path: Path to HDF5 file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        shuffle_buffer_size: Size of shuffle buffer for streamed shuffling
        ddp_enabled: Whether DDP is enabled
        rank: Process rank for DDP
        world_size: World size for DDP
        
    Returns:
        DataLoader for the dataset
    """
    dataset = HDF5MolecularIterableDataset(
        hdf5_path=hdf5_path, 
        shuffle=shuffle, 
        buffer_size=shuffle_buffer_size,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size
    )
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=iterable_collate_fn
    )

def check_and_create_hdf5_directories(args):
    """Create directories for HDF5 files if they don't exist yet"""
    import os

    # Handle separate HDF5 files (default behavior)
    for path in [args.train_hdf5, args.val_hdf5, args.test_hdf5]:
        if path:
            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)


# Graph & Feature Extraction Functions
@njit
def build_numba_adjacency_list(adj_matrix: np.ndarray):
    """Build adjacency list from matrix for fast BFS using numba."""
    n = adj_matrix.shape[0]
    adjacency_list = NumbaList()
    for v in range(n):
        row_nonzero = np.where(adj_matrix[v] > 0)[0]
        nbr_list = NumbaList()
        for nbr in row_nonzero:
            if nbr != v:  # skip self-loop
                nbr_list.append(nbr)
        adjacency_list.append(nbr_list)
    return adjacency_list

@njit
def compute_multi_hop_edges_bfs_numba(adj_list, max_hops):
    """
    Produces the same hop-by-hop edges as adjacency exponentiation,
    using a BFS frontier in edge-space. Returns a list of (2 x E) arrays.
    """
    n = len(adj_list)
    visited = np.zeros((n, n), dtype=boolean)

    # Hop 1
    hop1_list = []
    for v in range(n):
        for w in adj_list[v]:
            if not visited[v, w]:
                visited[v, w] = True
                hop1_list.append((v, w))
    
    hop1_array = np.empty((2, len(hop1_list)), dtype=np.int32)
    for i, (src, dst) in enumerate(hop1_list):
        hop1_array[0, i] = src
        hop1_array[1, i] = dst
    
    results = [hop1_array]
    frontier = hop1_list

    # Hops 2..max_hops
    for _hop in range(1, max_hops):
        new_edges = []
        for (u, v) in frontier:
            # expand from v -> w
            neighbors_v = adj_list[v]
            for w in neighbors_v:
                if w != u:
                    if not visited[u, w]:
                        visited[u, w] = True
                        new_edges.append((u, w))

        if len(new_edges) == 0:
            empty_arr = np.empty((2, 0), dtype=np.int32)
            results.append(empty_arr)
            break

        arr = np.empty((2, len(new_edges)), dtype=np.int32)
        for i, (src, dst) in enumerate(new_edges):
            arr[0, i] = src
            arr[1, i] = dst

        results.append(arr)
        frontier = new_edges

    while len(results) < max_hops:
        results.append(np.empty((2, 0), dtype=np.int32))

    return results



def compute_all(smiles: str, max_hops: int) -> Dict[str, Any] or None:
    """
    Compute multi-hops + features in one pass.
    Return None if SMILES invalid/unparseable or any essential step fails.
    
    Args:
        smiles: SMILES string for a molecule
        max_hops: Maximum number of hops for BFS
        
    Returns:
        Dictionary containing molecular features or None if processing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add H's and assign stereochemistry
    try:
        mol = Chem.AddHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        processed_smi = Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)
    except:
        return None

    # 1) Multi-hop BFS edges
    try:
        adj_matrix = Chem.GetAdjacencyMatrix(mol).astype(np.int32)
        adj_list_numba = build_numba_adjacency_list(adj_matrix)
        edge_indices_list = compute_multi_hop_edges_bfs_numba(adj_list_numba, max_hops)
        multi_hop_edges = edge_indices_list
    except:
        return None

    # 2) Atom features
    atom_features_list = []
    try:
        for atom in mol.GetAtoms():
            atom_type = atom.GetAtomicNum()
            degree = atom.GetTotalDegree()
            hydrogen_count = atom.GetTotalNumHs(includeNeighbors=True)
            hybridization = atom.GetHybridization()
            atom_features_list.append({
                'atom_type': atom_type,
                'hydrogen_count': hydrogen_count,
                'degree': degree,
                'hybridization': hybridization,
            })
    except:
        return None

    # Store atomic numbers in a single array
    try:
        atomic_numbers_array = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=np.int32
        )
    except:
        return None

    # 3) Chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    chiral_tensors = []
    for center_idx, chirality in chiral_centers:
        center_atom = mol.GetAtomWithIdx(center_idx)
        neighbors = [nbr.GetIdx() for nbr in center_atom.GetNeighbors()]
        chiral_tensors.append(np.array(neighbors, dtype=np.int32))

    # 4) Cis/Trans bonds
    cis_bonds_list = []
    trans_bonds_list = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo not in [Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE]:
                # skip STEREONONE, STEREOANY, etc.
                continue

            start_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Build neighbor lists excluding the double-bond partner
            start_neighbors = [nbr.GetIdx() for nbr in start_atom.GetNeighbors()
                               if nbr.GetIdx() != end_atom.GetIdx()]
            end_neighbors = [nbr.GetIdx() for nbr in end_atom.GetNeighbors()
                               if nbr.GetIdx() != start_atom.GetIdx()]

            # skip "symmetric" or near-symmetric bonds
            if len(set(start_neighbors + end_neighbors)) < 4:
                continue

            stereo_atoms = bond.GetStereoAtoms()
            if len(stereo_atoms) != 2:
                continue

            s_high = stereo_atoms[0]
            e_high = stereo_atoms[1]

            # Identify the "low" substituent on each side
            s_low_candidates = [x for x in start_neighbors if x != s_high]
            if not s_low_candidates:
                continue
            s_low = min(s_low_candidates, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

            e_low_candidates = [x for x in end_neighbors if x != e_high]
            if not e_low_candidates:
                continue
            e_low = min(e_low_candidates, key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())

            if stereo == Chem.BondStereo.STEREOE:  # E => opposite
                trans_bonds_list.append([s_high, e_high])  
                trans_bonds_list.append([s_low, e_low])    
                trans_bonds_list.append([e_high, s_high])  
                trans_bonds_list.append([e_low, s_low])

                # cross pairs = cis
                cis_bonds_list.append([s_high, e_low])
                cis_bonds_list.append([s_low, e_high])
                cis_bonds_list.append([e_low, s_high])
                cis_bonds_list.append([e_high, s_low])

            elif stereo == Chem.BondStereo.STEREOZ:  # Z => same
                cis_bonds_list.append([s_high, e_high])
                cis_bonds_list.append([s_low, e_low])
                cis_bonds_list.append([e_high, s_high])
                cis_bonds_list.append([e_low, s_low])

                # cross pairs = trans
                trans_bonds_list.append([s_high, e_low])
                trans_bonds_list.append([s_low, e_high])
                trans_bonds_list.append([e_low, s_high])
                trans_bonds_list.append([e_high, s_low])

    # 5) Total formal charge
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # 6) Convert atom features to index arrays
    mapped_atom_features = {
        'atom_type': [],
        'hydrogen_count': [],
        'degree': [],
        'hybridization': [],
    }

    for feat in atom_features_list:
        # Atomic number
        a_type = feat['atom_type']
        a_type_idx = ATOM_TYPES.index(a_type) if a_type in ATOM_TYPES else len(ATOM_TYPES)
        
        # Hydrogen count (create index on the fly)
        h_count = feat['hydrogen_count']
        h_count_idx = min(h_count, 8)  # Cap at 8 hydrogens (reasonable limit)
        
        # Degree
        deg = feat['degree']
        deg_idx = DEGREES.index(deg) if deg in DEGREES else len(DEGREES)
        
        # Hybridization
        hyb = feat['hybridization']
        hyb_idx = HYBRIDIZATIONS.index(hyb) if hyb in HYBRIDIZATIONS else len(HYBRIDIZATIONS)

        mapped_atom_features['atom_type'].append(a_type_idx)
        mapped_atom_features['hydrogen_count'].append(h_count_idx)
        mapped_atom_features['degree'].append(deg_idx)
        mapped_atom_features['hybridization'].append(hyb_idx)

    for k in mapped_atom_features:
        mapped_atom_features[k] = np.array(mapped_atom_features[k], dtype=np.int8)

    chiral_tensors = [np.array(x, dtype=np.int32) for x in chiral_tensors]
    cis_bonds_tensors = [np.array(x, dtype=np.int32) for x in cis_bonds_list]
    trans_bonds_tensors = [np.array(x, dtype=np.int32) for x in trans_bonds_list]

    return {
        "multi_hop_edges": multi_hop_edges,
        "atom_features": mapped_atom_features,
        "chiral_tensors": chiral_tensors,
        "cis_bonds_tensors": cis_bonds_tensors,
        "trans_bonds_tensors": trans_bonds_tensors,
        "total_charge": total_charge,
        "atomic_numbers": atomic_numbers_array,
        "processed_smiles": processed_smi
    }

# Worker Functions for Parallel Processing

def _worker_bfs(smiles_and_target, max_hops):
    """Worker function for parallel feature computation."""
    smi, tgt = smiles_and_target
    precomp = compute_all(smi, max_hops)
    if precomp is None:
        return None
    return {
        'smiles': smi,
        'target': tgt,
        'precomputed': precomp
    }

def _worker_process_smiles(item):
    """
    Worker function for processing SMILES in parallel.
    
    Args:
        item: Tuple containing (idx, smiles, max_hops)
    
    Returns:
        Tuple of (idx, precomp) where precomp is the result of compute_all
        or None if the SMILES couldn't be processed
    """
    idx, smiles, max_hops = item
    precomp = compute_all(smiles, max_hops)
    return (idx, precomp)

