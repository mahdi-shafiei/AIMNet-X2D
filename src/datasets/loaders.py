# datasets/loaders.py
"""
Data loader creation and collate functions for molecular datasets.
"""

from torch.utils.data import DataLoader
from .molecular import PyGSMILESDataset, HDF5MolecularIterableDataset, MyBatch


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
    world_size: int = 1,
    preprocessing_pipeline = None
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
        preprocessing_pipeline: Optional preprocessing pipeline for raw data
        
    Returns:
        DataLoader for the dataset
    """
    dataset = HDF5MolecularIterableDataset(
        hdf5_path=hdf5_path, 
        shuffle=shuffle, 
        buffer_size=shuffle_buffer_size,
        ddp_enabled=ddp_enabled,
        rank=rank,
        world_size=world_size,
        preprocessing_pipeline=preprocessing_pipeline
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=iterable_collate_fn
    )