# main.py

# Standard Libraries
import os
import sys
import argparse
import copy
import uuid
import time
import math
import yaml

# Third-Party Libraries
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import wandb
import tqdm
from datetime import datetime
import torch.nn as nn

# Local Module Imports
from utils import set_seed, safe_get_rank, is_main_process
from datasets import (
    load_dataset_simple,
    load_dataset_multitask,
    split_dataset,
    precompute_all_and_filter,
    precompute_and_write_hdf5_parallel_chunked,
    create_pyg_dataloader,
    create_iterable_pyg_dataloader,
    PyGSMILESDataset,
    ATOM_TYPES,
    DEGREES,
    HYBRIDIZATIONS,
    partial_parse_atomic_numbers,
    compute_sae_dict_from_atomic_numbers_list
)
from model import (
    GNN,
    SizeExtensiveNormalizer,
    MultiTaskSAENormalizer,
    MultiTargetStandardScaler,
    WeightedL1Loss
)
from training import (
    train_gnn,
    evaluate,
    extract_partial_charges,
    extract_embeddings_main
)
from inference import inference_main
from config import validate_args
from hyperparameter import _sample_hparam_value


def main(args):
    """
    Main function that handles the complete pipeline:
    - Data loading and processing
    - Model creation
    - Training and evaluation
    - Inference
    - Hyperparameter tuning
    
    Args:
        args: Command-line arguments
    """
    import torch.distributed as dist
    import copy
    import uuid

    set_seed(42)
    is_ddp = False
    local_rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = args.num_gpu_devices

    best_trial_metrics = None
    best_trial_hyperparams = None
    best_trial_num = -1
    best_test_metrics = None

    multitask_weights = None

    validate_args(args)

    # Generate a unique group ID for this sweep
    group_id = str(uuid.uuid4())

    # Load hyperparameter configuration from file
    if args.hyperparameter_file:
        with open(args.hyperparameter_file, "r") as f:
            hparam_config = yaml.safe_load(f)
    else:
        hparam_config = {}  # Empty config if no file

    #     is_ddp = False
    # local_rank = 0
    # world_size = 1
    
    if args.num_gpu_devices > 1:
        # Initialize distributed environment variables
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            if dist.is_available():
                dist.init_process_group(backend="nccl")
                world_size = dist.get_world_size()
                is_ddp = True
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
                print(f"[DDP] rank={dist.get_rank()} local_rank={local_rank} world_size={world_size}")
            else:
                print("torch.distributed is not available, falling back to single GPU")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("LOCAL_RANK not found in environment, falling back to single GPU")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single-Process] Using device: {device}")

    # Check if we're in inference mode and prioritize it
    if args.inference_csv or args.inference_hdf5:
        if not is_ddp or dist.get_rank() == 0:
            print("Running in inference mode - training will be skipped")
            
            # Auto-generate output path if not provided
            if not args.inference_output and args.inference_csv:
                args.inference_output = os.path.splitext(args.inference_csv)[0] + "_predictions.csv"
                print(f"Using auto-generated output path: {args.inference_output}")
                
            # Ensure output directory exists
            if args.inference_output:
                os.makedirs(os.path.dirname(os.path.abspath(args.inference_output)), exist_ok=True)
                
        inference_main(args, device, is_ddp, local_rank, world_size)
        
        if is_ddp:
            dist.barrier()
            
        if is_ddp and dist.is_initialized():
            dist.destroy_process_group()
            
        print("Inference completed successfully.")
        return



    # --- Data Loading (CORRECTED) ---
    if args.data_path:
        # ... (same as before, single CSV case) ...
        if args.train_data or args.val_data or args.test_data:
            raise ValueError("Cannot specify both --data_path and individual train/val/test dataset paths.")
        print(f"Reading data from {args.data_path} ...")

        if args.task_type == 'multitask':
            if args.multi_target_columns is None:
                raise ValueError("For --task_type=multitask, must specify --multi_target_columns.")
            multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]
            smiles_list, target_values = load_dataset_multitask(args.data_path, args.smiles_column, multi_cols)
        else:
            smiles_list, target_values = load_dataset_simple(args.data_path, args.smiles_column, args.target_column)

        print(f"Splitting dataset with train: {args.train_split}, val: {args.val_split}, test: {args.test_split}")
        smiles_train, target_train, smiles_val, target_val, smiles_test, target_test = split_dataset(
            smiles_list, target_values, args.train_split, args.val_split, args.test_split, task_type=args.task_type
        )
        # combine train/val for cross validation splitting later
        smiles_train_val = smiles_train + smiles_val
        target_train_val = target_train + target_val

    else:  # Separate train/val/test CSVs
        if not (args.train_data and args.val_data and args.test_data):
            raise ValueError("Must specify either --data_path or all of --train_data, --val_data, and --test_data.")

        print("Using separate CSV for train/val/test.")
        if args.task_type == 'multitask':
            if args.multi_target_columns is None:
                raise ValueError("For --task_type=multitask, must specify --multi_target_columns.")
            multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]
            smiles_train, target_train = load_dataset_multitask(args.train_data, args.smiles_column, multi_cols)
            smiles_val, target_val = load_dataset_multitask(args.val_data, args.smiles_column, multi_cols)
            smiles_test, target_test = load_dataset_multitask(args.test_data, args.smiles_column, multi_cols)
        else:
            smiles_train, target_train = load_dataset_simple(args.train_data, args.smiles_column, args.target_column)
            smiles_val, target_val = load_dataset_simple(args.val_data, args.smiles_column, args.target_column)
            smiles_test, target_test = load_dataset_simple(args.test_data, args.smiles_column, args.target_column)

        # DO NOT combine train/val here.  Keep them separate.
        target_train_val = target_train + target_val


    max_hops = args.num_shells

    # Decide if we have multi_cols or not:
    if args.task_type == 'multitask':
        if args.multi_target_columns is None:
            raise ValueError("For --task_type=multitask, must specify --multi_target_columns.")
        multi_cols = [c.strip() for c in args.multi_target_columns.split(',')]

        if args.multitask_weights is not None:
            wlist = [float(x.strip()) for x in args.multitask_weights.split(',')]
            multitask_weights = wlist  # Now defined in the correct scope

        elif args.multitask_weights is None:
            multitask_weights = np.ones(len(multi_cols), dtype=float)
    else:
        # single-task (regression or classification)
        multi_cols = None

    # NEW: Initialize an overall Wandb run for the entire hyperparameter sweep
    if (not is_ddp or dist.get_rank() == 0) and args.enable_wandb:
        sweep_wandb_run = wandb.init(
            project=args.wandb_project,
            group=group_id,  # Use the group ID
            name="Hyperparameter Sweep"  # A descriptive name for the sweep run
        )
        print(f"Wandb sweep initialized for rank 0. Project: {sweep_wandb_run.project}, Run ID: {sweep_wandb_run.id}, Group ID: {group_id}")


    # Possibly apply standard scaling (BEFORE dataset creation)
    if args.task_type in ['regression', 'multitask']:
        # Scale train_val together
        Y_train_val = np.array(target_train_val, dtype=np.float32)
        if len(Y_train_val.shape) == 1:
            Y_train_val = Y_train_val.reshape(-1, 1)
        std_scaler = MultiTargetStandardScaler()
        Y_train_val_scaled = std_scaler.fit_transform(Y_train_val)

        # Apply the SAME scaling to test data
        Y_test = np.array(target_test, dtype=np.float32)
        if len(Y_test.shape) == 1:
            Y_test = Y_test.reshape(-1, 1)
        Y_test_scaled = std_scaler.transform(Y_test)  # Use transform, not fit_transform

       # Convert back to lists
        if Y_train_val_scaled.shape[1] == 1:
            target_train_val = [float(x[0]) for x in Y_train_val_scaled]
            target_test = [float(x[0]) for x in Y_test_scaled]  # Apply to test as well
        else:
            target_train_val = Y_train_val_scaled.tolist()
            target_test = Y_test_scaled.tolist() # Apply to test

        num_tasks = Y_train_val.shape[1]

        #split train_val back into train and val AFTER scaling
        target_train = target_train_val[:len(smiles_train)]
        target_val = target_train_val[len(smiles_train):]

    else:
        num_tasks = 1
        std_scaler = None

    # --- Precompute Features and BFS (CORRECTED for separate datasets) ---
    if not args.iterable_dataset:
        print("==== Using the original in-memory PyGSMILESDataset approach ====")
        # Process train, val, and test SEPARATELY
        smiles_train_valid, train_targets_valid, train_precomputed = precompute_all_and_filter(
            smiles_train, target_train, max_hops, num_workers=args.num_workers
        )
        smiles_val_valid, val_targets_valid, val_precomputed = precompute_all_and_filter(
            smiles_val, target_val, max_hops, num_workers=args.num_workers
        )
        smiles_test_valid, test_targets_valid, test_precomputed = precompute_all_and_filter(
            smiles_test, target_test, max_hops, num_workers=args.num_workers
        )

        # Create SEPARATE datasets
        train_dataset = PyGSMILESDataset(smiles_train_valid, train_targets_valid, train_precomputed)
        val_dataset = PyGSMILESDataset(smiles_val_valid, val_targets_valid, val_precomputed)
        test_dataset = PyGSMILESDataset(smiles_test_valid, test_targets_valid, test_precomputed)

        # SAE (CORRECTED for separate datasets)
        if args.calculate_sae and args.task_type == 'regression':
            print("Applying SAE normalization (once) to single-task regression datasets ...")
            sae_normalizer = SizeExtensiveNormalizer()
            # Calculate SAE using only the TRAINING data
            sae_normalizer.calc_sae_from_dataset(train_dataset)
            sae_normalizer.normalize_dataset(train_dataset)  # Normalize train
            sae_normalizer.normalize_dataset(val_dataset)    # Normalize val
            sae_normalizer.normalize_dataset(test_dataset)   # Normalize test
        elif args.calculate_sae and args.task_type == 'multitask':
            if args.sae_subtasks is not None:
                subtask_indices = [int(x.strip()) for x in args.sae_subtasks.split(',')]
                print(f"Applying SAE normalization (per-subtask) for multitask. Subtasks = {subtask_indices}")
                multi_sae = MultiTaskSAENormalizer(subtask_indices)
                # Calculate SAE using only the TRAINING data
                multi_sae.calc_sae_from_dataset(train_dataset)
                multi_sae.normalize_dataset(train_dataset)  # Normalize train
                multi_sae.normalize_dataset(val_dataset)    # Normalize val
                multi_sae.normalize_dataset(test_dataset)   # Normalize test
            else:
                print("SAE subtasks not specified for multitask => skipping SAE.")
    # --- End of one-time calculations ---

    # --- Precompute Features and BFS for HDF5 ---
    if args.iterable_dataset:
        print("==== Using the HDF5 + IterableDataset approach ====")
        precompute_workers = args.precompute_num_workers if args.precompute_num_workers is not None else args.num_workers

        # Determine which HDF5 files to use and if precomputation is needed
        # Using separate HDF5 files
        train_hdf5_path = args.train_hdf5
        val_hdf5_path = args.val_hdf5
        test_hdf5_path = args.test_hdf5
        
        # Check if any of the files need to be created
        need_precompute = (not os.path.exists(train_hdf5_path)) or \
                        (not os.path.exists(val_hdf5_path)) or \
                        (not os.path.exists(test_hdf5_path))


        # If files don't exist, precompute and create them
        if need_precompute:
            if not is_ddp or dist.get_rank() == 0:
                print("One or more HDF5 files not found. Generating BFS => HDF5...")
                
                #------------------------------
                # SAE Normalization Logic
                #------------------------------
                # For single-task regression
                if args.calculate_sae and args.task_type == 'regression':
                    if not is_ddp or dist.get_rank() == 0:
                        print("Gathering atomic numbers once for train to compute SAE...")
                    
                    # Collect atomic numbers and targets from training data
                    if not is_ddp or dist.get_rank() == 0:
                        train_nums = []
                        good_train_smiles = []
                        good_train_targets = []
                        
                        for smi, tgt in tqdm.tqdm(zip(smiles_train, target_train), total=len(smiles_train)):
                            nums = partial_parse_atomic_numbers(smi)
                            if nums is not None:
                                train_nums.append(nums)
                                good_train_smiles.append(smi)
                                good_train_targets.append(tgt)

                        # Compute SAE dictionary using ONLY training data
                        sae_dict = compute_sae_dict_from_atomic_numbers_list(train_nums, good_train_targets)
                        print("SAE dictionary computed from training data:", sae_dict)
                        print("Applying shift to train, val, test...")

                        # Apply SAE shift to training data
                        shifted_train_targets = []
                        c_idx = 0
                        for smi, original_tgt in zip(smiles_train, target_train):
                            if c_idx < len(good_train_smiles) and smi == good_train_smiles[c_idx]:
                                shift_val = sum(sae_dict.get(n, 0.0) for n in train_nums[c_idx])
                                shifted_train_targets.append(original_tgt - shift_val)
                                c_idx += 1
                            else:
                                shifted_train_targets.append(original_tgt)
                        
                        # Apply SAE shift to validation data
                        val_nums = []
                        good_val_targets = []
                        for smi, tgt in tqdm.tqdm(zip(smiles_val, target_val), total=len(smiles_val)):
                            nums = partial_parse_atomic_numbers(smi)
                            val_nums.append(nums)
                            good_val_targets.append(tgt)
                            
                        shifted_val_targets = []
                        for nums, tval in zip(val_nums, good_val_targets):
                            shift = 0.0
                            if nums is not None:
                                for n in nums:
                                    shift += sae_dict.get(n, 0.0)
                            shifted_val_targets.append(tval - shift)

                        # Apply SAE shift to test data
                        test_nums = []
                        good_test_targets = []
                        for smi, tgt in tqdm.tqdm(zip(smiles_test, target_test), total=len(smiles_test)):
                            nums = partial_parse_atomic_numbers(smi)
                            test_nums.append(nums)
                            good_test_targets.append(tgt)
                            
                        shifted_test_targets = []
                        for nums, tval in zip(test_nums, good_test_targets):
                            shift = 0.0
                            if nums is not None:
                                for n in nums:
                                    shift += sae_dict.get(n, 0.0)
                            shifted_test_targets.append(tval - shift)
                        

                    if is_ddp:
                        dist.barrier()

                # For multi-task regressions with per-task SAE
                elif args.calculate_sae and args.task_type == 'multitask':
                    if args.sae_subtasks is not None:
                        subtask_indices = [int(x.strip()) for x in args.sae_subtasks.split(',')]
                        
                        if not is_ddp or dist.get_rank() == 0:
                            print(f"SAE normalization per-subtask for multitask. Subtasks={subtask_indices}")
                        
                        if not is_ddp or dist.get_rank() == 0:
                            # Convert targets to arrays for easier manipulation
                            train_array = np.array(target_train, dtype=np.float64)
                            val_array = np.array(target_val, dtype=np.float64)
                            test_array = np.array(target_test, dtype=np.float64)
                            
                            # Parse atomic numbers for all datasets
                            print("Partial parse for train set...")
                            train_atomic_nums = []
                            for smi in tqdm.tqdm(smiles_train):
                                nums = partial_parse_atomic_numbers(smi)
                                train_atomic_nums.append(nums)
                            
                            print("Partial parse for val set...")
                            val_atomic_nums = []
                            for smi in tqdm.tqdm(smiles_val):
                                nums = partial_parse_atomic_numbers(smi)
                                val_atomic_nums.append(nums)

                            print("Partial parse for test set...")
                            test_atomic_nums = []
                            for smi in tqdm.tqdm(smiles_test):
                                nums = partial_parse_atomic_numbers(smi)
                                test_atomic_nums.append(nums)

                            # For each subtask that needs SAE normalization
                            for st in subtask_indices:
                                print(f"Subtask {st} => compute SAE...")
                                
                                # Collect data for the current subtask (from training data only)
                                train_subtask_targets = []
                                train_subtask_nums = []
                                for nums, tvals in zip(train_atomic_nums, train_array):
                                    if nums is not None:
                                        train_subtask_targets.append(tvals[st])
                                        train_subtask_nums.append(nums)

                                # Compute SAE dictionary for this subtask
                                sae_dict_st = compute_sae_dict_from_atomic_numbers_list(
                                    train_subtask_nums,
                                    train_subtask_targets
                                )
                                # print(f"  SAE Dict for subtask {st} ({multi_cols[st]} if available): {sae_dict_st}")

                                # Apply SAE shift to training data for this subtask
                                for i, nums in enumerate(train_atomic_nums):
                                    if nums is not None:
                                        shift_val = sum(sae_dict_st.get(n, 0.0) for n in nums)
                                        train_array[i, st] -= shift_val
                                
                                # Apply SAE shift to validation data for this subtask
                                for i, nums in enumerate(val_atomic_nums):
                                    if nums is not None:
                                        shift_val = sum(sae_dict_st.get(n, 0.0) for n in nums)
                                        val_array[i, st] -= shift_val

                                # Apply SAE shift to test data for this subtask
                                for i, nums in enumerate(test_atomic_nums):
                                    if nums is not None:
                                        shift_val = sum(sae_dict_st.get(n, 0.0) for n in nums)
                                        test_array[i, st] -= shift_val

                            # Convert arrays back to lists
                            target_train = train_array.tolist()
                            target_val = val_array.tolist()
                            target_test = test_array.tolist()
                            

                        if is_ddp:
                            dist.barrier()
                    else:
                        if not is_ddp or dist.get_rank() == 0:
                            print("SAE subtasks not specified for multitask => skipping SAE.")


                #------------------------------
                # HDF5 File Creation
                #------------------------------
                if not is_ddp or dist.get_rank() == 0:
                    # Write to separate HDF5 files
                    print(f"Writing train BFS to HDF5: {train_hdf5_path}")
                    precompute_and_write_hdf5_parallel_chunked(
                        smiles_train, 
                        shifted_train_targets if args.calculate_sae and args.task_type == 'regression' else target_train, 
                        max_hops, 
                        train_hdf5_path,
                        chunk_size=1000, 
                        num_workers=precompute_workers,
                        task_type=args.task_type,
                        multi_target_columns=multi_cols
                    )
                    
                    print(f"Writing validation BFS to HDF5: {val_hdf5_path}")
                    precompute_and_write_hdf5_parallel_chunked(
                        smiles_val, 
                        shifted_val_targets if args.calculate_sae and args.task_type == 'regression' else target_val, 
                        max_hops, 
                        val_hdf5_path,
                        chunk_size=1000, 
                        num_workers=precompute_workers,
                        task_type=args.task_type,
                        multi_target_columns=multi_cols
                    )

                    # Write test HDF5 (same logic for both modes)
                    print(f"Writing test BFS to HDF5: {test_hdf5_path}")
                    precompute_and_write_hdf5_parallel_chunked(
                        smiles_test, 
                        shifted_test_targets if args.calculate_sae and args.task_type == 'regression' else target_test, 
                        max_hops, 
                        test_hdf5_path,
                        chunk_size=1000, 
                        num_workers=precompute_workers,
                        task_type=args.task_type,
                        multi_target_columns=multi_cols
                    )
                
                if is_ddp:
                    dist.barrier()

    
    for trial_num in range(args.num_trials):

        # Sample hyperparameters for this trial
        current_hparams = {}
        for hparam_name, hparam_value in hparam_config.items():
            current_hparams[hparam_name] = _sample_hparam_value(hparam_value)

        # if num_gpus > 1:
        #     if dist.is_initialized():
        #         dist.destroy_process_group()  # Destroy if already initialized
        #     dist.init_process_group(backend="nccl")
        #     local_rank = int(os.environ["LOCAL_RANK"])
        #     torch.cuda.set_device(local_rank)
        #     device = torch.device("cuda", local_rank)
        #     is_ddp = True  # Set is_ddp to True when using DDP
        #     world_size = dist.get_world_size()
        #     print(
        #         f"[DDP] rank={dist.get_rank()} local_rank={local_rank} world_size={world_size}"
        #     )
        # else:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     print(f"[Single-Process] Using device: {device}")

        # Create a copy of args and update
        current_args = copy.deepcopy(args)
        for key, value in current_hparams.items():
            # boolean handling
            action = parser._registry_get('action', key, None)
            if action is not None and action.const is not None:
                # For boolean actions, set to True/False based on the string value
                if str(value).lower() == 'true':
                    setattr(current_args, key, True)
                elif str(value).lower() == 'false':
                    setattr(current_args, key, False)
                else:
                    raise ValueError(f"Invalid value for boolean argument '{key}': {value}")
            else:
                # For non-boolean arguments
                try:
                    current_value = getattr(current_args, key, None)
                    if current_value is None:
                        # If attribute doesn't exist or is None, just set it directly
                        setattr(current_args, key, value)
                    else:
                        # Convert value to the same type as the current value
                        setattr(current_args, key, type(current_value)(value))
                except Exception as e:
                    # Handle hyperparameters that might not be in the original args
                    print(f"Warning: Error setting hyperparameter {key}: {str(e)}. Setting anyway as is.")
                    setattr(current_args, key, value)

        # Set default value for enable_wandb if not present
        if not hasattr(current_args, "enable_wandb"):
            setattr(current_args, "enable_wandb", False)

        # Print the hyperparameters for this run
        if (not is_ddp or dist.get_rank() == 0):
            if args.hyperparameter_file != None:
                print(f"----- Starting trial {trial_num + 1} / {args.num_trials} -----")
                print(f"Hyperparameters for this run: {current_hparams}")

        # If user wants to do inference only, skip training
        if current_args.inference_csv or current_args.inference_hdf5:
            inference_main(current_args, device, is_ddp, local_rank, world_size)
            if (not is_ddp or dist.get_rank() == 0) and current_args.enable_wandb:
                wandb.finish()
            if is_ddp:
                dist.barrier()
            continue  # Skip to the next hyperparameter set

        # --- Cross-Validation Loop (CONDITIONAL) ---


        if not current_args.iterable_dataset:
            # --- Using precomputed data for InMemoryDataset ---
            if is_ddp:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, #use train dataset
                    num_replicas=world_size,
                    rank=safe_get_rank(),
                    shuffle=True,
                    drop_last=False
                )
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset, #use val dataset
                    num_replicas=world_size,
                    rank=safe_get_rank(),
                    shuffle=True,
                    drop_last=False
                )
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset,
                    num_replicas=world_size,
                    rank=safe_get_rank(),
                    shuffle=False,
                    drop_last=False
                )
            else:
                train_sampler = None
                val_sampler = None
                test_sampler = None

            train_loader = create_pyg_dataloader(
                dataset=train_dataset,  #  train dataset
                batch_size=current_args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=current_args.num_workers,
                sampler=train_sampler
            )
            val_loader = create_pyg_dataloader(
                dataset=val_dataset,  # val dataset
                batch_size=current_args.batch_size,
                shuffle=(val_sampler is None),
                num_workers=current_args.num_workers,
                sampler=val_sampler
            )
            test_loader = create_pyg_dataloader(
                dataset=test_dataset,
                batch_size=current_args.batch_size,
                shuffle=False,
                num_workers=current_args.num_workers,
                sampler=test_sampler
            )

        else:
            # --- Using HDF5 for IterableDataset ---
            # Use separate train and val HDF5 files
            train_hdf5_path = current_args.train_hdf5
            val_hdf5_path = current_args.val_hdf5
            test_hdf5_path = current_args.test_hdf5
            
            print(f"Using separate HDF5 files: train={train_hdf5_path}, val={val_hdf5_path}, test={test_hdf5_path}")
            
            # Create train loader from train.h5
            train_loader = create_iterable_pyg_dataloader(
                hdf5_path=train_hdf5_path,
                batch_size=current_args.batch_size,
                shuffle=True,
                num_workers=current_args.num_workers,
                shuffle_buffer_size=current_args.shuffle_buffer_size,
                ddp_enabled=is_ddp,
                rank=safe_get_rank(),
                world_size=world_size
            )
            
            # Create val loader from val.h5
            val_loader = create_iterable_pyg_dataloader(
                hdf5_path=val_hdf5_path,
                batch_size=current_args.batch_size,
                shuffle=False,  # No shuffle for validation
                num_workers=current_args.num_workers,
                shuffle_buffer_size=current_args.shuffle_buffer_size,
                ddp_enabled=is_ddp,
                rank=safe_get_rank(),
                world_size=world_size
            )
                

            # Test loader is the same for both modes
            test_loader = create_iterable_pyg_dataloader(
                hdf5_path=test_hdf5_path,
                batch_size=current_args.batch_size,
                shuffle=False,
                num_workers=current_args.num_workers,
                shuffle_buffer_size=current_args.shuffle_buffer_size,
                ddp_enabled=is_ddp,
                rank=safe_get_rank(),
                world_size=world_size
            )

        # Model creation, DDP wrapping, Training
        hidden_dim = current_args.hidden_dim
        output_dim = num_tasks
        feature_sizes = {
            'atom_type': len(ATOM_TYPES) + 1,
            'degree': len(DEGREES) + 1,
            'hybridization': len(HYBRIDIZATIONS) + 1,
            'hydrogen_count': 9
        }
        model = GNN(
            feature_sizes=feature_sizes,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_shells=current_args.num_shells,
            num_message_passing_layers=current_args.num_message_passing_layers,
            ffn_hidden_dim=current_args.ffn_hidden_dim,
            ffn_num_layers=current_args.ffn_num_layers,
            pooling_type=current_args.pooling_type,
            task_type=current_args.task_type,
            embedding_dim=current_args.embedding_dim,
            use_partial_charges=current_args.use_partial_charges,
            use_stereochemistry=current_args.use_stereochemistry,
            ffn_dropout=current_args.ffn_dropout,
            activation_type=current_args.activation_type,
            shell_conv_num_mlp_layers=current_args.shell_conv_num_mlp_layers,
            shell_conv_dropout=current_args.shell_conv_dropout,
            attention_num_heads=current_args.attention_num_heads,
            attention_temperature=current_args.attention_temperature
        )

        # Optionally load pretrained & freeze
        if current_args.transfer_learning is not None:
            if not is_ddp or dist.get_rank() == 0:
                print(f"Loading pretrained weights from {current_args.transfer_learning}...")
            pretrained_weights = torch.load(current_args.transfer_learning, map_location=device)
            model.load_state_dict(pretrained_weights, strict=False)
            if current_args.freeze_pretrained:
                if not is_ddp or dist.get_rank() == 0:
                    print("Freezing pretrained layers except final output layer.")
                for name, param in model.named_parameters():
                    if "output_layer" not in name:
                        param.requires_grad = False

        model.to(device)
        if is_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )

        # Train using train_val_loader and a dummy val_loader
        training_start = time.time()
        trained_model = train_gnn(
            model,
            train_loader,  # Use train loader
            val_loader, #  Use val_loader
            test_loader,
            current_args.epochs,
            current_args.learning_rate,
            device,
            early_stopping=current_args.early_stopping,
            task_type=current_args.task_type,
            mixed_precision=current_args.mixed_precision,
            num_tasks=output_dim,
            multitask_weights=multitask_weights,
            std_scaler=std_scaler,
            is_ddp=is_ddp,
            current_args=current_args  # Pass current_args
        )
        training_end = time.time()
        training_time = training_end - training_start


        if not is_ddp or dist.get_rank() == 0:
            print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate on the test set
        if current_args.task_type == 'classification':
            final_criterion = nn.BCEWithLogitsLoss()
        elif current_args.task_type == 'multitask':
            #if multitask_weights is not None:
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            final_criterion = WeightedL1Loss(w_tensor)
            # else:
            #     final_criterion = nn.L1Loss()
        else:
            final_criterion = nn.L1Loss()

        test_metrics = evaluate(
            trained_model,
            test_loader,
            final_criterion,
            device,
            task_type=current_args.task_type,
            sae_normalizer=None,
            mixed_precision=current_args.mixed_precision,
            num_tasks=output_dim,
            std_scaler=std_scaler,
            is_ddp=is_ddp
        )

        best_val_loss = float('inf')
        # No CV, so we use the test metrics for best model selection
        if test_metrics['loss'] < best_val_loss:
            best_val_loss = test_metrics['loss']  # Use test loss
            best_model_state = {k: v.cpu() for k, v in trained_model.state_dict().items()}
            best_hyperparams = current_args
            best_std_scaler = std_scaler



        # --- This section is now OUTSIDE the CV loop (or runs if no CV) ---

        if not is_ddp or dist.get_rank() == 0:
            print("\nFinal Test Results:")
            if current_args.task_type == 'classification':
                print(f"Test Loss: {test_metrics['loss']:.4f}")
                print(f"Test Accuracy: {test_metrics['acc']:.4f}")
                print(f"Test Precision: {test_metrics['precision']:.4f}")
                print(f"Test Recall: {test_metrics['recall']:.4f}")
                print(f"Test F1: {test_metrics['f1']:.4f}")
            elif current_args.task_type == 'multitask':
                print(f"Test Loss: {test_metrics['loss']:.4f}")
                print(f"Test MAE (avg): {test_metrics.get('mae', 0):.4f}")
                print(f"Test RMSE (avg): {test_metrics.get('rmse', 0):.4f}")
                print(f"Test R2 (avg): {test_metrics.get('r2', 0):.4f}")
            else:
                print(f"Test Loss: {test_metrics['loss']:.4f}")
                print(f"Test MAE: {test_metrics['mae']:.4f}")
                print(f"Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"Test R2: {test_metrics['r2']:.4f}")

            if args.enable_wandb:
              # Log to the overall sweep Wandb run
                wandb_log_dict = {
                    "final_test_loss": test_metrics['loss'],
                    "final_test_mae": test_metrics.get('mae'),
                    "final_test_rmse": test_metrics.get('rmse'),
                    "final_test_r2": test_metrics.get('r2'),
                    "final_test_acc": test_metrics.get('acc'),
                    "final_test_precision": test_metrics.get('precision'),
                    "final_test_recall": test_metrics.get('recall'),
                    "final_test_f1": test_metrics.get('f1'),
                    "best_val_loss": best_val_loss, #log best val loss
                    "trial_num": trial_num #log trial number
                }

                sweep_wandb_run.log(wandb_log_dict)
                print("Test metrics and CV metrics (if applicable) logged to overall Wandb sweep.")
            
        # Create a dictionary with the current trial's metrics and hyperparameters
        current_trial_results = {
            'trial_num': trial_num,
            'test_metrics': test_metrics,
            'hyperparams': {k: getattr(current_args, k) for k in current_hparams.keys()}
        }
        
        # Update best trial tracking (use the leader node in DDP)
        if (not is_ddp or dist.get_rank() == 0):
            # Initialize best_trial_metrics on first trial
            if best_trial_metrics is None:
                best_trial_metrics = test_metrics['loss']
                best_trial_hyperparams = current_trial_results['hyperparams']
                best_trial_num = trial_num
                best_test_metrics = test_metrics
            # Update if current trial is better
            elif test_metrics['loss'] < best_trial_metrics:
                best_trial_metrics = test_metrics['loss']
                best_trial_hyperparams = current_trial_results['hyperparams']
                best_trial_num = trial_num
                best_test_metrics = test_metrics


        # Partial charge extraction (only rank 0 writes out)
        if current_args.output_partial_charges and current_args.use_partial_charges and (not is_ddp or dist.get_rank() == 0):
            print(f"Extracting partial charges on the test set (writing to {current_args.output_partial_charges} ...")
            if not current_args.iterable_dataset:
                single_proc_test_loader = create_pyg_dataloader(
                    test_dataset,
                    batch_size=current_args.batch_size,
                    shuffle=False,
                    num_workers=0
                )
                extraction_model = GNN(
                    feature_sizes=feature_sizes,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_shells=current_args.num_shells,
                    num_message_passing_layers=current_args.num_message_passing_layers,
                    ffn_hidden_dim=current_args.ffn_hidden_dim,
                    ffn_num_layers=current_args.ffn_num_layers,
                    pooling_type=current_args.pooling_type,
                    task_type=current_args.task_type,
                    embedding_dim=current_args.embedding_dim,
                    use_partial_charges=current_args.use_partial_charges,
                    use_stereochemistry=current_args.use_stereochemistry,
                    ffn_dropout=current_args.ffn_dropout,
                    activation_type=current_args.activation_type,
                    shell_conv_num_mlp_layers=current_args.shell_conv_num_mlp_layers,
                    shell_conv_dropout=current_args.shell_conv_dropout,
                    attention_num_heads=current_args.attention_num_heads,
                    attention_temperature=current_args.attention_temperature
                ).to(device)
                extraction_model.load_state_dict(best_model_state) #load best model
                extraction_model.eval()

                pc_results = extract_partial_charges(extraction_model, single_proc_test_loader, device)
                df_pc = pd.DataFrame(pc_results, columns=["smiles", "partial_charges"])
                df_pc.to_csv(current_args.output_partial_charges, index=False)
                print(f"Partial charges CSV saved to {current_args.output_partial_charges}")
            else:
                print("WARNING: partial charge extraction not trivially implemented for large IterableDataset. "
                      "You can replicate the same logic with a single-process pass over test.h5 if desired.")


        # Final barrier if DDP
        if is_ddp:
            dist.barrier()



    # --- Save the best model (outside the loop, only rank 0) ---
    if (not is_ddp or dist.get_rank() == 0) and best_model_state is not None:
        model_artifact = {
            "hyperparams": {
                "task_type": best_hyperparams.task_type,
                "num_shells": best_hyperparams.num_shells,
                "hidden_dim": best_hyperparams.hidden_dim,
                "num_message_passing_layers": best_hyperparams.num_message_passing_layers,
                "ffn_hidden_dim": best_hyperparams.ffn_hidden_dim,
                "ffn_num_layers": best_hyperparams.ffn_num_layers,
                "pooling_type": best_hyperparams.pooling_type,
                "embedding_dim": best_hyperparams.embedding_dim,
                "scaler_means": best_std_scaler.means.tolist() if best_std_scaler else None, #save best std scaler
                "scaler_stds": best_std_scaler.stds.tolist() if best_std_scaler else None, #save best std scaler
                "use_partial_charges": best_hyperparams.use_partial_charges,
                "use_stereochemistry": best_hyperparams.use_stereochemistry,
                "ffn_dropout": best_hyperparams.ffn_dropout,
                "learning_rate": best_hyperparams.learning_rate,
                # best validation loss
                "best_val_loss": best_val_loss,
                "activation_type": current_args.activation_type,
            },
            "state_dict": best_model_state
        }

        torch.save(model_artifact, args.model_save_path)  # Save to the path specified in args
        print(f"Saved best model (with hyperparams) to {args.model_save_path}")
        print(f"Best hyperparameters: {best_hyperparams}")

        if args.enable_wandb:
          # Log the best hyperparameters to the OVERALL sweep Wandb run
          sweep_wandb_run.config.update(vars(best_hyperparams))
          sweep_wandb_run.config.update({"best_val_loss": best_val_loss})

          # Save best model artifact to overall sweep
          wandb.save(args.model_save_path)
          print("Best hyperparameters and best model saved to overall Wandb sweep.")
          # Finish the overall sweep run
          sweep_wandb_run.finish()


    # ADD FINAL SUMMARY OUTPUT HERE
    if (not is_ddp or dist.get_rank() == 0) and args.num_trials > 1:
        print("\n" + "="*40)
        print("BEST HYPERPARAMETER SWEEP RESULTS")
        print("="*40)
        print(f"Best Trial: {best_trial_num + 1}/{args.num_trials}")
        print("\nBest Hyperparameters:")
        for param_name, param_value in best_trial_hyperparams.items():
            print(f"  {param_name}: {param_value}")
        
        print("\nBest Test Metrics:")
        if args.task_type == 'multitask':
            print(f"  Test Loss: {best_test_metrics['loss']:.5f}")
            print(f"  Test MAE (avg): {best_test_metrics.get('mae', 0):.5f}")
            print(f"  Test RMSE (avg): {best_test_metrics.get('rmse', 0):.5f}")
            print(f"  Test R2 (avg): {best_test_metrics.get('r2', 0):.5f}")
            
            # Add detailed per-task metrics display for best trial
            if 'mae_per_target' in best_test_metrics:
                print("\n  Detailed Test Metrics Per Task:")
                for i, (mae_i, rmse_i, r2_i) in enumerate(zip(
                    best_test_metrics['mae_per_target'],
                    best_test_metrics['rmse_per_target'],
                    best_test_metrics['r2_per_target']
                )):
                    task_name = f"Task {i}"
                    if 'multi_cols' in locals() and multi_cols and i < len(multi_cols):
                        task_name = multi_cols[i]
                    print(f"    [{task_name}] MAE={mae_i:.5f}, RMSE={rmse_i:.5f}, R2={r2_i:.5f}")
        else:
            print(f"  Test Loss: {best_test_metrics['loss']:.5f}")
            print(f"  Test MAE: {best_test_metrics['mae']:.5f}")
            print(f"  Test RMSE: {best_test_metrics['rmse']:.5f}")
            print(f"  Test R2: {best_test_metrics['r2']:.5f}")
        print("="*40)
        
        # If using wandb, log final best results
        if args.enable_wandb:
            wandb_best_dict = {
                "best_trial": best_trial_num + 1,
                "best_test_loss": best_test_metrics['loss'],
                "best_test_mae": best_test_metrics.get('mae'),
                "best_test_rmse": best_test_metrics.get('rmse'),
                "best_test_r2": best_test_metrics.get('r2')
            }
            # Add best hyperparameters to wandb
            for param_name, param_value in best_trial_hyperparams.items():
                wandb_best_dict[f"best_{param_name}"] = param_value
                
            sweep_wandb_run.log(wandb_best_dict)
            print("Best hyperparameter results logged to Wandb.")



    # Molecular embedding extraction
    if args.save_embeddings and (not is_ddp or dist.get_rank() == 0):
        print(f"Extracting and saving full dataset embeddings to {args.embeddings_output_path}")
        
        # Ensure directory exists
        embedding_dir = os.path.dirname(os.path.abspath(args.embeddings_output_path))
        if embedding_dir and not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)
        
        # Use the best model for extraction (or trained_model if best_model isn't available)
        embedding_model = None
        
        if best_model_state is not None:
            # Create a fresh model with the best hyperparameters
            feature_sizes = {
                'atom_type': len(ATOM_TYPES) + 1,
                'hydrogen_count': 9,  # 0-8 hydrogens (capped at 8)
                'degree': len(DEGREES) + 1,
                'hybridization': len(HYBRIDIZATIONS) + 1,
            }
            
            embedding_model = GNN(
                feature_sizes=feature_sizes,
                hidden_dim=best_hyperparams.hidden_dim,
                output_dim=output_dim,
                num_shells=best_hyperparams.num_shells,
                num_message_passing_layers=best_hyperparams.num_message_passing_layers,
                ffn_hidden_dim=best_hyperparams.ffn_hidden_dim,
                ffn_num_layers=best_hyperparams.ffn_num_layers,
                pooling_type=best_hyperparams.pooling_type,
                task_type=best_hyperparams.task_type,
                embedding_dim=best_hyperparams.embedding_dim,
                use_partial_charges=best_hyperparams.use_partial_charges,
                use_stereochemistry=best_hyperparams.use_stereochemistry,
                ffn_dropout=best_hyperparams.ffn_dropout,
                activation_type=best_hyperparams.activation_type,
                shell_conv_num_mlp_layers=current_args.shell_conv_num_mlp_layers,
                shell_conv_dropout=current_args.shell_conv_dropout,
                attention_num_heads=current_args.attention_num_heads,
                attention_temperature=current_args.attention_temperature

            ).to(device)
            
            embedding_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        else:
            # Use the last trained model if best_model is not available
            embedding_model = trained_model
            if isinstance(embedding_model, torch.nn.parallel.DistributedDataParallel):
                embedding_model = embedding_model.module
        
        # Put model in eval mode
        embedding_model.eval()
        
        # Extract embeddings from all datasets
        extract_embeddings_main(
            args, 
            embedding_model, 
            train_loader, 
            val_loader, 
            test_loader, 
            device
        )
        
        print(f"Successfully saved full dataset embeddings to {args.embeddings_output_path}")

    # Destroy if DDP
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

    print("Program finished successfully.")


###############################################################################
# Command-line interface
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Molecular Property Predictor")

    # Data paths
    parser.add_argument("--data_path", type=str, help="Path to single CSV file")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)

    parser.add_argument("--train_data", type=str, help="CSV file for train set")
    parser.add_argument("--val_data", type=str, help="CSV file for val set")
    parser.add_argument("--test_data", type=str, help="CSV file for test set")

    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--target_column", type=str, default="target")
    parser.add_argument("--multi_target_columns", type=str, default=None)

    # Model hyperparams
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_shells", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_save_path", type=str, default="gnn_model.pth")
    parser.add_argument("--early_stopping", action="store_true")

    parser.add_argument("--ffn_hidden_dim", type=int, default=None)
    parser.add_argument("--ffn_num_layers", type=int, default=3)
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="attention",
        choices=["attention", "mean", "max", "sum"],
        help="Type of pooling to use: 'attention', 'mean', 'max', or 'sum'. Default is 'attention'."
    )

    parser.add_argument("--num_message_passing_layers", type=int, default=3)

    # Task
    parser.add_argument("--task_type", type=str, default="regression",
                        choices=["regression", "multitask"])

    # SAE
    parser.add_argument("--calculate_sae", action="store_true")
    parser.add_argument("--sae_subtasks", type=str, default=None)

    # Iterable dataset
    parser.add_argument("--iterable_dataset", action="store_true")
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--multitask_weights", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=64)

    # Add these to the parser arguments
    parser.add_argument("--train_hdf5", type=str, default=None,
                    help="Path to train HDF5 file (if using separate train/val files)")
    parser.add_argument("--val_hdf5", type=str, default=None,
                    help="Path to validation HDF5 file (if using separate train/val files)")
    parser.add_argument("--test_hdf5", type=str, default="test.h5")


    parser.add_argument("--hyperparameter_file", type=str, default=None,
                        help="Text file containing lines of hyperparameters to run multiple experiments.")

    parser.add_argument("--use_partial_charges", action="store_true",
                        help="Activate partial charge calculations.")
    parser.add_argument("--use_stereochemistry", action="store_true",
                        help="Activate stereochemical feature calculations.")

    parser.add_argument("--inference_csv", type=str, default=None,
                        help="Path to input CSV for inference. If provided, training is skipped.")
    parser.add_argument("--inference_output", type=str, default="predictions.csv",
                        help="Where to save inference predictions.")

    parser.add_argument("--inference_mode", type=str, choices=["streaming", "inmemory", "iterable"], default=None,
                   help="Mode for inference: streaming (compute features on-the-fly), inmemory (precompute all in memory), iterable (use HDF5)")

    parser.add_argument("--output_partial_charges", type=str, default=None,
                        help="If specified (and --use_partial_charges is on), the script will output a CSV "
                             "with two columns: smiles and partial_charges (list).")

    parser.add_argument("--num_gpu_devices", type=int, default=1,
                    help="Number of GPU devices for DDP (use 1 for single-GPU or 0 for CPU).")

    parser.add_argument("--inference_hdf5", type=str, default=None,
                    help="Path to an HDF5 file containing precomputed BFS/features for inference.")

    parser.add_argument("--precompute_num_workers", type=int, default=None,
                    help="Number of workers for precomputation (BFS + features) in iterable mode. If None, defaults to --num_workers.")

    parser.add_argument(
        "--ffn_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for the feed-forward layers of the model."
    )

    parser.add_argument("--enable_wandb", action="store_true", help="Enable Wandb logging.")

    parser.add_argument("--num_trials", type=int, default=1,  # Default to 10 trials
                        help="Number of trials for hyperparameter tuning (for random search).")

    parser.add_argument("--loss_function", type=str, default="l1", choices=["l1", "mse"],
                        help="Choose between L1 loss and MSE loss.")

    # Hyperparameter Options
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience.")
    parser.add_argument("--lr_scheduler", type=str, default="ReduceLROnPlateau",
                        choices=[None, "ReduceLROnPlateau", "CosineAnnealingLR", "StepLR", "ExponentialLR"],
                        help="Learning rate scheduler to use.")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5,
                        help="Factor for ReduceLROnPlateau scheduler.")
    parser.add_argument("--lr_patience", type=int, default=10,
                        help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument("--lr_cosine_t_max", type=int, default=10,
                        help="T_max for CosineAnnealingLR scheduler.")
    parser.add_argument("--lr_step_size", type=int, default=10,
                        help="Step size for StepLR scheduler.")
    parser.add_argument("--lr_step_gamma", type=float, default=0.1,
                        help="Gamma for StepLR scheduler.")
    parser.add_argument("--lr_exp_gamma", type=float, default=0.95,
                        help="Gamma for ExponentialLR scheduler.")

    parser.add_argument("--activation_type", type=str, default="silu",
                    choices=["relu", "leakyrelu", "elu", "silu", "gelu"],
                    help="Type of activation function to use.")

    parser.add_argument("--wandb_project", type=str, default="gnn-molecular-property-prediction", 
                    help="Name of the Wandb project if enable_wandb is called.")


    # Transfer learning paths and controls
    parser.add_argument("--transfer_learning", type=str, default=None,
                       help="Path to pretrained model for transfer learning")
    parser.add_argument("--freeze_pretrained", action="store_true",
                       help="Freeze pretrained layers except output layer")
    parser.add_argument("--freeze_layers", type=str, default=None,
                       help="Comma-separated list of layer patterns to freeze")
    parser.add_argument("--unfreeze_layers", type=str, default=None,
                       help="Comma-separated list of layer patterns to explicitly unfreeze")
    parser.add_argument("--reset_output_layer", action="store_true",
                       help="Reset output layer when loading pretrained model")

    parser.add_argument("--layer_wise_lr_decay", action="store_true",
                   help="Enable layer-wise learning rate decay for transfer learning")
    parser.add_argument("--lr_decay_factor", type=float, default=0.8,
                    help="Decay factor for layer-wise learning rate (lower means more aggressive decay)")
    
    #MC Dropout
    parser.add_argument("--mc_samples", type=int, default=0,
                    help="Number of Monte Carlo dropout samples for uncertainty estimation (0 = disabled)")
                    
    parser.add_argument("--stream_chunk_size", type=int, default=1000,
                    help="Number of SMILES to process in each chunk during streamed inference")
    parser.add_argument("--stream_batch_size", type=int, default=None,
                    help="Batch size for model inference during streaming (if different from --batch_size)")

    parser.add_argument("--save_embeddings", action="store_true",
                    help="Extract and save molecular embeddings")
    parser.add_argument("--embeddings_output_path", type=str, default="molecular_embeddings.h5",
                        help="Path to save extracted embeddings")
    parser.add_argument("--include_atom_embeddings", action="store_true",
                help="Extract and save atom-level embeddings in addition to molecule embeddings")
    
    parser.add_argument("--attention_num_heads", type=int, default=4,
                   help="Number of attention heads in MultiHeadAttentionPoolingLayer")
    parser.add_argument("--attention_temperature", type=float, default=1.0,
                    help="Initial temperature for attention in MultiHeadAttentionPoolingLayer")

    parser.add_argument("--shell_conv_num_mlp_layers", type=int, default=2,
                   help="Number of MLP layers in ShellConvolutionLayer")
    parser.add_argument("--shell_conv_dropout", type=float, default=0.05,
                    help="Dropout rate for ShellConvolutionLayer MLP")
    

    args = parser.parse_args()
    main(args)


