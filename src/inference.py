# inference.py

# Standard libraries
import os
import time
import pickle

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool, cpu_count
import torch.distributed as dist

# Data processing and progress bar
import pandas as pd
import numpy as np
import tqdm

# Local imports
from datasets import _worker_process_smiles, MyBatch
from torch_geometric.data import Data
from model import (
    MultiTargetStandardScaler,
    GNN
)

from rdkit.Chem.rdchem import HybridizationType

# Atom feature constants
ATOM_TYPES = list(range(1, 119))  # Atomic numbers up to 118
DEGREES = list(range(6))          # Degrees from 0 to 5
HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2
]

def streamed_inference(args, model, device, feature_sizes, std_scaler=None, is_ddp=False, local_rank=0, world_size=1):
    """
    Performs streaming inference where SMILES are read from CSV in chunks,
    features are computed on-the-fly using parallel processing,
    predictions are made, and results are immediately written to disk.
    """
    model.eval()
    task_type = args.task_type
    max_hops = args.num_shells
    batch_size = args.stream_batch_size or args.batch_size
    
    # Determine number of workers for parallel feature computation
    num_workers = args.num_workers if args.num_workers > 0 else max(1, cpu_count() // 2)
    
    # Separate files for each rank in DDP mode
    output_file = args.inference_output
    if is_ddp and world_size > 1:
        base, ext = os.path.splitext(output_file)
        tmp_output_file = f"{base}_rank{local_rank}{ext}"
    else:
        tmp_output_file = output_file
    
    # Create output directory if it doesn't exist
    if not is_ddp or local_rank == 0:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Synchronize after directory creation
    if is_ddp and dist.is_initialized():
        dist.barrier()
    
    # Read CSV in chunks
    chunk_size = args.stream_chunk_size
    
    # For MC dropout
    mc_samples = args.mc_samples
    
    # Display start message
    if not is_ddp or local_rank == 0:
        print(f"Starting streamed inference with chunk size {chunk_size}, batch size {batch_size}, "
              f"and {num_workers} workers for feature computation")
    
    # Track valid/invalid SMILES counts
    valid_count = 0
    invalid_count = 0
    total_processed = 0
    
    try:
        # First, count total lines in the CSV to divide work evenly in DDP mode
        total_lines = 0
        if not is_ddp or local_rank == 0:
            try:
                with open(args.inference_csv, 'r') as f:
                    # Subtract 1 for header
                    total_lines = sum(1 for _ in f) - 1
                print(f"Total molecules to process: {total_lines}")
            except Exception as e:
                print(f"Warning: Could not count lines in CSV: {e}")
                print("Will proceed without line count.")
        
        # Broadcast total_lines from rank 0 to all processes
        if is_ddp and dist.is_initialized():
            total_lines_tensor = torch.tensor([total_lines], dtype=torch.long, device=device)
            dist.broadcast(total_lines_tensor, src=0)
            total_lines = total_lines_tensor.item()
        
        # In DDP mode, we need to divide the work by line number
        if is_ddp and world_size > 1:
            # Calculate lines per rank
            lines_per_rank = total_lines // world_size
            # Calculate start and end lines for this rank
            start_line = local_rank * lines_per_rank + 1  # +1 for header
            if local_rank == world_size - 1:
                # Last rank gets any remaining lines
                end_line = total_lines + 1  # +1 for header
            else:
                end_line = (local_rank + 1) * lines_per_rank + 1  # +1 for header
            
            # Calculate how many molecules this rank will process
            rank_molecules_count = end_line - start_line
            print(f"[Rank {local_rank}] Processing lines {start_line} to {end_line-1} (out of {total_lines}): {rank_molecules_count} molecules")
        else:
            # Single process mode
            start_line = 1  # Start after header
            end_line = None  # Process until end
            rank_molecules_count = total_lines
        
        # Determine headers based on task type and MC samples
        header = [args.smiles_column]
        if task_type == 'multitask':
            # For multitask, we need prediction_0, prediction_1, etc.
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                output_dim = model.module.output_layer.weight.shape[0]
            else:
                output_dim = model.output_layer.weight.shape[0]
            
            for i in range(output_dim):
                header.append(f"prediction_{i}")
                # Only add uncertainty columns if MC dropout is enabled
                if mc_samples > 0:
                    header.append(f"uncertainty_{i}")
        else:
            # For single-task, just prediction
            header.append("prediction")
            # Only add uncertainty column if MC dropout is requested
            if mc_samples > 0:
                header.append("uncertainty")

        # Now all ranks have the header variable defined
        # Each rank writes its own header
        with open(tmp_output_file, 'w') as f:
            f.write(','.join(header) + '\n')

        if is_ddp and dist.is_initialized():
            dist.barrier()  # Ensure all ranks have created their output files
        
        # Create a single progress bar for this rank's entire workload
        pbar = None
        if not is_ddp or local_rank == 0:  # Only show progress bar on rank 0 if DDP
            pbar = tqdm.tqdm(
                total=rank_molecules_count,
                desc=f"[Rank {local_rank}] Processing molecules",
                unit="mol"
            )
        
        # Process the CSV in sequential chunks by line number
        with Pool(processes=num_workers) as pool:
            # Use pandas to skip to the correct starting line and read until end_line
            # Note: pandas skiprows is 0-indexed, while our line numbers are 1-indexed
            skiprows = list(range(1, start_line))  # Skip header (row 0) and lines until start_line
            nrows = None if end_line is None else (end_line - start_line)
            
            # Read the CSV chunk by chunk, but only the lines assigned to this rank
            for chunk_idx, chunk_df in enumerate(pd.read_csv(args.inference_csv, skiprows=skiprows, nrows=nrows, chunksize=chunk_size)):
                # Extract SMILES from the chunk
                smiles_list = chunk_df[args.smiles_column].tolist()
                
                if not smiles_list:
                    continue
                
                # Prepare data for worker function
                process_inputs = [(idx, smi, max_hops) for idx, smi in enumerate(smiles_list)]
                
                # Process SMILES in the current chunk using parallel map - no progress bar here
                start_time = time.time()
                precomp_results = list(pool.imap(
                    _worker_process_smiles, 
                    process_inputs, 
                    chunksize=max(1, len(process_inputs) // (num_workers * 10))
                ))
                feature_compute_time = time.time() - start_time
                
                # Process results - collect valid molecules
                valid_smiles = []
                valid_indices = []
                precomp_data = []
                
                for idx, precomp in precomp_results:
                    if precomp is not None:
                        valid_smiles.append(smiles_list[idx])
                        valid_indices.append(idx)
                        precomp_data.append(precomp)
                        valid_count += 1
                    else:
                        invalid_count += 1
                
                if not valid_smiles:
                    print(f"[Rank {local_rank}] No valid SMILES in chunk {chunk_idx+1}, skipping")
                    continue
                
                # Create batch data objects for all valid molecules
                start_time = time.time()
                batch_data_objects = []
                
                for idx, (smi, precomp) in enumerate(zip(valid_smiles, precomp_data)):
                    try:
                        num_atoms = precomp['atom_features']['atom_type'].shape[0]
                        x_dummy = torch.ones((num_atoms, 1), dtype=torch.float)
                        
                        data_obj = Data()
                        data_obj.x = x_dummy
                        data_obj.smiles = smi
                        
                        # For inference, we set a dummy target
                        data_obj.target = torch.tensor([0.0], dtype=torch.float)
                        
                        # Process and convert precomputed data to PyTorch tensors
                        data_obj.multi_hop_edges = [torch.from_numpy(e).long() for e in precomp["multi_hop_edges"]]
                        
                        atom_feats_map = {}
                        for k, arr in precomp["atom_features"].items():
                            atom_feats_map[k] = torch.from_numpy(arr).long()
                        data_obj.atom_features_map = atom_feats_map
                        
                        data_obj.chiral_tensors = [torch.from_numpy(x).long() for x in precomp["chiral_tensors"]]
                        data_obj.cis_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["cis_bonds_tensors"]]
                        data_obj.trans_bonds_tensors = [torch.from_numpy(x).long() for x in precomp["trans_bonds_tensors"]]
                        
                        data_obj.total_charge = torch.tensor([precomp["total_charge"]], dtype=torch.float)
                        data_obj.atomic_numbers = torch.from_numpy(precomp["atomic_numbers"]).long()
                        
                        batch_data_objects.append(data_obj)
                    except Exception as e:
                        # Log error but continue processing other molecules
                        print(f"Error processing SMILES {idx}: {smi[:30]}... - {str(e)}")
                
                data_prep_time = time.time() - start_time
                
                # Create dataloader
                temp_loader = torch.utils.data.DataLoader(
                    batch_data_objects,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=MyBatch.from_data_list,
                    num_workers=0  # Using 0 to avoid serialization issues with complex objects
                )
                
                # Make predictions
                start_time = time.time()
                
                if mc_samples > 0:
                    # MC Dropout for uncertainty estimates
                    def enable_dropout(m):
                        if type(m) == nn.Dropout:
                            m.train()
                    
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        model.module.apply(enable_dropout)
                    else:
                        model.apply(enable_dropout)
                    
                    # Run multiple forward passes
                    sample_predictions = []
                    
                    for _ in range(mc_samples):
                        batch_predictions = []
                        
                        for batch in temp_loader:
                            if batch is None:
                                continue
                                
                            batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
                            batch_indices = batch.batch_indices.to(device)
                            batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
                            total_charges = batch.total_charges.to(device)
                            tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
                            cis_indices = batch.final_cis_tensor.to(device)
                            trans_indices = batch.final_trans_tensor.to(device)
                            
                            with torch.no_grad():
                                outputs, _, _ = model(
                                    batch_atom_features,
                                    batch_multi_hop_edges,
                                    batch_indices,
                                    total_charges,
                                    tetrahedral_indices,
                                    cis_indices,
                                    trans_indices
                                )
                                batch_predictions.append(outputs.detach().cpu().numpy())
                        
                        if batch_predictions:
                            sample_preds = np.concatenate(batch_predictions)
                            sample_predictions.append(sample_preds)
                    
                    if sample_predictions:
                        # Stack all samples: shape (n_samples, n_molecules, n_outputs)
                        stacked_predictions = np.stack(sample_predictions, axis=0)
                        # Mean across samples
                        mean_preds = np.mean(stacked_predictions, axis=0)
                        # Standard deviation across samples (uncertainty)
                        uncertainties = np.std(stacked_predictions, axis=0)
                        
                        # Apply inverse scaling if needed
                        if std_scaler is not None and task_type in ["regression", "multitask"]:
                            mean_preds = std_scaler.inverse_transform(mean_preds)
                        
                        all_preds = mean_preds
                        all_uncertainties = uncertainties
                    else:
                        all_preds = np.array([])
                        all_uncertainties = np.array([])
                else:
                    # Standard deterministic inference
                    batch_predictions = []
                    
                    for batch in temp_loader:
                        if batch is None:
                            continue
                            
                        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
                        batch_indices = batch.batch_indices.to(device)
                        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
                        total_charges = batch.total_charges.to(device)
                        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
                        cis_indices = batch.final_cis_tensor.to(device)
                        trans_indices = batch.final_trans_tensor.to(device)
                        
                        with torch.no_grad():
                            outputs, _, _ = model(
                                batch_atom_features,
                                batch_multi_hop_edges,
                                batch_indices,
                                total_charges,
                                tetrahedral_indices,
                                cis_indices,
                                trans_indices
                            )
                            batch_predictions.append(outputs.detach().cpu().numpy())
                    
                    if batch_predictions:
                        all_preds = np.concatenate(batch_predictions)
                        
                        # Apply inverse scaling if needed
                        if std_scaler is not None and task_type in ["regression", "multitask"]:
                            all_preds = std_scaler.inverse_transform(all_preds)
                        
                        # Create dummy uncertainties (all zeros) if MC dropout wasn't used
                        all_uncertainties = np.zeros_like(all_preds)
                    else:
                        all_preds = np.array([])
                        all_uncertainties = np.array([])
                
                inference_time = time.time() - start_time
                
                # Write results to file (append mode)
                if all_preds.size > 0:
                    # Determine output format based on task type
                    with open(tmp_output_file, 'a') as f:
                        for i, (orig_idx, pred_idx) in enumerate(zip(valid_indices, range(len(all_preds)))):
                            # Get the original SMILES
                            current_smiles = chunk_df.iloc[int(orig_idx)][args.smiles_column]
                            
                            # Prepare output line
                            line = [current_smiles]

                            # Add predictions (and uncertainties if enabled)
                            ncols = all_preds.shape[1] if len(all_preds.shape) > 1 else 1
                            if ncols == 1:
                                # Single task
                                line.append(str(all_preds[pred_idx].item() if len(all_preds.shape) > 1 else all_preds[pred_idx]))
                                # Only add uncertainty if MC dropout was used
                                if mc_samples > 0:
                                    line.append(str(all_uncertainties[pred_idx].item() if len(all_uncertainties.shape) > 1 else all_uncertainties[pred_idx]))
                            else:
                                # Multi-task
                                for j in range(ncols):
                                    line.append(str(all_preds[pred_idx, j]))
                                    # Only add uncertainty if MC dropout was used
                                    if mc_samples > 0:
                                        line.append(str(all_uncertainties[pred_idx, j]))
                            
                            # Write line to file
                            f.write(','.join(line) + '\n')
                
                # Update total count and update progress bar
                molecules_in_chunk = len(valid_indices)
                total_processed += molecules_in_chunk
                
                # Update the progress bar
                if pbar is not None:
                    pbar.update(molecules_in_chunk)
                
                # Print a simple summary for each chunk without progress bar
                print(f"[Rank {local_rank}] Processed chunk {chunk_idx+1}: {molecules_in_chunk} molecules, "
                      f"Total: {total_processed}/{rank_molecules_count} ({total_processed/rank_molecules_count*100:.1f}%), "
                      f"Time: feature={feature_compute_time:.2f}s, data={data_prep_time:.2f}s, inference={inference_time:.2f}s")
        
        # Close the progress bar if it exists
        if pbar is not None:
            pbar.close()
            
        # In DDP mode, we need to gather all tmp files and combine them
        if is_ddp and dist.is_initialized():
            dist.barrier()  # Wait for all ranks to finish processing
            
            # Only rank 0 combines the files
            if local_rank == 0:
                # Combine all temporary files into the final output file
                with open(output_file, 'w') as outfile:
                    # Write header first
                    outfile.write(','.join(header) + '\n')
                    
                    # Write all data from all ranks
                    for rank in range(world_size):
                        rank_file = f"{os.path.splitext(output_file)[0]}_rank{rank}{os.path.splitext(output_file)[1]}"
                        if os.path.exists(rank_file):
                            with open(rank_file, 'r') as infile:
                                # Skip header
                                next(infile, None)
                                # Copy content
                                for line in infile:
                                    outfile.write(line)
                
                # Remove temporary files after combining
                for rank in range(world_size):
                    rank_file = f"{os.path.splitext(output_file)[0]}_rank{rank}{os.path.splitext(output_file)[1]}"
                    if os.path.exists(rank_file):
                        os.remove(rank_file)
                        
                print(f"Combined all rank outputs into: {output_file}")
                
                # Verify the number of lines in the output file
                try:
                    with open(output_file, 'r') as f:
                        output_lines = sum(1 for _ in f) - 1  # Subtract header
                    
                    if output_lines != total_lines:
                        print(f"WARNING: Output file has {output_lines} lines, but expected {total_lines}.")
                        print("Some molecules may not have been processed correctly.")
                    else:
                        print(f"SUCCESS: Output file has the expected {output_lines} lines.")
                except:
                    pass
    
    except Exception as e:
        print(f"Error during streamed inference: {str(e)}")
        import traceback
        traceback.print_exc()

# 2. Modify the inference_main function to handle the new parameter
# Update the inference logic in the inference_main function (around line 4085):

def inference_main(args, device, is_ddp, local_rank, world_size):
    """
    Runs inference either on a precomputed HDF5 dataset, from raw SMILES CSV,
    or using streamed processing for large datasets.
    """
    start_time = time.time()

    # 1) Load model artifact & hyperparams
    print("Loading model from", args.model_save_path)
    model_artifact = torch.load(args.model_save_path, map_location=device)
    hyperparams = model_artifact["hyperparams"]
    state_dict = model_artifact["state_dict"]
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    # Possibly re-create std_scaler
    std_scaler = None
    if hyperparams.get("scaler_means") is not None:
        std_scaler = MultiTargetStandardScaler()
        std_scaler.means = np.array(hyperparams["scaler_means"])
        std_scaler.stds = np.array(hyperparams["scaler_stds"])
        std_scaler.fitted = True
        print("Standard scaler reconstructed from model hyperparameters")

    task_type = hyperparams["task_type"]
    print(f"Task type: {task_type}")
    
    # Build model
    feature_sizes = {
        'atom_type': len(ATOM_TYPES) + 1,
        'hydrogen_count': 9,  # 0-8 hydrogens (capped at 8)
        'degree': len(DEGREES) + 1,
        'hybridization': len(HYBRIDIZATIONS) + 1,
    }
    
    print("Building model...")

    # Get all required parameters from hyperparams or command line
    # Use the values from the model when available, otherwise from args
    shell_conv_num_mlp_layers = args.shell_conv_num_mlp_layers
    shell_conv_dropout = args.shell_conv_dropout
    attention_num_heads = args.attention_num_heads
    attention_temperature = args.attention_temperature
    
    # Check for the existence of the key first in the state dictionary
    has_mlp_layer_2 = any(k.startswith("message_passing_layers.0.mlp_blocks.2") for k in state_dict.keys())
    if has_mlp_layer_2:
        print("Detected 3 MLP layers in saved model, overriding shell_conv_num_mlp_layers to 3")
        shell_conv_num_mlp_layers = 3
    
    print(f"Using shell_conv_num_mlp_layers={shell_conv_num_mlp_layers}")

    # Determine output dimension from the state dictionary keys
    output_dim = None
    # Try different possible key patterns for output layer
    output_keys = [
        "output_layer.weight", 
        "module.output_layer.weight",
        "classifier.weight",
        "module.classifier.weight",
        "fc.weight",
        "module.fc.weight",
        "head.weight",
        "module.head.weight"
    ]
    
    for key in output_keys:
        if key in state_dict:
            output_dim = state_dict[key].shape[0]
            print(f"Found output dimension {output_dim} from key: {key}")
            break
    
    # If none of the common keys were found, try to infer from hyperparams
    if output_dim is None:
        if 'output_dim' in hyperparams:
            output_dim = hyperparams['output_dim']
            print(f"Using output dimension {output_dim} from hyperparameters")
        else:
            # As a fallback, scan all weight matrices to find a likely output layer
            output_candidates = []
            for key, value in state_dict.items():
                if key.endswith('.weight') and len(value.shape) == 2:
                    output_candidates.append((key, value.shape[0]))
            
            # Sort by dimension size (assuming smaller is more likely to be output)
            output_candidates.sort(key=lambda x: x[1])
            
            if output_candidates:
                # Choose the smallest output dimension as a heuristic
                output_dim = output_candidates[0][1]
                print(f"Inferred output dimension {output_dim} from key: {output_candidates[0][0]}")
            else:
                # Last resort - default to 1
                output_dim = 1
                print("WARNING: Could not determine output dimension, defaulting to 1")

    model = GNN(
        feature_sizes=feature_sizes,
        hidden_dim=hyperparams["hidden_dim"],
        output_dim=output_dim,
        num_shells=hyperparams["num_shells"],
        num_message_passing_layers=hyperparams["num_message_passing_layers"],
        ffn_hidden_dim=hyperparams["ffn_hidden_dim"],
        ffn_num_layers=hyperparams["ffn_num_layers"],
        pooling_type=hyperparams["pooling_type"],
        task_type=task_type,
        embedding_dim=hyperparams["embedding_dim"],
        use_partial_charges=hyperparams.get("use_partial_charges", False),
        use_stereochemistry=hyperparams.get("use_stereochemistry", False),
        ffn_dropout=hyperparams["ffn_dropout"],
        activation_type=hyperparams["activation_type"],
        shell_conv_num_mlp_layers=shell_conv_num_mlp_layers,
        shell_conv_dropout=shell_conv_dropout,
        attention_num_heads=attention_num_heads,
        attention_temperature=attention_temperature
    ).to(device)
    
    # Safely load state dict with flexible key matching
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        # Handle module prefix difference (DDP adds 'module.' prefix)
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' prefix
        
        # Try to find matching parameter in model
        if name in model_state_dict:
            # Direct match
            if param.size() == model_state_dict[name].size():
                model_state_dict[name].copy_(param)
            else:
                print(f"Size mismatch for {name}: model={model_state_dict[name].size()}, checkpoint={param.size()}")
        else:
            # Try to match by parameter name pattern
            found_match = False
            for model_key in model_state_dict.keys():
                # Check if the key has similar pattern (e.g., same layer but different naming)
                if model_key.split('.')[-1] == name.split('.')[-1]:
                    if param.size() == model_state_dict[model_key].size():
                        model_state_dict[model_key].copy_(param)
                        print(f"Mapped {name} to {model_key}")
                        found_match = True
                        break
            
            if not found_match:
                print(f"Parameter {name} not found in model")
    
    model.eval()
    print("Model loaded successfully")

    # Check if we should use streamed inference
    if args.inference_csv:
        if not is_ddp or dist.get_rank() == 0:
            print(f"Using parallel streamed inference mode from CSV: {args.inference_csv}")
        
        # Create directory for output file if needed
        output_dir = os.path.dirname(args.inference_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        # Call the streamed inference function
        inference_start = time.time()
        streamed_inference(
            args, 
            model, 
            device, 
            feature_sizes,
            std_scaler=std_scaler, 
            is_ddp=is_ddp, 
            local_rank=local_rank, 
            world_size=world_size
        )
        inference_end = time.time()
        
        if not is_ddp or dist.get_rank() == 0:
            print(f"Inference completed in {inference_end - inference_start:.2f} seconds")
        return

