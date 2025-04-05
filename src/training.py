# training.py


# Standard libraries
import math
import time
import gc
from datetime import datetime
import pickle

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For distributed training
import torch.distributed as dist

# Progress bar
import tqdm

# File storage
import h5py

# Local imports
from utils import get_layer_wise_learning_rates, is_main_process, safe_get_rank
from model import WeightedL1Loss, WeightedMSELoss
import numpy as np
import wandb


# Training

def train_gnn(
    model,
    train_loader,
    val_loader,
    test_loader,  # Keep test_loader for final evaluation after CV
    num_epochs,
    learning_rate,
    device,
    early_stopping=False,
    task_type='regression',
    mixed_precision=False,
    num_tasks=1,
    multitask_weights=None,
    std_scaler=None,
    is_ddp=False,
    current_args=None
):
    """
    Train a GNN model.
    
    Args:
        model: The GNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_epochs: Number of training epochs
        learning_rate: Base learning rate
        device: Device to train on
        early_stopping: Whether to use early stopping
        task_type: Type of task ('regression' or 'multitask')
        mixed_precision: Whether to use mixed precision training
        num_tasks: Number of tasks for multi-task learning
        multitask_weights: Optional weights for multi-task learning
        std_scaler: Optional standard scaler for normalization
        is_ddp: Whether distributed data parallel is enabled
        current_args: Additional arguments
        
    Returns:
        Trained model
    """
    # Initialize model weights
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.init_weights()
    else:
        model.init_weights()
        
    model.train()  # Set initial train mode

    # Setup optimizer based on layer-wise learning rate decay if enabled
    if current_args.layer_wise_lr_decay:
        model_to_use = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        parameter_groups = get_layer_wise_learning_rates(
            model_to_use,
            learning_rate, 
            decay_factor=current_args.lr_decay_factor
        )
        optimizer = torch.optim.Adam(parameter_groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup loss function
    if current_args.loss_function == 'l1':
        if task_type == 'multitask':
            #if multitask_weights is not None:
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedL1Loss(w_tensor)
            if safe_get_rank() == 0:
                print(f"Using WeightedL1Loss for multitask with weights = {multitask_weights}")
            # else:
            #     criterion = nn.L1Loss()
        elif task_type == 'regression':
            criterion = nn.L1Loss()
    elif current_args.loss_function == 'mse':  # MSE option
        if task_type == 'multitask':
            # if multitask_weights is not None:
            w_tensor = torch.tensor(multitask_weights, dtype=torch.float)
            criterion = WeightedMSELoss(w_tensor)  # Use new WeightedMSELoss class
            if safe_get_rank() == 0:
                print(f"Using WeightedMSELoss for multitask with weights = {multitask_weights}")
            # else:
            #     criterion = nn.MSELoss()
        elif task_type == 'regression':
            criterion = nn.MSELoss()
    else:
        raise ValueError(f"Invalid loss function: {current_args.loss_function}")

    # Setup learning rate scheduler
    if current_args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=current_args.lr_reduce_factor,
            patience=int(current_args.lr_patience),  # Half the main patience
            verbose=True
        )
    elif current_args.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=current_args.lr_cosine_t_max,
            eta_min=0,
            verbose=True
        )
    elif current_args.lr_scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=current_args.lr_step_size,
            gamma=current_args.lr_step_gamma,
            verbose=True
        )
    elif current_args.lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=current_args.lr_exp_gamma,
            verbose=True
        )
    else:
        scheduler = None  # No scheduler

    # Setup early stopping
    patience = current_args.patience  # Use patience from command-line arguments
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    best_model = None
    best_epoch = 0

    # Setup mixed precision scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device.type == 'cuda') else None

    def maybe_set_epoch(loader, epoch):
        # If using a DistributedSampler, call set_epoch for deterministic data splitting
        if hasattr(loader, "sampler") and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(epoch)

    # Track epoch times for performance monitoring
    epoch_times = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        maybe_set_epoch(train_loader, epoch)
        model.train()

        # Track local sums for global average training loss
        epoch_local_loss_sum = 0.0
        epoch_local_count = 0

        # Training loop
        pbar = tqdm.tqdm(enumerate(train_loader),
                       total=len(train_loader),
                       desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in pbar:
            if batch is None:
                continue

            batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
            batch_indices = batch.batch_indices.to(device)
            batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}

            if isinstance(batch.targets, list):
                continue
            else:
                targets = batch.targets.to(device)

            total_charges = batch.total_charges.to(device)
            tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
            cis_indices = batch.final_cis_tensor.to(device)
            trans_indices = batch.final_trans_tensor.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs, _, _ = model(
                        batch_atom_features,
                        batch_multi_hop_edges,
                        batch_indices,
                        total_charges,
                        tetrahedral_indices,
                        cis_indices,
                        trans_indices
                    )
                    
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, _, _ = model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                if torch.isnan(outputs).any():
                    print("NaN found in outputs!")
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accumulate local sums for train loss
            batch_size = targets.size(0)
            epoch_local_loss_sum += loss.item() * batch_size
            epoch_local_count += batch_size

        # All-reduce to get global average train loss (for DDP)
        if is_ddp and dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([epoch_local_loss_sum, epoch_local_count],
                                      dtype=torch.float, device=device)
            dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
            global_sum = local_tensor[0].item()
            global_count = local_tensor[1].item()
            epoch_train_loss = global_sum / global_count if global_count > 0 else 0.0
        else:
            epoch_train_loss = epoch_local_loss_sum / epoch_local_count if epoch_local_count > 0 else 0.0

        # Evaluate on validation set
        with torch.no_grad():
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                task_type=task_type,
                mixed_precision=mixed_precision,
                num_tasks=num_tasks,
                std_scaler=std_scaler,
                is_ddp=is_ddp
            )

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])  # For ReduceLROnPlateau, use validation loss
            else:
                scheduler.step()  # For others, just step

        epoch_end_time = time.time()  # End time for the current epoch
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # Handle best model saving and early stopping (main process only)
        if (not dist.is_initialized()) or (safe_get_rank() == 0):
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch + 1
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    raw_state = model.module.state_dict()
                else:
                    raw_state = model.state_dict()
                best_model = {k: v.cpu() for k, v in raw_state.items()}
                patience_counter = 0
                best_metrics = {
                    'epoch': best_epoch,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            else:
                patience_counter += 1

            stop_training = torch.tensor([0], dtype=torch.uint8, device=device)

            # Log metrics to wandb if enabled
            if current_args.enable_wandb:
                wandb_dict = {
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "val_loss": val_metrics['loss'],
                    "epoch_time": epoch_duration
                }

                if task_type == 'multitask':
                    wandb_dict.update({
                        "val_mae_avg": val_metrics.get('mae'),
                        "val_rmse_avg": val_metrics.get('rmse'),
                        "val_r2_avg": val_metrics.get('r2'),
                    })
                    if 'mae_per_target' in val_metrics:
                        for i, mae_i in enumerate(val_metrics['mae_per_target']):
                            wandb_dict[f"val_mae_target_{i}"] = mae_i
                        for i, rmse_i in enumerate(val_metrics['rmse_per_target']):
                            wandb_dict[f"val_rmse_target_{i}"] = rmse_i
                        for i, r2_i in enumerate(val_metrics['r2_per_target']):
                            wandb_dict[f"val_r2_target_{i}"] = r2_i
                else:
                    wandb_dict.update({
                        "val_mae": val_metrics.get('mae'),
                        "val_rmse": val_metrics.get('rmse'),
                        "val_r2": val_metrics.get('r2'),
                    })
                
                if is_main_process():
                    wandb.log(wandb_dict)

            # Print progress
            if task_type == 'multitask':
                print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.8f}")
                print(f"[Train Loss: {epoch_train_loss:.5f}] "
                      f"Val Loss: {val_metrics['loss']:.5f}, MAE: {val_metrics['mae']:.5f}, "
                      f"RMSE: {val_metrics['rmse']:.5f}, R2: {val_metrics['r2']:.5f}")
                if 'mae_per_target' in val_metrics:
                    for i, (mae_i, rmse_i, r2_i) in enumerate(zip(
                        val_metrics['mae_per_target'],
                        val_metrics['rmse_per_target'],
                        val_metrics['r2_per_target']
                    )):
                        print(f"  [Val Target {i}] MAE={mae_i:.5f}, RMSE={rmse_i:.5f}, R2={r2_i:.5f}")
            else:
                print(f"\nEpoch {epoch + 1}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.8f}")
                print(f"[Train Loss: {epoch_train_loss:.5f}] "
                      f"Val => Loss: {val_metrics['loss']:.5f}, MAE: {val_metrics['mae']:.5f}, "
                      f"RMSE: {val_metrics['rmse']:.5f}, R2: {val_metrics['r2']:.5f}")

            # Check early stopping
            if early_stopping and patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                stop_training = torch.tensor([1], dtype=torch.uint8, device=device)

        else:
            stop_training = torch.tensor([0], dtype=torch.uint8, device=device)

        # Broadcast early stopping signal to all processes (for DDP)
        if is_ddp and dist.is_initialized():
            dist.broadcast(stop_training, src=0)

        if stop_training.item() == 1:
            break

    # Load best model if available
    if best_model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict({k: v.to(device) for k, v in best_model.items()})
        else:
            model.load_state_dict({k: v.to(device) for k, v in best_model.items()})

    # Broadcast best model to all processes (for DDP)
    if is_ddp and dist.is_initialized():
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    # Calculate average epoch time
    if len(epoch_times) > 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
    else:
        avg_epoch_time = 0

    # Log best metrics to wandb
    if best_metrics is not None and (not dist.is_initialized() or safe_get_rank() == 0):
        if current_args.enable_wandb:
            wandb.run.summary.update(best_metrics)
            wandb.run.summary.update({"avg_epoch_time": avg_epoch_time})

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    return model

# Evaluation

@torch.no_grad()
def evaluate(
    model, 
    data_loader, 
    criterion, 
    device, 
    task_type='regression',
    sae_normalizer=None,  
    mixed_precision=False,
    num_tasks=1,
    std_scaler=None,
    is_ddp=False
):
    """
    Evaluate model on a given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to evaluate on
        task_type: Type of task ('regression' or 'multitask')
        sae_normalizer: Optional SAE normalizer
        mixed_precision: Whether to use mixed precision
        num_tasks: Number of tasks
        std_scaler: Optional standard scaler
        is_ddp: Whether distributed data parallel is enabled
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_size = 0

    all_preds_list = []
    all_targets_list = []
    total_loss = 0.0

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}

        if isinstance(batch.targets, list):
            continue
        else:
            targets = batch.targets.to(device)

        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        batch_size = targets.size(0)
        total_size += batch_size

        if mixed_precision and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs, _, _ = model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
                loss = criterion(outputs, targets)
        else:
            outputs, _, _ = model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices
            )
            if torch.isnan(outputs).any():
                print("NaN found in outputs!")
            loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size

        # Collect local predictions/targets
        all_preds_list.append(outputs.cpu().numpy())
        all_targets_list.append(targets.cpu().numpy())

    # Compute local average loss
    avg_loss = total_loss / (total_size if total_size > 0 else 1)

    # Calculate metrics based on task type
    if task_type == 'multitask':
        if len(all_preds_list) == 0:
            metrics = {'loss': avg_loss}
        else:
            # Debug
            if len(all_preds_list) > 0 and len(all_targets_list) > 0:
                Y_pred_debug = np.concatenate(all_preds_list, axis=0)
                Y_true_debug = np.concatenate(all_targets_list, axis=0)

                # Check each column for NaNs
                for col_idx in range(Y_pred_debug.shape[1]):
                    pred_nans = np.isnan(Y_pred_debug[:, col_idx]).sum()
                    true_nans = np.isnan(Y_true_debug[:, col_idx]).sum()
                    if pred_nans > 0 or true_nans > 0:
                        print(f"DEBUG - Column {col_idx}: Y_pred NaNs={pred_nans}, Y_true NaNs={true_nans}")

            Y_pred = np.concatenate(all_preds_list, axis=0)
            Y_true = np.concatenate(all_targets_list, axis=0)

            if std_scaler is not None:
                Y_pred = std_scaler.inverse_transform(Y_pred)
                Y_true = std_scaler.inverse_transform(Y_true)

            mae_vals = []
            rmse_vals = []
            r2_vals = []
            M = Y_true.shape[1]
            for m in range(M):
                mae_m = mean_absolute_error(Y_true[:, m], Y_pred[:, m])
                rmse_m = math.sqrt(mean_squared_error(Y_true[:, m], Y_pred[:, m]))
                r2_m = r2_score(Y_true[:, m], Y_pred[:, m])
                mae_vals.append(mae_m)
                rmse_vals.append(rmse_m)
                r2_vals.append(r2_m)

            mae_avg = float(np.mean(mae_vals))
            rmse_avg = float(np.mean(rmse_vals))
            r2_avg = float(np.mean(r2_vals))

            metrics = {
                'loss': avg_loss,
                'mae': mae_avg,
                'rmse': rmse_avg,
                'r2': r2_avg,
                'mae_per_target': mae_vals,
                'rmse_per_target': rmse_vals,
                'r2_per_target': r2_vals
            }
    else:
        # single-task regression
        if len(all_preds_list) == 0:
            metrics = {'loss': avg_loss}
        else:
            preds_np = np.concatenate(all_preds_list, axis=0)
            targets_np = np.concatenate(all_targets_list, axis=0)
            if std_scaler is not None:
                preds_np = std_scaler.inverse_transform(preds_np)
                targets_np = std_scaler.inverse_transform(targets_np)
            rmse_value = math.sqrt(mean_squared_error(targets_np, preds_np))
            metrics = {
                'loss': avg_loss,
                'mae': mean_absolute_error(targets_np, preds_np),
                'rmse': rmse_value,
                'r2': r2_score(targets_np, preds_np)
            }

    # Combine metrics across ranks for DDP
    if is_ddp and dist.is_available() and dist.is_initialized():
        # All-reduce the total_loss and total_size
        local_tensor = torch.tensor([total_loss, total_size], dtype=torch.float, device=device)
        dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        global_loss_sum = local_tensor[0].item()
        global_count = local_tensor[1].item()

        if global_count > 0:
            global_avg_loss = global_loss_sum / global_count
        else:
            global_avg_loss = 0.0

        # Gather predictions/targets on rank 0, then compute final metrics
        rank = safe_get_rank()
        if task_type == 'multitask':
            # Flatten local preds
            if len(all_preds_list) == 0:
                local_preds_np = np.zeros((0, num_tasks), dtype=np.float32)
                local_targs_np = np.zeros((0, num_tasks), dtype=np.float32)
            else:
                local_preds_np = np.concatenate(all_preds_list, axis=0).astype(np.float32)
                local_targs_np = np.concatenate(all_targets_list, axis=0).astype(np.float32)

            global_preds = gather_ndarray_to_rank0(local_preds_np, device)
            global_targs = gather_ndarray_to_rank0(local_targs_np, device)

            if rank == 0 and global_preds.shape[0] > 0:
                # Invert transform if needed
                if std_scaler is not None:
                    global_preds = std_scaler.inverse_transform(global_preds)
                    global_targs = std_scaler.inverse_transform(global_targs)

                M = global_targs.shape[1]
                mae_vals = []
                rmse_vals = []
                r2_vals = []
                for m in range(M):
                    mae_m = mean_absolute_error(global_targs[:, m], global_preds[:, m])
                    rmse_m = math.sqrt(mean_squared_error(global_targs[:, m], global_preds[:, m]))
                    r2_m = r2_score(global_targs[:, m], global_preds[:, m])
                    mae_vals.append(mae_m)
                    rmse_vals.append(rmse_m)
                    r2_vals.append(r2_m)

                final_metrics = {
                    'loss': global_avg_loss,
                    'mae': float(np.mean(mae_vals)),
                    'rmse': float(np.mean(rmse_vals)),
                    'r2': float(np.mean(r2_vals)),
                    'mae_per_target': mae_vals,
                    'rmse_per_target': rmse_vals,
                    'r2_per_target': r2_vals
                }
            elif rank == 0:
                # No data
                final_metrics = {'loss': global_avg_loss}
            else:
                final_metrics = {}
        else:
            # single-task regression
            if len(all_preds_list) == 0:
                local_preds_np = np.zeros((0,1), dtype=np.float32)
                local_targs_np = np.zeros((0,1), dtype=np.float32)
            else:
                local_preds_np = np.concatenate(all_preds_list, axis=0).astype(np.float32)
                local_targs_np = np.concatenate(all_targets_list, axis=0).astype(np.float32)

            global_preds = gather_ndarray_to_rank0(local_preds_np, device)
            global_targs = gather_ndarray_to_rank0(local_targs_np, device)

            if rank == 0 and global_preds.shape[0] > 0:
                if std_scaler is not None:
                    global_preds = std_scaler.inverse_transform(global_preds)
                    global_targs = std_scaler.inverse_transform(global_targs)

                mae_val = mean_absolute_error(global_targs, global_preds)
                rmse_val = math.sqrt(mean_squared_error(global_targs, global_preds))
                r2_val = r2_score(global_targs, global_preds)
                final_metrics = {
                    'loss': global_avg_loss,
                    'mae': mae_val,
                    'rmse': rmse_val,
                    'r2': r2_val
                }
            elif rank == 0:
                final_metrics = {'loss': global_avg_loss}
            else:
                final_metrics = {}

        # Broadcast final_metrics to all ranks
        final_metrics_pickled = None
        if rank == 0:
            import pickle
            final_metrics_pickled = pickle.dumps(final_metrics)

        # Rank 0 sends length to others
        if rank == 0:
            length_t = torch.tensor([len(final_metrics_pickled)], dtype=torch.long, device=device)
        else:
            length_t = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(length_t, src=0)

        # Broadcast the actual dictionary
        if rank != 0:
            final_metrics_pickled = bytearray(length_t.item())
        final_metrics_byte = torch.ByteTensor(list(final_metrics_pickled)).to(device)
        dist.broadcast(final_metrics_byte, src=0)
        
        # Unpickle on non-zero ranks
        if rank != 0:
            import pickle
            final_metrics = pickle.loads(final_metrics_byte.cpu().numpy().tobytes())

        # Overwrite metrics with final_metrics
        metrics = final_metrics

    return metrics

# Prediction functions

@torch.no_grad()
def predict_gnn(
    model,
    data_loader,
    device,
    task_type='regression',
    std_scaler=None,
    is_ddp=False
):
    """
    Returns a list of predictions in CPU numpy form.

    Args:
        model: Model to use for prediction
        data_loader: DataLoader with input data
        device: Device to run inference on
        task_type: Type of task ('regression' or 'multitask')
        std_scaler: Optional standard scaler
        is_ddp: Whether DDP is enabled
        
    Returns:
        Array of predictions for this rank's portion of the dataset
    """
    model.eval()
    preds_all = []

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        outputs, _, _ = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices
        )

        preds_all.append(outputs.detach().cpu().numpy())

    if len(preds_all) == 0:
        local_preds = np.array([])
    else:
        local_preds = np.concatenate(preds_all, axis=0)
        if task_type in ['regression', 'multitask'] and len(local_preds.shape) == 1:
            local_preds = local_preds.reshape(-1, 1)
        if std_scaler is not None and task_type in ['regression', 'multitask']:
            local_preds = std_scaler.inverse_transform(local_preds)

    # If DDP is enabled, combine predictions across all ranks
    if is_ddp and dist.is_initialized():
        # Convert local predictions to tensor
        local_tensor = torch.from_numpy(local_preds).float().to(device)
        local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=device)
        sizes_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes_list, local_size)

        max_size = max(s.item() for s in sizes_list)
        # Pad local tensor if needed
        if local_size < max_size:
            pad = (max_size - local_size).item()
            local_tensor = F.pad(local_tensor, (0, 0, 0, pad))  # pad rows

        # Gather predictions from all ranks
        gathered = [torch.zeros((max_size, local_tensor.shape[1]), device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_tensor)
        
        # Combine predictions on rank 0
        cat_list = []
        for i in range(dist.get_world_size()):
            cat_list.append(gathered[i][:sizes_list[i], :].cpu().numpy())

        # Return combined predictions on rank 0, local predictions on other ranks
        if dist.get_rank() == 0:
            all_preds_np = np.concatenate(cat_list, axis=0)
            return all_preds_np
        else:
            return local_preds
    else:
        return local_preds

@torch.no_grad()
def predict_with_mc_dropout(
    model,
    data_loader,
    device,
    num_samples=30,
    task_type='regression',
    std_scaler=None,
    is_ddp=False
):
    """
    Get predictions with uncertainty estimates using Monte Carlo Dropout.
    
    Args:
        model: Model to use for prediction
        data_loader: DataLoader with input data
        device: Device to run inference on
        num_samples: Number of MC dropout samples
        task_type: Type of task ('regression' or 'multitask')
        std_scaler: Optional standard scaler
        is_ddp: Whether DDP is enabled
        
    Returns:
        Tuple of (mean_predictions, uncertainties)
    """
    # Set model to eval mode
    model.eval()
    
    # Enable dropout layers during inference
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    
    # Apply to model (handle DDP wrapper if present)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.apply(enable_dropout)
    else:
        model.apply(enable_dropout)
    
    all_predictions = []
    
    # Add tqdm for Monte Carlo samples
    for _ in tqdm.tqdm(range(num_samples), desc="MC Dropout Samples", disable=is_ddp and dist.get_rank() != 0):
        sample_preds = []
        
        # Use a nested tqdm for batches
        batch_iterator = tqdm.tqdm(
            data_loader, 
            desc=f"Sample {_ + 1}/{num_samples}", 
            leave=False,
            disable=is_ddp and dist.get_rank() != 0
        )
        
        for batch in batch_iterator:
            if batch is None:
                continue
                
            batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
            batch_indices = batch.batch_indices.to(device)
            batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
            total_charges = batch.total_charges.to(device)
            tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
            cis_indices = batch.final_cis_tensor.to(device)
            trans_indices = batch.final_trans_tensor.to(device)

            outputs, _, _ = model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices
            )
            
            sample_preds.append(outputs.detach().cpu().numpy())
        
        if len(sample_preds) > 0:
            # Concatenate all batch predictions for this MC sample
            sample_pred_array = np.concatenate(sample_preds, axis=0)
            
            # Apply inverse scaling if needed
            if std_scaler is not None and task_type in ['regression', 'multitask']:
                sample_pred_array = std_scaler.inverse_transform(sample_pred_array)
                
            all_predictions.append(sample_pred_array)
    
    # If we have predictions, calculate statistics
    if len(all_predictions) > 0:
        # Stack to shape [num_samples, num_molecules, num_tasks]
        stacked_predictions = np.stack(all_predictions, axis=0)
        
        # Calculate mean and standard deviation across samples
        mean_predictions = np.mean(stacked_predictions, axis=0)
        uncertainties = np.std(stacked_predictions, axis=0)
        
        return mean_predictions, uncertainties
    else:
        return np.array([]), np.array([])

@torch.no_grad()
def predict_gnn_with_smiles(model, data_loader, device, task_type, std_scaler=None, is_ddp=False):
    """
    Get predictions with corresponding SMILES strings.
    
    Args:
        model: Model to use for prediction
        data_loader: DataLoader with input data
        device: Device to run inference on
        task_type: Type of task ('regression' or 'multitask')
        std_scaler: Optional standard scaler
        is_ddp: Whether DDP is enabled
        
    Returns:
        Tuple of (predictions, smiles_list)
    """
    model.eval()
    all_preds = []
    all_smiles = []

    for batch in data_loader:
        if batch is None:
            continue

        # Prepare inputs
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        outputs, _, _ = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices
        )

        preds_np = outputs.detach().cpu().numpy()
        all_preds.append(preds_np)
        # "batch.smiles_list" contains the SMILES strings for this batch
        all_smiles.extend(batch.smiles_list)

    if len(all_preds) == 0:
        return np.array([], dtype=np.float32), []

    preds_array = np.concatenate(all_preds, axis=0)
    # Apply inverse scaling if needed
    if std_scaler is not None and task_type in ["regression", "multitask"]:
        preds_array = std_scaler.inverse_transform(preds_array)

    return preds_array, all_smiles

# Embedding Extraction and Partial Charge Extraction

def extract_all_embeddings(model, train_loader, val_loader, test_loader, device, output_path, 
                          embedding_type='pooled', include_atom_embeddings=False):
    """
    Extract molecular embeddings from all data loaders (train, validation, test)
    and save them to a structured HDF5 file.
    
    Args:
        model: The trained GNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run inference on
        output_path: Path to save the embeddings
        embedding_type: Type of embedding to extract ('pooled' by default)
        include_atom_embeddings: Whether to also extract atom-level embeddings
    """
    model.eval()
    
    # Dictionary to store all embeddings and metadata
    dataset_embeddings = {
        'train': {'mol_embeddings': [], 'smiles': [], 'atom_embeddings': {}, 'atom_counts': {}},
        'val': {'mol_embeddings': [], 'smiles': [], 'atom_embeddings': {}, 'atom_counts': {}},
        'test': {'mol_embeddings': [], 'smiles': [], 'atom_embeddings': {}, 'atom_counts': {}}
    }
    
    # Create hooks for embeddings
    mol_embeddings_hook = []
    atom_embeddings_hook = []
    
    def mol_hook_fn(module, input, output):
        mol_embeddings_hook.append(output[0].detach().cpu().numpy())
    
    def atom_hook_fn(module, input, output):
        atom_embeddings_hook.append(output.detach().cpu().numpy())
    
    # Register the hooks at the appropriate layers
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        mol_hook = model.module.pooling.register_forward_hook(mol_hook_fn)
        if include_atom_embeddings:
            atom_hook = model.module.concat_self_other.register_forward_hook(atom_hook_fn)
    else:
        mol_hook = model.pooling.register_forward_hook(mol_hook_fn)
        if include_atom_embeddings:
            atom_hook = model.concat_self_other.register_forward_hook(atom_hook_fn)
    
    # Process each dataset
    for dataset_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"Extracting embeddings from {dataset_name} set...")
        
        # Process each batch
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc=f"Processing {dataset_name}")):
            if batch is None:
                continue
            
            # Clear the hook lists before each batch
            mol_embeddings_hook.clear()
            atom_embeddings_hook.clear()
            
            # Prepare batch data for forward pass
            batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
            batch_indices = batch.batch_indices.to(device)
            batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
            total_charges = batch.total_charges.to(device)
            tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
            cis_indices = batch.final_cis_tensor.to(device)
            trans_indices = batch.final_trans_tensor.to(device)

            # Forward pass to trigger the hooks
            _ = model(
                batch_atom_features,
                batch_multi_hop_edges,
                batch_indices,
                total_charges,
                tetrahedral_indices,
                cis_indices,
                trans_indices
            )
            
            # Save molecule embeddings
            batch_mol_embeddings = mol_embeddings_hook[0]
            dataset_embeddings[dataset_name]['mol_embeddings'].append(batch_mol_embeddings)
            
            # Save SMILES
            dataset_embeddings[dataset_name]['smiles'].extend(batch.smiles_list)
            
            # Process atom embeddings if requested
            if include_atom_embeddings:
                batch_atom_embeddings = atom_embeddings_hook[0]
                batch_indices_np = batch_indices.cpu().numpy()
                
                # Group atoms by molecule
                unique_mol_indices = np.unique(batch_indices_np)
                
                # For each molecule in this batch
                for mol_idx in unique_mol_indices:
                    # Get SMILES for this molecule
                    mol_smiles = batch.smiles_list[mol_idx]
                    
                    # Find atoms belonging to this molecule
                    mask = (batch_indices_np == mol_idx)
                    mol_atoms = batch_atom_embeddings[mask]
                    
                    # Store atom embeddings by SMILES
                    current_idx = len(dataset_embeddings[dataset_name]['mol_embeddings']) * loader.batch_size + mol_idx
                    dataset_embeddings[dataset_name]['atom_embeddings'][current_idx] = mol_atoms
                    dataset_embeddings[dataset_name]['atom_counts'][mol_smiles] = len(mol_atoms)
    
    # Remove the hooks
    mol_hook.remove()
    if include_atom_embeddings:
        atom_hook.remove()
    
    # Combine and finalize embeddings
    results = {}
    for dataset_name, data in dataset_embeddings.items():
        if data['mol_embeddings']:
            # Stack molecule embeddings
            results[f"{dataset_name}_mol_embeddings"] = np.vstack(data['mol_embeddings'])
            results[f"{dataset_name}_smiles"] = data['smiles']
            
            if include_atom_embeddings:
                results[f"{dataset_name}_atom_embeddings"] = data['atom_embeddings']
                results[f"{dataset_name}_atom_counts"] = data['atom_counts']
    
    # Total molecule counts
    train_count = len(results.get('train_smiles', []))
    val_count = len(results.get('val_smiles', []))
    test_count = len(results.get('test_smiles', []))
    total_count = train_count + val_count + test_count
    
    print(f"Extracted embeddings for {total_count} molecules: {train_count} train, {val_count} validation, {test_count} test")
    
    # Save the embeddings to HDF5
    save_embeddings_to_hdf5(results, output_path, include_atom_embeddings)
    
    return results

def save_embeddings_to_hdf5(results, output_path, include_atom_embeddings=False):
    """
    Save extracted embeddings to an HDF5 file with a structured format.
    
    Args:
        results: Dictionary containing embeddings and metadata
        output_path: Path to save the HDF5 file
        include_atom_embeddings: Whether atom embeddings are included
    """
    with h5py.File(output_path, 'w') as f:
        # Create dataset groups
        train_group = f.create_group('train')
        val_group = f.create_group('validation')
        test_group = f.create_group('test')
        
        # Create metadata group
        metadata = f.create_group('metadata')
        
        # Save dataset sizes
        train_size = len(results.get('train_smiles', []))
        val_size = len(results.get('val_smiles', []))
        test_size = len(results.get('test_smiles', []))
        
        metadata.attrs['train_size'] = train_size
        metadata.attrs['validation_size'] = val_size
        metadata.attrs['test_size'] = test_size
        metadata.attrs['total_size'] = train_size + val_size + test_size
        metadata.attrs['include_atom_embeddings'] = include_atom_embeddings
        
        # Save molecule embeddings and SMILES
        for dataset_name, group in [('train', train_group), ('val', val_group), ('test', test_group)]:
            # Check if this dataset has data
            mol_embeddings_key = f"{dataset_name}_mol_embeddings"
            smiles_key = f"{dataset_name}_smiles"
            
            if mol_embeddings_key in results and smiles_key in results:
                mol_embeddings = results[mol_embeddings_key]
                smiles = results[smiles_key]
                
                # Save molecular embeddings
                group.create_dataset('mol_embeddings', data=mol_embeddings)
                
                # Save SMILES strings
                dt = h5py.special_dtype(vlen=str)
                smiles_dataset = group.create_dataset('smiles', (len(smiles),), dtype=dt)
                for i, smi in enumerate(smiles):
                    smiles_dataset[i] = smi
                
                # Save atom embeddings if available
                if include_atom_embeddings:
                    atom_key = f"{dataset_name}_atom_embeddings"
                    if atom_key in results:
                        atom_group = group.create_group('atom_embeddings')
                        atom_counts = group.create_dataset('atom_counts', (len(smiles),), dtype=np.int32)
                        
                        atom_embeddings = results[atom_key]
                        for idx, atoms in atom_embeddings.items():
                            atom_dataset = atom_group.create_dataset(f'mol_{idx}', data=atoms)
                        
                        # Save atom counts
                        counts_dict = results.get(f"{dataset_name}_atom_counts", {})
                        for i, smi in enumerate(smiles):
                            atom_counts[i] = counts_dict.get(smi, 0)
    
    print(f"Successfully saved embeddings to: {output_path}")

@torch.no_grad()
def extract_partial_charges(model, data_loader, device):
    """
    Iterates over the data_loader, does a forward pass, 
    and collects partial charges if available.
    
    Args:
        model: Model to extract partial charges from
        data_loader: DataLoader with input data
        device: Device to run inference on
        
    Returns:
        List of tuples (smiles, [q1, q2, ...]) for each molecule
    """
    model.eval()
    results = []

    for batch_idx, batch in enumerate(data_loader):
        if batch is None:
            continue
        batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
        batch_indices = batch.batch_indices.to(device)
        batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
        total_charges = batch.total_charges.to(device)
        tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
        cis_indices = batch.final_cis_tensor.to(device)
        trans_indices = batch.final_trans_tensor.to(device)

        # Forward pass with partial charges
        _, _, partial_charges = model(
            batch_atom_features,
            batch_multi_hop_edges,
            batch_indices,
            total_charges,
            tetrahedral_indices,
            cis_indices,
            trans_indices
        )

        # Skip if partial_charges is None
        if partial_charges is None:
            for smi in batch.smiles_list:
                results.append((smi, []))
            continue

        # Group partial charges by molecule
        unique_mols = batch_indices.unique().tolist()
        for mol_id in unique_mols:
            smi = batch.smiles_list[mol_id]
            mask = (batch_indices == mol_id)
            q_vals = partial_charges[mask].detach().cpu().numpy().tolist()
            results.append((smi, q_vals))

    return results


def extract_embeddings_main(args, model, train_loader, val_loader, test_loader, device):
    """
    Extract molecular embeddings from the trained model for all datasets.
    
    This function:
    1. Runs molecules through the trained GNN model
    2. Extracts embeddings from the pooling layer (molecule-level representations)
    3. Optionally extracts atom-level embeddings 
    4. Saves everything to an HDF5 file for downstream analysis
    
    Args:
        args: Command line arguments
        model: The trained GNN model
        train_loader: DataLoader for training data 
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run extraction on
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING MOLECULAR EMBEDDINGS")
    print(f"{'='*80}")
    print(f"• Output file: {args.embeddings_output_path}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create hooks for embeddings (pooling output = molecular embeddings)
    mol_embeddings = []
    atom_embeddings = []
    batch_smiles = []
    
    # Define hook functions
    def mol_hook_fn(module, input, output):
        mol_embeddings.append(output[0].detach().cpu().numpy())
    
    def atom_hook_fn(module, input, output):
        atom_embeddings.append(output.detach().cpu().numpy())
    
    # Register hooks at appropriate layers
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        mol_hook = model.module.pooling.register_forward_hook(mol_hook_fn)
        if args.include_atom_embeddings:
            atom_hook = model.module.concat_self_other.register_forward_hook(atom_hook_fn)
    else:
        mol_hook = model.pooling.register_forward_hook(mol_hook_fn)
        if args.include_atom_embeddings:
            atom_hook = model.concat_self_other.register_forward_hook(atom_hook_fn)
    
    # Process datasets
    dataset_embeddings = {}
    
    # Function to process a dataset
    def process_dataset(name, loader):
        print(f"Processing {name} dataset...")
        
        # Clear collections for this dataset
        mol_embeddings.clear()
        atom_embeddings.clear()
        batch_smiles.clear()
        
        # Process batches
        total_molecules = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc=f"Extracting {name} embeddings"):
                if batch is None:
                    continue
                
                # Add SMILES to collection
                batch_smiles.extend(batch.smiles_list)
                total_molecules += len(batch.smiles_list)
                
                # Prepare batch for model
                batch_multi_hop_edges = batch.multi_hop_edge_indices.to(device)
                batch_indices = batch.batch_indices.to(device)
                batch_atom_features = {k: v.to(device) for k, v in batch.atom_features_map.items()}
                total_charges = batch.total_charges.to(device)
                tetrahedral_indices = batch.final_tetrahedral_chiral_tensor.to(device)
                cis_indices = batch.final_cis_tensor.to(device)
                trans_indices = batch.final_trans_tensor.to(device)
                
                # Forward pass triggers hooks
                _ = model(
                    batch_atom_features,
                    batch_multi_hop_edges,
                    batch_indices,
                    total_charges,
                    tetrahedral_indices,
                    cis_indices,
                    trans_indices
                )
        
        # Process molecule embeddings
        mols_emb = np.vstack(mol_embeddings) if mol_embeddings else np.array([])
        
        # Return results
        result = {
            'embeddings': mols_emb,
            'smiles': batch_smiles,
            'count': total_molecules
        }
        
        # Process atom embeddings if requested
        if args.include_atom_embeddings and atom_embeddings:
            # Create a mapping from SMILES to atom embeddings
            atom_emb_map = {}
            atom_count_map = {}
            
            # Group atom embeddings by molecule
            for batch_idx, atom_emb_batch in enumerate(atom_embeddings):
                batch_data = loader.dataset.data_list[batch_idx] if hasattr(loader.dataset, 'data_list') else None
                
                if batch_data is not None:
                    # For in-memory dataset
                    mol_smiles = batch_data.smiles
                    atom_emb_map[mol_smiles] = atom_emb_batch
                    atom_count_map[mol_smiles] = len(atom_emb_batch)
            
            result['atom_embeddings'] = atom_emb_map
            result['atom_counts'] = atom_count_map
        
        return result
    
    # Process each dataset
    if train_loader:
        dataset_embeddings['train'] = process_dataset('train', train_loader)
        print(f"• Extracted embeddings for {dataset_embeddings['train']['count']:,} training molecules")
    
    if val_loader:
        dataset_embeddings['validation'] = process_dataset('validation', val_loader)
        print(f"• Extracted embeddings for {dataset_embeddings['validation']['count']:,} validation molecules")
    
    if test_loader:
        dataset_embeddings['test'] = process_dataset('test', test_loader)
        print(f"• Extracted embeddings for {dataset_embeddings['test']['count']:,} test molecules")
    
    # Remove hooks
    mol_hook.remove()
    if args.include_atom_embeddings:
        atom_hook.remove()
    
    # Save embeddings to HDF5
    with h5py.File(args.embeddings_output_path, 'w') as f:
        # Create metadata
        metadata = f.create_group('metadata')
        metadata.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['include_atom_embeddings'] = args.include_atom_embeddings
        
        # Add model information
        model_info = metadata.create_group('model')
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_info.attrs['hidden_dim'] = model.module.hidden_dim if hasattr(model.module, 'hidden_dim') else 0
            model_info.attrs['num_shells'] = model.module.num_shells if hasattr(model.module, 'num_shells') else 0
            emb_dim = model_info.attrs['embedding_dim'] = model.module.embedding_dim if hasattr(model.module, 'embedding_dim') else 0
        else:
            model_info.attrs['hidden_dim'] = model.hidden_dim if hasattr(model, 'hidden_dim') else 0
            model_info.attrs['num_shells'] = model.num_shells if hasattr(model, 'num_shells') else 0
            model_info.attrs['embedding_dim'] = model.embedding_dim if hasattr(model, 'embedding_dim') else 0
        
        # Store embeddings for each dataset
        for dataset_name, data in dataset_embeddings.items():
            # Create dataset group
            dataset_group = f.create_group(dataset_name)
            
            # Store molecular embeddings
            embedding_shape = data['embeddings'].shape
            dataset_group.create_dataset('embeddings', data=data['embeddings'])
            
            # Store SMILES
            dt = h5py.special_dtype(vlen=str)
            smiles_dataset = dataset_group.create_dataset('smiles', (len(data['smiles']),), dtype=dt)
            for i, smi in enumerate(data['smiles']):
                smiles_dataset[i] = smi
            
            # Store atom embeddings if available
            if args.include_atom_embeddings and 'atom_embeddings' in data and 'atom_counts' in data:
                atoms_group = dataset_group.create_group('atom_embeddings')
                atom_counts = dataset_group.create_dataset('atom_counts', (len(data['smiles']),), dtype=np.int32)
                
                # Store atom embeddings for each molecule
                for i, (smi, count) in enumerate(data['atom_counts'].items()):
                    if smi in data['atom_embeddings']:
                        mol_atom_emb = data['atom_embeddings'][smi]
                        atoms_group.create_dataset(f'mol_{i}', data=mol_atom_emb)
                        atom_counts[i] = count
    
    total_count = sum(data['count'] for data in dataset_embeddings.values())
    print(f"\nSuccessfully saved embeddings for {total_count:,} molecules to: {args.embeddings_output_path}")
    print(f"{'='*80}\n")

