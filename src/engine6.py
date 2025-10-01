"""
Contains functions for training, validating, and testing a PyTorch model.
"""
import sys
import math
import numpy as np
from pathlib import Path
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch.nn as nn

import utils
import data_setup
################################################################################

def calculate_class_weights(dataloader):
    class_counts = [0] * 3
    for _, targets in dataloader:
        for target in targets:
            class_counts[target.item()] += 1
    total_count = sum(class_counts)
    class_weights = [total_count / count for count in class_counts]
    return torch.tensor(class_weights, dtype=torch.float)

def run_epoch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              is_training: bool = False,
              criterion: torch.nn.Module = None,
              is_regression: bool = False):
    """Function to run a single training or validation_epoch (depending on
    whether 'is_training' is True or False):
    if is_training==True:
        Turns a target PyTorch model to training mode (with: model.train()) and
        then runs through all of the required training steps (forward pass on
        the training set, loss calculation, optimizer step).
    if is_training==False:
        Turns a target PyTorch model still to training mode (with: model.train())
        and then performs a forward pass on a validation dataset. NB: To get the
        losses, 'model.train()' needs to be on! If we set instead 'model.eval()',
        then losses will NOT be computed!

    Args:
    model: A PyTorch model to be trained.
    dataloader: A PyTorch DataLoader providing the data.
    optimizer: A PyTorch optimizer to use for training the model.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    is_training: Boolean flag indicating whether the model is in training or validation mode.

    Returns:
    A tuple of the train/valid losses in the current epoch, in the form:
        np.array([loss_tot, loss_classifier, loss_mask, loss_box_reg, loss_objectness, loss_rpn_box_reg]).
    """
    
    # # Calculate class weights
    # class_weights = calculate_class_weights(dataloader).to(device)
    
    # Define the CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
   # Select default criterion if not provided
    if criterion is None:
        criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    
    # modes
    if is_training:
        model.train()
        # keep frozen backbone in eval during warm-up to avoid BN/stat drift
        if getattr(model, "_backbone_frozen", False) and hasattr(model, "backbone"):
            model.backbone.eval()
            # Plain models with staged-freeze: keep explicitly-frozen modules in eval()
        if getattr(model, "_staged_freeze", False) and not getattr(model, "_unfrozen", False):
            for mod in getattr(model, "_frozen_modules", []):
                try:
                    mod.eval()
                except Exception:
                    pass
    else:
        model.eval()
        
    # # Put model in TRAIN mode
    # model.train()
    # Initialize losses
    loss_tot = 0
    # Loop through DataLoader batches
    for batch, (images, targets) in enumerate(dataloader):
        # # Images: list/tuple of tensors -> [B, C, H, W]
        # if isinstance(images, (list, tuple)):
        #     images = torch.stack([img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
        #                           for img in images], dim=0)
        # images = images.to(device).float()

        # # Targets: handle regression vectors vs scalar class labels
        # if isinstance(targets, (list, tuple)):
        #     if is_regression:
        #         # Each target is a vector -> stack to [B, D]
        #         tgt_list = []
        #         for t in targets:
        #             t = torch.as_tensor(t, dtype=torch.float32)
        #             t = t.view(-1)  # ensure 1D [D]
        #             tgt_list.append(t)
        #         targets = torch.stack(tgt_list, dim=0)
        #     else:
        #         # Each target should be a scalar class index -> [B]
        #         tgt_list = []
        #         for t in targets:
        #             if isinstance(t, torch.Tensor):
        #                 if t.dim() == 0:
        #                     tt = t
        #                 else:
        #                     tt = t.reshape(-1)[0]
        #             else:
        #                 tt = torch.as_tensor(t)
        #                 tt = tt.reshape(-1)[0]
        #             tgt_list.append(tt.to(dtype=torch.long))
        #         targets = torch.stack(tgt_list, dim=0)
        # # Final dtypes/devices
        # targets = targets.to(device)
        # if is_regression:
        #     targets = targets.float()
        #     if targets.dim() == 1:
        #         targets = targets.unsqueeze(1)  # [B] -> [B, 1]
        # else:
        #     targets = targets.long()
        
        images = torch.stack(images)
        images = images.to(device)
        
        
        targets = torch.stack(targets)
        targets = targets.to(device)
        
    
        
        # Forward pass
        if is_training:
            output = model(images)
            
        else:
            with torch.inference_mode():
                output = model(images)
        
        loss_dict = criterion(output, targets)       
        
        # If in training mode, backpropagate the error, update the weights
        if is_training:
            # Optimizer zero grad
            optimizer.zero_grad()
            # Compute gradients
            loss_dict.backward()
            # Backpropagation (update weights)
            optimizer.step()

        # If the loss is NaN or infinite, stop the training/validation
        if math.isnan(loss_dict) or math.isinf(loss_dict):
            print(f"Loss is NaN or infinite at batch {batch}. Stopping {'training' if is_training else 'validation'}!")
            exit(1)

        # Logging (accumulate losses)
        loss_tot         += loss_dict
        
    # Adjust metrics to get average loss and accuracy per batch
    n_batches        = len(dataloader)
    loss_tot         = loss_tot         / n_batches
    

    return np.array([loss_tot.cpu().detach().numpy()])

################################################################################

# Add a small helper to read param-group LRs by name
def _get_param_group_lrs(optimizer: torch.optim.Optimizer):
    lrs = {}
    for i, g in enumerate(optimizer.param_groups):
        name = g.get("name", f"group_{i}")
        lrs[name] = g["lr"]
    return lrs

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler,
          config,
          writer: torch.utils.tensorboard.writer.SummaryWriter = None,
          checkpoint_file = None,
          optimizer_builder=None,
         ):
    """Trains and validate a PyTorch model.

    Passes a target PyTorch models through a training run_epoch() step and a
    validation run_epoch() step, for a number of epochs, doing both of these in
    the same loop. Calculates, prints and stores evaluation metrics throughout.
    It saves the best model based on the validation loss.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    valid_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    lr_scheduler: The learning rate scheduler.
    config:
        - epochs: An integer indicating how many epochs to train for.
        - device: A target device to compute on (e.g. "cuda" or "cpu").
        - use_scheduler: Boolean flag indicating whether to use or not the lr_scheduler.
    writer: to log information for tensorboard.
    checkpoint_file: file on which the checkpoints are saved.

    Returns:
    A dictionary of training and validation losses. Each metric has a value
    in a list, one value for each epoch.
    """
    epochs        = config.NUM_EPOCHS
    device        = config.DEVICE
    use_scheduler = config.USE_SCHEDULER
    lr            = config.LEARNING_RATE
    
     # Toggle: regression vs classification
    regression_mode = getattr(config, "USE_REGRESSION_TARGETS", False)

    # Choose criterion based on mode (you can switch to SmoothL1Loss if preferred)
    if regression_mode:
        criterion = nn.MSELoss()
    else:
        # class_weights optional:
        # class_weights = calculate_class_weights(train_dataloader).to(device)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = nn.CrossEntropyLoss()

    # Create empty dictionary which will contain all the losses per epoch
    results = {"epoch": [],
               "learning_rate": [],
               "train_loss_tot": [],
               "valid_loss_tot": [],
                }

    # Initialize the best validation loss to infinity
    best_epoch      = 0
    best_valid_loss = float('inf')

    # Loop through training and validation steps, for a number of epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        time_start_epoch = time.time()
        print('\n\n###############################################################')
        print(f"Epoch: {epoch+1}   |  Mode: {'REGRESSION' if regression_mode else 'CLASSIFICATION'}")
        print('###############################################################\n')
        
        # ---- Warm-up unfreeze at the start of the target epoch ----
        # if (getattr(config, "WARMUP_FREEZE_EPOCHS", 0) > 0
        #     and hasattr(model, "adapter")  # only applies to the 6->3 adapter setup
        #     and epoch == getattr(config, "WARMUP_FREEZE_EPOCHS", 0)
        #     and not getattr(model, "_unfrozen", False)):
        if (getattr(config, "WARMUP_FREEZE_EPOCHS", 0) > 0
            and epoch == getattr(config, "WARMUP_FREEZE_EPOCHS", 0)
            and not getattr(model, "_unfrozen", False)):
            print(f"[INFO] Warm-up ended at epoch {epoch}: unfreezing backbone and rebuilding optimizer/scheduler.")
            # # Unfreeze backbone
            # for p in model.backbone.parameters():
            #     p.requires_grad = True
            # # Ensure head and adapter remain trainable
            # if hasattr(model.backbone, "fc"):
            #     for p in model.backbone.fc.parameters():
            #         p.requires_grad = True
            # elif hasattr(model.backbone, "classifier"):
            #     for p in model.backbone.classifier.parameters():
            #         p.requires_grad = True
            # for p in model.adapter.parameters():
            #     p.requires_grad = True
            
            # # mark as unfrozen so run_epoch will not force backbone.eval()
            # setattr(model, "_backbone_frozen", False)
            # Unfreeze paths depending on model type
            # Adapter-wrapped backbones
            if hasattr(model, "backbone"):
                for p in model.backbone.parameters():
                    p.requires_grad = True
                # keep head+adapter trainable
                if hasattr(model.backbone, "fc"):
                    for p in model.backbone.fc.parameters(): p.requires_grad = True
                if hasattr(model.backbone, "classifier"):
                    for p in model.backbone.classifier.parameters(): p.requires_grad = True
                if hasattr(model, "adapter"):
                    for p in model.adapter.parameters(): p.requires_grad = True
                setattr(model, "_backbone_frozen", False)
                            # Plain torchvision models (e.g., resnet50_6ch_pretrained)
            if getattr(model, "_staged_freeze", False):
                # Unfreeze only what we froze
                for mod in getattr(model, "_frozen_modules", []):
                    for p in mod.parameters():
                        p.requires_grad = True
                # Also ensure head stays trainable
                if hasattr(model, "fc"):
                    for p in model.fc.parameters(): p.requires_grad = True
                if hasattr(model, "classifier"):
                    for p in model.classifier.parameters(): p.requires_grad = True
                model._staged_freeze = False
                model._frozen_modules = []

            # Rebuild optimizer so newly-unfrozen params are included
            if optimizer_builder is not None:
                optimizer = optimizer_builder(model, config)
            else:
                # Fallback: simple AdamW over all trainable params
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    params,
                    lr=getattr(config, "LEARNING_RATE", 5e-5),
                    betas=getattr(config, "BETAS", (0.9, 0.999)),
                    eps=getattr(config, "EPS", 1e-8),
                    weight_decay=getattr(config, "WEIGHT_DECAY", 1e-3),
                    amsgrad=getattr(config, "AMSGRAD", False),
                )

            # Ensure param_groups have 'initial_lr' so StepLR can resume cleanly
            for g in optimizer.param_groups:
                if "initial_lr" not in g:
                    g["initial_lr"] = g["lr"]
            
            # Recreate scheduler bound to the new optimizer
            if getattr(config, "USE_SCHEDULER", True):
                sched = getattr(config, "SCHEDULER", "StepLR")
                if sched == "StepLR":
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=config.SCHEDULER_STEP,
                        gamma=config.SCHEDULER_GAMMA,
                        last_epoch=epoch - 1  # keep schedule aligned
                    )
                elif sched == "OneCycleLR":
                    total_steps = (epochs - epoch) * len(train_dataloader)
                    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer=optimizer,
                        max_lr=getattr(config, "LEARNING_RATE", 5e-5),
                        total_steps=total_steps,
                    )
                elif sched == "ReduceLROnPlateau":
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=optimizer,
                        mode="min",
                        factor=getattr(config, "SCHED_FACTOR", 0.5),
                        patience=getattr(config, "SCHED_PATIENCE", 4),
                        threshold=1e-4,
                        cooldown=getattr(config, "SCHED_COOLDOWN", 0),
                        min_lr=getattr(config, "MIN_LR", 1e-7),
                        verbose=True
                                            )
            setattr(model, "_unfrozen", True)
        
        #----------------------------------------------------------------------#
        # Training epoch
        #----------------------------------------------------------------------#
        
        train_losses = np.zeros(1) # 5 losses + 1 total
        train_losses = run_epoch(model=model,
                                 dataloader=train_dataloader,
                                 optimizer=optimizer,
                                 device=device,
                                 is_training=True,
                                 criterion=criterion,
                                 is_regression=regression_mode)
        
        # -------------------------- Scheduler stepping ------------------------ #
        
        # if use_scheduler and lr_scheduler is not None and getattr(config, "SCHEDULER", "StepLR") == "StepLR":
        #     # Update the learning rate according to the `lr_scheduler` chosen
        #     lr_scheduler.step()
        #     lr = lr_scheduler.get_last_lr()[0]
        # else:
        #     # fallback to optimizer LR
        #     lr = optimizer.param_groups[0]["lr"]

        sched = getattr(config, "SCHEDULER", "StepLR")
        if use_scheduler and lr_scheduler is not None:
            if sched == "StepLR":
                lr_scheduler.step()
            elif sched == "ReduceLROnPlateau":
                # Step on validation metric later, after we compute valid_losses
                pass
            # OneCycleLR is stepped per-batch; if you use it, move stepping into run_epoch.
        
        # Determine LR to log: prefer 'backbone' group if named, else first group
        lr = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone"),
                  optimizer.param_groups[0]["lr"])

        #----------------------------------------------------------------------#
        # Validation epoch (pt.1)
        #----------------------------------------------------------------------#
        valid_losses = np.zeros_like(train_losses)
        if valid_dataloader is not None:
            with torch.inference_mode():
                valid_losses = run_epoch(model=model,
                                         dataloader=valid_dataloader,
                                         optimizer=None,
                                         device=device,
                                         is_training=False,
                                         criterion=criterion,
                                         is_regression=regression_mode)
                
        # If ReduceLROnPlateau, step now with the validation loss
        if use_scheduler and lr_scheduler is not None and sched == "ReduceLROnPlateau":
            val_metric = float(valid_losses[0]) if valid_dataloader is not None else float(train_losses[0])
            lr_scheduler.step(val_metric)
            # refresh lr after potential drop
            lr = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone"),
                      optimizer.param_groups[0]["lr"])


        
        
        #----------------------------------------------------------------------#
        # Print on standard output the losses
        #----------------------------------------------------------------------#
        print(f"lr            : {lr:.3e}")
        print(f"train_loss_tot: {train_losses[0]:.4f}")
        print(f"valid_loss_tot: {valid_losses[0]:.4f}")
        
        #----------------------------------------------------------------------#
        # Update the results dictionary with the new losses
        #----------------------------------------------------------------------#
        results["epoch"].append(epoch+1)
        results["learning_rate"].append(lr)
        # Optional: add per-group LR columns
        group_lrs = _get_param_group_lrs(optimizer)
        results.setdefault("lr_backbone", []).append(group_lrs.get("backbone", float("nan")))
        results.setdefault("lr_head", []).append(group_lrs.get("head", float("nan")))
        results.setdefault("lr_adapter", []).append(group_lrs.get("adapter", float("nan")))
        results.setdefault("lr_stem", []).append(group_lrs.get("stem", float("nan")))
        results["train_loss_tot"].append(train_losses[0])
        results["valid_loss_tot"].append(valid_losses[0])
        

        #----------------------------------------------------------------------#
        # Log tensorboard information
        #----------------------------------------------------------------------#
        # Check if there's a writer, if so, log information to it
        if writer is not None:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss_tot": train_losses[0],
                                                "valid_loss_tot": valid_losses[0]},
                               global_step=epoch)
            # writer.close()

        #----------------------------------------------------------------------#
        # Save model if validation loss is improved
        #----------------------------------------------------------------------#
        # If the current validation loss is lower than the best validation
        # loss seen up to now, save the model checkpoint
        valid_loss_tot = valid_losses[0]
        if valid_loss_tot < best_valid_loss:
            best_valid_loss = valid_loss_tot
            best_epoch = epoch
            if checkpoint_file is not None:
                # Save model weights only
                torch.save(obj=model.state_dict(), 
                           f=str(checkpoint_file))
                # # Save full model
                # torch.save(obj=model, 
                #            f=str(Path(checkpoint_file.parent))+'/'+str(config.MODEL_NAME)+'_full.pt')

        #----------------------------------------------------------------------#
        # Print information on standard output
        #----------------------------------------------------------------------#
        #if torch.cuda.is_available():
        #    utils.show_gpu_memory(device)

        epoch_time = time.time() - time_start_epoch
        print('\n[INFO] Epoch {} completed in {:.1f} seconds.\n'.format(epoch+1, epoch_time))

        #----------------------------------------------------------------------#
    
    if writer is not None:
        writer.close()

    #--------------------------------------------------------------------------#
    # Save model weights at the FINAL epoch (not needed, but just in case)
    #--------------------------------------------------------------------------#
    save_last_epoch = False
    if save_last_epoch:
        torch.save(obj=model.state_dict(), 
                f=str(Path(checkpoint_file.parent))+'/'+str(config.MODEL_NAME)+'_lastepoch.pt')

    #--------------------------------------------------------------------------#
    # Save metadata about the whole training process
    #--------------------------------------------------------------------------#
    training_metadata = {
        'best_valid_loss': float(best_valid_loss),
        'epoch_best_valid_loss': best_epoch, # epoch at which the best model is saved (best valid loss)
        #'epochs': epochs,
        #'batch_size': config.BATCH_SIZE,
        #'learning_rate': lr_scheduler.get_last_lr()[0],
        #'learning_rate': config.LEARNING_RATE,
        #'momentum': config.MOMENTUM,
        #'weight_decay': config.WEIGHT_DECAY,
        #'trainable_params': config.TRAINABLE_NAMES,
        'save_dir': str(Path(checkpoint_file.parent)),
        'checkpoint_file': str(checkpoint_file),
        'model_name': config.MODEL_NAME}
    with open(Path(checkpoint_file.parent/'training_metadata.json'), 'w') as f:
        json.dump(training_metadata, f)
    print('\nBest validation loss (model saved) at epoch:', best_epoch)

    # Return the dict 'results' with the info on the losses
    return results

################################################################################
# SOURCE: https://github.com/pytorch/vision/blob/v0.15.2/references/detection/engine.py
################################################################################


# ---- 1) Adapter weight importance (linear adapter only) ----
def adapter_weight_importance(model: torch.nn.Module):
    """
    Returns a dict: channel_index -> sum_j |W[j, channel_index]| for a 1x1 Conv2d adapter.
    """
    if not hasattr(model, "adapter") or not isinstance(model.adapter, nn.Conv2d):
        return None
    W = model.adapter.weight.data  # [3, in_ch, 1, 1]
    if W.dim() != 4 or W.size(2) != 1 or W.size(3) != 1:
        return None
    in_ch = W.size(1)
    imp = {}
    with torch.no_grad():
        # L1 norm across output channels for each input channel
        for c in range(in_ch):
            imp[c] = W[:, c, 0, 0].abs().sum().item()
    return imp

# ---- 2) Gradient-based sensitivity per input channel ----
def channel_gradient_importance(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                device: torch.device,
                                target_class: int = None,
                                max_batches: int = 5):
    """
    Computes mean absolute gradient magnitude per input channel over a few batches.
    If target_class is None, uses CrossEntropy loss; else uses mean logit of target_class.
    Returns np.array of shape [C].
    """
    model.eval()
    grads_accum = None
    batches = 0

    for b, (images, targets) in enumerate(dataloader):
        if b >= max_batches:
            break
        if isinstance(images, (list, tuple)):
            images = torch.stack(images, dim=0)
        if isinstance(targets, (list, tuple)):
            targets = torch.stack(targets, dim=0)

        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()
        images.requires_grad_(True)

        logits = model(images)
        if target_class is None:
            loss = nn.CrossEntropyLoss()(logits, targets)
            scalar = loss
        else:
            scalar = logits[:, target_class].mean()

        model.zero_grad(set_to_none=True)
        scalar.backward()

        g = images.grad.detach().abs().mean(dim=(0, 2, 3))  # [C]
        grads_accum = g if grads_accum is None else grads_accum + g
        images.requires_grad_(False)
        batches += 1

    if grads_accum is None or batches == 0:
        return None
    return (grads_accum / batches).cpu().numpy()

# ---- 3) Channel ablation: delta validation loss when zeroing one channel ----
def channel_ablation_delta_loss(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                device: torch.device,
                                max_batches: int = 50,
                                fill: float = 0.0):
    """
    For each channel c: computes (loss_with_channel_zeroed - baseline_loss) averaged over batches.
    fill=0.0 is 'zero in normalized space', i.e., mean pixel in original space.
    Returns np.array of shape [C].
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    deltas = None
    batches = 0

    with torch.inference_mode():
        for b, (images, targets) in enumerate(dataloader):
            if b >= max_batches:
                break
            if isinstance(images, (list, tuple)):
                images = torch.stack(images, dim=0)
            if isinstance(targets, (list, tuple)):
                targets = torch.stack(targets, dim=0)

            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).long()
            B, C, H, W = images.shape

            # Baseline loss
            logits = model(images)
            base_loss = criterion(logits, targets).item()

            # Per-channel ablation
            delta_batch = []
            for c in range(C):
                x_mask = images.clone()
                x_mask[:, c, :, :] = fill
                logits_m = model(x_mask)
                loss_m = criterion(logits_m, targets).item()
                delta_batch.append(loss_m - base_loss)

            delta_batch = np.array(delta_batch)  # [C]
            deltas = delta_batch if deltas is None else deltas + delta_batch
            batches += 1

    if deltas is None or batches == 0:
        return None
    return deltas / batches

# ---- Convenience: run all three analyses and pretty-print ----
def evaluate_channel_importance(model, valid_loader, device, target_class=None, max_batches=10):
    w_imp = adapter_weight_importance(model)
    g_imp = channel_gradient_importance(model, valid_loader, device, target_class=target_class, max_batches=max_batches)
    d_imp = channel_ablation_delta_loss(model, valid_loader, device, max_batches=max_batches, fill=0.0)

    print("\n[CHANNEL IMPORTANCE]")
    if w_imp is not None:
        print("Adapter |W| L1 per input channel:")
        print(", ".join([f"ch{c}={w_imp[c]:.4f}" for c in sorted(w_imp.keys())]))
    else:
        print("Adapter weight importance: N/A (nonlinear adapter or no 1x1 Conv).")

    if g_imp is not None:
        print("Grad-based sensitivity (mean |dL/dx_c|) per channel:")
        print(", ".join([f"ch{i}={v:.3e}" for i, v in enumerate(g_imp)]))
    else:
        print("Grad-based sensitivity: N/A.")

    if d_imp is not None:
        print("Ablation delta loss (zeroing channel) per channel (higher = more important):")
        print(", ".join([f"ch{i}={v:.4f}" for i, v in enumerate(d_imp)]))
    else:
        print("Ablation delta loss: N/A.")

    return {"adapter_L1": w_imp, "grad_sensitivity": g_imp, "ablation_delta": d_imp}