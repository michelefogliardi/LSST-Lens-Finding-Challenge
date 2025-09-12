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
              is_training: bool = False):
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
    criterion = nn.CrossEntropyLoss()
    
    # modes
    if is_training:
        model.train()
       
    else:
        model.eval()
        
    # # Put model in TRAIN mode
    # model.train()
    # Initialize losses
    loss_tot = 0
    # Loop through DataLoader batches
    for batch, (images, targets) in enumerate(dataloader):
        
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

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler,
          config,
          writer: torch.utils.tensorboard.writer.SummaryWriter = None,
          checkpoint_file = None
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
        print(f"Epoch: {epoch+1}")
        print('###############################################################\n')
        
        #----------------------------------------------------------------------#
        # Training epoch
        #----------------------------------------------------------------------#
        train_losses = np.zeros(1) # 5 losses + 1 total
        train_losses = run_epoch(model=model,
                                 dataloader=train_dataloader,
                                 optimizer=optimizer,
                                 device=device,
                                 is_training=True)
        # if use_scheduler:
        #     # Update the learning rate according to the `lr_scheduler` chosen
        #     lr_scheduler.step()
        #     lr = lr_scheduler.get_last_lr()[0]
        sched = getattr(config, "SCHEDULER", "StepLR")
        if use_scheduler and lr_scheduler is not None:
            if sched == "StepLR":
                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
            elif sched == "ReduceLROnPlateau":
                # Step on validation metric later, after we compute valid_losses
                pass
        
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
                                         is_training=False)
        
        # If ReduceLROnPlateau, step now with the validation loss
        if use_scheduler and lr_scheduler is not None and sched == "ReduceLROnPlateau":
            val_metric = float(valid_losses[0]) if valid_dataloader is not None else float(train_losses[0])
            lr_scheduler.step(val_metric)
            # refresh lr after potential drop
            lr = optimizer.param_groups[0]["lr"]

        
        
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
            writer.close()

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
