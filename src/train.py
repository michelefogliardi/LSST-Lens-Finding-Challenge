"""
Trains a PyTorch Mask R-CNN model using device-agnostic code.
"""
import os



import time
import torch
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = "cuda" if torch.cuda.is_available() else "cpu"
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import data_setup, engine3, model, utils

################################################################################
# Create config class where to store some info on the model
################################################################################

    
   
class config:
    ### Set random seed
    SEED = 42
    ### Set version of the code
    VERSION = '0.0.1'
    print('[INFO] Version of the code:', VERSION)
    ### Set name of the model
    MODEL = 'zoobot'
    MODEL_NAME = MODEL
    # SAVED_MODEL = '/dati4/mfogliardi/training/ggsl/models/0.0.1_2025-08-25_15-13-18/resnet50.pt'
    ### Set device
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('[INFO] Using device:', DEVICE)
    ### Survey
    TYPE='EUCLID'
    # Set height and width of the images
    HEIGHT = 100
    WIDTH  = 100
    ### Set normalization type
    V_NORMALIZE = 'v3'
    
    MEAN = [0.182512, 0.187020, 0.507372] # final_train.csv 2nd try
    STD = [0.188272, 0.244794, 0.296535] # final_train.csv 2nd try
    # MEAN = [0.011, 0.187, 0.507]
    # STD  = [0.030, 0.244, 0.296]

    ### Set paths
    ROOT      = '/dati4/mfogliardi/training/ggsl'
    TEST_DATA_CSV  = '/dati4/mfogliardi/training/ggsl/csv/final_test.csv'
    TRAIN_DATA_CSV = '/dati4/mfogliardi/training/ggsl/csv/final_train.csv'
    VALID_DATA_CSV = '/dati4/mfogliardi/training/ggsl/csv/final_val.csv'
    DATA_CSV  = '/dati4/mfogliardi/training/ggsl/csv/art_test.csv'
    ### Set path to the code
    CODE_PATH = '/dati4/mfogliardi/training/ggsl/lo_zibaldone/'
    ### Set number of classes (our dataset has only two: GGSL and notGGSL)
    NUM_CLASSES = 2
    ### Total number of epochs for the training
    NUM_EPOCHS = 50
    ### Set batch size
    BATCH_SIZE = 200
    ### Optimizer
    #OPTIMIZER = 'SGD'
    OPTIMIZER = 'AdamW'
    #OPTIMIZER = 'Adam'
    if OPTIMIZER=='SGD':
        # Set parameters of the optimizer (SGD)
        LEARNING_RATE = 0.001
        MOMENTUM      = 0.9
        WEIGHT_DECAY  = 0.0005
    elif OPTIMIZER=='AdamW':
        # Set parameters of the optimizer (AdamW)
        LEARNING_RATE = 0.00005
        BETAS         = (0.9, 0.999)
        EPS           = 1e-08
        WEIGHT_DECAY  = 1e-03
        AMSGRAD       = False
    elif OPTIMIZER=='Adam':
        # Set parameters of the optimizer (Adam)
        LEARNING_RATE = 0.00005 
        BETAS         = (0.9, 0.999)
        EPS           = 1e-08
        WEIGHT_DECAY  = 0 
        AMSGRAD       = False 
    ### Scheduler
    # SCHEDULER = 'StepLR'
    #SCHEDULER = 'OneCycleLR'
    SCHEDULER = 'ReduceLROnPlateau'
    if SCHEDULER=='StepLR':
        SCHEDULER_STEP  = 10
        SCHEDULER_GAMMA = 0.5
    if SCHEDULER == 'ReduceLROnPlateau':
        SCHED_FACTOR = 0.5
        SCHED_PATIENCE = 8
        MIN_LR = 1e-7
    ### If True, use the pretrained weights
    USE_PRETRAINED = True
    
    ### If True, use the learning rate scheduler during training
    USE_SCHEDULER = True
    NUM_WORKERS = 0 # os.cpu_count()//2
    

    
# # Set the number of worker processes for loading data
# config.NUM_WORKERS = os.cpu_count()//2
# Set random seed
utils.fix_all_seeds(config.SEED)


################################################################################
# Create base folder, dataset folder, and download dataset
################################################################################

### Check paths
# Path to script
code_path = config.CODE_PATH
# Path to image folder
image_path = config.ROOT
# Check if the folder exists
if Path(image_path).is_dir():
    print("[INFO] Dataset ready in the folder:", image_path)

################################################################################
# Create folder where to store the model and all the info on the training
################################################################################

checkpoint_file, timestamp = utils.create_model_dir(config=config)

# ################################################################################
# # Create train, valid, and test (DataSet and) DataLoaders
# ################################################################################

data_loader_train, \
data_loader_valid, \
data_loader_test = data_setup.create_dataloaders(config=config)


################################################################################
# Instantiate the Mask R-CNN model
################################################################################

# Instantiate an instance of the model from the "model.py" script
model = model.get_mask_rcnn_model(config=config)
# Add attributes to the model for the device and model name
model.device = config.DEVICE
model.name   = config.MODEL_NAME
# Add attributes to the model for the device and model name using a dictionary

# # Move model to the right device
model.to(config.DEVICE)
# Move model to the right device
# device = torch.device(config.DEVICE)
# model.to(device)


# summary(model, (3, 100, 100))


# model.load_state_dict(torch.load(f=config.SAVED_MODEL))


################################################################################
# Construct Optimizer and LR scheduler
################################################################################

## Select trainable parameters
params = [p for p in model.parameters() if p.requires_grad]


### Construct optimizer
if config.OPTIMIZER=='SGD':
    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(params=params,
                                lr=config.LEARNING_RATE,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
elif config.OPTIMIZER=='AdamW':
    # AdamW optimizer (includes weight decay for regularization)
    optimizer = torch.optim.AdamW(params=params,
                                  lr=config.LEARNING_RATE,
                                  betas=config.BETAS,
                                  eps=config.EPS,
                                  weight_decay=config.WEIGHT_DECAY,
                                  amsgrad=config.AMSGRAD)           
elif config.OPTIMIZER=='Adam':
    # Adam optimizer
    optimizer = torch.optim.Adam(params=params,
                                 lr=config.LEARNING_RATE,
                                 betas=config.BETAS,
                                 eps=config.EPS,
                                 weight_decay=config.WEIGHT_DECAY,
                                 amsgrad=config.AMSGRAD)           
else:
    print('`config.OPTIMIZER` is not defined correctly! Exiting...')
    exit(1)
                            
                            
### Construct learning rate scheduler
if config.SCHEDULER=='StepLR':
    # Decrease the learning rate by a factor SCHEDULER_GAMMA every SCHEDULER_STEP epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   step_size=config.SCHEDULER_STEP,
                                                   gamma=config.SCHEDULER_GAMMA,
                                                   verbose=False)
elif config.SCHEDULER=='OneCycleLR':
    # Sets the learning rate of each parameter group according to the 1-cycle learning rate policy
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                       max_lr=config.LEARNING_RATE,
                                                       total_steps=config.NUM_EPOCHS*len(data_loader_train),
                                                       verbose=False)
elif config.SCHEDULER=='ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=getattr(config, "SCHED_FACTOR", 0.5),
        patience=getattr(config, "SCHED_PATIENCE", 5),
        threshold=1e-4,
        cooldown=getattr(config, "SCHED_COOLDOWN", 0),
        min_lr=getattr(config, "MIN_LR", 1e-7),
        verbose=True
    )
else:
    print('`config.SCHEDULER` is not defined correctly! Exiting...')
    exit(1)


################################################################################
# Train the Net
################################################################################

print('\n--- START TRAINING ---')
writer = SummaryWriter(log_dir=Path(checkpoint_file.parent))
time_start = time.time()
results = engine3.train(model=model,
                       train_dataloader=data_loader_train,
                       valid_dataloader=data_loader_valid,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       config=config,
                       writer=writer,
                       checkpoint_file=checkpoint_file,
                       )
train_time = time.time() - time_start
print('--- END TRAINING ---\n')
print('\n[INFO] Training completed in {:.1f} seconds.'.format(train_time))

################################################################################
# Export and some extra information
################################################################################

utils.save_parameters(model=model,
                      file_name=Path(checkpoint_file.parent/'show_parameters.txt'))

df_results = utils.save_df_results_csv(results=results,
                                       file_name=Path(checkpoint_file.parent/'df_results.csv'))

utils.save_fig_losses(df_results=df_results,
                      file_name=Path(checkpoint_file.parent/'fig_losses.pdf'))

utils.save_config(config=config,
                  file_name=Path(checkpoint_file.parent/'config.txt'))

utils.save_model_summary3(model=model,
                         config=config,
                         file_name=Path(checkpoint_file.parent/'model_summary.txt'))

# ################################################################################
# # If the device is a GPU, empty the cache
# ################################################################################

# if config.DEVICE.type != 'cpu':
#     torch.cuda.empty_cache()

# ################################################################################
# # END
# ################################################################################
