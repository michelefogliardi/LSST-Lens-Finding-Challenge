"""
Trains a PyTorch Mask R-CNN model using device-agnostic code.
"""
import os



import time
import torch
import torch.nn as nn
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = "cuda" if torch.cuda.is_available() else "cpu"
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import data_setup, engine6, model, utils

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
    MODEL = 'zoobot_5ch_adapter'
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
    
    # MEAN = [0.284284, 0.310959, 0.331269] # g/r/i asinh 99%
    # STD  = [0.263660, 0.306905, 0.333308]
    
    # MEAN = [0.275266, 0.298942, 0.345367] # u_g/r/i_z asinh 99%
    # STD  = [0.259621, 0.298554, 0.349212]
    
    # MEAN = [0.011, 0.187, 0.507]
    # STD  = [0.030, 0.244, 0.296]
    
    MEAN = [0.161927, 0.158478, 0.194141, 0.189002, 0.228415]
    STD  = [0.242562, 0.237847, 0.261295, 0.260213, 0.285261]

    ### Set paths
    ROOT      = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/'
    TEST_DATA_CSV  = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/merged_test.csv'
    TRAIN_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/merged_train.csv'
    VALID_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/merged_valid.csv'


    DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/merged_train.csv'
    
    ### Set path to the code
    CODE_PATH = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/'
    ### Set number of classes (our dataset has only two: GGSL and notGGSL)
    NUM_CLASSES = 2
    ### Total number of epochs for the training
    NUM_EPOCHS = 60
    ### Set batch size
    BATCH_SIZE = 1500
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
    # SCHEDULER = 'OneCycleLR'
    SCHEDULER = 'ReduceLROnPlateau'
    if SCHEDULER=='StepLR':
        SCHEDULER_STEP  = 20
        SCHEDULER_GAMMA = 0.5
    if SCHEDULER == 'ReduceLROnPlateau':
        SCHED_FACTOR = 0.5
        SCHED_PATIENCE = 8
        MIN_LR = 1e-7
    ### If True, use the pretrained weights
    USE_PRETRAINED = True
    ADAPTER_INIT = 'average_all'
    BACKBONE_LR_MULT = 1.0
    HEAD_LR_MULT = 5.0
    ADAPTER_LR_MULT = 5.0
    WARMUP_FREEZE_EPOCHS = 4
    BACKBONE_CKPT = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/models/zoobot_riz_asinh99_rightnorm/zoobot.pt'
    # FREEZE_BACKBONE_STAGES = 0  # used by 'resnet50_6ch_pretrained'
    # BACKBONE_LR_MULT = 1.0 # used by 'resnet50_6ch_pretrained'
    # HEAD_LR_MULT = 5.0 # used by 'resnet50_6ch_pretrained'
    # STEM_LR_MULT = 5.0 # used by 'resnet50_6ch_pretrained'
    # FREEZE_ONLY_LAYER1 = True # used by 'resnet50_6ch_pretrained', it is STAGE0 for zoobot instead
    ### If True, use the learning rate scheduler during training
    USE_SCHEDULER = True
    NUM_WORKERS = 0 # os.cpu_count()//2
    USE_REGRESSION_TARGETS = True
    if USE_REGRESSION_TARGETS:
        # Use enriched, imputed CSVs for regression:
        TEST_DATA_CSV  = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_test_lens_wparams_vdisp_imputed.csv'
        TRAIN_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_train_lens_wparams_vdisp_imputed.csv'
        VALID_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_valid_lens_wparams_vdisp_imputed.csv'
    # Optional: replace (magnitude, PA) pairs for shear and source ellipticity with
    # spin-2 components (e1 = m cos 2θ, e2 = m sin 2θ). This keeps dimensionality the same
    # and avoids angle wrap/undefined PA at small magnitudes.
    USE_SPIN2_COMPONENTS = True

    

    
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
data_loader_test = data_setup.create_multichannel_dataloaders(config=config)



# data_loader_train, \
# data_loader_valid, \
# data_loader_test = data_setup.create_dataloaders_weighted(config=config)

################################################################################
# Instantiate the Mask R-CNN model
################################################################################

# Instantiate an instance of the model from the "model.py" script
model = model.get_cnn_model(config=config)
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


# model.load_state_dict(torch.load(f=config.SAVED_MODEL));


################################################################################
# Construct Optimizer and LR scheduler
################################################################################

def build_optimizer(model, config):
    """Create optimizer with param groups:
       - adapter (if present): higher LR
       - head (fc/classifier): higher LR
       - backbone: lower LR
    """
    base_lr = getattr(config, "LEARNING_RATE", 5e-5)
    wd = getattr(config, "WEIGHT_DECAY", 1e-3)
    betas = getattr(config, "BETAS", (0.9, 0.999))
    eps = getattr(config, "EPS", 1e-8)
    amsgrad = getattr(config, "AMSGRAD", False)

    # Multipliers allow easy tuning from config
    backbone_mult = getattr(config, "BACKBONE_LR_MULT", 1.0)
    head_mult = getattr(config, "HEAD_LR_MULT", 5.0)
    adapter_mult = getattr(config, "ADAPTER_LR_MULT", 5.0)
    stem_mult     = getattr(config, "STEM_LR_MULT", 5.0)

    param_groups = []

    def add_group(params, lr_mult, name):
        params = [p for p in params if p.requires_grad]
        if params:
            param_groups.append({
                "params": params,
                "lr": base_lr * lr_mult,
                "weight_decay": wd,
                "name": name
            })

    # Adapter (for resnet50_6ch_adapter)
    if hasattr(model, "adapter"):
        add_group(model.adapter.parameters(), adapter_mult, "adapter")

    # Head (resnet: fc; wrapper: backbone.fc; vgg: classifier)
   
    if hasattr(model, "fc"):
        add_group(model.fc.parameters(), head_mult, "head")
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
        add_group(model.backbone.fc.parameters(), head_mult, "head")
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        add_group(model.classifier.parameters(), head_mult, "head")
       
    # Stem (conv1/bn1): important for 6ch_pretrained, give higher LR
    stem_params = []
    if hasattr(model, "conv1"):
        stem_params += list(model.conv1.parameters())
    if hasattr(model, "bn1"):
        stem_params += list(model.bn1.parameters())
    if hasattr(model, "stem"):
        stem_params += list(model.stem.parameters())
    if hasattr(model, "downsample_layers"):
        stem_params += list(model.downsample_layers[0].parameters())
    add_group(stem_params, stem_mult, "stem")

    # Backbone = everything else that still requires grad
    # included = set()
    # for g in param_groups:
    #     for p in g["params"]:
    #         included.add(p)

    # backbone_params = [p for p in model.parameters() if p.requires_grad and p not in included]
    included_ids = set()
    for g in param_groups:
        for p in g["params"]:
            included_ids.add(id(p))
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in included_ids]
    add_group(backbone_params, backbone_mult, "backbone")

    opt_name = getattr(config, "OPTIMIZER", "AdamW").lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=getattr(config, "MOMENTUM", 0.9),
            weight_decay=wd,
            nesterov=True,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=wd, amsgrad=amsgrad
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=wd, amsgrad=amsgrad
        )

    # Optional: small log to verify
    try:
        for i, g in enumerate(optimizer.param_groups):
            name = g.get("name", f"group_{i}")
            lr = g["lr"]
            count = sum(p.numel() for p in g["params"])
            print(f"[OPT] {name}: lr={lr:.2e}, params={count}")
    except Exception:
        pass

    return optimizer

# Warm-up freeze (adapter variant): freeze backbone, train adapter+head
if getattr(config, "WARMUP_FREEZE_EPOCHS", 0) > 0 and hasattr(model, "adapter"):
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Keep head trainable: prefer fc if present, else classifier (VGG)
    if hasattr(model.backbone, "fc"):
        for p in model.backbone.fc.parameters():
            p.requires_grad = True
    elif hasattr(model.backbone, "classifier"):
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True
    for p in model.adapter.parameters():
        p.requires_grad = True
    # mark backbone as frozen so run_epoch can keep it in eval() during warm-up
    setattr(model, "_backbone_frozen", True)
    print("[INFO] Warm-up: backbone frozen, adapter+head trainable")

optimizer = build_optimizer(model, config)

# ### Select trainable parameters
# params = [p for p in model.parameters() if p.requires_grad]


# ### Construct optimizer
# if config.OPTIMIZER=='SGD':
#     # Stochastic Gradient Descent
#     optimizer = torch.optim.SGD(params=params,
#                                 lr=config.LEARNING_RATE,
#                                 momentum=config.MOMENTUM,
#                                 weight_decay=config.WEIGHT_DECAY)
# elif config.OPTIMIZER=='AdamW':
#     # AdamW optimizer (includes weight decay for regularization)
#     optimizer = torch.optim.AdamW(params=params,
#                                   lr=config.LEARNING_RATE,
#                                   betas=config.BETAS,
#                                   eps=config.EPS,
#                                   weight_decay=config.WEIGHT_DECAY,
#                                   amsgrad=config.AMSGRAD)           
# elif config.OPTIMIZER=='Adam':
#     # Adam optimizer
#     optimizer = torch.optim.Adam(params=params,
#                                  lr=config.LEARNING_RATE,
#                                  betas=config.BETAS,
#                                  eps=config.EPS,
#                                  weight_decay=config.WEIGHT_DECAY,
#                                  amsgrad=config.AMSGRAD)           
# else:
#     print('`config.OPTIMIZER` is not defined correctly! Exiting...')
#     exit(1)
                            
                            
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
results = engine6.train(model=model,
                       train_dataloader=data_loader_train,
                       valid_dataloader=data_loader_valid,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       config=config,
                       writer=writer,
                       checkpoint_file=checkpoint_file,
                       optimizer_builder=lambda m, cfg=config: build_optimizer(m, cfg))
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

utils.save_model_summary(model=model,
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