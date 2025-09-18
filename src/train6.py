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
import data_setup, engine, model, utils

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
    MODEL = 'zoobot_euclid_6ch_adapter'
    MODEL_NAME = MODEL
    SAVED_MODEL = '/dati4/mfogliardi/training/ggsl/models/zoobot_6ch_2ndtry_freeze4_5e5_step_id3/zoobot_6ch_adapter.pt'
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
    # MEAN = [0.412, 0.200, 0.085] # train.csv
    # STD = [0.140, 0.236, 0.121] # train.csv
    # MEAN = [0.446, 0.201, 0.084] # merged_train.csv
    # STD = [0.134, 0.239, 0.126] # merged_train.csv
    # MEAN = [0.207, 0.201, 0.504] # merged_train.csv 2nd try
    # STD = [0.151, 0.239, 0.300] # merged_train.csv 2nd try
    
    # MEAN = [0.287, 0.187, 0.078] # final_train.csv normal (beta=1)
    # STD = [0.135, 0.229, 0.122] # final_train.csv normal (beta=1)
    # MEAN = [0.182, 0.187, 0.507] # final_train.csv 2nd try
    # STD = [0.147, 0.229, 0.294] # final_train.csv 2nd try
    
    # MEAN = [0.011392, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch pow
    # STD  = [0.030090, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
    
    # MEAN = [0.182512, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch 2ndtry
    # STD  = [0.188272, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
    
    MEAN = [0.182772, 0.187033, 0.507411, 0.172818, 0.025821, 0.017266] # 6ch 2ndtry zoom0
    STD  = [0.188229, 0.244853, 0.296725, 0.180817, 0.535088, 0.734407]
    
    # MEAN = [0.182513, 0.187020, 0.507372, 0.014899, 0.025455, 0.262853] # 6ch asinh fam
    # STD  = [0.188272, 0.244794, 0.296535, 0.093289, 0.094730, 0.047978]
    
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
    NUM_EPOCHS = 60
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
    ADAPTER_INIT = 'identity_first3'
    BACKBONE_LR_MULT = 1.0
    HEAD_LR_MULT = 5.0
    ADAPTER_LR_MULT = 5.0
    WARMUP_FREEZE_EPOCHS = 4
    BACKBONE_CKPT = '/dati4/mfogliardi/training/ggsl/models/zoobot_euclid_2nd_try_final_train_zoom0/zoobot_euclid.pt'
    # FREEZE_BACKBONE_STAGES = 0  # used by 'resnet50_6ch_pretrained'
    # BACKBONE_LR_MULT = 1.0 # used by 'resnet50_6ch_pretrained'
    # HEAD_LR_MULT = 5.0 # used by 'resnet50_6ch_pretrained'
    # STEM_LR_MULT = 5.0 # used by 'resnet50_6ch_pretrained'
    # FREEZE_ONLY_LAYER1 = True # used by 'resnet50_6ch_pretrained', it is STAGE0 for zoobot instead
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
data_loader_test = data_setup.create_multichannel_dataloaders(config=config)



# data_loader_train, \
# data_loader_valid, \
# data_loader_test = data_setup.create_dataloaders_weighted(config=config)

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
results = engine.train(model=model,
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