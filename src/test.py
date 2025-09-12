import torch
import torch.nn as nn
from torchvision import models
from data_setup import create_dataloaders, create_multichannel_dataloaders
import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import timm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def clip_percentile(img, perc=98):
        """
        Clips the pixel values of an image based on the given percentile. Each
        channel is clipped independently!

        Parameters:
        img (numpy.ndarray): The input image.
        perc (float): The percentile value to clip the pixel values. Default is 98.

        Returns:
        numpy.ndarray: The clipped image.

        Raises:
        AssertionError: If the percentile value is not between 0 and 100.
        """
        assert perc<100
        assert perc>0
        if len(img.shape) != 3 or img.shape[0] != 3:
            print('img.shape =', img.shape)
            raise ValueError("Input image must have shape: (3, height, width)")
        if len(np.shape(img))==2:
            img = np.expand_dims(img, axis=0)
        lim = 100-perc
        img_clipped = np.zeros_like(img)
        for i in range(img.shape[0]):
            img_i   = img[i,:,:]
            img_i_f = img_i.flatten()
            p1 = np.percentile(img_i_f, 100-(perc+lim/2.))
            p2 = np.percentile(img_i_f,     (perc+lim/2.))
            clipped = np.clip(img_i, p1, p2)
            img_clipped[i,:,:] = clipped
        if np.shape(img_clipped)[0]==1:
            img_clipped = np.squeeze(img_clipped)
        assert img_clipped.shape[0]==3
        return img_clipped
    
 

class config:
    ### Set random seed
    SEED = 42
    ### Set version of the code
    VERSION = '0.0.1'
    print('[INFO] Version of the code:', VERSION)
    ### Set name of the model
    MODEL = 'zoobot_6ch_adapter' # 'vgg16', 'vgg19', 'efficientnet-b0', 'zoobot'
    MODEL_NAME = MODEL
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
    # MEAN = [0.332, 0.148, 0.050] # q1_lenses.csv
    # STD = [0.132, 0.204, 0.101] # q1_lenses.csv

    # MEAN = [0.207, 0.201, 0.504] # merged_train.csv 2nd try
    # STD = [0.151, 0.239, 0.300] # merged_train.csv 2nd try
    # MEAN = [0.287, 0.187, 0.078] # final_train.csv normal (beta=1)
    # STD = [0.135, 0.229, 0.122] # final_train.csv normal (beta=1)
    # MEAN = [0.182, 0.187, 0.507] # final_train.csv 2nd try
    # STD = [0.147, 0.229, 0.294] # final_train.csv 2nd try

    # MEAN = [0.011392, 0.017252, 0.025831, 0.501423, 0.507372, 0.187020] # wrong sequence of channels
    # STD  = [0.030090, 0.733123, 0.534643, 0.189828, 0.296535, 0.244794]
    
    # MEAN = [0.011392, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch pow
    # STD  = [0.030090, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
    
    MEAN = [0.182512, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch 2ndtry
    STD  = [0.188272, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
    
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
    SCHEDULER = 'StepLR'
    #SCHEDULER = 'OneCycleLR'
    if SCHEDULER=='StepLR':
        SCHEDULER_STEP  = 20
        SCHEDULER_GAMMA = 0.5
    ### If True, use the pretrained weights
    USE_PRETRAINED = True
    ### If True, use the learning rate scheduler during training
    USE_SCHEDULER = True
    NUM_WORKERS = 0 # os.cpu_count()//2
    


     
    
# # Set the number of worker processes for loading data
# config.NUM_WORKERS = os.cpu_count()//2
# Set random seed
utils.fix_all_seeds(config.SEED)



model_path = '/dati4/mfogliardi/training/ggsl/models/zoobot_6ch_2ndtry_freeze4_5e5_step_id3/zoobot_6ch_adapter.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config.MODEL == 'resnet50':
    # Load the ResNet50 model
    model = models.resnet50(weights=None)
    # Modify the last fully connected layer to have 3 output classes
    num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 3)  # 3 output classes
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),  # First reduce to 1024
        nn.ReLU(),
        nn.Dropout(0.3),        # Dropout to prevent overfitting
        nn.Linear(1024, 512),   # Then reduce to 512
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),    # Then to 256
        nn.ReLU(),
        nn.Linear(256, 128),    # Then to 128
        nn.ReLU(),
        nn.Linear(128, config.NUM_CLASSES)       # Final output layer (3 classes)
    )
    
elif config.MODEL == 'vgg16':
    
    model = models.vgg16(weights=None)
    
    # Modify the classifier to have 3 output classes
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs, 2048),  # First reduce to 2048
        nn.ReLU(),
        nn.Dropout(0.3),        # Dropout to prevent overfitting

        nn.Linear(2048, 1024),   # Then reduce to 1024
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(1024, 512),   # Then reduce to 512
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(512, 256),    # Then to 256
        nn.ReLU(),

        nn.Linear(256, 128),    # Then to 128
        nn.ReLU(),

        nn.Linear(128, config.NUM_CLASSES)       # Final output layer (3 classes)
    )
    
elif config.MODEL == 'vgg19':
        # Load the VGG16 model
        if config.USE_PRETRAINED:
            model = models.vgg19(weights="DEFAULT")
        else:
            model = models.vgg19(weights=None)
        
        # Modify the classifier to have 3 output classes
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # First reduce to 2048
            nn.ReLU(),
            nn.Dropout(0.3),        # Dropout to prevent overfitting

            nn.Linear(2048, 1024),   # Then reduce to 1024
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),   # Then reduce to 512
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),    # Then to 256
            nn.ReLU(),

            nn.Linear(256, 128),    # Then to 128
            nn.ReLU(),

            nn.Linear(128, config.NUM_CLASSES)       # Final output layer (3 classes)
        )
        
elif config.MODEL == 'efficientnet-b0':
        
        model = models.efficientnet_b0(weights=None)
        
        # Modify the classifier to have 3 output classes
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # First reduce to 2048
            nn.ReLU(),
            nn.Dropout(0.3),        # Dropout to prevent overfitting

            nn.Linear(2048, 1024),   # Then reduce to 1024
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),   # Then reduce to 512
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),    # Then to 256
            nn.ReLU(),

            nn.Linear(256, 128),    # Then to 128
            nn.ReLU(),

            nn.Linear(128, config.NUM_CLASSES)       # Final output layer (3 classes)
        )

elif config.MODEL == 'zoobot':
        # Load the Zoobot encoder with its classifier
        model = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_base', pretrained=True, num_classes=config.NUM_CLASSES)

elif config.MODEL == 'zoobot_6ch_adapter':
        
        class ZoobotWithAdapter(nn.Module):
            def __init__(self, num_in_ch: int, num_classes: int,
                         use_pretrained: bool = True,
                         init_mode: str = "identity_first3",
                         backbone_ckpt: str = None):
                super().__init__()
                # Load Zoobot ConvNeXt-base backbone via timm (3-ch)
                self.backbone = timm.create_model(
                    'hf_hub:mwalmsley/zoobot-encoder-convnext_base',
                    pretrained=use_pretrained,
                    num_classes=num_classes
                )
                # Expose/alias classifier head for optimizer param groups
                # timm ConvNeXt uses .head as the classifier; create .fc alias
                if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Module):
                    self.backbone.fc = self.backbone.head

                # 6->3 learnable adapter (linear per-pixel mixing)
                self.adapter = nn.Conv2d(num_in_ch, 3, kernel_size=1, bias=False)
                with torch.no_grad():
                    w = self.adapter.weight  # [3, num_in_ch, 1, 1]
                    w.zero_()
                    if init_mode == "identity_first3" and num_in_ch >= 3:
                        for c in range(3):
                            w[c, c, 0, 0] = 1.0
                    elif init_mode == "average_all":
                        w[:, :, 0, 0] = 1.0 / float(num_in_ch)
                    else:
                        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')

                # Optionally load your domain-finetuned 3-ch Zoobot checkpoint (backbone only)
                if backbone_ckpt is not None and os.path.isfile(backbone_ckpt):
                    sd = torch.load(backbone_ckpt, map_location="cpu")
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    # strip 'module.' if present
                    sd = {k.replace("module.", ""): v for k, v in sd.items()}
                    # drop classifier weights (timm convnext head.*)
                    sd = {k: v for k, v in sd.items() if not (k.startswith("head.") or k.startswith("fc."))}
                    missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
                    print(f"[INFO] Loaded Zoobot backbone from {backbone_ckpt} (missing: {len(missing)}, unexpected: {len(unexpected)})")

            def forward(self, x):
                x = self.adapter(x)  # [B,6,H,W] -> [B,3,H,W]
                return self.backbone(x)

        model = ZoobotWithAdapter(
            num_in_ch=6,
            num_classes=config.NUM_CLASSES,
            use_pretrained=getattr(config, "USE_PRETRAINED", True),
            init_mode=getattr(config, "ADAPTER_INIT", "identity_first3"),
            backbone_ckpt=getattr(config, "BACKBONE_CKPT", None)
        )

elif config.MODEL == 'zoobot_6ch_pretrained':
        # ConvNeXt-base (Zoobot encoder), inflate stem conv 3->6 and fine-tune
        model = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_base',
                                  pretrained=getattr(config, "USE_PRETRAINED", True),
                                  num_classes=config.NUM_CLASSES)

        # Helper to find the first conv of the stem for ConvNeXt variants
        def _get_stem_conv(m: nn.Module):
            # timm convnext has either .stem (Sequential[Conv2d, LayerNorm]) or .downsample_layers[0][0]
            if hasattr(m, "stem") and isinstance(m.stem, nn.Sequential) and len(m.stem) > 0 and isinstance(m.stem[0], nn.Conv2d):
                return ("stem", 0, m.stem[0])
            if hasattr(m, "downsample_layers"):  # some convnext impls
                dl0 = m.downsample_layers[0]
                if isinstance(dl0, nn.Sequential) and len(dl0) > 0 and isinstance(dl0[0], nn.Conv2d):
                    return ("downsample_layers.0", 0, dl0[0])
            return (None, None, None)

        parent_name, idx, old_conv = _get_stem_conv(model)
        if old_conv is None:
            raise RuntimeError("Could not locate Zoobot/ConvNeXt stem Conv2d to inflate.")

        new_in_ch = 6
        new_conv = nn.Conv2d(
            in_channels=new_in_ch,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            w = old_conv.weight.data  # [Cout, 3, k, k]
            if w.shape[1] != 3:
                raise RuntimeError(f"Unexpected stem conv in_ch={w.shape[1]} (expected 3).")
            w_mean = w.mean(dim=1, keepdim=True)  # [Cout,1,k,k]
            extra = w_mean.repeat(1, new_in_ch - 3, 1, 1)  # [Cout,3,k,k]
            new_w = torch.cat([w, extra], dim=1)  # [Cout,6,k,k]
            new_conv.weight.copy_(new_w)

        # Install the new conv into the model
        if parent_name == "stem":
            model.stem[0] = new_conv
        elif parent_name == "downsample_layers.0":
            model.downsample_layers[0][0] = new_conv

        # Expose classifier head as .fc for optimizer grouping
        if hasattr(model, "head") and isinstance(model.head, nn.Module):
            model.fc = model.head

        # Staged freezing (warm-up)
        model._staged_freeze = False
        model._frozen_modules = []

        freeze_stages = getattr(config, "FREEZE_BACKBONE_STAGES", 0)  # 0: none, 1: stem, 2: stem+stage0
        freeze_only_stage0 = getattr(config, "FREEZE_ONLY_LAYER1", False)  # recommended: True

        if freeze_only_stage0:
            # Keep stem trainable, freeze first stage blocks
            if hasattr(model, "stages") and len(model.stages) > 0:
                for p in model.stages[0].parameters():
                    p.requires_grad = False
                model._frozen_modules.append(model.stages[0])
                model._staged_freeze = True
            elif hasattr(model, "stages_0"):  # fallback if custom attr naming
                for p in model.stages_0.parameters():
                    p.requires_grad = False
                model._frozen_modules.append(model.stages_0)
                model._staged_freeze = True
        else:
            # Legacy numeric: not recommended to freeze stem for 6ch inflation
            if freeze_stages >= 1:
                # Freeze stem (conv + norm)
                if hasattr(model, "stem"):
                    for p in model.stem.parameters():
                        p.requires_grad = False
                    model._frozen_modules.append(model.stem)
                    model._staged_freeze = True
                elif hasattr(model, "downsample_layers"):
                    for p in model.downsample_layers[0].parameters():
                        p.requires_grad = False
                    model._frozen_modules.append(model.downsample_layers[0])
                    model._staged_freeze = True
            if freeze_stages >= 2:
                # Freeze first stage of blocks
                if hasattr(model, "stages") and len(model.stages) > 0:
                    for p in model.stages[0].parameters():
                        p.requires_grad = False
                    model._frozen_modules.append(model.stages[0])
                    model._staged_freeze = True
   

elif config.MODEL == 'resnet50_6ch':
        # Load ResNet50 without pretrained weights (since they're for 3 channels)
        model = models.resnet50(weights=None)
        
        # Modify the first convolutional layer for 6 input channels
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new conv layer weights
        # Use Xavier/He initialization for the new 6-channel input layer
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Modify the last fully connected layer to have config.NUM_CLASSES output classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # First reduce to 1024
            nn.ReLU(),
            nn.Dropout(0.3),            # Dropout to prevent overfitting

            nn.Linear(1024, 512),       # Then reduce to 512
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),        # Then to 256
            nn.ReLU(),

            nn.Linear(256, 128),        # Then to 128
            nn.ReLU(),

            nn.Linear(128, config.NUM_CLASSES)  # Final output layer
        )

elif config.MODEL == 'vgg16_6ch':
        # Load VGG16 without pretrained weights (since they're for 3 channels)
        model = models.vgg16(weights=None)
        
        # Modify the first convolutional layer for 6 input channels
        # Original VGG16 first layer: Conv2d(3, 64, kernel_size=3, padding=1)
        # New: Conv2d(6, 64, kernel_size=3, padding=1)
        
        # Get the first conv layer parameters (except input channels)
        first_conv = model.features[0]
        out_channels = first_conv.out_channels
        kernel_size = first_conv.kernel_size
        stride = first_conv.stride
        padding = first_conv.padding
        
        # Replace the first convolutional layer
        model.features[0] = nn.Conv2d(6, out_channels, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, bias=False)
        
        # Initialize the new conv layer weights
        nn.init.kaiming_normal_(model.features[0].weight, mode='fan_out', nonlinearity='relu')
        
        # Modify the classifier to have config.NUM_CLASSES output classes
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # First reduce to 2048
            nn.ReLU(),
            nn.Dropout(0.3),            # Dropout to prevent overfitting

            nn.Linear(2048, 1024),      # Then reduce to 1024
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),       # Then reduce to 512
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),        # Then to 256
            nn.ReLU(),

            nn.Linear(256, 128),        # Then to 128
            nn.ReLU(),

            nn.Linear(128, config.NUM_CLASSES)  # Final output layer
        )

elif config.MODEL == 'vgg19_6ch_adapter':
        
        class VGG19WithAdapter(nn.Module):
            def __init__(self, num_in_ch: int, num_classes: int,
                         use_pretrained: bool = True,
                         init_mode: str = "identity_first3",
                         backbone_ckpt: str = None):
                super().__init__()
                weights = "DEFAULT" if use_pretrained else None
                self.backbone = models.vgg19(weights=weights)

                # Replace the last classifier layer with your MLP head
                num_ftrs = self.backbone.classifier[6].in_features
                self.backbone.classifier[6] = nn.Sequential(
                    nn.Linear(num_ftrs, 2048), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, num_classes)
                )

                # 6->3 learnable adapter (linear per-pixel mixing)
                self.adapter = nn.Conv2d(num_in_ch, 3, kernel_size=1, bias=False)
                with torch.no_grad():
                    w = self.adapter.weight  # [3, num_in_ch, 1, 1]
                    w.zero_()
                    if init_mode == "identity_first3" and num_in_ch >= 3:
                        for c in range(3):
                            w[c, c, 0, 0] = 1.0
                    elif init_mode == "average_all":
                        w[:, :, 0, 0] = 1.0 / float(num_in_ch)
                    else:
                        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')

                # Optional: load a domain-finetuned VGG19 checkpoint (3-ch), ignore head
                if backbone_ckpt is not None and os.path.isfile(backbone_ckpt):
                    sd = torch.load(backbone_ckpt, map_location="cpu")
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    sd = {k.replace("module.", ""): v for k, v in sd.items()}
                    # Drop the last classifier block keys (we just replaced classifier[6])
                    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.6")}
                    missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
                    print(f"[INFO] Loaded VGG19 backbone from {backbone_ckpt} (missing: {len(missing)}, unexpected: {len(unexpected)})")

                # Expose classifier at top-level so your optimizer groups detect the 'head'
                self.classifier = self.backbone.classifier

            def forward(self, x):
                x = self.adapter(x)   # [B,6,H,W] -> [B,3,H,W]
                return self.backbone(x)

        model = VGG19WithAdapter(
            num_in_ch=6,
            num_classes=config.NUM_CLASSES,
            use_pretrained=getattr(config, "USE_PRETRAINED", True),
            init_mode=getattr(config, "ADAPTER_INIT", "identity_first3"),
            backbone_ckpt=getattr(config, "BACKBONE_CKPT", None)
        )
    

elif config.MODEL == 'resnet50_6ch_adapter':
        
        class ResNet50WithAdapter(nn.Module):
            def __init__(self, num_in_ch: int, num_classes: int, use_pretrained: bool, init_mode: str = "identity_first3", backbone_ckpt: str = None):
                super().__init__()
                base_weights = "DEFAULT" if use_pretrained else None
                self.backbone = models.resnet50(weights=base_weights)  # keep 3-ch conv1 intact

                # Replace classifier head
                num_ftrs = self.backbone.fc.in_features
                self.backbone.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, num_classes)
                )

                # 6->3 learnable adapter (linear mixing per-pixel)
                self.adapter = nn.Conv2d(num_in_ch, 3, kernel_size=1, bias=False)

                # Initialization:
                with torch.no_grad():
                    w = self.adapter.weight  # [3, num_in_ch, 1, 1]
                    w.zero_()
                    if init_mode == "identity_first3" and num_in_ch >= 3:
                        # Pass-through first 3 channels exactly; extras start at zero contribution
                        for c in range(3):
                            w[c, c, 0, 0] = 1.0
                    elif init_mode == "average_all":
                        # Start as grayscale average into each RGB channel
                        w[:, :, 0, 0] = 1.0 / float(num_in_ch)
                    else:
                        # Fallback: small random
                        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='linear')
                # Optionally load your domain-finetuned 3-ch checkpoint (backbone only)
                if backbone_ckpt is not None and os.path.isfile(backbone_ckpt):
                    sd = torch.load(backbone_ckpt, map_location="cpu")
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    # strip 'module.' if saved with DataParallel
                    sd = {k.replace("module.", ""): v for k, v in sd.items()}
                    # drop classifier weights
                    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
                    # load into the 3-ch backbone
                    missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
                    print(f"[INFO] Loaded backbone from {backbone_ckpt} (missing: {len(missing)}, unexpected: {len(unexpected)})")


            def forward(self, x):
                x = self.adapter(x)   # [B,6,H,W] -> [B,3,H,W]
                return self.backbone(x)


        model = ResNet50WithAdapter(
            num_in_ch=6,
            num_classes=config.NUM_CLASSES,
            use_pretrained=getattr(config, "USE_PRETRAINED", True),
            init_mode=getattr(config, "ADAPTER_INIT", "identity_first3"),
            backbone_ckpt=getattr(config, "BACKBONE_CKPT", None)
    )
   
elif config.MODEL == 'resnet50_6ch_pretrained':
        # Use pretrained weights if requested, then inflate conv1 to 6 channels
        base_weights = "DEFAULT" if getattr(config, "USE_PRETRAINED", False) else None
        model = models.resnet50(weights=base_weights)

        old_conv1 = model.conv1  # [64, 3, 7, 7] on torchvision ResNet50
        new_in_ch = 6
        new_conv1 = nn.Conv2d(
            in_channels=new_in_ch,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )

        if base_weights is not None:
            # Inflate 3->6 channels. Strategy: keep RGB as-is, fill extra channels with mean(RGB).
            with torch.no_grad():
                w = old_conv1.weight.data  # [64, 3, 7, 7]
                w_mean = w.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                extra = w_mean.repeat(1, new_in_ch - w.size(1), 1, 1)  # [64, 3, 7, 7]
                new_w = torch.cat([w, extra], dim=1)  # [64, 6, 7, 7]
                new_conv1.weight.copy_(new_w)
        else:
            # He init when training from scratch
            nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')

        model.conv1 = new_conv1

        # Track staged freeze for warm-up unfreeze later
        model._staged_freeze = False
        model._frozen_modules = []
        # Optionally freeze early stages to stabilize fine-tuning
        freeze_stages = getattr(config, "FREEZE_BACKBONE_STAGES", 1)  # 0: none, 1: stem, 2: +layer1
        freeze_only_layer1 = getattr(config, "FREEZE_ONLY_LAYER1", False)
        if freeze_only_layer1:
            for p in model.layer1.parameters():
                p.requires_grad = False
            model._frozen_modules.append(model.layer1)
            model._staged_freeze = True
        else:
            # legacy numeric policy
            if freeze_stages >= 1:
                for p in model.conv1.parameters():
                    p.requires_grad = False
                for p in model.bn1.parameters():
                    p.requires_grad = False
                model._frozen_modules += [model.conv1, model.bn1]
                model._staged_freeze = True
            if freeze_stages >= 2:
                for p in model.layer1.parameters():
                    p.requires_grad = False
                model._frozen_modules.append(model.layer1)
                model._staged_freeze = True

        # Classifier head
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, config.NUM_CLASSES)
        )
   
model.load_state_dict(torch.load(model_path))
model = model.to(device)
# Create the DataLoader for the test set
# _, data_loader_valid, data_loader_test = create_dataloaders(config=config)
_, data_loader_valid, data_loader_test = create_multichannel_dataloaders(config=config)
##############################################################################################    


def evaluate_model_and_save_results(model, data_loader_test, device, config, save_dir='/dati4/mfogliardi/training/ggsl/models/results/'):
    """
    Evaluate model performance and save mispredicted objects to PDF.
    
    Args:
        model: Trained PyTorch model
        data_loader_test: Test data loader
        device: Device to run evaluation on
        config: Configuration object
        save_dir: Directory to save results
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize evaluation variables
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    # Collect misclassified examples
    misclassified_examples = []
    
    # Initialize counters for class statistics
    class_totals = [0, 0]
    class_correct = [0, 0]
    prediction_counts = [0, 0]
    
    print("Starting model evaluation...")
    
    with torch.no_grad():
        for batch, (images, targets) in enumerate(data_loader_test):
            images = torch.stack(images)
            images = images.to(device)
            targets = torch.stack(targets)  
            targets = targets.to(device)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store all predictions and targets
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update class-specific counters and collect misclassified examples
            for i, (img, target, pred, prob) in enumerate(zip(images, targets, predicted, probabilities)):
                class_idx = target.item()
                pred_idx = pred.item()
                class_totals[class_idx] += 1
                prediction_counts[pred_idx] += 1
                
                if target == pred:
                    class_correct[class_idx] += 1
                else:
                    # Get the filename/identifier for this sample
                    # This depends on how your data loader is structured
                    sample_idx = batch * data_loader_test.batch_size + i
                    cutout_name = data_loader_test.dataset.get_filename(sample_idx)  # You'll need to implement this
                    
                    # Collect misclassified example
                    misclassified_examples.append({
                        'image': img.cpu(),
                        'true_class': target.item(),
                        'predicted_class': pred.item(),
                        'probabilities': prob.cpu().numpy(),
                        'batch': batch,
                        'index_in_batch': i,
                        'cutout_name': cutout_name
                    })
            
            if batch % 10 == 0:
                print(f"Processed batch {batch}/{len(data_loader_test)}")
    
    # Calculate metrics
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None, labels=[0, 1])
    
    # Print results
    print(f'\nOverall accuracy: {accuracy * 100:.2f}%')
    print(f'Total misclassified: {len(misclassified_examples)}')
    
    class_names = ["notGGSL", "GGSL"]
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f'Class {i} ({class_names[i]}): Precision: {p:.2f}, Recall: {r:.2f}, F1-score: {f:.2f}')
    
    print("\nClass-wise Performance:")
    for i in range(len(class_totals)):
        if class_totals[i] > 0:
            acc = class_correct[i]/class_totals[i]*100
            print(f"Class {i} ({class_names[i]}): {class_correct[i]}/{class_totals[i]} correct ({acc:.2f}%)")
    
    print("\nPrediction Distribution:")
    for i in range(len(prediction_counts)):
        print(f"Class {i} ({class_names[i]}): predicted {prediction_counts[i]} times")
    
    # Save misclassified examples to PDF
    if misclassified_examples:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = os.path.join(save_dir, f'misclassified_examples_{timestamp}.pdf')
        
        print(f"\nSaving {len(misclassified_examples)} misclassified examples to {pdf_filename}")
        
        with PdfPages(pdf_filename) as pdf:
            # Create one page per misclassified object with 1x3 layout for channels
            for idx, example in enumerate(misclassified_examples):
                img = example['image']
                true_class = example['true_class']
                pred_class = example['predicted_class']
                probs = example['probabilities']
                cutout_name = example['cutout_name']
                
                # Create figure with 1x3 subplots for the three channels
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'Object {idx+1}/{len(misclassified_examples)} - {cutout_name}\n' +
                           f'True: {class_names[true_class]} | Predicted: {class_names[pred_class]} | Confidence: {probs[pred_class]:.3f}', 
                           fontsize=14, weight='bold')
                
                channel_names = ['MTF', 'Asinh_Low', 'Asinh_High']
                
                for ch in range(3):
                    axes[ch].imshow(img[ch], cmap='viridis')
                    axes[ch].set_title(f'{channel_names[ch]}', fontsize=12)
                    axes[ch].axis('off')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Print progress every 50 objects
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(misclassified_examples)} misclassified objects")
        
        print(f"PDF saved successfully!")
    
    # Return evaluation metrics
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_correct': class_correct,
        'class_totals': class_totals,
        'prediction_counts': prediction_counts,
        'misclassified_count': len(misclassified_examples),
        'total_samples': total
    }
    
    return results


def evaluate_model_and_save_results_by_subcategory(
    model, data_loader_test, device, config, 
    save_dir='/dati4/mfogliardi/training/ggsl/models/results/'
):
    """
    Evaluate model performance and save mispredicted objects to PDF, grouped by subcategory.
    """
    import collections

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    # Subcategory definitions
    subcategories = {
        "sim_ggsl": "/dati4/mfogliardi/training/ggsl/dataset/ggsl/",
        "sim_notggsl": "/dati4/mfogliardi/training/ggsl/dataset/notggsl/",
        "q1lenses": "/dati4/mfogliardi/training/ggsl/dataset/q1lenses",
        "artefacts": "/dati4/mfogliardi/training/ggsl/artefacts/",
        "trickygalaxies": "/dati4/mfogliardi/training/ggsl/dataset/trickygirls/",
    }
    subcat_names = list(subcategories.keys())

    # Per-subcategory stats
    subcat_stats = {
        name: {
            "class_totals": [0, 0],
            "class_correct": [0, 0],
            "prediction_counts": [0, 0],
            "misclassified_examples": []
        } for name in subcat_names
    }

    print("Starting model evaluation by subcategory...")

    with torch.no_grad():
        for batch, (images, targets) in enumerate(data_loader_test):
            images = torch.stack(images).to(device)
            targets = torch.stack(targets).to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            for i, (img, target, pred, prob) in enumerate(zip(images, targets, predicted, probabilities)):
                sample_idx = batch * data_loader_test.batch_size + i
                cutout_name = data_loader_test.dataset.get_filename(sample_idx)
                # Identify subcategory
                subcat = None
                for name, path in subcategories.items():
                    if path in cutout_name:
                        subcat = name
                        break
                if subcat is None:
                    continue  # skip if not matched

                class_idx = target.item()
                pred_idx = pred.item()
                subcat_stats[subcat]["class_totals"][class_idx] += 1
                subcat_stats[subcat]["prediction_counts"][pred_idx] += 1

                if target == pred:
                    subcat_stats[subcat]["class_correct"][class_idx] += 1
                else:
                    subcat_stats[subcat]["misclassified_examples"].append({
                        'image': img.cpu(),
                        'true_class': class_idx,
                        'predicted_class': pred_idx,
                        'probabilities': prob.cpu().numpy(),
                        'batch': batch,
                        'index_in_batch': i,
                        'cutout_name': cutout_name
                    })

            if batch % 10 == 0:
                print(f"Processed batch {batch}/{len(data_loader_test)}")

    # Print and save results for each subcategory
    class_names = ["notGGSL", "GGSL"]
    for subcat in subcat_names:
        stats = subcat_stats[subcat]
        total_subcat = sum(stats["class_totals"])
        if total_subcat == 0:
            print(f"\nSubcategory {subcat}: No samples found.")
            continue

        correct_subcat = sum(stats["class_correct"])
        accuracy = correct_subcat / total_subcat if total_subcat > 0 else 0
        print(f"\nSubcategory: {subcat}")
        print(f"  Total samples: {total_subcat}")
        print(f"  Overall accuracy: {accuracy*100:.2f}%")
        print(f"  Total misclassified: {len(stats['misclassified_examples'])}")

        for i in range(2):
            if stats["class_totals"][i] > 0:
                acc = stats["class_correct"][i] / stats["class_totals"][i] * 100
                print(f"  Class {i} ({class_names[i]}): {stats['class_correct'][i]}/{stats['class_totals'][i]} correct ({acc:.2f}%)")
            else:
                print(f"  Class {i} ({class_names[i]}): 0 samples")

        print("  Prediction Distribution:")
        for i in range(2):
            print(f"    Class {i} ({class_names[i]}): predicted {stats['prediction_counts'][i]} times")

        # Save misclassified examples to PDF, ordered by true class
        if stats["misclassified_examples"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = os.path.join(save_dir, f'misclassified_{subcat}_{timestamp}.pdf')
            print(f"  Saving {len(stats['misclassified_examples'])} misclassified examples to {pdf_filename}")

            # Order by true class: first all class 0, then class 1
            ordered_examples = sorted(
                stats["misclassified_examples"], 
                key=lambda x: x['true_class']
            )

            with PdfPages(pdf_filename) as pdf:
                for idx, example in enumerate(ordered_examples):
                    img = example['image']
                    true_class = example['true_class']
                    pred_class = example['predicted_class']
                    probs = example['probabilities']
                    cutout_name = example['cutout_name']

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(
                        f'Object {idx+1}/{len(ordered_examples)} - {cutout_name}\n'
                        f'True: {class_names[true_class]} | Predicted: {class_names[pred_class]} | Confidence: {probs[pred_class]:.3f}',
                        fontsize=14, weight='bold'
                    )
                    channel_names = ['MTF', 'Asinh_Low', 'Asinh_High']
                    for ch in range(3):
                        axes[ch].imshow(img[ch], cmap='viridis')
                        axes[ch].set_title(f'{channel_names[ch]}', fontsize=12)
                        axes[ch].axis('off')
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    if (idx + 1) % 50 == 0:
                        print(f"    Processed {idx + 1}/{len(ordered_examples)} misclassified objects")
            print("  PDF saved successfully!")

    return subcat_stats

# Usage:
subcat_results = evaluate_model_and_save_results_by_subcategory(
    model=model,
    data_loader_test=data_loader_test,
    device=device,
    config=config,
    save_dir='/dati4/mfogliardi/training/ggsl/models/results/subdivided/q1test'
)
# # Usage - replace your existing evaluation code with this:
# results = evaluate_model_and_save_results(
#     model=model, 
#     data_loader_test=data_loader_test, 
#     device=device, 
#     config=config,
#     save_dir='/dati4/mfogliardi/training/ggsl/models/results/'
# 
# # RESULTS:
    ######################################################################
    # zoobot norm
    # Overall accuracy: 98.63%
    # Total misclassified: 18
    # Class 0 (notGGSL): Precision: 0.99, Recall: 0.99, F1-score: 0.99
    # Class 1 (GGSL): Precision: 0.98, Recall: 0.98, F1-score: 0.98

    # Class-wise Performance:
    # Class 0 (notGGSL): 802/810 correct (99.01%)
    # Class 1 (GGSL): 497/507 correct (98.03%)
    ######################################################################
    # VGG19 norm
    # Overall accuracy: 98.03%
    # Total misclassified: 26
    # Class 0 (notGGSL): Precision: 0.99, Recall: 0.98, F1-score: 0.98
    # Class 1 (GGSL): Precision: 0.97, Recall: 0.98, F1-score: 0.97

    # Class-wise Performance:
    # Class 0 (notGGSL): 796/810 correct (98.27%)
    # Class 1 (GGSL): 495/507 correct (97.63%)
    ######################################################################
    # resnet50
    # Overall accuracy: 97.49%
    # Total misclassified: 33
    # Class 0 (notGGSL): Precision: 0.99, Recall: 0.97, F1-score: 0.98
    # Class 1 (GGSL): Precision: 0.95, Recall: 0.99, F1-score: 0.97

    # Class-wise Performance:
    # Class 0 (notGGSL): 783/810 correct (96.67%)
    # Class 1 (GGSL): 501/507 correct (98.82%)
    ######################################################################
    # zoobot 2nd try
    # Overall accuracy: 98.03%
    # Total misclassified: 26
    # Class 0 (notGGSL): Precision: 0.98, Recall: 0.98, F1-score: 0.98
    # Class 1 (GGSL): Precision: 0.97, Recall: 0.97, F1-score: 0.97

    # Class-wise Performance:
    # Class 0 (notGGSL): 797/810 correct (98.40%)
    # Class 1 (GGSL): 494/507 correct (97.44%)
    ######################################################################
    # resnet 50 2nd try
    # Overall accuracy: 98.48%
    # Total misclassified: 20
    # Class-wise Performance:
    # Class 0 (notGGSL): 797/810 correct (98.40%)
    # Class 1 (GGSL): 500/507 correct (98.62%)

    # Prediction Distribution:
    # Class 0 (notGGSL): predicted 804 times
    # Class 1 (GGSL): predicted 513 times
    #######################################################################
    # zoobot 2nd try on final test
    # Overall accuracy: 97.59%
    # Total misclassified: 45
    # Class 0 (notGGSL): Precision: 0.98, Recall: 0.98, F1-score: 0.98
    # Class 1 (GGSL): Precision: 0.96, Recall: 0.96, F1-score: 0.96

    # Class-wise Performance:
    # Class 0 (notGGSL): 1283/1305 correct (98.31%)
    # Class 1 (GGSL): 543/566 correct (95.94%) # 55q1/60

    # Prediction Distribution:
    # Class 0 (notGGSL): predicted 1306 times
    # Class 1 (GGSL): predicted 565 times
    #######################################################################
    # resnet50 2nd try on final test
    # Overall accuracy: 98.24%
    # Total misclassified: 33
    # Class 0 (notGGSL): Precision: 0.98, Recall: 0.99, F1-score: 0.99
    # Class 1 (GGSL): Precision: 0.98, Recall: 0.96, F1-score: 0.97

    # Class-wise Performance:
    # Class 0 (notGGSL): 1296/1305 correct (99.31%)
    # Class 1 (GGSL): 542/566 correct (95.76%) # 54q1/60

    # Prediction Distribution:
    # Class 0 (notGGSL): predicted 1320 times
    # Class 1 (GGSL): predicted 551 times
    #######################################################################
    # vgg19 2nd try on final test
    # Overall accuracy: 97.97%
    # Total misclassified: 38
    # Class 0 (notGGSL): Precision: 0.99, Recall: 0.98, F1-score: 0.99
    # Class 1 (GGSL): Precision: 0.96, Recall: 0.97, F1-score: 0.97

    # Class-wise Performance:
    # Class 0 (notGGSL): 1285/1305 correct (98.47%)
    # Class 1 (GGSL): 548/566 correct (96.82%) # 54q1/60

    # Prediction Distribution:
    # Class 0 (notGGSL): predicted 1303 times
    # Class 1 (GGSL): predicted 568 times