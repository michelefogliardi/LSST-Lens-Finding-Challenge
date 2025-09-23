
import torch
import torch.nn as nn
from torchvision import models
from data_setup import create_dataloaders, create_multichannel_dataloaders
import utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import timm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import aplpy
from astropy.io import fits
from PIL import Image

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
    MODEL = 'zoobot' # 'vgg16', 'vgg19', 'efficientnet-b0', 'zoobot'
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
    
    
    # MEAN = [0.284284, 0.310959, 0.331269] # g/r/i asinh 99%
    # STD  = [0.263660, 0.306905, 0.333308]
    MEAN = [0.161927, 0.158478, 0.194141] # g/r/i asinh 99% right norm
    STD  = [0.242562, 0.237847, 0.261295]
    
    # MEAN = [0.275266, 0.298942, 0.345367] # u_g/r/i_z asinh 99%
    # STD  = [0.259621, 0.298554, 0.349212]
    # MEAN = [0.162559, 0.158490, 0.189509] # u_g/r/i_z asinh 99% right norm
    # STD  = [0.244132, 0.237820, 0.258841]

    # MEAN = [0.161927, 0.158478, 0.194141, 0.189002, 0.228415] # rizgy asinh 99% right norm
    # STD  = [0.242562, 0.237847, 0.261295, 0.260213, 0.285261]
    
    
    
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



model_path = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/models/zoobot_riz_asinh99_rightnorm/zoobot.pt'

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

elif config.MODEL == 'zoobot_5ch_adapter':
        
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

                # 5->3 learnable adapter (linear per-pixel mixing)
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
                x = self.adapter(x)  # [B,5,H,W] -> [B,3,H,W]
                return self.backbone(x)

        model = ZoobotWithAdapter(
            num_in_ch=5,
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
_, data_loader_valid, data_loader_test = create_dataloaders(config=config)
# _, data_loader_valid, data_loader_test = create_multichannel_dataloaders(config=config)
##############################################################################################    


def evaluate_model_and_save_results(model, data_loader_test, device, config, 
                                    save_dir='/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/results/'):
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
    save_dir='/dati4/mfogliardi/training/ggsl/models/results/',
    model_name='model',
    csv_path=None
):
    """
    Evaluate model performance and save mispredicted objects to PDF, grouped by subcategory.
    """
    import collections

    # Resolve CSV path default
    if csv_path is None:
        csv_path = getattr(config, "TEST_DATA_CSV", 
                           "/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/merged_test_results.csv")

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    # Collect per-cutout probabilities for class=1 (lens)
    scores_paths = []
    scores_probs = []

    # Subcategory definitions
    subcategories = {
        "sim_ggsl": "/astrodata/lensing_challenge_lsst/data_slsim_lenses/",
        "sim_notggsl": "/astrodata/lensing_challenge_lsst/data_slsim_nonlenses/",
        "hsc_lenses": "/astrodata/lensing_challenge_lsst/data_hsc_lenses/",
        "hsc__nonlenses": "/astrodata/lensing_challenge_lsst/data_hsc_nonlenses/",
        
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

                # Record probability for class=1 (lens)
                try:
                    prob_lens = float(prob[1].item())
                except Exception:
                    # Fallback if tensor indexing differs
                    prob_lens = float(torch.nn.functional.softmax(prob.unsqueeze(0), dim=1)[0, 1].item())
                scores_paths.append(cutout_name)
                scores_probs.append(prob_lens)
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

                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    fig.suptitle(
                        f'Object {idx+1}/{len(ordered_examples)} - {cutout_name}\n'
                        f'True: {class_names[true_class]} | Predicted: {class_names[pred_class]} | Confidence: {probs[pred_class]:.3f}',
                        fontsize=14, weight='bold'
                    )
                    channel_names = ['R', 'I', 'Z', 'RGB']
                    for ch in range(3):
                        axes[ch].imshow(img[ch], cmap='viridis')
                        axes[ch].set_title(f'{channel_names[ch]}', fontsize=12)
                        axes[ch].axis('off')
                    

                    # img shape: (3, H, W), already stretched

                    # --- New: Create RGB using aplpy and FITS temp files ---
                    def make_rgb_from_arrays(path, out_png="temp_rgb.png"):
                        with fits.open(path) as hdul:
                            data_r = hdul[3].data.astype(float)
                            data_g = hdul[2].data.astype(float)
                            data_b = hdul[1].data.astype(float)
                            header_r = hdul[3].header
                            header_g = hdul[2].header
                            header_b = hdul[1].header

                            # Write each band to a temporary FITS file with its header
                            tmp_r = "tmp_r.fits"
                            tmp_g = "tmp_g.fits"
                            tmp_b = "tmp_b.fits"
                            fits.writeto(tmp_r, data_r, header=header_r, overwrite=True)
                            fits.writeto(tmp_g, data_g, header=header_g, overwrite=True)
                            fits.writeto(tmp_b, data_b, header=header_b, overwrite=True)
                            
                        aplpy.make_rgb_image([tmp_r, tmp_g, tmp_b], out_png,
                                             stretch_r='linear', stretch_g='linear', stretch_b='linear',
                                             pmin_r=0, pmax_r=99, pmin_g=0, pmax_g=99, pmin_b=0, pmax_b=99)
                        os.remove(tmp_r)
                        os.remove(tmp_g)
                        os.remove(tmp_b)
                        return out_png

                    try:
                        rgb_png = make_rgb_from_arrays(cutout_name, out_png=f"temp_rgb_{idx}.png")
                        rgb_img = np.array(Image.open(rgb_png))
                        axes[3].imshow(rgb_img)
                        axes[3].set_title(f'{channel_names[3]}', fontsize=12)
                        axes[3].axis('off')
                        
                    except Exception as e:
                        print(f"Warning: RGB creation failed for {cutout_name}: {e}")
                        # Try to show the image if it was created
                        if os.path.exists(f"temp_rgb_{idx}.png"):
                            rgb_img = np.array(Image.open(f"temp_rgb_{idx}.png"))
                            axes[3].imshow(rgb_img)
                            axes[3].set_title(f'{channel_names[3]} (error)', fontsize=12)
                            axes[3].axis('off')
                        else:
                            axes[3].set_title(f'{channel_names[3]} (failed)', fontsize=12)
                            axes[3].axis('off')
                    finally:
                        if os.path.exists(f"temp_rgb_{idx}.png"):
                            os.remove(f"temp_rgb_{idx}.png")
                                        
                    
                                        
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    if (idx + 1) % 50 == 0:
                        print(f"    Processed {idx + 1}/{len(ordered_examples)} misclassified objects")
            print("  PDF saved successfully!")

    # After evaluation, write class=1 probabilities to CSV
    try:
        import pandas as pd  # local import to avoid hard dependency at module import time
        df = pd.read_csv(csv_path)
        # Sanitize column name a bit
        colname = f"scores_{str(model_name).replace(' ', '_').replace('/', '_')}"

        # Prefer path-based mapping when possible
        wrote_by_path = False
        if "path" in df.columns:
            def _norm(p):
                try:
                    return os.path.abspath(os.path.normpath(str(p)))
                except Exception:
                    return str(p)
            df["_path_norm"] = df["path"].apply(_norm)
            mapping = { _norm(p): s for p, s in zip(scores_paths, scores_probs) }
            df[colname] = df["_path_norm"].map(mapping)
            missing = int(df[colname].isna().sum())
            if missing == 0:
                wrote_by_path = True
            # Clean helper
            df.drop(columns=["_path_norm"], inplace=True, errors="ignore")

        if not wrote_by_path:
            # Positional fallback if lengths match
            if len(df) == len(scores_probs):
                df[colname] = pd.Series(scores_probs, index=df.index)
            else:
                print(f"[ERROR] Cannot write scores: CSV rows {len(df)} != scores {len(scores_probs)}")
                return subcat_stats

        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved class=1 probabilities to '{csv_path}' in column '{colname}'")
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {csv_path}. Skipping score write.")
    except Exception as e:
        print(f"[ERROR] Failed to update CSV: {e}")

    return subcat_stats

# Usage:
subcat_results = evaluate_model_and_save_results_by_subcategory(
    model=model,
    data_loader_test=data_loader_test,
    device=device,
    config=config,
    save_dir='/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/results/zoobot_riz_asinh99_rightnorm',
    model_name='zoobot_riz_asinh99',
    csv_path='/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/merged_test_results.csv'
)
# # Usage - replace your existing evaluation code with this:
# results = evaluate_model_and_save_results(
#     model=model, 
#     data_loader_test=data_loader_test, 
#     device=device, 
#     config=config,
#     save_dir='/dati4/mfogliardi/training/ggsl/models/results/'
# )
