"""
Contains PyTorch model code to instantiate the Mask R-CNN model.
"""
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from torchsummary import summary
import utils, data_setup
import os

################################################################################

def get_model_instance_segmentation(config):
    

    # ResNet50 has a downsampling factor of 32, eg: IN:(3,224,224) --> OUT:(2048,7,7)
    # src: https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
    subsampling_resnet50 = 32.
    rescale_factor = round(config.WIDTH/subsampling_resnet50)
    min_size       = int(rescale_factor*subsampling_resnet50)
    max_size       = min_size

    if config.MODEL == 'resnet50':
        # Load the ResNet50 model
        if config.USE_PRETRAINED:
            model = models.resnet50(weights="DEFAULT")
        else:
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
        # Load the VGG16 model
        if config.USE_PRETRAINED:
            model = models.vgg16(weights="DEFAULT")
        else:
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
    
    elif config.MODEL == 'efficientnet-b0':
        # Load the EfficientNet-B0 model
        if config.USE_PRETRAINED:
            model = models.efficientnet_b0(weights="DEFAULT")
        else:
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
        
        # Print the encoder summary
        
        # summary(model, (3, 100, 100))  # Adjust the input size as needed
    elif config.MODEL == 'zoobot_euclid':
        # Load the Zoobot encoder with its classifier
        model = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-euclid-convnext-base', pretrained=True, num_classes=config.NUM_CLASSES)
        
        # Print the encoder summary
        
        # summary(model, (3, 100, 100))  # Adjust the input size as needed
    
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
        model._staged_freeze  = False
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
            num_in_ch=getattr(config, "INPUT_CHANNELS", 6),
            num_classes=config.NUM_CLASSES,
            use_pretrained=getattr(config, "USE_PRETRAINED", True),
            init_mode=getattr(config, "ADAPTER_INIT", "identity_first3"),
            backbone_ckpt=getattr(config, "BACKBONE_CKPT", None)
    )
        
    elif config.MODEL == 'vgg16_6ch':
        # Load VGG16 without pretrained weights (since they're for 3 channels)
        model = models.vgg16(weights=None)
        
        # Modify the first convolutional layer for 6 input channels
        # Original VGG16 first layer: Conv2d(3, 64, kernel_size=3, padding=1)
        # New: Conv2d(6, 64, kernel_size=3, padding=1)
        
        # Get the first conv layer parameters (except input channels)
        first_conv   = model.features[0]
        out_channels = first_conv.out_channels
        kernel_size  = first_conv.kernel_size
        stride       = first_conv.stride
        padding      = first_conv.padding
        
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

    
    else:
        raise ValueError('The chosen model is not defined!')

    return model

################################################################################


def get_cnn_model(config):
    model = get_model_instance_segmentation(config=config)
    
    return model

################################################################################


# if __name__ == "__main__":
    
#     class Config:
#         MODEL = 'zoobot'  # Choose from: resnet50, vgg16, efficientnet-b0, zoobot, resnet50_6ch, resnet50_6ch_pretrained, resnet50_6ch_adapter, vgg16_6ch
#         NUM_CLASSES = 3  # Example: number of output classes
#         USE_PRETRAINED = True
#         WIDTH = 100

#     model = get_cnn_model(config=Config)
#     print(model)
