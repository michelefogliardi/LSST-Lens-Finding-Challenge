import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip

import astropy.visualization as astrovi
from astropy.visualization import make_lupton_rgb, AsinhStretch
from astropy.coordinates import SkyCoord
import astropy.units as u

import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler
import sys
import ast
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import pyregion
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import exposure
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, sobel
from collections import Counter
# Add these imports at the top
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from scipy.ndimage import gaussian_filter1d
import warnings
# Silence NumPy MaskedArray partition warning globally
warnings.filterwarnings(
    "ignore",
    message=r".*'partition' will ignore the 'mask'.*",
    category=UserWarning,
    module=r"numpy\.core\.fromnumeric"
)
import utils
from io import BytesIO

class config:
    ### Set random seed
    SEED = 42
    ### Set version of the code
    VERSION = '0.0.1'
    print('[INFO] Version of the code:', VERSION)
    ### Set name of the model
    MODEL = 'resnet50'
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
    
    # MEAN = [0.287, 0.187, 0.078] # final_train.csv normal (beta=1)
    # STD = [0.135, 0.229, 0.122] # final_train.csv normal (beta=1)
    
    # MEAN = [0.182, 0.187, 0.507] # final_train.csv 2nd try
    # STD = [0.147, 0.229, 0.294] # final_train.csv 2nd try
    
    # MEAN = [0.011392, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch pow
    # STD  = [0.030090, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
    
    MEAN = [0.182512, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch 2ndtry
    STD  = [0.188272, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
    
    # MEAN = [0.163443, 0.148762, 0.516259, 0.128465, 0.009395, 0.004910] # shared test set 
    # STD = [0.153274, 0.221103, 0.309718, 0.148990, 0.381343, 0.589721]
    
    # MEAN = [0.316990, 0.216337, 0.501836, 0.451666, 0.612710, 0.843656] # ERO
    # STD  = [0.213938, 0.285839, 0.289018, 0.420396, 74.925231, 74.732706]
    
    # MEAN = [0.339397, 0.236873, 0.505158, 0.107595, 0.002220, -0.014500] # 0416 scale
    # STD  = [0.212314, 0.292032, 0.329197, 0.125507, 0.021190, 0.583026]
    
    # MEAN = [0.182513, 0.187020, 0.507372, 0.014899, 0.025455, 0.262853] # 6ch asinh fam
    # STD  = [0.188272, 0.244794, 0.296535, 0.093289, 0.094730, 0.047978]
    
    # MEAN = [0.011392, 0.187020, 0.507372] # 3ch con power stretch al posto di mtf del 2nd try
    # STD  = [0.030090, 0.244794, 0.296535]
    
    ### Set paths
    ROOT      = '/dati4/mfogliardi/training/ggsl'
    TEST_DATA_CSV  = '/dati4/mfogliardi/training/ggsl/csv/merged_test_q1.csv'
    TRAIN_DATA_CSV = '/dati4/mfogliardi/training/ggsl/csv/train.csv'
    VALID_DATA_CSV = '/dati4/mfogliardi/training/ggsl/csv/val.csv'
    # DATA_CSV  = '/dati4/mfogliardi/training/ggsl/csv/final_train.csv'
    # DATA_CSV = '/dati4/mfogliardi/training/ggsl/dataset/M0416/M0416_10arcmin.csv' #0416
    # DATA_CSV = '/dati4/mfogliardi/training/ggsl/dataset/shared_test_set/cutouts_paths_id_with_idstr.csv' # shared test set
    # DATA_CSV = '/dati4/mfogliardi/training/ggsl/dataset/shared_test_set/discovery_paths_id.csv' # discovery engine set
    
    DATA_CSV = '/dati4/mfogliardi/training/ggsl/dataset/shared_test_set/testAB/zoobot_6ch_adapter_lens_candidates_Q1_lenses.csv'
    LENS_CSV  = '/dati4/mfogliardi/training/ggsl/csv/q1_lenses.csv'
    ### Set path to the code
    CODE_PATH = '/dati4/mfogliardi/training/ggsl/lo_zibaldone/'
    ### Set number of classes (our dataset has only two: GGSL and notGGSL)
    NUM_CLASSES = 2
    ### Total number of epochs for the training
    NUM_EPOCHS = 50
    ### Set batch size
    BATCH_SIZE = 50
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


path_to_ABELL2390 = '/astrodata/Euclid/ERO/A2390/data' 
# ---- SAAMIE SHARED TEST SET -------
# All_strong_lens_discovery_engine_cutouts_reserved_test_set_true_no_train.csv 
#  29639 XX
#     216 X
#     118 C
#      13 B
#       6 A
# ---- SAAMIE SHARED TEST SET -------

# # # Set the number of worker processes for loading data
# # config.NUM_WORKERS = os.cpu_count()//2
# # Set random seed
# utils.fix_all_seeds(config.SEED)

def SigmaClipping(data,
                  n=3,
                  n_iter=1,
                  mu = None,
                  sigma = None):
    """
    Perform the 'sigma clipping' algorithm with 'n_iter' iterations, keeping
    a number 'n' of std.
    """
    x = np.array( data.flatten() )
    for i in range(n_iter):
        # x     = x[ x>(mu-n*sigma) ]
        # x     = x[ x<(mu+n*sigma) ]
        # Calculate lower and upper bounds
        lower_bound = mu - n * sigma
        upper_bound = mu + n * sigma
        
        # Replace values below the lower bound with the lower bound
        x[x < lower_bound] = lower_bound
        
        # Replace values above the upper bound with the upper bound
        x[x > upper_bound] = upper_bound
    # reshape the image x to its original size
    x = x.reshape(data.shape)   
    return x

class GGSL_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, config, csv_path=None, transforms=None):
        self.root             = config.ROOT
        self.img_csv          = csv_path if csv_path is not None else config.DATA_CSV
        self.df_cutouts       = pd.read_csv(self.img_csv)
        self.transforms       = transforms
        self.v_normalize      = config.V_NORMALIZE
        self.filenames        = self.df_cutouts.iloc[:,0].tolist()  # Assuming first column is filenames
        self.mean_value      = config.MEAN
        self.std_value       = config.STD
        
        # Initialize pipeline components
        self.data_loader_factory = DataLoaderFactory()
        self.stretch_pipeline = StretchPipeline(self)

    #--------------------------------------------------------------------------#
    
    def clip_percentile(self, data, lower_percentile=0, upper_percentile=98.0):
        """
        Clip the data to the specified percentiles.
        
        Parameters:
        - data: numpy array of data to be clipped
        - lower_percentile: lower percentile for clipping
        - upper_percentile: upper percentile for clipping
        
        Returns:
        - Clipped data
        """
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        return np.clip(data, lower_bound, upper_bound)

    def asinh_stretch_ds9(self, data, percent=99.9):
        """Apply asinh stretch similar to DS9 with percent clipping"""
        
        # assert nan values are not present
        if np.isnan(data).any():
            # print('nan values are present')
            # if nan values are present, replace them with 0
            data = np.nan_to_num(data)
        
        # Calculate the value at the specified percentile
        vmax = np.nanpercentile(data, percent)
       
        # Avoid division by zero
        if vmax == 0 or np.isnan(vmax):
            vmax = 0.001  # Use a safe default
        # Scale data 
        scaled_data = data / vmax
        
        # Apply asinh transformation similar to DS9
        beta = 0.1  # DS9-like scale factor - may need tuning
        transformed = np.arcsinh(scaled_data / beta) / np.arcsinh(1.0 / beta)
        
        # Clip to [0, 1] range
        return np.clip(transformed, 0,transformed.max())

    def mtf_arcsinh_scaling(self, image, beta=5, percentile_min=0, percentile_max=99):
        """
        Apply a non-linear MTF-like scaling to enhance contrast in astronomical images.

        Parameters:
        - image: 2D NumPy array (float or uint16)
        - beta: float, controls stretch strength (larger = less aggressive)
        - percentile_min, percentile_max: percentiles for clipping low/high extremes

        Returns:
        - scaled image in [0, 1], dtype float32
        """
        # image = image.astype(np.float32)
        # assert nan values are not present
        if np.isnan(image).any():
            # print('nan values are present')
            # if nan values are present, replace them with 0
            image = np.nan_to_num(image)

        # Robust range estimation using percentiles
        vmin = np.percentile(image, percentile_min)
        vmax = np.percentile(image, percentile_max)

        # Prevent division by zero
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        # Normalize
        scaled = (image - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0, 1)

        # Apply arcsinh stretch
        stretched = np.arcsinh(beta * scaled) / np.arcsinh(beta)
        
        return stretched #.astype(np.float32)

    #--------------------------------------------------------------------------#
    
    def gradient_enhanced_stretch(self, data, alpha=0.3):
        """
        Enhance edges and structures using gradient information.
        Better preserves separation between lens and source.
        """
        
        # Clean data
        data = np.nan_to_num(data)
        
        # Calculate gradient magnitude
        grad_x = sobel(data, axis=0)
        grad_y = sobel(data, axis=1)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        grad_norm = gradient_mag / (gradient_mag.max() + 1e-8)
        
        # Enhance original with gradient information
        enhanced = data + alpha * grad_norm * data
        
        # Apply mild stretch
        vmin, vmax = np.percentile(enhanced, [1, 99])
        stretched = np.clip((enhanced - vmin) / (vmax - vmin), 0, 1)
        
        return stretched
    
    def multiscale_balanced_stretch(self, data, scales=[1, 3, 5]):
        """
        Multi-scale enhancement that preserves both lens and source features.
        """
        
        data = np.nan_to_num(data)
        
        # Create multi-scale representation
        multiscale = np.zeros_like(data)
        
        for scale in scales:
            # Gaussian at different scales
            smooth = gaussian_filter(data, sigma=scale)
            # High-pass component
            detail = data - smooth
            # Add weighted detail
            weight = 1.0 / scale
            multiscale += weight * detail
        
        # Add back the base image
        result = data + 0.3 * multiscale
        
        # Gentle stretch
        vmin, vmax = np.percentile(result, [1, 99])
        stretched = np.clip((result - vmin) / (vmax - vmin), 0, 1)
        
        return stretched
    
    def lens_optimized_stretch(self, data, beta=0.5, lens_percentile=99):
        """
        Conservative stretch optimized for lens detection.
        Preserves central/lens features while suppressing background.
        """
        data = np.nan_to_num(data)
        
        # Conservative clipping focused on lens features
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, lens_percentile)  # Lower than usual
        
        # Prevent division by zero
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        
        # Normalize
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        
        # Very gentle arcsinh stretch
        stretched = np.arcsinh(beta * normalized) / np.arcsinh(beta)
        
        return stretched
    
    def adaptive_histogram_stretch(self, data, clip_limit=0.02):
        """
        Local contrast enhancement that adapts to image content.
        """
        
        data = np.nan_to_num(data)
        
        # Convert to uint16 for CLAHE
        data_uint = (data * 65535).astype(np.uint16)
        
        # Apply CLAHE
        enhanced = exposure.equalize_adapthist(
            data_uint, 
            clip_limit=clip_limit,
            nbins=256
        )
        # clip the enhaced image to 99 percentile
        # enhanced = self.clip_percentile(enhanced, lower_percentile=0, upper_percentile=99.9)
        return enhanced.astype(np.float32)
    
    
    def optimized_adaptive_stretch(self, data, clip_limits=[0.01, 0.05, 0.1]):
        """
        Optimized multi-scale adaptive histogram stretch.
        """
        
        data = np.nan_to_num(data)
        min_data = np.min(data)
        max_data = np.max(data)
        
        # data = (data - min_data) / (max_data - min_data)
        
        data_uint = (data * 65535).astype(np.uint16)

        

        
        
        enhanced_versions = []
        for clip_limit in clip_limits:
            enhanced = exposure.equalize_adapthist(
                data_uint, 
                clip_limit=clip_limit,
                nbins=512,
                kernel_size=None  # Let it choose optimal tile size
            )
            enhanced_versions.append(enhanced)
        
        # Weighted average (emphasize conservative versions)
        weights = [0.3, 0.5, 0.2]  # More weight on moderate enhancement
        weighted_result = np.average(enhanced_versions, axis=0, weights=weights)
        
        return weighted_result.astype(np.float32)
    
    def improved_gradient_enhanced_stretch(self, data, alpha=0.3, sigma_smooth=0.5, edge_threshold=0.1):
        """
        Improved gradient enhancement with noise reduction and adaptive parameters.
        """
        
        data = np.nan_to_num(data)
        
        # Pre-smooth to reduce noise in gradient calculation
        if sigma_smooth > 0:
            smoothed = gaussian_filter(data, sigma=sigma_smooth)
        else:
            smoothed = data
        
        # Calculate gradient magnitude with noise reduction
        grad_x = sobel(smoothed, axis=0)
        grad_y = sobel(smoothed, axis=1)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Adaptive normalization based on edge strength
        grad_norm = gradient_mag / (np.percentile(gradient_mag, 95) + 1e-8)
        grad_norm = np.clip(grad_norm, 0, 1)
        
        # Apply threshold to focus on significant edges
        edge_mask = gradient_mag > np.percentile(gradient_mag, edge_threshold * 100)
        
        # Adaptive alpha based on local image properties
        local_std = gaussian_filter(data**2, sigma=3) - gaussian_filter(data, sigma=3)**2
        adaptive_alpha = alpha * (1 + 0.5 * local_std / (local_std.max() + 1e-8))
        
        # Enhanced with adaptive weighting
        enhanced = data + adaptive_alpha * grad_norm * data * edge_mask
        
        # More robust stretch
        vmin, vmax = np.percentile(enhanced, [0, 99])
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        stretched = np.clip((enhanced - vmin) / (vmax - vmin), 0, 1)
        
        return stretched

    def improved_optimized_adaptive_stretch(self, data, clip_limits=[0.01, 0.03, 0.08], 
                                       adaptive_weights=True):
        """
        Improved adaptive stretch with dynamic weighting.
        """
        
        data = np.nan_to_num(data)
        data_uint = (data * 65535).astype(np.uint16)
        
        enhanced_versions = []
        quality_scores = []
        
        for clip_limit in clip_limits:
            enhanced = exposure.equalize_adapthist(
                data_uint, 
                clip_limit=clip_limit,
                nbins=256,
                kernel_size=None
            )
            enhanced_versions.append(enhanced)
            
            # Calculate quality score based on contrast and noise
            if adaptive_weights:
                contrast = enhanced.std()
                # Estimate noise (high-frequency content)
                noise_estimate = np.abs(enhanced - gaussian_filter(enhanced, 1)).mean()
                quality = contrast / (noise_estimate + 1e-8)
                quality_scores.append(quality)
        
        if adaptive_weights and len(quality_scores) > 0:
            # Normalize quality scores to get weights
            quality_scores = np.array(quality_scores)
            weights = quality_scores / quality_scores.sum()
        else:
            # Default weights favoring moderate enhancement
            weights = np.array([0.2, 0.6, 0.2])
        
        weighted_result = np.average(enhanced_versions, axis=0, weights=weights)
        
        return weighted_result.astype(np.float32)
    
    #--------------------------------------------------------------------------#
    
    def analyze_nan_images(self, max_samples=10):
        """Analyze images that contain NaN values."""
        nan_images = []
        
        for idx in range(min(len(self.df_cutouts), max_samples)):
            img_path = os.path.join(self.root, self.df_cutouts.iloc[idx,0])
            
            try:
                if 'artefacts' in img_path:
                    hdul = fits.open(img_path)
                    vis = hdul[0].data[0]
                    hdul.close()
                else:
                    vis = np.load(img_path)
                    vis = np.squeeze(vis)
                
                if np.isnan(vis).any():
                    nan_count = np.isnan(vis).sum()
                    total_pixels = vis.size
                    nan_percentage = (nan_count / total_pixels) * 100
                    
                    print(f'[NaN ANALYSIS] Image {idx}: {img_path}')
                    print(f'  - Shape: {vis.shape}')
                    print(f'  - NaN count: {nan_count}/{total_pixels} ({nan_percentage:.2f}%)')
                    print(f'  - Min/Max (excluding NaN): {np.nanmin(vis):.4f}/{np.nanmax(vis):.4f}')
                    print(f'  - Mean/Std (excluding NaN): {np.nanmean(vis):.4f}/{np.nanstd(vis):.4f}')
                    
                    nan_images.append((idx, img_path, nan_count, nan_percentage))
                    
            except Exception as e:
                print(f'[ERROR] Could not process image {idx}: {e}')
        
        return nan_images

    def get_filename(self, idx):
        """Return the filename for the sample at index idx"""
        return self.filenames[idx]
    
    def _identify_dataset_type(self, img_path: str) -> str:
        """Identify dataset type from path."""
        path_parts = img_path.split('/')
        if len(path_parts) < 6:
            return DatasetTypeEnum.SIMULATED
            
        split_5 = path_parts[5]
        split_6 = path_parts[6] if len(path_parts) > 6 else ""
        
        if split_5 == 'artefacts':
            return DatasetTypeEnum.ARTEFACTS
        elif split_5 == 'dataset':
            if split_6 == 'q1lenses':
                return DatasetTypeEnum.Q1_LENSES
            elif split_6 == 'trickygirls':
                return DatasetTypeEnum.TRICKY_GALAXIES
            elif split_6 == '2390':
                return DatasetTypeEnum.ERO_2390
            elif split_6 == 'M0416':
                return DatasetTypeEnum.M0416
            elif  split_6 == 'discovery' or split_6 == 'shared_test_set':
                return DatasetTypeEnum.Q1
            else:
                return DatasetTypeEnum.SIMULATED
        else:
            return DatasetTypeEnum.SIMULATED
    
    #--------------------------------------------------------------------------#

    def z_normalize(self, img, mean_value, std_value):
        """
        Standardize the image as ~N(0,1).
        """            
        # img_znorm = np.zeros_like(img)
        # # print('img.shape:', img.shape)
        # for i in range(img.shape[2]):                
        #         img_znorm[i] = (img[i]-mean_value[i]) / std_value[i]
                
        # img has shape (H, W, C), mean_value and std_value have shape (C,)
        # Convert to numpy arrays for broadcasting
        mean_array = np.array(mean_value)
        std_array = np.array(std_value)
        
        # Broadcasting will handle the normalization across all channels at once
        img_znorm = (img - mean_array) / std_array
            
        return img_znorm

    def normalize_0_1(self, img):
        """
        Normalize each channel of the image to the range [0, 1] independently.
        
        Args:
            img: numpy array of shape (H, W, C) where C is the number of channels
        
        Returns:
            Normalized image with each channel in range [0, 1]
        """
        
        # Handle different input shapes
        if len(img.shape) == 2:
            # Single channel image (H, W)
            min_value = img.min()
            max_value = img.max()
            
            if max_value - min_value < 1e-8:  # Avoid division by zero
                return np.zeros_like(img)
            
            normalized = (img - min_value) / (max_value - min_value)
            return np.clip(normalized, 0, 1)
        
        elif len(img.shape) == 3:
            # Multi-channel image (H, W, C)
            normalized_img = np.zeros_like(img)
            
            for ch in range(img.shape[2]):  # Loop through channels
                channel_data = img[:, :, ch]
                
                min_value = channel_data.min()
                max_value = channel_data.max()
                
                # Avoid division by zero for constant channels
                if max_value - min_value < 1e-8:
                    normalized_img[:, :, ch] = np.zeros_like(channel_data)
                else:
                    normalized_channel = (channel_data - min_value) / (max_value - min_value)
                    normalized_img[:, :, ch] = np.clip(normalized_channel, 0, 1)
            
            return normalized_img
        
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}. Expected (H, W) or (H, W, C)")
    #--------------------------------------------------------------------------#
    
    # def __getitem__(self, idx):
        
    #     img_path = os.path.join(self.root, self.df_cutouts.iloc[idx,0])
        
    #     split_5 = img_path.split('/')[5]
    #     split_6 = img_path.split('/')[6]
    #     # print('[INFO] Split 5:', split_5)
        
    #     # Initialize variables
    #     vis_mtf = None
    #     vis_asinh_low = None
    #     vis_asinh_high = None
        
    #     # handling fits cutouts
    #     if split_5 == 'artefacts':
            
    #         ### Fits header, WCS, pixelscale
    #         hdul = fits.open(img_path)
            
    #         # data = hdul[0].data[0]
    #         # wcs = WCS(hdul[0].header)
    #         vis = hdul[0].data[0]
    #         # y = hdul[1].data[1]
    #         # jh = hdul[2].data[2]
            
    #         # # first try
    #         # vis_mtf = self.mtf_arcsinh_scaling(vis, beta=1, percentile_min=0, percentile_max=99)
    #         # vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)   
    #         # vis_asinh_high = self.asinh_stretch_ds9(vis, percent=99.9)
            
    #         # second try
    #         vis_mtf = self.gradient_enhanced_stretch(vis, alpha=0.3)
    #         vis_asinh_high = self.optimized_adaptive_stretch(vis, clip_limits=[0.1, 0.5, 1])
    #         vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)
            
            
    #         hdul.close()
    
    #     # handling fits cutouts
    #     elif split_5 == 'dataset' and split_6 != 'q1lenses' and split_6 != 'trickygirls':

    #         vis = np.load(img_path)
    #          # Squeeze out any extra dimensions
    #         vis = np.squeeze(vis)
            
    #         # vis_mtf = self.mtf_arcsinh_scaling(vis, beta=1, percentile_min=0, percentile_max=99)
    #         # vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)
    #         # vis_asinh_high = self.asinh_stretch_ds9(vis, percent=99.9)
            
    #         # 2nd try
    #         vis_mtf = self.gradient_enhanced_stretch(vis, alpha=0.3)
    #         vis_asinh_high = self.optimized_adaptive_stretch(vis, clip_limits=[0.1, 0.5, 1])
    #         vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)

    #     elif split_5 == 'dataset' and split_6 == 'q1lenses':
    #         ### Fits header, WCS, pixelscale
    #         hdul = fits.open(img_path)
    #         # hdul.info()
            
    #         # data = hdul[0].data[0]
    #         # wcs = WCS(hdul[0].header)
    #         vis = hdul[0].data
    #         # y = hdul[1].data[1]
    #         # jh = hdul[2].data[2]
    #         # print(f"[INFO] Loaded cutout shape: {vis.shape} for {img_path}")
    #         # Ensure all cutouts are resized to 100x100
    #         if vis.shape != (100, 100):
    #             # Calculate zoom factors for both dimensions
    #             zoom_factor_h = 100 / vis.shape[0]
    #             zoom_factor_w = 100 / vis.shape[1]
    #             vis = zoom(vis, (zoom_factor_h, zoom_factor_w), order=1)  # order=1 for bilinear interpolation
    #             # print(f"[INFO] Resized cutout from {hdul[0].data.shape} to {vis.shape}")
            
    #         # vis_mtf = self.mtf_arcsinh_scaling(vis, beta=1, percentile_min=0, percentile_max=99)
    #         # vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)   
    #         # vis_asinh_high = self.asinh_stretch_ds9(vis, percent=99.9)
            
    #         # 2nd try
    #         vis_mtf = self.gradient_enhanced_stretch(vis, alpha=0.3)
    #         vis_asinh_high = self.optimized_adaptive_stretch(vis, clip_limits=[0.1, 0.5, 1])
    #         vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)
            
    #         # # improved 2nd try (non usato)
    #         # vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)
    #         # vis_asinh_high = self.improved_gradient_enhanced_stretch(vis, alpha=0.3, sigma_smooth=0.5, edge_threshold=0.1)
    #         # vis_mtf = self.improved_optimized_adaptive_stretch(vis, clip_limits=[0.1, 0.5, 1], adaptive_weights=True)

            
    
            
    #         hdul.close()
        
    #     elif split_5 == 'dataset' and split_6 == 'trickygirls':
    #         try:
    #             ### Fits header, WCS, pixelscale
    #             hdul = fits.open(img_path)
    #             # hdul.info()
                
    #             # Check if the file has valid data
    #             if len(hdul) == 0 or hdul[0].data is None:
    #                 hdul.close()
    #                 raise ValueError(f"Empty or corrupted FITS file: {img_path}")
                
    #             # data = hdul[0].data[0]
    #             # wcs = WCS(hdul[0].header)
    #             vis = hdul[0].data
                
    #             # Check if vis is valid
    #             if vis is None or vis.size == 0:
    #                 hdul.close()
    #                 raise ValueError(f"Empty data array in FITS file: {img_path}")
                
    #             # y = hdul[1].data[1]
    #             # jh = hdul[2].data[2]
    #             # print(f"[INFO] Loaded cutout shape: {vis.shape} for {img_path}")
    #             # Ensure all cutouts are resized to 100x100
    #             if vis.shape != (100, 100):
    #                 # Calculate zoom factors for both dimensions
    #                 zoom_factor_h = 100 / vis.shape[0]
    #                 zoom_factor_w = 100 / vis.shape[1]
    #                 vis = zoom(vis, (zoom_factor_h, zoom_factor_w), order=1)  # order=1 for bilinear interpolation
    #                 # print(f"[INFO] Resized cutout from {hdul[0].data.shape} to {vis.shape}")
                
    #             # vis_mtf = self.mtf_arcsinh_scaling(vis, beta=1, percentile_min=0, percentile_max=99)
    #             # vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)   
    #             # vis_asinh_high = self.asinh_stretch_ds9(vis, percent=99.9)
                
    #             # 2nd try
    #             vis_mtf = self.gradient_enhanced_stretch(vis, alpha=0.3)
    #             vis_asinh_high = self.optimized_adaptive_stretch(vis, clip_limits=[0.1, 0.5, 1])
    #             vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)
                
    #             # # improved 2nd try (non usato)
    #             # vis_asinh_low = self.asinh_stretch_ds9(vis, percent=99)
    #             # vis_asinh_high = self.improved_gradient_enhanced_stretch(vis, alpha=0.3, sigma_smooth=0.5, edge_threshold=0.1)
    #             # vis_mtf = self.improved_optimized_adaptive_stretch(vis, clip_limits=[0.1, 0.5, 1], adaptive_weights=True)

    #             hdul.close()
                
    #         except Exception as e:
    #             print(f"[ERROR] Corrupted trickygirls file at index {idx}: {img_path} - {e}")
    #             # Skip this corrupted file and try the next one
    #             if idx + 1 < len(self.df_cutouts):
    #                 return self.__getitem__(idx + 1)
    #             else:
    #                 # If this is the last file, try a previous one
    #                 if idx > 0:
    #                     return self.__getitem__(idx - 1)
    #                 else:
    #                     # If somehow we can't find any valid file, raise an error
    #                     raise RuntimeError(f"No valid files found in dataset starting from index {idx}")
          
    #     assert np.shape(vis_mtf)==np.shape(vis_asinh_low)
    #     assert np.shape(vis_asinh_high)==np.shape(vis_mtf)

    #     img = np.stack([vis_mtf, vis_asinh_low, vis_asinh_high])
        
        
    #     obj_class = self.df_cutouts.iloc[idx,1] # index of class column in the csv file 
    #     obj_class = torch.as_tensor(obj_class, dtype=torch.uint8)
        
    #     target    = obj_class
        
    #     # print(np.shape(img)) # = (3, 100, 100)
        
    #     resized_image = np.moveaxis(img, 0, -1)
    #     assert np.shape(resized_image)[-1]==3
    #     img = resized_image
    #     # print(np.shape(img)) # = (100, 100, 3)
    #     # resized_image = torch.as_tensor(resized_image, dtype=torch.float32)
        
    #     # # Ensure the image has the correct shape (H, W, C) for torchvision transforms
    #     # assert img.shape[-1] == 3, f"Expected 3 channels, got {img.shape}"
    
    #     # transforms1 = []
    #     # transforms1.append(ToTensor())
    #     # transforms1.append(ConvertImageDtype(torch.float))
    #     # self.transforms1 = Compose(transforms1)
    #     # resized_image,target = self.transforms1(resized_image, target)
        
    #     img = self.z_normalize(img, self.mean_value, self.std_value)
        
    #     if self.transforms is not None:
    #         # if target != torch.as_tensor(0, dtype=torch.uint8):
    #         img, target = self.transforms(img, target)


    #     return img, target

    #--------------------------------------------------------------------------#
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, self.df_cutouts.iloc[idx,0])
            
            # Identify dataset type and get appropriate loader
            dataset_type = self._identify_dataset_type(img_path)
            data_loader = self.data_loader_factory.create_loader(dataset_type)
            
            # Load raw data
            vis = data_loader.load_data(img_path)
            
            # Apply stretch pipeline to get 3-channel image
            img = self.stretch_pipeline.apply_stretches(vis)
            
            # Get target
            obj_class = self.df_cutouts.iloc[idx,1]
            target = torch.as_tensor(obj_class, dtype=torch.uint8)
            
            
            # Apply normalization
            img = self.z_normalize(img, self.mean_value, self.std_value)
            # img = self.normalize_0_1(img)
            # Apply transforms if specified
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            
            return img, target
            
        except Exception as e:
            print(f"[ERROR] Failed to load sample {idx}: {e}")
            # Fallback strategy: try next valid sample
            return self._get_fallback_sample(idx)
    
    def _get_fallback_sample(self, failed_idx: int):
        """Safe fallback when a sample fails to load."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Try next sample (with wraparound)
                fallback_idx = (failed_idx + attempt + 1) % len(self.df_cutouts)
                return self.__getitem__(fallback_idx)
            except Exception:
                continue
        
        # If all attempts fail, create a dummy sample
        dummy_img = np.zeros((100, 100, 3), dtype=np.float32)
        dummy_target = torch.as_tensor(0, dtype=torch.uint8)
        return dummy_img, dummy_target
    
    #--------------------------------------------------------------------------#

    def __len__(self):
        return len(self.df_cutouts)

    #--------------------------------------------------------------------------#   


class DatasetTypeEnum:
    ARTEFACTS = "artefacts"
    SIMULATED = "simulated"  
    Q1_LENSES = "q1lenses"
    TRICKY_GALAXIES = "trickygalaxies"
    ERO_2390 = "2390"
    M0416 = "M0416"
    Q1 = "Q1"

class DataLoaderFactory:
    """Factory for creating appropriate data loaders based on dataset type."""
    
    @staticmethod
    def create_loader(dataset_type: str):
        loaders = {
            DatasetTypeEnum.ARTEFACTS: ArtefactsLoader(),
            DatasetTypeEnum.SIMULATED: SimulatedLoader(),
            DatasetTypeEnum.Q1_LENSES: Q1LensesLoader(),
            DatasetTypeEnum.TRICKY_GALAXIES: TrickyGalaxiesLoader(),
            DatasetTypeEnum.ERO_2390: ERO2390Loader(),
            DatasetTypeEnum.M0416: M0416Loader(),
            DatasetTypeEnum.Q1: Q1Loader()
        }
        return loaders.get(dataset_type, SimulatedLoader())

class BaseDataLoader:
    """Base class for all data loaders."""
    
    def load_data(self, img_path: str) -> np.ndarray:
        raise NotImplementedError
    
    def validate_data(self, data: np.ndarray) -> np.ndarray:
        """Common validation logic."""
        if data is None or data.size == 0:
            raise ValueError(f"Empty data array")
        return np.nan_to_num(data)
    
    @staticmethod
    def _sky_stats(d: np.ndarray, clip_sigma=3.0, iters=3):
        d = np.asarray(d, dtype=np.float64)
        d = np.where(np.isfinite(d), d, 0.0)
        clipped = sigma_clip(d, sigma=clip_sigma, maxiters=iters,
                             cenfunc='median', stdfunc='std', masked=True)
        sky = np.nanmedian(clipped)
        resid = d - sky
        mad = np.median(np.abs(resid - np.median(resid)))
        noise = 1.4826 * mad
        if not np.isfinite(noise) or noise < 1e-8:
            noise = np.std(clipped.filled(sky))
            if noise < 1e-8:
                noise = 1.0
        return sky, noise

    @staticmethod
    def sky_subtract_only(data: np.ndarray, clip_sigma=3.0, iters=3) -> np.ndarray:
        d = np.asarray(data, dtype=np.float64)
        d = np.where(np.isfinite(d), d, 0.0)
        sky, _ = BaseDataLoader._sky_stats(d, clip_sigma=clip_sigma, iters=iters)
        return d - sky

    @staticmethod
    def rescale_to_q1(arr: np.ndarray, zp_ero: float, zp_q1: float) -> np.ndarray:
        # Convert ERO scale to Q1-like using zeropoints (constant factor)
        scale = 10 ** ((zp_q1 - zp_ero) / 2.5)
        return arr * scale

    @staticmethod
    def sky_subtract_and_whiten(data: np.ndarray, clip_sigma=3.0, iters=3) -> np.ndarray:
        d = np.asarray(data, dtype=np.float64)
        d = np.where(np.isfinite(d), d, 0.0)
        clipped = sigma_clip(d, sigma=clip_sigma, maxiters=iters, cenfunc='median', stdfunc='std', masked=True)
        sky = np.nanmedian(clipped)
        resid = d - sky
        mad = np.median(np.abs(resid - np.median(resid)))
        noise = 1.4826 * mad
        if not np.isfinite(noise) or noise < 1e-8:
            noise = np.std(clipped.filled(sky))
            if noise < 1e-8:
                noise = 1.0
        return (d - sky) / noise

class ArtefactsLoader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        try:
            vis = hdul[0].data[0]
            return self.validate_data(vis)
        finally:
            hdul.close()

class Q1LensesLoader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        try:
            vis = hdul[0].data
            vis = self.validate_data(vis)
            # print(vis.shape)
            
            # Ensure 100x100 size
            if vis.shape != (100, 100):
                zoom_factor_h = 100 / vis.shape[0]
                zoom_factor_w = 100 / vis.shape[1]
                vis = zoom(vis, (zoom_factor_h, zoom_factor_w), order=0)
            
            return vis
        finally:
            hdul.close()

class TrickyGalaxiesLoader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        try:
            hdul = fits.open(img_path)
            if len(hdul) == 0 or hdul[0].data is None:
                hdul.close()
                raise ValueError(f"Empty FITS file: {img_path}")
            
            vis = hdul[0].data
            vis = self.validate_data(vis)
            
            # Ensure 100x100 size
            if vis.shape != (100, 100):
                zoom_factor_h = 100 / vis.shape[0]
                zoom_factor_w = 100 / vis.shape[1]
                vis = zoom(vis, (zoom_factor_h, zoom_factor_w), order=0)
            
            hdul.close()
            return vis
            
        except Exception as e:
            raise ValueError(f"Corrupted trickygalaxies file: {img_path} - {e}")

class SimulatedLoader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        vis = np.load(img_path)
        return self.validate_data(np.squeeze(vis))

class ERO2390Loader(BaseDataLoader):
    
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        try:
            vis = hdul[1].data  # VIS is in HDU 1
            vis = self.validate_data(vis)

            # Preprocess ERO to be Q1-like (sky -> scale), matching pixel_distribution1.py 'sky_scale'
            mode = 'sky_scale'

            if mode == 'raw':
                pass  # do nothing
            elif mode == 'sky':
                vis = self.sky_subtract_only(vis)
            elif mode == 'sky_scale':
                # 1) sky subtract
                vis = self.sky_subtract_only(vis)
                # 2) rescale to Q1 zeropoint
                zp_ero = 30.132
                zp_q1  = 24.6
                vis = self.rescale_to_q1(vis, zp_ero=zp_ero, zp_q1=zp_q1)
            elif mode == 'whiten':
                # legacy/debug: sky subtract + noise whitening (not recommended here)
                vis = self.sky_subtract_and_whiten(vis)
            else:
                raise ValueError(f"Unknown ERO_PREPROC mode: {mode}")

            # Ensure 100x100 size
            if vis.shape != (100, 100):
                zh = 100 / vis.shape[0]; zw = 100 / vis.shape[1]
                vis = zoom(vis, (zh, zw), order=0)

            return vis
        finally:
            hdul.close()

class M0416Loader(BaseDataLoader):
    
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        try:
            vis = hdul[0].data  # VIS is in HDU 0
            vis = self.validate_data(vis)

            # Preprocess ERO to be Q1-like (sky -> scale), matching pixel_distribution1.py 'sky_scale'
            mode = 'sky_scale'

            if mode == 'raw':
                pass  # do nothing
            elif mode == 'sky':
                vis = self.sky_subtract_only(vis)
            elif mode == 'sky_scale':
                # 1) sky subtract
                # vis = self.sky_subtract_only(vis)
                # 2) rescale to Q1 zeropoint
                zp_ero = 23.900
                zp_q1  = 24.6
                vis = self.rescale_to_q1(vis, zp_ero=zp_ero, zp_q1=zp_q1)
            elif mode == 'whiten':
                # legacy/debug: sky subtract + noise whitening (not recommended here)
                vis = self.sky_subtract_and_whiten(vis)
            else:
                raise ValueError(f"Unknown ERO_PREPROC mode: {mode}")

            # Ensure 100x100 size
            if vis.shape != (100, 100):
                zh = 100 / vis.shape[0]; zw = 100 / vis.shape[1]
                vis = zoom(vis, (zh, zw), order=0)

            return vis
        finally:
            hdul.close()

class Q1Loader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        try:
            vis = hdul[0].data
            vis = self.validate_data(vis)
            
            # Ensure 100x100 size
            if vis.shape != (100, 100):
                zoom_factor_h = 100 / vis.shape[0]
                zoom_factor_w = 100 / vis.shape[1]
                vis = zoom(vis, (zoom_factor_h, zoom_factor_w), order=0)
            
            return vis
        finally:
            hdul.close()

class StretchPipeline:
    """Handles all stretch operations for the 3-channel output."""
    
    def __init__(self, stretch_methods):
        self.stretch_methods = stretch_methods
    
    def apply_stretches(self, data: np.ndarray) -> np.ndarray:
        """Apply the three stretch methods and return stacked channels."""
        
        # Apply your current 2nd try stretches
        vis_mtf = self.stretch_methods.gradient_enhanced_stretch(data, alpha=0.3) # 2ND TRY ###########################
        # vis_mtf = OptimizedVISChannelProcessor(self.stretch_methods).apply_ds9_stretch(data, pmax=99.5, bias=0.5, contrast=1, 
        #                                                                                stretch='power', stretch_param=2, output_range=(0, 1))
        vis_asinh = self.stretch_methods.asinh_stretch_ds9(data, percent=99)
        vis_clahe = self.stretch_methods.optimized_adaptive_stretch(data, clip_limits=[0.1, 0.5, 1])
        
        # Validate shapes
        assert vis_mtf.shape == vis_asinh.shape == vis_clahe.shape

        # Stack and move axis for torchvision compatibility
        img = np.stack([vis_mtf, vis_asinh, vis_clahe])
        img = np.moveaxis(img, 0, -1)  # (3, H, W) -> (H, W, 3)
        
        return img


################################################################################
################################################################################
# Replace your current StretchPipeline class with this enhanced version
class EnhancedStretchPipeline:
    """
    Enhanced stretch pipeline with 6 optimized VIS channels.
    """
    
    def __init__(self, stretch_methods, num_channels=6):
        self.stretch_methods = stretch_methods
        self.num_channels = num_channels
        self.channel_processor = OptimizedVISChannelProcessor(stretch_methods)
        
    def apply_stretches(self, data: np.ndarray) -> np.ndarray:
        """Apply multi-channel stretches based on configuration."""
        
        if self.num_channels == 3:
            # Original 3-channel implementation
            vis_mtf = self.stretch_methods.gradient_enhanced_stretch(data, alpha=0.3)
            vis_asinh_high = self.stretch_methods.optimized_adaptive_stretch(
                data, clip_limits=[0.1, 0.5, 1])
            vis_asinh_low = self.stretch_methods.asinh_stretch_ds9(data, percent=99)
            
            img = np.stack([vis_mtf, vis_asinh_low, vis_asinh_high])
            img = np.moveaxis(img, 0, -1)
            
        elif self.num_channels == 6:
            # New 6-channel implementation
            img = self.channel_processor.process_six_channels(data)
            # img = self.channel_processor.process_six_channels_asinh_family(data)
        
        elif self.num_channels == 9:
            # New 6-channel implementation
            img = self.channel_processor.process_nine_channels(data)
        
        elif self.num_channels == 'all' or (isinstance(self.num_channels, int) and self.num_channels > 9):
            img_all = self.channel_processor.process_all_channels(data)  # (H,W,C_all)
            if isinstance(self.num_channels, int):
                # keep first K deterministically
                C_keep = min(self.num_channels, img_all.shape[-1])
                img = img_all[..., :C_keep]
            else:
                img = img_all
            
        else:
            raise ValueError(f"Unsupported number of channels: {self.num_channels}")
        
        return img

# Add the OptimizedVISChannelProcessor class
class OptimizedVISChannelProcessor:
    """6-channel VIS-only processor with minimized redundancy."""
    
    def __init__(self, stretch_methods):
        self.stretch_methods = stretch_methods
    
    def conservative_photometric_stretch(self, data, percentile=99):
        """Channel 1: Preserves photometric accuracy."""
        data = np.nan_to_num(data)
        vmin = np.percentile(data, 1)
        vmax = np.percentile(data, percentile)
        
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        
        # Simple linear stretch - preserve the natural range
        stretched = np.clip(((data - vmin) / (vmax - vmin)), 0, percentile)
        
        result = stretched
        # # Apply gentle asinh but don't force to [0,1]
        # beta = 1
        # result = np.arcsinh(beta * stretched) / np.arcsinh(beta)
        return result

    def faint_feature_enhancer(self, data, boost_scales=[1, 2, 4]):
        """Channel 2: Maximizes visibility of faint features."""
        data = np.nan_to_num(data)
        enhanced = np.zeros_like(data)
        
        vmin = np.percentile(data, 0)
        vmax = np.percentile(data, 99.5)
        
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        
        min_value = np.min(data)
        max_value = np.max(data)
        # Simple linear stretch - preserve the natural range
        # stretched = (data - vmin) / (vmax - vmin)
        stretched = (data - min_value) / (max_value - min_value)
        vmax = np.percentile(stretched, 99.9)
        vmin = np.percentile(stretched, 1)
        stretched = np.clip(stretched, vmin, vmax)  # Ensure within [0, 1]
        
        for scale in boost_scales:
            smoothed = gaussian_filter(stretched, sigma=scale)
            detail = stretched - smoothed
            brightness_threshold = np.percentile(smoothed, 90)
            faint_mask = smoothed < brightness_threshold
            amplification = np.where(faint_mask, 10 / scale, 0.5 / scale)
            enhanced += detail * amplification
        
        result = data + enhanced
        # return self.robust_normalize(result, lower_p=1, upper_p=99)
        # return self.outlier_robust_normalize(result)
        return result
    
    def structure_boundary_enhancer(self, data, edge_scales=[0.5, 1, 2]):
        """Channel 3: Emphasizes structural boundaries."""
        data = np.nan_to_num(data)
        edge_responses = []
        vmax = np.percentile(data, 99.5)
        vmin = np.percentile(data, 1)
        data = np.clip(data, vmin, vmax)
        # # apply asinh stretch
        # beta = 1
        # data = np.arcsinh(beta * data) / np.arcsinh(beta)
        
        for sigma in edge_scales:
            smoothed = gaussian_filter(data, sigma=sigma)
            grad_x = sobel(smoothed, axis=0)
            grad_y = sobel(smoothed, axis=1)
            edge_mag = np.sqrt(grad_x**2 + grad_y**2)
            edge_responses.append(edge_mag)
        
        combined_edges = np.zeros_like(data)
        weights = [0.3, 0.5, 0.2]
        
        for edge_resp, weight in zip(edge_responses, weights):
            combined_edges += weight * edge_resp
        
        edge_norm = combined_edges / (np.percentile(combined_edges, 99) + 1e-8)
        # print(f"[INFO] Edge norm max: {edge_norm.max()}, min: {edge_norm.min()}")
        # edge_norm = np.clip(edge_norm, 0, edge_norm.max())
        enhanced = data + 0.6 * edge_norm * data
        
        # return self.robust_normalize(enhanced)
        # return self.outlier_robust_normalize(enhanced)
        return enhanced
    
    def radial_profile_preserving_stretch(self, data, ring_radii=[5, 10, 15, 20]):
        """Channel 4: Optimized for radial structures."""
        data = np.nan_to_num(data)
        center = np.array(data.shape) // 2
        y, x = np.ogrid[:data.shape[0], :data.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        max_radius = int(r.max())
        radial_profile = np.zeros(max_radius + 1)
        
        for radius in range(max_radius + 1):
            mask = (r >= radius - 0.5) & (r < radius + 0.5)
            if mask.any():
                radial_profile[radius] = np.median(data[mask])
        
        # Smooth radial profile
        smooth_profile = gaussian_filter1d(radial_profile, sigma=2)
        radial_bg = np.interp(r.flatten(), range(len(smooth_profile)), 
                            smooth_profile).reshape(data.shape)
        
        residual = data - radial_bg
        ring_enhancement = np.ones_like(data)
        for radius in ring_radii:
            ring_mask = (r > radius - 2) & (r < radius + 2)
            ring_enhancement[ring_mask] *= 1.3
        
        enhanced = radial_bg + residual * ring_enhancement
        # return self.robust_normalize(enhanced)
        # return self.outlier_robust_normalize(enhanced)
        return enhanced
    
    def multi_scale_contrast_enhancer(self, data, scales=[0.1, 0.5, 1]):
        """Channel 5: Multi-scale contrast enhancement."""
        data = np.nan_to_num(data)
        enhanced_versions = []
        
        for scale in scales:
            sigma1 = scale
            sigma2 = scale * 1.6
            gaussian1 = gaussian_filter(data, sigma=sigma1)
            gaussian2 = gaussian_filter(data, sigma=sigma2)
            dog = gaussian1 - gaussian2
            enhanced = data + 0.2 * dog
            enhanced_versions.append(enhanced)
        
        weights = [0.5, 0.25, 0.25]
        result = np.average(enhanced_versions, axis=0, weights=weights)
        # return self.robust_normalize(result)
        # return self.outlier_robust_normalize(result)
        return result
    
    def optimized_gradient_stretch(self, data, alpha=0.5):
        """Channel 6: Optimized gradient enhancement."""
        data = np.nan_to_num(data)
        smoothed = data # gaussian_filter(data, sigma=0.001)
        
        grad_x = sobel(smoothed, axis=0)
        grad_y = sobel(smoothed, axis=1)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_norm = gradient_mag / (np.percentile(gradient_mag, 99.9) + 1e-8)
        grad_norm = np.clip(grad_norm, 0, 1)
        enhanced = data + alpha * grad_norm * data
        
        # return self.robust_normalize(enhanced)
        # return self.outlier_robust_normalize(enhanced)
        return enhanced

    def pure_edge_detector(self, data):
        """Pure edge map, not additive enhancement."""
        data = np.nan_to_num(data)
        
        # Multi-scale edge detection
        grad_x = sobel(data, axis=0)
        grad_y = sobel(data, axis=1)
        edge_map = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return pure edge map (not data + edges)
        return edge_map
    
    def logarithmic_compressor(self, data):
        """Logarithmic compression for high dynamic range."""
        data = np.nan_to_num(data)
        
        # Shift to positive values
        data_positive = data - data.min() + 1e-6
        
        # Apply log compression
        log_compressed = np.log10(data_positive + 1)
        
        return log_compressed
    
    def power_law_compressor(self, data, gamma=2.0, normalization='adaptive'):
        """
        Channel 5: Power law compression (replaces multi_scale_contrast).
        
        Args:
            gamma: Power law exponent 
                   - gamma > 1: Compresses bright features, enhances faint
                   - gamma < 1: Enhances bright features, compresses faint
            normalization: How to handle dynamic range
        """
        data = np.nan_to_num(data)
        
        # Ensure data is positive for power law
        data_min = np.min(data)
        if data_min <= 0:
            data_shifted = data - data_min + 1e-6
        else:
            data_shifted = data
        
        if normalization == 'adaptive':
            # Normalize to [0,1] range first, then apply power law
            vmin = np.percentile(data_shifted, 0)
            vmax = np.percentile(data_shifted, 95)
            
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
                
            normalized = (data_shifted - vmin) / (vmax - vmin)
            # normalized = np.clip(normalized, 0, 1)
            
            # Apply power law
            power_compressed = np.power(normalized, gamma)
            
        elif normalization == 'robust':
            # Apply power law directly with robust scaling
            power_compressed = np.power(data_shifted, gamma)
            
            # Robust normalization after power transform
            vmin = np.percentile(power_compressed, 0)
            vmax = np.percentile(power_compressed, 99.9)
            
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
                
            power_compressed = (power_compressed - vmin) / (vmax - vmin)
            
        else:  # 'direct'
            # Direct power law without normalization
            power_compressed = np.power(data_shifted, gamma)
        
        return power_compressed
    
    def ds9_log_stretch(self, data, bias=0.5, percent=99):
        """
        DS9-accurate log stretch implementation.
        """
        data = np.nan_to_num(data)
        
        # Use percentile for max (more robust than absolute max) # E INVECE NO
        data_min = np.min(data)
        data_max = np.max(data) # np.percentile(data, percent)

        if data_max <= data_min:
            data_max = data_min + 1e-10
        
        # Normalize to [0, 1]
        normalized = (data - data_min) / (data_max - data_min)
        # normalized = np.clip(normalized, 0, 1)
        
        # DS9's actual log formula
        if bias <= 0:
            bias = 1e-10
        
        # Actual to DS9's actual implementation da internet
        log_result = np.log10(bias * normalized + 1) / np.log10(bias + 1)
    
        
        return log_result 
    
    def ds9_pow_stretch(self, data, bias=1000, percent=99.5):
        """
        DS9-accurate power stretch implementation.
        """
        data = np.nan_to_num(data)
        
        # Use percentile for max (more robust than absolute max) # E INVECE NO
        data_min = np.min(data)
        data_max = np.percentile(data, percent)
        print(f"[INFO] Data min: {data_min}, Data max: {data_max} for {percent}%")
        data = np.clip(data, data_min, data_max)  # Ensure no outliers


        if data_max <= data_min:
            data_max = data_min + 1e-10
        
        # Normalize to [0, 1]
        normalized = (data - data_min) / (data_max - data_min)
        

        # DS9's actual square root formula
        if bias <= 0:
            bias = 1e-10
        
        pow_result = ((bias**normalized)-1) / (bias)        
        
        
        return pow_result 

    def ds9_sqrt_stretch(self, data, bias=0.5, percent=99.5):
        """
        DS9-accurate square root stretch implementation.
        """
        data = np.nan_to_num(data)
        
        # Use percentile for max (more robust than absolute max) # E INVECE NO
        data_min = np.min(data)
        data_max = np.max(data) # np.percentile(data, percent)


        if data_max <= data_min:
            data_max = data_min + 1e-10
        
        # Normalize to [0, 1]
        normalized = (data - data_min) / (data_max - data_min)
        # normalized = np.clip(normalized, 0, 1)

        # DS9's actual square root formula
        if bias <= 0:
            bias = 1e-10
        
        # sqrt_result = np.sqrt(bias * normalized + 1) / np.sqrt(bias + 1)
        
        # # This is DS9's exact sqrt formula
        # sqrt_bias_plus_1 = np.sqrt(bias + 1)
        # sqrt_result = (np.sqrt(bias * normalized + 1) - 1) / (sqrt_bias_plus_1 - 1)
        sqrt_result = np.sqrt(normalized)
                
        
        return sqrt_result 
    
    def sqrt_compressor(self, data, pre_scaling='percentile'):
        """
        Channel 6: Square root compression (replaces optimized_gradient).
        
        Square root provides balanced compression between linear and log.
        Particularly good for preserving mid-range features.
        """
        data = np.nan_to_num(data)
        
        # Ensure data is positive
        data_min = np.min(data)
        if data_min < 0:
            data_shifted = data - data_min + 1e-6
        else:
            data_shifted = data + 1e-6  # Small offset to avoid sqrt(0)
        
        if pre_scaling == 'percentile':
            # Scale data using percentiles before sqrt
            vmin = np.percentile(data_shifted, 0.5)
            vmax = np.percentile(data_shifted, 99.5)
            
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
                
            scaled = (data_shifted - vmin) / (vmax - vmin)
            scaled = np.clip(scaled, 0, None)  # Ensure non-negative
            
            sqrt_compressed = np.sqrt(scaled)
            
        elif pre_scaling == 'robust':
            # Use median and MAD for robust scaling
            median = np.median(data_shifted)
            mad = np.median(np.abs(data_shifted - median))
            
            # Scale using robust statistics
            scaled = data_shifted / (median + 3 * mad)
            scaled = np.clip(scaled, 0, None)
            
            sqrt_compressed = np.sqrt(scaled)
            
        else:  # 'direct'
            sqrt_compressed = np.sqrt(data_shifted)
        
        return sqrt_compressed
    
    def robust_normalize(self, data, lower_p=0.5, upper_p=99.5):
        """Robust normalization handling outliers."""
        vmin, vmax = np.percentile(data, [lower_p, upper_p])
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        return np.clip((data - vmin) / (vmax - vmin), 0, 1)
    
    def outlier_robust_normalize(self, data, sigma_clip=5):
        """
        Remove extreme outliers but preserve natural dynamic range.
        Don't force to [0,1]!
        """
        # Calculate robust statistics
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Clip extreme outliers only (5-sigma equivalent)
        lower_bound = median - sigma_clip * 1.4826 * mad
        upper_bound = median + sigma_clip * 1.4826 * mad
        
        # Clip outliers but DON'T normalize to [0,1]
        clipped = np.clip(data, lower_bound, upper_bound)
        
        return clipped
    


    def apply_ds9_stretch(self, data, pmin=1, pmax=99, bias=0.5, contrast=1, stretch='linear', stretch_param=None, output_range=(0, 1)):
        """
        Apply DS9-style stretch directly to a numpy array.
        
        Parameters:
        -----------
        data : np.array
            Input image data
        pmin, pmax : float
            Percentiles for determining vmin/vmax (default: 1, 99)
        bias : float
            Bias parameter [0, 1] (default: 0.5)
        contrast : float
            Contrast parameter [0, +inf] (default: 1)
        stretch : str
            Stretch type: 'linear', 'sqrt', 'power', 'log', 'asinh', 'sinh', 'squared'
        stretch_param : float
            Parameter for power, log, and asinh stretches
        output_range : tuple
            Output range for stretched data (default: (0, 1))
        
        Returns:
        --------
        np.array
            Stretched image data
        """
        
        # Handle NaN values
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            return np.full_like(data, output_range[0])
        
        # substitute 0 to nan
        data = np.where(finite_mask, data, 0)

        # Create the stretch function
        if stretch == 'linear':
            stretch_func = astrovi.LinearStretch()
        elif stretch == 'sqrt':
            stretch_func = astrovi.SqrtStretch()
        elif stretch == 'power':
            if stretch_param is None:
                raise ValueError("stretch_param must be provided for power stretch.")
            stretch_func = astrovi.PowerStretch(stretch_param)
        elif stretch == 'log':
            if stretch_param is None:
                stretch_param = 1000
            stretch_func = astrovi.LogStretch(stretch_param)
        elif stretch == 'asinh':
            if stretch_param is None:
                stretch_param = 0.1
            stretch_func = astrovi.AsinhStretch(stretch_param)
        elif stretch == 'sinh':
            if stretch_param is None:
                stretch_param = 1./3.
            stretch_func = astrovi.SinhStretch(stretch_param)
        elif stretch == 'squared':
            stretch_func = astrovi.SquaredStretch()
        else:
            raise ValueError(f'Unknown stretch: {stretch}.')
        
        # Apply bias and contrast
        composite_stretch = astrovi.CompositeStretch(
            stretch_func, astrovi.ContrastBiasStretch(contrast, bias)
        )
        
        # Normalize data to [0, 1] range first
        # data_normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        # print(f"[INFO] minval: {minval}, maxval: {maxval} for pmin: {pmin}, pmax: {pmax}")  
        
        # Apply the stretch
        stretched_data = composite_stretch(data)
        # Calculate percentiles for vmin/vmax
        # vmin, vmax = np.percentile(finite_data, [pmin, pmax])
        vmax_stretched = np.percentile(stretched_data, pmax)
        vmin_stretched = np.percentile(stretched_data, pmin)
        # print(f"[INFO] vmin: {vmin}, vmax: {vmax}, vmax stretched: {vmax_stretched}")
        result = np.clip(stretched_data, vmin_stretched, vmax_stretched)  # Ensure within [0, 1]
        # minval = np.min(result)
        # maxval = np.max(result)
        # result = (result - minval) / (maxval - minval)
        # # Scale to desired output range
        # output_min, output_max = output_range
        # result = stretched_data * (output_max - output_min) + output_min
        
        # Handle NaN values in output
        result = np.where(finite_mask, result, np.nan)
        
        return result
    
    
    
    
    def process_six_channels(self, data):
        """Process VIS data into 6 optimized channels."""
        # ch1 = self.apply_ds9_stretch(data, pmax=99.5, bias=0.5, contrast=1, stretch='power', stretch_param=2, output_range=(0, 1)) # 6ch pow
        ch1 = self.stretch_methods.gradient_enhanced_stretch(data, alpha=0.3) # 6ch 2ndtry
        ch2 = self.stretch_methods.asinh_stretch_ds9(data, percent=99)
        ch3 = self.stretch_methods.optimized_adaptive_stretch(data, clip_limits=[0.1, 0.5, 1])
        
        # ch4 = self.stretch_methods.gradient_enhanced_stretch(data, alpha=0.3)
        ch4 = self.apply_ds9_stretch(data, pmax=99.5, bias=0.5, contrast=1, stretch='log', stretch_param=None, output_range=(0, 1))
        ch5 = self.structure_boundary_enhancer(data)
        ch6 = self.faint_feature_enhancer(data)
        
        
        
        channels = np.stack([ch1, ch2, ch3, ch4, ch5, ch6])
        img = np.moveaxis(channels, 0, -1)  # (6, H, W) -> (H, W, 6)
        return img
    
    def process_nine_channels(self, data):
        """Process VIS data into 6 optimized channels."""
        ch1 = self.conservative_photometric_stretch(data)
        ch2 = self.faint_feature_enhancer(data)
        # ch3 = self.structure_boundary_enhancer(data)
        ch3 = self.pure_edge_detector(data)  # Use pure edge map
        
        
        # ch4 = self.ds9_log_stretch(data, bias=1000)  # Alternative channel 3
        # ch4 = self.radial_profile_preserving_stretch(data)
        ch4 = self.apply_ds9_stretch(data, pmax=99.5, bias=0.5, contrast=1, stretch='log', stretch_param=None, output_range=(0, 1))
        
        # ch5 = self.multi_scale_contrast_enhancer(data)
        # ch5 = self.ds9_pow_stretch(data, bias=1000)  # Alternative channel 4
        ch5 = self.apply_ds9_stretch(data, pmax=99.5, bias=0.5, contrast=1, stretch='power', stretch_param=2, output_range=(0, 1))
        
        # ch6 = self.ds9_sqrt_stretch(data)
        # ch6 = self.optimized_gradient_stretch(data)
        ch6 = self.apply_ds9_stretch(data, pmax=99.5, bias=0.5, contrast=1, stretch='sinh', stretch_param=None, output_range=(0, 1))
        
        ch7 = self.stretch_methods.gradient_enhanced_stretch(data, alpha=0.3)
        ch8 = self.stretch_methods.optimized_adaptive_stretch(data, clip_limits=[0.1, 0.5, 1])
        ch9 = self.stretch_methods.asinh_stretch_ds9(data, percent=99)
        

        channels = np.stack([ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9])
        img = np.moveaxis(channels, 0, -1)  # (9, H, W) -> (H, W, 9)
        return img

    def process_all_channels(self, data):
        """
        Build a rich VIS-only channel set (~20), covering your defined stretches
        with complementary parameterizations. Returns (H, W, C_all).
        """
        d = np.nan_to_num(data)
        chans = []
        names = []

        # Core 3 (your 3-ch baseline family)
        chans.append(self.stretch_methods.gradient_enhanced_stretch(d, alpha=0.3)); names.append("GradEnhanced_a0.3")
        chans.append(self.stretch_methods.asinh_stretch_ds9(d, percent=99));        names.append("Asinh_p99")
        chans.append(self.stretch_methods.optimized_adaptive_stretch(d, clip_limits=[0.1, 0.5, 1])); names.append("OptAdaptive_std")

        # DS9 family
        chans.append(self.apply_ds9_stretch(d, pmax=99.0, bias=0.5, contrast=1, stretch='log'));   names.append("DS9_Log")
        chans.append(self.apply_ds9_stretch(d, pmax=99.0, bias=0.5, contrast=1, stretch='power', stretch_param=2)); names.append("DS9_Pow_g2")
        chans.append(self.apply_ds9_stretch(d, pmax=99.0, bias=0.5, contrast=1, stretch='sqrt'));  names.append("DS9_Sqrt")
        chans.append(self.apply_ds9_stretch(d, pmax=99.0, bias=0.5, contrast=1, stretch='sinh'));  names.append("DS9_Sinh")
        chans.append(self.apply_ds9_stretch(d, pmax=99.0, bias=0.5, contrast=1, stretch='asinh', stretch_param=0.1)); names.append("DS9_Asinh_0.1")

        # Edge/structure and faint features
        chans.append(self.pure_edge_detector(d));                               names.append("Edge_Pure")
        chans.append(self.structure_boundary_enhancer(d));                      names.append("BoundaryEnh")
        chans.append(self.faint_feature_enhancer(d));                           names.append("FaintFeatEnh")

        # Contrast/multi-scale
        chans.append(self.multi_scale_contrast_enhancer(d, scales=[0.1,0.5,1])); names.append("MSContrast")
        chans.append(self.optimized_gradient_stretch(d, alpha=0.5));             names.append("GradOpt_a0.5")
        chans.append(self.stretch_methods.improved_gradient_enhanced_stretch(d, alpha=0.3, sigma_smooth=0.5, edge_threshold=0.1)); names.append("GradEnhanced_IMP")
        chans.append(self.stretch_methods.improved_optimized_adaptive_stretch(d));               names.append("OptAdaptive_IMP")

        # Photometric/radial
        chans.append(self.conservative_photometric_stretch(d, percentile=99));   names.append("Photometric_p99")
        chans.append(self.radial_profile_preserving_stretch(d));                 names.append("RadialPreserve")

        # Compression variants
        chans.append(self.logarithmic_compressor(d));                            names.append("LogCompressor")
        chans.append(self.power_law_compressor(d, gamma=2.0, normalization='adaptive')); names.append("PowerLaw_g2")
        chans.append(self.power_law_compressor(d, gamma=0.5, normalization='adaptive')); names.append("PowerLaw_g0.5")
        chans.append(self.sqrt_compressor(d, pre_scaling='percentile'));         names.append("SqrtCompressor")

        # Stack -> (C, H, W) -> (H, W, C)
        channels = np.stack(chans, axis=0)
        img = np.moveaxis(channels, 0, -1)
        self._all_channel_names = names
        return img

    def process_six_channels_asinh_family(self, data):
        """
        6-ch set: first 3 = your 3-ch baseline,
        next 3 = Asinh variants targeting different regimes.
        Order: [GradEnhanced, Asinh_p99, OptAdaptive, Asinh_beta=0.05, Asinh_beta=0.15, Asinh_p99.7]
        """
        d = np.nan_to_num(data)
        # First 3 match exactly your 3-ch ResNet training
        ch1 = self.stretch_methods.gradient_enhanced_stretch(d, alpha=0.3)
        ch2 = self.stretch_methods.asinh_stretch_ds9(d, percent=99)
        ch3 = self.stretch_methods.optimized_adaptive_stretch(d, clip_limits=[0.1, 0.5, 1])
        # Asinh variants:
        # - Smaller beta => more log-like (strong core compression, faint boosted)
        # - Larger beta => more linear-like (preserve bright structure)
        ch4 = self.apply_ds9_stretch(d, pmax=99.5, bias=0.5, contrast=1.2, stretch='asinh', stretch_param=0.15)
        ch5 = self.apply_ds9_stretch(d, pmax=99.7, bias=0.5, contrast=1, stretch='asinh', stretch_param=0.15)
       # Higher clipping to keep more of bright structures
        ch6 = self.apply_ds9_stretch(d, pmax=99.9, bias=0.5, contrast=0.5, stretch='asinh', stretch_param=0.15)
        channels = np.stack([ch1, ch2, ch3, ch4, ch5, ch6])
        img = np.moveaxis(channels, 0, -1)
        self._six_asinh_names = ["GradEnhanced_a0.3", "Asinh_p99", "OptAdaptive_std",
                                 "DS9_Asinh_b0.05", "DS9_Asinh_b0.15", "Asinh_p99.7"]
        return img

    def get_six_asinh_names(self):
        return getattr(self, "_six_asinh_names", None)
    
    def get_all_channel_names(self):
        return getattr(self, "_all_channel_names", None)

# Add ChannelAnalysisTools class
class ChannelAnalysisTools:
    """Comprehensive tools for analyzing multi-channel contributions."""
    
    def __init__(self, num_channels=9):
        self.num_channels = num_channels
        if num_channels == 9:
            self.channel_names = [
                'Cons._Phot.', 'Faint_Feat_Enh', 
                'Pure Edge det.','Log compr.',
                'Power_Law_Compr.', 'Sqrt_Compr.',
                'Grad._Enhanced', 'Opt._Adaptive', 'Asinh_Stretch'
            ]
        elif num_channels == 6:
            self.channel_names = [
                'Power_Stretch', 
                'Asinh_Stretch', 
                'Opt._Adaptive', 
                'Grad._Enhanced',
                'Pure Edge det.', 
                'Faint_Feat_Enh' 
            ]
    
    def compute_channel_statistics(self, dataset, sample_size=1000):
        """Compute comprehensive statistics for each channel."""
        if sample_size > len(dataset):
            sample_size = len(dataset)
        
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        channel_stats = {
            'means': np.zeros(self.num_channels),
            'stds': np.zeros(self.num_channels),
            'mins': np.zeros(self.num_channels),
            'maxs': np.zeros(self.num_channels),
            'correlation_matrix': np.zeros((self.num_channels, self.num_channels))
        }
        
        all_channel_data = [[] for _ in range(self.num_channels)]
        
        for idx in tqdm(indices, desc="Computing channel statistics"):
            try:
                img, _ = dataset[idx]
                
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                if img.shape[0] == self.num_channels:
                    img = np.transpose(img, (1, 2, 0))
                
                for ch in range(self.num_channels):
                    ch_data = img[:, :, ch].flatten()
                    all_channel_data[ch].extend(ch_data)
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        for ch in range(self.num_channels):
            data = np.array(all_channel_data[ch])
            channel_stats['means'][ch] = np.mean(data)
            channel_stats['stds'][ch] = np.std(data)
            channel_stats['mins'][ch] = np.min(data)
            channel_stats['maxs'][ch] = np.max(data)
        
        # Compute correlation matrix
        correlation_data = np.zeros((min(10000, len(all_channel_data[0])), self.num_channels))
        sample_indices = np.random.choice(len(all_channel_data[0]), 
                                        min(10000, len(all_channel_data[0])), replace=False)
        
        for ch in range(self.num_channels):
            correlation_data[:, ch] = np.array(all_channel_data[ch])[sample_indices]
        
        channel_stats['correlation_matrix'] = np.corrcoef(correlation_data.T)
        return channel_stats
    
    def generate_analysis_report(self, dataset, sample_size=1000):
        """Generate comprehensive analysis report."""
        print("=" * 60)
        print("MULTI-CHANNEL ANALYSIS REPORT")
        print("=" * 60)
        
        stats = self.compute_channel_statistics(dataset, sample_size)
        
        print("\n1. CHANNEL STATISTICS:")
        print("-" * 30)
        for i, name in enumerate(self.channel_names):
            print(f"{name}:")
            print(f"  Mean: {stats['means'][i]:.4f}")
            print(f"  Std:  {stats['stds'][i]:.4f}")
            print(f"  Range: [{stats['mins'][i]:.4f}, {stats['maxs'][i]:.4f}]")
        
        print("\n2. CORRELATION MATRIX:")
        print("-" * 30)
        correlation_df = pd.DataFrame(
            stats['correlation_matrix'], 
            columns=self.channel_names,
            index=self.channel_names
        )
        print(correlation_df.round(3))
        
        return stats

# Update the GGSL_Dataset class to support configurable channels
class GGSL_Dataset_MultiChannel(GGSL_Dataset):
    """Enhanced GGSL Dataset with configurable multi-channel support."""
    
    def __init__(self, config, csv_path=None, transforms=None, num_channels=3):
        super().__init__(config, csv_path, transforms)
        self.num_channels = num_channels
        
        # # Update mean and std for different channel configurations
        # if num_channels == 6:
            
        #     self.mean_value = [0.339397, 0.236873, 0.505158, 0.107595, 0.002220, -0.014500] # 0416 scale
        #     self.std_value  = [0.212314, 0.292032, 0.329197, 0.125507, 0.021190, 0.583026]

        #     # self.mean_value = [0.011392, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch pow
        #     # self.std_value = [0.030090, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]   
            
        #     # self.mean_value = [0.182512, 0.187020, 0.507372, 0.501423, 0.025831, 0.017252] # 6ch 2ndtry
        #     # self.std_value  = [0.188272, 0.244794, 0.296535, 0.189828, 0.534643, 0.733123]
            
        #     # self.mean_value = [0.316990, 0.216337, 0.501836, 0.451666, 0.612710, 0.843656] # ERO
        #     # self.std_value = [0.213938, 0.285839, 0.289018, 0.420396, 74.925231, 74.732706]
            
              
            
        #     # self.mean_value = [0.339545, 0.206452, 0.515560, 0.086415, 0.001191, -0.015427] # 0416 sky scale
        #     # self.std_value  = [0.212401, 0.286092, 0.338191, 0.114127, 0.020359, 0.582938]

        #     # self.mean_value = [0.182513, 0.187020, 0.507372, 0.014899, 0.025455, 0.262853] # 6ch asinh fam
        #     # self.std_value  = [0.188272, 0.244794, 0.296535, 0.093289, 0.094730, 0.047978]
            
        # Update mean and std for different channel configurations
        if num_channels == 9:
            
            self.mean_value = [0.198894, 0.020038, 0.047231, 0.025180, 0.024282, 0.027003, 0.182513, 0.507372, 0.187020]  
            self.std_value = [0.355429, 0.691650, 7.434228, 0.687494, 0.662065, 0.882627, 0.188272, 0.296535, 0.244794]
        if num_channels == 'all' or (isinstance(self.num_channels, int) and self.num_channels > 9):
            self.mean_value = [ 
                               0.18544532704114228,
                                0.1890263188922253,
                                0.5073528119877891,
                                0.5049551716039067,
                                0.011637009405990412,
                                0.21706667060382806,
                                0.021507105763886463,
                                0.1749055255195379,
                                0.07884580448283088,
                                0.025141017436927954,
                                0.01621996561400811,
                                0.0233888243779388,
                                0.025959189001210303,
                                0.2445807712958751,
                                0.3343811745152648,
                                0.204238886249775,
                                0.024200575905782087,
                                0.01023494924261971,
                                32.57983365313886,
                                0.7057067953502657,
                                0.3409324412158834]
  
            self.std_value = [
                                0.18971419748503535,
                                0.2463617740069781,
                                0.2968730895761633,
                                0.18846410174766967,
                                0.02998522037605397,
                                0.1406498871313746,
                                0.030459653897075195,
                                0.17502622950564184,
                                3.01917151972728,
                                0.6513288216259373,
                                0.7592618802010203,
                                0.6864784224927026,
                                0.8847393908219333,
                                0.20791020827819054,
                                0.43757035461526234,
                                0.5791157052296704,
                                0.708265410581812,
                                0.04338344792661475,
                                17233.34155479176,
                                0.3085866956730574,
                                0.19633275599645683
                            ]
    
    
            
        # Replace stretch pipeline with enhanced version
        self.stretch_pipeline = EnhancedStretchPipeline(self, num_channels)

################################################################################
################################################################################
################################################################################

# def get_transform(train):
#     transforms = []
#     transforms.append(ToTensor())
#     transforms.append(ConvertImageDtype(torch.float))
#     if train:
#         transforms.append(RandomHorizontalFlip(0.5))
#         transforms.append(RandomVerticalFlip(0.5))
#         # transforms.append(RandomRotation(30)) # ZIO PRETE
#     return Compose(transforms)


def get_transform(train, apply_class_specific_aug=False):
    transforms = []
    transforms.append(ToTensor())
    transforms.append(ConvertImageDtype(torch.float))
    if train:
        if apply_class_specific_aug:
            # Apply stronger augmentations for class 1 (minority class)
            transforms.append(ClassConditionalRandomHorizontalFlip(0.8, target_classes=[1]))
            transforms.append(ClassConditionalRandomVerticalFlip(0.8, target_classes=[1]))
            # Lighter augmentations for class 0 (majority class)
            transforms.append(ClassConditionalRandomHorizontalFlip(0.3, target_classes=[0]))
            transforms.append(ClassConditionalRandomVerticalFlip(0.3, target_classes=[0]))
        else:
            # Regular augmentations for all classes
            transforms.append(RandomHorizontalFlip(0.5))
            transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)

# Add the class-conditional transform classes
class ClassConditionalRandomHorizontalFlip(nn.Module):
    """Apply horizontal flip only to specific classes."""
    
    def __init__(self, p=0.5, target_classes=[1]):
        super().__init__()
        self.p = p
        self.target_classes = target_classes
    
    def forward(self, image: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if target is not None and target.item() in self.target_classes:
            if torch.rand(1) < self.p:
                image = F.hflip(image)
        return image, target

class ClassConditionalRandomVerticalFlip(nn.Module):
    """Apply vertical flip only to specific classes."""
    
    def __init__(self, p=0.5, target_classes=[1]):
        super().__init__()
        self.p = p
        self.target_classes = target_classes
    
    def forward(self, image: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if target is not None and target.item() in self.target_classes:
            if torch.rand(1) < self.p:
                image = F.vflip(image)
        return image, target

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

################################################################################

class ToTensor(nn.Module):
    """
    Convert a `PIL Image` or `ndarray` to `tensor` and scale the values accordingly.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to 
    a torch.FloatTensor of shape (C x H x W) in the float range [0.0, 1.0] if the 
    PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the `numpy.ndarray` has `dtype = np.uint8`.

    In the other cases, tensors are returned without scaling.

    This is the same as the class `ToTensor()` the can be found in the
    standard `torchvision` library, but defined in this way it also accepts a
    (optional) dictionary as second argument, where we store all the information
    on the bounding boxes and the binary masks.
    """
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

################################################################################

class ConvertImageDtype(nn.Module):
    """Convert a tensor image to the given `dtype`."""
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

################################################################################

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """Perform an horizontal flip of the image, and adjust as consequence also
    the information on the bounding boxes and the binary masks."""
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            
        return image, target

class RandomVerticalFlip(T.RandomVerticalFlip):
    """Perform a vertical flip of the image, and adjust as consequence also
    the information on the bounding boxes and the binary masks."""
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            
        return image, target

class RandomRotation(nn.Module):
    """Rotate the image by angle."""
    def __init__(self, degrees: float):
        super().__init__()
        self.degrees = degrees

    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])
        image = F.rotate(image, angle)
        return image, target

################################################################################

def create_dataloaders(config):
    """Creates training and testing DataLoaders.

    Takes in a data directory path and turns it into PyTorch Datasets and then
    into PyTorch DataLoaders.

    Args:
    config: A class containing some parameters for the configuration of the net.
        - config.BATCH_SIZE: Number of samples per batch in each of the DataLoaders.
        - config.NUM_WORKERS: An integer for number of workers per DataLoader.
        - config.DATA_PATH: Path to the directory that contains the images.

    Returns:
    A tuple of (data_loader_train, data_loader_valid, data_loader_test).
    Example usage:
        data_loader_train, data_loader_valid, data_loader_test = \
        = create_dataloaders(config=config)
    """

    # Print csv files containing the info on the simulated images (dataset)
    print('[INFO] Train CSV Dataset file:\n\t', str(config.TRAIN_DATA_CSV))
    print('[INFO] Valid CSV Dataset file:\n\t', str(config.VALID_DATA_CSV))
    print('[INFO] Test CSV Dataset file:\n\t', str(config.TEST_DATA_CSV))

    # Use our dataset and defined transformations
    dataset_train = GGSL_Dataset(config=config,
                                                csv_path=config.TRAIN_DATA_CSV, 
                                             transforms=get_transform(train=True,
                                                                      apply_class_specific_aug=True))
    dataset_valid = GGSL_Dataset(config=config,
                                                csv_path=config.VALID_DATA_CSV,
                                             transforms=get_transform(train=False))
    dataset_test  = GGSL_Dataset(config=config,
                                                csv_path=config.TEST_DATA_CSV,
                                             transforms=get_transform(train=False))

    # # Split the dataset in train, validation, and test set
    # train_perc    = 0.85 # 0.80
    # valid_perc    = 0.15 # (1.-train_perc)/2.
    
    # train_split   = int(train_perc*len(dataset))
    # valid_split   = int(valid_perc*len(dataset))
    # torch.manual_seed(42)
    # indices       = torch.randperm(len(dataset)).tolist()
    # dataset_train = torch.utils.data.Subset(dataset,       indices[:train_split])
    # dataset_valid = torch.utils.data.Subset(dataset_valid, indices[train_split:(train_split+valid_split)])
    
    print('[INFO] Number of images in Train Set =', len(dataset_train))
    print('[INFO] Number of images in Valid Set =', len(dataset_valid))
    print('[INFO] Number of images in Test  Set =', len(dataset_test))
    #print(f"Train data:\n{dataset_train}\nValid data:\n{dataset_valid}\nTest data:\n{dataset_test}")

    # Define training, validation, and test DataLoaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=config.BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid,
                                                    batch_size=config.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn)
    data_loader_test  = torch.utils.data.DataLoader(dataset_test,
                                                    batch_size=config.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=config.NUM_WORKERS,
                                                    collate_fn=collate_fn)

    return data_loader_train, data_loader_valid, data_loader_test


# Helper function to create datasets with different channel configurations
def create_multichannel_dataloaders(config, num_channels=6):
    """Create dataloaders with configurable number of channels."""
    
    print(f'[INFO] Creating {num_channels}-channel dataloaders...')
    
    dataset_train = GGSL_Dataset_MultiChannel(
        config=config,
        csv_path=config.TRAIN_DATA_CSV, 
        transforms=get_transform(train=True, apply_class_specific_aug=True),
        num_channels=num_channels
    )
    
    dataset_valid = GGSL_Dataset_MultiChannel(
        config=config,
        csv_path=config.VALID_DATA_CSV,
        transforms=get_transform(train=False),
        num_channels=num_channels
    )
    
    dataset_test = GGSL_Dataset_MultiChannel(
        config=config,
        csv_path=config.TEST_DATA_CSV,
        transforms=get_transform(train=False),
        num_channels=num_channels
    )
    
    print(f'[INFO] Train: {len(dataset_train)}, Valid: {len(dataset_valid)}, Test: {len(dataset_test)}')
    
    # Create dataloaders
    dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': True,
        'collate_fn': collate_fn
    }
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, **dataloader_kwargs
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, shuffle=False, **dataloader_kwargs
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, **dataloader_kwargs
    )
    
    return data_loader_train, data_loader_valid, data_loader_test

# train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(config)

# # call one example from the train dataloader, from the test dataloader, and from the valid dataloader
# train_example = next(iter(train_dataloader))
# valid_example = next(iter(valid_dataloader))
# test_example  = next(iter(test_dataloader))
# # Convert batch tuples to tensors for easier processing
# train_images = torch.stack(train_example[0])
# valid_images = torch.stack(valid_example[0])
# test_images = torch.stack(test_example[0])


# # print the mean and std of the entire batch in the train, valid, and test dataloaders
# print('[INFO] Train batch mean and std:', train_images.mean(), train_images.std())
# print('[INFO] Valid batch mean and std:', valid_images.mean(), valid_images.std())
# print('[INFO] Test batch mean and std:', test_images.mean(), test_images.std())

# # print the shape of the batches
# print('[INFO] Train batch shape:', train_images.shape)
# print('[INFO] Valid batch shape:', valid_images.shape)
# print('[INFO] Test batch shape:', test_images.shape)

# # Optional: print per-channel statistics
# print('\n[INFO] Per-channel statistics:')
# for i, name in enumerate(['Train', 'Valid', 'Test']):
#     images = [train_images, valid_images, test_images][i]
#     for ch, ch_name in enumerate(['MTF', 'Asinh_Low', 'Asinh_High']):
#         ch_mean = images[:, ch].mean()
#         ch_std = images[:, ch].std()
#         print(f'[INFO] {name} {ch_name} channel - mean: {ch_mean:.4f}, std: {ch_std:.4f}')


#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#--------------------------------------------------------------------------# 
# Little test here to check if the dataset is working
#--------------------------------------------------------------------------# 
################################################################################

# # Create the dataset
# dataset = GGSL_Dataset(config=config, csv_path=config.DATA_CSV, transforms=get_transform(train=False))

# data  = dataset[6] # get the first element of the dataset

# # #print example[0] and example [1] type
# # print(type(data[0])) # = <class 'torch.Tensor'>
# # print(data[0].shape) # = torch.Size([3, 100, 100])
# # print(data[1]) # = tensor(1, dtype=torch.uint8) - class of the object
# # print(data[0].mean(), data[0].std()) # 0.10511417952822158 0.17409599629508193
# # print(data[0][1].min(), data[0][1].max()) # tot = 0.0 / 1.3510160527673551 --- asinh = 


# # plot the panel with the two images side by side
# plt.figure(figsize=(18, 6))
# plt.subplot(1, 3, 1)
# plt.imshow(data[0][0].cpu().numpy(), cmap='viridis')
# plt.colorbar()
# plt.title('MTF Arcsinh Stretch')
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(data[0][1].cpu().numpy(), cmap='viridis')
# plt.colorbar()  
# plt.title('Arcsinh Low Stretch')
# plt.subplot(1, 3, 3)
# plt.imshow(data[0][2].cpu().numpy(), cmap='viridis')
# plt.colorbar()
# plt.title('Arcsinh High Stretch')
# plt.axis('off')
# plt.tight_layout()
# plt.savefig('ggsl_dataset_example_3ch.png', dpi=300)
# plt.show()  

def compute_channel_statistics(config, sample_size=None, save_results=True):
    """
    Compute mean and standard deviation for each channel in the multi-channel dataset.
    
    Args:
        config: Configuration object with dataset parameters
        sample_size: Number of samples to use (None = use all samples)
        save_results: Whether to save results to a file
    
    Returns:
        Dictionary with statistics for each channel
    """
    print("[INFO] Computing channel statistics for 9-channel dataset...")
    
    # Create dataset without transforms to get raw normalized data
    dataset = GGSL_Dataset_MultiChannel(
        config=config,
        csv_path=config.DATA_CSV,
        transforms=None,  # No transforms to get raw stretched data
        num_channels=6
    )
    
    if sample_size is None:
        sample_size = len(dataset)
    else:
        sample_size = min(sample_size, len(dataset))
    
    print(f"[INFO] Using {sample_size} samples from {len(dataset)} total samples")
    
    # Initialize accumulators for computing statistics
    channel_sums = np.zeros(6)
    channel_squared_sums = np.zeros(6)
    channel_mins = np.full(6, np.inf)
    channel_maxs = np.full(6, -np.inf)
    total_pixels_per_channel = 0
    
    # Process samples in batches to avoid memory issues
    batch_size = 1000
    num_batches = (sample_size + batch_size - 1) // batch_size
    
    print("[INFO] Processing samples...")
    
    for batch_idx in tqdm(range(num_batches), desc="Computing statistics"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, sample_size)
        
        batch_data = []
        
        # Load batch of samples
        for idx in range(start_idx, end_idx):
            try:
                # Get sample (before z-normalization)
                img_path = os.path.join(dataset.root, dataset.df_cutouts.iloc[idx, 0])
                dataset_type = dataset._identify_dataset_type(img_path)
                data_loader = dataset.data_loader_factory.create_loader(dataset_type)
                
                # Load raw data and apply stretches
                vis = data_loader.load_data(img_path)
                img = dataset.stretch_pipeline.apply_stretches(vis)
                
                # Convert to tensor format if needed
                if isinstance(img, np.ndarray):
                    if img.shape[-1] == 6:  # (H, W, 6)
                        img = np.transpose(img, (2, 0, 1))  # -> (6, H, W)

                batch_data.append(img)
                
            except Exception as e:
                print(f"[WARNING] Failed to process sample {idx}: {e}")
                continue
        
        if not batch_data:
            continue
            
        # Convert batch to numpy array
        batch_array = np.array(batch_data)  # Shape: (batch_size, 9, H, W)
        
        # Update statistics for each channel
        for ch in range(6):
            channel_data = batch_array[:, ch, :, :].flatten()
            
            # Update sums
            channel_sums[ch] += np.sum(channel_data)
            channel_squared_sums[ch] += np.sum(channel_data ** 2)
            
            # Update min/max
            channel_mins[ch] = min(channel_mins[ch], np.min(channel_data))
            channel_maxs[ch] = max(channel_maxs[ch], np.max(channel_data))
            
            # Count pixels (same for all channels)
            if ch == 0:
                total_pixels_per_channel += len(channel_data)
    
    # Compute final statistics
    channel_means = channel_sums / total_pixels_per_channel
    channel_vars = (channel_squared_sums / total_pixels_per_channel) - (channel_means ** 2)
    channel_stds = np.sqrt(np.maximum(channel_vars, 0))  # Ensure non-negative
    
    # Channel names for reference
    channel_names = [
        'Grad.Enhanced', # 'Pow_Stretch',
        'Asinh_Stretch',
        'Optimized_Adaptive',
        'Gradient_Enhanced-LogDs9',
        'Pure Edge det.', #'Structure_Boundary',
        'Faint_Feature_Enhanced'
    ]
    
    # Create results dictionary
    results = {
        'means': channel_means.tolist(),
        'stds': channel_stds.tolist(),
        'mins': channel_mins.tolist(),
        'maxs': channel_maxs.tolist(),
        'channel_names': channel_names,
        'total_samples_processed': sample_size,
        'total_pixels_per_channel': total_pixels_per_channel
    }
    
    # Print results
    print("\n" + "="*60)
    print("CHANNEL STATISTICS RESULTS")
    print("="*60)
    print(f"Total samples processed: {sample_size}")
    print(f"Total pixels per channel: {total_pixels_per_channel}")
    print("\nPer-channel statistics:")
    print("-" * 60)
    
    for i, name in enumerate(channel_names):
        print(f"Channel {i+1:2d} - {name:<25}: "
              f"Mean={channel_means[i]:.6f}, "
              f"Std={channel_stds[i]:.6f}")
        print(f"{'':>30} "
              f"Range=[{channel_mins[i]:.6f}, {channel_maxs[i]:.6f}]")
    
    # Print formatted arrays for easy copying to config
    print("\n" + "-"*60)
    print("FOR CONFIG.PY - Copy these values:")
    print("-"*60)
    print("MEAN_6CH = [", end="")
    for i, mean in enumerate(channel_means):
        if i > 0:
            print(", ", end="")
        print(f"{mean:.6f}", end="")
    print("]")
    
    print("STD_6CH  = [", end="")
    for i, std in enumerate(channel_stds):
        if i > 0:
            print(", ", end="")
        print(f"{std:.6f}", end="")
    print("]")
    
    # Save results to file
    if save_results:
        import json
        output_file = os.path.join(config.ROOT, 'channel_statistics_6ch_2ndtry.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to: {output_file}")
        
        # Also save as CSV for easy viewing
        import pandas as pd
        df_stats = pd.DataFrame({
            'Channel': [f"Ch{i+1}_{name}" for i, name in enumerate(channel_names)],
            'Mean': channel_means,
            'Std': channel_stds,
            'Min': channel_mins,
            'Max': channel_maxs
        })
        csv_file = os.path.join(config.ROOT, 'channel_statistics_6ch_2ndtry.csv')
        df_stats.to_csv(csv_file, index=False)
        print(f"[INFO] CSV saved to: {csv_file}")
    
    return results

def compute_all_channel_statistics(config, num_channels='all', sample_size=None, save_results=True, out_prefix='channel_statistics'):
    """
    Compute per-channel mean and std for the given multi-channel pipeline.
    - num_channels: 'all' to use process_all_channels, or an integer (e.g., 6, 9, 20).
    - sample_size: limit number of samples (None = all).
    - Returns a dict with means, stds, mins, maxs, channel_names.
    """
    import json
    from datetime import datetime

    # Build dataset that emits raw stretched channels (no z-norm, no transforms)
    dataset = GGSL_Dataset_MultiChannel(
        config=config,
        csv_path=config.DATA_CSV,
        transforms=None,
        num_channels=num_channels
    )

    n_total = len(dataset)
    if sample_size is None:
        sample_size = n_total
    else:
        sample_size = min(sample_size, n_total)

    print(f"[INFO] Computing channel statistics for num_channels={num_channels} on {sample_size}/{n_total} samples...")

    # Determine channel count and names from the first sample (pre z-norm)
    # We bypass __getitem__ to avoid z_normalize: load and apply stretches directly.
    def _load_stretched(idx):
        img_path = os.path.join(dataset.root, dataset.df_cutouts.iloc[idx, 0])
        dtype = dataset._identify_dataset_type(img_path)
        loader = dataset.data_loader_factory.create_loader(dtype)
        vis = loader.load_data(img_path)
        return dataset.stretch_pipeline.apply_stretches(vis)  # (H,W,C)

    # Find a valid first sample
    first_img = None
    first_idx = 0
    for i in range(n_total):
        try:
            first_img = _load_stretched(i)
            first_idx = i
            break
        except Exception as e:
            print(f"[WARN] Skipping sample {i}: {e}")
            continue
    if first_img is None:
        raise RuntimeError("Could not load any sample to infer channel count.")

    H, W, C = first_img.shape
    # Try to get channel names (only available for 'all' via processor)
    channel_names = None
    try:
        proc = getattr(dataset.stretch_pipeline, "channel_processor", None)
        if proc and hasattr(proc, "get_all_channel_names"):
            channel_names = proc.get_all_channel_names()
            if isinstance(num_channels, int) and channel_names is not None:
                channel_names = channel_names[:C]
    except Exception:
        pass
    if channel_names is None or len(channel_names) != C:
        channel_names = [f"Ch{i}" for i in range(C)]

    # Accumulators (float64 for stability)
    sums = np.zeros(C, dtype=np.float64)
    sq_sums = np.zeros(C, dtype=np.float64)
    mins = np.full(C, np.inf, dtype=np.float64)
    maxs = np.full(C, -np.inf, dtype=np.float64)
    counts = np.zeros(C, dtype=np.int64)

    # Indices to process
    rng = np.random.default_rng(seed=42)
    indices = np.arange(n_total)
    # Sample without replacement if needed
    if sample_size < n_total:
        indices = rng.choice(indices, size=sample_size, replace=False)
    else:
        indices = indices[:sample_size]

    for k, idx in enumerate(tqdm(indices, desc="Computing statistics")):
        try:
            img = _load_stretched(idx)  # (H,W,C)
            if img.shape[-1] != C:
                # Skip inconsistent shapes
                print(f"[WARN] Sample {idx} has different channel count: {img.shape[-1]} != {C}. Skipping.")
                continue

            # Flatten per-channel
            flat = img.reshape(-1, C)  # (H*W, C)
            # NaN-safe
            flat = np.nan_to_num(flat, copy=False)

            # Update accumulators
            sums += flat.sum(axis=0)
            sq_sums += np.square(flat, dtype=np.float64).sum(axis=0)
            mins = np.minimum(mins, flat.min(axis=0))
            maxs = np.maximum(maxs, flat.max(axis=0))
            counts += flat.shape[0]

        except Exception as e:
            print(f"[WARN] Skipping sample {idx} due to error: {e}")
            continue

    # Final stats
    counts_safe = np.maximum(counts, 1)
    means = sums / counts_safe
    vars_ = sq_sums / counts_safe - np.square(means)
    vars_[vars_ < 0] = 0.0
    stds = np.sqrt(vars_)

    # Results
    results = {
        "num_channels": int(C),
        "channel_names": channel_names,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "total_samples_processed": int(sample_size),
        "total_pixels_per_channel": int(counts_safe[0]) if C > 0 else 0,
    }

    # Pretty print for copy-paste
    print("\n" + "="*60)
    print(f"CHANNEL STATISTICS ({C} channels)")
    print("="*60)
    for i, name in enumerate(channel_names):
        print(f"{i:02d} {name:<24} mean={means[i]:.6f} std={stds[i]:.6f} "
              f"range=[{mins[i]:.6f}, {maxs[i]:.6f}]")

    print("\nFor config (copy/paste):")
    print("MEAN_ALL = [", ", ".join(f"{m:.6f}" for m in means), "]")
    print("STD_ALL  = [", ", ".join(f"{s:.6f}" for s in stds), "]")

    # Save
    if save_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(config.ROOT, "channel_stats")
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, f"{out_prefix}_{C}ch_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved JSON: {json_path}")

        # CSV
        try:
            import pandas as pd
            df = pd.DataFrame({
                "channel": [f"{i:02d}_{n}" for i, n in enumerate(channel_names)],
                "mean": means,
                "std": stds,
                "min": mins,
                "max": maxs,
            })
            csv_path = os.path.join(out_dir, f"{out_prefix}_{C}ch_{ts}.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Saved CSV:  {csv_path}")
        except Exception as e:
            print(f"[WARN] CSV save failed: {e}")

    return results

# Example usage and testing
if __name__ == "__main__":
    
    # Compute statistics for 9-channel dataset
    compute_stats = False
    
    if compute_stats:
        print("\n[INFO] Computing channel statistics...")
        
        # Choose one of these options:
        
        # Option 1: Fast computation with subset of samples
        stats_fast = compute_channel_statistics(config, sample_size=None, save_results=True)
        # stats_fast = compute_all_channel_statistics(config, num_channels='all', sample_size=3000, save_results=True, out_prefix='channel_statistics')
        print("\n[INFO] Fast statistics computed.")
        
    # Test 6-channel implementation
    print("[INFO] Testing 6-channel dataset...")
    
    # Create 6-channel dataset
    dataset_6ch = GGSL_Dataset_MultiChannel(
        config=config, 
        csv_path=config.DATA_CSV, 
        transforms=get_transform(train=False),
        num_channels=6
    )
    # plot the first 6-channel image
    sample_img, sample_label = dataset_6ch[9] # 4008 10390 4293 3048 8855
    print(f"6-channel sample shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")
    plt.figure(figsize=(9, 6))
    for i in range(6):
        plt.subplot(3, 3, i + 1)
        plt.imshow(sample_img[i, :, :], cmap='viridis')
        plt.title(f'Channel {i+1}')
        plt.colorbar()  
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('ggsl_6ch_field_q1_order0.png', dpi=300)
    plt.show()
    # for i in range(len(dataset_6ch)):
    #     sample_img, sample_label = dataset_6ch[i]
    #     print(f"6-channel sample shape: {sample_img.shape}")
    #     print(f"Sample label: {sample_label}")
    #     plt.figure(figsize=(9, 6))
    #     for j in range(6):
    #         plt.subplot(3, 3, j + 1)
    #         plt.imshow(sample_img[j, :, :], cmap='viridis')
    #         plt.title(f'Channel {j+1}')
    #         plt.colorbar()  
    #         plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'/dati4/mfogliardi/training/ggsl/lo_zibaldone/candidates/ggsl_6ch_field{i}.png', dpi=300)
    # # plt.show()
    
    
    # Test sample
    # sample_img, sample_label = dataset_6ch[6]
    # print(f"6-channel sample shape: {sample_img.shape}")
    # print(f"Sample label: {sample_label}")
    
    # # Run channel analysis
    # analyzer = ChannelAnalysisTools(num_channels=6)
    # analysis_results = analyzer.generate_analysis_report(dataset_6ch, sample_size=100)










nan_analysis = False

if nan_analysis == True:
    # Analyze NaN images before creating dataloaders
    print('\n[INFO] Analyzing images for NaN values...')
    dataset_temp = GGSL_Dataset(config=config, csv_path=config.TRAIN_DATA_CSV, transforms=None)
    nan_analysis = dataset_temp.analyze_nan_images(max_samples=100)

    print(f'\n[WARNING] Found {len(nan_analysis)} images with NaN values:')
    for idx, path, count, percentage in nan_analysis:
        print(f'  - Image {idx}: {count} NaNs ({percentage:.2f}%)')




def collate_fn(batch):
    return tuple(zip(*batch))


def merge_and_shuffle_csvs(csv_pairs, output_dir='/dati4/mfogliardi/training/ggsl/csv/', seed=42):
    """
    Merge pairs of CSV files and shuffle all rows, saving as new files.
    
    Args:
        csv_pairs: List of tuples, each containing (csv1_path, csv2_path, output_name)
        output_dir: Directory to save the merged CSV files
        seed: Random seed for reproducible shuffling
        
    Returns:
        List of output file paths
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(seed)
    output_paths = []
    
    for csv1_path, csv2_path, output_name in csv_pairs:
        try:
            # Read both CSV files
            df1 = pd.read_csv(csv1_path)
            df2 = pd.read_csv(csv2_path)
            
            print(f'[INFO] Merging {csv1_path} ({len(df1)} rows) with {csv2_path} ({len(df2)} rows)')
            
            # Check if columns match
            if list(df1.columns) != list(df2.columns):
                print(f'[WARNING] Column mismatch between {csv1_path} and {csv2_path}')
                print(f'  CSV1 columns: {list(df1.columns)}')
                print(f'  CSV2 columns: {list(df2.columns)}')
            
            # Concatenate the dataframes
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # Shuffle the rows
            shuffled_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            # Create output path
            output_path = os.path.join(output_dir, output_name)
            
            # Save the merged and shuffled CSV
            shuffled_df.to_csv(output_path, index=False)
            
            print(f'[INFO] Saved merged CSV: {output_path} ({len(shuffled_df)} total rows)')
            output_paths.append(output_path)
            
            # Print class distribution if there's a class column (assuming second column is class)
            if len(shuffled_df.columns) > 1:
                class_counts = shuffled_df.iloc[:, 1].value_counts()
                print(f'  Class distribution: {dict(class_counts)}')
            
        except Exception as e:
            print(f'[ERROR] Failed to merge {csv1_path} and {csv2_path}: {e}')
    
    return output_paths

# # Usage example:
# if __name__ == "__main__":
#     # Define the CSV pairs to merge
#     csv_pairs = [
#         (
#             '/dati4/mfogliardi/training/ggsl/csv/merged_test.csv',
#             '/dati4/mfogliardi/training/ggsl/csv/q1_lenses.csv',
#             'merged_test_q1.csv'
#         )]
# #         (
# #             '/dati4/mfogliardi/training/ggsl/csv/art_val.csv',
# #             '/dati4/mfogliardi/training/ggsl/csv/val.csv',
# #             'merged_val.csv'
# #         ),
# #         (
# #             '/dati4/mfogliardi/training/ggsl/csv/art_test.csv',
# #             '/dati4/mfogliardi/training/ggsl/csv/test.csv',
# #             'merged_test.csv'
# #         )
# #     ]
    
#     # Merge and shuffle the CSV files
#     print('[INFO] Starting CSV merging process...')
#     merged_paths = merge_and_shuffle_csvs(csv_pairs, seed=42)
    
#     print(f'\n[INFO] Created merged CSV files:')
#     for path in merged_paths:
#         print(f'  - {path}')


def merge_and_shuffle_multiple_csvs(csv_groups, output_dir='/dati4/mfogliardi/training/ggsl/csv/', seed=42):
    """
    Merge multiple CSV files for each split (train/val/test) and shuffle all rows.
    
    Args:
        csv_groups: List of tuples, each containing (list_of_csv_paths, output_name)
        output_dir: Directory to save the merged CSV files
        seed: Random seed for reproducible shuffling
        
    Returns:
        List of output file paths
    """

    np.random.seed(seed)
    output_paths = []
    
    for csv_paths, output_name in csv_groups:
        try:
            dataframes = []
            total_rows = 0
            
            print(f'\n[INFO] Merging {len(csv_paths)} CSV files for {output_name}:')
            
            # Read all CSV files
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    dataframes.append(df)
                    total_rows += len(df)
                    print(f'  - {csv_path}: {len(df)} rows')
                else:
                    print(f'  - [WARNING] File not found: {csv_path}')
            
            if not dataframes:
                print(f'[ERROR] No valid CSV files found for {output_name}')
                continue
            
            # Check if all dataframes have the same columns
            first_columns = list(dataframes[0].columns)
            for i, df in enumerate(dataframes[1:], 1):
                if list(df.columns) != first_columns:
                    print(f'[WARNING] Column mismatch in file {i+1}')
                    print(f'  Expected: {first_columns}')
                    print(f'  Found: {list(df.columns)}')
            
            # Concatenate all dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)
            
            # Shuffle the rows
            shuffled_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            
            # Create output path
            output_path = os.path.join(output_dir, output_name)
            
            # Save the merged and shuffled CSV
            shuffled_df.to_csv(output_path, index=False)
            
            print(f'[INFO] Saved merged CSV: {output_path} ({len(shuffled_df)} total rows)')
            output_paths.append(output_path)
            
            # Print class distribution if there's a class column (assuming second column is class)
            if len(shuffled_df.columns) > 1:
                class_counts = shuffled_df.iloc[:, 1].value_counts()
                print(f'  Class distribution: {dict(class_counts)}')
                
                # Print percentage distribution
                class_percentages = shuffled_df.iloc[:, 1].value_counts(normalize=True) * 100
                print(f'  Class percentages: {dict(class_percentages.round(2))}')
            
        except Exception as e:
            print(f'[ERROR] Failed to merge files for {output_name}: {e}')
    
    return output_paths

# Usage function for your specific use case
def create_final_merged_datasets(base_dir='/dati4/mfogliardi/training/ggsl/csv/', seed=42):
    """
    Create the final merged datasets by combining all train/val/test CSV files.
    """
    
    # Define the CSV groups to merge
    csv_groups = [
        # Training set
        ([
            os.path.join(base_dir, 'art_train.csv'),
            os.path.join(base_dir, 'q1_lenses_train.csv'),  # Assuming this exists
            os.path.join(base_dir, 'trickygirls_train.csv'),
            os.path.join(base_dir, 'train.csv')
        ], 'final_train.csv'),
        
        # Validation set
        ([
            os.path.join(base_dir, 'art_val.csv'),
            os.path.join(base_dir, 'q1_lenses_val.csv'),  # Assuming this exists
            os.path.join(base_dir, 'trickygirls_val.csv'),
            os.path.join(base_dir, 'val.csv')
        ], 'final_val.csv'),
        
        # Test set
        ([
            os.path.join(base_dir, 'art_test.csv'),
            os.path.join(base_dir, 'q1_lenses_test.csv'),  # Assuming this exists
            os.path.join(base_dir, 'trickygirls_test.csv'),
            os.path.join(base_dir, 'test.csv')
        ], 'final_test.csv')
    ]
    
    print('[INFO] Creating final merged datasets...')
    merged_paths = merge_and_shuffle_multiple_csvs(csv_groups, output_dir=base_dir, seed=seed)
    
    print(f'\n[INFO] Created final merged CSV files:')
    for path in merged_paths:
        print(f'  - {path}')
        
        # Print summary statistics for each final file
        try:
            df = pd.read_csv(path)
            print(f'    Total samples: {len(df)}')
            if len(df.columns) > 1:
                class_dist = df.iloc[:, 1].value_counts()
                print(f'    Class distribution: {dict(class_dist)}')
        except Exception as e:
            print(f'    Error reading summary: {e}')
    
    return merged_paths



# # Execute the merging
# if __name__ == "__main__":
#     create_final_merged = True
#     if create_final_merged:
#         final_paths = create_final_merged_datasets(seed=42)

# [INFO] Merging 4 CSV files for final_train.csv:
#   - /dati4/mfogliardi/training/ggsl/csv/art_train.csv: 2117 rows
#   - /dati4/mfogliardi/training/ggsl/csv/q1_lenses_train.csv: 494 rows
#   - /dati4/mfogliardi/training/ggsl/csv/trickygirls_train.csv: 4202 rows
#   - /dati4/mfogliardi/training/ggsl/csv/train.csv: 8785 rows
# [INFO] Saved merged CSV: /dati4/mfogliardi/training/ggsl/csv/final_train.csv (15598 total rows)
#   Class distribution: {0: 10856, 1: 4742}
#   Class percentages: {0: 69.6, 1: 30.4}

# [INFO] Merging 4 CSV files for final_val.csv:
#   - /dati4/mfogliardi/training/ggsl/csv/art_val.csv: 128 rows
#   - /dati4/mfogliardi/training/ggsl/csv/q1_lenses_val.csv: 30 rows
#   - /dati4/mfogliardi/training/ggsl/csv/trickygirls_val.csv: 250 rows
#   - /dati4/mfogliardi/training/ggsl/csv/val.csv: 546 rows
# [INFO] Saved merged CSV: /dati4/mfogliardi/training/ggsl/csv/final_val.csv (954 total rows)
#   Class distribution: {0: 668, 1: 286}
#   Class percentages: {0: 70.02, 1: 29.98}

# [INFO] Merging 4 CSV files for final_test.csv:
#   - /dati4/mfogliardi/training/ggsl/csv/art_test.csv: 251 rows
#   - /dati4/mfogliardi/training/ggsl/csv/q1_lenses_test.csv: 59 rows
#   - /dati4/mfogliardi/training/ggsl/csv/trickygirls_test.csv: 495 rows
#   - /dati4/mfogliardi/training/ggsl/csv/test.csv: 1066 rows
# [INFO] Saved merged CSV: /dati4/mfogliardi/training/ggsl/csv/final_test.csv (1871 total rows)
#   Class distribution: {0: 1305, 1: 566}
#   Class percentages: {0: 69.75, 1: 30.25}

# [INFO] Created final merged CSV files:
#   - /dati4/mfogliardi/training/ggsl/csv/final_train.csv
#     Total samples: 15598
#     Class distribution: {0: 10856, 1: 4742}
#   - /dati4/mfogliardi/training/ggsl/csv/final_val.csv
#     Total samples: 954
#     Class distribution: {0: 668, 1: 286}
#   - /dati4/mfogliardi/training/ggsl/csv/final_test.csv
#     Total samples: 1871
#     Class distribution: {0: 1305, 1: 566}




