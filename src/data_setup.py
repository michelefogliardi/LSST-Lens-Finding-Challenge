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
    
    MEAN = [0.161927, 0.158478, 0.194141] 
    STD  = [0.242562, 0.237847, 0.261295]  
    # MEAN = [0.161927, 0.158478, 0.194141, 0.189002, 0.228415]
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
        self.num_channels     = 3  # Default to 3 channels (can be modified later)
        
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

    #--------------------------------------------------------------------------#
    # PRINCIPAL STRETCHES
    #--------------------------------------------------------------------------#
    
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
        
        # Clip to [0, max] range
        return np.clip(transformed, 0,transformed.max())
    
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
        stretched = np.clip((enhanced), vmin, vmax)
        stretched = (stretched - vmin) / (vmax - vmin)

        return stretched
    
    def optimized_adaptive_stretch(self, data, clip_limits=[0.01, 0.05, 0.1]):
        """
        Optimized multi-scale adaptive histogram stretch.
        """
        
        data = np.nan_to_num(data)
        min_data = np.min(data)
        max_data = np.max(data)
        
        data = (data - min_data) / (max_data - min_data)
        
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
        if self.num_channels == 5:
            return DatasetTypeEnum.BASE_5CH
        return DatasetTypeEnum.BASE
    
    
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
    
    
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, self.df_cutouts.iloc[idx,0])
            
            # Identify dataset type and get appropriate loader
            dataset_type = self._identify_dataset_type(img_path)
            data_loader = self.data_loader_factory.create_loader(dataset_type)
            
            # Load raw data
            data = data_loader.load_data(img_path)
            
            # Apply stretch pipeline to get 3-channel image
            # img = self.stretch_pipeline.asinh_filters(data) # 3CH
            # img = self.stretch_pipeline.mtf_filters(data)
            img = self.stretch_pipeline.apply_stretches(data) # 5CH
            
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
    BASE = "base"
    NPY = "npy"
    BASE_5CH = "base_5ch"

class DataLoaderFactory:
    """Factory for creating appropriate data loaders based on dataset type."""
    
    @staticmethod
    def create_loader(dataset_type: str):
        loaders = {
            DatasetTypeEnum.BASE: BaseFitsLoader(),
            DatasetTypeEnum.NPY: NPYLoader(),
            DatasetTypeEnum.BASE_5CH: BaseFitsLoader_5CH()  
        }
        return loaders.get(dataset_type, BaseFitsLoader())

class BaseDataLoader:
    """Base class for all data loaders."""
    
    def load_data(self, img_path: str) -> np.ndarray:
        raise NotImplementedError
    
    def validate_data(self, data: np.ndarray) -> np.ndarray:
        """Common validation logic."""
        if data is None or data.size == 0:
            raise ValueError(f"Empty data array")
        return np.nan_to_num(data)
    
class BaseFitsLoader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        # hdul.info()
        try:
            g = hdul[1].data
            g_val = self.validate_data(g)
            r = hdul[2].data
            r_val = self.validate_data(r)
            i = hdul[3].data
            i_val = self.validate_data(i)
            z = hdul[4].data
            z_val = self.validate_data(z)
            y = hdul[5].data
            y_val = self.validate_data(y)
            g_r = (g_val + r_val / 2.)
            z_y = (z_val + y_val / 2.)
            # return np.stack([r_val, i_val, z_val], axis=0)  # (3, H, W)
            return np.stack([g_r, i_val, z_y], axis=0)  # (3, H, W)
        

        finally:
            hdul.close()

class BaseFitsLoader_5CH(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        hdul = fits.open(img_path)
        # hdul.info()
        try:
            g = hdul[1].data
            g_val = self.validate_data(g)
            r = hdul[2].data
            r_val = self.validate_data(r)
            i = hdul[3].data
            i_val = self.validate_data(i)
            z = hdul[4].data
            z_val = self.validate_data(z)
            y = hdul[5].data
            y_val = self.validate_data(y)


            return np.stack([g_val, r_val, i_val, z_val, y_val], axis=0)  # (5, H, W)

        finally:
            hdul.close()


        

        
            
class NPYLoader(BaseDataLoader):
    def load_data(self, img_path: str) -> np.ndarray:
        vis = np.load(img_path)
        return self.validate_data(np.squeeze(vis))

class StretchPipeline:
    """Handles all stretch operations for the 3-channel output."""
    
    def __init__(self, stretch_methods):
        self.stretch_methods = stretch_methods
    
    def apply_stretches(self, data: np.ndarray) -> np.ndarray:
        """Apply the three stretch methods and return stacked channels."""
        
        # Apply your current 2nd try stretches
        vis_mtf = self.stretch_methods.gradient_enhanced_stretch(data, alpha=0.3) # 2ND TRY ######
        vis_asinh = self.stretch_methods.asinh_stretch_ds9(data, percent=99)
        vis_clahe = self.stretch_methods.optimized_adaptive_stretch(data, clip_limits=[0.1, 0.5, 1])
        
        # Validate shapes
        assert vis_mtf.shape == vis_asinh.shape == vis_clahe.shape

        # Stack and move axis for torchvision compatibility
        img = np.stack([vis_mtf, vis_asinh, vis_clahe])
        img = np.moveaxis(img, 0, -1)  # (3, H, W) -> (H, W, 3)
        
        return img
    
    def asinh_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply the three stretch methods and return stacked channels."""
        
        # Apply your current 2nd try stretches
        r_asinh = self.stretch_methods.asinh_stretch_ds9(data[0], percent=99)
        i_asinh = self.stretch_methods.asinh_stretch_ds9(data[1], percent=99)
        z_asinh = self.stretch_methods.asinh_stretch_ds9(data[2], percent=99)
        
        
        # Validate shapes
        assert r_asinh.shape == i_asinh.shape == z_asinh.shape 

        # Stack and move axis for torchvision compatibility
        img = np.stack([r_asinh, i_asinh, z_asinh])
        img = np.moveaxis(img, 0, -1)  # (3, H, W) -> (H, W, 3)
        
        return img
    
    def mtf_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply the three stretch methods and return stacked channels."""
        
        # Apply your current 2nd try stretches
        g_asinh = self.stretch_methods.gradient_enhanced_stretch(data[0], alpha=0.3) 
        r_asinh = self.stretch_methods.gradient_enhanced_stretch(data[1], alpha=0.3)
        i_asinh = self.stretch_methods.gradient_enhanced_stretch(data[2], alpha=0.3)

        # Validate shapes
        assert g_asinh.shape == r_asinh.shape == i_asinh.shape 

        # Stack and move axis for torchvision compatibility
        img = np.stack([g_asinh, r_asinh, i_asinh])
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
            
        elif self.num_channels == 5:
            # New 5-channel implementation
            img = self.channel_processor.process_five_channels(data)

        else:
            raise ValueError(f"Unsupported number of channels: {self.num_channels}")
        
        return img

# Add the OptimizedVISChannelProcessor class
class OptimizedVISChannelProcessor:
    """5-channel VIS-only processor with minimized redundancy."""
    
    def __init__(self, stretch_methods):
        self.stretch_methods = stretch_methods
    
    def process_five_channels(self, data):
        """Process VIS data into 5 optimized channels."""

        # Apply your current 2nd try stretches
        g_asinh = self.stretch_methods.asinh_stretch_ds9(data[0], percent=99)
        r_asinh = self.stretch_methods.asinh_stretch_ds9(data[1], percent=99)
        i_asinh = self.stretch_methods.asinh_stretch_ds9(data[2], percent=99)
        z_asinh = self.stretch_methods.asinh_stretch_ds9(data[3], percent=99)
        y_asinh = self.stretch_methods.asinh_stretch_ds9(data[4], percent=99)

        channels = np.stack([r_asinh, i_asinh, z_asinh, g_asinh, y_asinh])
        img = np.moveaxis(channels, 0, -1)  # (5, H, W) -> (H, W, 5)
        return img
    
    
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
def create_multichannel_dataloaders(config, num_channels=5):
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



def compute_5channel_statistics(config, sample_size=None, save_results=True):
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
        num_channels=5
    )
    
    if sample_size is None:
        sample_size = len(dataset)
    else:
        sample_size = min(sample_size, len(dataset))
    
    print(f"[INFO] Using {sample_size} samples from {len(dataset)} total samples")
    
    # Initialize accumulators for computing statistics
    channel_sums = np.zeros(5)
    channel_squared_sums = np.zeros(5)
    channel_mins = np.full(5, np.inf)
    channel_maxs = np.full(5, -np.inf)
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
                    if img.shape[-1] == 5:  # (H, W, 5)
                        img = np.transpose(img, (2, 0, 1))  # -> (5, H, W)

                batch_data.append(img)
                
            except Exception as e:
                print(f"[WARNING] Failed to process sample {idx}: {e}")
                continue
        
        if not batch_data:
            continue
            
        # Convert batch to numpy array
        batch_array = np.array(batch_data)  # Shape: (batch_size, 9, H, W)
        
        # Update statistics for each channel
        for ch in range(5):
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
        'g asinh',
        'r asinh',
        'i asinh',
        'z asinh',
        'y asinh'
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
    print("MEAN = [", end="")
    for i, mean in enumerate(channel_means):
        if i > 0:
            print(", ", end="")
        print(f"{mean:.6f}", end="")
    print("]")
    
    print("STD  = [", end="")
    for i, std in enumerate(channel_stds):
        if i > 0:
            print(", ", end="")
        print(f"{std:.6f}", end="")
    print("]")
    
    # Save results to file
    if save_results:
        import json
        output_file = os.path.join(config.ROOT, 'channel_statistics_5ch_asinh.json')
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
        csv_file = os.path.join(config.ROOT, 'channel_statistics_5ch_asinh.csv')
        df_stats.to_csv(csv_file, index=False)
        print(f"[INFO] CSV saved to: {csv_file}")
    
    return results

def compute_3channel_statistics(config, sample_size=None, save_results=True):
    """
    Compute mean and standard deviation for each channel in the multi-channel dataset.
    
    Args:
        config: Configuration object with dataset parameters
        sample_size: Number of samples to use (None = use all samples)
        save_results: Whether to save results to a file
    
    Returns:
        Dictionary with statistics for each channel
    """
    print("[INFO] Computing channel statistics for 3-channel dataset...")
    
    # Create dataset without transforms to get raw normalized data
    dataset = GGSL_Dataset(
        config=config,
        csv_path=config.DATA_CSV,
        transforms=None,  # No transforms to get raw stretched data
    )
    
    if sample_size is None:
        sample_size = len(dataset)
    else:
        sample_size = min(sample_size, len(dataset))
    
    print(f"[INFO] Using {sample_size} samples from {len(dataset)} total samples")
    
    # Initialize accumulators for computing statistics
    channel_sums = np.zeros(3)
    channel_squared_sums = np.zeros(3)
    channel_mins = np.full(3, np.inf)
    channel_maxs = np.full(3, -np.inf)
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
                img = dataset.stretch_pipeline.asinh_filters(vis)
                
                # Convert to tensor format if needed
                if isinstance(img, np.ndarray):
                    if img.shape[-1] == 3:  # (H, W, 3)
                        img = np.transpose(img, (2, 0, 1))  # -> (3, H, W)

                batch_data.append(img)
                
            except Exception as e:
                print(f"[WARNING] Failed to process sample {idx}: {e}")
                continue
        
        if not batch_data:
            continue
            
        # Convert batch to numpy array
        batch_array = np.array(batch_data)  # Shape: (batch_size, 3, H, W)
        
        # Update statistics for each channel
        for ch in range(3):
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
        'g asinh',
        'r asinh',
        'i asinh'
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
    print("MEAN = [", end="")
    for i, mean in enumerate(channel_means):
        if i > 0:
            print(", ", end="")
        print(f"{mean:.6f}", end="")
    print("]")
    
    print("STD  = [", end="")
    for i, std in enumerate(channel_stds):
        if i > 0:
            print(", ", end="")
        print(f"{std:.6f}", end="")
    print("]")
    
    # Save results to file
    if save_results:
        import json
        output_file = os.path.join(config.ROOT, 'channel_statistics_3ch_asinh.json')
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
        csv_file = os.path.join(config.ROOT, 'channel_statistics_3ch_asinh.csv')
        df_stats.to_csv(csv_file, index=False)
        print(f"[INFO] CSV saved to: {csv_file}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    
    # Compute statistics for 9-channel dataset
    compute_stats = False
    
    if compute_stats:
        print("\n[INFO] Computing channel statistics...")
        
        # Choose one of these options:
        
        # Option 1: Fast computation with subset of samples
        # stats_fast = compute_5channel_statistics(config, sample_size=500, save_results=True)
        stats_fast = compute_3channel_statistics(config, sample_size=30000, save_results=True)
        print("\n[INFO] Fast statistics computed.")
    
    
    # # Test 5-channel implementation
    # print("[INFO] Testing 5-channel dataset...")

    # # Create 5-channel dataset
    # dataset_5ch = GGSL_Dataset_MultiChannel(
    #     config=config, 
    #     csv_path=config.DATA_CSV, 
    #     transforms=get_transform(train=False),
    #     num_channels= 5
        
    # )
    # # plot the first 5-channel image
    # sample_img, sample_label = dataset_5ch[4] 
    # print(f"5-channel sample shape: {sample_img.shape}")
    # print(f"Sample label: {sample_label}")
    # plt.figure(figsize=(12, 6))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(sample_img[i, :, :], cmap='viridis')
    #     plt.title(f'Channel {i+1}')
    #     plt.colorbar()  
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('test.png', dpi=300)
    # plt.show()
    # Test 3-channel implementation
    print("[INFO] Testing 3-channel dataset...")

    # Create 3-channel dataset
    dataset_3ch = GGSL_Dataset(
        config=config, 
        csv_path=config.DATA_CSV, 
        transforms=get_transform(train=False),
        
    )
    # plot the first 3-channel image
    sample_img, sample_label = dataset_3ch[4] 
    print(f"3-channel sample shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")
    plt.figure(figsize=(9, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(sample_img[i, :, :], cmap='viridis')
        plt.title(f'Channel {i+1}')
        plt.colorbar()  
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('test.png', dpi=300)
    plt.show()
    
    
    
    
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





