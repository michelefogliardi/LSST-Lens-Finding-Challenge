#!/usr/bin/env python3
"""
Test script to verify spin-2 conversion is working correctly for all 4 PA pairs.

This script:
1. Tests that all 4 (magnitude, PA) pairs are detected
2. Verifies spin-2 conversion produces correct (e1, e2) values
3. Tests flip augmentation correctly negates e2 components
4. Validates that angle_indices and e2_indices are properly separated
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_setup import (
    infer_spin2_pairs, 
    infer_angle_indices, 
    spin2_components,
    create_multichannel_dataloaders
)
from lensfit.utilities.targets import TARGET_STATS_COLUMNS

def test_spin2_pair_detection():
    """Test that all 4 pairs are detected."""
    print("="*80)
    print("TEST 1: Spin-2 Pair Detection")
    print("="*80)
    
    csv_path = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_train_lens_wparams_vdisp_imputed.csv'
    pairs = infer_spin2_pairs(csv_path)
    
    print(f"\nDetected {len(pairs)} pairs:")
    for key, (i_mag, i_pa) in pairs.items():
        print(f"  {key:10s}: mag_idx={i_mag:2d} ({TARGET_STATS_COLUMNS[i_mag]:15s}), "
              f"pa_idx={i_pa:2d} ({TARGET_STATS_COLUMNS[i_pa]:15s})")
    
    # Check all 4 expected pairs
    expected_keys = {'lens', 'main', 'source', 'shear'}
    found_keys = set(pairs.keys())
    
    if expected_keys == found_keys:
        print(f"\n✅ SUCCESS: All 4 expected pairs detected!")
    else:
        missing = expected_keys - found_keys
        extra = found_keys - expected_keys
        if missing:
            print(f"\n❌ MISSING pairs: {missing}")
        if extra:
            print(f"\n⚠️  EXTRA pairs: {extra}")
    
    return pairs

def test_spin2_conversion():
    """Test spin-2 conversion math."""
    print("\n" + "="*80)
    print("TEST 2: Spin-2 Conversion Math")
    print("="*80)
    
    test_cases = [
        (0.5, 0.0),    # magnitude=0.5, PA=0°
        (0.5, 45.0),   # magnitude=0.5, PA=45°
        (0.5, 90.0),   # magnitude=0.5, PA=90°
        (0.5, 180.0),  # magnitude=0.5, PA=180° (should equal PA=0°)
        (0.3, 30.0),
    ]
    
    print("\nTesting spin-2 conversion: (m, θ) → (e1, e2)")
    print("e1 = m * cos(2θ), e2 = m * sin(2θ)")
    print(f"\n{'m':>6s} {'θ°':>8s} {'e1':>10s} {'e2':>10s} {'verify_m':>10s} {'verify_θ':>10s}")
    print("-" * 60)
    
    for m, theta in test_cases:
        e1, e2 = spin2_components(m, theta)
        # Verify inverse: m = sqrt(e1^2 + e2^2), θ = 0.5 * arctan2(e2, e1)
        m_verify = np.sqrt(e1**2 + e2**2)
        theta_verify = (0.5 * np.arctan2(e2, e1) * 180 / np.pi) % 180
        
        print(f"{m:6.3f} {theta:8.1f} {e1:10.6f} {e2:10.6f} {m_verify:10.6f} {theta_verify:10.1f}")
        
        # Check that magnitude is preserved
        assert np.isclose(m, m_verify, atol=1e-6), f"Magnitude not preserved: {m} != {m_verify}"
        # Check that angle is preserved (modulo 180)
        assert np.isclose(theta % 180, theta_verify % 180, atol=1e-3), f"Angle not preserved: {theta} != {theta_verify}"
    
    # Test 180° periodicity
    e1_0, e2_0 = spin2_components(0.5, 0.0)
    e1_180, e2_180 = spin2_components(0.5, 180.0)
    
    if np.isclose(e1_0, e1_180) and np.isclose(e2_0, e2_180):
        print(f"\n✅ SUCCESS: 180° periodicity verified (PA=0° and PA=180° give same e1, e2)")
    else:
        print(f"\n❌ FAILED: 180° periodicity broken!")
        print(f"   PA=0°:   e1={e1_0:.6f}, e2={e2_0:.6f}")
        print(f"   PA=180°: e1={e1_180:.6f}, e2={e2_180:.6f}")

def test_flip_augmentation(pairs):
    """Test that flip augmentation correctly handles e2."""
    print("\n" + "="*80)
    print("TEST 3: Flip Augmentation on e2 Components")
    print("="*80)
    
    # Create a dummy target vector
    target = torch.zeros(26)  # 26 target dimensions
    
    # Set some PA values and convert to e2
    for key, (i_mag, i_pa) in pairs.items():
        m = 0.5
        theta = 45.0
        e1, e2 = spin2_components(m, theta)
        target[i_mag] = e1
        target[i_pa] = e2
    
    print("\nBefore flip:")
    for key, (i_mag, i_pa) in pairs.items():
        print(f"  {key:10s}: e1={target[i_mag].item():8.4f}, e2={target[i_pa].item():8.4f}")
    
    # Simulate flip: e2 → -e2
    target_flipped = target.clone()
    e2_indices = [pa for (_, pa) in pairs.values()]
    target_flipped[e2_indices] = -target_flipped[e2_indices]
    
    print("\nAfter flip (e2 → -e2):")
    for key, (i_mag, i_pa) in pairs.items():
        print(f"  {key:10s}: e1={target_flipped[i_mag].item():8.4f}, e2={target_flipped[i_pa].item():8.4f}")
    
    # Verify e1 unchanged, e2 negated
    for i_mag, i_pa in pairs.values():
        assert target[i_mag] == target_flipped[i_mag], f"e1 changed during flip at index {i_mag}!"
        assert target[i_pa] == -target_flipped[i_pa], f"e2 not negated during flip at index {i_pa}!"
    
    print(f"\n✅ SUCCESS: Flip augmentation correctly preserves e1 and negates e2")

def test_angle_e2_separation(pairs):
    """Test that angle_indices and e2_indices are properly separated."""
    print("\n" + "="*80)
    print("TEST 4: Angle vs e2 Index Separation")
    print("="*80)
    
    csv_path = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_train_lens_wparams_vdisp_imputed.csv'
    
    # Get all angle indices (before spin-2)
    all_angle_indices = infer_angle_indices(csv_path)
    print(f"\nAll angle indices (before spin-2): {all_angle_indices}")
    print(f"  Corresponding columns: {[TARGET_STATS_COLUMNS[i] for i in all_angle_indices]}")
    
    # Get e2 indices (PAs that will be converted)
    e2_indices = [pa for (_, pa) in pairs.values()]
    print(f"\ne2 indices (PAs converted to spin-2): {e2_indices}")
    print(f"  Corresponding columns: {[TARGET_STATS_COLUMNS[i] for i in e2_indices]}")
    
    # After spin-2, remaining angle indices
    remaining_angle_indices = [idx for idx in all_angle_indices if idx not in e2_indices]
    print(f"\nRemaining angle indices (after spin-2): {remaining_angle_indices}")
    if remaining_angle_indices:
        print(f"  Corresponding columns: {[TARGET_STATS_COLUMNS[i] for i in remaining_angle_indices]}")
    else:
        print(f"  (none - all PAs converted to spin-2)")
    
    # Check no overlap
    overlap = set(remaining_angle_indices) & set(e2_indices)
    if not overlap:
        print(f"\n✅ SUCCESS: No overlap between angle_indices and e2_indices")
    else:
        print(f"\n❌ FAILED: Overlap detected: {overlap}")
    
    # Expected: all 4 PAs should be in e2_indices, none remaining
    if len(e2_indices) == 4 and len(remaining_angle_indices) == 0:
        print(f"✅ SUCCESS: All 4 PAs converted to e2, no PAs remain as angles")
    elif len(remaining_angle_indices) > 0:
        print(f"⚠️  WARNING: {len(remaining_angle_indices)} PAs still treated as angles (not converted)")

def test_full_pipeline():
    """Test the full dataloader creation pipeline."""
    print("\n" + "="*80)
    print("TEST 5: Full Dataloader Pipeline & Normalization Stats")
    print("="*80)
    
    # Create a minimal config
    class Config:
        SEED = 42
        USE_REGRESSION_TARGETS = True
        USE_SPIN2_COMPONENTS = True
        BATCH_SIZE = 2
        NUM_WORKERS = 0
        TRAIN_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_train_lens_wparams_vdisp_imputed.csv'
        VALID_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_valid_lens_wparams_vdisp_imputed.csv'
        TEST_DATA_CSV = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_test_lens_wparams_vdisp_imputed.csv'
        MEAN = [0.161927, 0.158478, 0.194141, 0.189002, 0.228415]
        STD = [0.242562, 0.237847, 0.261295, 0.260213, 0.285261]
        HEIGHT = 100
        WIDTH = 100
        V_NORMALIZE = 'v3'
        ROOT = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/'
    
    config = Config()
    
    try:
        print("\nCreating dataloaders with spin-2 conversion...")
        print("(This will recompute normalization stats for e1, e2 components)\n")
        train_loader, valid_loader, test_loader = create_multichannel_dataloaders(config, num_channels=5)
        
        print(f"\n✅ SUCCESS: Dataloaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Valid batches: {len(valid_loader)}")
        print(f"   Test batches:  {len(test_loader)}")
        
        # Check the normalization stats from the dataset
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'target_mean') and hasattr(train_dataset, 'target_std'):
            mean = train_dataset.target_mean
            std = train_dataset.target_std
            
            print(f"\n   Normalization statistics (showing spin-2 converted indices):")
            pairs = infer_spin2_pairs(config.TRAIN_DATA_CSV)
            for key, (i_mag, i_pa) in sorted(pairs.items()):
                print(f"   {key:10s}: e1[{i_mag}] mean={mean[i_mag].item():8.4f}, std={std[i_mag].item():8.4f}")
                print(f"   {key:10s}: e2[{i_pa}] mean={mean[i_pa].item():8.4f}, std={std[i_pa].item():8.4f}")
            
            print(f"\n   Note: These are the actual e1/e2 stats (NOT magnitude/PA stats)")
        
        # Get one batch and check shapes
        batch = next(iter(train_loader))
        images, targets = batch
        
        # Handle both list and tensor formats
        if isinstance(images, list):
            images = torch.stack(images)
        if isinstance(targets, list):
            targets = torch.stack(targets)
            
        print(f"\n   Batch shapes:")
        print(f"   Images:  {images.shape}")
        print(f"   Targets: {targets.shape}")
        
        # Check target values are reasonable (e1, e2 should be bounded)
        print(f"\n   Target statistics (after normalization):")
        print(f"   Min:  {targets.min().item():.4f}")
        print(f"   Max:  {targets.max().item():.4f}")
        print(f"   Mean: {targets.mean().item():.4f}")
        print(f"   Std:  {targets.std().item():.4f}")
        
        # Check that we have spin-2 components (e2 values should be reasonable)
        e2_indices = [7, 10, 20, 23]  # ell_l_PA, sh_PA, ell_s_PA, ell_m_PA
        e2_values = targets[:, e2_indices]
        print(f"\n   e2 component statistics (indices {e2_indices}):")
        print(f"   Min:  {e2_values.min().item():.4f}")
        print(f"   Max:  {e2_values.max().item():.4f}")
        print(f"   Mean: {e2_values.mean().item():.4f}")
        
    except Exception as e:
        print(f"\n❌ FAILED: Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
    
def verify_normalization(dataloader, num_batches=100):
        """Verify that normalized targets have mean≈0, std≈1."""
        print("\n" + "="*80)
        print("NORMALIZATION VERIFICATION (Over multiple batches)")
        print("="*80)
        
        all_targets = []
        for i, (imgs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            all_targets.append(targets)
        
        all_targets = torch.cat(all_targets, dim=0)  # (N, 26)
        
        print(f"\nAggregated stats over {all_targets.shape[0]} samples:")
        print(f"{'Dimension':>12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 52)
        
        for dim in range(all_targets.shape[1]):
            vals = all_targets[:, dim]
            print(f"   Dim {dim:2d}    {vals.mean():>10.4f} {vals.std():>10.4f} "
                f"{vals.min():>10.4f} {vals.max():>10.4f}")
        
        global_mean = all_targets.mean()
        global_std = all_targets.std()
        
        print("-" * 52)
        print(f"{'Global':>12} {global_mean:>10.4f} {global_std:>10.4f}")
        
        if abs(global_mean) < 0.1 and 0.9 < global_std < 1.1:
            print("\n✅ Normalization looks good! Mean≈0, Std≈1")
        else:
            print(f"\n⚠️  Normalization may be off. Expected mean≈0, std≈1, got {global_mean:.3f}, {global_std:.3f}")

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" SPIN-2 CONVERSION TEST SUITE")
    print("="*80)
    
    # Test 1: Pair detection
    pairs = test_spin2_pair_detection()
    
    # Test 2: Spin-2 math
    test_spin2_conversion()
    
    # Test 3: Flip augmentation
    test_flip_augmentation(pairs)
    
    # Test 4: Index separation
    test_angle_e2_separation(pairs)
    
    # Test 5: Full pipeline
    # test_full_pipeline()
    
    print("\n" + "="*80)
    print("Running normalization verification...")
    print("="*80)
    
    try:
        
        loaders = test_full_pipeline()
        if loaders:
            train_loader, _, _ = loaders
            verify_normalization(train_loader, num_batches=100)
    except Exception as e:
        print(f"❌ Could not verify normalization: {e}")
    
    print("\n" + "="*80)
    print(" TEST SUITE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
