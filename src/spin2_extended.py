"""
Extended spin-2 conversion for ALL Position Angles.

This module provides an updated infer_spin2_pairs() function that converts
ALL four ellipticity (magnitude, PA) pairs to spin-2 components:

1. Lens ellipticity:   (ell_l, ell_l_PA)   → (ell_l, ell_l_e2)
2. Main ellipticity:   (ell_m, ell_m_PA)   → (ell_m, ell_m_e2)  
3. Source ellipticity: (ell_s, ell_s_PA)   → (ell_s, ell_s_e2)
4. Shear:              (sh, sh_PA)         → (sh, sh_e2)

USAGE:
Replace the infer_spin2_pairs() function in data_setup.py with this version.
"""

import pandas as pd
import copy
from typing import Dict, Tuple, Optional
from lensfit.utilities.targets import RegressionTargetSelector


def infer_spin2_pairs_extended(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """Infer ALL (magnitude_index, pa_index) pairs for targets to be converted to spin-2.
    
    Converts 4 pairs:
    - Lens ellipticity: (ell_l, ell_l_PA)
    - Main ellipticity: (ell_m, ell_m_PA)
    - Source ellipticity: (ell_s, ell_s_PA)
    - Shear: (sh, sh_PA)
    
    Returns:
        Dict mapping keys ('lens', 'main', 'source', 'shear') to (mag_idx, pa_idx) tuples.
    """
    df_head = pd.read_csv(csv_path, nrows=1)
    sel = RegressionTargetSelector(columns=df_head.columns, nan_policy="fill_zero")
    row = df_head.iloc[0].to_dict()
    vec = sel.vectorize_mapping(row)
    M = len(vec)

    # Try to get the selector's output names (preferred)
    names = None
    for attr in ["output_columns", "selected_columns", "feature_names", "columns", "names"]:
        if hasattr(sel, attr):
            cand = list(getattr(sel, attr))
            if len(cand) == M:
                names = cand
                break

    def idx_by_name(name: str) -> Optional[int]:
        if names is None or name is None:
            return None
        try:
            return names.index(name)
        except ValueError:
            return None

    def idx_by_perturb(key: str, delta: float = 1e-3) -> Optional[int]:
        """Find output index corresponding to CSV column `key` by perturbation."""
        if key not in row:
            return None
        base_vec = vec
        try:
            v = float(row[key])
        except Exception:
            return None
        if v != v:  # NaN check
            return None
        row2 = copy.deepcopy(row)
        row2[key] = v + delta
        try:
            vec2 = sel.vectorize_mapping(row2)
        except Exception:
            return None
        diffs = [i for i, (a, b) in enumerate(zip(base_vec, vec2)) if abs(a - b) > 1e-6]
        if len(diffs) == 1:
            return diffs[0]
        return None

    def find_pair(mag_candidates, pa_candidates, key_name):
        """Generic function to find (magnitude, PA) index pair."""
        i_pa = None
        for nm in pa_candidates:
            i_pa = idx_by_name(nm)
            if i_pa is not None:
                break
        if i_pa is None:
            for key in pa_candidates:
                i_pa = idx_by_perturb(key)
                if i_pa is not None:
                    break
        
        i_m = None
        for nm in mag_candidates:
            i_m = idx_by_name(nm)
            if i_m is not None:
                break
        if i_m is None:
            for key in mag_candidates:
                i_m = idx_by_perturb(key)
                if i_m is not None:
                    break
        
        if i_m is not None and i_pa is not None:
            return (i_m, i_pa)
        return None

    pairs: Dict[str, Tuple[int, int]] = {}

    # 1. Lens ellipticity pair
    lens_pair = find_pair(
        mag_candidates=['ell_l', 'ell_lens', 'ellipticity_l'],
        pa_candidates=['ell_l_PA', 'ell_lens_PA', 'ellipticity_l_PA', 'ell_l_pa'],
        key_name='lens'
    )
    if lens_pair:
        pairs['lens'] = lens_pair

    # 2. Main ellipticity pair
    main_pair = find_pair(
        mag_candidates=['ell_m', 'ell_main', 'ellipticity_m'],
        pa_candidates=['ell_m_PA', 'ell_main_PA', 'ellipticity_m_PA', 'ell_m_pa'],
        key_name='main'
    )
    if main_pair:
        pairs['main'] = main_pair

    # 3. Source ellipticity pair
    source_pair = find_pair(
        mag_candidates=['ell_s', 'ell_source', 'ellipticity_s'],
        pa_candidates=['ell_s_PA', 'ell_source_PA', 'ellipticity_s_PA', 'ell_s_pa'],
        key_name='source'
    )
    if source_pair:
        pairs['source'] = source_pair

    # 4. Shear pair
    shear_pair = find_pair(
        mag_candidates=['sh', 'shear'],
        pa_candidates=['sh_PA', 'shear_PA', 'sh_pa'],
        key_name='shear'
    )
    if shear_pair:
        pairs['shear'] = shear_pair

    if pairs:
        print(f"[INFO] Spin-2 pairs inferred (EXTENDED): {pairs}")
    else:
        print("[WARN] No spin-2 pairs could be inferred from CSV headers or perturbation.")
    
    return pairs


# Example usage
if __name__ == "__main__":
    # Test with your training CSV
    csv_path = '/astrodata/mfogliardi/lsst_challenge/LSST-Lens-Finding-Challenge/src/lensfit/csv/merged_train_lens_wparams_vdisp_imputed.csv'
    pairs = infer_spin2_pairs_extended(csv_path)
    
    print("\n" + "="*60)
    print("DETECTED SPIN-2 PAIRS")
    print("="*60)
    for key, (i_mag, i_pa) in pairs.items():
        print(f"{key:10s}: magnitude_idx={i_mag:2d}, pa_idx={i_pa:2d}")
    
    # Verify dimensionality
    from lensfit.utilities.targets import TARGET_STATS_COLUMNS
    print(f"\nOriginal target dimension: {len(TARGET_STATS_COLUMNS)}")
    print(f"Number of spin-2 pairs: {len(pairs)}")
    print(f"After spin-2 conversion: {len(TARGET_STATS_COLUMNS)} dimensions")
    print("(magnitude stays, PA becomes e2 - same total dimension)")
