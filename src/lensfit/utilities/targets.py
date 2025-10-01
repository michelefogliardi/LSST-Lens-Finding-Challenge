#!/usr/bin/env python3
"""
Utilities to select and vectorize regression targets from lens parameter CSVs.

Given CSV columns like:
  path,img_id,Lens ID,zlens,mag_lens_g,mag_lens_r,mag_lens_i,mag_lens_z,mag_lens_y,ell_l,ell_l_PA,Rein,sh,sh_PA,srcx,srcy,mag_src_g,mag_src_r,mag_src_i,mag_src_z,mag_src_y,zsrc,ell_s,ell_s_PA,Reff_s,n_s_sers,ell_m,ell_m_PA,Reff_l,n_l_sers,vel_disp,RA,Dec

We want to use all columns as regression targets EXCEPT identifier/non-regression fields:
  - path, img_id, Lens ID, n_l_sers, RA, Dec

This module provides:
  - select_target_columns: compute the list of target columns given all columns.
  - RegressionTargetSelector: class that keeps the target column list and vectorizes rows.

Notes:
  - Values are coerced to float. Non-convertible values become NaN; you can choose a policy
    to handle them (error, fill_zero, fill_value).
  - If you already imputed -999/±inf, vectors should be clean.
"""
from __future__ import annotations
from typing import Iterable, List, Mapping, Sequence

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import math

# Default columns to exclude from regression targets
DEFAULT_EXCLUDE = [
    "path",
    "img_id",
    "Lens ID",
    "n_l_sers",
    "n_s_sers",
    "RA",
    "Dec",
]

# Precomputed normalization statistics for regression target columns.
# Order matches the target column order produced by select_target_columns(all_columns, DEFAULT_EXCLUDE)
# i.e., the original CSV column order with the DEFAULT_EXCLUDE removed.
# Columns (26):
#   ['zlens', 'mag_lens_g', 'mag_lens_r', 'mag_lens_i', 'mag_lens_z', 'mag_lens_y',
#    'ell_l', 'ell_l_PA', 'Rein', 'sh', 'sh_PA', 'srcx', 'srcy',
#    'mag_src_g', 'mag_src_r', 'mag_src_i', 'mag_src_z', 'mag_src_y',
#    'zsrc', 'ell_s', 'ell_s_PA', 'Reff_s', 'ell_m', 'ell_m_PA', 'Reff_l', 'vel_disp']
TARGET_STATS_COLUMNS: List[str] = [
    "zlens",
    "mag_lens_g",
    "mag_lens_r",
    "mag_lens_i",
    "mag_lens_z",
    "mag_lens_y",
    "ell_l",
    "ell_l_PA",
    "Rein",
    "sh",
    "sh_PA",
    "srcx",
    "srcy",
    "mag_src_g",
    "mag_src_r",
    "mag_src_i",
    "mag_src_z",
    "mag_src_y",
    "zsrc",
    "ell_s",
    "ell_s_PA",
    "Reff_s",
    "ell_m",
    "ell_m_PA",
    "Reff_l",
    "vel_disp",
]

# Mean and Std values sourced from provided stats CSV for the above columns, in order
_MEAN_VALUES = [
    0.7235545249120725,
    22.992973300751967,
    21.76841234943388,
    20.788510816014234,
    20.270878982794482,
    19.97453734051518,
    0.5561474218208695,
    89.73713018596483,
    0.9105003656052448,
    0.01944653732133647,
    90.45301754974803,
    0.005385774134640611,
    0.05553213360122507,
    24.76233310753724,
    24.452480979404548,
    24.284612879488638,
    24.164868581928957,
    24.06530881843601,
    2.3977920172199854,
    0.533820041978406,
    90.13712109054137,
    0.10943322965898476,
    0.5395661453094581,
    89.84061511861319,
    0.6348490964888803,
    241.04461965879312,
]
_STD_VALUES = [
    0.26596412306423883,
    1.1606409061133536,
    1.2772269867545272,
    1.1710854767125805,
    1.0403615937477693,
    0.944560285025306,
    0.2994803683600868,
    51.94540125277479,
    0.38913281578431314,
    0.018227255475347807,
    52.06178657319362,
    0.5446498644352256,
    0.5324682016575858,
    0.8730120915021825,
    0.7791696785615215,
    0.758931610287011,
    0.858363388505483,
    1.0080145389223178,
    0.8103747129938357,
    0.2134390309676792,
    51.940001955051166,
    0.06545149912815854,
    0.2902513960637833,
    51.94155293425428,
    0.19205509151400202,
    54.874332342582896,
]

# Expose tensors if torch is available; otherwise expose Python lists as a fallback
if TORCH_AVAILABLE:
    MEAN = torch.tensor(_MEAN_VALUES, dtype=torch.float32)
    STD = torch.tensor(_STD_VALUES, dtype=torch.float32)
else:
    MEAN = _MEAN_VALUES  # type: ignore[assignment]
    STD = _STD_VALUES    # type: ignore[assignment]


def select_target_columns(all_columns: Sequence[str], exclude: Iterable[str] | None = None) -> List[str]:
    """
    Return the list of target columns, excluding identifier/non-regression fields.

    - Preserves original column order.
    - Exact, case-sensitive matching for names in `exclude`.
    """
    excl = set(DEFAULT_EXCLUDE if exclude is None else list(exclude))
    return [c for c in all_columns if c not in excl]


def _to_float(val) -> float:
    if val is None:
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().strip("\"\'")
    if s == "":
        return float("nan")
    try:
        v = float(s)
        return v
    except Exception:
        return float("nan")


class RegressionTargetSelector:
    """
    Keep a fixed list of target column names and provide helpers to vectorize rows.

    Params:
      - columns: full ordered list of CSV columns
      - exclude: optional custom exclude list; defaults to DEFAULT_EXCLUDE
      - nan_policy: one of {"error", "fill_zero", "fill_value"}
      - fill_value: value to use when nan_policy == "fill_value"
    """

    def __init__(
        self,
        columns: Sequence[str],
        exclude: Iterable[str] | None = None,
        nan_policy: str = "fill_zero",
        fill_value: float = 0.0,
    ) -> None:
        self.all_columns = list(columns)
        self.target_columns = select_target_columns(self.all_columns, exclude)
        if nan_policy not in {"error", "fill_zero", "fill_value"}:
            raise ValueError("nan_policy must be one of {'error','fill_zero','fill_value'}")
        self.nan_policy = nan_policy
        self.fill_value = float(fill_value)

    def vectorize_mapping(self, row: Mapping[str, object]) -> list[float]:
        """Vectorize a dict-like row (e.g., pandas Series .to_dict()) into a float list in target order."""
        vec: list[float] = []
        for col in self.target_columns:
            v = _to_float(row.get(col))
            if math.isnan(v):
                if self.nan_policy == "error":
                    raise ValueError(f"Non-numeric or missing value for target column '{col}'")
                elif self.nan_policy == "fill_zero":
                    v = 0.0
                else:
                    v = self.fill_value
            vec.append(v)
        return vec

    def vectorize_sequence(self, row_values: Sequence[object]) -> list[float]:
        """
        Vectorize given raw row sequence aligned to self.all_columns.
        """
        index = {c: i for i, c in enumerate(self.all_columns)}
        vec: list[float] = []
        for col in self.target_columns:
            i = index[col]
            raw = row_values[i] if i < len(row_values) else None
            v = _to_float(raw)
            if math.isnan(v):
                if self.nan_policy == "error":
                    raise ValueError(f"Non-numeric or missing value for target column '{col}'")
                elif self.nan_policy == "fill_zero":
                    v = 0.0
                else:
                    v = self.fill_value
            vec.append(v)
        return vec

    def vectorize_tensor(self, row: Mapping[str, object] | Sequence[object]):
        """Return a torch.float32 tensor for the row targets (if torch is available)."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is not available; install torch or use vectorize_mapping/sequence")
        if isinstance(row, Mapping):
            data = self.vectorize_mapping(row)
        else:
            data = self.vectorize_sequence(row)  # type: ignore[arg-type]
        return torch.tensor(data, dtype=torch.float32)


# Convenience function for quick usage
def make_selector_from_columns(columns: Sequence[str], **kwargs) -> RegressionTargetSelector:
    return RegressionTargetSelector(columns=columns, **kwargs)

"""
Example usage (inside a Dataset):

    import pandas as pd
    from lensfit.utilities.targets import RegressionTargetSelector

    df = pd.read_csv("/path/to/merged_train_lens_wparams_vdisp_imputed.csv")
    selector = RegressionTargetSelector(columns=df.columns, nan_policy="fill_zero")

    class RegressionDataset(torch.utils.data.Dataset):
        def __init__(self, df, selector):
            self.df = df
            self.selector = selector
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # image_tensor = ...  # load your image as before
            targets = self.selector.vectorize_mapping(row.to_dict())
            targets = torch.tensor(targets, dtype=torch.float32)
            return image_tensor, targets
"""
