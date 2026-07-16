"""Peak and band detection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_peaks(
    df: pd.DataFrame,
    mode: str = "maxima",
    prominence: float | None = None,
    distance: int = 10,
    height: float | None = None,
    max_peaks: int = 20,
) -> pd.DataFrame:
    """Detect maxima or minima in an extracted spectrum."""
    if df.empty or len(df) < 3:
        return pd.DataFrame(columns=["peak_id", "x", "y", "prominence"])

    y = df["y"].to_numpy(dtype=float)
    signal = -y if mode == "minima" else y

    kwargs = {"distance": max(1, int(distance))}
    if prominence is not None and prominence > 0:
        kwargs["prominence"] = float(prominence)
    if height is not None:
        kwargs["height"] = float(height)

    indices, props = find_peaks(signal, **kwargs)
    if len(indices) == 0:
        return pd.DataFrame(columns=["peak_id", "x", "y", "prominence"])

    prominences = props.get("prominences", np.full(len(indices), np.nan))
    result = pd.DataFrame({
        "peak_id": np.arange(1, len(indices) + 1),
        "x": df.iloc[indices]["x"].to_numpy(dtype=float),
        "y": df.iloc[indices]["y"].to_numpy(dtype=float),
        "prominence": prominences,
    })

    # Sort most relevant peaks first, then keep max_peaks.
    if "prominence" in result.columns and not result["prominence"].isna().all():
        result = result.sort_values("prominence", ascending=False)
    else:
        result = result.reindex(result["y"].abs().sort_values(ascending=False).index)

    result = result.head(int(max_peaks)).sort_values("x").reset_index(drop=True)
    result["peak_id"] = np.arange(1, len(result) + 1)
    return result
