"""Spectral post-processing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SpectrumType = Literal["Absorção / UV-Vis", "Raman", "FTIR"]


@dataclass
class SpectrumPreset:
    x_label: str
    y_label: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    invert_x: bool
    peak_mode: str


def get_spectrum_preset(spectrum_type: SpectrumType) -> SpectrumPreset:
    """Return default axis settings for each supported spectrum type."""
    if spectrum_type == "Absorção / UV-Vis":
        return SpectrumPreset(
            x_label="Comprimento de onda / nm",
            y_label="Absorbância",
            x_min=300.0,
            x_max=900.0,
            y_min=0.0,
            y_max=1.5,
            invert_x=False,
            peak_mode="maxima",
        )
    if spectrum_type == "Raman":
        return SpectrumPreset(
            x_label="Deslocamento Raman / cm⁻¹",
            y_label="Intensidade",
            x_min=100.0,
            x_max=3500.0,
            y_min=0.0,
            y_max=1.0,
            invert_x=False,
            peak_mode="maxima",
        )
    if spectrum_type == "FTIR":
        return SpectrumPreset(
            x_label="Número de onda / cm⁻¹",
            y_label="Transmitância ou Absorbância",
            x_min=4000.0,
            x_max=400.0,
            y_min=0.0,
            y_max=100.0,
            invert_x=True,
            peak_mode="minima",
        )
    raise ValueError(f"Unsupported spectrum type: {spectrum_type}")


def normalize_y(df: pd.DataFrame, method: str = "max") -> pd.DataFrame:
    """Normalize y values."""
    out = df.copy()
    y = out["y"].to_numpy(dtype=float)
    if len(y) == 0:
        return out

    if method == "max":
        denom = np.nanmax(np.abs(y))
        if denom != 0:
            out["y"] = y / denom
    elif method == "minmax":
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        if ymax != ymin:
            out["y"] = (y - ymin) / (ymax - ymin)
    return out


def smooth_y(df: pd.DataFrame, method: str = "moving_average", window: int = 7, polyorder: int = 2) -> pd.DataFrame:
    """Smooth y values."""
    out = df.copy()
    if out.empty or window <= 1:
        return out

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, len(out) if len(out) % 2 == 1 else len(out) - 1))

    if window < 3:
        return out

    y = out["y"].to_numpy(dtype=float)
    if method == "savitzky_golay" and window > polyorder:
        out["y"] = savgol_filter(y, window_length=window, polyorder=int(polyorder), mode="interp")
    else:
        out["y"] = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return out


def baseline_correction(df: pd.DataFrame, strength: float = 0.05) -> pd.DataFrame:
    """Apply a simple linear baseline correction between low-percentile edges."""
    out = df.copy()
    if len(out) < 5:
        return out
    y = out["y"].to_numpy(dtype=float)
    x = out["x"].to_numpy(dtype=float)

    n_edge = max(3, int(len(out) * float(strength)))
    left_y = float(np.nanmedian(y[:n_edge]))
    right_y = float(np.nanmedian(y[-n_edge:]))
    baseline = np.interp(x, [x[0], x[-1]], [left_y, right_y])
    out["y"] = y - baseline
    return out


def area_under_curve(df: pd.DataFrame) -> float:
    """Numerically integrate y(x) using the trapezoidal rule."""
    if df is None or len(df) < 2:
        return 0.0

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return 0.0

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def summarize_curve(df: pd.DataFrame) -> dict:
    """Return deterministic summary statistics for extracted spectrum."""
    if df.empty:
        return {
            "n_points": 0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "area": None,
        }
    return {
        "n_points": int(len(df)),
        "x_min": float(df["x"].min()),
        "x_max": float(df["x"].max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
        "area": area_under_curve(df),
    }
