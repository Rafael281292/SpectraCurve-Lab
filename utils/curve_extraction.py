"""Curve extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

Aggregation = Literal["median", "mean", "min", "max", "path"]


@dataclass
class ExtractionQuality:
    n_pixels: int
    n_columns_with_curve: int
    coverage_fraction: float
    estimated_quality: str
    message: str


def _cluster_ys(ys: np.ndarray, max_gap: int = 3) -> list[float]:
    """Group y pixels in a column into vertical clusters and return cluster centroids."""
    if len(ys) == 0:
        return []
    ys = np.sort(np.asarray(ys, dtype=float))
    clusters = [[ys[0]]]
    for value in ys[1:]:
        if value - clusters[-1][-1] <= max_gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.median(cluster)) for cluster in clusters]


def _extract_by_aggregation(mask: np.ndarray, aggregation: Aggregation, min_pixels_per_column: int) -> pd.DataFrame:
    """Original extraction mode: one aggregate y value per x column."""
    height, width = mask.shape
    rows = []
    for x in range(width):
        ys = np.where(mask[:, x])[0]
        if len(ys) < min_pixels_per_column:
            rows.append((x, np.nan))
            continue

        if aggregation == "median":
            y = float(np.median(ys))
        elif aggregation == "mean":
            y = float(np.mean(ys))
        elif aggregation == "min":
            y = float(np.min(ys))
        elif aggregation == "max":
            y = float(np.max(ys))
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")
        rows.append((x, y))
    return pd.DataFrame(rows, columns=["pixel_x", "pixel_y"])


def _extract_by_continuous_path(
    mask: np.ndarray,
    min_pixels_per_column: int = 1,
    cluster_gap: int = 3,
    max_jump: float = 35.0,
    jump_weight: float = 1.0,
    gap_penalty: float = 8.0,
) -> pd.DataFrame:
    """
    Extract a smooth, continuous curve path through candidate pixels.

    Instead of taking the median of all selected pixels in each column, this routine creates one
    or more y-candidates per x-column and uses dynamic programming to choose the path with the
    smallest vertical jumps. This is much more robust when grid lines, axes, text fragments or
    multiple candidate pixels contaminate the mask.
    """
    height, width = mask.shape
    candidates: list[list[float]] = []
    columns: list[int] = []

    for x in range(width):
        ys = np.where(mask[:, x])[0]
        if len(ys) < min_pixels_per_column:
            candidates.append([])
            continue
        y_candidates = _cluster_ys(ys, max_gap=cluster_gap)
        candidates.append(y_candidates)
        if y_candidates:
            columns.append(x)

    if not columns:
        return pd.DataFrame(columns=["pixel_x", "pixel_y"])

    # If only one candidate appears in each valid column, no DP is necessary.
    if all(len(candidates[x]) <= 1 for x in columns):
        return pd.DataFrame(
            [(x, candidates[x][0]) for x in columns if candidates[x]],
            columns=["pixel_x", "pixel_y"],
        )

    # Dynamic programming across valid x columns.
    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []

    first_x = columns[0]
    dp.append(np.zeros(len(candidates[first_x]), dtype=float))
    back.append(np.full(len(candidates[first_x]), -1, dtype=int))

    for i in range(1, len(columns)):
        x_prev = columns[i - 1]
        x_cur = columns[i]
        y_prev = np.asarray(candidates[x_prev], dtype=float)
        y_cur = np.asarray(candidates[x_cur], dtype=float)
        dx_gap = max(0, x_cur - x_prev - 1)

        cost = np.empty(len(y_cur), dtype=float)
        prev_idx = np.empty(len(y_cur), dtype=int)
        for j, yc in enumerate(y_cur):
            jumps = np.abs(y_prev - yc)
            # Penalize abrupt jumps and gaps. Large jumps are not forbidden, but discouraged.
            transition = dp[-1] + jump_weight * jumps + gap_penalty * dx_gap
            transition += np.where(jumps > max_jump, (jumps - max_jump) * 2.5, 0.0)
            k = int(np.argmin(transition))
            cost[j] = transition[k]
            prev_idx[j] = k
        dp.append(cost)
        back.append(prev_idx)

    # Backtrack best path.
    idx = int(np.argmin(dp[-1]))
    path = []
    for i in range(len(columns) - 1, -1, -1):
        x = columns[i]
        y = candidates[x][idx]
        path.append((x, y))
        idx = int(back[i][idx]) if i > 0 else -1
    path.reverse()

    return pd.DataFrame(path, columns=["pixel_x", "pixel_y"])


def remove_curve_outliers(pixel_df: pd.DataFrame, window: int = 21, z_threshold: float = 4.0) -> pd.DataFrame:
    """Remove abrupt isolated y outliers using a rolling median residual."""
    if pixel_df.empty or len(pixel_df) < max(5, window):
        return pixel_df.copy()

    out = pixel_df.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(out) if len(out) % 2 == 1 else len(out) - 1)
    if window < 5:
        return out

    y = out["pixel_y"].to_numpy(dtype=float)
    med = pd.Series(y).rolling(window=window, center=True, min_periods=1).median().to_numpy()
    resid = np.abs(y - med)
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if not np.isfinite(mad) or mad == 0:
        # Fallback: remove only very large deviations from local median.
        threshold = np.nanpercentile(resid, 98)
    else:
        threshold = max(float(z_threshold) * 1.4826 * mad, np.nanpercentile(resid, 90))
    keep = resid <= threshold
    out.loc[~keep, "pixel_y"] = np.nan
    out["pixel_y"] = out["pixel_y"].interpolate(method="linear", limit_direction="both")
    return out.dropna(subset=["pixel_y"]).reset_index(drop=True)


def resample_pixel_curve(pixel_df: pd.DataFrame, width: int, step: int = 1) -> pd.DataFrame:
    """Resample a pixel curve to a regular x grid by interpolation."""
    if pixel_df.empty:
        return pixel_df.copy()

    step = max(1, int(step))
    df = pixel_df.sort_values("pixel_x").drop_duplicates("pixel_x").reset_index(drop=True)
    x = df["pixel_x"].to_numpy(dtype=float)
    y = df["pixel_y"].to_numpy(dtype=float)

    if len(x) < 2:
        return df

    x_new = np.arange(max(0, int(np.nanmin(x))), min(width - 1, int(np.nanmax(x))) + 1, step)
    y_new = np.interp(x_new, x, y)
    return pd.DataFrame({"pixel_x": x_new.astype(float), "pixel_y": y_new.astype(float)})


def extract_pixel_curve(
    mask: np.ndarray,
    aggregation: Aggregation = "path",
    min_pixels_per_column: int = 1,
    interpolate_missing: bool = True,
    remove_outliers: bool = True,
    resample: bool = True,
) -> pd.DataFrame:
    """
    Convert a binary curve mask into one y pixel per x pixel column.

    The image coordinate system is used here: x grows to the right, y grows downward.
    Calibration to scientific coordinates is done elsewhere.

    Recommended mode is `aggregation="path"`, which tracks a continuous curve and usually
    reconstructs spectra better than a simple column median.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D boolean array.")

    height, width = mask.shape
    if aggregation == "path":
        df = _extract_by_continuous_path(mask, min_pixels_per_column=min_pixels_per_column)
    else:
        df = _extract_by_aggregation(mask, aggregation=aggregation, min_pixels_per_column=min_pixels_per_column)

    if interpolate_missing:
        # Reindex to all columns so gaps are interpolated before optional resampling.
        if not df.empty:
            full = pd.DataFrame({"pixel_x": np.arange(width, dtype=float)})
            df = full.merge(df, on="pixel_x", how="left")
            df["pixel_y"] = df["pixel_y"].interpolate(method="linear", limit_direction="both")

    df = df.dropna(subset=["pixel_y"]).reset_index(drop=True)

    if remove_outliers:
        df = remove_curve_outliers(df)

    if resample:
        df = resample_pixel_curve(df, width=width, step=1)

    return df


def estimate_extraction_quality(mask: np.ndarray, pixel_curve: Optional[pd.DataFrame] = None) -> ExtractionQuality:
    """Estimate extraction quality with simple deterministic heuristics."""
    n_pixels = int(mask.sum())
    width = mask.shape[1]
    if pixel_curve is None or pixel_curve.empty:
        n_columns = 0
    else:
        n_columns = int(pixel_curve["pixel_x"].nunique())

    coverage = n_columns / max(width, 1)
    if n_pixels == 0 or coverage < 0.15:
        quality = "Baixa"
        message = "Poucos pixels foram detectados. Ajuste o threshold, a cor ou a região do gráfico."
    elif coverage < 0.55:
        quality = "Média"
        message = "A curva foi detectada parcialmente. Pode haver lacunas ou interferência de grade/eixos."
    else:
        # Add a rough smoothness check.
        if pixel_curve is not None and len(pixel_curve) > 5:
            dy = np.diff(pixel_curve["pixel_y"].to_numpy(dtype=float))
            roughness = float(np.nanmedian(np.abs(dy)))
            if roughness > mask.shape[0] * 0.10:
                quality = "Média"
                message = "A cobertura é boa, mas há saltos abruptos. Tente o modo 'Caminho contínuo', ajuste o threshold ou recorte melhor a imagem."
            else:
                quality = "Alta"
                message = "A curva apresenta boa cobertura horizontal e continuidade adequada."
        else:
            quality = "Alta"
            message = "A curva apresenta boa cobertura horizontal e deve gerar uma reconstrução consistente."

    return ExtractionQuality(
        n_pixels=n_pixels,
        n_columns_with_curve=n_columns,
        coverage_fraction=float(coverage),
        estimated_quality=quality,
        message=message,
    )


def downsample_curve(df: pd.DataFrame, max_points: int = 1200) -> pd.DataFrame:
    """Downsample a curve for responsive plotting/export when images are very wide."""
    if len(df) <= max_points:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)
