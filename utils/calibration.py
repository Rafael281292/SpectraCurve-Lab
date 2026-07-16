"""Pixel-to-data calibration utilities."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Scale = Literal["linear", "log10"]


def _map_axis(values: np.ndarray, src_min: float, src_max: float, dst_min: float, dst_max: float, scale: Scale) -> np.ndarray:
    """Map from pixel coordinate to data coordinate for one axis."""
    if abs(float(src_max) - float(src_min)) < 1e-12:
        raise ValueError("Os dois pontos de calibração de um eixo não podem estar na mesma posição em pixel.")

    if scale == "linear":
        return dst_min + (values - src_min) * (dst_max - dst_min) / (src_max - src_min)

    if scale == "log10":
        if dst_min <= 0 or dst_max <= 0:
            raise ValueError("Escala log10 exige limites positivos no eixo.")
        log_min = np.log10(dst_min)
        log_max = np.log10(dst_max)
        mapped_log = log_min + (values - src_min) * (log_max - log_min) / (src_max - src_min)
        return 10**mapped_log

    raise ValueError(f"Escala inválida: {scale}")


def calibrate_pixel_curve(
    pixel_df: pd.DataFrame,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_scale: Scale = "linear",
    y_scale: Scale = "linear",
    invert_x: bool = False,
    invert_y: bool = False,
) -> pd.DataFrame:
    """
    Convert image pixel coordinates into calibrated scientific coordinates.

    This mode assumes the full cropped image is exactly the plot area:
    x_min/x_max are mapped to the left/right edges and y_min/y_max to the
    bottom/top edges, unless invert_x/invert_y are selected.
    """
    if pixel_df.empty:
        return pd.DataFrame(columns=["x", "y", "pixel_x", "pixel_y"])

    x0, x1 = (width - 1, 0) if invert_x else (0, width - 1)

    # If invert_y is False, high data y is at top of the plot; if True, high data y is at bottom.
    y0, y1 = (0, height - 1) if invert_y else (height - 1, 0)

    x = _map_axis(pixel_df["pixel_x"].to_numpy(dtype=float), x0, x1, x_min, x_max, x_scale)
    y = _map_axis(pixel_df["pixel_y"].to_numpy(dtype=float), y0, y1, y_min, y_max, y_scale)

    out = pd.DataFrame({
        "x": x,
        "y": y,
        "pixel_x": pixel_df["pixel_x"].to_numpy(dtype=float),
        "pixel_y": pixel_df["pixel_y"].to_numpy(dtype=float),
    })
    out = out.sort_values("x").reset_index(drop=True)
    return out


def validate_axis_points(points: dict[str, tuple[float, float] | None]) -> None:
    """Validate the four calibration points required for point-based calibration."""
    required = ["x_min", "x_max", "y_min", "y_max"]
    missing = [key for key in required if not points.get(key)]
    if missing:
        raise ValueError(
            "Calibração incompleta. Marque os quatro pontos na imagem: "
            + ", ".join(missing)
            + "."
        )

    x0 = float(points["x_min"][0])
    x1 = float(points["x_max"][0])
    y0 = float(points["y_min"][1])
    y1 = float(points["y_max"][1])

    if abs(x1 - x0) < 2:
        raise ValueError("Os pontos x mínimo e x máximo estão muito próximos no eixo horizontal.")
    if abs(y1 - y0) < 2:
        raise ValueError("Os pontos y mínimo e y máximo estão muito próximos no eixo vertical.")


def calibrate_pixel_curve_from_points(
    pixel_df: pd.DataFrame,
    x_min_px: float,
    x_max_px: float,
    y_min_px: float,
    y_max_px: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_scale: Scale = "linear",
    y_scale: Scale = "linear",
) -> pd.DataFrame:
    """
    Convert pixel coordinates into data coordinates using four user-marked axis points.

    The user marks where the numerical values x_min, x_max, y_min and y_max are located
    in the image. Only the x coordinate of the x points and the y coordinate of the y
    points are used. This is more accurate than using the full cropped image boundaries,
    because the actual plot area rarely matches the crop perfectly.
    """
    if pixel_df.empty:
        return pd.DataFrame(columns=["x", "y", "pixel_x", "pixel_y"])

    x_pixels = pixel_df["pixel_x"].to_numpy(dtype=float)
    y_pixels = pixel_df["pixel_y"].to_numpy(dtype=float)

    x = _map_axis(x_pixels, float(x_min_px), float(x_max_px), float(x_min), float(x_max), x_scale)
    y = _map_axis(y_pixels, float(y_min_px), float(y_max_px), float(y_min), float(y_max), y_scale)

    out = pd.DataFrame({
        "x": x,
        "y": y,
        "pixel_x": x_pixels,
        "pixel_y": y_pixels,
    })
    out = out.sort_values("x").reset_index(drop=True)
    return out
