"""Image processing utilities for SpectraCurve Lab."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal

import cv2
import numpy as np
from PIL import Image, ImageEnhance

ExtractionMode = Literal["auto_color", "contrast", "color", "hsv"]
ColorName = Literal["blue", "red", "green", "black", "custom"]


@dataclass
class CropBox:
    """Crop limits in pixel coordinates for the useful plot area."""

    left: int
    right: int
    top: int
    bottom: int

    def clamp(self, width: int, height: int) -> "CropBox":
        left = max(0, min(self.left, width - 1))
        right = max(left + 1, min(self.right, width))
        top = max(0, min(self.top, height - 1))
        bottom = max(top + 1, min(self.bottom, height))
        return CropBox(left=left, right=right, top=top, bottom=bottom)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array."""
    return np.asarray(image.convert("RGB"))


def crop_image(rgb: np.ndarray, crop_box: CropBox) -> np.ndarray:
    """Crop the useful plot area from an RGB image."""
    height, width = rgb.shape[:2]
    box = crop_box.clamp(width=width, height=height)
    return rgb[box.top : box.bottom, box.left : box.right]


def enhance_image(
    rgb: np.ndarray,
    contrast: float = 1.0,
    brightness: float = 1.0,
    sharpen: bool = False,
) -> np.ndarray:
    """Apply simple visual enhancement before curve extraction."""
    image = Image.fromarray(rgb)
    image = ImageEnhance.Contrast(image).enhance(float(contrast))
    image = ImageEnhance.Brightness(image).enhance(float(brightness))
    if sharpen:
        arr = np.asarray(image)
        blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
        arr = cv2.addWeighted(arr, 1.6, blur, -0.6, 0)
        return arr
    return np.asarray(image)


def remove_light_background(rgb: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Return a mask with pixels that are not close to white."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < int(threshold)


def _color_distance_mask(rgb: np.ndarray, target_rgb: Tuple[int, int, int], tolerance: int) -> np.ndarray:
    """Mask pixels near a target RGB color."""
    arr = rgb.astype(np.int16)
    target = np.array(target_rgb, dtype=np.int16).reshape(1, 1, 3)
    dist = np.linalg.norm(arr - target, axis=2)
    return dist <= int(tolerance)


def _hsv_color_mask(rgb: np.ndarray, color_name: ColorName, custom_rgb: Tuple[int, int, int], tolerance: int) -> np.ndarray:
    """Color segmentation in HSV, more robust against antialiasing and brightness changes."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # OpenCV hue ranges from 0 to 179.
    if color_name == "blue":
        mask = (h >= 85) & (h <= 135) & (s >= 35) & (v >= 30)
    elif color_name == "red":
        mask = (((h >= 0) & (h <= 12)) | ((h >= 165) & (h <= 179))) & (s >= 35) & (v >= 30)
    elif color_name == "green":
        mask = (h >= 35) & (h <= 85) & (s >= 35) & (v >= 30)
    elif color_name == "black":
        mask = v <= max(20, min(180, int(tolerance)))
    else:
        # Convert custom color to HSV and threshold hue/saturation/value locally.
        target = np.uint8([[list(custom_rgb)]])
        target_hsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV)[0, 0]
        target_h = int(target_hsv[0])
        hue_tol = max(4, min(35, int(tolerance / 4)))
        dh = np.minimum(np.abs(h.astype(int) - target_h), 180 - np.abs(h.astype(int) - target_h))
        mask = (dh <= hue_tol) & (s >= 25) & (v >= 25)

    return mask.astype(bool)


def _auto_colored_trace_mask(rgb: np.ndarray, min_saturation: int = 45, max_value: int = 250) -> np.ndarray:
    """
    Detect colored spectral traces while rejecting gray grid lines, axes and black text.

    This mode is usually the best choice for plots exported by matplotlib/plotly/origin
    because colored curves have high saturation, while grid lines and axes are gray/black
    and therefore have low saturation. It intentionally does NOT detect black curves; for
    black-only plots use contrast mode and crop tightly around the plot area.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Keep only saturated, visible pixels. This removes gray gridlines and black axes.
    mask = (s >= int(min_saturation)) & (v <= int(max_value)) & (v >= 20)

    # Remove near-background pixels common in antialiasing of very pale grid lines.
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    color_range = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    mask &= color_range >= 25
    return mask.astype(bool)


def _remove_long_straight_lines(mask: np.ndarray, row_fraction: float = 0.45, col_fraction: float = 0.45) -> np.ndarray:
    """
    Remove axis/grid-like horizontal and vertical lines.

    A row/column with too many selected pixels is more likely to be an axis/grid line than a
    spectrum trace. This helps a lot for black curves where axes are also black.
    """
    cleaned = mask.copy().astype(bool)
    height, width = cleaned.shape
    if height == 0 or width == 0:
        return cleaned

    row_hits = cleaned.sum(axis=1) / max(width, 1)
    col_hits = cleaned.sum(axis=0) / max(height, 1)

    # Use dilation around line rows/columns so antialiased grid pixels are removed too.
    rows_to_remove = np.where(row_hits >= float(row_fraction))[0]
    cols_to_remove = np.where(col_hits >= float(col_fraction))[0]
    for r in rows_to_remove:
        r0 = max(0, r - 1)
        r1 = min(height, r + 2)
        cleaned[r0:r1, :] = False
    for c in cols_to_remove:
        c0 = max(0, c - 1)
        c1 = min(width, c + 2)
        cleaned[:, c0:c1] = False
    return cleaned


def _remove_border_pixels(mask: np.ndarray, border_px: int = 2) -> np.ndarray:
    """Remove pixels on plot borders, usually axes/frame remnants."""
    cleaned = mask.copy().astype(bool)
    if border_px <= 0:
        return cleaned
    cleaned[:border_px, :] = False
    cleaned[-border_px:, :] = False
    cleaned[:, :border_px] = False
    cleaned[:, -border_px:] = False
    return cleaned


def _remove_small_components(mask: np.ndarray, min_component_area: int = 6) -> np.ndarray:
    """Remove connected components below a minimum area."""
    mask_u8 = mask.astype(np.uint8)
    if min_component_area <= 0:
        return mask_u8.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= int(min_component_area):
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def keep_relevant_components(
    mask: np.ndarray,
    max_components: int = 8,
    min_width_fraction: float = 0.02,
) -> np.ndarray:
    """
    Keep components that look like useful curve fragments.

    Score favors components with horizontal extent and area. This prevents tiny text/noise from
    contaminating the reconstructed spectrum while keeping broken curve segments.
    """
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    height, width = mask.shape
    components = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if w < max(2, int(width * float(min_width_fraction))):
            continue
        # Spectral traces usually have meaningful horizontal span. Favor width strongly.
        # This helps reject small colored marks/legends while keeping the main curve.
        score = area * (1.0 + 3.0 * w / max(width, 1))
        components.append((score, label))

    components.sort(reverse=True)
    keep_labels = {label for _, label in components[: int(max_components)]}
    cleaned = np.zeros_like(mask_u8)
    for label in keep_labels:
        cleaned[labels == label] = 1
    return cleaned.astype(bool)


def make_curve_mask(
    rgb: np.ndarray,
    mode: ExtractionMode = "contrast",
    threshold: int = 120,
    color_name: ColorName = "blue",
    color_tolerance: int = 80,
    custom_rgb: Tuple[int, int, int] = (0, 0, 255),
    remove_grid: bool = True,
    min_component_area: int = 6,
    remove_axes: bool = True,
    keep_components: bool = True,
    max_components: int = 8,
) -> np.ndarray:
    """
    Create a binary mask of candidate curve pixels.

    Parameters
    ----------
    rgb:
        RGB image array, preferably already cropped to the plot area.
    mode:
        `auto_color` detects saturated colored traces and ignores gray axes/grid lines.
        `contrast` detects dark/high-contrast pixels, `color` uses RGB distance, and `hsv`
        uses hue/saturation segmentation.
    threshold:
        Grayscale threshold for contrast mode. Lower values keep only darker pixels.
    color_name:
        Predefined curve color for color/hsv mode.
    color_tolerance:
        RGB Euclidean distance tolerance or black HSV value threshold.
    custom_rgb:
        Target RGB if color_name == "custom".
    remove_grid:
        If True, applies morphological filtering to reduce isolated grid/noise.
    min_component_area:
        Remove connected components smaller than this number of pixels.
    remove_axes:
        If True, removes long horizontal/vertical line artifacts such as axes and grid.
    keep_components:
        If True, keeps only the most relevant curve-like connected components.
    max_components:
        Maximum number of curve fragments to keep.
    """
    if mode == "auto_color":
        # Recommended default for colored spectra: removes gray grid/eixos/text automatically.
        # color_tolerance is reused as a saturation threshold in this mode.
        mask = _auto_colored_trace_mask(rgb, min_saturation=max(20, int(color_tolerance // 2)), max_value=252)
    elif mode == "contrast":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray < int(threshold)
    elif mode == "hsv":
        mask = _hsv_color_mask(rgb, color_name, custom_rgb, int(color_tolerance))
    else:
        targets = {
            "blue": (40, 90, 220),
            "red": (220, 60, 60),
            "green": (40, 160, 80),
            "black": (20, 20, 20),
            "custom": custom_rgb,
        }
        mask = _color_distance_mask(rgb, targets[color_name], int(color_tolerance))

    mask = mask.astype(bool)
    mask = _remove_border_pixels(mask, border_px=2)

    if remove_axes:
        mask = _remove_long_straight_lines(mask, row_fraction=0.42, col_fraction=0.42)

    mask_u8 = mask.astype(np.uint8)
    if remove_grid:
        # Opening removes isolated noise; closing reconnects small curve gaps.
        open_kernel = np.ones((2, 2), dtype=np.uint8)
        close_kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    mask = _remove_small_components(mask_u8.astype(bool), min_component_area=min_component_area)

    if keep_components:
        mask = keep_relevant_components(mask, max_components=max_components)

    return mask.astype(bool)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the detected mask on top of an RGB image for validation."""
    overlay = rgb.copy()
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError("Mask and image must have the same height and width.")
    overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def _hue_distance(h1: float, h2: float) -> float:
    """Circular hue distance for OpenCV hue values in [0, 179]."""
    d = abs(float(h1) - float(h2))
    return min(d, 180.0 - d)


def _hue_to_label(hue: float) -> str:
    """Approximate human-readable color label from OpenCV hue."""
    h = float(hue) % 180
    if h < 10 or h >= 165:
        return "vermelha"
    if h < 25:
        return "laranja"
    if h < 35:
        return "amarela"
    if h < 85:
        return "verde"
    if h < 135:
        return "azul"
    if h < 165:
        return "roxa"
    return "colorida"


def split_colored_curve_masks(
    rgb: np.ndarray,
    min_saturation: int = 45,
    max_value: int = 252,
    min_component_area: int = 12,
    min_width_fraction: float = 0.05,
    hue_merge_tolerance: float = 12.0,
    max_curves: int = 6,
    remove_grid: bool = True,
    remove_axes: bool = True,
) -> list[dict]:
    """
    Automatically split multiple colored curves into separate masks.

    This routine is intended for plots with curves of different colors. It first keeps only
    saturated colored pixels, removes line/noise artifacts, finds connected components, and
    then merges fragments with similar hue into curve groups. It is robust for multi-color
    spectra exported from common plotting software. Curves with exactly the same color are a
    harder problem and should be separated with the component-based mode or manually.
    """
    if rgb.ndim != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")

    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Candidate colored pixels. This rejects gray grid lines, black axes and text.
    mask = _auto_colored_trace_mask(rgb, min_saturation=int(min_saturation), max_value=int(max_value))
    mask = _remove_border_pixels(mask, border_px=2)
    if remove_axes:
        mask = _remove_long_straight_lines(mask, row_fraction=0.42, col_fraction=0.42)

    if remove_grid:
        mask_u8 = mask.astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        mask = mask_u8.astype(bool)

    mask = _remove_small_components(mask, min_component_area=min_component_area)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    groups: list[dict] = []
    min_width = max(2, int(width * float(min_width_fraction)))

    # Build components and merge fragments by hue.
    components = []
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        hgt = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        if w < min_width:
            # Small colored fragments are often legend/text/noise.
            continue
        comp_mask = labels == label
        comp_h = np.median(h[comp_mask])
        comp_s = np.median(s[comp_mask])
        comp_v = np.median(v[comp_mask])
        # Favor wide components over large but compact symbols.
        score = float(area) * (1.0 + 3.0 * w / max(width, 1))
        components.append({
            "label": label,
            "mask": comp_mask,
            "hue": float(comp_h),
            "sat": float(comp_s),
            "val": float(comp_v),
            "area": area,
            "width": w,
            "height": hgt,
            "score": score,
            "bbox": (x, y, w, hgt),
        })

    components.sort(key=lambda item: item["score"], reverse=True)

    for comp in components:
        placed = False
        for group in groups:
            if _hue_distance(comp["hue"], group["hue"]) <= float(hue_merge_tolerance):
                group["mask"] |= comp["mask"]
                total_area = group["area"] + comp["area"]
                if total_area > 0:
                    # Weighted circular approximation is acceptable for close hues.
                    group["hue"] = (group["hue"] * group["area"] + comp["hue"] * comp["area"]) / total_area
                group["area"] = total_area
                group["score"] += comp["score"]
                group["components"] += 1
                placed = True
                break
        if not placed:
            groups.append({
                "name": f"Curva {len(groups) + 1} ({_hue_to_label(comp['hue'])})",
                "color_label": _hue_to_label(comp["hue"]),
                "hue": comp["hue"],
                "area": comp["area"],
                "score": comp["score"],
                "components": 1,
                "mask": comp["mask"].copy(),
            })

    groups.sort(key=lambda item: item["score"], reverse=True)
    out = []
    for i, group in enumerate(groups[: int(max_curves)]):
        # Clean each group again.
        curve_mask = _remove_small_components(group["mask"], min_component_area=max(3, int(min_component_area // 2)))
        out.append({
            "name": f"Curva {i + 1} ({group['color_label']})",
            "color_label": group["color_label"],
            "hue": group["hue"],
            "area": int(curve_mask.sum()),
            "components": int(group["components"]),
            "mask": curve_mask.astype(bool),
        })
    return out


def split_component_curve_masks(
    base_mask: np.ndarray,
    max_curves: int = 6,
    min_component_area: int = 20,
    min_width_fraction: float = 0.10,
) -> list[dict]:
    """
    Split a binary mask into separate connected-component masks.

    Use this when multiple curves have similar colors but appear as separated components.
    It is less reliable when a single curve is broken into many pieces or when curves cross.
    """
    if base_mask.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    height, width = base_mask.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(base_mask.astype(np.uint8), connectivity=8)
    components = []
    min_width = max(2, int(width * float(min_width_fraction)))
    for label in range(1, labels_count):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_area) or w < min_width:
            continue
        score = float(area) * (1.0 + 2.5 * w / max(width, 1))
        components.append((score, label, area, w, h))
    components.sort(reverse=True)
    out = []
    for i, (_, label, area, w, h) in enumerate(components[: int(max_curves)]):
        curve_mask = labels == label
        out.append({
            "name": f"Curva {i + 1}",
            "color_label": "componente",
            "hue": None,
            "area": int(area),
            "components": 1,
            "mask": curve_mask.astype(bool),
        })
    return out


def overlay_multiple_masks(rgb: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Overlay multiple masks with distinct colors for visual validation."""
    overlay = rgb.copy()
    palette = np.array([
        [255, 0, 0],      # red
        [0, 120, 255],    # blue/orange-ish contrast depending image
        [0, 180, 0],
        [180, 0, 220],
        [255, 150, 0],
        [0, 180, 180],
    ], dtype=np.uint8)
    for i, mask in enumerate(masks):
        if mask.shape[:2] != rgb.shape[:2]:
            raise ValueError("Mask and image must have the same height and width.")
        overlay[mask.astype(bool)] = palette[i % len(palette)]
    return overlay
