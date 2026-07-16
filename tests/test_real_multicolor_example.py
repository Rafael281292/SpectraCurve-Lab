from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils.calibration import calibrate_pixel_curve_from_points
from utils.curve_extraction import estimate_extraction_quality, extract_pixel_curve
from utils.image_processing import split_selected_color_masks
from utils.peak_detection import detect_peaks


ROOT = Path(__file__).resolve().parents[1]


def test_real_example_extracts_three_ordered_colored_curves():
    """Integration test using the multicolor Y6 example delivered with the project."""
    rgb = np.array(Image.open(ROOT / "examples" / "Fig3-a.jpg").convert("RGB"))
    x0, x1, y0, y1 = 270, 1985, 32, 1388
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1]

    specs = [
        {"name": "Y6", "rgb": [31, 119, 180]},
        {"name": "Y6-2Se", "rgb": [93, 172, 226]},
        {"name": "Y6-2Te", "rgb": [23, 67, 96]},
    ]
    masks = split_selected_color_masks(
        crop,
        specs,
        tolerance=18,
        min_component_area=10,
        remove_grid=True,
        remove_axes=True,
        max_components_per_curve=10,
    )

    assert [item["name"] for item in masks] == ["Y6", "Y6-2Se", "Y6-2Te"]
    assert all(not np.any(masks[i]["mask"] & masks[j]["mask"])
               for i in range(3) for j in range(i + 1, 3))

    main_peaks = {}
    for item in masks:
        pixels = extract_pixel_curve(
            item["mask"], aggregation="path", interpolate_missing=False,
            remove_outliers=True, resample=False,
        )
        quality = estimate_extraction_quality(item["mask"], pixels)
        assert quality.estimated_quality == "Alta"
        assert quality.coverage_fraction > 0.95

        curve = calibrate_pixel_curve_from_points(
            pixels,
            x_min_px=330 - x0,
            x_max_px=1952 - x0,
            y_min_px=1394 - y0,
            y_max_px=29 - y0,
            x_min=380.0,
            x_max=860.0,
            y_min=0.0,
            y_max=1.0,
        )
        peaks = detect_peaks(curve, prominence=0.03, distance=80, max_peaks=5)
        strongest = peaks.sort_values("prominence", ascending=False).iloc[0]
        main_peaks[item["name"]] = float(strongest["x"])

    assert 680 < main_peaks["Y6"] < 695
    assert 695 < main_peaks["Y6-2Se"] < 710
    assert 720 < main_peaks["Y6-2Te"] < 740
    assert main_peaks["Y6"] < main_peaks["Y6-2Se"] < main_peaks["Y6-2Te"]
