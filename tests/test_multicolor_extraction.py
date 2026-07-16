import cv2
import numpy as np

from utils.image_processing import split_selected_color_masks


def test_manual_selected_colors_create_independent_masks():
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    x = np.arange(10, 210, dtype=np.int32)
    blue_y = (35 + 8 * np.sin(x / 18.0)).astype(np.int32)
    red_y = (82 + 7 * np.cos(x / 21.0)).astype(np.int32)
    cv2.polylines(image, [np.column_stack([x, blue_y])], False, (31, 119, 180), 3)
    cv2.polylines(image, [np.column_stack([x, red_y])], False, (214, 39, 40), 3)

    results = split_selected_color_masks(
        image,
        [
            {"name": "Curva azul", "hex": "#1F77B4"},
            {"name": "Curva vermelha", "hex": "#D62728"},
        ],
        tolerance=20,
        min_component_area=4,
        remove_grid=False,
        remove_axes=False,
    )

    assert [item["name"] for item in results] == ["Curva azul", "Curva vermelha"]
    assert results[0]["rgb"] == [31, 119, 180]
    assert results[0]["hex"] == "#1F77B4"
    assert results[1]["rgb"] == [214, 39, 40]
    assert results[1]["hex"] == "#D62728"
    assert not np.any(results[0]["mask"] & results[1]["mask"])
    assert results[0]["mask"].sum() > 300
    assert results[1]["mask"].sum() > 300
