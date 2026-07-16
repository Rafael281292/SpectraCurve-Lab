import pandas as pd

from tools.spectral_tools import SpectralContext, SpectralToolRegistry, build_tool_schemas


def make_context():
    a = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
    b = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.8, 0.0]})
    peaks = pd.DataFrame({"curve": ["A"], "peak_id": [1], "x": [1.0], "y": [1.0], "prominence": [1.0]})
    return SpectralContext(
        "Raman",
        {"A": a, "B": b},
        peaks_df=peaks,
        quality={"estimated_quality": "Alta"},
        curve_metadata={
            "A": {"ordinal": 1, "name": "A", "color_label": "azul", "rgb": [31, 119, 180], "hex": "#1F77B4"},
            "B": {"ordinal": 2, "name": "B", "color_label": "vermelha", "rgb": [214, 39, 40], "hex": "#D62728"},
        },
    )


def test_ollama_tool_schemas():
    for schema in build_tool_schemas(["A", "B"]):
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert schema["function"]["parameters"]["additionalProperties"] is False


def test_summary_and_comparison():
    registry = SpectralToolRegistry(make_context())
    assert '"ok": true' in registry.execute("get_curve_summary", {"curve_name": "A"})
    comparison = registry.execute("compare_curves", {"curve_a": "A", "curve_b": "B"})
    assert '"pearson_correlation"' in comparison


def test_unknown_curve_is_safe_error():
    registry = SpectralToolRegistry(make_context())
    result = registry.execute("get_curve_summary", {"curve_name": "C"})
    assert '"ok": false' in result


def test_curve_catalog_and_color_resolution():
    registry = SpectralToolRegistry(make_context())
    catalog = registry.execute("list_detected_curves", {})
    assert '"count": 2' in catalog
    by_color = registry.execute("get_curve_by_color", {"color_reference": "curva azul"})
    assert '"resolved_curve_name": "A"' in by_color
    by_hex = registry.execute("get_curve_by_color", {"color_reference": "#D62728"})
    assert '"resolved_curve_name": "B"' in by_hex
    by_order = registry.execute("get_curve_by_color", {"color_reference": "segunda curva"})
    assert '"resolved_curve_name": "B"' in by_order
