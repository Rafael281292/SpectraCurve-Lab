from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:  # pragma: no cover - fallback for environments without the component
    streamlit_image_coordinates = None

from utils.calibration import calibrate_pixel_curve, calibrate_pixel_curve_from_points, validate_axis_points
from utils.curve_extraction import downsample_curve, estimate_extraction_quality, extract_pixel_curve
from utils.database import insert_analysis, load_history, init_db
from utils.image_processing import (
    CropBox,
    crop_image,
    enhance_image,
    make_curve_mask,
    overlay_mask,
    overlay_multiple_masks,
    pil_to_rgb_array,
    split_colored_curve_masks,
    split_component_curve_masks,
)
from utils.mock_ai import generate_mock_diagnostic
from utils.peak_detection import detect_peaks
from utils.report import build_report
from utils.spectra_analysis import (
    baseline_correction,
    get_spectrum_preset,
    normalize_y,
    smooth_y,
    summarize_curve,
)

APP_TITLE = "SpectraCurve Lab"
APP_SUBTITLE = "Extração, reconstrução e análise de espectros de Absorção, Raman e FTIR a partir de imagens."


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


@st.cache_data(show_spinner=False)
def load_example_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_line_plot(df: pd.DataFrame, x_label: str, y_label: str, title: str):
    fig = px.line(df, x="x", y="y", title=title, labels={"x": x_label, "y": y_label})
    fig.update_layout(height=480, hovermode="x unified")
    return fig


def make_multi_line_plot(long_df: pd.DataFrame, x_label: str, y_label: str, title: str):
    """Plot multiple extracted curves using the long format: curve, x, y."""
    fig = px.line(long_df, x="x", y="y", color="curve", title=title, labels={"x": x_label, "y": y_label, "curve": "Curva"})
    fig.update_layout(height=520, hovermode="x unified")
    return fig


def _format_point(point: tuple[float, float] | None) -> str:
    if not point:
        return "não marcado"
    return f"({point[0]:.1f}, {point[1]:.1f}) px"


def draw_calibration_markers(image_rgb: np.ndarray, points: dict[str, tuple[float, float] | None]) -> Image.Image:
    """Return an RGB PIL image with the currently marked calibration points."""
    img = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = {
        "x_min": "#2563eb",
        "x_max": "#1d4ed8",
        "y_min": "#dc2626",
        "y_max": "#b91c1c",
    }
    labels = {
        "x_min": "x min",
        "x_max": "x max",
        "y_min": "y min",
        "y_max": "y max",
    }
    for key, point in points.items():
        if not point:
            continue
        x, y = float(point[0]), float(point[1])
        r = 7
        color = colors.get(key, "#22c55e")
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)
        draw.line((x - 15, y, x + 15, y), fill=color, width=2)
        draw.line((x, y - 15, x, y + 15), fill=color, width=2)
        draw.rectangle((x + 10, y - 17, x + 86, y + 4), fill="white", outline=color, width=1)
        draw.text((x + 13, y - 15), labels.get(key, key), fill=color)
    return img


def show_click_calibration_widget(
    image_rgb: np.ndarray,
    active_key: str,
    widget_key: str,
    display_width: int = 620,
):
    """Display a scaled image and capture one click using streamlit-image-coordinates.

    The image is intentionally resized before being sent to the component.
    This avoids the calibration canvas becoming larger than the browser viewport.
    Click coordinates are then converted back to coordinates in the original cropped image.
    """
    if streamlit_image_coordinates is None:
        st.error("O componente streamlit-image-coordinates não está instalado. Rode: pip install streamlit-image-coordinates")
        return None

    points = st.session_state.calibration_points
    marked = draw_calibration_markers(image_rgb, points)
    original_w, original_h = marked.size

    display_w = int(max(280, min(display_width, original_w)))
    scale = display_w / original_w
    display_h = max(1, int(original_h * scale))
    display_img = marked.resize((display_w, display_h), Image.Resampling.LANCZOS)

    st.caption(f"Imagem exibida em {display_w} × {display_h} px. Coordenadas convertidas automaticamente para o recorte original de {original_w} × {original_h} px.")

    coords = streamlit_image_coordinates(display_img, key=widget_key)
    if coords is None:
        return None

    x = float(coords["x"]) / scale
    y = float(coords["y"]) / scale
    return active_key, (x, y)




def derive_plot_roi_from_axis_points(
    points: dict[str, tuple[float, float] | None],
    width: int,
    height: int,
    padding_px: int = 0,
) -> tuple[int, int, int, int]:
    """Derive the useful plot rectangle from the four user-marked axis points.

    The horizontal plot limits come from the x coordinates of x_min/x_max.
    The vertical plot limits come from the y coordinates of y_min/y_max.
    Returns (left, right, top, bottom) in the coordinate system of the displayed/cropped image.
    """
    validate_axis_points(points)

    x0 = float(points["x_min"][0])
    x1 = float(points["x_max"][0])
    y0 = float(points["y_min"][1])
    y1 = float(points["y_max"][1])

    left = int(np.floor(min(x0, x1))) - int(padding_px)
    right = int(np.ceil(max(x0, x1))) + int(padding_px)
    top = int(np.floor(min(y0, y1))) - int(padding_px)
    bottom = int(np.ceil(max(y0, y1))) + int(padding_px)

    left = max(0, min(left, width - 2))
    right = max(left + 2, min(right, width))
    top = max(0, min(top, height - 2))
    bottom = max(top + 2, min(bottom, height))

    return left, right, top, bottom


def shift_axis_points(
    points: dict[str, tuple[float, float] | None],
    dx: int,
    dy: int,
) -> dict[str, tuple[float, float] | None]:
    """Shift calibration points after cropping the extraction region."""
    shifted: dict[str, tuple[float, float] | None] = {}
    for key, point in points.items():
        if point is None:
            shifted[key] = None
        else:
            shifted[key] = (float(point[0]) - dx, float(point[1]) - dy)
    return shifted


def curves_dict_to_long_df(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert {curve_name: df[x,y,...]} to a single long DataFrame."""
    frames = []
    for name, df in curves.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["curve"] = name
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["curve", "x", "y", "pixel_x", "pixel_y"])
    return pd.concat(frames, ignore_index=True)


def curves_dict_to_wide_df(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert extracted curves to a wide table with one y column per curve."""
    wide = None
    for name, df in curves.items():
        if df is None or df.empty:
            continue
        tmp = df[["x", "y"]].copy().rename(columns={"y": name})
        if wide is None:
            wide = tmp
        else:
            wide = pd.merge(wide, tmp, on="x", how="outer")
    if wide is None:
        return pd.DataFrame()
    return wide.sort_values("x").reset_index(drop=True)


def show_metric_cards(summary: dict, peaks_df: pd.DataFrame | None, quality_label: str | None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pontos extraídos", summary.get("n_points", 0) or 0)
    c2.metric("Picos/bandas", 0 if peaks_df is None else len(peaks_df))
    area = summary.get("area", None)
    c3.metric("Área aproximada", "—" if area is None else f"{area:.3g}")
    c4.metric("Qualidade", quality_label or "—")


with st.sidebar:
    st.title("📈 SpectraCurve Lab")
    st.caption("Protótipo sem IA real: interpretação e recomendações são simuladas.")
    st.divider()
    st.markdown("**Fluxo sugerido**")
    st.markdown(
        """
1. Envie a imagem do espectro  
2. Escolha Absorção, Raman ou FTIR  
3. Ajuste a região útil do gráfico  
4. Calibre os eixos  
5. Extraia a curva  
6. Detecte picos e gere relatório  
"""
    )
    st.divider()
    st.info("Para melhores resultados, use imagens com fundo claro, uma curva principal e eixos bem definidos.")

st.title(APP_TITLE)
st.write(APP_SUBTITLE)

# Session state defaults
for key, value in {
    "image_rgb": None,
    "filename": None,
    "crop_rgb": None,
    "mask": None,
    "overlay": None,
    "curve_df": pd.DataFrame(),
    "multi_curves": {},
    "multi_curve_long_df": pd.DataFrame(),
    "multi_curve_mode": False,
    "multi_qualities": {},
    "peaks_df": pd.DataFrame(),
    "diagnostic": "",
    "report": "",
    "quality": None,
    "last_saved_signature": None,
    "calibration_points": {"x_min": None, "x_max": None, "y_min": None, "y_max": None},
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


tabs = st.tabs(
    [
        "Início",
        "Upload e configuração",
        "Pré-processamento",
        "Calibração e extração",
        "Espectro reconstruído",
        "Picos/Bandas",
        "Diagnóstico simulado",
        "Relatório",
        "Histórico",
    ]
)

with tabs[0]:
    st.header("Objetivo da aplicação")
    st.write(
        """
O **SpectraCurve Lab** é um protótipo para digitalizar espectros científicos disponíveis apenas como imagens.
Ele extrai a curva, converte os pixels em coordenadas reais, reconstrói o espectro, detecta picos/bandas e gera um relatório técnico.

Nesta versão, **nenhum LLM ou modelo generativo foi integrado**. A seção de diagnóstico é simulada para demonstrar onde um agente de IA poderia atuar no futuro.
"""
    )

    st.subheader("Modos suportados")
    c1, c2, c3 = st.columns(3)
    c1.success("Absorção / UV-Vis\n\nλmax, bandas, área e comparação com espectros teóricos.")
    c2.info("Raman\n\nPicos vibracionais, intensidades relativas e comparação com espectros calculados.")
    c3.warning("FTIR\n\nBandas características, mínimos de transmitância e eixo x invertido.")

    st.subheader("Entregas do protótipo")
    st.markdown(
        """
- Upload de imagem de espectro
- Calibração manual dos eixos
- Extração por contraste ou cor
- Reconstrução interativa com Plotly
- Suavização, normalização e baseline simples
- Detecção de picos/bandas
- Exportação CSV
- Relatório técnico em TXT
- Histórico local com SQLite
"""
    )

with tabs[1]:
    st.header("Upload e configuração do espectro")

    example_paths = {
        "Nenhum": None,
        "Exemplo — Absorção": "examples/absorption_example.png",
        "Exemplo — Raman": "examples/raman_example.png",
        "Exemplo — FTIR": "examples/ftir_example.png",
    }

    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded = st.file_uploader("Envie uma imagem do espectro", type=["png", "jpg", "jpeg", "webp"])
    with col_b:
        example_choice = st.selectbox("Ou carregue uma imagem de exemplo", list(example_paths.keys()))

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.session_state.image_rgb = pil_to_rgb_array(image)
        st.session_state.filename = uploaded.name
    elif example_paths[example_choice] is not None:
        image = load_example_image(example_paths[example_choice])
        st.session_state.image_rgb = pil_to_rgb_array(image)
        st.session_state.filename = Path(example_paths[example_choice]).name

    spectrum_type = st.selectbox("Tipo de espectro", ["Absorção / UV-Vis", "Raman", "FTIR"])
    preset = get_spectrum_preset(spectrum_type)

    st.session_state.spectrum_type = spectrum_type
    st.session_state.preset = preset

    if st.session_state.image_rgb is not None:
        rgb = st.session_state.image_rgb
        st.image(rgb, caption=f"Imagem carregada: {st.session_state.filename} | Resolução: {rgb.shape[1]} × {rgb.shape[0]} px", use_container_width=True)
    else:
        st.info("Envie uma imagem ou selecione um exemplo para começar.")

with tabs[2]:
    st.header("Pré-processamento e região útil do gráfico")

    if st.session_state.image_rgb is None:
        st.info("Carregue uma imagem na aba 'Upload e configuração'.")
    else:
        rgb = st.session_state.image_rgb
        height, width = rgb.shape[:2]

        st.subheader("1. Recorte da área útil")
        st.caption("Ajuste os limites para deixar apenas a região interna do gráfico. Isso evita que texto, legenda e eixos atrapalhem a extração.")

        col1, col2, col3, col4 = st.columns(4)
        left_pct = col1.slider("Esquerda (%)", 0, 40, 8)
        right_pct = col2.slider("Direita (%)", 60, 100, 94)
        top_pct = col3.slider("Topo (%)", 0, 40, 10)
        bottom_pct = col4.slider("Base (%)", 60, 100, 88)

        crop_box = CropBox(
            left=int(width * left_pct / 100),
            right=int(width * right_pct / 100),
            top=int(height * top_pct / 100),
            bottom=int(height * bottom_pct / 100),
        )
        cropped = crop_image(rgb, crop_box)

        st.subheader("2. Melhorias visuais")
        c1, c2, c3 = st.columns(3)
        contrast = c1.slider("Contraste", 0.5, 3.0, 1.2, 0.1)
        brightness = c2.slider("Brilho", 0.5, 2.0, 1.0, 0.1)
        sharpen = c3.checkbox("Realçar contornos", value=False)

        processed = enhance_image(cropped, contrast=contrast, brightness=brightness, sharpen=sharpen)
        st.session_state.crop_rgb = processed
        st.session_state.crop_box = crop_box

        c_left, c_right = st.columns(2)
        c_left.image(cropped, caption="Área recortada", use_container_width=True)
        c_right.image(processed, caption="Imagem após pré-processamento", use_container_width=True)

with tabs[3]:
    st.header("Calibração e extração da curva")

    if st.session_state.crop_rgb is None:
        st.info("Configure o recorte na aba 'Pré-processamento'.")
    else:
        preset = st.session_state.get("preset", get_spectrum_preset("Absorção / UV-Vis"))
        spectrum_type = st.session_state.get("spectrum_type", "Absorção / UV-Vis")
        crop_rgb = st.session_state.crop_rgb
        h, w = crop_rgb.shape[:2]

        st.subheader("1. Calibração dos eixos")
        st.caption(
            "Agora você pode marcar na imagem onde ficam x mínimo, x máximo, y mínimo e y máximo, "
            "e digitar o valor real correspondente a cada ponto. Esse modo é mais preciso do que assumir "
            "que o recorte coincide exatamente com a área do gráfico."
        )

        calibration_mode = st.radio(
            "Modo de calibração",
            ["points", "crop_edges"],
            format_func=lambda x: {
                "points": "Marcar min/máx de cada eixo na imagem",
                "crop_edges": "Usar bordas do recorte como limites",
            }[x],
            horizontal=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        x_min = col1.number_input("Valor real em x mínimo", value=float(preset.x_min), format="%.6f")
        x_max = col2.number_input("Valor real em x máximo", value=float(preset.x_max), format="%.6f")
        y_min = col3.number_input("Valor real em y mínimo", value=float(preset.y_min), format="%.6f")
        y_max = col4.number_input("Valor real em y máximo", value=float(preset.y_max), format="%.6f")

        col5, col6 = st.columns(2)
        x_scale = col5.selectbox("Escala x", ["linear", "log10"])
        y_scale = col6.selectbox("Escala y", ["linear", "log10"])

        # As opções de inversão só são necessárias quando usamos as bordas do recorte.
        # No modo por pontos, a direção do eixo é definida pelos pontos marcados e pelos valores digitados.
        invert_x = False
        invert_y = False

        if calibration_mode == "points":
            st.markdown("**Marcação dos pontos de calibração**")
            st.caption(
                "Escolha o ponto que deseja marcar, depois clique diretamente na imagem. "
                "Para x mínimo/x máximo, clique no local do respectivo tick no eixo x. "
                "Para y mínimo/y máximo, clique no local do respectivo tick no eixo y."
            )

            point_key = st.radio(
                "Ponto que será marcado no próximo clique",
                ["x_min", "x_max", "y_min", "y_max"],
                format_func=lambda x: {
                    "x_min": "x mínimo",
                    "x_max": "x máximo",
                    "y_min": "y mínimo",
                    "y_max": "y máximo",
                }[x],
                horizontal=True,
            )

            display_width = st.slider(
                "Largura da imagem para marcação",
                min_value=350,
                max_value=900,
                value=620,
                step=50,
                help="Reduza este valor se a imagem estiver maior que a tela. A calibração continua correta porque os cliques são convertidos para a escala original.",
            )

            point_rows = []
            for label, internal_key in [
                ("x mínimo", "x_min"),
                ("x máximo", "x_max"),
                ("y mínimo", "y_min"),
                ("y máximo", "y_max"),
            ]:
                pt = st.session_state.calibration_points.get(internal_key)
                point_rows.append({
                    "Ponto": label,
                    "Status": "marcado" if pt else "não marcado",
                    "Coordenada no recorte": _format_point(pt),
                })
            st.dataframe(pd.DataFrame(point_rows), use_container_width=True, hide_index=True)

            b1, b2 = st.columns([1, 3])
            with b1:
                if st.button("Limpar pontos marcados"):
                    st.session_state.calibration_points = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
                    st.rerun()
            with b2:
                st.caption("Clique na imagem reduzida abaixo. Use o controle de largura para aproximar ou afastar a visualização.")

            clicked = show_click_calibration_widget(
                crop_rgb,
                point_key,
                widget_key=f"calib_{point_key}_{display_width}",
                display_width=display_width,
            )
            if clicked is not None:
                key_clicked, point = clicked
                st.session_state.calibration_points[key_clicked] = point
                st.success(f"Ponto {key_clicked} marcado em {_format_point(point)}. Selecione o próximo ponto e clique novamente.")

            with st.expander("Como marcar corretamente", expanded=False):
                st.markdown(
                    """
- **x mínimo**: clique exatamente na posição horizontal do menor valor do eixo x.
- **x máximo**: clique exatamente na posição horizontal do maior valor do eixo x.
- **y mínimo**: clique exatamente na posição vertical do menor valor do eixo y.
- **y máximo**: clique exatamente na posição vertical do maior valor do eixo y.
- Para FTIR, você pode digitar `x mínimo = 4000` e `x máximo = 400` se o gráfico estiver invertido. Não precisa marcar opção extra de inversão.
"""
                )
        else:
            col7, col8 = st.columns(2)
            invert_x = col7.checkbox("Eixo x invertido", value=bool(preset.invert_x))
            invert_y = col8.checkbox("Eixo y invertido", value=False)
            st.info(
                "Neste modo, a aplicação assume que a borda esquerda/direita e inferior/superior do recorte "
                "coincidem exatamente com os limites do gráfico. Se a reconstrução ficar deslocada, use o modo por marcação."
            )

        # Região usada na extração.
        # Por padrão, usa a imagem pré-processada inteira.
        # No modo por pontos, pode ser recortada automaticamente usando as posições marcadas nos eixos.
        source_rgb = crop_rgb
        active_calibration_points = st.session_state.calibration_points
        source_roi = (0, w, 0, h)  # left, right, top, bottom no sistema do recorte atual
        auto_axis_roi = False

        if calibration_mode == "points":
            st.markdown("**Região útil para extração**")
            auto_axis_roi = st.checkbox(
                "Delimitar automaticamente a área útil usando os pontos marcados nos eixos",
                value=True,
                help=(
                    "Quando ativado, a aplicação usa x mínimo/x máximo para definir as bordas esquerda/direita "
                    "e y mínimo/y máximo para definir topo/base da região de extração. Assim, você não precisa "
                    "ajustar manualmente o recorte com precisão na aba anterior."
                ),
            )
            axis_roi_padding = st.slider(
                "Margem extra ao redor da área útil (px)",
                min_value=0,
                max_value=80,
                value=4,
                step=2,
                help="Use uma margem pequena. Margens grandes podem incluir legenda, texto, eixo ou grade fora da área útil.",
            )

            if auto_axis_roi:
                try:
                    left_roi, right_roi, top_roi, bottom_roi = derive_plot_roi_from_axis_points(
                        st.session_state.calibration_points,
                        width=w,
                        height=h,
                        padding_px=axis_roi_padding,
                    )
                    source_rgb = crop_rgb[top_roi:bottom_roi, left_roi:right_roi].copy()
                    active_calibration_points = shift_axis_points(
                        st.session_state.calibration_points,
                        dx=left_roi,
                        dy=top_roi,
                    )
                    source_roi = (left_roi, right_roi, top_roi, bottom_roi)

                    st.success(
                        f"Região útil definida pelos pontos: esquerda={left_roi}, direita={right_roi}, "
                        f"topo={top_roi}, base={bottom_roi}."
                    )
                    st.image(
                        source_rgb,
                        caption="Região útil definida automaticamente pelos pontos dos eixos",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(
                        "A região útil ainda não pôde ser definida automaticamente. "
                        f"Marque os quatro pontos dos eixos primeiro. Detalhe: {exc}"
                    )

        source_h, source_w = source_rgb.shape[:2]

        st.subheader("2. Máscara da curva")
        st.caption("Para gráficos com várias curvas, use o modo múltiplas curvas. O melhor caso é quando cada curva tem uma cor diferente.")

        mode_col1, mode_col2 = st.columns([1, 2])
        extraction_scope = mode_col1.radio(
            "Modo de extração",
            ["single", "multi_color", "multi_component"],
            format_func=lambda x: {
                "single": "Uma curva",
                "multi_color": "Múltiplas curvas por cor",
                "multi_component": "Múltiplas curvas por componentes",
            }[x],
            help=(
                "Use 'Múltiplas curvas por cor' para curvas azul/vermelha/verde/etc. "
                "Use 'componentes' somente quando as curvas tiverem a mesma cor e estiverem separadas."
            ),
        )
        max_curves = mode_col2.slider("Máximo de curvas a detectar", 1, 10, 5)

        c1, c2, c3, c4 = st.columns(4)
        extraction_method = c1.selectbox(
            "Método",
            ["auto_color", "hsv", "color", "contrast"],
            format_func=lambda x: {
                "auto_color": "Automático para curva colorida",
                "contrast": "Contraste",
                "color": "Cor RGB",
                "hsv": "Cor HSV",
            }[x],
            help="Use 'Automático para curva colorida' quando a curva for azul, vermelha ou verde e a grade/eixos forem cinza/preto.",
        )
        threshold = c2.slider("Threshold", 0, 255, 130, help="Usado principalmente no modo Contraste. Para curvas coloridas, prefira o modo automático ou HSV.")
        color_name = c3.selectbox("Cor da curva", ["blue", "red", "green", "black", "custom"], format_func=lambda x: {"blue": "Azul", "red": "Vermelho", "green": "Verde", "black": "Preto", "custom": "Personalizada"}[x])
        color_tolerance = c4.slider("Tolerância/saturação", 5, 180, 90, help="No modo automático, controla a saturação mínima; em HSV/RGB controla a tolerância de cor.")

        custom_rgb = (0, 0, 255)
        if color_name == "custom":
            color_hex = st.color_picker("Cor personalizada", "#0000ff")
            custom_rgb = tuple(int(color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

        c5, c6, c7 = st.columns(3)
        remove_grid = c5.checkbox("Reduzir grade/ruído fino", value=True)
        min_area = c6.slider("Área mínima de componente", 0, 200, 12 if extraction_scope != "single" else 6)
        aggregation = c7.selectbox("Rastreamento da curva", ["path", "median", "mean", "min", "max"], format_func=lambda x: {"path": "Caminho contínuo", "median": "Mediana por coluna", "mean": "Média por coluna", "min": "Pixel superior", "max": "Pixel inferior"}[x])

        c8, c9 = st.columns(2)
        remove_axes = c8.checkbox("Remover eixos/grade longos", value=True)
        keep_components = c9.checkbox("Manter só componentes relevantes", value=True, disabled=extraction_scope != "single")

        st.subheader("3. Pós-processamento dos dados")
        p1, p2, p3, p4 = st.columns(4)
        apply_normalization = p1.checkbox("Normalizar y", value=False)
        normalization_method = p1.selectbox("Método de normalização", ["max", "minmax"])
        apply_smoothing = p2.checkbox("Suavizar", value=False, help="Deixe desligado primeiro. Ligue apenas após conferir se a máscara está correta.")
        smoothing_method = p2.selectbox("Método de suavização", ["moving_average", "savitzky_golay"])
        smoothing_window = p3.slider("Janela de suavização", 3, 51, 7, step=2)
        apply_baseline = p4.checkbox("Corrigir baseline simples", value=False)

        def calibrate_and_process(pixel_df: pd.DataFrame) -> pd.DataFrame:
            """Calibrate a pixel curve using the selected calibration mode and apply post-processing."""
            if calibration_mode == "points":
                points = active_calibration_points
                validate_axis_points(points)
                curve_df_local = calibrate_pixel_curve_from_points(
                    pixel_df,
                    x_min_px=points["x_min"][0],
                    x_max_px=points["x_max"][0],
                    y_min_px=points["y_min"][1],
                    y_max_px=points["y_max"][1],
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    x_scale=x_scale,
                    y_scale=y_scale,
                )
            else:
                curve_df_local = calibrate_pixel_curve(
                    pixel_df,
                    width=source_w,
                    height=source_h,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    invert_x=invert_x,
                    invert_y=invert_y,
                )

            if apply_baseline:
                curve_df_local = baseline_correction(curve_df_local)
            if apply_smoothing:
                curve_df_local = smooth_y(curve_df_local, method=smoothing_method, window=smoothing_window)
            if apply_normalization:
                curve_df_local = normalize_y(curve_df_local, method=normalization_method)
            return curve_df_local

        run_extraction = st.button("Extrair curva(s)", type="primary")

        if run_extraction:
            try:
                if extraction_scope == "multi_color":
                    curve_masks_info = split_colored_curve_masks(
                        source_rgb,
                        min_saturation=max(20, int(color_tolerance // 2)),
                        min_component_area=max(1, int(min_area)),
                        min_width_fraction=0.04,
                        hue_merge_tolerance=12.0,
                        max_curves=max_curves,
                        remove_grid=remove_grid,
                        remove_axes=remove_axes,
                    )
                    if not curve_masks_info:
                        raise ValueError("Nenhuma curva colorida foi detectada. Tente reduzir a tolerância/saturação, recortar melhor a imagem ou usar extração por contraste/componentes.")
                    masks = [item["mask"] for item in curve_masks_info]
                    overlay = overlay_multiple_masks(source_rgb, masks)

                    curves = {}
                    qualities = {}
                    for item in curve_masks_info:
                        pixel_df = extract_pixel_curve(
                            item["mask"],
                            aggregation=aggregation,
                            interpolate_missing=True,
                            remove_outliers=True,
                            resample=True,
                        )
                        quality = estimate_extraction_quality(item["mask"], pixel_df)
                        curve_df_i = calibrate_and_process(pixel_df)
                        curves[item["name"]] = curve_df_i
                        qualities[item["name"]] = quality

                    long_df = curves_dict_to_long_df(curves)
                    first_curve = next(iter(curves.values())) if curves else pd.DataFrame()
                    combined_mask = np.zeros((source_h, source_w), dtype=bool)
                    for m in masks:
                        combined_mask |= m
                    combined_quality = estimate_extraction_quality(combined_mask, first_curve)

                    st.session_state.mask = combined_mask
                    st.session_state.overlay = overlay
                    st.session_state.curve_df = first_curve
                    st.session_state.multi_curves = curves
                    st.session_state.multi_curve_long_df = long_df
                    st.session_state.multi_curve_mode = True
                    st.session_state.multi_qualities = qualities
                    st.session_state.quality = combined_quality
                    st.session_state.peaks_df = pd.DataFrame()
                    st.success(f"{len(curves)} curva(s) detectada(s) por cor.")

                elif extraction_scope == "multi_component":
                    base_mask = make_curve_mask(
                        source_rgb,
                        mode=extraction_method,
                        threshold=threshold,
                        color_name=color_name,
                        color_tolerance=color_tolerance,
                        custom_rgb=custom_rgb,
                        remove_grid=remove_grid,
                        min_component_area=max(1, int(min_area)),
                        remove_axes=remove_axes,
                        keep_components=False,
                    )
                    curve_masks_info = split_component_curve_masks(
                        base_mask,
                        max_curves=max_curves,
                        min_component_area=max(1, int(min_area)),
                        min_width_fraction=0.08,
                    )
                    if not curve_masks_info:
                        raise ValueError("Nenhum componente com aparência de curva foi detectado. Ajuste threshold/cor, recorte ou área mínima.")
                    masks = [item["mask"] for item in curve_masks_info]
                    overlay = overlay_multiple_masks(source_rgb, masks)

                    curves = {}
                    qualities = {}
                    for item in curve_masks_info:
                        pixel_df = extract_pixel_curve(
                            item["mask"],
                            aggregation=aggregation,
                            interpolate_missing=True,
                            remove_outliers=True,
                            resample=True,
                        )
                        quality = estimate_extraction_quality(item["mask"], pixel_df)
                        curve_df_i = calibrate_and_process(pixel_df)
                        curves[item["name"]] = curve_df_i
                        qualities[item["name"]] = quality

                    long_df = curves_dict_to_long_df(curves)
                    first_curve = next(iter(curves.values())) if curves else pd.DataFrame()
                    combined_mask = np.zeros((source_h, source_w), dtype=bool)
                    for m in masks:
                        combined_mask |= m
                    combined_quality = estimate_extraction_quality(combined_mask, first_curve)

                    st.session_state.mask = combined_mask
                    st.session_state.overlay = overlay
                    st.session_state.curve_df = first_curve
                    st.session_state.multi_curves = curves
                    st.session_state.multi_curve_long_df = long_df
                    st.session_state.multi_curve_mode = True
                    st.session_state.multi_qualities = qualities
                    st.session_state.quality = combined_quality
                    st.session_state.peaks_df = pd.DataFrame()
                    st.success(f"{len(curves)} curva(s) detectada(s) por componentes.")

                else:
                    mask = make_curve_mask(
                        source_rgb,
                        mode=extraction_method,
                        threshold=threshold,
                        color_name=color_name,
                        color_tolerance=color_tolerance,
                        custom_rgb=custom_rgb,
                        remove_grid=remove_grid,
                        min_component_area=min_area,
                        remove_axes=remove_axes,
                        keep_components=keep_components,
                    )
                    overlay = overlay_mask(source_rgb, mask)
                    pixel_df = extract_pixel_curve(
                        mask,
                        aggregation=aggregation,
                        interpolate_missing=True,
                        remove_outliers=True,
                        resample=True,
                    )
                    quality = estimate_extraction_quality(mask, pixel_df)
                    curve_df = calibrate_and_process(pixel_df)

                    st.session_state.mask = mask
                    st.session_state.overlay = overlay
                    st.session_state.curve_df = curve_df
                    st.session_state.multi_curves = {}
                    st.session_state.multi_curve_long_df = pd.DataFrame()
                    st.session_state.multi_curve_mode = False
                    st.session_state.quality = quality
                    st.session_state.peaks_df = pd.DataFrame()
                    st.success("Curva extraída com sucesso.")

                st.session_state.extraction_settings = {
                    "Modo de extração": extraction_scope,
                    "Método": extraction_method,
                    "Threshold": threshold,
                    "Cor": color_name,
                    "Tolerância de cor": color_tolerance,
                    "Rastreamento": aggregation,
                    "Remover eixos/grade": remove_axes,
                    "Componentes relevantes": keep_components,
                    "Máximo de curvas": max_curves,
                    "Normalização": apply_normalization,
                    "Suavização": apply_smoothing,
                    "Baseline": apply_baseline,
                    "Região útil automática": bool(auto_axis_roi),
                    "ROI usado": source_roi,
                }
                axis_settings = {
                    "Modo de calibração": calibration_mode,
                    "x mínimo": x_min,
                    "x máximo": x_max,
                    "y mínimo": y_min,
                    "y máximo": y_max,
                    "Escala x": x_scale,
                    "Escala y": y_scale,
                    "Rótulo x": preset.x_label,
                    "Rótulo y": preset.y_label,
                }
                if calibration_mode == "points":
                    axis_settings.update({
                        "x mínimo px": _format_point(st.session_state.calibration_points.get("x_min")),
                        "x máximo px": _format_point(st.session_state.calibration_points.get("x_max")),
                        "y mínimo px": _format_point(st.session_state.calibration_points.get("y_min")),
                        "y máximo px": _format_point(st.session_state.calibration_points.get("y_max")),
                    })
                else:
                    axis_settings.update({
                        "Eixo x invertido": invert_x,
                        "Eixo y invertido": invert_y,
                    })
                st.session_state.axis_settings = axis_settings
            except Exception as exc:
                st.error(f"Erro durante a extração: {exc}")

        if st.session_state.overlay is not None:
            q = st.session_state.quality
            c_left, c_right = st.columns(2)
            c_left.image(source_rgb, caption="Imagem usada para extração", use_container_width=True)
            c_right.image(st.session_state.overlay, caption="Máscara sobreposta em vermelho", use_container_width=True)
            st.info(f"Qualidade estimada: **{q.estimated_quality}** — {q.message}")
            st.warning(
                "Validação obrigatória: a máscara vermelha deve cobrir somente a curva. "
                "Se a máscara cobrir eixos, grade, legenda ou texto, a reconstrução ficará errada. "
                "Nesse caso, ajuste o recorte e use o método 'Automático para curva colorida' ou 'Cor HSV'."
            )

with tabs[4]:
    st.header("Espectro reconstruído")
    curve_df = st.session_state.curve_df
    multi_mode = bool(st.session_state.get("multi_curve_mode", False))
    multi_curves = st.session_state.get("multi_curves", {})
    long_df = st.session_state.get("multi_curve_long_df", pd.DataFrame())

    if curve_df is None or curve_df.empty:
        st.info("Extraia uma curva primeiro.")
    else:
        preset = st.session_state.get("preset", get_spectrum_preset("Absorção / UV-Vis"))

        if multi_mode and multi_curves:
            summary = {"n_points": int(len(long_df)), "area": None}
            show_metric_cards(summary, st.session_state.get("peaks_df"), st.session_state.quality.estimated_quality if st.session_state.quality else None)
            st.success(f"{len(multi_curves)} curva(s) extraída(s).")
            st.plotly_chart(
                make_multi_line_plot(long_df, preset.x_label, preset.y_label, "Espectros reconstruídos a partir da imagem"),
                use_container_width=True,
            )

            st.subheader("Dados extraídos")
            table_mode = st.radio("Formato da tabela", ["long", "wide"], horizontal=True, format_func=lambda x: "Longo" if x == "long" else "Largo")
            if table_mode == "long":
                st.dataframe(long_df[["curve", "x", "y", "pixel_x", "pixel_y"]], use_container_width=True)
                st.download_button(
                    "Baixar dados de todas as curvas em CSV",
                    data=dataframe_to_csv_bytes(long_df[["curve", "x", "y", "pixel_x", "pixel_y"]]),
                    file_name="spectracurve_multi_curves_long.csv",
                    mime="text/csv",
                )
            else:
                wide_df = curves_dict_to_wide_df(multi_curves)
                st.dataframe(wide_df, use_container_width=True)
                st.download_button(
                    "Baixar tabela larga em CSV",
                    data=dataframe_to_csv_bytes(wide_df),
                    file_name="spectracurve_multi_curves_wide.csv",
                    mime="text/csv",
                )
        else:
            summary = summarize_curve(curve_df)
            show_metric_cards(summary, st.session_state.get("peaks_df"), st.session_state.quality.estimated_quality if st.session_state.quality else None)
            st.plotly_chart(make_line_plot(curve_df, preset.x_label, preset.y_label, "Espectro reconstruído a partir da imagem"), use_container_width=True)

            st.download_button(
                "Baixar dados extraídos em CSV",
                data=dataframe_to_csv_bytes(curve_df[["x", "y"]]),
                file_name="spectracurve_extracted_data.csv",
                mime="text/csv",
            )

with tabs[5]:
    st.header("Detecção de picos e bandas")

    curve_df = st.session_state.curve_df
    multi_mode = bool(st.session_state.get("multi_curve_mode", False))
    multi_curves = st.session_state.get("multi_curves", {})

    if curve_df is None or curve_df.empty:
        st.info("Extraia uma curva primeiro.")
    else:
        preset = st.session_state.get("preset", get_spectrum_preset("Absorção / UV-Vis"))
        spectrum_type = st.session_state.get("spectrum_type", "Absorção / UV-Vis")

        if multi_mode and multi_curves:
            selected_curve_name = st.selectbox("Selecionar curva para análise de picos", list(multi_curves.keys()))
            curve_for_peaks = multi_curves[selected_curve_name]
        else:
            selected_curve_name = "Curva única"
            curve_for_peaks = curve_df

        st.caption("Para FTIR em transmitância, use 'mínimos'. Para absorbância/Raman/UV-Vis, use 'máximos'.")
        c1, c2, c3, c4 = st.columns(4)
        default_mode = "minima" if spectrum_type == "FTIR" else "maxima"
        peak_mode = c1.selectbox("Tipo de banda", ["maxima", "minima"], index=0 if default_mode == "maxima" else 1, format_func=lambda x: "Máximos" if x == "maxima" else "Mínimos")
        prominence = c2.number_input("Proeminência mínima", min_value=0.0, value=0.02, step=0.01, format="%.4f")
        distance = c3.slider("Distância mínima entre picos", 1, 300, 20)
        max_peaks = c4.slider("Máximo de picos", 1, 50, 15)

        detect_all = False
        if multi_mode and multi_curves:
            detect_all = st.checkbox("Detectar picos em todas as curvas", value=False)

        if st.button("Detectar picos/bandas", type="primary"):
            if detect_all and multi_mode and multi_curves:
                all_peaks = []
                for name, df_i in multi_curves.items():
                    peaks_i = detect_peaks(df_i, mode=peak_mode, prominence=prominence, distance=distance, max_peaks=max_peaks)
                    if not peaks_i.empty:
                        peaks_i.insert(0, "curve", name)
                        all_peaks.append(peaks_i)
                peaks_df = pd.concat(all_peaks, ignore_index=True) if all_peaks else pd.DataFrame()
            else:
                peaks_df = detect_peaks(curve_for_peaks, mode=peak_mode, prominence=prominence, distance=distance, max_peaks=max_peaks)
                if not peaks_df.empty:
                    peaks_df.insert(0, "curve", selected_curve_name)
            st.session_state.peaks_df = peaks_df
            st.success(f"{len(peaks_df)} pico(s)/banda(s) detectado(s).")

        peaks_df = st.session_state.peaks_df
        if peaks_df is not None and not peaks_df.empty:
            if multi_mode and multi_curves:
                if "curve" in peaks_df.columns and len(peaks_df["curve"].unique()) > 1:
                    fig = make_multi_line_plot(st.session_state.multi_curve_long_df, preset.x_label, preset.y_label, "Picos/bandas detectados em múltiplas curvas")
                    for name in peaks_df["curve"].unique():
                        sub = peaks_df[peaks_df["curve"] == name]
                        fig.add_scatter(x=sub["x"], y=sub["y"], mode="markers", name=f"Picos — {name}")
                else:
                    fig = make_line_plot(curve_for_peaks, preset.x_label, preset.y_label, f"Picos/bandas — {selected_curve_name}")
                    fig.add_scatter(x=peaks_df["x"], y=peaks_df["y"], mode="markers", name="Picos/bandas")
            else:
                fig = make_line_plot(curve_df, preset.x_label, preset.y_label, "Picos/bandas detectados")
                fig.add_scatter(x=peaks_df["x"], y=peaks_df["y"], mode="markers", name="Picos/bandas")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(peaks_df, use_container_width=True)
            st.download_button(
                "Baixar tabela de picos em CSV",
                data=dataframe_to_csv_bytes(peaks_df),
                file_name="spectracurve_peaks.csv",
                mime="text/csv",
            )
        else:
            st.info("Nenhum pico detectado ainda, ou os parâmetros atuais estão restritivos.")

with tabs[6]:
    st.header("Diagnóstico simulado com IA")
    curve_df = st.session_state.curve_df

    if curve_df is None or curve_df.empty:
        st.info("Extraia uma curva primeiro.")
    else:
        summary = summarize_curve(curve_df)
        peaks_df = st.session_state.get("peaks_df", pd.DataFrame())
        quality = st.session_state.get("quality")
        diagnostic = generate_mock_diagnostic(
            spectrum_type=st.session_state.get("spectrum_type", "Absorção / UV-Vis"),
            curve_summary=summary,
            peaks_df=peaks_df,
            quality_label=quality.estimated_quality if quality else "Não estimada",
            quality_message=quality.message if quality else "Qualidade ainda não estimada.",
            extraction_method=st.session_state.get("extraction_settings", {}).get("Método", "não informado"),
        )
        st.session_state.diagnostic = diagnostic
        st.warning("Diagnóstico simulado: esta seção não utiliza LLM ou IA generativa nesta versão.")
        st.text_area("Texto gerado para demonstração", diagnostic, height=420)

with tabs[7]:
    st.header("Relatório técnico")
    curve_df = st.session_state.curve_df

    if curve_df is None or curve_df.empty:
        st.info("Extraia uma curva primeiro.")
    else:
        summary = summarize_curve(curve_df)
        peaks_df = st.session_state.get("peaks_df", pd.DataFrame())
        diagnostic = st.session_state.get("diagnostic") or generate_mock_diagnostic(
            spectrum_type=st.session_state.get("spectrum_type", "Absorção / UV-Vis"),
            curve_summary=summary,
            peaks_df=peaks_df,
            quality_label=st.session_state.quality.estimated_quality if st.session_state.quality else "Não estimada",
            quality_message=st.session_state.quality.message if st.session_state.quality else "Qualidade ainda não estimada.",
            extraction_method=st.session_state.get("extraction_settings", {}).get("Método", "não informado"),
        )

        report = build_report(
            filename=st.session_state.get("filename", "imagem_sem_nome"),
            spectrum_type=st.session_state.get("spectrum_type", "Absorção / UV-Vis"),
            axis_info=st.session_state.get("axis_settings", {}),
            extraction_info=st.session_state.get("extraction_settings", {}),
            curve_summary=summary,
            peaks_df=peaks_df,
            diagnostic_text=diagnostic,
        )
        st.session_state.report = report
        st.text_area("Relatório", report, height=520)

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "Baixar relatório TXT",
            data=report.encode("utf-8"),
            file_name="spectracurve_report.txt",
            mime="text/plain",
        )
        c2.download_button(
            "Baixar dados CSV",
            data=dataframe_to_csv_bytes(curve_df[["x", "y"]]),
            file_name="spectracurve_extracted_data.csv",
            mime="text/csv",
        )

        if c3.button("Salvar análise no histórico"):
            signature = f"{st.session_state.get('filename')}|{summary.get('n_points')}|{len(peaks_df)}|{st.session_state.quality.estimated_quality if st.session_state.quality else 'NA'}"
            insert_analysis(
                filename=st.session_state.get("filename", "imagem_sem_nome"),
                spectrum_type=st.session_state.get("spectrum_type", "não informado"),
                extraction_method=st.session_state.get("extraction_settings", {}).get("Método", "não informado"),
                n_points=summary.get("n_points", 0),
                n_peaks=len(peaks_df),
                quality=st.session_state.quality.estimated_quality if st.session_state.quality else "Não estimada",
            )
            st.session_state.last_saved_signature = signature
            st.success("Análise salva no histórico.")

with tabs[8]:
    st.header("Histórico local de análises")
    history = load_history()
    if history.empty:
        st.info("Nenhuma análise salva ainda.")
    else:
        st.dataframe(history, use_container_width=True)
        st.download_button(
            "Baixar histórico CSV",
            data=dataframe_to_csv_bytes(history),
            file_name="spectracurve_history.csv",
            mime="text/csv",
        )
