"""Mock AI diagnostics for the prototype stage."""
from __future__ import annotations

import pandas as pd


def generate_mock_diagnostic(
    spectrum_type: str,
    curve_summary: dict,
    peaks_df: pd.DataFrame,
    quality_label: str,
    quality_message: str,
    extraction_method: str,
) -> str:
    """Generate deterministic placeholder text showing where a future LLM would act."""
    n_points = curve_summary.get("n_points", 0)
    n_peaks = 0 if peaks_df is None else len(peaks_df)

    if peaks_df is not None and not peaks_df.empty:
        main_peak = peaks_df.iloc[0]
        peak_text = f"O principal pico/banda detectado aparece em x ≈ {main_peak['x']:.3g}, com intensidade y ≈ {main_peak['y']:.3g}."
    else:
        peak_text = "Nenhum pico/banda foi detectado com os parâmetros atuais. Ajuste a proeminência, distância mínima ou suavização."

    if spectrum_type == "Absorção / UV-Vis":
        interpretation = (
            "Para espectros de absorção, um agente de IA futuro poderia sugerir atribuições preliminares "
            "para bandas eletrônicas, identificar deslocamentos batocrômicos/hipsocrômicos e comparar o espectro com dados de literatura ou TD-DFT."
        )
    elif spectrum_type == "Raman":
        interpretation = (
            "Para espectros Raman, um agente de IA futuro poderia sugerir associações entre picos e modos vibracionais, "
            "destacar bandas diagnósticas e comparar o resultado com espectros calculados."
        )
    else:
        interpretation = (
            "Para FTIR, um agente de IA futuro poderia reconhecer regiões funcionais, sugerir atribuições vibracionais e "
            "avaliar deslocamentos de bandas associados a interações químicas ou mudanças estruturais."
        )

    return f"""Esta interpretação é simulada. Nenhum modelo de IA ou LLM foi integrado nesta versão.

Resumo automático:
- Tipo de espectro: {spectrum_type}
- Método de extração: {extraction_method}
- Pontos extraídos: {n_points}
- Picos/bandas detectados: {n_peaks}
- Qualidade estimada da extração: {quality_label}

{quality_message}

{peak_text}

Motivação para integração futura com IA:
{interpretation}

Na versão futura, o agente também poderá orientar a calibração dos eixos, identificar problemas de extração, comparar múltiplos espectros e gerar relatórios científicos contextualizados.
"""
