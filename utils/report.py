"""Report generation utilities."""
from __future__ import annotations

import pandas as pd


def build_report(
    filename: str,
    spectrum_type: str,
    axis_info: dict,
    extraction_info: dict,
    curve_summary: dict,
    peaks_df: pd.DataFrame,
    diagnostic_text: str,
    curve_catalog: dict | None = None,
) -> str:
    """Build a plain text technical report."""
    lines = []
    lines.append("Relatório técnico — SpectraCurve Lab")
    lines.append("=" * 46)
    lines.append("")
    lines.append(f"Arquivo analisado: {filename}")
    lines.append(f"Tipo de espectro: {spectrum_type}")
    lines.append("")

    lines.append("1. Calibração dos eixos")
    lines.append("-" * 28)
    for key, value in axis_info.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("2. Configuração de extração")
    lines.append("-" * 31)
    for key, value in extraction_info.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("3. Curvas e cores associadas")
    lines.append("-" * 29)
    if curve_catalog:
        for name, info in curve_catalog.items():
            lines.append(
                f"{info.get('ordinal', '-')}. {name}: cor={info.get('color_label')}, "
                f"RGB={info.get('rgb')}, HEX={info.get('hex')}, modo={info.get('selection_mode')}"
            )
    else:
        lines.append("Catálogo de cores não disponível.")
    lines.append("")

    lines.append("4. Resumo numérico da curva principal")
    lines.append("-" * 37)
    for key, value in curve_summary.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("5. Picos/bandas detectados")
    lines.append("-" * 28)
    if peaks_df is not None and not peaks_df.empty:
        for _, row in peaks_df.iterrows():
            lines.append(
                f"Pico {int(row['peak_id'])}: x = {row['x']:.6g}, y = {row['y']:.6g}, proeminência = {row.get('prominence', float('nan')):.6g}"
            )
    else:
        lines.append("Nenhum pico/banda detectado com os parâmetros atuais.")
    lines.append("")

    lines.append("6. Diagnóstico com IA generativa")
    lines.append("-" * 34)
    lines.append(diagnostic_text.strip())
    lines.append("")

    lines.append("7. Limitações")
    lines.append("-" * 13)
    lines.append("A extração depende da qualidade da imagem, da calibração manual dos eixos e dos parâmetros de máscara.")
    lines.append("A interpretação do LLM depende da qualidade dos dados e não substitui validação por especialista ou comparação com referências.")
    lines.append("Gráficos com múltiplas curvas, baixa resolução ou eixos muito poluídos podem exigir ajuste manual.")

    return "\n".join(lines)
