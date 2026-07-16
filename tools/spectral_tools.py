"""Deterministic tools exposed to the local Ollama model.

The LLM never receives authority to fabricate calculations. It may request these
functions, which operate on curve data already present in the Streamlit session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from utils.spectra_analysis import summarize_curve


@dataclass
class SpectralContext:
    spectrum_type: str
    curves: dict[str, pd.DataFrame]
    peaks_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: dict[str, Any] = field(default_factory=dict)
    axis_info: dict[str, Any] = field(default_factory=dict)
    extraction_info: dict[str, Any] = field(default_factory=dict)
    curve_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    filename: str = "imagem_sem_nome"

    @property
    def curve_names(self) -> list[str]:
        return list(self.curves.keys())


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    return value


def _json(data: dict[str, Any]) -> str:
    return json.dumps(_finite_or_none(data), ensure_ascii=False, sort_keys=True)



def _normalize_reference(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9#]+", " ", text.casefold()).strip()


def _metadata_catalog(context: SpectralContext) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for ordinal, name in enumerate(context.curve_names, start=1):
        info = dict(context.curve_metadata.get(name, {}))
        info.setdefault("ordinal", ordinal)
        info.setdefault("name", name)
        info.setdefault("color_label", None)
        info.setdefault("rgb", None)
        info.setdefault("hex", None)
        catalog.append(info)
    return catalog

def build_tool_schemas(curve_names: list[str]) -> list[dict[str, Any]]:
    """Create Ollama-compatible function schemas with constrained arguments."""
    names = curve_names or ["Curva única"]
    all_names = names + ["Todas"]

    def tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    empty_parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    return [
        tool(
            "get_analysis_context",
            "Retorna o tipo de espectro, catálogo das curvas com nomes/cores, arquivo, calibração e método de extração. Use para entender o contexto geral antes de interpretar.",
            empty_parameters,
        ),
        tool(
            "list_detected_curves",
            "Lista todas as curvas reconhecidas e seus metadados: ordem, nome, rótulo de cor, RGB, HEX e modo de seleção.",
            empty_parameters,
        ),
        tool(
            "get_curve_by_color",
            "Resolve uma referência do usuário por nome, cor, RGB, HEX ou ordem, como 'azul', '#1F77B4', 'segunda curva' ou o nome personalizado.",
            {
                "type": "object",
                "properties": {
                    "color_reference": {
                        "type": "string",
                        "description": "Nome, cor, HEX, RGB ou posição ordinal usada para identificar a curva.",
                    }
                },
                "required": ["color_reference"],
                "additionalProperties": False,
            },
        ),
        tool(
            "get_curve_summary",
            "Calcula estatísticas determinísticas de uma curva: quantidade de pontos, limites de x e y e área trapezoidal.",
            {
                "type": "object",
                "properties": {
                    "curve_name": {
                        "type": "string",
                        "enum": names,
                        "description": "Nome exato da curva disponível no contexto.",
                    }
                },
                "required": ["curve_name"],
                "additionalProperties": False,
            },
        ),
        tool(
            "get_detected_peaks",
            "Retorna somente os picos ou bandas já calculados pelo algoritmo determinístico de detecção. Não cria novos picos.",
            {
                "type": "object",
                "properties": {
                    "curve_name": {
                        "type": "string",
                        "enum": all_names,
                        "description": "Curva específica ou 'Todas'.",
                    }
                },
                "required": ["curve_name"],
                "additionalProperties": False,
            },
        ),
        tool(
            "compare_curves",
            "Compara duas curvas na faixa de x sobreposta e retorna correlação, RMSE, MAE, diferença média, diferença máxima e áreas.",
            {
                "type": "object",
                "properties": {
                    "curve_a": {"type": "string", "enum": names},
                    "curve_b": {"type": "string", "enum": names},
                },
                "required": ["curve_a", "curve_b"],
                "additionalProperties": False,
            },
        ),
        tool(
            "get_extraction_quality",
            "Retorna a avaliação determinística da qualidade de extração e sua justificativa.",
            empty_parameters,
        ),
    ]


class SpectralToolRegistry:
    """Routes model function calls to deterministic Python implementations."""

    def __init__(self, context: SpectralContext):
        self.context = context

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "get_analysis_context":
                result = self.get_analysis_context()
            elif name == "list_detected_curves":
                result = self.list_detected_curves()
            elif name == "get_curve_by_color":
                result = self.get_curve_by_color(arguments["color_reference"])
            elif name == "get_curve_summary":
                result = self.get_curve_summary(arguments["curve_name"])
            elif name == "get_detected_peaks":
                result = self.get_detected_peaks(arguments["curve_name"])
            elif name == "compare_curves":
                result = self.compare_curves(arguments["curve_a"], arguments["curve_b"])
            elif name == "get_extraction_quality":
                result = self.get_extraction_quality()
            else:
                result = {"ok": False, "error": f"Ferramenta desconhecida: {name}"}
        except (KeyError, ValueError, TypeError) as exc:
            result = {"ok": False, "error": str(exc), "tool": name}
        except Exception as exc:  # defensive boundary: tool errors must reach the model safely
            result = {"ok": False, "error": f"Falha interna na ferramenta: {exc}", "tool": name}
        return _json(result)

    def _curve(self, name: str) -> pd.DataFrame:
        if name not in self.context.curves:
            raise ValueError(f"Curva '{name}' não está disponível.")
        df = self.context.curves[name]
        if df is None or df.empty:
            raise ValueError(f"Curva '{name}' está vazia.")
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError(f"Curva '{name}' não contém colunas x e y.")
        return df[["x", "y"]].dropna().copy()

    def get_analysis_context(self) -> dict[str, Any]:
        return {
            "ok": True,
            "filename": self.context.filename,
            "spectrum_type": self.context.spectrum_type,
            "available_curves": self.context.curve_names,
            "curve_catalog": _metadata_catalog(self.context),
            "axis_info": self.context.axis_info,
            "extraction_info": self.context.extraction_info,
        }

    def list_detected_curves(self) -> dict[str, Any]:
        catalog = _metadata_catalog(self.context)
        return {
            "ok": True,
            "count": len(catalog),
            "curves": catalog,
        }

    def _resolve_curve_reference(self, reference: str) -> str:
        normalized = _normalize_reference(reference)
        if not normalized:
            raise ValueError("Informe uma cor, nome, código HEX ou ordem da curva.")

        ordinal_words = {
            "primeira": 1, "primeiro": 1,
            "segunda": 2, "segundo": 2,
            "terceira": 3, "terceiro": 3,
            "quarta": 4, "quarto": 4,
            "quinta": 5, "quinto": 5,
            "sexta": 6, "sexto": 6,
            "setima": 7, "setimo": 7,
            "oitava": 8, "oitavo": 8,
            "nona": 9, "nono": 9,
            "decima": 10, "decimo": 10,
        }
        ordinal = None
        number_match = re.search(r"\b(10|[1-9])\b", normalized)
        if number_match:
            ordinal = int(number_match.group(1))
        else:
            for word, value in ordinal_words.items():
                if re.search(rf"\b{word}\b", normalized):
                    ordinal = value
                    break

        catalog = _metadata_catalog(self.context)
        if ordinal is not None:
            ordinal_matches = [item for item in catalog if int(item.get("ordinal", -1)) == ordinal]
            if len(ordinal_matches) == 1:
                return str(ordinal_matches[0]["name"])

        scored: list[tuple[int, str]] = []
        for item in catalog:
            name = str(item.get("name", ""))
            fields = [
                name,
                item.get("color_label"),
                item.get("hex"),
                item.get("detected_hex"),
                " ".join(str(v) for v in (item.get("rgb") or [])),
                " ".join(str(v) for v in (item.get("detected_rgb") or [])),
            ]
            normalized_fields = [
                normalized_field
                for field in fields
                if field is not None
                for normalized_field in [_normalize_reference(field)]
                if normalized_field
            ]
            score = 0
            for field in normalized_fields:
                if normalized == field:
                    score = max(score, 100)
                elif len(field) >= 3 and (normalized in field or field in normalized):
                    score = max(score, 70)
                else:
                    tokens = set(normalized.split())
                    overlap = len(tokens.intersection(field.split()))
                    score = max(score, overlap * 15)
            if score > 0:
                scored.append((score, name))

        if not scored:
            raise ValueError(
                f"Nenhuma curva corresponde a '{reference}'. Curvas disponíveis: {', '.join(self.context.curve_names)}."
            )
        best_score = max(score for score, _ in scored)
        best_names = sorted({name for score, name in scored if score == best_score})
        if len(best_names) != 1:
            raise ValueError(
                f"A referência '{reference}' é ambígua entre: {', '.join(best_names)}. Use o nome ou HEX exato."
            )
        return best_names[0]

    def get_curve_by_color(self, color_reference: str) -> dict[str, Any]:
        curve_name = self._resolve_curve_reference(color_reference)
        metadata = dict(self.context.curve_metadata.get(curve_name, {}))
        summary = summarize_curve(self._curve(curve_name))
        return {
            "ok": True,
            "input_reference": color_reference,
            "resolved_curve_name": curve_name,
            "metadata": metadata,
            "summary": summary,
        }

    def get_curve_summary(self, curve_name: str) -> dict[str, Any]:
        summary = summarize_curve(self._curve(curve_name))
        return {
            "ok": True,
            "curve_name": curve_name,
            "metadata": self.context.curve_metadata.get(curve_name, {}),
            "summary": summary,
        }

    def get_detected_peaks(self, curve_name: str) -> dict[str, Any]:
        peaks = self.context.peaks_df
        if peaks is None or peaks.empty:
            return {
                "ok": True,
                "curve_name": curve_name,
                "count": 0,
                "peaks": [],
                "message": "Nenhum pico ou banda foi calculado com os parâmetros atuais.",
            }

        selected = peaks.copy()
        if curve_name != "Todas" and "curve" in selected.columns:
            selected = selected[selected["curve"].astype(str) == curve_name]

        allowed = [c for c in ["curve", "peak_id", "x", "y", "prominence"] if c in selected.columns]
        records = selected[allowed].replace({np.nan: None}).to_dict(orient="records")
        return {
            "ok": True,
            "curve_name": curve_name,
            "count": len(records),
            "peaks": records,
        }

    def compare_curves(self, curve_a: str, curve_b: str) -> dict[str, Any]:
        if curve_a == curve_b:
            raise ValueError("Selecione duas curvas diferentes para comparação.")

        a = self._curve(curve_a).sort_values("x").drop_duplicates("x")
        b = self._curve(curve_b).sort_values("x").drop_duplicates("x")
        left = max(float(a["x"].min()), float(b["x"].min()))
        right = min(float(a["x"].max()), float(b["x"].max()))
        if right <= left:
            raise ValueError("As curvas não possuem intervalo de x sobreposto.")

        n = int(min(max(len(a), len(b), 50), 1000))
        x_grid = np.linspace(left, right, n)
        ya = np.interp(x_grid, a["x"].to_numpy(float), a["y"].to_numpy(float))
        yb = np.interp(x_grid, b["x"].to_numpy(float), b["y"].to_numpy(float))
        diff = ya - yb

        std_a = float(np.std(ya))
        std_b = float(np.std(yb))
        correlation = None if std_a == 0 or std_b == 0 else float(np.corrcoef(ya, yb)[0, 1])
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))

        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        return {
            "ok": True,
            "curve_a": curve_a,
            "curve_b": curve_b,
            "metadata_a": self.context.curve_metadata.get(curve_a, {}),
            "metadata_b": self.context.curve_metadata.get(curve_b, {}),
            "overlap": {"x_min": left, "x_max": right, "n_samples": n},
            "metrics": {
                "pearson_correlation": correlation,
                "rmse": rmse,
                "mae": mae,
                "mean_signed_difference": float(np.mean(diff)),
                "max_absolute_difference": float(np.max(np.abs(diff))),
                "area_a_overlap": float(integrate(ya, x_grid)),
                "area_b_overlap": float(integrate(yb, x_grid)),
            },
            "note": "Correlação alta indica semelhança de forma, não igualdade absoluta.",
        }

    def get_extraction_quality(self) -> dict[str, Any]:
        if not self.context.quality:
            return {"ok": True, "estimated_quality": "Não estimada", "message": "A qualidade ainda não foi calculada."}
        return {"ok": True, **self.context.quality}
