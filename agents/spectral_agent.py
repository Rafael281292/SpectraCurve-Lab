"""Local Ollama tool-calling orchestration for SpectraCurve Lab."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from tools.spectral_tools import SpectralContext, SpectralToolRegistry, build_tool_schemas

try:
    from ollama import Client, RequestError, ResponseError
except ImportError:  # allows the rest of the app to load before dependencies are installed
    Client = None  # type: ignore[assignment]
    RequestError = ResponseError = Exception  # type: ignore[misc,assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_HOST = "http://localhost:11434"


@dataclass
class AgentResult:
    text: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    usage: dict[str, int | float | None] = field(default_factory=dict)


def _load_system_prompt() -> str:
    prompt = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    examples_path = ROOT / "prompts" / "few_shot_examples.json"
    if examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
        prompt += "\n\n<few_shot_examples>\n" + json.dumps(examples, ensure_ascii=False, indent=2) + "\n</few_shot_examples>"
    prompt += (
        "\n\n<ollama_agent_loop>"
        "Você está em um loop de ferramentas. Antes da resposta final, chame pelo menos uma ferramenta relevante. "
        "Você pode chamar ferramentas adicionais em rodadas seguintes."
        "</ollama_agent_loop>"
    )
    return prompt


def _normalize_host(host: str | None) -> str:
    value = (host or "").strip() or DEFAULT_HOST
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value.rstrip("/")


def get_environment_ollama_host() -> str:
    """Return an Ollama host configured by environment, or the local default."""
    return _normalize_host(os.getenv("OLLAMA_HOST", DEFAULT_HOST))


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _model_name(item: Any) -> str:
    return str(_get(item, "model", _get(item, "name", "")) or "").strip()


def list_installed_models(host: str = DEFAULT_HOST, timeout: float = 4.0) -> list[str]:
    """List models currently installed in an Ollama server."""
    if Client is None:
        raise RuntimeError("A biblioteca 'ollama' não está instalada. Execute: pip install -r requirements.txt")
    client = Client(host=_normalize_host(host), timeout=timeout)
    response = client.list()
    models = _get(response, "models", []) or []
    return sorted({name for item in models if (name := _model_name(item))})


def _message_to_mapping(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {
        "role": _get(message, "role", "assistant"),
        "content": _get(message, "content", "") or "",
        "thinking": _get(message, "thinking", None),
        "tool_calls": _get(message, "tool_calls", None),
    }


def _tool_calls(message: Any) -> list[Any]:
    return list(_get(message, "tool_calls", None) or [])


def _tool_call_parts(call: Any) -> tuple[str, dict[str, Any]]:
    function = _get(call, "function", {})
    name = str(_get(function, "name", "") or "")
    arguments = _get(function, "arguments", {}) or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


def _accumulate_usage(total: dict[str, int], response: Any) -> None:
    for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration"):
        value = _get(response, key, 0) or 0
        try:
            total[key] = total.get(key, 0) + int(value)
        except (TypeError, ValueError):
            pass


def _usage_dict(total: dict[str, int]) -> dict[str, int | float | None]:
    prompt_tokens = total.get("prompt_eval_count", 0)
    output_tokens = total.get("eval_count", 0)
    duration_ns = total.get("total_duration", 0)
    duration_s = duration_ns / 1_000_000_000 if duration_ns else 0.0
    tokens_per_second = output_tokens / duration_s if output_tokens and duration_s else None
    return {
        "prompt_tokens": prompt_tokens or None,
        "output_tokens": output_tokens or None,
        "total_tokens": (prompt_tokens + output_tokens) or None,
        "duration_seconds": round(duration_s, 3) if duration_s else None,
        "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else None,
    }


def run_spectral_agent(
    *,
    question: str,
    context: SpectralContext,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.9,
    thinking: bool = False,
    max_output_tokens: int = 1000,
    context_window: int = 4096,
    max_tool_rounds: int = 4,
    keep_alive: str = "5m",
) -> AgentResult:
    """Run a bounded Ollama tool-calling loop and return a grounded answer."""
    if Client is None:
        raise RuntimeError("A biblioteca 'ollama' não está instalada. Execute: pip install -r requirements.txt")
    if not question.strip():
        raise ValueError("Digite uma pergunta para o assistente.")
    if not model.strip():
        raise ValueError("Selecione um modelo Ollama.")

    normalized_host = _normalize_host(host)
    client = Client(host=normalized_host, timeout=180.0)
    registry = SpectralToolRegistry(context)
    tools = build_tool_schemas(context.curve_names)

    safe_context = {
        "filename": context.filename,
        "spectrum_type": context.spectrum_type,
        "available_curves": context.curve_names,
        "curve_catalog": context.curve_metadata,
        "has_detected_peaks": bool(context.peaks_df is not None and not context.peaks_df.empty),
    }
    user_content = (
        "<analysis_context>\n"
        + json.dumps(safe_context, ensure_ascii=False, indent=2)
        + "\n</analysis_context>\n"
        + "<user_request>\n"
        + question.strip()
        + "\n</user_request>"
    )

    messages: list[Any] = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    trace: list[dict[str, Any]] = []
    usage_total: dict[str, int] = {}

    try:
        for round_index in range(max_tool_rounds + 1):
            response = client.chat(
                model=model.strip(),
                messages=messages,
                tools=tools,
                stream=False,
                think=bool(thinking),
                keep_alive=keep_alive,
                options={
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "num_predict": int(max_output_tokens),
                    "num_ctx": int(context_window),
                },
            )
            _accumulate_usage(usage_total, response)
            message = _get(response, "message", {})
            messages.append(_message_to_mapping(message))
            calls = _tool_calls(message)

            if calls:
                for call in calls:
                    name, args = _tool_call_parts(call)
                    output = registry.execute(name, args)
                    try:
                        parsed_output = json.loads(output)
                    except json.JSONDecodeError:
                        parsed_output = {"raw": output}
                    trace.append(
                        {
                            "round": round_index + 1,
                            "tool": name,
                            "arguments": args,
                            "output": parsed_output,
                            "invoked_by": "model",
                        }
                    )
                    messages.append({"role": "tool", "tool_name": name, "content": output})
                continue

            text = str(_get(message, "content", "") or "").strip()
            if trace:
                return AgentResult(
                    text=text or "O modelo não retornou texto. Tente reformular a pergunta.",
                    tool_trace=trace,
                    model=model,
                    usage=_usage_dict(usage_total),
                )

            # Ollama does not expose a provider-level tool_choice='required'. If the
            # local model ignores the instruction, the application injects one safe,
            # deterministic context tool result and asks for a grounded continuation.
            fallback_output = registry.execute("get_analysis_context", {})
            trace.append(
                {
                    "round": round_index + 1,
                    "tool": "get_analysis_context",
                    "arguments": {},
                    "output": json.loads(fallback_output),
                    "invoked_by": "application_fallback",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "A aplicação executou a ferramenta get_analysis_context porque nenhuma ferramenta foi chamada. "
                        "Use este resultado como evidência e chame outras ferramentas quando a pergunta exigir valores, picos, comparação ou qualidade:\n"
                        f"<tool_result>{fallback_output}</tool_result>"
                    ),
                }
            )

        return AgentResult(
            text="O limite de chamadas de ferramentas foi atingido antes da resposta final.",
            tool_trace=trace,
            model=model,
            usage=_usage_dict(usage_total),
        )
    except ResponseError as exc:
        status = _get(exc, "status_code", None)
        detail = str(_get(exc, "error", str(exc)) or str(exc))
        if status == 404 or "not found" in detail.lower():
            raise RuntimeError(
                f"O modelo '{model}' não está instalado. Execute no terminal: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"O servidor Ollama retornou um erro: {detail}") from exc
    except RequestError as exc:
        raise RuntimeError(f"A requisição ao Ollama é inválida: {exc}") from exc
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(
            f"Não foi possível conectar ao Ollama em {normalized_host}. Inicie o serviço com 'ollama serve'."
        ) from exc
    except Exception as exc:
        message = str(exc)
        if any(token in message.lower() for token in ("connection refused", "failed to connect", "connect error")):
            raise RuntimeError(
                f"Não foi possível conectar ao Ollama em {normalized_host}. Inicie o serviço com 'ollama serve'."
            ) from exc
        raise
