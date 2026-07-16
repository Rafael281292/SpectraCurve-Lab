from types import SimpleNamespace

import pandas as pd

import agents.spectral_agent as agent
from tools.spectral_tools import SpectralContext


class FakeClient:
    last_instance = None

    def __init__(self, **kwargs):
        self.calls = []
        self.kwargs = kwargs
        FakeClient.last_instance = self

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="get_curve_summary",
                    arguments={"curve_name": "Curva única"},
                )
            )
            message = SimpleNamespace(
                role="assistant",
                content="",
                thinking="",
                tool_calls=[tool_call],
                model_dump=lambda exclude_none=True: {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_curve_summary",
                                "arguments": {"curve_name": "Curva única"},
                            },
                        }
                    ],
                },
            )
            return SimpleNamespace(
                message=message,
                prompt_eval_count=10,
                eval_count=4,
                total_duration=1_000_000_000,
                load_duration=100_000_000,
            )

        message = SimpleNamespace(
            role="assistant",
            content="## Resumo\nCurva analisada com dados da ferramenta.",
            thinking="",
            tool_calls=[],
            model_dump=lambda exclude_none=True: {
                "role": "assistant",
                "content": "## Resumo\nCurva analisada com dados da ferramenta.",
            },
        )
        return SimpleNamespace(
            message=message,
            prompt_eval_count=20,
            eval_count=10,
            total_duration=2_000_000_000,
            load_duration=0,
        )

    def list(self):
        return SimpleNamespace(
            models=[SimpleNamespace(model="qwen3:1.7b"), SimpleNamespace(model="qwen3:4b")]
        )


def make_context():
    curve = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
    return SpectralContext("Raman", {"Curva única": curve})


def test_agent_executes_tool_and_returns_final_text(monkeypatch):
    monkeypatch.setattr(agent, "Client", FakeClient)

    result = agent.run_spectral_agent(
        question="Resuma a curva.",
        context=make_context(),
    )

    assert "Curva analisada" in result.text
    assert result.tool_trace[0]["tool"] == "get_curve_summary"
    assert result.tool_trace[0]["invoked_by"] == "model"
    assert result.usage["total_tokens"] == 44
    calls = FakeClient.last_instance.calls
    assert calls[0]["model"] == "qwen3:1.7b"
    assert calls[0]["options"]["temperature"] == 0.2
    assert calls[0]["options"]["num_ctx"] == 4096
    assert calls[0]["tools"][0]["function"]["name"] == "get_analysis_context"


def test_list_installed_models(monkeypatch):
    monkeypatch.setattr(agent, "Client", FakeClient)
    assert agent.list_installed_models() == ["qwen3:1.7b", "qwen3:4b"]
