# Relatório de validação da entrega

Data de preparação: 16/07/2026

## Testes executados

| Validação | Resultado |
|---|---|
| Compilação de `app.py`, `agents`, `tools`, `utils`, `scripts` e `tests` | Aprovada |
| Pytest | **8 passed** |
| Integração com `examples/Fig3-a.jpg` | Aprovada |
| Máscaras CIELAB não sobrepostas | Aprovada |
| Cobertura horizontal das três curvas | 96,91%–98,78% |
| Ordem dos máximos Y6 < Y6-2Se < Y6-2Te | Aprovada |
| Inicialização do Streamlit | Aprovada |
| Endpoint de saúde Streamlit | `ok` |
| Validação estrutural do PPTX | Aprovada, sem overflow |

## Resultados principais da imagem

| Curva | Cobertura | Máximo principal |
|---|---:|---:|
| Y6 | 98,31% | 687,47 nm |
| Y6-2Se | 96,91% | 701,97 nm |
| Y6-2Te | 98,78% | 730,09 nm |

## Validação do Ollama

O ambiente de preparação não possui o executável Ollama nem modelos locais. Por isso, não foi executada inferência real e não foram inventadas métricas de latência ou qualidade textual.

O script abaixo está incluído para execução automática no computador de apresentação:

```bash
ollama serve
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
python scripts/run_llm_experiments.py
```
