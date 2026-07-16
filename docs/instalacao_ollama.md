# Instalação e validação do Ollama

## Ubuntu

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verifique:

```bash
ollama -v
```

Inicie o serviço, quando necessário:

```bash
ollama serve
```

Em outra janela do terminal, baixe o modelo padrão:

```bash
ollama pull qwen3:1.7b
```

Confirme a instalação:

```bash
ollama list
```

Teste uma conversa direta:

```bash
ollama run qwen3:1.7b
```

## Testar a API local

```bash
curl http://localhost:11434/api/tags
```

## Executar o SpectraCurve Lab

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

No painel lateral, o campo **Servidor Ollama** deve ser:

```text
http://localhost:11434
```

## Problemas comuns

### Não foi possível conectar

Execute:

```bash
ollama serve
```

ou, em instalações com systemd:

```bash
sudo systemctl start ollama
sudo systemctl status ollama
```

### Modelo não encontrado

```bash
ollama pull qwen3:1.7b
```

### Resposta lenta

- mantenha `num_ctx=4096`;
- mantenha thinking desativado;
- teste `qwen3:0.6b`;
- feche aplicações que consomem muita memória;
- verifique GPU e memória com `nvidia-smi` e `free -h`.
