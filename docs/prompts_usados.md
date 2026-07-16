# Prompts usados com agente de codificação

Este arquivo registra as principais instruções usadas durante o desenvolvimento e as mudanças de arquitetura. Os textos foram revisados manualmente antes de serem incorporados ao projeto.

## Primeira etapa — avaliação intermediária

### Prompt 1 — estrutura inicial

Crie uma aplicação Streamlit chamada SpectraCurve Lab para extração e análise de espectros de Absorção, Raman e FTIR a partir de imagens. A aplicação deve ter upload de imagem, calibração dos eixos, extração da curva, reconstrução do espectro, detecção de picos, download CSV, diagnóstico simulado e relatório técnico. Não integre nenhum LLM nesta etapa.

### Prompt 2 — modularização

Separe o código em módulos Python: processamento de imagem, calibração, extração de curva, análise espectral, detecção de picos, diagnóstico simulado, relatório e banco de dados local.

### Prompt 3 — múltiplas curvas e calibração

Melhore a extração para reconhecer múltiplas curvas por cor ou componentes. Permita marcar x mínimo, x máximo, y mínimo e y máximo diretamente na imagem e use esses pontos para delimitar a região útil.

## Segunda etapa — IA generativa

### Prompt 4 — primeira arquitetura considerada

Integre um LLM ao SpectraCurve Lab usando chamadas diretas, function calling e ferramentas determinísticas. Mantenha os cálculos em Python e faça o modelo interpretar apenas os resultados retornados pelas ferramentas.

### Prompt 5 — system prompt científico

Crie um system prompt com persona de especialista em espectroscopia, regras contra invenção de picos, distinção entre fatos e hipóteses, formato de saída em Markdown, tags XML e proteção básica contra prompt injection.

### Prompt 6 — tools tipadas

Implemente ferramentas com JSON Schema para resumir curvas, retornar picos detectados, comparar curvas, consultar o contexto e avaliar a qualidade de extração. Inclua tratamento de erros.

### Prompt 7 — interface e rastreabilidade

Substitua a aba de diagnóstico simulado por uma aba de assistente real. Permita configurar modelo e parâmetros. Exiba as tools chamadas, seus argumentos, resultados e métricas de geração.

### Prompt 8 — migração para Ollama

Substitua a integração de API paga por Ollama local. Use o SDK oficial Python, preserve o loop de tool calling, remova API keys, adote qwen3:1.7b como modelo padrão para hardware limitado e atualize interface, testes, documentação e instruções de instalação.

### Prompt 9 — fallback de tool calling

Como o Ollama não expõe um parâmetro nativo equivalente a tool choice obrigatório, faça o system prompt exigir pelo menos uma tool. Caso o modelo não chame nenhuma, execute get_analysis_context como fallback determinístico, registre a origem da chamada e solicite uma continuação fundamentada.

## Iteração e refinamento

### Abordagem abandonada

Foi considerada uma API paga de alta capacidade. Essa alternativa oferecia maior robustez, mas exigia chave e faturamento separados. A arquitetura foi migrada para Ollama para atender ao requisito de não gerar pagamento adicional.

### Correções manuais

- Os schemas foram convertidos para o formato de tools do Ollama.
- A dependência de chave de API foi removida.
- O modelo padrão foi reduzido para `qwen3:1.7b` devido ao hardware disponível.
- Foi adicionado fallback quando um modelo pequeno não chama tools.
- O thinking não é exibido ao usuário.
- Temperatura e top-p receberam protocolo de experimentação separado.
- O README foi reescrito para documentar trade-offs de privacidade, custo, memória e qualidade.
- Foram adicionados testes do loop do agente com cliente Ollama simulado.

## Partes desenvolvidas com IA de codificação

- estrutura inicial do Streamlit;
- modularização;
- schemas das tools;
- loop do agente;
- testes automatizados;
- documentação inicial.

## Partes revisadas ou ajustadas manualmente

- regras científicas do system prompt;
- separação entre cálculo e interpretação;
- limites de atribuição espectroscópica;
- valores padrão de parâmetros;
- escolha do modelo compatível com o hardware;
- mensagens de erro e instruções de execução;
- validação das tools e dos testes.

### Prompt 8 — seleção manual de múltiplas cores

> Permita selecionar várias cores, gerar uma máscara independente para cada uma e armazenar cada curva com nome, RGB e HEX. Preserve essa associação no gráfico, no CSV e nas tools do modelo.

**Decisões incorporadas:**

- segmentação perceptual no espaço CIELAB;
- atribuição exclusiva de cada pixel à cor selecionada mais próxima;
- catálogo estruturado por curva;
- tools para listar curvas e resolver referências por cor, HEX ou ordem;
- testes de não sobreposição das máscaras.
