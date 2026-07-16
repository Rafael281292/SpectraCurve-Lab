# Roteiro de apresentação — 3 minutos

## Slide 1 — 0:00 a 0:25

“Este é o SpectraCurve Lab, uma aplicação que digitaliza curvas científicas presentes em imagens, reconstrói os dados e usa IA generativa local para interpretar resultados. O ponto principal é que o modelo não inventa os cálculos: ele consulta funções Python determinísticas.”

## Slide 2 — 0:25 a 0:55

“O problema é que muitos dados são publicados apenas como figuras. Nesta imagem de teste existem três curvas azuis com tonalidades próximas. A aplicação cria uma máscara CIELAB para cada cor e armazena nome, ordem, RGB e HEX, preservando a identidade das curvas.”

## Slide 3 — 0:55 a 1:25

“No teste automático, as três curvas foram extraídas com cobertura entre 96,9% e 98,8%. Os máximos principais foram 687,5, 702,0 e 730,1 nanômetros, mostrando o deslocamento progressivo para maiores comprimentos de onda. As máscaras não se sobrepuseram.”

## Slide 4 — 1:25 a 2:05

“Escolhi Ollama com qwen3:1.7b porque não exige API paga, preserva privacidade e suporta tool calling. Usei o SDK oficial diretamente, sem LangChain, porque existe apenas um agente e sete ferramentas. A temperatura padrão é 0.2 e o top-p é 0.9 para priorizar estabilidade.”

## Slide 5 — 2:05 a 2:35

“O system prompt usa tags XML, few-shot e regras contra invenção. As tools listam curvas, resolvem uma referência por cor ou ordem, retornam resumos, picos, comparações e qualidade. Se o modelo ignorar as tools, a aplicação injeta um contexto determinístico e registra o fallback.”

## Slide 6 — 2:35 a 3:00

“Foram aprovados oito testes automatizados, incluindo a imagem real. As principais limitações são os pontos pretos, que se confundem com eixos e textos, e a menor capacidade de modelos locais pequenos. Como próximo passo, eu adicionaria detecção específica de marcadores e uma biblioteca espectral validada.”

# Respostas rápidas

## Por que temperatura 0.2?

Porque a tarefa é técnico-científica e exige consistência. Valores maiores ampliam variação e risco de especulação.

## Por que não LangChain?

Porque há um agente, sete tools e nenhum grafo complexo. O SDK direto reduz dependências e facilita explicar e depurar o fluxo.

## O modelo reconhece a imagem?

Não diretamente nesta arquitetura. Python faz segmentação e extração; o LLM interpreta resultados estruturados.

## Como as cores são preservadas?

Cada curva possui máscara independente, nome, ordem, RGB e HEX. Esses metadados ficam disponíveis nas tools.

## O que acontece com prompt malicioso?

O input é delimitado, o system prompt proíbe substituir regras e números devem vir das tools. Isso reduz, mas não elimina completamente o risco.

## O que mudaria com um modelo pago maior?

A interpretação e a obediência às tools provavelmente melhorariam, mas haveria custo por chamada e envio de dados a um provedor externo.
