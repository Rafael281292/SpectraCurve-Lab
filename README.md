---
title: SpectraCurve Lab
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SpectraCurve Lab

**SpectraCurve Lab** é um protótipo de aplicação web para **extração, reconstrução e análise de espectros científicos a partir de imagens**. O foco inicial é trabalhar com espectros de **Absorção / UV-Vis**, **Raman** e **FTIR**.

A aplicação permite que o usuário envie uma imagem de gráfico, selecione o tipo de espectro, calibre os eixos, extraia uma ou mais curvas, reconstrua os dados em formato numérico, detecte picos/bandas, exporte CSV e gere um relatório técnico.

> **Importante:** esta versão **não integra LLM, OCR avançado ou modelo generativo**. As partes em que uma IA poderia atuar futuramente aparecem como **diagnóstico simulado**, mensagens de apoio e fluxos de interface. O objetivo desta etapa é entregar a interface e a estrutura funcional do protótipo.

---

## Links da entrega

- **Endpoint público:** https://spectracurve-lab-skfeenda2vvypwezrdiel5.streamlit.app/
- **Repositório GitHub:** https://github.com/Rafael281292/SpectraCurve-Lab
- **Ferramenta de codificação com IA usada:** OpenAI Codex
- **Framework principal:** Streamlit
- **Linguagem:** Python
- **Banco local:** SQLite
- **Deploy previsto:** Hugging Face Spaces com Docker

---

## Checklist de aderência à atividade

| Exigência da atividade | Como foi atendida no projeto |
|---|---|
| Endpoint público funcional | A aplicação foi preparada para deploy em Hugging Face Spaces com Docker. O link deve ser inserido na seção **Links da entrega** após o deploy. |
| Problema real e desafiador | O sistema trata a digitalização de espectros publicados apenas como imagem, problema comum em análise científica e reprodutibilidade de dados. |
| Interface e estrutura completas | O protótipo possui upload, seleção de tipo de espectro, pré-processamento, calibração, extração, reconstrução, análise de picos, exportação, relatório e histórico. |
| Não integrar LLM/modelo real | A aplicação usa apenas processamento clássico de imagem e diagnóstico simulado. |
| Repositório GitHub | O projeto deve ser entregue em repositório com estrutura clara, `.gitignore`, `requirements.txt`, `Dockerfile` e commits incrementais. |
| README detalhado | Este README documenta problema, solução, escolhas de design, uso do Codex, prompts, acertos, falhas, partes manuais e próximos passos. |

---

## 1. Problema escolhido

Em muitos artigos científicos, relatórios, dissertações, apresentações e materiais técnicos, espectros aparecem apenas como imagens. Quando os dados brutos não estão disponíveis, torna-se difícil:

- comparar resultados de diferentes trabalhos;
- reproduzir análises;
- calcular deslocamentos de bandas;
- identificar picos quantitativamente;
- comparar dados experimentais com simulações;
- montar bancos de dados a partir de gráficos publicados;
- reutilizar dados em Python, Excel, Origin, MATLAB ou outros softwares científicos.

Esse problema é comum em áreas como química computacional, espectroscopia molecular, física molecular, ciência de materiais, polímeros e caracterização experimental.

---

## 2. Solução proposta

O **SpectraCurve Lab** propõe um fluxo guiado para transformar uma imagem de gráfico em dados numéricos aproximados.

Fluxo principal:

1. Upload da imagem do espectro.
2. Escolha do tipo de espectro: Absorção / UV-Vis, Raman ou FTIR.
3. Recorte da área útil do gráfico.
4. Calibração dos eixos.
5. Extração de uma ou várias curvas.
6. Conversão de pixels para coordenadas reais.
7. Reconstrução do espectro em formato numérico.
8. Pós-processamento: suavização, normalização e correção simples de baseline.
9. Detecção de picos/bandas.
10. Exportação dos dados em CSV.
11. Diagnóstico simulado.
12. Geração de relatório técnico.
13. Registro no histórico local.

A ideia central é permitir que um usuário recupere uma aproximação dos dados `x, y` a partir de um gráfico publicado como imagem.

---

## 3. Como a IA será integrada futuramente

Nesta avaliação intermediária, a aplicação **não usa IA generativa em tempo de execução**. Porém, o fluxo foi planejado para permitir integração futura com um agente de IA.

Possíveis usos futuros de IA:

- reconhecer automaticamente o tipo de espectro;
- detectar a região útil do gráfico;
- identificar e calibrar eixos por OCR;
- separar curva, legenda, eixo, texto e linhas de grade;
- sugerir parâmetros de extração quando a máscara estiver ruim;
- interpretar picos/bandas de UV-Vis, Raman e FTIR;
- comparar espectros extraídos com dados de literatura ou simulações;
- sugerir atribuições vibracionais ou eletrônicas;
- gerar relatórios científicos contextualizados;
- explicar ao usuário as limitações da extração.

No protótipo atual, essas funcionalidades aparecem como **mock/placeholder**, principalmente na parte de diagnóstico simulado.

---

## 4. Funcionalidades implementadas

- Interface Streamlit com múltiplas abas.
- Upload de imagens `.png`, `.jpg`, `.jpeg` e `.webp`.
- Três modos espectrais: Absorção / UV-Vis, Raman e FTIR.
- Presets de eixo para cada tipo de espectro.
- Recorte manual da região útil do gráfico.
- Pré-processamento de imagem com contraste, brilho e realce.
- Calibração manual pelos valores mínimo/máximo dos eixos.
- Calibração por pontos marcados diretamente na imagem.
- Delimitação automática da região útil com base nos pontos dos eixos.
- Extração de curva por contraste.
- Extração de curva por cor.
- Extração de múltiplas curvas por cor.
- Extração de múltiplas curvas por componentes/caminhos.
- Reconstrução dos dados em coordenadas reais.
- Visualização da máscara detectada.
- Visualização do espectro reconstruído.
- Suavização por média móvel ou Savitzky-Golay.
- Normalização por máximo ou min-max.
- Correção de baseline simples.
- Detecção de picos/bandas com `scipy.signal.find_peaks`.
- Exportação de dados em CSV.
- Relatório técnico em TXT.
- Histórico local em SQLite.
- Botão para resetar extração/reconstrução.
- Exemplos de Absorção, Raman e FTIR.

---

## 5. Escolhas de design

### 5.1 Por que Streamlit?

A aplicação foi construída em **Streamlit** porque o objetivo da atividade era criar uma interface funcional e navegável rapidamente, com upload de arquivos, botões, formulários, abas, gráficos e visualizações.

Alternativas consideradas:

- **FastAPI + React:** arquitetura mais robusta e escalável, mas exigiria mais tempo para implementar frontend e backend.
- **Gradio:** simples para protótipos de IA, mas menos conveniente para uma interface com muitas etapas, abas e parâmetros.
- **Flask puro:** flexível, mas exigiria mais HTML, CSS e JavaScript manual.

Como o foco desta etapa era a **estrutura da interface** e não uma API final de produção, Streamlit foi a alternativa mais adequada.

### 5.2 Por que modularizar em `utils/`?

A lógica da aplicação foi separada em módulos para evitar que todo o projeto ficasse concentrado em `app.py`.

Principais módulos:

```text
utils/
├── image_processing.py
├── calibration.py
├── curve_extraction.py
├── spectra_analysis.py
├── peak_detection.py
├── mock_ai.py
├── report.py
└── database.py
```

Essa estrutura facilita manutenção, testes e substituição futura de partes clássicas por modelos de IA.

### 5.3 Por que usar processamento clássico de imagem?

A atividade exige que nenhum LLM ou modelo de IA seja integrado nesta fase. Por isso, a extração usa técnicas clássicas:

- limiarização;
- filtros de contraste;
- segmentação por cor;
- máscaras;
- componentes conectados;
- rastreamento da curva por coluna;
- interpolação;
- conversão pixel → coordenada real.

A IA deve entrar apenas em uma etapa futura, principalmente para OCR, separação semântica dos elementos do gráfico e interpretação dos espectros.

### 5.4 Por que SQLite?

O SQLite foi escolhido para manter um histórico local simples das análises sem exigir servidor externo de banco de dados. Para um protótipo individual, ele é suficiente e reduz a complexidade de deploy.

### 5.5 Por que Docker no Hugging Face Spaces?

O projeto usa bibliotecas como OpenCV, SciPy, Plotly e Streamlit. O Docker ajuda a controlar o ambiente de execução e reduz problemas de dependência no deploy.

---

## 6. Estrutura do projeto

```text
spectracurve-lab/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .streamlit/
│   └── config.toml
├── utils/
│   ├── image_processing.py
│   ├── calibration.py
│   ├── curve_extraction.py
│   ├── spectra_analysis.py
│   ├── peak_detection.py
│   ├── mock_ai.py
│   ├── report.py
│   └── database.py
├── examples/
│   ├── absorption_example.png
│   ├── raman_example.png
│   └── ftir_example.png
├── docs/
│   ├── prompts_usados.md
│   ├── fluxo_da_aplicacao.md
│   └── limitacoes.md
└── data/
```

---

## 7. Como rodar localmente

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

---

## 8. Como usar

1. Abra a aplicação.
2. Vá em **Upload e configuração**.
3. Envie uma imagem ou carregue um exemplo.
4. Escolha o tipo de espectro.
5. Vá em **Pré-processamento** e ajuste a região útil do gráfico.
6. Vá em **Calibração e extração**.
7. Digite os valores reais dos eixos ou marque os pontos diretamente na imagem.
8. Escolha o método de extração: contraste, cor ou múltiplas curvas.
9. Clique em **Extrair curva**.
10. Confira a máscara detectada e o gráfico reconstruído.
11. Ajuste suavização, normalização ou baseline se necessário.
12. Detecte picos/bandas.
13. Exporte CSV e relatório.
14. Use **Resetar extração/reconstrução** caso queira refazer a etapa de extração sem trocar a imagem.

---

## 9. Deploy

A aplicação foi preparada para deploy em ambiente web usando **Streamlit Community Cloud**

### 9.1 Arquivos necessários na raiz

```text
app.py
README.md
requirements.txt
Dockerfile
.dockerignore
utils/
examples/
docs/
```

### 9.2 Exemplo de execução local com Docker

```bash
docker build -t spectracurve-lab .
docker run -p 7860:7860 spectracurve-lab
```

Depois, acessar:

```text
http://localhost:7860
```

### 9.3 Deploy no Hugging Face Spaces

1. Criar um Space no Hugging Face.
2. Escolher SDK: **Docker**.
3. Subir os arquivos do projeto para o repositório do Space.
4. Aguardar o build.
5. Copiar o link público do Space.
6. Inserir o link em **Links da entrega**.

---

## 10. Uso do agente de codificação com IA

### 10.1 Ferramenta usada

A ferramenta de codificação com IA usada no desenvolvimento foi o **OpenAI Codex**.

O Codex foi usado para:

- gerar a estrutura inicial da aplicação;
- criar componentes da interface em Streamlit;
- modularizar funções em arquivos dentro de `utils/`;
- implementar funções de processamento de imagem;
- criar funções de calibração;
- criar funções de exportação;
- criar o banco SQLite local;
- sugerir correções de bugs;
- refatorar trechos grandes do código;
- melhorar a documentação.

O desenvolvimento foi iterativo. Em vez de pedir a aplicação inteira em um único comando, o processo foi dividido em partes: gerar uma funcionalidade, testar, identificar problemas, pedir correção e ajustar manualmente.

---

## 11. Prompts representativos usados no Codex

Abaixo estão prompts representativos usados durante o desenvolvimento. Eles mostram o tipo de solicitação feita ao agente de codificação.

### Prompt 1 — Estrutura inicial da aplicação

```text
Crie uma aplicação em Streamlit para extrair curvas de espectros científicos a partir de imagens. A aplicação deve ter upload de imagem, seleção do tipo de espectro, abas para pré-processamento, calibração, extração da curva, análise de picos, exportação CSV e relatório. Não integre nenhum modelo de IA ainda; use apenas diagnóstico simulado.
```

**O que funcionou:** o Codex gerou uma primeira versão da interface com abas, upload de arquivo e campos principais.

**O que precisei ajustar manualmente:** reorganizei nomes, adaptei o fluxo para Absorção, Raman e FTIR e revisei textos para uso científico.

---

### Prompt 2 — Modularização

```text
Separe a lógica da aplicação em módulos dentro da pasta utils. Crie arquivos para processamento de imagem, calibração dos eixos, extração da curva, análise espectral, detecção de picos, relatório, banco de dados e diagnóstico simulado. O app.py deve ficar responsável principalmente pela interface.
```

**O que funcionou:** o Codex ajudou a criar uma estrutura mais organizada.

**O que precisei ajustar manualmente:** revisei imports, nomes de funções e dependências entre módulos, porque algumas chamadas ficaram inconsistentes.

---

### Prompt 3 — Extração de curva

```text
Implemente funções para detectar uma curva em uma imagem de gráfico usando contraste e também usando cor. A função deve retornar os pixels da curva e permitir reconstruir a curva em coordenadas reais após a calibração dos eixos.
```

**O que funcionou:** a extração por contraste funcionou bem para gráficos simples com fundo branco.

**O que precisei ajustar manualmente:** ajustei limiares, remoção de ruído, interpolação e casos em que eixos ou grades eram confundidos com a curva.

---

### Prompt 4 — Calibração dos eixos

```text
Adicione uma etapa de calibração em que o usuário informa os valores reais de x mínimo, x máximo, y mínimo e y máximo. A aplicação deve converter coordenadas de pixel para coordenadas reais e funcionar também para FTIR, onde o eixo x pode estar invertido.
```

**O que funcionou:** o Codex gerou a função inicial de conversão pixel → valor real.

**O que precisei ajustar manualmente:** corrigi o eixo y, pois em imagens o eixo vertical cresce para baixo, enquanto nos gráficos científicos o eixo y cresce para cima. Também revisei o caso de FTIR com eixo x invertido.

---

### Prompt 5 — Calibração por pontos

```text
Implemente uma calibração alternativa em que o usuário clique diretamente na imagem para marcar x mínimo, x máximo, y mínimo e y máximo. Depois, ele deve digitar o valor real correspondente a cada ponto. Use esses pontos para definir a área útil do gráfico.
```

**O que funcionou:** o Codex sugeriu uma estrutura usando cliques na imagem e `st.session_state`.


### Prompt 6 — Múltiplas curvas

```text
Adicione suporte para gráficos com múltiplas curvas. A aplicação deve tentar separar curvas por cor e reconstruir todas as curvas detectadas, exportando os dados em formato longo e formato largo.
```

**O que funcionou:** o Codex criou uma estratégia inicial por segmentação de cor.

**O que precisei ajustar manualmente:** a primeira versão confundia tons próximos, misturava legenda com curva e tinha dificuldade com curvas da mesma cor. Testei manualmente imagens de exemplo e ajustei parâmetros e fluxo da interface.

---

### Prompt 7 — Diagnóstico simulado

```text
Crie uma função de diagnóstico simulado para espectros UV-Vis, Raman e FTIR. A função deve receber informações da curva e dos picos detectados e retornar um texto técnico curto, deixando claro que ainda não há integração com IA real.
```

**O que funcionou:** o Codex gerou um mock útil para demonstrar como a IA poderia funcionar futuramente.

---

### Prompt 8 — Histórico em SQLite

```text
Implemente um histórico local em SQLite para salvar análises realizadas, incluindo tipo de espectro, data, parâmetros principais, número de pontos extraídos e resumo do resultado.
```

**O que funcionou:** o Codex criou a estrutura inicial do banco local.

**O que precisei ajustar manualmente:** ajustei o caminho do banco para ser relativo ao projeto e reduzi dependências do diretório em que o terminal foi aberto.

---

### Prompt 9 — Correção de bug em análise espectral

```text
Estou recebendo erro na função summarize_curve ao calcular a área sob a curva. Revise o módulo spectra_analysis.py e corrija a função para calcular área, máximos, mínimos e estatísticas sem quebrar quando os dados estiverem vazios ou incompletos.
```

**O que funcionou:** o Codex sugeriu validações para dados vazios e incompletos.

**O que precisei ajustar manualmente:** revisei os retornos para manter mensagens consistentes na interface.

---

### Prompt 10 — Reset da extração

```text
Adicione um botão para resetar apenas a extração e a reconstrução, sem apagar a imagem carregada nem a calibração dos eixos. O botão deve limpar máscara, curvas extraídas, picos, diagnóstico, relatório e tabelas.
```

**O que funcionou:** o Codex ajudou a identificar chaves do `st.session_state` relacionadas à extração.

---

## 12. Prompts que funcionaram melhor

Os prompts mais eficientes foram os específicos, com:

- arquivo a ser alterado;
- nome da função;
- entradas e saídas esperadas;
- comportamento esperado;
- restrição explícita de não integrar IA real.

Exemplo de prompt eficiente:

```text
No arquivo utils/calibration.py, crie uma função pixel_to_data_coordinates que receba arrays de pixels x/y, dimensões da imagem e os valores reais xmin, xmax, ymin, ymax. A função deve retornar um DataFrame com colunas x e y. Lembre que em imagem o eixo y cresce para baixo.
```

Esse tipo de prompt produziu código mais fácil de testar e integrar.

---

## 13. Prompts que não funcionaram bem

Prompts muito amplos geraram código grande demais e com bugs de integração.

Exemplo de prompt problemático:

```text
Melhore toda a aplicação e deixe a extração perfeita para qualquer gráfico.
```

Problemas observados:

- o Codex tentava alterar muitos arquivos ao mesmo tempo;
- algumas funções mudavam de assinatura sem atualizar todas as chamadas;
- apareciam parâmetros duplicados;
- a interface ficava com controles demais;
- uma melhoria em um exemplo piorava outro caso.

Depois disso, passei a solicitar mudanças menores e mais verificáveis.

---
### 14 Decisões científicas

Foram definidas manualmente as categorias de espectro mais relevantes:

- Absorção / UV-Vis;
- Raman;
- FTIR.

Também foram escolhidos manualmente os tipos de análise do protótipo:

- detecção de picos;
- cálculo de área;
- normalização;
- suavização;
- baseline simples;
- exportação CSV.

### 14.3 Testes com imagens

Os testes de qualidade da extração foram feitos manualmente usando imagens de exemplo. Em cada teste, verifiquei se a aplicação:

- pegava a curva correta;
- confundia eixo com curva;
- confundia legenda com curva;
- perdia partes da curva;
- reconstruía a curva com escala invertida;
- exportava dados coerentes.

### 14.4 Ajustes de interface

Foram feitos ajustes manuais em:

- nomes das abas;
- textos explicativos;
- ordem dos controles;
- mensagens de aviso;
- botões de reset;
- instruções para calibração por pontos;
- organização dos resultados.

### 14.5 Correções de bugs

Algumas correções exigiram intervenção manual, principalmente quando o erro dependia do fluxo completo da aplicação. Exemplos:

- limpar corretamente o `st.session_state`;
- evitar usar calibração antiga em uma imagem nova;
- manter a imagem carregada após resetar apenas a extração;
- corrigir casos de curva vazia;
- impedir que o app quebrasse quando nenhum pico fosse detectado;
- ajustar FTIR com eixo x decrescente;
- revisar a separação de múltiplas curvas;
- evitar que legenda fosse tratada como curva.

### 14.6 Documentação

A organização final do README e a análise crítica do processo foram feitas manualmente, com apoio de IA apenas para estruturação e revisão textual.

---

## 15. O que funcionou bem

### 15.1 Interface inicial

O Codex foi eficiente para gerar rapidamente a estrutura inicial em Streamlit, incluindo upload, abas, botões e visualizações.

### 15.2 Modularização

A separação em módulos funcionou bem. Isso facilitou a manutenção e permitiu pedir melhorias específicas para arquivos específicos.

### 15.3 Exportação e relatório

As funções de exportação CSV e relatório foram simples de implementar com apoio do Codex.

### 15.4 Detecção de picos

A integração com `scipy.signal.find_peaks` funcionou bem como protótipo. Os parâmetros de distância, proeminência e altura são ajustáveis pelo usuário.

### 15.5 Mock de IA

A criação do diagnóstico simulado foi adequada para esta fase da atividade, pois demonstra onde a IA entraria sem violar a restrição de não integrar modelo real.

### 15.6 Iteração com o agente

O uso mais produtivo do Codex ocorreu quando o desenvolvimento foi quebrado em tarefas pequenas, como corrigir uma função específica ou adicionar um controle isolado na interface.

---

## 16. O que não funcionou perfeitamente

### 16.1 Extração de curvas complexas

A maior dificuldade foi extrair corretamente curvas em imagens com:

- baixa resolução;
- linhas muito finas;
- múltiplas curvas próximas;
- curvas da mesma cor;
- legenda sobreposta;
- linhas de grade;
- texto próximo à curva;
- antialiasing forte.

Nesses casos, a segmentação por cor ou contraste pode confundir objetos da imagem.

## 17. Histórico de desenvolvimento e commits

Para evidenciar o uso incremental do agente de codificação, o repositório deve conter commits ao longo do desenvolvimento, por exemplo:

```text
1. Estrutura inicial do app Streamlit
2. Modularização em utils
3. Implementação da calibração
4. Extração de curva por contraste e cor
5. Múltiplas curvas e exportação CSV
6. Diagnóstico simulado e relatório
7. Correções de bugs e README final
8. Deploy no Hugging Face Spaces
```

Essa seção descreve a organização esperada do histórico. O histórico real pode ser conferido na aba de commits do repositório GitHub.

---

## 18. Limitações atuais

- A aplicação não usa OCR.
- A aplicação não integra LLM.
- A aplicação não usa modelo generativo.
- A separação automática de múltiplas curvas ainda pode falhar em casos difíceis.
- A extração pode confundir linhas de grade, eixos, legenda ou texto com a curva.
- Imagens de baixa resolução reduzem bastante a qualidade da reconstrução.
- FTIR em transmitância pode exigir análise de mínimos em vez de máximos.
- O histórico SQLite local pode não ser persistente em ambientes de deploy temporários.

---

## 19. Próximos passos

- Calibração automática por OCR.
- Detecção automática da área útil do gráfico.
- Extração mais robusta de múltiplas curvas.
- Ferramenta manual para corrigir pontos da curva extraída.
- Comparação com CSV de referência.
- Interpretação com agente de IA.
- Geração de relatórios científicos mais completos.
- Integração com bases de dados espectroscópicas.
- Salvamento persistente em nuvem.
- Testes automatizados para imagens de exemplo.

---

## 21. Observação final sobre o uso de IA

A IA foi usada como **agente de codificação**, não como funcionalidade final da aplicação. Portanto:

- o usuário final da aplicação não conversa com um LLM;
- não há chamada à API de IA;
- não há geração real de interpretação científica por modelo;
- o diagnóstico apresentado é simulado;
- o foco da entrega é a interface funcional, o fluxo da aplicação e a documentação do processo de desenvolvimento com auxílio do Codex.
