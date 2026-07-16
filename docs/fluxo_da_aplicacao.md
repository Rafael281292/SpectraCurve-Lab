# Fluxo da aplicação

1. O usuário envia uma imagem de espectro.
2. Seleciona Absorção / UV-Vis, Raman ou FTIR.
3. Ajusta a região útil do gráfico.
4. Marca os pontos de calibração e informa os valores reais dos eixos.
5. Para múltiplas curvas, o usuário seleciona várias cores ou usa detecção automática.
6. A aplicação cria uma máscara independente para cada cor e armazena nome, RGB e HEX.
7. Os pixels de cada máscara são convertidos em dados x,y.
8. O usuário aplica pós-processamento opcional.
9. A aplicação reconstrói o espectro.
10. O algoritmo detecta picos ou bandas.
11. O usuário faz uma pergunta ao assistente.
12. O modelo local do Ollama seleciona tools.
13. As tools calculam e retornam JSON determinístico.
14. O modelo gera uma interpretação cautelosa.
15. A interface mostra resposta, rastreamento, tokens e duração.
16. O sistema gera relatório e permite exportar dados.

## Calibração por pontos

O usuário marca x mínimo, x máximo, y mínimo e y máximo diretamente na imagem e informa os valores correspondentes. Esses pontos também delimitam a região útil para conversão dos pixels em coordenadas científicas.
