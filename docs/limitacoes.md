# Limitações atuais

## Extração da imagem

- O reconhecimento de eixos é manual.
- A aplicação não lê automaticamente números dos ticks.
- Imagens com baixa resolução prejudicam a segmentação.
- Curvas sobrepostas, cruzadas ou com a mesma cor exigem ajustes manuais.
- Grade, legenda, eixos e textos podem contaminar a máscara.

## Interpretação com Ollama

- A resposta depende da qualidade da curva e dos picos calculados.
- Modelos locais pequenos podem ignorar ferramentas ou produzir análise superficial.
- A aplicação utiliza um fallback determinístico quando nenhuma tool é chamada.
- O modelo não possui acesso a uma biblioteca espectroscópica validada.
- Sem estrutura molecular, composição ou referência, atribuições permanecem hipóteses.
- Prompt injection pode ser mitigado, mas não eliminado completamente.
- A velocidade depende do hardware e do tamanho do modelo.
- Um servidor Ollama precisa estar em execução e o modelo precisa estar instalado.

## Múltiplas curvas

- A seleção manual funciona melhor quando as curvas têm cores visualmente distintas.
- Tolerância CIELAB muito baixa pode criar lacunas; tolerância muito alta pode incluir legenda ou símbolos da mesma cor.
- A atribuição pela cor evita sobreposição de máscaras, mas curvas com exatamente a mesma cor ainda exigem o modo por componentes.
- O modo por componentes pode falhar quando a curva está fragmentada.
- Legendas ou marcadores com a mesma cor da curva podem contaminar a máscara se estiverem dentro da região útil.

## Deploy

- `localhost:11434` só funciona quando Ollama e Streamlit estão na mesma máquina.
- Um deploy público precisa incluir o Ollama ou acessar um servidor remoto.
- A execução local é a abordagem principal da avaliação.

## Limitação científica

O sistema é uma ferramenta de apoio. A interpretação gerada não substitui o dado bruto, referências espectrais, cálculos teóricos ou avaliação por especialista.
