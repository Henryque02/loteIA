# LoteIA — Documento de Apresentação

> *Vale a pena desenvolver este terreno?* — e qual a **chance** de a resposta ser sim.

Documento-guia do projeto para a banca e para mim, como roteiro. Foca no
**objetivo** e em **como cada peça contribui** — não no código. A narrativa é a
espinha (problema → dado → preço → incerteza → finanças → produto); ao lado de cada
peça vai o selo da **Camada (C1–C6)** da disciplina. A tabela final amarra os dois.

---

## 1. O que é e por que existe

**O problema.** Decidir comprar uma gleba e lotear é uma decisão de milhões. Hoje
ela costuma ser resumida a **um número seco** — um VPL ou uma TIR de planilha. Esse
número esconde a única coisa que importa de verdade: a **incerteza**. O preço de
venda do lote não é conhecido, é uma *aposta*; a velocidade de venda é uma aposta; o
custo da gleba e da infraestrutura também. Uma planilha entrega "VPL = R$ X" como se
fosse um fato, quando na prática é o centro de uma nuvem larga de resultados
possíveis. O incorporador decide no escuro achando que está no claro.

**A proposta.** O LoteIA troca o número seco por uma **probabilidade**: *qual a
chance de o empreendimento superar a rentabilidade-alvo?* Em vez de "a TIR é 15%",
ele responde "há 77% de chance de a TIR superar a sua meta". É uma resposta honesta
sobre risco, no formato em que a decisão realmente é tomada.

**O diferencial.** A maioria das ferramentas tenta prever "sucesso" diretamente com
um modelo. Isso é indefensável — não existe dado público de resultado de
empreendimento para treinar tal modelo. O LoteIA faz o oposto: o **Machine Learning
modela só o que dá para medir** (o preço de terreno por m², com dado público de
verdade), e a viabilidade é **calculada** (fluxo de caixa) e **simulada** (Monte
Carlo). Cada elo é auditável. É um simulador, não um adivinhador.

---

## 2. A ideia central em uma frase + o fluxo

> **Estimar o preço do terreno com incerteza e propagar essa incerteza por um modelo
> financeiro, para responder com uma probabilidade — não com um chute de número.**

O fluxo, de ponta a ponta:

```
   DADO PÚBLICO            PREÇO/m²            PREÇO DO LOTE        FLUXO DE CAIXA
  (ITBI, GeoSampa,   →   com INCERTEZA    →   (área × preço,   →   (receita no tempo
   IBGE, OSM)            (faixa, não ponto)    com a faixa)        − gleba − infra)
                              │ C1                                       │
                              ▼                                          ▼
                                         MONTE CARLO  ───────────►  PROBABILIDADE
                                    (sorteia preço, absorção,      de superar a
                                     custos milhares de vezes)     meta de retorno
                                              │ C2                       │
                                              ▼                          ▼
                                        distribuição de            "77% de chance"
                                          TIR / VPL
```

Lendo em uma frase: **dado público → preço/m² com incerteza → preço do lote → fluxo
de caixa → Monte Carlo → probabilidade.** Tudo o que vem depois (geo, SHAP, tornado,
otimizador, automação) ou *melhora* esse fluxo ou o *explica*.

---

## 3. As regras que tornam o projeto defensável

São as decisões metodológicas que sustentam o projeto diante de uma banca. Cada uma
fecha uma porta de crítica.

1. **O ML modela apenas o preço/m² de terreno (o lado da receita).** Viabilidade e
   "sucesso do empreendimento" **nunca** são alvo de ML — são *calculados* e
   *simulados*. Não há dado público de resultado de empreendimento; inventar um alvo
   seria fabricar a resposta.

2. **Validação separada por tempo (walk-forward).** Treina com transações de
   2023+2024 e testa em 2025 — o ano mais novo e fechado. Nunca se embaralha o tempo
   aleatoriamente: isso seria vazamento temporal (o modelo "veria o futuro") e
   destruiria a defensabilidade. O número que reportamos é honesto porque foi medido
   **fora do período de treino**.

3. **Isolar o terreno antes de calcular o preço/m².** Imóvel com construção não é
   terra pura — o valor embute a benfeitoria. Filtramos para área construída ≈ 0
   *antes* de derivar o alvo, senão o preço/m² fica contaminado.

4. **Só dado público.** ITBI/IPTU de São Paulo, GeoSampa, IBGE, OpenStreetMap. Nada
   proprietário. O projeto é reproduzível por qualquer um com as mesmas fontes.

5. **Incerteza sempre.** O modelo de preço reporta um intervalo/distribuição, não só
   um ponto. A saída de viabilidade é probabilística, não um número seco. Essa é a
   tese inteira condensada em uma regra.

---

## 4. Como cada peça contribui

A narrativa, peça por peça, com o selo da Camada onde se aplica.

### Os dados — a matéria-prima
As **guias de ITBI pagas** da Prefeitura de SP (2023, 2024, 2025) trazem cada
transação imobiliária real: valor, área do terreno, área construída, uso e a chave
cadastral **SQL** (Setor-Quadra-Lote). Filtrando terrenos puros e dividindo valor por
área, obtemos o **preço/m² observado** — o alvo do modelo. A camada geográfica
(GeoSampa, IBGE, OSM) anexa contexto a cada lote pela SQL. **231 mil** transações em
2025 dão lastro estatístico.

### O modelo de preço/m² — o coração da receita
Um modelo de regressão aprende o preço/m² de terreno a partir de atributos do lote e
do entorno: área, testada, distância ao centro e à estação, comércio e escolas no
raio de 1 km, renda do setor, posição (x, y) e zona. É a única coisa que o ML faz — e
faz bem o suficiente para bater um baseline forte (ver Seção 5).

### `C1` — Incerteza com CQR (ponto → faixa)
Um modelo que diz "R$ 1.000/m²" e ponto final é frágil. Aqui usamos **regressão
quantílica conformalizada (CQR, via MAPIE)**: a saída vira uma **faixa** (ex.: R$
700–1.350/m²) com cobertura calibrada. Em vez de um ponto, entregamos a incerteza do
preço — que é exatamente o que o Monte Carlo precisa consumir.

### `C5` — Enriquecimento geográfico (o experimento que prova valor)
A Camada 5 testa uma hipótese: *atributos geográficos melhoram a previsão de preço?*
O experimento compara o modelo só-cadastro contra o modelo geo+zona, **no mesmo
holdout temporal**. Resultado: o geo **reduz o erro (MAE) em 10,7%** — passa o
critério de ≥5% e justifica a complexidade extra. É ciência aplicada, não enfeite.

### `C3` — Explicabilidade com SHAP (por que esse preço?)
Para cada estimativa, o **SHAP** decompõe o preço previsto nos fatores que o
empurraram para cima ou para baixo (proximidade de estação, renda do setor, área…).
Transforma o modelo de caixa-preta em algo que o avaliador entende e em que confia.

### O motor financeiro — do preço ao VPL/TIR
O preço do lote (área × preço/m²) vira **receita distribuída no tempo** conforme a
absorção (ritmo de venda). Subtraem-se **gleba, infraestrutura** e custos
operacionais (comissão, impostos, marketing, admin, licenciamento). Daí saem **VPL** e
**TIR** — a linguagem financeira da decisão.

### `C2` — Monte Carlo (propaga a incerteza → distribuição)
Aqui a tese se realiza. Cada input incerto é uma **distribuição** (preço como faixa
do CQR; absorção e custos como triangulares min–moda–max). O motor sorteia milhares
de cenários e produz uma **distribuição de TIR/VPL** — e daí a **probabilidade de
superar a meta**. Preço e velocidade de venda são acoplados por uma **cópula
gaussiana** (`rho_mercado`): mercado quente = preço alto *e* venda rápida juntos,
porque na vida real eles andam juntos. O número seco vira uma nuvem com chance.

### `C3` — Sensibilidade / tornado (o que mais mexe?)
Uma análise **OAT (one-at-a-time)** varia cada input isolado e mede o impacto no
resultado, ordenando num **gráfico de tornado**. Responde "onde devo focar a
due diligence?" — talvez o custo da gleba mexa muito mais o resultado que a taxa de
venda. Orienta a decisão para o que realmente pesa.

### `C6` — Otimização do melhor uso (HBU)
*Highest-and-best-use*: dada a área vendável, qual **configuração de lotes** (número e
tamanho) **maximiza a probabilidade de retorno**? O otimizador liga o modelo de preço
(o preço/m² muda com o tamanho do lote) ao Monte Carlo e busca a melhor configuração.
Sai de "este projeto é viável?" para "qual o **melhor** projeto possível aqui?".

### `C4` — O produto (API + Streamlit + n8n)
Tudo isso vira uma ferramenta usável:
- **API (FastAPI)** — `POST /viabilidade` (a probabilidade + faixas + tornado + SHAP),
  `POST /otimizar` (HBU), `GET /quadra/{setor}/{quadra}` (localização no mapa),
  `GET /health`. Carrega os modelos uma vez na inicialização.
- **Streamlit** — painel do analista (mexer em `rho`, custos, meta) e **modo
  relatório** (abre já preenchido a partir de um link).
- **n8n** — automação ponta a ponta: o cliente preenche um **Google Form**, o fluxo
  valida, chama a API e devolve por **e-mail** o veredito + um **link para o relatório
  interativo**. O cliente não toca em código nem em planilha.

---

## 5. Validação e resultados honestos

Tudo medido **walk-forward**: treino em **2023+2024**, teste no holdout de **2025**.

| Métrica (holdout 2025) | Baseline (mediana por setor) | Modelo geo+zona | Ganho |
|---|---|---|---|
| **MAE** (erro médio, R$/m²) | 1.533 | **1.369** | **−10,7%** |
| **MAPE mediana** | 42,2% | **35,7%** | melhor |
| **R²** | 0,222 | **0,252** | melhor |

O baseline não é fraco de propósito — é a mediana de preço por setor, uma referência
dura de bater em imóveis. Superá-lo em **10,7% de MAE fora do tempo de treino** é um
ganho real e defensável.

**Cobertura do intervalo (CQR).** Intervalo nominal de **80%**; cobertura empírica
medida em 2025: **73,4%**. À primeira vista parece "abaixo do alvo" — mas é
exatamente o sinal de honestidade. A garantia conformal vale sob *trocabilidade*
(distribuição estável); ao validar num **ano novo**, o mercado se deslocou e a
cobertura cai um pouco. Reportar **73,4% medido fora do tempo** é mais defensável do
que exibir "80% perfeito" calibrado no próprio ano de teste — isso seria vazamento
disfarçado de qualidade. A leve subcobertura é uma **força metodológica**: prova que
o número não foi cozinhado.

---

## 6. Limitações (ditas antes que a banca pergunte)

- **Não existe dado público de resultado de empreendimento.** Por isso a viabilidade
  é *calculada e simulada*, nunca prevista por ML. É uma escolha de honestidade, e
  também o teto do que dá para fazer só com dado público.
- **Custos (gleba, infra) são inputs incertos, não previstos.** O usuário os fornece
  como faixas; o modelo não os adivinha. Eles entram no Monte Carlo como distribuições,
  não como features de treino — misturar os dois lados contaminaria o modelo de preço.
- **A cobertura do intervalo cai fora do tempo.** Esperado e assumido (Seção 5);
  recalibrar para "80% bonito" exigiria usar 2025 na calibração, o que seria vazamento.
- **Geografia limitada a São Paulo.** As fontes (ITBI, GeoSampa, zoneamento) são
  municipais; estender para outra cidade exige refazer a camada de dados.
- **Anos anteriores a 2023 ficaram de fora.** Os arquivos antigos de ITBI
  embutem um snapshot cadastral tardio, que enviesaria o isolamento de terreno (viés
  de sobrevivência).

---

## 7. Roteiro de fala / como apresentar

**Antes de começar (na minha máquina):** subir os dois serviços. Os artefatos já
estão treinados localmente, então basta:

```
make api     # FastAPI em localhost:8000
make front   # Streamlit em localhost:8501 (noutro terminal)
```

**Ordem da demo:**

1. **Abrir com a tese (30s).** "Comprar uma gleba é uma decisão de milhões decidida
   com um número seco que esconde o risco. O LoteIA troca esse número por uma
   probabilidade."
2. **A visão do cliente.** Abrir o **relatório** (o link que o n8n mandaria): mapa da
   quadra, a **probabilidade de viabilidade**, os histogramas de TIR/VPL. "Foi isso
   que ele recebeu por e-mail, sem preencher nada além de um formulário."
3. **A visão do analista (onde mora o valor).** No painel do Streamlit:
   - mexer no **`rho_mercado`** e mostrar a distribuição mudar — "é a incerteza de
     mercado entrando no resultado";
   - mostrar o **tornado** — "o que mais move o resultado é X, não Y";
   - rodar o **otimizador (HBU)** — "qual o melhor tamanho de lote aqui".
4. **Fechar com a honestidade.** O modelo geo bate o baseline em **10,7% fora do
   tempo**; o intervalo cobre **73,4%** e *por que isso é uma força*. "Cada elo é
   público e auditável. É um simulador, não um adivinhador."

**Frases-âncora:**
- "Uma probabilidade, não um número seco."
- "O ML só precifica o terreno; a viabilidade é simulada."
- "Validado no futuro (2025), não no próprio período de treino."
- "Incerteza não é um detalhe — é o produto."

---

## 8. Mapa Camadas C1–C6 ↔ seções

| Camada | O que entrega | Onde aparece neste documento |
|---|---|---|
| **C1** | Incerteza do preço (CQR / intervalo) | §2 (fluxo), §4 (Incerteza CQR), §5 (cobertura 73,4%) |
| **C2** | Monte Carlo → distribuição e probabilidade | §1–2 (a tese), §4 (Monte Carlo + `rho_mercado`) |
| **C3** | Explicabilidade (SHAP) e sensibilidade (tornado) | §4 (SHAP; tornado), §7 (demo do tornado) |
| **C4** | Produto/automação (API + Streamlit + n8n) | §4 (o produto), §7 (roteiro da demo) |
| **C5** | Enriquecimento geográfico (experimento) | §4 (geo), §5 (−10,7% MAE) |
| **C6** | Otimização do melhor uso (HBU) | §4 (HBU), §7 (demo do otimizador) |

E as bases que sustentam tudo (não são "Camada" numerada, mas são o alicerce): os
**dados públicos** (§4, regra 4), o **modelo de preço/m²** (§4, regra 1) e a
**validação temporal** (§3 regra 2, §5).

---

*Fonte dos números: artefatos `models/preco_m2.joblib` e `preco_m2_quantilico.joblib`
(re-treino walk-forward 2023+2024 → 2025). Regras e arquitetura: `CLAUDE.md`.*
