# LoteIA — Plano de Execução

**Projeto final de LIA — Engenharia de Computação**
Período: 3 de junho → 22 de junho · Grupo: individual
*(Nome "LoteIA" é provisório — Lote + IA. Troca fácil, é só um find-and-replace.)*

---

## 0. O que é isto e por que ele se destaca

LoteIA é um **simulador probabilístico de viabilidade de loteamentos** (condomínios
horizontais / subdivisão de glebas em lotes). Dado um terreno candidato — sua
localização, área e parâmetros do projeto —, o sistema estima quanto os lotes
valem, propaga a incerteza dessa estimativa por um modelo financeiro e devolve
**não um número, mas uma probabilidade**: "este empreendimento tem X% de chance
de superar a rentabilidade-alvo".

O diferencial frente a um projeto acadêmico comum:

1. **É real.** O domínio é o core business da Opes Data (estudos de viabilidade
   econômico-financeira de empreendimentos imobiliários), empresa onde o autor
   estagia. O pitch deixa de ser "exercício hipotético" e vira "ferramenta
   relevante para uma empresa real, num dos maiores mercados imobiliários do
   país". A Opes entra como **usuário-alvo e validador de domínio** — nenhum dado
   proprietário é usado; o projeto roda 100% em dado público.

2. **A profundidade joga na força do autor.** O coração do projeto é modelagem
   estocástica (propagação de incerteza, simulação de Monte Carlo, otimização) —
   modelar problema complexo de engenharia, não estatística pura.

3. **Honestidade metodológica embutida.** O projeto não finge prever o sucesso de
   um empreendimento (não existe dado público de resultado). Ele **simula**
   viabilidade a partir de inputs estimados — uma distinção que é, por si só, um
   ponto forte de banca (seção 2).

**Princípio de execução:** fatia vertical primeiro. Ao fim da Fase 1 já existe um
sistema rodando de ponta a ponta (versão mínima). As fases seguintes aprofundam
cada camada sem nunca quebrar o que já funciona.

---

## 2. A decisão metodológica central — não prevemos sucesso, simulamos viabilidade

**Leia antes de escrever qualquer linha de modelo.** Esta seção é o coração da
defensabilidade.

O instinto ingênuo seria "treinar um modelo que prevê se um empreendimento dá
certo". Isso é **impossível e indefensável** com dado público: não existe base
rotulada de "loteamentos que deram certo × que deram errado" — esse dado é
proprietário das incorporadoras. Um modelo treinado assim seria um chute
disfarçado.

A abordagem correta inverte o problema. A viabilidade financeira **não é uma
coisa a prever — é uma conta determinística** (fluxo de caixa → TIR/VPL). O que é
genuinamente incerto são os *inputs* dessa conta, principalmente o **preço de
venda do lote** e a **velocidade de absorção**. Então:

> **O ML estima os inputs incertos (com sua incerteza). O motor financeiro
> calcula a viabilidade. A simulação de Monte Carlo propaga a incerteza dos
> inputs até a saída, transformando-a numa distribuição de probabilidade.**

Isso é honesto, é defensável, e não precisa de nenhum dado de resultado. Quando a
banca perguntar "como você validou que o empreendimento é viável?", a resposta é
"eu não afirmo que é — eu quantifico a probabilidade dele ser, dada a incerteza
do mercado, e mostro de que variável essa probabilidade mais depende".

---

## 3. As 6 camadas de profundidade

O **núcleo** é a fatia vertical: um modelo de preço pontual + um cálculo de
viabilidade pontual + API + front mínimo. As 6 camadas o transformam de trivial
em projeto de destaque.

- **Camada 1 — Incerteza no preço.** Em vez de prever "o m² do lote vale R$ X", o
  modelo prevê uma *distribuição* (regressão quantílica, ou predição conformal
  para um intervalo com cobertura garantida). É o análogo da calibração: não
  basta o número, é preciso saber quão confiável ele é.

- **Camada 2 — Simulação de Monte Carlo da viabilidade.** O coração de
  engenharia. Amostra-se milhares de cenários a partir da incerteza dos inputs
  (preço do lote, absorção, custo de infraestrutura) e roda-se o fluxo de caixa
  do loteamento em cada um → distribuição de TIR e VPL → **probabilidade de
  superar a rentabilidade-alvo**. Substitui a "matriz de custo" do ChurnGuard por
  algo mais rico: decisão sob incerteza.

- **Camada 3 — Explicabilidade.** SHAP no modelo de preço (o que faz o m² do lote
  subir: localização, infraestrutura do entorno, área). E **análise de
  sensibilidade** na saída financeira (a viabilidade depende mais do preço de
  venda? da velocidade de absorção? do custo de obra?). Mostra que o sistema não
  é caixa-preta.

- **Camada 4 — Orquestração e produto (n8n).** O incorporador entra com um
  terreno (via formulário ou Google Sheets) → o fluxo pontua → devolve um
  relatório probabilístico e dispara alerta. Zona de conforto do autor, baixo
  risco, fica perto do fim.

- **Camada 5 — Enriquecimento geoespacial dos dados (experimento).** O análogo da
  camada de dados sintéticos do ChurnGuard, mas mais defensável: enriquecer o
  dataset com features geoespaciais (distância a vias principais, comércio,
  escolas via OpenStreetMap; renda por setor censitário via IBGE) e medir, contra
  uma baseline, se isso melhora o modelo de preço. Critério de sucesso definido
  antes; resultado negativo bem conduzido é entrega válida.

- **Camada 6 — Otimização (highest-and-best-use).** A camada mais difícil e a
  primeira a ser cortada se faltar tempo. Dado um terreno, qual **configuração**
  do loteamento (número e tamanho dos lotes, faseamento da venda) maximiza a
  probabilidade de retorno? Transforma o produto de avaliador em otimizador.

---

## 4. O modelo financeiro do loteamento (primer)

Para quem vem da engenharia e não das finanças: o fluxo de caixa de um loteamento
é um problema de modelagem temporal, não de contabilidade complexa.

- **Receita (VGV).** Soma do preço de venda de todos os lotes, *distribuída no
  tempo* conforme a velocidade de absorção (quão rápido os lotes vendem) e muitas
  vezes *parcelada* (o comprador paga em prestações — daí a "carteira de
  recebíveis").
- **Custos.** Aquisição da gleba + infraestrutura (terraplenagem, vias, drenagem,
  redes de água/esgoto/energia, paisagismo) + licenciamento + marketing/comissões
  + impostos + administração.
- **Indicadores.** Como receitas e custos ocorrem em momentos diferentes, traz-se
  tudo a valor presente: **VPL** (valor presente líquido — soma descontada) e
  **TIR** (taxa interna de retorno — a rentabilidade efetiva do projeto). A
  viabilidade é, essencialmente, "a TIR supera a rentabilidade-alvo / o VPL é
  positivo?".

As fórmulas são fechadas e simples. A dificuldade — e o mérito — está em modelar
a *incerteza* de cada input e propagá-la (Camada 2), não na aritmética financeira
em si.

---

## 5. Dados — DECISÃO TRAVADA: São Paulo (ITBI ⨝ IPTU)

*(Risco nº 1 resolvido. O caminho de dado está definido; segue o pipeline.)*

O lado da **receita** (preço de venda do lote por m²) vem de **dado público de
transação real de São Paulo**, cruzando duas bases abertas da Prefeitura por uma
chave comum, o **SQL** (Setor-Quadra-Lote):

- **Transações de ITBI** (Secretaria da Fazenda, mensal desde 2019): cada linha é
  uma transação paga, com SQL, endereço, **valor da transação** e cartório. Nomes
  das partes são omitidos (sigilo fiscal — sem problema de privacidade). Falta a
  área e o tipo do imóvel.
- **Cadastro do IPTU** (GeoSampa / Base dos Dados): traz **área do terreno**, área
  construída, endereço, zoneamento e o SQL, em download aberto único.

**Pipeline de ETL (Fase 1):**

1. Baixar e empilhar os arquivos anuais de ITBI (2019→2025) → valor + SQL + endereço.
2. Juntar com o cadastro do IPTU pelo SQL → anexa área do terreno, área
   construída e zoneamento.
3. **Isolar terrenos**: filtrar área construída ≈ 0 (transações de terra pura).
4. Alvo do modelo: **preço/m² de terreno = valor da transação ÷ área do terreno**.
5. Geocodificar (GeoSampa é georreferenciado) para as features espaciais da Camada 5.

**O lado do custo** (aquisição de gleba + infraestrutura) **não vem de dado de
treino** — entra como input incerto na Monte Carlo (Camada 2) e é interrogado pela
análise de sensibilidade (Camada 3). Referências de custo de infraestrutura a
definir na Fase 2 (SINAPI e benchmarks de custo por lote).

**Dois pontos que isso destrava:**

- **Validação temporal honesta.** Com dado desde 2019, treina-se no passado
  (≤2023) e testa-se no futuro (≥2024) — validação que o Telco do ChurnGuard nunca
  permitiu, e forte argumento de banca.
- **O papel da Opes.** SP não é Goiânia; esse foi o trade-off aceito. A Opes
  reentra como **validadora de domínio** e usuária-alvo: a metodologia roda em dado
  real de SP e é interpretada com o conhecimento do mercado onde o autor atua.
  Nenhum dado proprietário é usado.

*(Alternativas equivalentes, se necessário: BH, Recife e Porto Alegre também abrem
ITBI. SP foi escolhida por ser a mais completa e pelo cruzamento ITBI+IPTU
confirmado.)*

---

## 6. Arquitetura de componentes

```
CAMADA DE DADOS
  Dados públicos de preço de lote/terreno
    + enriquecimento geoespacial (OSM, IBGE)   [Camada 5]
            ▼
CAMADA DE ML (offline / treino)
  • Modelo de preço/m² com incerteza (.joblib)  [núcleo + Camada 1]
  • Explicador SHAP                              [Camada 3]
            ▼
CAMADA DE SIMULAÇÃO
  Motor de fluxo de caixa do loteamento
    + Monte Carlo (TIR/VPL → probabilidade)      [Camada 2]
    + otimizador de configuração                 [Camada 6]
            ▼
CAMADA DE SERVIÇO — FastAPI
  POST /viabilidade → { prob_retorno, faixa_preco,
                        distribuicao_tir, fatores_shap, sensibilidade }
            ▼
CAMADA DE ORQUESTRAÇÃO — n8n                      [Camada 4]
  Entrada de terreno (Sheets/form) → pontua →
  gera relatório → alerta
            ▼
CAMADA DE APRESENTAÇÃO — front (Lovable/Streamlit)
  Entrada do terreno · distribuição de retorno ·
  fatores de preço · análise de sensibilidade
```

---

## 7. Workflow do n8n (Camada 4)

1. **Trigger** — nova linha numa Google Sheet de terrenos candidatos (ou webhook
   de um formulário).
2. **HTTP Request → `POST /viabilidade`** — envia os parâmetros do terreno.
3. **Formatação** — monta o relatório (probabilidade de retorno, faixa de preço
   estimada, variável mais sensível).
4. **Grava** o resultado de volta na planilha / banco.
5. **Notifica** — alerta com o veredito ("Terreno X: 78% de chance de superar a
   meta; resultado mais sensível à velocidade de absorção").

---

## 8. Cronograma (3–22 de junho)

### Fase 1 — Fatia vertical (3–8 jun)
*O sistema mínimo roda de ponta a ponta.*
- **Extrair e cruzar ITBI ⨝ IPTU de São Paulo pelo SQL** (pipeline na seção 5) — fazer primeiro.
- EDA do dataset de preços; modelo de preço baseline.
- Motor financeiro mínimo (VGV vs custo → TIR/VPL pontual).
- FastAPI mínima `/viabilidade`; front esqueleto.
- **Marco:** entra terreno, sai um veredito de viabilidade pontual.

### Fase 2 — Profundidade (9–14 jun)
*O modelo deixa de ser trivial.*
- **Camada 1:** incerteza no preço (quantílica/conformal).
- **Camada 2:** Monte Carlo → distribuição de TIR/VPL → probabilidade.
- **Camada 3:** SHAP no preço + análise de sensibilidade financeira.
- **Marco:** a saída vira probabilística e explicável.

### Fase 3 — Camadas experimentais + integração (15–19 jun)
*Os diferenciais e a automação.*
- **Camada 5:** enriquecimento geoespacial (experimento controlado).
- **Camada 6:** otimização de configuração (se o tempo permitir).
- **Camada 4:** workflow n8n completo + front final.
- **Marco:** sistema completo, automatizado.

### Finalização (20–22 jun)
- Relatório / documentação.
- Pitch: problema → decisão metodológica → demo → resultado.
- Ensaio. Margem para imprevistos.

---

## 9. Riscos e mitigação

- **Fonte de dados (ex-risco nº 1) — RESOLVIDO.** Definido: SP, cruzando ITBI ⨝
  IPTU pelo SQL (seção 5). Risco residual baixo: validar a taxa de cobertura do
  join e o volume de terrenos após o filtro de área construída ≈ 0.
- **Camada 6 (otimização) é a mais difícil.** Mitigação: é a última e a primeira a
  ser cortada; o sistema é completo e apresentável sem ela.
- **ML "só" um regressor de preço.** Mitigação: a profundidade está deliberadamente
  na incerteza (C1) + simulação (C2) + sensibilidade (C3), não no regressor cru.
- **Custo de infraestrutura de loteamento mal estimado.** Mitigação: tratar como
  input incerto na Monte Carlo e reportar via análise de sensibilidade.
- **Atraso.** Mitigação: a fatia vertical da Fase 1 garante um entregável sempre.

---

## 10. Checklist de entregáveis

- [x] Fonte de dados definida (SP: ITBI ⨝ IPTU) — [ ] extração e EDA documentadas
- [ ] Modelo de preço/m² comparado contra baseline
- [ ] Incerteza no preço (intervalo/distribuição) — C1
- [ ] Motor financeiro do loteamento (VGV, fluxo, TIR/VPL)
- [ ] Simulação de Monte Carlo → probabilidade de retorno — C2
- [ ] SHAP (preço) + análise de sensibilidade (financeiro) — C3
- [ ] API FastAPI servindo a viabilidade completa
- [ ] Workflow n8n + entrada via Sheets/form — C4
- [ ] Experimento de enriquecimento geoespacial documentado — C5
- [ ] Otimizador de configuração — C6 (opcional)
- [ ] Front-end (Lovable/Streamlit)
- [ ] Relatório e pitch
