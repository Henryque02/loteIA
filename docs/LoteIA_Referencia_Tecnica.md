# LoteIA — Referência técnica (Q&A da banca)

> Documento de apoio, mais técnico, para perguntas da banca. Cobre os dados, as
> features do modelo de preço, o que varia no Monte Carlo e o que o VPL significa.
> Ordem: dados → modelo → simulação → leitura do resultado.

## 1. Os dados

**Fonte primária — ITBI.** ITBI é o *Imposto sobre a Transmissão de Bens Imóveis*, o imposto municipal pago em toda transferência onerosa de imóvel. A consequência prática: cada **guia de ITBI paga** é o registro de uma transação imobiliária real. As guias da Prefeitura de SP (Secretaria da Fazenda) trazem, por transação: valor, área do terreno (embutida), área construída, uso e a chave cadastral **SQL** (Setor-Quadra-Lote).

**Filtro de terreno puro.** Usa só lotes essencialmente sem construção: filtra para `uso = TERRENO` e **área construída ≈ 0** (< 1 m²). O porquê é metodológico — um imóvel com construção não é terra pura, o valor embute a benfeitoria; se não isolar, o preço/m² fica contaminado pelo prédio. Por isso o filtro acontece *antes* de derivar o alvo.

**O alvo.** Em cima dos terrenos puros: preço/m² observado = valor ÷ área. Mediana ~R$ 1.076/m². Depois da limpeza sobram alguns milhares de terrenos puros (~2,5 mil no treino de 2023+2024, mais o holdout de 2025).

**Anos e enriquecimento.** ITBI de 2023, 2024 e 2025. A esse núcleo a camada geográfica anexa contexto por localização: GeoSampa (centroide da quadra, zoneamento), OpenStreetMap (estações, comércio, escolas) e IBGE (renda do setor). Só dado público — reproduzível por qualquer um com as mesmas fontes.

**Validação temporal.** Treina em 2023+2024, testa em 2025 — o ano mais novo e fechado, sem embaralhar o tempo (walk-forward). O número reportado é honesto porque foi medido fora do período de treino.

**Duas decisões de dados que valem citar:**
- *Fonte da área:* usa a área embutida no próprio ITBI (cobertura 100%), não o `lote_cidadao` do GeoSampa (só ~0,6%). Onde os dois coincidem, batem em 100% dos casos — o que valida a escolha.
- *Recorte de anos:* anos anteriores a 2023 ficaram de fora — os arquivos antigos de ITBI embutem o cadastro *atual*, então um lote vendido vazio em 2019 e construído depois seria descartado por engano (viés de sobrevivência).

## 2. O modelo de preço/m²

**Alvo (o que prediz):** o preço por m² de terreno (R$/m²), derivado de transação real do ITBI (valor ÷ área, em terrenos puros). Não prediz viabilidade nem sucesso — só o preço/m².

**Features (10 no total)** — atributos do lote e do entorno:

| Feature | Fonte | O que captura |
|---|---|---|
| `area_terreno` | ITBI | tamanho do lote (m²) |
| `testada` | ITBI | frente do lote (m) — esquina/largura influencia |
| `dist_centro` | GeoSampa | distância à Praça da Sé (marco zero) — gradiente centro→periferia |
| `dist_estacao` | OSM | distância à estação de transporte mais próxima |
| `n_comercio_1km` | OSM | comércios no raio de 1 km — vitalidade local |
| `n_escola_1km` | OSM | escolas no raio de 1 km — infraestrutura do bairro |
| `renda_setor` | IBGE | renda do setor censitário — poder aquisitivo da região |
| `x`, `y` | GeoSampa | coordenadas (UTM) do centroide da quadra — posição absoluta |
| `zona` | GeoSampa | sigla do zoneamento (PDE 2014) — potencial construtivo (categórica) |

**Algoritmo e artefatos:** gradient boosting (`HistGradientBoostingRegressor`). Dois modelos, mesmo conjunto de features — o pontual (`preco_m2.joblib`, usado pelo SHAP) e o quantílico/CQR (`preco_m2_quantilico.joblib`, que dá a faixa [10%, mediana, 90%]). Pré-processamento: numéricas escalonadas + imputação; `zona` em one-hot.

**Não são features:** custos (gleba, infra) — são inputs da simulação; e a viabilidade — é calculada/simulada, nunca alvo de ML.

## 3. O que varia no Monte Carlo

**Três inputs são sorteados a cada cenário** (os incertos dominantes):

1. **Preço do lote** — vem da faixa do CQR (preço/m² [inf, med, sup] × área), modelado como triangular (min, moda, max). É a incerteza que o ML produz.
2. **Custo de infraestrutura** — triangular. No fluxo do n8n, o valor único informado vira triangular (moda −20%/+40%).
3. **Velocidade de venda (absorção)** — os meses de venda, triangular. No n8n, o valor único vira triangular (da metade ao dobro da moda).

**Ficam fixos (não sorteados):** custo da gleba (você negocia, então é conhecido), nº de lotes, meses de obra, taxa-alvo, parcelas, mês de início das vendas, e todos os custos operacionais (comissão, impostos, marketing, admin, licenciamento).

> **Nota de precisão:** a apresentação agrupa "custos (gleba, infra)" como incertos, mas no código só a infra é sorteada — a gleba entra como valor fixo. Se perguntarem "o que varia?", a resposta exata é: **preço, custo de infra e velocidade de venda**.

**Um refinamento opcional** (default desligado; sofisticação opt-in):
- **Cópula gaussiana (`rho_mercado`):** acopla preço e absorção por um "estado de mercado" comum — mercado quente = preço alto coincide com venda rápida (menos meses). No app o analista controla num slider (~0,5 típico); `rho=0` = independentes. Engrossa as caudas conjuntas → VPL mais largo e probabilidade mais conservadora.

**Como roda:** para cada um dos ~10.000 cenários, sorteia (preço, infra, absorção) → monta o fluxo de caixa → calcula VPL e TIR. Saída: as distribuições de VPL e TIR, e `prob_viavel` = fração dos cenários com VPL > 0.

## 4. O que o VPL representa

**Definição.** O VPL é a soma de *todas* as entradas e saídas do projeto, ao longo de toda a linha do tempo, cada uma trazida ao valor de hoje (descontada). Inclui as saídas (gleba no mês 0, infra na obra) **e** as entradas (vendas), tudo somado já descontado. Não é "o caixa no fim" — é o líquido de tudo, em reais de hoje.

**A sutileza que muda a interpretação:** o desconto é feito à **taxa-alvo** (a rentabilidade exigida). Por isso:
- VPL = 0 → o projeto rende *exatamente* a meta (não é prejuízo; é "bateu o alvo na mosca").
- VPL > 0 → **supera** a meta: cria valor acima do retorno exigido.
- VPL < 0 → fica abaixo da meta (pode até dar lucro em caixa, mas rende menos do que se queria).

Um VPL de R$ 4 mi **não** quer dizer "você terá R$ 4 mi no fim". Quer dizer: *"este projeto vale R$ 4 mi a mais, em reais de hoje, do que simplesmente render a taxa-alvo"*. É lucro em excesso da exigência, a valor presente.

**No histograma de VPL:** cada barra conta quantos dos ~10.000 cenários deram aquele VPL. A linha vermelha em R$ 0 é o "empata com a meta". A fração à direita dela = probabilidade de superar a meta = a probabilidade de viabilidade — o mesmo número que a fração à direita da "meta" no gráfico da TIR (porque VPL > 0 à taxa-alvo e TIR > meta são o mesmo evento).

---

*Fontes no repositório: dados e features em `src/loteia/data/` (incl. `features.py`); modelo de incerteza em `model/uncertainty.py` (CQR/MAPIE); Monte Carlo em `finance/montecarlo.py`; fluxo de caixa, VPL e TIR em `finance/cashflow.py`. Métricas walk-forward (2023+2024 → 2025) em `notebooks/03_validacao_temporal.ipynb`.*
