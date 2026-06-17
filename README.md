# LoteIA — viabilidade probabilística de loteamentos

Simulador que responde *"vale a pena desenvolver este terreno?"* com uma
**probabilidade**, não com um número seco. Um modelo de machine learning estima o
**preço de venda do terreno por m²** (com incerteza); um motor de fluxo de caixa e
uma simulação de **Monte Carlo** transformam essa estimativa na **probabilidade de
o empreendimento superar a rentabilidade-alvo**.

Projeto final da disciplina LIA. Usa **apenas dados públicos** (ITBI/IPTU de São
Paulo, GeoSampa, IBGE, OpenStreetMap).

---

## Princípio central

> O ML modela **apenas o preço/m² do terreno** (o lado da receita). Viabilidade e
> "sucesso do empreendimento" **nunca** são alvo de ML — são *calculados* (fluxo de
> caixa) e *simulados* (Monte Carlo).

Não existe dado público de resultado de empreendimento, então não há um alvo a
"prever" para viabilidade. O que se prevê é o preço da terra; o resto é finança
explícita e propagação de incerteza. A saída é sempre uma distribuição, não um
ponto.

---

## Arquitetura em camadas

| Camada | O que faz | Onde mora |
|---|---|---|
| **C1 — Incerteza** | Regressão quantílica conformal (CQR/MAPIE): preço/m² como **intervalo**, não ponto | `model/uncertainty.py` |
| **C2 — Monte Carlo** | Propaga a incerteza dos inputs (preço, infra, absorção) pelo fluxo de caixa → distribuição de VPL/TIR → P(viável) | `finance/montecarlo.py`, `finance/cashflow.py` |
| **C3 — Explicabilidade** | SHAP (por que este preço?) + tornado de sensibilidade (o que mais move o VPL?) | `model/explain.py`, `finance/sensitivity.py` |
| **C4 — Orquestração** | Google Form → valida → API → e-mail + link do relatório (n8n) | `n8n/` |
| **C5 — Geoespacial** | Enriquece o terreno com distância ao centro/estação, POIs, renda, zoneamento | `data/geo.py`, `data/features.py`, `model/experimento_geo.py` |
| **C6 — Otimizador** | *Highest-and-best-use*: qual tamanho de lote maximiza a P(viável) | `optimize/config_search.py` |

---

## Stack

- **Python 3.12+** · pandas · scikit-learn · SHAP
- **Incerteza:** MAPIE (CQR — regressão quantílica conformalizada)
- **Geo:** geopandas · pyproj · scipy (GeoSampa WFS, Overpass/OSM, IBGE)
- **API:** FastAPI + uvicorn · artefatos serializados em `joblib`
- **Front:** Streamlit · **Automação:** n8n
- **Dependências:** geridas por `uv` (`pyproject.toml` + `uv.lock`); o pacote
  `loteia` é instalado pelo build-system (hatchling, layout `src/`)

---

## Instalação e uso

```bash
uv sync          # instala as dependências E o pacote loteia (editável)

make data        # baixa e processa ITBI de SP (2023/24/25) → parquet em data/interim
make geo         # quadras (GeoSampa) + POIs (OSM) + renda (IBGE) + zoneamento → caches
make train       # treina o modelo pontual + o quantílico → models/*.joblib
make api         # sobe a FastAPI em localhost:8000
make front       # sobe o Streamlit em localhost:8501 (outro terminal, com a API no ar)

make test        # pytest
make lint        # ruff check src tests
```

> `uv sync` exige a seção `[build-system]` no `pyproject.toml` — sem ela
> `import loteia` falha e todos os comandos `make` quebram.

---

## Estrutura do repositório

```
src/loteia/
├── config.py            # caminhos e parâmetros globais (ANO_CORTE_TREINO, SEED)
├── data/                # download e preparo dos dados
│   ├── download.py      #   ITBI (xlsx por ano) → parquet
│   ├── join.py          #   limpeza, filtro de terrenos, preço/m²
│   ├── geo.py           #   downloads geoespaciais (C5), com cache
│   └── features.py      #   engenharia de atributos geo (distâncias, POIs, renda, zona)
├── model/
│   ├── train.py         #   treino do regressor + split temporal (walk-forward)
│   ├── uncertainty.py   #   C1 — modelo quantílico (CQR)
│   ├── explain.py       #   C3 — SHAP global e local
│   └── experimento_geo.py  # C5 — experimento controlado: geo melhora o modelo?
├── finance/
│   ├── cashflow.py      #   fluxo de caixa → VPL, TIR
│   ├── montecarlo.py    #   C2 — simulação de Monte Carlo
│   └── sensitivity.py   #   C3 — tornado de sensibilidade
├── optimize/
│   └── config_search.py # C6 — otimizador de configuração (opcional)
├── api/main.py          # FastAPI
└── report.py            # modo relatório do front (payload a partir de query params)

app/streamlit_app.py     # front interativo + modo relatório (link do n8n)
n8n/                     # workflow da Camada 4 + setup
docs/                    # apresentação e referência técnica (Q&A da banca)
tests/                   # pytest (um arquivo por módulo)
```

`data/` e `models/` **não** são versionados (são reconstruídos pelos comandos `make`).

---

## Modelagem

**Alvo:** preço/m² de terreno puro. Imóveis com construção são **isolados** (área
construída ≈ 0) antes de calcular o alvo, para não contaminar o preço da terra.

**Validação por tempo (walk-forward).** Treina com transações até o ano de corte
(`ANO_CORTE_TREINO`, hoje 2024 = 2023+2024) e testa no ano mais novo e fechado
(2025). Nunca se embaralha aleatoriamente no tempo — isso seria vazamento temporal
e destruiria a defensabilidade. Quando surge um ano novo, o corte avança.

**Incerteza (C1).** O preço sai como intervalo (CQR/MAPIE, cobertura nominal 80%).
A cobertura empírica no teste é reportada por `make train` — sob deslocamento
temporal ela tende a ficar abaixo do nominal, e isso é registrado com honestidade.

**Custos não são features.** Gleba, infraestrutura e absorção são **inputs
incertos da simulação**, não features de treino do modelo de preço. Os dois lados
(receita estimada por ML × custos assumidos) nunca se misturam.

---

## API

Carrega os artefatos uma vez na inicialização.

| Rota | Descrição |
|---|---|
| `POST /viabilidade` | P(viável), faixa de preço/m², distribuição de TIR/VPL, SHAP, sensibilidade |
| `POST /otimizar` | ranking de tamanhos de lote por P(viável) (C6) |
| `GET /quadra/{setor}/{quadra}` | centroide da quadra fiscal (lat/lon) |
| `GET /health` | status |

---

## Front (Streamlit)

Dois modos no mesmo app:

- **Interativo** (`make front`): abas **Viabilidade** e **Otimizador de
  configuração**, com bloco "Terreno" e "Premissas comuns" compartilhados.
- **Relatório** (link do n8n): se a URL traz os parâmetros da análise, a página
  renderiza o resultado pronto, read-only — o cliente não preenche nada
  (`loteia/report.py`).

---

## Orquestração (n8n — Camada 4)

O cliente preenche um **Google Form**; o fluxo valida, chama a API e devolve por
**e-mail** o veredito + um **link para o relatório no Streamlit**. Workflow em
`n8n/workflow_viabilidade.json`; setup, campos do formulário e contrato da API em
[n8n/README.md](n8n/README.md). Nenhuma alteração no código Python — só consome a
API existente.

---

