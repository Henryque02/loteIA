# LoteIA — Contexto do Projeto (CLAUDE.md)

Simulador probabilístico de viabilidade de loteamentos. O ML estima o preço de
venda de terreno por m²; um motor financeiro + simulação de Monte Carlo
transformam essa estimativa (com incerteza) numa **probabilidade de o
empreendimento superar a rentabilidade-alvo**. Projeto final da disciplina LIA.

## Stack
- Python 3.12+ · pandas, scikit-learn, SHAP · FastAPI · joblib
- Incerteza: regressão quantílica / conformal (MAPIE — CQR)
- Geo (C5): geopandas, pyproj, scipy (OSM/Overpass, GeoSampa WFS, IBGE)
- Front: Streamlit (ou Lovable) · Automação: n8n
- Deps geridas por `uv` (`pyproject.toml` + `uv.lock`); o pacote `loteia` é
  instalado pelo build-system (hatchling, layout `src/`).

## Comandos
- `uv sync` — instala as deps **e** o pacote `loteia` (editável). Exige a seção
  `[build-system]` no pyproject; sem ela `import loteia` falha e todo `make` quebra.
- `make data` — baixa e processa ITBI de SP (2023/24/25) → parquet em `data/interim`
- `make geo` — baixa quadras (GeoSampa WFS), POIs (OSM), renda (IBGE) e
  zoneamento → caches em `data/interim` (Camada 5)
- `make train` — treina o modelo pontual + o quantílico (gera `models/*.joblib`)
- `make api` — sobe a FastAPI (uvicorn) em `localhost:8000`
- `make front` — sobe o Streamlit em `localhost:8501` (noutro terminal, com a API no ar)
- `make test` — roda os testes (pytest)
- `make lint` — `ruff check src tests`

## Mapa do repositório
- `src/loteia/data/` — download, join por **SQL** (= Setor-Quadra-Lote, a chave do
  cadastro; é uma coluna, NÃO banco de dados — junta-se com pandas `merge`),
  features geoespaciais (`features.py`, `geo.py`)
- `src/loteia/model/` — treino, incerteza (C1), explicação SHAP (C3),
  experimento geo (C5, `experimento_geo.py`)
- `src/loteia/finance/` — fluxo de caixa, Monte Carlo (C2), sensibilidade (C3)
- `src/loteia/optimize/` — highest-and-best-use (C6, opcional)
- `src/loteia/api/` — FastAPI: `POST /viabilidade`, `POST /otimizar` (C6),
  `GET /quadra/{setor}/{quadra}`, `GET /health`
- `src/loteia/report.py` — modo relatório do front (reconstrói o payload a partir
  de query params; é o que o n8n usa para abrir o Streamlit já preenchido)
- `n8n/` — workflow da Camada 4 (Google Form → valida → API → email + link Streamlit)
- `notebooks/` — só exploração; a lógica mora em `src/loteia`
- `data/` e `models/` — NÃO versionar (estão no .gitignore)

## REGRAS CRÍTICAS (não violar)
1. **O ML modela APENAS o preço/m² de terreno (lado da receita).** Viabilidade e
   "sucesso do empreendimento" NUNCA são alvo de ML — são *calculados* (fluxo de
   caixa) e *simulados* (Monte Carlo). Não existe dado público de resultado de
   empreendimento; não invente um alvo.
2. **Validação separada por TEMPO (walk-forward).** Treinar com transações até o
   ano de corte (`ANO_CORTE_TREINO`, hoje 2024 = 2023+2024) e testar no ano mais
   novo e fechado (hoje 2025). Ao surgir um ano novo, o corte avança e o anterior
   vira treino. Nunca embaralhar aleatoriamente no tempo — isso é vazamento
   temporal e destrói a defensabilidade.
3. **Isolar terrenos** (área construída ≈ 0) ANTES de calcular preço/m². Imóvel com
   construção não é terra pura e contamina o alvo.
4. **Só dado público.** ITBI/IPTU de SP, GeoSampa, IBGE, OpenStreetMap. NUNCA usar
   dado proprietário da Opes Data.
5. **Custos (gleba, infraestrutura) são inputs incertos da simulação**, não
   features de treino do modelo de preço. Não misturar os dois lados.
6. **Incerteza sempre.** O modelo de preço reporta intervalo/distribuição, não só
   ponto. A saída de viabilidade é probabilística, não um número seco.

## Convenções
- Type hints; formatação com ruff/black. Seeds fixas para reprodutibilidade.
- Artefatos serializados em `.joblib` dentro de `models/`.
- A API carrega os artefatos uma vez na inicialização.
- Classes pickladas (modelos) são importadas pela referência canônica do módulo
  ao rodar como script, para que o artefato carregue na API (não como `__main__.*`).
