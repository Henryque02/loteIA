# LoteIA — Contexto do Projeto (CLAUDE.md)

Simulador probabilístico de viabilidade de loteamentos. O ML estima o preço de
venda de terreno por m²; um motor financeiro + simulação de Monte Carlo
transformam essa estimativa (com incerteza) numa **probabilidade de o
empreendimento superar a rentabilidade-alvo**. Projeto final da disciplina LIA.

## Stack
- Python 3.11+ · pandas, scikit-learn, SHAP · FastAPI · joblib
- Incerteza: regressão quantílica / conformal (MAPIE)
- Front: Streamlit (ou Lovable) · Automação: n8n

## Comandos
- `uv sync` (ou `uv pip install -r requirements.txt`) — instalar deps
- `make data` — baixar e processar ITBI + IPTU de SP
- `make train` — treinar o modelo de preço (gera models/*.joblib)
- `make api` — subir a FastAPI (uvicorn) em localhost
- `make test` — rodar os testes (pytest)

## Mapa do repositório
- `src/loteia/data/` — download, join por **SQL** (= Setor-Quadra-Lote, a chave do
  cadastro; é uma coluna, NÃO banco de dados — junta-se com pandas `merge`) e features
- `src/loteia/model/` — treino, incerteza (C1), explicação SHAP (C3)
- `src/loteia/finance/` — fluxo de caixa, Monte Carlo (C2), sensibilidade (C3)
- `src/loteia/optimize/` — highest-and-best-use (C6, opcional)
- `src/loteia/api/` — FastAPI `/viabilidade`
- `notebooks/` — só exploração; a lógica mora em `src/loteia`
- `data/` e `models/` — NÃO versionar (estão no .gitignore)

## REGRAS CRÍTICAS (não violar)
1. **O ML modela APENAS o preço/m² de terreno (lado da receita).** Viabilidade e
   "sucesso do empreendimento" NUNCA são alvo de ML — são *calculados* (fluxo de
   caixa) e *simulados* (Monte Carlo). Não existe dado público de resultado de
   empreendimento; não invente um alvo.
2. **Validação separada por TEMPO.** Treinar com transações ≤2023 e testar com
   ≥2024. Nunca embaralhar aleatoriamente no tempo — isso é vazamento temporal e
   destrói a defensabilidade.
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
