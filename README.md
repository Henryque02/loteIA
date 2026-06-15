# LoteIA
Simulador probabilístico de viabilidade de loteamentos.
Veja CLAUDE.md para contexto, comandos e regras do projeto.

## Rodando

```bash
uv sync          # dependências
make data        # ITBI 2023/24 → parquet
make geo         # quadras (GeoSampa) + OSM + renda IBGE
make train       # modelo pontual + quantílico (artefatos em models/)
make api         # FastAPI em localhost:8000
make front       # Streamlit em localhost:8501 (em outro terminal, com a API no ar)
```

## Orquestração (n8n, Camada 4)

Workflow Google Form → análise → email em `n8n/workflow_viabilidade.json`.
Setup e campos do formulário em [n8n/README.md](n8n/README.md).
