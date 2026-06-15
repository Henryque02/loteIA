# Camada 4 — Orquestração (n8n)

Workflow `workflow_viabilidade.json`: o cliente preenche um **Google Form**, o
fluxo valida os dados, chama a **API de viabilidade** e devolve o veredito por
**email**. Camada 4 do LoteIA — nenhuma alteração no código Python; só consome a
API existente.

## Fluxo

```
Google Form → (Sheets) → Montar payload → Validar campos ─┬─ faltou dado → Email "faltam dados"
                                                          └─ ok → Validar localização (/quadra)
                                                                   ├─ 404 → Email "setor/quadra não localizados"
                                                                   └─ ok → POST /viabilidade → Formatar veredito
                                                                            → Gravar na planilha → Email ao solicitante
```

11 nós: `Google Sheets Trigger` · 2× `Code` (montar payload / formatar veredito) ·
2× `IF` (validação de campos e de localização) · 2× `HTTP Request`
(`GET /quadra`, `POST /viabilidade`) · `Google Sheets` (gravar) · 3× `Email` (SMTP).

## Campos do Google Form

Crie um Google Form e vincule as respostas a uma planilha. **Os títulos das
perguntas devem casar exatamente com os nomes abaixo** (são lidos pelo nó
"Montar payload"); se preferir outros títulos, ajuste as chaves no código do nó.

**Obrigatórios** (se faltar, o cliente recebe email pedindo o que falta):

| Pergunta (cabeçalho da coluna) | Exemplo |
|---|---|
| Setor | `085` |
| Quadra | `013` |
| Área do lote (m²) | `300` |
| Nº de lotes | `80` |
| Custo da gleba (R$) | `25000000` |
| Custo de infra (R$) | `20000000` |
| Meses de obra | `18` |
| Meses de venda | `24` |
| Taxa-alvo anual (%) | `15` (aceita `15` ou `0,15`) |
| Email | `cliente@exemplo.com` |

**Opcionais** (o fluxo aplica defaults de mercado, transparentes, se vazios):

| Pergunta | Default aplicado |
|---|---|
| Parcelas | `12` |
| Comissão (%) | `6%` |
| Impostos (%) | `4%` |
| Marketing (%) | `3%` |
| Admin mensal (R$) | `0` |
| Licenciamento (R$) | `0` |
| Início das vendas (mês) | `6` |
| Testada (m) | `≈ √área` |

O fluxo também transforma os valores únicos em distribuições para a Monte Carlo:
**custo de infra** vira triangular (moda ±, −20%/+40%) e **meses de venda** vira
triangular (moda, metade/dobro). Números no formato brasileiro
(`25.000.000`, `0,15`) são aceitos.

## Setup

1. **Importar**: n8n → *Workflows* → *Import from File* → `workflow_viabilidade.json`.
2. **Credenciais** (cada nó marcado "(configurar)"):
   - *Google Sheets OAuth2* — no trigger e no nó "Gravar na planilha".
   - *SMTP* (ou troque os nós Email por *Gmail*) — nos 3 nós de email; ajuste o
     `fromEmail`.
3. **Planilhas**: aponte o trigger para a planilha de **respostas do Form** e o
   nó "Gravar na planilha" para a planilha de **Resultados** (crie uma aba
   `Resultados`). Atualize o link no nó "Enviar email ao solicitante".
4. **API**: defina a variável de ambiente do n8n `API_BASE_URL`
   (default `http://localhost:8000`). Suba a API com `make api`.
   - n8n e API na mesma máquina → `http://localhost:8000`.
   - n8n em nuvem/contêiner → exponha a API por um túnel (ex.: `ngrok http 8000`)
     e use a URL pública em `API_BASE_URL`.
5. **Ativar** o workflow.

## Verificação rápida (sem n8n)

O contrato que o fluxo usa pode ser conferido direto na API:

```bash
make api  # sobe em localhost:8000
curl -s -X POST localhost:8000/viabilidade -H 'Content-Type: application/json' \
  -d '{"terreno":{"setor":"085","quadra":"013","area_lote_m2":300,"testada":17},
       "projeto":{"n_lotes":80,"custo_gleba":25000000,
         "custo_infra":{"minimo":16000000,"moda":20000000,"maximo":28000000},
         "meses_obra":18,"meses_vendas":{"minimo":12,"moda":24,"maximo":48},
         "n_parcelas":12,"taxa_alvo_anual":0.15,"mes_inicio_vendas":6,
         "custos":{"comissao_pct":0.06,"impostos_pct":0.04,"marketing_pct":0.03,
                   "admin_mensal":0,"licenciamento":0}},
       "n_sims":5000,"incluir_distribuicao":false}'
```

O teste ponta-a-ponta (Form → email) é manual, na sua instância n8n com as
credenciais configuradas.
