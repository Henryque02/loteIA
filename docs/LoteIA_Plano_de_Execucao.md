# LoteIA — Plano de Execução

**Projeto final de LIA — Engenharia de Computação** · Grupo: individual
**Estado:** implementação das 6 camadas concluída e testada. Fase atual:
endurecimento (reprodutibilidade), fortalecimento da validação temporal e
preparação da banca.
*(Nome "LoteIA" é provisório — Lote + IA. Troca fácil, é só um find-and-replace.)*

> **Snapshot de revisão (15 jun):** o sistema roda de ponta a ponta; **135 testes
> passam, 5 pulados** (os pulados exigem o artefato real treinado). Um bug de
> empacotamento que travava todo `make`/pytest foi corrigido (ver §11). O pipeline
> reproduz os números esperados: **2.500 terrenos limpos**, **mediana R$ 1.076/m²**.

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
   estagia. A Opes entra como **usuário-alvo e validador de domínio** — nenhum
   dado proprietário é usado; o projeto roda 100% em dado público.

2. **A profundidade joga na força do autor.** O coração do projeto é modelagem
   estocástica (propagação de incerteza, simulação de Monte Carlo, otimização) —
   modelar problema complexo de engenharia, não estatística pura.

3. **Honestidade metodológica embutida.** O projeto não finge prever o sucesso de
   um empreendimento (não existe dado público de resultado). Ele **simula**
   viabilidade a partir de inputs estimados — uma distinção que é, por si só, um
   ponto forte de banca (§2).

---

## 2. A decisão metodológica central — não prevemos sucesso, simulamos viabilidade

**Leia antes de tocar em qualquer modelo.** Esta seção é o coração da defensabilidade.

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

Quando a banca perguntar "como você validou que o empreendimento é viável?", a
resposta é "eu não afirmo que é — eu quantifico a probabilidade dele ser, dada a
incerteza do mercado, e mostro de que variável essa probabilidade mais depende".

---

## 3. As 6 camadas de profundidade — todas implementadas

O **núcleo** é a fatia vertical: modelo de preço + cálculo de viabilidade + API +
front. As 6 camadas o transformam de trivial em projeto de destaque.

- **[x] Camada 1 — Incerteza no preço.** Regressão quantílica conformalizada (CQR,
  via MAPIE): o modelo prevê uma *faixa* `[inf, med, sup]` com cobertura calibrada,
  não só um ponto. `ModeloQuantilico` em `model/uncertainty.py`; cobertura empírica
  medida no teste. O conjunto de calibração sai de um split aleatório **dentro do
  treino** — a regra temporal vale para treino×teste, não para a calibração.

- **[x] Camada 2 — Monte Carlo da viabilidade.** O coração de engenharia
  (`finance/montecarlo.py`). Amostra milhares de cenários da incerteza dos inputs
  (preço, absorção, custo de infra como triangulares) → roda o fluxo de caixa em
  cada um → distribuição de TIR/VPL → **probabilidade de VPL > 0 à taxa-alvo**.
  Dois refinamentos, ambos com default desligado (caso-base simples, sofisticação
  opt-in):
  - **Cópula gaussiana** acoplando preço↔absorção (`rho_mercado`): mercado quente =
    preço alto coincide com venda rápida.
  - **Decomposição sistemática/idiossincrática** do preço (`frac_idiossincratica`):
    o ruído por-lote diversifica ∝ 1/√n_lotes; o nível de mercado, não.

- **[x] Camada 3 — Explicabilidade.** SHAP no modelo de preço (`model/explain.py`,
  com teste de aditividade) + **análise de sensibilidade** (tornado one-at-a-time
  em `finance/sensitivity.py`): de qual variável a viabilidade mais depende.

- **[x] Camada 4 — Orquestração e produto (n8n).** Workflow de 11 nós
  (`n8n/workflow_viabilidade.json`): **Google Form → valida campos → valida
  localização (`/quadra`) → `POST /viabilidade` → grava na planilha (log) → email
  ao solicitante** com o veredito + **link para o relatório interativo no
  Streamlit**. Nenhuma alteração no Python; só consome a API. Setup e campos do
  Form em `n8n/README.md`.

- **[x] Camada 5 — Enriquecimento geoespacial (experimento).** Features de
  distância (centro, estação), contagem em 1 km (comércio, escola) via OSM, renda
  por setor censitário (IBGE) e zona (zoneamento GeoSampa). **Critério de sucesso
  definido ANTES** (`experimento_geo.py`): o modelo enriquecido deve reduzir o MAE
  no teste em ≥5% frente ao de 3 features; o script reporta SUCESSO/NEGATIVO.

- **[x] Camada 6 — Otimização (highest-and-best-use).** `optimize/config_search.py`
  + `POST /otimizar`: dado uma gleba, compara configurações (tamanho/número de
  lotes) e ordena pela probabilidade de viabilidade, ligando o modelo de preço ao
  simulador (o preço/m² responde ao tamanho do lote candidato).

---

## 4. O modelo financeiro do loteamento (primer)

Para quem vem da engenharia e não das finanças: o fluxo de caixa de um loteamento
é um problema de modelagem temporal, não de contabilidade complexa.

- **Receita (VGV).** Soma do preço de venda dos lotes, *distribuída no tempo*
  conforme a absorção e muitas vezes *parcelada* (carteira de recebíveis).
- **Custos.** Gleba (à vista no mês 0) + infraestrutura (distribuída nos meses de
  obra) + licenciamento + marketing/comissões + impostos + administração mensal.
- **Indicadores.** Traz-se tudo a valor presente: **VPL** (soma descontada) e
  **TIR** (rentabilidade efetiva, por bisseção). Viável = VPL > 0 à taxa-alvo
  (equivale a TIR > alvo). Implementado em `finance/cashflow.py`.

A dificuldade — e o mérito — está em modelar a *incerteza* de cada input e
propagá-la (Camada 2), não na aritmética em si.

---

## 5. Dados — DECISÃO TRAVADA E EXECUTADA: São Paulo (ITBI ⨝ IPTU)

O lado da **receita** vem de **transação real de São Paulo**, cruzando duas bases
abertas da Prefeitura pela chave **SQL** (Setor-Quadra-Lote):

- **ITBI** (Secretaria da Fazenda, mensal): cada linha é uma transação paga, com
  SQL, endereço, valor e — crucialmente — **área já embutida** (snapshot do
  cadastro IPTU anexado ao arquivo).
- **Cadastro IPTU** (GeoSampa): área, uso, zoneamento por SQL.

**Decisão de fonte da área (verificada empiricamente, no notebook de EDA):**
- Usa-se a **área embutida no próprio ITBI** → **cobertura 100%** dos registros.
- O `lote_cidadao` do GeoSampa cobriria só ~0,6% dos terrenos (lotes antigos
  ausentes do snapshot atual) — descartado. Onde os dois casam, a área bate em
  100% dos casos (|dif| ≤ 1 m²), validando o uso da área embutida.

**Pipeline executado:** empilha ITBI 2023/24 → isola terrenos (uso=TERRENO &
área construída < 1 m²) → preço/m² = valor ÷ área do terreno → filtros de limpeza
(compra e venda, 100% transmitido, fração ideal 1, valor > R$1.000). Resultado:
**63.588 terrenos isolados → 2.500 limpos para o alvo**, mediana **R$ 1.076/m²**,
estável entre anos (R$ 1.104 em 2023, R$ 1.059 em 2024).

**O lado do custo** não vem de treino — entra como input incerto na Monte Carlo
(C2) e é interrogado pela sensibilidade (C3).

**Extensão multi-ano (próximo passo, ver §9):** `data/tpcl.py` já implementa o
join com cadastro **ano-alinhado** (a cada transação, o cadastro do ano dela, não
o atual) para evitar viés de snapshot ao usar anos antigos — mas exige obter o
cadastro histórico manualmente (Base dos Dados / e-SIC).

---

## 6. Arquitetura de componentes

```
CAMADA DE DADOS
  ITBI ⨝ IPTU (SP) por SQL · isolamento de terrenos
    + enriquecimento geoespacial (OSM, IBGE, zoneamento)   [Camada 5]
            ▼
CAMADA DE ML (offline / treino)
  • Modelo de preço/m² pontual (.joblib)                    [núcleo]
  • Modelo quantílico CQR — faixa de preço (.joblib)        [Camada 1]
  • Explicador SHAP                                          [Camada 3]
            ▼
CAMADA DE SIMULAÇÃO
  Motor de fluxo de caixa do loteamento (VPL/TIR)
    + Monte Carlo (→ probabilidade) + cópula/idiossincrático [Camada 2]
    + otimizador de configuração                             [Camada 6]
    + tornado de sensibilidade                               [Camada 3]
            ▼
CAMADA DE SERVIÇO — FastAPI
  POST /viabilidade → { prob_viavel, faixa_preco_m2,
                        tir_anual, vpl, sensibilidade,
                        fatores_shap, distribuicoes }
  POST /otimizar · GET /quadra/{setor}/{quadra} · GET /health
            ▼
CAMADA DE ORQUESTRAÇÃO — n8n                                 [Camada 4]
  Google Form → valida → API → grava (log) → email + link
            ▼
CAMADA DE APRESENTAÇÃO — Streamlit (ou Lovable)
  Modo interativo (formulário) e modo relatório (URL do n8n):
  mapa · histogramas TIR/VPL · tornado · fatores SHAP
```

---

## 7. Workflow do n8n (Camada 4) — implementado

`Google Sheets Trigger` (respostas do Form) → `Code` (montar payload, com
defaults de mercado e parsing de número BR) → `IF` (campos completos?) → `HTTP`
(`GET /quadra`, valida localização) → `IF` (achou?) → `HTTP` (`POST /viabilidade`)
→ `Code` (formatar veredito + montar URL do relatório Streamlit) → `Google Sheets`
(gravar log) → `Email` (veredito ao solicitante). Dois ramos de email de exceção
(faltou dado / local não encontrado).

Notas de integração (em `n8n/README.md`): a API deve ouvir em `0.0.0.0`; n8n via
Docker acessa o host por `host.docker.internal` (exige `extra_hosts`); os nós de
email podem ser SMTP **ou Gmail** (OAuth, mais simples).

---

## 8. Estado atual e o que falta (a ~1 semana da entrega)

**Concluído:** as 6 camadas (C1–C6), a fatia vertical, o pipeline de dados real,
a API completa, o workflow n8n, o front Streamlit (interativo + relatório), a
suíte de testes (135 passando) e a correção de empacotamento (§11).

**O que falta, em ordem de prioridade:**

1. **Fortalecer a validação temporal (§9, risco nº 1).** Hoje treina só com 2023 e
   testa só com 2024 — um ano de cada lado. Mínimo: explicitar no relatório que é
   uma validação *forward de um ano* e mostrar a estabilidade da mediana entre anos.
   Ideal, se houver tempo: adicionar anos via o caminho `tpcl.py`.
2. **Treinar e versionar os artefatos finais** (`make data && make geo && make
   train`) e rodar a integração real da API (os 5 testes hoje pulados).
3. **Relatório / documentação:** problema → decisão metodológica (§2) → pipeline →
   resultados (cobertura do CQR, ganho do experimento geo, exemplo de viabilidade).
4. **Pitch e ensaio:** foco nas decisões não-óbvias de cada camada. Margem para
   imprevistos.

---

## 9. Riscos e mitigação

- **Validação temporal fina (risco nº 1 atual).** Só 2023/2024. "Treinar no
  passado, testar no futuro" é o argumento-chave de defensabilidade, hoje
  demonstrado sobre um único ano de cada lado. Mitigação: ser explícito sobre isso
  e/ou estender anos via `tpcl.py` (já implementado, falta dado histórico).
- **`rho_mercado=0.5` como default do produto é uma suposição.** Mitigação: ter a
  justificativa pronta e mostrar sensibilidade rodando com `rho=0`.
- **Renda IBGE é Censo 2010 aplicada a transações de 2023/24.** Defensável como
  enriquecimento estático, mas vale uma frase de ressalva no relatório.
- **Camada 6 é a mais difícil.** Já implementada; se algo der errado na banca, o
  sistema é completo e apresentável sem ela.
- **ML "só" um regressor de preço.** A profundidade está deliberadamente na
  incerteza (C1) + simulação (C2) + sensibilidade (C3), não no regressor cru.

---

## 10. Checklist de entregáveis

- [x] Fonte de dados definida e **executada** (SP: ITBI ⨝ IPTU) — EDA documentada
      nos notebooks (2.500 terrenos, mediana R$ 1.076/m²)
- [x] Modelo de preço/m² comparado contra baseline (mediana por setor)
- [x] Incerteza no preço (faixa via CQR/MAPIE) — C1
- [x] Motor financeiro do loteamento (VGV, fluxo, VPL/TIR) — `finance/cashflow.py`
- [x] Simulação de Monte Carlo → probabilidade de retorno — C2
- [x] SHAP (preço) + análise de sensibilidade (financeiro) — C3
- [x] API FastAPI servindo a viabilidade completa (+ `/otimizar`, `/quadra`)
- [x] Workflow n8n + entrada via Google Form — C4
- [x] Experimento de enriquecimento geoespacial com critério pré-definido — C5
- [x] Otimizador de configuração — C6
- [x] Front-end Streamlit (interativo + modo relatório)
- [x] Suíte de testes (135 passando) e correção de empacotamento
- [ ] Validação temporal fortalecida / explicitada no relatório
- [ ] Artefatos finais treinados e integração real da API verificada
- [ ] Relatório e pitch

---

## 11. Correção de empacotamento aplicada (nota técnica)

Faltava `[build-system]` no `pyproject.toml`. Sem ela, o `uv` tratava o projeto
como "não-pacote" e **não instalava `loteia`** no ambiente; com o layout `src/`,
isso quebrava `import loteia` e, por tabela, **todos os `make` e a coleta inteira
do pytest** num checkout limpo (só funcionava rodando de dentro de `src/`).
Adicionado:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/loteia"]
```

Depois `uv sync` (instala `loteia==0.1.0` editável) → **135 testes passam, 5
pulados**. (A `description` do pyproject também saiu do placeholder padrão.)
