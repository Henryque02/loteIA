"""Front Streamlit do LoteIA: consome a FastAPI (mesmo caminho do n8n).
Aba 1: viabilidade de um projeto definido. Aba 2: otimizador de configuração.
Modo relatório: se a URL traz os parâmetros (link do n8n), renderiza a análise
pronta e pula o formulário. Suba a API antes: `make api` (front com `make front`)."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.ticker import FuncFormatter

from loteia.report import (
    payload_de_params,
    tem_params_relatorio,
)

_VERDE, _VERMELHO, _AZUL = "#2ca02c", "#d62728", "#3b7dd8"


def _fmt_reais(v: float, _=None) -> str:
    """Formata valor em R$ com sufixo legível (mi/mil) — mata o '1e7' do eixo."""
    a = abs(v)
    if a >= 1_000_000:
        return f"R$ {v / 1_000_000:,.1f} mi".replace(".0 mi", " mi")
    if a >= 1_000:
        return f"R$ {v / 1_000:,.0f} mil"
    return f"R$ {v:,.0f}"


def _fmt_pct(v: float, _=None) -> str:
    return f"{v * 100:.0f}%"

API = os.environ.get("LOTEIA_API", "http://localhost:8000")

st.set_page_config(page_title="LoteIA", page_icon="📐", layout="wide")
st.title("📐 LoteIA — viabilidade probabilística de loteamentos")


def _post(rota: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API}{rota}", json=payload, timeout=120)
    except requests.ConnectionError:
        st.error(f"API fora do ar em {API} — rode `make api` antes.")
        return None
    if r.status_code != 200:
        detalhe = r.json().get("detail", r.text) if r.text else r.status_code
        st.error(f"Erro da API ({r.status_code}): {detalhe}")
        return None
    return r.json()


def _viabilidade_cacheada(params) -> dict | None:
    """Cache leve do /viabilidade no modo relatório: simula uma única vez por
    link e reusa nos reruns (toggle do mapa, botões) em vez de re-simular a cada
    interação. Falhas não são cacheadas — tenta de novo no próximo run."""
    chave = "viab_" + str(sorted(dict(params).items()))
    corpo = st.session_state.get(chave)
    if corpo is None:
        corpo = _post("/viabilidade", payload_de_params(params))
        if corpo is not None:
            st.session_state[chave] = corpo
    return corpo


def _quadra_latlon(setor: str, quadra: str) -> dict | None:
    try:
        r = requests.get(f"{API}/quadra/{setor}/{quadra}", timeout=30)
        return r.json() if r.status_code == 200 else None
    except requests.ConnectionError:
        return None


def _mapa_toggle(setor: str, quadra: str, key: str, *, mostrar_status: bool = True):
    """Status da quadra + toggle 'Ver no mapa', recolhido por padrão (sem gap).
    Ligado, renderiza o mapa; veio em branco (tiles externos do st.map)? desliga
    e liga de novo para forçar nova tentativa de carregar."""
    loc = _quadra_latlon(setor, quadra)
    if mostrar_status:
        if loc:
            st.success("Quadra localizada ✔")
        else:
            st.warning("Quadra não localizada (confira setor/quadra ou suba a API).")
    if st.toggle("🗺️ Ver no mapa", key=key):
        if loc:
            st.map(pd.DataFrame({"lat": [loc["lat"]], "lon": [loc["lon"]]}), zoom=14)
        else:
            st.caption("Localização indisponível para mostrar no mapa.")


def _estilo(ax):
    """Estilo limpo: sem spines de topo/direita, fonte do título maior."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.title.set_size(13)
    ax.title.set_weight("bold")


def _barras_horizontais(rotulos, valores, titulo, rotulo_fn):
    """Barras horizontais com rótulo de valor em cada barra. `rotulo_fn(v)`
    formata o texto (R$ mi, %, etc.); cor por sinal (verde positivo, vermelho)."""
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(rotulos) + 1.2))
    cores = [_VERMELHO if v < 0 else _VERDE for v in valores]
    barras = ax.barh(rotulos, valores, color=cores)
    ax.bar_label(barras, labels=[rotulo_fn(v) for v in valores],
                 padding=4, fontsize=10)
    ax.set_title(titulo)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xticks([])  # o número está no rótulo da barra
    margem = max(abs(min(valores)), abs(max(valores))) * 0.28 or 1
    ax.set_xlim(min(0, min(valores)) - margem, max(0, max(valores)) + margem)
    _estilo(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _histograma(valores, titulo, alvo=None, formato="moeda", rotulo_alvo="meta"):
    """Histograma com eixo X formatado (moeda em R$ mi ou percentual)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(valores, bins=40, color=_AZUL, alpha=0.9)
    if alvo is not None:
        ax.axvline(alvo, color=_VERMELHO, linestyle="--", linewidth=1.6,
                   label=rotulo_alvo)
        ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(
        FuncFormatter(_fmt_reais if formato == "moeda" else _fmt_pct)
    )
    ax.set_yticks([])
    ax.set_title(titulo)
    _estilo(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _faixa(rotulo: str, minimo: float, moda: float, maximo: float, passo: float,
           disabled: bool = False):
    c1, c2, c3 = st.columns(3)
    return (
        c1.number_input(f"{rotulo} (mín)", value=minimo, step=passo, disabled=disabled),
        c2.number_input(f"{rotulo} (moda)", value=moda, step=passo, disabled=disabled),
        c3.number_input(f"{rotulo} (máx)", value=maximo, step=passo, disabled=disabled),
    )


def _render_relatorio(corpo: dict):
    """Renderiza o relatório de viabilidade (métricas + gráficos) a partir da
    resposta de /viabilidade. Usado pela aba interativa e pelo modo relatório."""
    prob = corpo["prob_viavel"]
    faixa = corpo["faixa_preco_m2"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidade de superar a taxa-alvo", f"{prob:.0%}")
    m2.metric("Preço/m² (10%)", f"R$ {faixa['inf']:,.0f}")
    m3.metric("Preço/m² (mediana)", f"R$ {faixa['med']:,.0f}")
    m4.metric("Preço/m² (90%)", f"R$ {faixa['sup']:,.0f}")

    _NOMES_INPUT = {
        "preco_lote": "Preço de venda do lote",
        "custo_infra": "Custo de infraestrutura",
        "meses_vendas": "Velocidade de venda",
    }
    # rótulos legíveis para as features cruas do modelo (gráfico SHAP);
    # fallback para o nome cru caso surja uma feature nova não mapeada
    _NOMES_FEATURE = {
        "dist_centro": "Distância ao centro",
        "dist_estacao": "Distância até a estação mais próxima",
        "area_terreno_itbi": "Área do terreno (m²)",
        "x": "Coordenada leste (X)",
        "y": "Coordenada norte (Y)",
        "n_escola_1km": "Escolas em 1 km",
        "zona": "Zoneamento",
        "n_comercio_1km": "Comércios em 1 km",
        "testada": "Frente do lote (m)",
        "renda_setor": "Renda média da região (IBGE)",
    }

    dist = corpo.get("distribuicoes")
    g1, g2 = st.columns(2)
    with g1:
        if dist and dist["tir_anual"]:
            _histograma(dist["tir_anual"], "Rentabilidade anual do projeto (TIR)",
                        alvo=corpo["taxa_alvo_anual"], formato="pct",
                        rotulo_alvo="meta")
            st.caption("**TIR** = taxa interna de retorno: a rentabilidade efetiva "
                       "do projeto ao ano. Quanto mais à direita da meta, melhor.")
    with g2:
        if dist:
            _histograma(dist["vpl"], "Lucro a valor presente (VPL)",
                        alvo=0.0, formato="moeda", rotulo_alvo="empata (R$ 0)")
            st.caption("**VPL** = valor presente líquido: o lucro do projeto trazido "
                       "para hoje. Acima de R$ 0 = projeto cria valor.")

    g3, g4 = st.columns(2)
    with g3:
        sens = pd.DataFrame(corpo["sensibilidade"])
        rotulos = [_NOMES_INPUT.get(v, v) for v in sens["variavel"]][::-1]
        _barras_horizontais(rotulos, sens["amplitude"].tolist()[::-1],
                            "O que mais muda o resultado", _fmt_reais)
        st.caption("Quanto o lucro (VPL) varia quando cada fator vai do pior ao "
                   "melhor cenário. A barra maior é o fator decisivo.")
    with g4:
        if corpo["fatores_shap"]:
            contrib = corpo["fatores_shap"]["contribuicoes"]
            total = sum(abs(v) for v in contrib.values()) or 1
            ordem = sorted(contrib, key=lambda k: abs(contrib[k]))
            rotulos = [_NOMES_FEATURE.get(k, k) for k in ordem]
            _barras_horizontais(rotulos, [contrib[k] for k in ordem],
                                "O que explica o preço do terreno",
                                lambda v: f"{v / total * 100:+.0f}%")
            st.caption("Peso de cada fator na estimativa do preço/m² (verde puxa "
                       "para cima, vermelho para baixo).")


def _render_otimizacao(corpo: dict):
    """Renderiza o ranking do otimizador (métrica + tabela + barras). Usado pela
    aba Otimizador e pelo comparativo opcional do modo relatório."""
    ranking = pd.DataFrame(corpo["configuracoes"])
    melhor = ranking.iloc[0]
    st.metric(
        "Melhor configuração",
        f"lotes de {melhor['area_lote_m2']:.0f} m² "
        f"({melhor['n_lotes']:.0f} lotes)",
        f"P(viável) = {melhor['prob_viavel']:.0%}",
    )
    st.dataframe(
        ranking.style.format(
            {
                "area_lote_m2": "{:.0f}",
                "n_lotes": "{:.0f}",
                "prob_viavel": "{:.1%}",
                "vpl_mediano": "R$ {:,.0f}",
                "tir_anual_mediana": "{:.1%}",
            },
            na_rep="—",
        ),
        use_container_width=True,
    )
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(
        ranking["area_lote_m2"].astype(int).astype(str) + " m²",
        ranking["prob_viavel"],
        color="#1f77b4",
    )
    ax.set_ylabel("P(viável)")
    ax.set_ylim(0, 1)
    ax.set_title("Probabilidade de viabilidade por tamanho de lote")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------- modo relatório (link do n8n) ----------------
_params = st.query_params
if tem_params_relatorio(_params):
    emp = _params.get("empreendimento", "")
    setor_r, quadra_r = _params.get("setor", ""), _params.get("quadra", "")
    st.subheader(f"Relatório de viabilidade — {emp}" if emp
                 else "Relatório de viabilidade")

    # painel read-only: mesmo layout do make front, porém com os campos
    # desabilitados e preenchidos com os dados do cliente vindos da URL (Form → n8n)
    def _f(chave: str, default: float = 0.0) -> float:
        v = _params.get(chave)
        return float(v) if v not in (None, "") else float(default)

    with st.expander("📋 Dados do empreendimento", expanded=True):
        t1, t2, t3 = st.columns(3)
        t1.text_input("Setor fiscal", value=setor_r, disabled=True)
        t2.text_input("Quadra fiscal", value=quadra_r, disabled=True)
        t3.number_input("Frente do lote (m)", value=_f("testada", 10),
                        disabled=True)

        j1, j2 = st.columns(2)
        j1.number_input("Área do lote (m²)", value=_f("area_lote_m2"), disabled=True)
        j2.number_input("Nº de lotes", value=int(_f("n_lotes")), disabled=True)

        p1, p2 = st.columns(2)
        p1.number_input("Custo da gleba (R$)", value=_f("custo_gleba"), disabled=True)
        p2.number_input("Meses de obra", value=int(_f("meses_obra")), disabled=True)

        _faixa("Custo de infra (R$)", _f("infra_min"), _f("infra_moda"),
               _f("infra_max"), 1e6, disabled=True)
        _faixa("Meses de venda", _f("vendas_min"), _f("vendas_moda"),
               _f("vendas_max"), 6.0, disabled=True)

        st.markdown("**Custos operacionais**")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.number_input("Comissão %VGV", value=_f("comissao_pct"), disabled=True,
                        format="%.2f")
        d2.number_input("Impostos %VGV", value=_f("impostos_pct"), disabled=True,
                        format="%.2f")
        d3.number_input("Marketing %VGV", value=_f("marketing_pct"), disabled=True,
                        format="%.2f")
        d4.number_input("Admin mensal (R$)", value=_f("admin_mensal"), disabled=True)
        d5.number_input("Licenciamento (R$)", value=_f("licenciamento"), disabled=True)

        st.markdown("**Meta e cronograma de vendas**")
        s1, s2, s3 = st.columns(3)
        s1.number_input("Parcelas por venda", value=int(_f("n_parcelas", 1)),
                        disabled=True)
        s2.number_input("Mês início vendas", value=int(_f("mes_inicio_vendas", 1)),
                        disabled=True)
        s3.number_input("Taxa-alvo anual", value=_f("taxa_alvo_anual"), disabled=True,
                        format="%.2f")

    _mapa_toggle(setor_r, quadra_r, key="mapa_relatorio", mostrar_status=False)

    st.divider()
    st.markdown("##### Resultado da análise")
    corpo = _viabilidade_cacheada(_params)
    if corpo:
        _render_relatorio(corpo)
    st.caption("Resultado probabilístico (simulação de Monte Carlo) — não é "
               "garantia de resultado do empreendimento.")

    # comparativo opcional (highest-and-best-use) rodado SÓ ao clicar — não pesa
    # o carregamento inicial. Inputs do otimizador derivados do próprio projeto.
    st.divider()
    area_lote_cli = _f("area_lote_m2")
    n_lotes_cli = int(_f("n_lotes")) or 1
    v_min, v_moda, v_max = _f("vendas_min"), _f("vendas_moda"), _f("vendas_max")
    if st.button("📊 Comparar outros tamanhos de lote"):
        if not (area_lote_cli and v_min and v_moda and v_max):
            st.warning("Faltam dados (área do lote ou prazo de vendas) no link "
                       "para montar o comparativo.")
        else:
            # área vendável = área do lote × nº de lotes; candidatos = leque em
            # torno do tamanho do cliente; absorção (lotes/mês) = nº de lotes ÷
            # meses de venda (duração curta → absorção alta, daí o min/max invertem)
            candidatos = sorted({round(area_lote_cli * m)
                                 for m in (0.5, 0.75, 1.0, 1.5, 2.0)})
            corpo_otim = _post(
                "/otimizar",
                {
                    "terreno": {"setor": setor_r, "quadra": quadra_r,
                                "testada": _f("testada", 10)},
                    "gleba": {
                        "area_vendavel_m2": area_lote_cli * n_lotes_cli,
                        "candidatos_area_lote": [float(c) for c in candidatos],
                        "custo_gleba": _f("custo_gleba"),
                        "custo_infra": {"minimo": _f("infra_min"),
                                        "moda": _f("infra_moda"),
                                        "maximo": _f("infra_max")},
                        "meses_obra": int(_f("meses_obra")),
                        "absorcao_lotes_mes": {"minimo": n_lotes_cli / v_max,
                                               "moda": n_lotes_cli / v_moda,
                                               "maximo": n_lotes_cli / v_min},
                        "taxa_alvo_anual": _f("taxa_alvo_anual"),
                        "n_parcelas": int(_f("n_parcelas", 1)),
                        "mes_inicio_vendas": int(_f("mes_inicio_vendas", 1)),
                        "custos": {"comissao_pct": _f("comissao_pct"),
                                   "impostos_pct": _f("impostos_pct"),
                                   "marketing_pct": _f("marketing_pct"),
                                   "admin_mensal": _f("admin_mensal"),
                                   "licenciamento": _f("licenciamento")},
                    },
                    "n_sims": 3000,
                    "rho_mercado": _f("rho_mercado", 0.5),
                },
            )
            if corpo_otim:
                st.caption(
                    f"Comparando o seu lote de **{area_lote_cli:.0f} m²** com "
                    "outros tamanhos, na mesma gleba e premissas. A velocidade "
                    "de venda vem do prazo que você informou."
                )
                _render_otimizacao(corpo_otim)
    st.stop()


# ---------- terreno + mercado (compartilhado, fora dos formulários) ----------
# Antes na barra lateral; agora na área principal para as duas abas lerem. Tem
# de ficar FORA dos st.form, senão o Otimizador não enxergaria os valores.
with st.expander("📍 Terreno", expanded=True):
    t1, t2, t3 = st.columns(3)
    setor = t1.text_input("Setor fiscal (3 dígitos)", value="085")
    quadra = t2.text_input("Quadra fiscal (3 dígitos)", value="013")
    testada = t3.number_input("Frente do lote (m)", value=10.0,
                              min_value=1.0, step=1.0)
    rho_mercado = st.slider(
        "Correlação preço↔absorção", 0.0, 0.95, 0.5, 0.05,
        help="É uma CORRELAÇÃO (acoplamento), não o nível de vendas: liga preço e "
        "velocidade de venda na simulação. 0 = sorteados de forma independente; "
        ">0 = mercado quente (preço alto coincide com venda rápida), gerando "
        "caudas conjuntas mais realistas.",
    )
    _mapa_toggle(setor, quadra, key="mapa_interativo")

# ---------- premissas comuns às duas abas (fora dos formulários) ----------
# Definidas uma única vez: as duas abas (Viabilidade e Otimizador) leem estas
# variáveis, garantindo o mesmo empreendimento nas duas análises.
with st.expander("⚙️ Premissas comuns — valem para as duas abas", expanded=True):
    p1, p2 = st.columns(2)
    custo_gleba = p1.number_input("Custo da gleba (R$)", value=60_000_000.0, step=1e6)
    meses_obra = p2.number_input("Meses de obra", value=18, step=1)

    infra = _faixa("Custo de infra (R$)", 15e6, 20e6, 28e6, 1e6)

    st.markdown("**Custos operacionais**")
    d1, d2, d3, d4, d5 = st.columns(5)
    comissao = d1.number_input("Comissão %VGV", value=0.06, step=0.01, format="%.2f")
    impostos = d2.number_input("Impostos %VGV", value=0.04, step=0.01, format="%.2f")
    marketing = d3.number_input("Marketing %VGV", value=0.03, step=0.01, format="%.2f")
    admin = d4.number_input("Admin mensal (R$)", value=30_000.0, step=10_000.0)
    licenciamento = d5.number_input("Licenciamento (R$)", value=500_000.0, step=100_000.0)

    st.markdown("**Meta e cronograma de vendas**")
    s1, s2, s3 = st.columns(3)
    n_parcelas = s1.number_input("Parcelas por venda", value=24, step=6)
    mes_inicio = s2.number_input("Mês início vendas", value=6, step=1, min_value=1)
    taxa_alvo = s3.number_input("Taxa-alvo anual", value=0.18, step=0.01, format="%.2f")

aba_viab, aba_otim = st.tabs(["Viabilidade", "Otimizador de configuração"])

# ---------------- aba 1: viabilidade ----------------
with aba_viab:
    with st.form("form_viabilidade"):
        st.subheader("Projeto")
        c1, c2, c3 = st.columns(3)
        area_lote = c1.number_input("Área do lote (m²)", value=300.0, step=50.0)
        n_lotes = c2.number_input("Nº de lotes", value=100, step=10)
        n_sims = c3.number_input("Simulações", value=5_000, step=1_000)

        vendas = _faixa("Meses de venda", 12.0, 24.0, 48.0, 6.0)
        st.caption("Velocidade de venda como **duração** (meses até vender tudo) — "
                   "mesmo conceito da 'Velocidade de venda' do otimizador, lá "
                   "expressa como taxa (lotes/mês).")
        st.caption("Custos, infra, gleba, obra e meta vêm das **Premissas comuns** "
                   "acima.")

        rodar = st.form_submit_button("Simular viabilidade", type="primary")

    if rodar:
        corpo = _post(
            "/viabilidade",
            {
                "terreno": {
                    "setor": setor, "quadra": quadra,
                    "area_lote_m2": area_lote, "testada": testada,
                },
                "projeto": {
                    "n_lotes": int(n_lotes),
                    "custo_gleba": custo_gleba,
                    "custo_infra": dict(zip(("minimo", "moda", "maximo"), infra)),
                    "meses_obra": int(meses_obra),
                    "meses_vendas": dict(zip(("minimo", "moda", "maximo"), vendas)),
                    "n_parcelas": int(n_parcelas),
                    "taxa_alvo_anual": taxa_alvo,
                    "mes_inicio_vendas": int(mes_inicio),
                    "custos": {
                        "comissao_pct": comissao,
                        "impostos_pct": impostos,
                        "marketing_pct": marketing,
                        "admin_mensal": admin,
                        "licenciamento": licenciamento,
                    },
                },
                "n_sims": int(n_sims),
                "incluir_distribuicao": True,
                "rho_mercado": rho_mercado,
            },
        )
        if corpo:
            _render_relatorio(corpo)


# ---------------- aba 2: otimizador ----------------
with aba_otim:
    with st.form("form_otimizar"):
        st.subheader("Gleba")
        c1, c2 = st.columns(2)
        area_vendavel = c1.number_input("Área vendável (m²)", value=30_000.0, step=5_000.0)
        n_sims_o = c2.number_input("Simulações ", value=3_000, step=1_000)

        candidatos = st.text_input(
            "Tamanhos de lote candidatos (m², separados por vírgula)",
            value="150, 250, 400, 600",
        )
        absorcao = _faixa("Velocidade de venda (lotes/mês)", 2.0, 5.0, 10.0, 1.0)
        st.caption("Custos, infra, gleba, obra e meta vêm das **Premissas comuns** "
                   "acima.")

        rodar_o = st.form_submit_button("Otimizar configuração", type="primary")

    if rodar_o:
        try:
            lista_candidatos = [float(x) for x in candidatos.split(",") if x.strip()]
        except ValueError:
            st.error("Candidatos inválidos: use números separados por vírgula.")
            lista_candidatos = []
        if lista_candidatos:
            corpo = _post(
                "/otimizar",
                {
                    "terreno": {"setor": setor, "quadra": quadra, "testada": testada},
                    "gleba": {
                        "area_vendavel_m2": area_vendavel,
                        "candidatos_area_lote": lista_candidatos,
                        "custo_gleba": custo_gleba,
                        "custo_infra": dict(zip(("minimo", "moda", "maximo"), infra)),
                        "meses_obra": int(meses_obra),
                        "absorcao_lotes_mes": dict(
                            zip(("minimo", "moda", "maximo"), absorcao)
                        ),
                        "taxa_alvo_anual": taxa_alvo,
                        "n_parcelas": int(n_parcelas),
                        "mes_inicio_vendas": int(mes_inicio),
                        "custos": {
                            "comissao_pct": comissao,
                            "impostos_pct": impostos,
                            "marketing_pct": marketing,
                            "admin_mensal": admin,
                            "licenciamento": licenciamento,
                        },
                    },
                    "n_sims": int(n_sims_o),
                    "rho_mercado": rho_mercado,
                },
            )
            if corpo:
                _render_otimizacao(corpo)
