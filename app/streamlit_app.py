"""Front Streamlit do LoteIA: consome a FastAPI (mesmo caminho do n8n).
Aba 1: viabilidade de um projeto definido. Aba 2: otimizador de configuração.
Modo relatório: se a URL traz os parâmetros (link do n8n), renderiza a análise
pronta e pula o formulário. Suba a API antes: `make api` (front com `make front`)."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from loteia.report import payload_de_params, tem_params_relatorio

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


def _quadra_latlon(setor: str, quadra: str) -> dict | None:
    try:
        r = requests.get(f"{API}/quadra/{setor}/{quadra}", timeout=30)
        return r.json() if r.status_code == 200 else None
    except requests.ConnectionError:
        return None


def _barras_horizontais(rotulos: list[str], valores: list[float], titulo: str):
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(rotulos) + 1))
    cores = ["#d62728" if v < 0 else "#2ca02c" for v in valores]
    ax.barh(rotulos, valores, color=cores)
    ax.set_title(titulo)
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _histograma(valores: list[float], titulo: str, alvo: float | None = None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(valores, bins=40, color="#1f77b4", alpha=0.85)
    if alvo is not None:
        ax.axvline(alvo, color="#d62728", linestyle="--", label="alvo")
        ax.legend()
    ax.set_title(titulo)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _faixa(rotulo: str, minimo: float, moda: float, maximo: float, passo: float):
    c1, c2, c3 = st.columns(3)
    return (
        c1.number_input(f"{rotulo} (mín)", value=minimo, step=passo),
        c2.number_input(f"{rotulo} (moda)", value=moda, step=passo),
        c3.number_input(f"{rotulo} (máx)", value=maximo, step=passo),
    )


def _render_relatorio(corpo: dict):
    """Renderiza o relatório de viabilidade (métricas + gráficos) a partir da
    resposta de /viabilidade. Usado pela aba interativa e pelo modo relatório."""
    prob = corpo["prob_viavel"]
    faixa = corpo["faixa_preco_m2"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(superar a taxa-alvo)", f"{prob:.0%}")
    m2.metric("Preço/m² (10%)", f"R$ {faixa['inf']:,.0f}")
    m3.metric("Preço/m² (mediana)", f"R$ {faixa['med']:,.0f}")
    m4.metric("Preço/m² (90%)", f"R$ {faixa['sup']:,.0f}")

    dist = corpo.get("distribuicoes")
    g1, g2 = st.columns(2)
    with g1:
        if dist and dist["tir_anual"]:
            _histograma(dist["tir_anual"], "Distribuição da TIR anual",
                        alvo=corpo["taxa_alvo_anual"])
    with g2:
        if dist:
            _histograma(dist["vpl"], "Distribuição do VPL (R$)", alvo=0.0)

    g3, g4 = st.columns(2)
    with g3:
        sens = pd.DataFrame(corpo["sensibilidade"])
        _barras_horizontais(sens["variavel"].tolist()[::-1],
                            sens["amplitude"].tolist()[::-1],
                            "Sensibilidade do VPL (tornado)")
    with g4:
        if corpo["fatores_shap"]:
            contrib = corpo["fatores_shap"]["contribuicoes"]
            ordem = sorted(contrib, key=lambda k: abs(contrib[k]))
            _barras_horizontais(ordem, [contrib[k] for k in ordem],
                                "Fatores do preço/m² (SHAP, R$/m²)")


# ---------------- modo relatório (link do n8n) ----------------
_params = st.query_params
if tem_params_relatorio(_params):
    emp = _params.get("empreendimento", "")
    setor_r, quadra_r = _params.get("setor", ""), _params.get("quadra", "")
    st.subheader(f"Relatório de viabilidade — {emp}" if emp
                 else "Relatório de viabilidade")
    st.caption(f"Setor fiscal {setor_r} · quadra {quadra_r}")
    loc = _quadra_latlon(setor_r, quadra_r)
    if loc:
        st.map(pd.DataFrame({"lat": [loc["lat"]], "lon": [loc["lon"]]}), zoom=14)
    corpo = _post("/viabilidade", payload_de_params(_params))
    if corpo:
        _render_relatorio(corpo)
    st.caption("Resultado probabilístico (simulação de Monte Carlo) — não é "
               "garantia de resultado do empreendimento.")
    st.stop()


# ---------------- sidebar: terreno ----------------
with st.sidebar:
    st.header("Terreno")
    setor = st.text_input("Setor fiscal (3 dígitos)", value="085")
    quadra = st.text_input("Quadra fiscal (3 dígitos)", value="013")
    testada = st.number_input("Testada (m)", value=10.0, min_value=1.0, step=1.0)
    st.divider()
    st.subheader("Mercado")
    rho_mercado = st.slider(
        "Correlação preço↔absorção", 0.0, 0.95, 0.5, 0.05,
        help="Mercado quente: preço alto coincide com venda rápida. "
        "0 = inputs independentes; mais alto = caudas conjuntas mais realistas.",
    )
    st.caption(f"API: {API}")

    local = _quadra_latlon(setor, quadra)
    if local:
        st.success("Quadra localizada ✔")
        st.map(
            pd.DataFrame({"lat": [local["lat"]], "lon": [local["lon"]]}),
            zoom=14,
        )
    else:
        st.warning("Quadra não localizada (confira setor/quadra ou suba a API).")

aba_viab, aba_otim = st.tabs(["Viabilidade", "Otimizador de configuração"])

# ---------------- aba 1: viabilidade ----------------
with aba_viab:
    with st.form("form_viabilidade"):
        st.subheader("Projeto")
        c1, c2, c3, c4 = st.columns(4)
        area_lote = c1.number_input("Área do lote (m²)", value=300.0, step=50.0)
        n_lotes = c2.number_input("Nº de lotes", value=100, step=10)
        custo_gleba = c3.number_input("Custo da gleba (R$)", value=60_000_000.0, step=1e6)
        meses_obra = c4.number_input("Meses de obra", value=18, step=1)

        infra = _faixa("Custo de infra (R$)", 15e6, 20e6, 28e6, 1e6)
        vendas = _faixa("Meses de venda", 12.0, 24.0, 48.0, 6.0)

        st.markdown("**Custos operacionais**")
        d1, d2, d3, d4, d5 = st.columns(5)
        comissao = d1.number_input("Comissão %VGV", value=0.06, step=0.01, format="%.2f")
        impostos = d2.number_input("Impostos %VGV", value=0.04, step=0.01, format="%.2f")
        marketing = d3.number_input("Marketing %VGV", value=0.03, step=0.01, format="%.2f")
        admin = d4.number_input("Admin mensal (R$)", value=30_000.0, step=10_000.0)
        licenciamento = d5.number_input("Licenciamento (R$)", value=500_000.0, step=100_000.0)

        c5, c6, c7, c8 = st.columns(4)
        n_parcelas = c5.number_input("Parcelas por venda", value=24, step=6)
        mes_inicio = c6.number_input("Mês início vendas", value=6, step=1, min_value=1)
        taxa_alvo = c7.number_input("Taxa-alvo anual", value=0.18, step=0.01, format="%.2f")
        n_sims = c8.number_input("Simulações", value=5_000, step=1_000)

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
        c1, c2, c3 = st.columns(3)
        area_vendavel = c1.number_input("Área vendável (m²)", value=30_000.0, step=5_000.0)
        custo_gleba_o = c2.number_input("Custo da gleba (R$) ", value=60_000_000.0, step=1e6)
        meses_obra_o = c3.number_input("Meses de obra ", value=18, step=1)

        candidatos = st.text_input(
            "Tamanhos de lote candidatos (m², separados por vírgula)",
            value="150, 250, 400, 600",
        )
        infra_o = _faixa("Custo de infra (R$) ", 15e6, 20e6, 28e6, 1e6)
        absorcao = _faixa("Absorção (lotes/mês)", 2.0, 5.0, 10.0, 1.0)

        st.markdown("**Custos operacionais**")
        e1, e2, e3, e4, e5 = st.columns(5)
        comissao_o = e1.number_input("Comissão %VGV ", value=0.06, step=0.01, format="%.2f")
        impostos_o = e2.number_input("Impostos %VGV ", value=0.04, step=0.01, format="%.2f")
        marketing_o = e3.number_input("Marketing %VGV ", value=0.03, step=0.01, format="%.2f")
        admin_o = e4.number_input("Admin mensal (R$) ", value=30_000.0, step=10_000.0)
        licenciamento_o = e5.number_input("Licenciamento (R$) ", value=500_000.0, step=100_000.0)

        c4, c5, c6, c7 = st.columns(4)
        n_parcelas_o = c4.number_input("Parcelas por venda ", value=24, step=6)
        mes_inicio_o = c5.number_input("Mês início vendas ", value=6, step=1, min_value=1)
        taxa_alvo_o = c6.number_input("Taxa-alvo anual ", value=0.18, step=0.01, format="%.2f")
        n_sims_o = c7.number_input("Simulações ", value=3_000, step=1_000)

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
                        "custo_gleba": custo_gleba_o,
                        "custo_infra": dict(zip(("minimo", "moda", "maximo"), infra_o)),
                        "meses_obra": int(meses_obra_o),
                        "absorcao_lotes_mes": dict(
                            zip(("minimo", "moda", "maximo"), absorcao)
                        ),
                        "taxa_alvo_anual": taxa_alvo_o,
                        "n_parcelas": int(n_parcelas_o),
                        "mes_inicio_vendas": int(mes_inicio_o),
                        "custos": {
                            "comissao_pct": comissao_o,
                            "impostos_pct": impostos_o,
                            "marketing_pct": marketing_o,
                            "admin_mensal": admin_o,
                            "licenciamento": licenciamento_o,
                        },
                    },
                    "n_sims": int(n_sims_o),
                    "rho_mercado": rho_mercado,
                },
            )
            if corpo:
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
