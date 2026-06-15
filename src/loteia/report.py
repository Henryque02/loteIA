"""Modo relatório do front: reconstrói o payload de /viabilidade a partir dos
parâmetros (flat) embutidos numa URL. Permite que o n8n monte um link que abre
a página do Streamlit já com a análise pronta (sem o cliente preencher nada).

Função pura, sem dependência de Streamlit — compartilhável e testável.
"""
from collections.abc import Mapping

# campos mínimos que caracterizam um link de relatório (vs. acesso interativo)
OBRIGATORIOS: tuple[str, ...] = ("setor", "n_lotes")


def tem_params_relatorio(params: Mapping[str, str]) -> bool:
    """True se a URL carrega os parâmetros de uma análise (modo relatório)."""
    return all(params.get(k) not in (None, "") for k in OBRIGATORIOS)


def _reais(v: float) -> str:
    """R$ com sufixo legível e decimal brasileiro (1,5 mi)."""
    a = abs(v)
    if a >= 1_000_000:
        s = f"{v / 1_000_000:.1f}".rstrip("0").rstrip(".").replace(".", ",")
        return f"R$ {s} mi"
    if a >= 1_000:
        return f"R$ {v / 1_000:.0f} mil"
    return f"R$ {v:.0f}"


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def resumo_respostas(params: Mapping[str, str]) -> list[tuple[str, str]]:
    """Painel read-only das respostas do cliente (rótulo, valor formatado).

    Mostra o que foi pedido no Form, formatado (R$, %, m²), para o relatório.
    """
    def f(chave: str, default: float = 0.0) -> float:
        v = params.get(chave)
        return float(v) if v not in (None, "") else float(default)

    itens: list[tuple[str, str]] = []
    emp = params.get("empreendimento")
    if emp:
        itens.append(("Empreendimento", emp))
    itens += [
        ("Localização", f"Setor {params.get('setor', '')} · "
                        f"Quadra {params.get('quadra', '')}"),
        ("Área do lote", f"{f('area_lote_m2'):.0f} m²"),
        ("Nº de lotes", f"{int(f('n_lotes'))}"),
        ("Custo da gleba", _reais(f("custo_gleba"))),
        ("Custo de infra", _reais(f("infra_moda"))),
        ("Prazo de obra", f"{int(f('meses_obra'))} meses"),
        ("Prazo de vendas", f"{int(f('vendas_moda'))} meses"),
        ("Taxa-alvo (a.a.)", _pct(f("taxa_alvo_anual"))),
        ("Comissão/impostos/marketing",
         f"{_pct(f('comissao_pct'))} / {_pct(f('impostos_pct'))} / "
         f"{_pct(f('marketing_pct'))}"),
    ]
    return itens


def payload_de_params(params: Mapping[str, str]) -> dict:
    """Reconstrói o corpo de POST /viabilidade a partir dos query params.

    Tudo chega como string (query string). Faixas vêm como `*_min/_moda/_max`.
    Campos opcionais ausentes caem em defaults seguros (iguais aos do n8n).
    """
    def f(chave: str, default: float = 0.0) -> float:
        v = params.get(chave)
        return float(v) if v not in (None, "") else float(default)

    def i(chave: str, default: int = 0) -> int:
        return int(round(f(chave, default)))

    return {
        "terreno": {
            "setor": params.get("setor", ""),
            "quadra": params.get("quadra", ""),
            "area_lote_m2": f("area_lote_m2"),
            "testada": f("testada", 10),
        },
        "projeto": {
            "n_lotes": i("n_lotes"),
            "custo_gleba": f("custo_gleba"),
            "custo_infra": {
                "minimo": f("infra_min"),
                "moda": f("infra_moda"),
                "maximo": f("infra_max"),
            },
            "meses_obra": i("meses_obra"),
            "meses_vendas": {
                "minimo": f("vendas_min"),
                "moda": f("vendas_moda"),
                "maximo": f("vendas_max"),
            },
            "n_parcelas": i("n_parcelas", 1),
            "taxa_alvo_anual": f("taxa_alvo_anual"),
            "mes_inicio_vendas": i("mes_inicio_vendas", 1),
            "custos": {
                "comissao_pct": f("comissao_pct"),
                "impostos_pct": f("impostos_pct"),
                "marketing_pct": f("marketing_pct"),
                "admin_mensal": f("admin_mensal"),
                "licenciamento": f("licenciamento"),
            },
        },
        "n_sims": i("n_sims", 5000),
        "incluir_distribuicao": True,
        "rho_mercado": f("rho_mercado", 0.5),
    }
