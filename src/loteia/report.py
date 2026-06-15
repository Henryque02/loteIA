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
