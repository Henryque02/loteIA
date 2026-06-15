"""Camada 3 - analise de sensibilidade: de qual variavel a viabilidade mais depende."""
import pandas as pd

from loteia.finance.cashflow import Custos, fluxo_loteamento, vpl
from loteia.finance.montecarlo import Incerteza


def sensibilidade_tornado(
    *,
    n_lotes: int,
    custo_gleba: float,
    meses_obra: int,
    preco_lote: Incerteza,
    custo_infra: Incerteza,
    meses_vendas: Incerteza,
    taxa_alvo_anual: float,
    n_parcelas: int = 1,
    custos: Custos = Custos(),
    mes_inicio_vendas: int = 1,
) -> pd.DataFrame:
    """Tornado (one-at-a-time): varia cada input do mínimo ao máximo com os
    demais fixos na moda e mede o VPL à taxa-alvo nas duas pontas.

    Retorna colunas [variavel, vpl_no_minimo, vpl_no_maximo, amplitude],
    ordenado por amplitude decrescente — a 1ª linha é a variável da qual a
    viabilidade mais depende.
    """
    taxa_mensal = (1.0 + taxa_alvo_anual) ** (1 / 12) - 1.0
    incertos = {
        "preco_lote": preco_lote,
        "custo_infra": custo_infra,
        "meses_vendas": meses_vendas,
    }

    def _vpl(valores: dict[str, float]) -> float:
        fluxo = fluxo_loteamento(
            n_lotes=n_lotes,
            preco_lote=valores["preco_lote"],
            custo_gleba=custo_gleba,
            custo_infra=valores["custo_infra"],
            meses_obra=meses_obra,
            meses_vendas=max(1, round(valores["meses_vendas"])),
            n_parcelas=n_parcelas,
            comissao_pct=custos.comissao_pct,
            impostos_pct=custos.impostos_pct,
            marketing_pct=custos.marketing_pct,
            admin_mensal=custos.admin_mensal,
            licenciamento=custos.licenciamento,
            mes_inicio_vendas=mes_inicio_vendas,
        )
        return vpl(fluxo, taxa_mensal)

    modas = {nome: inc.moda for nome, inc in incertos.items()}
    linhas = []
    for nome, inc in incertos.items():
        v_min = _vpl({**modas, nome: inc.minimo})
        v_max = _vpl({**modas, nome: inc.maximo})
        linhas.append(
            {
                "variavel": nome,
                "vpl_no_minimo": v_min,
                "vpl_no_maximo": v_max,
                "amplitude": abs(v_max - v_min),
            }
        )

    return (
        pd.DataFrame(linhas)
        .sort_values("amplitude", ascending=False)
        .reset_index(drop=True)
    )
