"""Motor de fluxo de caixa do loteamento -> VPL e TIR.
Receita = preco do lote distribuido no tempo (absorcao). Custos = gleba + infra."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Custos:
    """Custos operacionais do loteamento, além de gleba e infra.

    Percentuais incidem sobre o VGV (casados com cada recebimento); admin é
    mensal ao longo do horizonte; licenciamento é pago à vista no mês 0.
    """

    comissao_pct: float = 0.0
    impostos_pct: float = 0.0
    marketing_pct: float = 0.0
    admin_mensal: float = 0.0
    licenciamento: float = 0.0


def vpl(fluxo: ArrayLike, taxa: float) -> float:
    """Valor presente líquido do fluxo (índice = período) à taxa por período."""
    f = np.asarray(fluxo, dtype=float)
    t = np.arange(len(f))
    return float(np.sum(f / (1.0 + taxa) ** t))


def tir(fluxo: ArrayLike, lo: float = -0.99, hi: float = 10.0) -> float:
    """Taxa interna de retorno por período (raiz do VPL, por bisseção).

    Exige fluxo com entrada e saída (inversão de sinal); senão ValueError.
    """
    f = np.asarray(fluxo, dtype=float)
    if not (np.any(f > 0) and np.any(f < 0)):
        raise ValueError("TIR indefinida: fluxo sem inversão de sinal.")
    v_lo, v_hi = vpl(f, lo), vpl(f, hi)
    if v_lo * v_hi > 0:
        raise ValueError("TIR fora do intervalo de busca [-99%, 1000%].")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        v_mid = vpl(f, mid)
        if abs(v_mid) < 1e-9 or hi - lo < 1e-12:
            return mid
        if v_lo * v_mid <= 0:
            hi = mid
        else:
            lo, v_lo = mid, v_mid
    return (lo + hi) / 2.0


def tir_anual(tir_mensal: float) -> float:
    """Converte TIR mensal em anual por capitalização composta."""
    return (1.0 + tir_mensal) ** 12 - 1.0


def fluxo_loteamento(
    n_lotes: int,
    preco_lote: float,
    custo_gleba: float,
    custo_infra: float,
    meses_obra: int,
    meses_vendas: int,
    n_parcelas: int = 1,
    *,
    comissao_pct: float = 0.0,
    impostos_pct: float = 0.0,
    marketing_pct: float = 0.0,
    admin_mensal: float = 0.0,
    licenciamento: float = 0.0,
    mes_inicio_vendas: int = 1,
) -> np.ndarray:
    """Fluxo de caixa mensal do loteamento (índice 0 = aquisição da gleba).

    Custos fixos:
    - Gleba + licenciamento pagos à vista no mês 0.
    - Infraestrutura distribuída uniformemente nos meses 1..meses_obra.
    - Administração: valor fixo mensal ao longo de todo o horizonte.

    Receita e custos sobre o VGV (comissão + impostos + marketing) ocorrem
    casados com cada recebimento — vendas uniformes em
    `mes_inicio_vendas .. mes_inicio_vendas+meses_vendas-1`, cada venda recebida
    em n_parcelas mensais a partir do mês da venda. O custo % é debitado na
    mesma parcela em que a receita entra (logo escala com o VGV).
    """
    ultimo_recebimento = mes_inicio_vendas + meses_vendas + n_parcelas - 2
    n_meses = 1 + max(meses_obra, ultimo_recebimento)
    f = np.zeros(n_meses)

    f[0] -= custo_gleba + licenciamento
    f[1 : meses_obra + 1] -= custo_infra / meses_obra
    f -= admin_mensal

    pct_venda = comissao_pct + impostos_pct + marketing_pct
    parcela_bruta = n_lotes * preco_lote / meses_vendas / n_parcelas
    parcela_liquida = parcela_bruta * (1.0 - pct_venda)
    for mes_venda in range(mes_inicio_vendas, mes_inicio_vendas + meses_vendas):
        f[mes_venda : mes_venda + n_parcelas] += parcela_liquida
    return f
