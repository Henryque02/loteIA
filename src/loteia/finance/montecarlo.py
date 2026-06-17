"""Camada 2 - simulacao de Monte Carlo.
Amostra a incerteza dos inputs (preco, absorcao, custo) -> distribuicao de TIR/VPL
-> probabilidade de superar a rentabilidade-alvo."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, triang

from loteia.config import SEED
from loteia.finance.cashflow import Custos, fluxo_loteamento, tir, tir_anual, vpl


@dataclass(frozen=True)
class Incerteza:
    """Input incerto modelado como distribuição triangular (min, moda, max)."""

    minimo: float
    moda: float
    maximo: float

    def amostrar(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.maximo <= self.minimo:
            return np.full(n, float(self.moda))
        return rng.triangular(self.minimo, self.moda, self.maximo, size=n)

    def quantil(self, u: np.ndarray) -> np.ndarray:
        """Inversa da CDF (ppf) nos uniformes u — base para amostragem por cópula."""
        if self.maximo <= self.minimo:
            return np.full_like(np.asarray(u, dtype=float), float(self.moda))
        larg = self.maximo - self.minimo
        c = (self.moda - self.minimo) / larg
        return triang.ppf(u, c, loc=self.minimo, scale=larg)


def amostrar_mercado_correlacionado(
    preco_lote: Incerteza,
    meses_vendas: Incerteza,
    rho: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra preço e meses de venda acoplados por um 'estado de mercado' comum
    (cópula gaussiana). rho>0 = mercado quente: preço alto coincide com venda
    rápida (menos meses) → correlação negativa entre as séries. rho=0 = indep.
    """
    if rho <= 0.0:
        return preco_lote.amostrar(rng, n), meses_vendas.amostrar(rng, n)
    m = rng.standard_normal(n)  # fator latente de mercado
    e_p = rng.standard_normal(n)
    e_v = rng.standard_normal(n)
    raiz = np.sqrt(1.0 - rho * rho)
    z_p = rho * m + raiz * e_p          # preço carrega +rho no mercado
    z_v = -rho * m + raiz * e_v         # meses carrega -rho (quente → menos meses)
    precos = preco_lote.quantil(norm.cdf(z_p))
    meses = meses_vendas.quantil(norm.cdf(z_v))
    return precos, meses


def simular_viabilidade(
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
    rho_mercado: float = 0.0,
    n_sims: int = 10_000,
    seed: int = SEED,
) -> dict:
    """Propaga a incerteza dos inputs pelo fluxo de caixa.

    Viável = VPL > 0 descontado à taxa-alvo (equivale a TIR > alvo).
    `custos` (comissão/impostos/marketing/admin/licenciamento) são parâmetros
    fixos do projeto, não dimensões incertas — as incertezas dominantes são
    preço, infra e absorção.

    Refinamento (default = comportamento independente):
    - `rho_mercado` (>0): acopla preço e absorção por um estado de mercado comum.

    Retorna prob_viavel e as distribuições de VPL e TIR anual (NaN onde a TIR é
    indefinida, ex.: fluxo sem inversão de sinal).
    """
    rng = np.random.default_rng(seed)
    precos = preco_lote.amostrar(rng, n_sims)
    infras = custo_infra.amostrar(rng, n_sims)
    vendas_f = meses_vendas.amostrar(rng, n_sims)
    if rho_mercado > 0.0:
        precos, vendas_f = amostrar_mercado_correlacionado(
            preco_lote, meses_vendas, rho_mercado, n_sims, rng
        )
    vendas = np.maximum(1, np.round(vendas_f)).astype(int)

    taxa_alvo_mensal = (1.0 + taxa_alvo_anual) ** (1 / 12) - 1.0
    vpls = np.empty(n_sims)
    tirs = np.full(n_sims, np.nan)
    for i in range(n_sims):
        fluxo = fluxo_loteamento(
            n_lotes=n_lotes,
            preco_lote=precos[i],
            custo_gleba=custo_gleba,
            custo_infra=infras[i],
            meses_obra=meses_obra,
            meses_vendas=vendas[i],
            n_parcelas=n_parcelas,
            comissao_pct=custos.comissao_pct,
            impostos_pct=custos.impostos_pct,
            marketing_pct=custos.marketing_pct,
            admin_mensal=custos.admin_mensal,
            licenciamento=custos.licenciamento,
            mes_inicio_vendas=mes_inicio_vendas,
        )
        vpls[i] = vpl(fluxo, taxa_alvo_mensal)
        try:
            tirs[i] = tir_anual(tir(fluxo))
        except ValueError:
            pass

    return {
        "prob_viavel": float(np.mean(vpls > 0)),
        "vpl": vpls,
        "tir_anual": tirs,
        "taxa_alvo_anual": taxa_alvo_anual,
    }
