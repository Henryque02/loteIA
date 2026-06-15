"""Camada 6 (opcional) - highest-and-best-use.
Busca a configuracao do loteamento (n. e tamanho dos lotes, faseamento) que
maximiza a probabilidade de retorno."""
from typing import Callable

import numpy as np
import pandas as pd

from loteia.config import SEED
from loteia.finance.cashflow import Custos
from loteia.finance.montecarlo import Incerteza, simular_viabilidade


def precificador_do_modelo(modelo, exemplo: pd.DataFrame) -> Callable[[float], Incerteza]:
    """Liga o modelo de preço ao otimizador: o intervalo de preço/m² previsto
    para um lote do tamanho candidato vira a Incerteza do preço do lote.

    `exemplo` é UMA linha com as demais features do terreno (setor, testada,
    features geo...); a área é sobrescrita a cada candidato.
    """

    def precificar(area_lote: float) -> Incerteza:
        X = exemplo.copy()
        X["area_terreno_itbi"] = area_lote
        faixa = modelo.predict_intervalo(X).iloc[0]
        return Incerteza(
            faixa["preco_m2_inf"] * area_lote,
            faixa["preco_m2_med"] * area_lote,
            faixa["preco_m2_sup"] * area_lote,
        )

    return precificar


def otimizar_configuracao(
    *,
    area_vendavel_m2: float,
    candidatos_area_lote: list[float],
    precificador: Callable[[float], Incerteza],
    custo_gleba: float,
    custo_infra: Incerteza,
    meses_obra: int,
    absorcao_lotes_mes: Incerteza,
    taxa_alvo_anual: float,
    n_parcelas: int = 1,
    custos: Custos = Custos(),
    mes_inicio_vendas: int = 1,
    rho_mercado: float = 0.0,
    n_sims: int = 5_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compara configurações de lote e ordena pela probabilidade de viabilidade.

    Para cada tamanho candidato: n_lotes = área vendável ÷ área do lote; o
    `precificador` dá a distribuição do preço daquele lote (o modelo de preço
    captura o efeito do tamanho); a absorção (lotes/mês) vira meses de venda
    (absorção alta → venda rápida, por isso o intervalo se inverte).
    """
    linhas = []
    for area_lote in candidatos_area_lote:
        n_lotes = int(area_vendavel_m2 // area_lote)
        if n_lotes == 0:
            continue
        meses_vendas = Incerteza(
            n_lotes / absorcao_lotes_mes.maximo,
            n_lotes / absorcao_lotes_mes.moda,
            n_lotes / absorcao_lotes_mes.minimo,
        )
        r = simular_viabilidade(
            n_lotes=n_lotes,
            custo_gleba=custo_gleba,
            meses_obra=meses_obra,
            preco_lote=precificador(area_lote),
            custo_infra=custo_infra,
            meses_vendas=meses_vendas,
            taxa_alvo_anual=taxa_alvo_anual,
            n_parcelas=n_parcelas,
            custos=custos,
            mes_inicio_vendas=mes_inicio_vendas,
            rho_mercado=rho_mercado,
            n_sims=n_sims,
            seed=seed,
        )
        linhas.append(
            {
                "area_lote_m2": area_lote,
                "n_lotes": n_lotes,
                "prob_viavel": r["prob_viavel"],
                "vpl_mediano": float(np.median(r["vpl"])),
                "tir_anual_mediana": float(np.nanmedian(r["tir_anual"]))
                if np.isfinite(r["tir_anual"]).any()
                else float("nan"),
            }
        )

    return (
        pd.DataFrame(linhas)
        .sort_values(["prob_viavel", "vpl_mediano"], ascending=False)
        .reset_index(drop=True)
    )
