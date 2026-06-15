"""Testes da Camada 6: busca da configuração de loteamento que maximiza P(viável)."""

import pandas as pd
import pytest

from loteia.finance.montecarlo import Incerteza
from loteia.optimize.config_search import (
    otimizar_configuracao,
    precificador_do_modelo,
)


class _ModeloFixo:
    """Dublê do ModeloQuantilico: preço/m² fixo em (900, 1000, 1100)."""

    def predict_intervalo(self, df: pd.DataFrame) -> pd.DataFrame:
        assert "area_terreno_itbi" in df.columns
        n = len(df)
        return pd.DataFrame(
            {"preco_m2_inf": [900.0] * n,
             "preco_m2_med": [1000.0] * n,
             "preco_m2_sup": [1100.0] * n},
            index=df.index,
        )


class TestPrecificadorDoModelo:
    def test_escala_o_intervalo_pela_area_do_lote(self):
        exemplo = pd.DataFrame([{"setor": "085", "testada": 10.0}])
        prec = precificador_do_modelo(_ModeloFixo(), exemplo)
        inc = prec(300.0)
        assert inc.minimo == pytest.approx(900.0 * 300)
        assert inc.moda == pytest.approx(1000.0 * 300)
        assert inc.maximo == pytest.approx(1100.0 * 300)

    def test_passa_a_area_do_candidato_para_o_modelo(self):
        chamadas = []

        class Espiao(_ModeloFixo):
            def predict_intervalo(self, df):
                chamadas.append(df["area_terreno_itbi"].iloc[0])
                return super().predict_intervalo(df)

        exemplo = pd.DataFrame([{"setor": "085", "testada": 10.0}])
        prec = precificador_do_modelo(Espiao(), exemplo)
        prec(150.0)
        prec(600.0)
        assert chamadas == [150.0, 600.0]


def _precificador(area_lote: float) -> Incerteza:
    """Preço do lote sintético: R$ 1.000/m² ± 20%, sem prêmio por tamanho."""
    base = 1_000.0 * area_lote
    return Incerteza(0.8 * base, base, 1.2 * base)


def _otimizar(**kw):
    base = dict(
        area_vendavel_m2=30_000.0,
        candidatos_area_lote=[150.0, 300.0, 600.0],
        precificador=_precificador,
        custo_gleba=10_000_000.0,
        custo_infra=Incerteza(8_000_000, 10_000_000, 12_000_000),
        meses_obra=12,
        absorcao_lotes_mes=Incerteza(2, 4, 8),
        taxa_alvo_anual=0.15,
        n_sims=300,
        seed=42,
    )
    base.update(kw)
    return otimizar_configuracao(**base)


class TestOtimizarConfiguracao:
    def test_uma_linha_por_candidato_ordenada_pela_prob(self):
        r = _otimizar()
        assert len(r) == 3
        assert r["prob_viavel"].is_monotonic_decreasing

    def test_n_lotes_respeita_a_area_vendavel(self):
        r = _otimizar()
        for _, linha in r.iterrows():
            assert linha["n_lotes"] == int(30_000.0 // linha["area_lote_m2"])

    def test_probs_validas(self):
        r = _otimizar()
        assert r["prob_viavel"].between(0, 1).all()

    def test_melhor_configuracao_e_a_primeira(self):
        r = _otimizar()
        assert r.iloc[0]["prob_viavel"] == r["prob_viavel"].max()

    def test_premio_de_preco_muda_o_vencedor(self):
        # se o m² do lote pequeno vale o dobro, o lote pequeno deve vencer
        def premium(area_lote: float) -> Incerteza:
            m2 = 2_000.0 if area_lote <= 200 else 1_000.0
            base = m2 * area_lote
            return Incerteza(0.9 * base, base, 1.1 * base)

        r = _otimizar(precificador=premium)
        assert r.iloc[0]["area_lote_m2"] == 150.0

    def test_reprodutivel_com_seed(self):
        r1, r2 = _otimizar(), _otimizar()
        assert r1["prob_viavel"].tolist() == r2["prob_viavel"].tolist()

    def test_custos_operacionais_reduzem_prob(self):
        from loteia.finance.cashflow import Custos

        sem = _otimizar()
        com = _otimizar(custos=Custos(comissao_pct=0.06, impostos_pct=0.02,
                                      admin_mensal=30_000.0))
        # com custos, a melhor prob não pode ser maior que sem custos
        assert com["prob_viavel"].max() <= sem["prob_viavel"].max()
