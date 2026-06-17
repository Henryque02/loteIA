"""Testes da simulação de Monte Carlo (Camada 2): probabilidade de viabilidade."""
import numpy as np
import pytest

from loteia.finance.cashflow import Custos
from loteia.finance.montecarlo import (
    Incerteza,
    amostrar_mercado_correlacionado,
    simular_viabilidade,
)


PROJETO = dict(
    n_lotes=50,
    custo_gleba=2_000_000.0,
    meses_obra=12,
    n_parcelas=1,
    taxa_alvo_anual=0.15,
)


def _sim(n_sims=500, seed=42, **kw):
    base = dict(
        PROJETO,
        preco_lote=Incerteza(90_000, 100_000, 110_000),
        custo_infra=Incerteza(1_500_000, 2_000_000, 2_500_000),
        meses_vendas=Incerteza(12, 24, 48),
    )
    base.update(kw)
    return simular_viabilidade(n_sims=n_sims, seed=seed, **base)


class TestIncerteza:
    def test_degenerada_amostra_constante(self):
        rng = np.random.default_rng(0)
        x = Incerteza(100.0, 100.0, 100.0).amostrar(rng, 10)
        assert np.allclose(x, 100.0)

    def test_amostras_dentro_dos_limites(self):
        rng = np.random.default_rng(0)
        x = Incerteza(10.0, 20.0, 30.0).amostrar(rng, 1000)
        assert x.min() >= 10.0 and x.max() <= 30.0


class TestSimularViabilidade:
    def test_caso_degenerado_lucrativo_da_prob_1(self):
        # sem incerteza e muito lucrativo → P(viável) = 1
        r = _sim(
            preco_lote=Incerteza(500_000, 500_000, 500_000),
            custo_infra=Incerteza(1_000_000, 1_000_000, 1_000_000),
            meses_vendas=Incerteza(12, 12, 12),
        )
        assert r["prob_viavel"] == pytest.approx(1.0)

    def test_caso_degenerado_inviavel_da_prob_0(self):
        # receita não cobre nem a gleba → P(viável) = 0
        r = _sim(
            preco_lote=Incerteza(10_000, 10_000, 10_000),
            custo_infra=Incerteza(2_000_000, 2_000_000, 2_000_000),
            meses_vendas=Incerteza(12, 12, 12),
        )
        assert r["prob_viavel"] == pytest.approx(0.0)

    def test_prob_entre_0_e_1(self):
        r = _sim()
        assert 0.0 <= r["prob_viavel"] <= 1.0

    def test_reprodutivel_com_seed(self):
        r1, r2 = _sim(seed=7), _sim(seed=7)
        assert r1["prob_viavel"] == r2["prob_viavel"]
        assert np.array_equal(r1["vpl"], r2["vpl"])

    def test_preco_maior_nao_reduz_prob(self):
        barato = _sim(preco_lote=Incerteza(60_000, 70_000, 80_000))
        caro = _sim(preco_lote=Incerteza(120_000, 140_000, 160_000))
        assert caro["prob_viavel"] >= barato["prob_viavel"]

    def test_saida_tem_distribuicoes(self):
        r = _sim(n_sims=200)
        assert len(r["vpl"]) == 200
        assert len(r["tir_anual"]) == 200
        assert np.isfinite(r["vpl"]).all()

    def test_custos_operacionais_reduzem_prob(self):
        sem = _sim()
        com = _sim(custos=Custos(comissao_pct=0.06, impostos_pct=0.02,
                                 marketing_pct=0.03, admin_mensal=20_000.0,
                                 licenciamento=500_000.0))
        assert com["prob_viavel"] <= sem["prob_viavel"]
        assert np.median(com["vpl"]) < np.median(sem["vpl"])

    def test_custos_default_inalterado(self):
        # Custos() vazio reproduz o resultado sem custos operacionais
        a = _sim(seed=7)
        b = _sim(seed=7, custos=Custos())
        assert a["prob_viavel"] == b["prob_viavel"]
        assert np.array_equal(a["vpl"], b["vpl"])

    def test_rho_zero_reproduz_resultado(self):
        a = _sim(seed=11)
        b = _sim(seed=11, rho_mercado=0.0)
        assert np.array_equal(a["vpl"], b["vpl"])


class TestMercadoCorrelacionado:
    def test_rho_zero_quase_sem_correlacao(self):
        rng = np.random.default_rng(0)
        preco = Incerteza(90_000, 100_000, 110_000)
        meses = Incerteza(12, 24, 48)
        p, m = amostrar_mercado_correlacionado(preco, meses, rho=0.0, n=20_000, rng=rng)
        assert abs(np.corrcoef(p, m)[0, 1]) < 0.05

    def test_rho_positivo_mercado_quente_preco_alto_venda_rapida(self):
        # mercado quente: preço alto coincide com MENOS meses de venda → corr < 0
        rng = np.random.default_rng(0)
        preco = Incerteza(90_000, 100_000, 110_000)
        meses = Incerteza(12, 24, 48)
        p, m = amostrar_mercado_correlacionado(preco, meses, rho=0.8, n=20_000, rng=rng)
        assert np.corrcoef(p, m)[0, 1] < -0.4

    def test_amostras_respeitam_limites(self):
        rng = np.random.default_rng(1)
        preco = Incerteza(90_000, 100_000, 110_000)
        meses = Incerteza(12, 24, 48)
        p, m = amostrar_mercado_correlacionado(preco, meses, rho=0.6, n=5_000, rng=rng)
        assert p.min() >= 90_000 and p.max() <= 110_000
        assert m.min() >= 12 and m.max() <= 48
