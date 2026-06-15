"""Testes da Camada 3 (financeiro): análise de sensibilidade tornado."""
import pytest

from loteia.finance.cashflow import Custos
from loteia.finance.montecarlo import Incerteza
from loteia.finance.sensitivity import sensibilidade_tornado


def _tornado(**kw):
    base = dict(
        n_lotes=50,
        custo_gleba=2_000_000.0,
        meses_obra=12,
        preco_lote=Incerteza(90_000, 100_000, 110_000),
        custo_infra=Incerteza(1_500_000, 2_000_000, 2_500_000),
        meses_vendas=Incerteza(12, 24, 48),
        taxa_alvo_anual=0.15,
    )
    base.update(kw)
    return sensibilidade_tornado(**base)


class TestSensibilidadeTornado:
    def test_uma_linha_por_input_incerto(self):
        t = _tornado()
        assert set(t["variavel"]) == {"preco_lote", "custo_infra", "meses_vendas"}

    def test_amplitude_nao_negativa_e_ordenada(self):
        t = _tornado()
        assert (t["amplitude"] >= 0).all()
        assert t["amplitude"].is_monotonic_decreasing

    def test_input_sem_incerteza_tem_amplitude_zero(self):
        t = _tornado(custo_infra=Incerteza(2_000_000, 2_000_000, 2_000_000))
        linha = t[t["variavel"] == "custo_infra"].iloc[0]
        assert linha["amplitude"] == pytest.approx(0.0)

    def test_incerteza_dominante_fica_em_primeiro(self):
        # faixa de preço enorme → preco_lote deve liderar o tornado
        t = _tornado(preco_lote=Incerteza(10_000, 100_000, 500_000))
        assert t.iloc[0]["variavel"] == "preco_lote"

    def test_preco_maior_aumenta_vpl(self):
        t = _tornado()
        linha = t[t["variavel"] == "preco_lote"].iloc[0]
        assert linha["vpl_no_maximo"] > linha["vpl_no_minimo"]

    def test_custo_maior_reduz_vpl(self):
        t = _tornado()
        linha = t[t["variavel"] == "custo_infra"].iloc[0]
        assert linha["vpl_no_maximo"] < linha["vpl_no_minimo"]

    def test_custos_operacionais_baixam_o_tornado(self):
        # com custos operacionais, todo o nível de VPL do tornado cai
        sem = _tornado()
        com = _tornado(custos=Custos(comissao_pct=0.06, admin_mensal=20_000.0))
        pico_sem = sem["vpl_no_maximo"].max()
        pico_com = com["vpl_no_maximo"].max()
        assert pico_com < pico_sem
