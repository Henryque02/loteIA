"""Testes do motor financeiro (VPL/TIR em casos conhecidos)."""
import numpy as np
import pytest

from loteia.finance.cashflow import fluxo_loteamento, tir, tir_anual, vpl


class TestVpl:
    def test_taxa_zero_e_a_soma(self):
        assert vpl([-100.0, 60.0, 60.0], 0.0) == pytest.approx(20.0)

    def test_caso_conhecido_vpl_nulo(self):
        # investe 100, recebe 110 em 1 período, taxa 10% → VPL = 0
        assert vpl([-100.0, 110.0], 0.10) == pytest.approx(0.0)

    def test_desconto_de_dois_periodos(self):
        # 121 daqui a 2 períodos a 10% vale 100 hoje
        assert vpl([-100.0, 0.0, 121.0], 0.10) == pytest.approx(0.0)


class TestTir:
    def test_caso_conhecido_um_periodo(self):
        assert tir([-100.0, 110.0]) == pytest.approx(0.10)

    def test_caso_conhecido_dois_periodos(self):
        assert tir([-100.0, 0.0, 121.0]) == pytest.approx(0.10)

    def test_sem_inversao_de_sinal_e_erro(self):
        with pytest.raises(ValueError):
            tir([-100.0, -50.0])

    def test_tir_anual_compoe_12_meses(self):
        # 1% a.m. ≈ 12,68% a.a.
        assert tir_anual(0.01) == pytest.approx(0.1268, abs=1e-3)


class TestFluxoLoteamento:
    def _fluxo(self, **kw):
        base = dict(
            n_lotes=10, preco_lote=100_000.0, custo_gleba=300_000.0,
            custo_infra=200_000.0, meses_obra=4, meses_vendas=10, n_parcelas=1,
        )
        base.update(kw)
        return fluxo_loteamento(**base)

    def test_conservacao_do_total(self):
        f = self._fluxo()
        assert f.sum() == pytest.approx(10 * 100_000 - 300_000 - 200_000)

    def test_gleba_paga_no_mes_zero(self):
        f = self._fluxo()
        assert f[0] == pytest.approx(-300_000.0)

    def test_infra_uniforme_durante_a_obra(self):
        # sem receita, meses 1..4 carregam só a infra
        f = self._fluxo(preco_lote=0.0)
        assert np.allclose(f[1:5], -50_000.0)
        assert np.allclose(f[5:], 0.0)

    def test_vendas_uniformes_a_vista(self):
        # sem custos de obra, meses 1..10 recebem 1 lote cada
        f = self._fluxo(custo_infra=0.0)
        assert np.allclose(f[1:11], 100_000.0)

    def test_parcelamento_estende_o_fluxo(self):
        # última venda no mês 10, em 12 parcelas → último recebimento no mês 21
        f = self._fluxo(n_parcelas=12)
        assert len(f) == 22
        assert f[21] != 0.0
        assert f.sum() == pytest.approx(10 * 100_000 - 300_000 - 200_000)

    def test_projeto_lucrativo_tem_tir_positiva(self):
        assert tir(self._fluxo()) > 0.0


class TestCustosOperacionais:
    """Custos novos (comissão/impostos/marketing %VGV, admin mensal, licenciamento)."""

    def _fluxo(self, **kw):
        base = dict(
            n_lotes=10, preco_lote=100_000.0, custo_gleba=300_000.0,
            custo_infra=200_000.0, meses_obra=4, meses_vendas=10, n_parcelas=1,
        )
        base.update(kw)
        return fluxo_loteamento(**base)

    def test_default_sem_custos_novos_inalterado(self):
        # sem os novos parâmetros, o fluxo é idêntico ao motor antigo
        f = self._fluxo()
        assert f.sum() == pytest.approx(10 * 100_000 - 300_000 - 200_000)

    def test_comissao_reduz_total_por_pct_do_vgv(self):
        vgv = 10 * 100_000
        f = self._fluxo(comissao_pct=0.06)
        antes = 10 * 100_000 - 300_000 - 200_000
        assert f.sum() == pytest.approx(antes - 0.06 * vgv)

    def test_comissao_escala_com_vgv(self):
        base = self._fluxo().sum()
        com = self._fluxo(comissao_pct=0.06).sum()
        com_dobro = self._fluxo(preco_lote=200_000.0, comissao_pct=0.06).sum()
        # ao dobrar o VGV, o desconto de comissão dobra
        assert (base - com) == pytest.approx(0.06 * 10 * 100_000)
        assert (self._fluxo(preco_lote=200_000.0).sum() - com_dobro) == pytest.approx(
            0.06 * 10 * 200_000
        )

    def test_impostos_e_marketing_tambem_descontam_vgv(self):
        vgv = 10 * 100_000
        f = self._fluxo(impostos_pct=0.02, marketing_pct=0.03)
        antes = 10 * 100_000 - 300_000 - 200_000
        assert f.sum() == pytest.approx(antes - 0.05 * vgv)

    def test_admin_mensal_debita_todo_mes(self):
        f0 = self._fluxo()
        f = self._fluxo(admin_mensal=1_000.0)
        assert f.sum() == pytest.approx(f0.sum() - 1_000.0 * len(f))

    def test_licenciamento_upfront_no_mes_zero(self):
        f = self._fluxo(licenciamento=80_000.0)
        # mês 0 carrega gleba + licenciamento
        assert f[0] == pytest.approx(-(300_000.0 + 80_000.0))

    def test_mes_inicio_vendas_atrasa_receita(self):
        # vendas só começam no mês 7 → meses 1..6 sem receita de venda
        f = self._fluxo(custo_infra=0.0, mes_inicio_vendas=7)
        assert np.allclose(f[1:7], 0.0)
        assert f[7] == pytest.approx(100_000.0)

    def test_cada_custo_reduz_vpl(self):
        base = vpl(self._fluxo(), 0.01)
        for kw in (
            {"comissao_pct": 0.06},
            {"impostos_pct": 0.02},
            {"marketing_pct": 0.03},
            {"admin_mensal": 2_000.0},
            {"licenciamento": 100_000.0},
        ):
            assert vpl(self._fluxo(**kw), 0.01) < base

    def test_conservacao_com_todos_os_custos(self):
        vgv = 10 * 100_000
        f = self._fluxo(
            comissao_pct=0.06, impostos_pct=0.02, marketing_pct=0.03,
            admin_mensal=1_000.0, licenciamento=80_000.0,
        )
        esperado = (
            vgv - 300_000 - 200_000 - 0.11 * vgv - 1_000.0 * len(f) - 80_000.0
        )
        assert f.sum() == pytest.approx(esperado)
