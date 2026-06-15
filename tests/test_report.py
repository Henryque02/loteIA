"""Testes do modo relatório: reconstrução do payload a partir da URL (query params)."""
import pytest

from loteia.report import (
    payload_de_params,
    resumo_respostas,
    tem_params_relatorio,
)


def _params_completos(**over):
    base = {
        "empreendimento": "Jardim X",
        "setor": "085", "quadra": "013", "area_lote_m2": "300", "testada": "17",
        "n_lotes": "80", "custo_gleba": "25000000",
        "infra_min": "16000000", "infra_moda": "20000000", "infra_max": "28000000",
        "meses_obra": "18",
        "vendas_min": "12", "vendas_moda": "24", "vendas_max": "48",
        "n_parcelas": "12", "taxa_alvo_anual": "0.15", "mes_inicio_vendas": "6",
        "comissao_pct": "0.06", "impostos_pct": "0.04", "marketing_pct": "0.03",
        "admin_mensal": "20000", "licenciamento": "300000",
        "rho_mercado": "0.5",
    }
    base.update(over)
    return base


class TestTemParamsRelatorio:
    def test_detecta_quando_completo(self):
        assert tem_params_relatorio(_params_completos())

    def test_falso_sem_setor_ou_lotes(self):
        assert not tem_params_relatorio({})
        assert not tem_params_relatorio({"setor": "085"})  # falta n_lotes


class TestPayloadDeParams:
    def test_monta_terreno_e_projeto(self):
        p = payload_de_params(_params_completos())
        assert p["terreno"] == {
            "setor": "085", "quadra": "013", "area_lote_m2": 300.0, "testada": 17.0,
        }
        assert p["projeto"]["n_lotes"] == 80
        assert p["projeto"]["custo_gleba"] == 25_000_000.0

    def test_reconstroi_faixas_triangulares(self):
        p = payload_de_params(_params_completos())
        assert p["projeto"]["custo_infra"] == {
            "minimo": 16_000_000.0, "moda": 20_000_000.0, "maximo": 28_000_000.0,
        }
        assert p["projeto"]["meses_vendas"] == {
            "minimo": 12.0, "moda": 24.0, "maximo": 48.0,
        }

    def test_custos_e_rho(self):
        p = payload_de_params(_params_completos())
        assert p["projeto"]["custos"]["comissao_pct"] == 0.06
        assert p["rho_mercado"] == 0.5
        assert p["incluir_distribuicao"] is True

    def test_tipos_inteiros(self):
        p = payload_de_params(_params_completos())
        for k in ("n_lotes", "meses_obra", "n_parcelas", "mes_inicio_vendas"):
            assert isinstance(p["projeto"][k], int)

    def test_opcionais_ausentes_usam_default(self):
        # sem custos/rho/n_parcelas → defaults seguros
        minimo = {
            "setor": "085", "quadra": "013", "area_lote_m2": "300",
            "n_lotes": "80", "custo_gleba": "25000000",
            "infra_min": "16000000", "infra_moda": "20000000", "infra_max": "28000000",
            "meses_obra": "18",
            "vendas_min": "12", "vendas_moda": "24", "vendas_max": "48",
            "taxa_alvo_anual": "0.15",
        }
        p = payload_de_params(minimo)
        assert p["projeto"]["n_parcelas"] == 1
        assert p["projeto"]["custos"]["comissao_pct"] == 0.0
        assert p["rho_mercado"] == pytest.approx(0.5)
        assert p["terreno"]["testada"] == 10.0  # default


class TestResumoRespostas:
    def test_devolve_pares_rotulo_valor_formatados(self):
        itens = dict(resumo_respostas(_params_completos()))
        assert itens["Empreendimento"] == "Jardim X"
        assert itens["Localização"] == "Setor 085 · Quadra 013"
        assert itens["Área do lote"] == "300 m²"
        assert itens["Nº de lotes"] == "80"
        assert itens["Custo da gleba"] == "R$ 25 mi"
        assert itens["Taxa-alvo (a.a.)"] == "15%"

    def test_formata_milhoes_com_decimal(self):
        itens = dict(resumo_respostas(_params_completos(custo_gleba="1500000")))
        assert itens["Custo da gleba"] == "R$ 1,5 mi"

    def test_sem_empreendimento_omite_a_linha(self):
        p = _params_completos()
        del p["empreendimento"]
        assert "Empreendimento" not in dict(resumo_respostas(p))
