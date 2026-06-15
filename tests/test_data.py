"""Testes do pipeline de dados (join por SQL, filtro de terrenos)."""
import pandas as pd
import pytest

from loteia.data.join import calcular_preco_m2, filtrar_terrenos, norm_sql10


class TestNormSql10:
    def test_zfill_e_descarta_verificador(self):
        # 9 dígitos → zfill(11) → [:10] descarta o 11º
        s = pd.Series([100800297.0])
        assert norm_sql10(s)[0] == "0010080029"

    def test_11_digitos_descarta_verificador(self):
        s = pd.Series([13944000322.0])
        assert norm_sql10(s)[0] == "1394400032"

    def test_nulo_vira_na(self):
        s = pd.Series([float("nan"), None])
        r = norm_sql10(s)
        assert pd.isna(r[0])
        assert pd.isna(r[1])

    def test_setor_preservado(self):
        # os 3 primeiros dígitos do sql10 devem ser o setor fiscal
        s = pd.Series([8500130026.0])  # setor 085
        assert norm_sql10(s)[0][:3] == "085"


class TestFiltrarTerrenos:
    def _df(self):
        return pd.DataFrame({
            "uso_desc": ["TERRENO", "TERRENO", "RESIDÊNCIA", "TERRENO"],
            "area_construida_itbi": [0.0, 50.0, 0.0, None],
        })

    def test_exclui_construido(self):
        out = filtrar_terrenos(self._df())
        assert len(out) == 2  # linhas 0 (ac=0) e 3 (ac=None→0)

    def test_exclui_uso_diferente(self):
        out = filtrar_terrenos(self._df())
        assert set(out["uso_desc"]) == {"TERRENO"}


class TestCalcularPrecoM2:
    def _terreno(self, **overrides):
        base = {
            "natureza": "1.Compra e venda",
            "proporcao_transmitida": 100.0,
            "fracao_ideal": 1.0,
            "valor_transacao": 500_000.0,
            "area_terreno_itbi": 500.0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_calcula_preco(self):
        out = calcular_preco_m2(self._terreno())
        assert len(out) == 1
        assert out["preco_m2"].iloc[0] == pytest.approx(1000.0)

    def test_exclui_natureza_nao_onerosa(self):
        out = calcular_preco_m2(self._terreno(natureza="2.Cessão de direitos"))
        assert len(out) == 0

    def test_exclui_proporcao_parcial(self):
        out = calcular_preco_m2(self._terreno(proporcao_transmitida=50.0))
        assert len(out) == 0

    def test_exclui_fracao_ideal_menor(self):
        out = calcular_preco_m2(self._terreno(fracao_ideal=0.5))
        assert len(out) == 0

    def test_exclui_valor_infimo(self):
        out = calcular_preco_m2(self._terreno(valor_transacao=1.0))
        assert len(out) == 0
