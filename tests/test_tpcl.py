"""Testes do join ITBI ⨝ cadastro TPCL ano-alinhado (evita viés de snapshot)."""
import pandas as pd
import pytest

from loteia.data.join import calcular_preco_m2, filtrar_terrenos
from loteia.data.tpcl import carregar_cadastro_tpcl, juntar_cadastro_ano_alinhado


def _cadastro(ano, linhas):
    return pd.DataFrame(linhas).assign(ano_cadastro=ano)


class TestCarregarCadastroTpcl:
    def test_le_csv_e_normaliza_sql(self, tmp_path):
        p = tmp_path / "tpcl_2019.csv"
        pd.DataFrame({
            "sql": [8500130026], "area_terreno": [500.0],
            "area_construida": [0.0], "uso_desc": ["TERRENO"],
        }).to_csv(p, index=False)
        cad = carregar_cadastro_tpcl(2019, caminho=p)
        assert cad["sql10"].iloc[0] == "0850013002"  # zfill(11)[:10]
        assert {"sql10", "area_terreno", "area_construida", "uso_desc"} <= set(cad)

    def test_sem_caminho_e_sem_cache_orienta_a_fonte(self):
        # sem arquivo local, falha com instrução clara (não inventa dado)
        with pytest.raises(FileNotFoundError, match="TPCL"):
            carregar_cadastro_tpcl(1990, caminho=None)


class TestJoinAnoAlinhado:
    def _itbi(self):
        # mesma SQL vendida em 2019 e 2024
        return pd.DataFrame({
            "sql10": ["0850013002", "0850013002", "0850013003"],
            "ano": [2019, 2024, 2019],
            "valor_transacao": [500_000.0, 900_000.0, 100_000.0],
            "natureza": ["1.x", "1.x", "1.x"],
            "proporcao_transmitida": [100.0, 100.0, 100.0],
            "fracao_ideal": [1.0, 1.0, 1.0],
        })

    def _cadastros(self):
        # em 2019 o lote 002 era TERRENO vazio; em 2024 já está construído
        return {
            2019: pd.DataFrame({
                "sql10": ["0850013002", "0850013003"],
                "area_terreno": [500.0, 200.0],
                "area_construida": [0.0, 0.0],
                "uso_desc": ["TERRENO", "TERRENO"],
            }),
            2024: pd.DataFrame({
                "sql10": ["0850013002"],
                "area_terreno": [500.0],
                "area_construida": [180.0],
                "uso_desc": ["RESIDÊNCIA"],
            }),
        }

    def test_usa_cadastro_do_ano_da_transacao(self):
        out = juntar_cadastro_ano_alinhado(self._itbi(), self._cadastros())
        # a venda de 2019 enxerga o cadastro de 2019 (terreno vazio)
        v2019 = out[(out["sql10"] == "0850013002") & (out["ano"] == 2019)].iloc[0]
        assert v2019["uso_desc"] == "TERRENO"
        assert v2019["area_construida_itbi"] == 0.0
        # a venda de 2024 enxerga o cadastro de 2024 (construído)
        v2024 = out[(out["sql10"] == "0850013002") & (out["ano"] == 2024)].iloc[0]
        assert v2024["uso_desc"] == "RESIDÊNCIA"

    def test_corrige_vies_de_snapshot(self):
        # ano-alinhado: a venda de terreno de 2019 SOBREVIVE ao filtro;
        # com o snapshot atual (2024) ela seria descartada (vira construído).
        out = juntar_cadastro_ano_alinhado(self._itbi(), self._cadastros())
        terrenos = filtrar_terrenos(out)
        assert ((terrenos["sql10"] == "0850013002") & (terrenos["ano"] == 2019)).any()
        assert not ((terrenos["sql10"] == "0850013002") & (terrenos["ano"] == 2024)).any()

    def test_preco_usa_area_do_ano(self):
        out = juntar_cadastro_ano_alinhado(self._itbi(), self._cadastros())
        preco = calcular_preco_m2(filtrar_terrenos(out))
        v = preco[(preco["sql10"] == "0850013002") & (preco["ano"] == 2019)].iloc[0]
        assert v["preco_m2"] == pytest.approx(500_000.0 / 500.0)

    def test_transacao_sem_cadastro_do_ano_e_descartada(self):
        itbi = self._itbi()
        cadastros = {2024: self._cadastros()[2024]}  # falta 2019
        out = juntar_cadastro_ano_alinhado(itbi, cadastros)
        # vendas de 2019 ficam sem área (NaN) → caem no filtro de terreno/preço
        assert out[out["ano"] == 2019]["area_terreno_itbi"].isna().all()
