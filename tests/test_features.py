"""Testes da Camada 5: geocodificação por quadra e features geoespaciais."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from loteia.data.features import (
    contar_no_raio,
    dist_mais_proximo,
    geocodificar,
    montar_centroides,
    montar_features_geo,
    renda_por_ponto,
    zona_por_ponto,
)


def _quadrado(x0: float, y0: float, lado: float = 10.0) -> Polygon:
    return Polygon([(x0, y0), (x0 + lado, y0), (x0 + lado, y0 + lado), (x0, y0 + lado)])


@pytest.fixture
def gdf_quadras():
    return gpd.GeoDataFrame(
        {
            "cd_setor_fiscal": ["085", "085"],
            "cd_quadra_fiscal": ["013", "014"],
            "geometry": [_quadrado(0, 0), _quadrado(100, 100)],
        },
        crs="EPSG:31983",
    )


@pytest.fixture
def centroides(gdf_quadras):
    return montar_centroides(gdf_quadras)


class TestMontarCentroides:
    def test_centroide_do_quadrado(self, centroides):
        linha = centroides[centroides["quadra"] == "013"].iloc[0]
        assert linha["x"] == pytest.approx(5.0)
        assert linha["y"] == pytest.approx(5.0)

    def test_chaves_setor_quadra(self, centroides):
        assert set(centroides.columns) >= {"setor", "quadra", "x", "y"}


class TestGeocodificar:
    def test_anexa_xy_pelo_setor_quadra(self, centroides):
        df = pd.DataFrame({"sql10": ["0850130026", "0850140001"]})
        out = geocodificar(df, centroides)
        assert out["x"].tolist() == pytest.approx([5.0, 105.0])

    def test_quadra_ausente_vira_nan(self, centroides):
        df = pd.DataFrame({"sql10": ["0999990001"]})
        out = geocodificar(df, centroides)
        assert np.isnan(out["x"].iloc[0])


class TestDistancias:
    def test_dist_mais_proximo(self):
        pontos = np.array([[0.0, 0.0], [10.0, 0.0]])
        alvos = np.array([[0.0, 3.0], [13.0, 0.0]])
        d = dist_mais_proximo(pontos, alvos)
        assert d == pytest.approx([3.0, 3.0])

    def test_contar_no_raio(self):
        pontos = np.array([[0.0, 0.0]])
        alvos = np.array([[1.0, 0.0], [0.0, 2.0], [50.0, 0.0]])
        assert contar_no_raio(pontos, alvos, raio=5.0).tolist() == [2]


class TestRendaPorPonto:
    @pytest.fixture
    def gdf_renda(self):
        return gpd.GeoDataFrame(
            {"renda": [1000.0, 5000.0],
             "geometry": [_quadrado(0, 0), _quadrado(100, 100)]},
            crs="EPSG:31983",
        )

    def test_ponto_dentro_do_setor(self, gdf_renda):
        xy = np.array([[5.0, 5.0], [105.0, 105.0]])
        assert renda_por_ponto(xy, gdf_renda).tolist() == pytest.approx([1000.0, 5000.0])

    def test_ponto_fora_vira_nan(self, gdf_renda):
        xy = np.array([[500.0, 500.0]])
        assert np.isnan(renda_por_ponto(xy, gdf_renda)[0])


class TestZonaPorPonto:
    @pytest.fixture
    def gdf_zona(self):
        return gpd.GeoDataFrame(
            {"zona": ["ZM", "ZER"],
             "geometry": [_quadrado(0, 0), _quadrado(100, 100)]},
            crs="EPSG:31983",
        )

    def test_ponto_recebe_sigla_da_zona(self, gdf_zona):
        xy = np.array([[5.0, 5.0], [105.0, 105.0]])
        assert zona_por_ponto(xy, gdf_zona).tolist() == ["ZM", "ZER"]

    def test_ponto_fora_vira_sem_zona(self, gdf_zona):
        xy = np.array([[500.0, 500.0]])
        # categórica: fora de qualquer polígono → sentinela "SEM_ZONA" (não NaN)
        assert zona_por_ponto(xy, gdf_zona).tolist() == ["SEM_ZONA"]


class TestMontarFeaturesGeo:
    def test_anexa_todas_as_colunas(self, centroides):
        df = pd.DataFrame({"sql10": ["0850130026", "0999990001"]})
        gdf_renda = gpd.GeoDataFrame(
            {"renda": [2000.0], "geometry": [_quadrado(0, 0)]}, crs="EPSG:31983"
        )
        gdf_zona = gpd.GeoDataFrame(
            {"zona": ["ZM"], "geometry": [_quadrado(0, 0)]}, crs="EPSG:31983"
        )
        out = montar_features_geo(
            df,
            centroides=centroides,
            estacoes=np.array([[5.0, 8.0]]),
            comercios=np.array([[6.0, 5.0], [5000.0, 5000.0]]),
            escolas=np.array([[5.0, 5.0]]),
            gdf_renda=gdf_renda,
            gdf_zona=gdf_zona,
            marco_zero=(0.0, 5.0),
        )
        esperadas = {
            "dist_centro", "dist_estacao", "n_comercio_1km", "n_escola_1km",
            "renda_setor", "zona", "x", "y",
        }
        assert esperadas <= set(out.columns)
        linha = out.iloc[0]
        assert linha["dist_centro"] == pytest.approx(5.0)
        assert linha["dist_estacao"] == pytest.approx(3.0)
        assert linha["n_comercio_1km"] == 1
        assert linha["renda_setor"] == pytest.approx(2000.0)
        assert linha["zona"] == "ZM"
        # linha sem geocodificação propaga NaN/sem-zona sem quebrar
        assert np.isnan(out.iloc[1]["dist_centro"])
        assert out.iloc[1]["zona"] == "SEM_ZONA"
