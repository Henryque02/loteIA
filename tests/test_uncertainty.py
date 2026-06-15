"""Testes da Camada 1: intervalo de preço com cobertura garantida (CQR/MAPIE)."""
import numpy as np
import pytest

from loteia.model.train import carregar_artefato, salvar_artefato, split_temporal
from loteia.model.uncertainty import ModeloQuantilico, cobertura_empirica
from tests.test_model import _df_sintetico


@pytest.fixture(scope="module")
def conjuntos():
    df = _df_sintetico(n=800, seed=1)
    return split_temporal(df, ano_corte=2023)


@pytest.fixture(scope="module")
def modelo(conjuntos):
    train, _ = conjuntos
    return ModeloQuantilico(cobertura=0.8).fit(train)


class TestModeloQuantilico:
    def test_intervalo_ordenado(self, modelo, conjuntos):
        _, test = conjuntos
        pred = modelo.predict_intervalo(test)
        assert (pred["preco_m2_inf"] <= pred["preco_m2_med"]).all()
        assert (pred["preco_m2_med"] <= pred["preco_m2_sup"]).all()

    def test_cobertura_proxima_da_nominal(self, modelo, conjuntos):
        _, test = conjuntos
        cob = cobertura_empirica(modelo, test)
        assert cob >= 0.70  # nominal 0.80, margem para variação amostral

    def test_log_alvo_intervalo_ordenado_positivo_e_coberto(self, conjuntos):
        train, test = conjuntos
        m = ModeloQuantilico(cobertura=0.8, log_alvo=True).fit(train)
        pred = m.predict_intervalo(test)
        assert (pred["preco_m2_inf"] > 0).all()  # back-transform expm1 > 0
        assert (pred["preco_m2_inf"] <= pred["preco_m2_med"]).all()
        assert (pred["preco_m2_med"] <= pred["preco_m2_sup"]).all()
        assert cobertura_empirica(m, test) >= 0.70

    def test_intervalo_nao_degenerado(self, modelo, conjuntos):
        # dados têm ruído → o intervalo precisa ter largura real
        _, test = conjuntos
        pred = modelo.predict_intervalo(test)
        largura = pred["preco_m2_sup"] - pred["preco_m2_inf"]
        assert (largura > 0).all()

    def test_setor_desconhecido_nao_quebra(self, modelo, conjuntos):
        _, test = conjuntos
        novo = test.head(1).copy()
        novo["setor"] = "999"
        pred = modelo.predict_intervalo(novo)
        assert np.isfinite(pred.to_numpy()).all()

    def test_reprodutivel(self, conjuntos):
        train, test = conjuntos
        p1 = ModeloQuantilico(cobertura=0.8).fit(train).predict_intervalo(test)
        p2 = ModeloQuantilico(cobertura=0.8).fit(train).predict_intervalo(test)
        assert np.allclose(p1.to_numpy(), p2.to_numpy())

    def test_features_parametrizaveis(self, conjuntos):
        # alvo multiplicado por um fator observável → modelo que o vê
        # produz intervalos mais estreitos do que o que não vê
        import numpy as np

        train, test = conjuntos
        rng = np.random.default_rng(3)
        train = train.copy()
        test = test.copy()
        train["fator"] = rng.uniform(0.5, 2.0, len(train))
        test["fator"] = rng.uniform(0.5, 2.0, len(test))
        train["preco_m2"] = train["preco_m2"] * train["fator"]
        test["preco_m2"] = test["preco_m2"] * test["fator"]

        sem = ModeloQuantilico(0.8).fit(train)
        com = ModeloQuantilico(
            0.8, features_num=["area_terreno_itbi", "testada", "fator"]
        ).fit(train)
        largura = lambda m: (  # noqa: E731
            m.predict_intervalo(test)["preco_m2_sup"]
            - m.predict_intervalo(test)["preco_m2_inf"]
        ).median()
        assert largura(com) < largura(sem)

    def test_roundtrip_joblib(self, modelo, conjuntos, tmp_path):
        _, test = conjuntos
        caminho = tmp_path / "quantilico.joblib"
        salvar_artefato(caminho, modelo_quantilico=modelo)
        art = carregar_artefato(caminho)
        p1 = art["modelo_quantilico"].predict_intervalo(test)
        p2 = modelo.predict_intervalo(test)
        assert np.allclose(p1.to_numpy(), p2.to_numpy())
