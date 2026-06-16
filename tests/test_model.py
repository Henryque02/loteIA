"""Testes do treino do modelo de preço/m² (split temporal, baseline, métricas)."""
import numpy as np
import pandas as pd
import pytest

from loteia.model.train import (
    BaselineMedianaSetor,
    avaliar,
    salvar_artefato,
    carregar_artefato,
    split_temporal,
    treinar_modelo,
)


def _df_sintetico(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Transações sintéticas: preço/m² determinado pelo setor + ruído leve."""
    rng = np.random.default_rng(seed)
    setores = rng.choice(["010", "085", "200"], size=n)
    base = pd.Series(setores).map({"010": 800.0, "085": 3000.0, "200": 1500.0})
    return pd.DataFrame({
        "setor": setores,
        "area_terreno_itbi": rng.uniform(100, 1000, size=n),
        "testada": rng.uniform(5, 30, size=n),
        "ano": rng.choice([2022, 2023, 2024], size=n),
        "preco_m2": base * rng.normal(1.0, 0.05, size=n),
    })


class TestSplitTemporal:
    def test_treino_ate_o_corte_teste_depois(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        assert (train["ano"] <= 2023).all()
        assert (test["ano"] > 2023).all()

    def test_nao_perde_linhas(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        assert len(train) + len(test) == len(df)

    def test_corte_padrao_treina_ate_2024_testa_2025(self):
        # walk-forward: ao incluir 2025, o corte global avança para 2024 —
        # treino = 2023+2024, teste = 2025. Nenhum ano é descartado.
        from loteia.config import ANO_CORTE_TREINO

        assert ANO_CORTE_TREINO == 2024
        df = pd.DataFrame({
            "setor": ["010"] * 6,
            "area_terreno_itbi": [300.0] * 6,
            "testada": [10.0] * 6,
            "ano": [2023, 2023, 2024, 2024, 2025, 2025],
            "preco_m2": [800.0] * 6,
        })
        train, test = split_temporal(df)  # usa o corte global
        assert set(train["ano"]) == {2023, 2024}
        assert set(test["ano"]) == {2025}


class TestBaselineMedianaSetor:
    def test_prediz_mediana_do_setor(self):
        X = pd.DataFrame({"setor": ["A", "A", "B", "B"]})
        y = pd.Series([100.0, 200.0, 1000.0, 2000.0])
        bl = BaselineMedianaSetor().fit(X, y)
        pred = bl.predict(pd.DataFrame({"setor": ["A", "B"]}))
        assert pred[0] == pytest.approx(150.0)
        assert pred[1] == pytest.approx(1500.0)

    def test_setor_desconhecido_usa_mediana_global(self):
        X = pd.DataFrame({"setor": ["A", "A", "B", "B"]})
        y = pd.Series([100.0, 200.0, 1000.0, 2000.0])
        bl = BaselineMedianaSetor().fit(X, y)
        pred = bl.predict(pd.DataFrame({"setor": ["Z"]}))
        assert pred[0] == pytest.approx(600.0)  # mediana global de y


class TestTreinarAvaliar:
    def test_modelo_supera_baseline_global(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        modelo = treinar_modelo(train)
        met = avaliar(modelo, test)
        # erro de prever sempre a mediana global do treino
        mae_global = (test["preco_m2"] - train["preco_m2"].median()).abs().mean()
        assert met["mae"] < mae_global

    def test_avaliar_retorna_metricas(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        modelo = treinar_modelo(train)
        met = avaliar(modelo, test)
        assert {"mae", "mape_mediana", "r2"} <= set(met)
        assert all(np.isfinite(v) for v in met.values())

    def test_reprodutivel_com_seed(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        m1 = treinar_modelo(train)
        m2 = treinar_modelo(train)
        assert np.allclose(m1.predict(test), m2.predict(test))

    def test_features_parametrizaveis(self):
        # feature extra que determina o preço → modelo com ela tem de ganhar
        df = _df_sintetico()
        rng = np.random.default_rng(1)
        df["fator_oculto"] = rng.uniform(0.5, 2.0, size=len(df))
        df["preco_m2"] = df["preco_m2"] * df["fator_oculto"]
        train, test = split_temporal(df, ano_corte=2023)

        base = treinar_modelo(train)
        extra = treinar_modelo(
            train,
            features_num=["area_terreno_itbi", "testada", "fator_oculto"],
        )
        met_base = avaliar(base, test)
        met_extra = avaliar(extra, test)
        assert met_extra["mae"] < met_base["mae"]

    def test_log_alvo_preve_em_escala_real_e_positiva(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        modelo = treinar_modelo(train, log_alvo=True)
        pred = modelo.predict(test)
        # predição volta à escala de R$/m² (positiva, ordem de grandeza do alvo)
        assert (pred > 0).all()
        assert 100 < np.median(pred) < 10_000

    def test_log_alvo_reprodutivel(self):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        m1 = treinar_modelo(train, log_alvo=True)
        m2 = treinar_modelo(train, log_alvo=True)
        assert np.allclose(m1.predict(test), m2.predict(test))


class TestArtefato:
    def test_roundtrip_joblib(self, tmp_path):
        df = _df_sintetico()
        train, test = split_temporal(df, ano_corte=2023)
        modelo = treinar_modelo(train)
        caminho = tmp_path / "preco_m2.joblib"
        salvar_artefato(caminho, modelo=modelo, metricas={"mae": 1.0})
        art = carregar_artefato(caminho)
        assert art["metricas"]["mae"] == 1.0
        assert np.allclose(art["modelo"].predict(test), modelo.predict(test))
