"""Testes da Camada 3 (preço): SHAP global e local sobre o modelo de preço."""
import pytest

from loteia.model.explain import explicar_global, explicar_local
from loteia.model.train import FEATURES, split_temporal, treinar_modelo
from tests.test_model import _df_sintetico


@pytest.fixture(scope="module")
def contexto():
    train, test = split_temporal(_df_sintetico(n=600, seed=2), ano_corte=2023)
    return treinar_modelo(train), train, test


class TestExplicarGlobal:
    def test_uma_importancia_por_feature(self, contexto):
        modelo, train, _ = contexto
        imp = explicar_global(modelo, train)
        assert set(imp.index) == set(FEATURES)

    def test_importancias_nao_negativas_e_ordenadas(self, contexto):
        modelo, train, _ = contexto
        imp = explicar_global(modelo, train)
        assert (imp >= 0).all()
        assert imp.is_monotonic_decreasing

    def test_setor_domina_no_dado_sintetico(self, contexto):
        # no sintético o preço é função do setor → deve ser a feature nº 1
        modelo, train, _ = contexto
        imp = explicar_global(modelo, train)
        assert imp.index[0] == "setor"


class TestExplicarLocal:
    def test_um_valor_por_feature(self, contexto):
        modelo, _, test = contexto
        shap_vals, _ = explicar_local(modelo, test.head(1))
        assert set(shap_vals.index) == set(FEATURES)

    def test_aditividade_base_mais_shap_e_a_predicao(self, contexto):
        # propriedade SHAP: base + soma das contribuições = predição do modelo
        modelo, _, test = contexto
        linha = test.head(1)
        shap_vals, base = explicar_local(modelo, linha)
        pred = modelo.predict(linha[FEATURES])[0]
        assert base + shap_vals.sum() == pytest.approx(pred, rel=1e-6)
