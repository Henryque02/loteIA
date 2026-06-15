"""Camada 3 - explicabilidade: SHAP global e local sobre o modelo de preco."""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from loteia.model.train import FEATURES


def _shap_values(
    modelo: Pipeline, df: pd.DataFrame
) -> tuple[np.ndarray, float, list[str]]:
    """Valores SHAP (linhas × features, na ordem das features do modelo) e base.

    Modelo com log-alvo (TransformedTargetRegressor): as contribuições saem na
    escala do alvo de treino (log-R$/m²); o ranking de importância é o mesmo.
    """
    features = getattr(modelo, "features_", FEATURES)
    pipe = getattr(modelo, "regressor_", modelo)  # desembrulha TTR se houver
    Xt = pipe.named_steps["prep"].transform(df[features])
    explicador = shap.TreeExplainer(pipe.named_steps["gbr"])
    valores = explicador.shap_values(Xt)
    base = float(np.asarray(explicador.expected_value).ravel()[0])
    return valores, base, features


def explicar_global(modelo: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Importância global: média de |SHAP| por feature, em ordem decrescente."""
    valores, _, features = _shap_values(modelo, df)
    imp = pd.Series(np.abs(valores).mean(axis=0), index=features)
    return imp.sort_values(ascending=False)


def explicar_local(modelo: Pipeline, linha: pd.DataFrame) -> tuple[pd.Series, float]:
    """Contribuição de cada feature para UMA predição: (shap_por_feature, base).

    Vale a aditividade: base + soma das contribuições = predição.
    """
    valores, base, features = _shap_values(modelo, linha)
    return pd.Series(valores[0], index=features), base


if __name__ == "__main__":
    from loteia.config import MODELS
    from loteia.data.download import ler_ano
    from loteia.data.join import calcular_preco_m2, filtrar_terrenos
    from loteia.model.train import carregar_artefato, split_temporal

    df = pd.concat([ler_ano(a) for a in (2023, 2024)], ignore_index=True)
    df = calcular_preco_m2(filtrar_terrenos(df))
    train, _ = split_temporal(df)

    modelo = carregar_artefato(MODELS / "preco_m2.joblib")["modelo"]
    print("importância global (média |SHAP|, R$/m²):")
    print(explicar_global(modelo, train).round(1))
