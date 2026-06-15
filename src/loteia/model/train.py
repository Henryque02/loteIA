"""Treina o modelo de preco/m2 de terreno.
REGRA: split por TEMPO (treina <= ANO_CORTE_TREINO ; testa depois). Sem shuffle."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from loteia.config import ANO_CORTE_TREINO, MODELS, SEED

FEATURES_NUM: list[str] = ["area_terreno_itbi", "testada"]
FEATURES_CAT: list[str] = ["setor"]
FEATURES: list[str] = FEATURES_NUM + FEATURES_CAT
ALVO = "preco_m2"


def split_temporal(
    df: pd.DataFrame, ano_corte: int = ANO_CORTE_TREINO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em (treino, teste) pelo ano da transação. Nunca embaralhar no tempo."""
    train = df[df["ano"] <= ano_corte].copy()
    test = df[df["ano"] > ano_corte].copy()
    return train, test


class BaselineMedianaSetor:
    """Baseline: mediana do preço/m² por setor fiscal; setor não visto usa a global."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineMedianaSetor":
        self.mediana_global_ = float(y.median())
        self.medianas_ = y.groupby(X["setor"].values).median().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X["setor"].map(self.medianas_).fillna(self.mediana_global_).to_numpy()
        )


def montar_preprocessador(
    features_num: list[str] | None = None,
    features_cat: list[str] | None = None,
) -> ColumnTransformer:
    """Numéricas passam direto; categóricas viram ordinal (desconhecido → -1)."""
    return ColumnTransformer(
        [
            ("num", "passthrough", features_num or FEATURES_NUM),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                features_cat or FEATURES_CAT,
            ),
        ]
    )


def treinar_modelo(
    train_df: pd.DataFrame,
    features_num: list[str] | None = None,
    features_cat: list[str] | None = None,
    log_alvo: bool = False,
) -> Pipeline:
    """Treina o gradient boosting sobre as features dadas, com seed fixa.

    `log_alvo=True` treina em log1p(preço/m²) e devolve expm1 — o alvo é
    log-normal (cauda pesada); `.predict` continua em escala real de R$/m².
    """
    features_num = features_num or FEATURES_NUM
    features_cat = features_cat or FEATURES_CAT
    pipe = Pipeline(
        [
            ("prep", montar_preprocessador(features_num, features_cat)),
            (
                "gbr",
                HistGradientBoostingRegressor(
                    loss="absolute_error", random_state=SEED
                ),
            ),
        ]
    )
    modelo = (
        TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        if log_alvo
        else pipe
    )
    modelo.features_ = features_num + features_cat
    modelo.fit(train_df[modelo.features_], train_df[ALVO])
    return modelo


def avaliar(modelo, df: pd.DataFrame) -> dict[str, float]:
    """MAE, MAPE mediano e R² no conjunto dado."""
    features = getattr(modelo, "features_", FEATURES)
    y = df[ALVO]
    pred = modelo.predict(df[features])
    ape = np.abs((y - pred) / y)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "mape_mediana": float(np.median(ape)),
        "r2": float(r2_score(y, pred)),
    }


def salvar_artefato(caminho: Path, **artefato) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, caminho)


def carregar_artefato(caminho: Path) -> dict:
    return joblib.load(caminho)


if __name__ == "__main__":
    from loteia.model.experimento_geo import (
        FEATURES_FINAIS_CAT,
        FEATURES_FINAIS_NUM,
        montar_dataset_enriquecido,
    )

    # rodando como script, a classe local seria picklada como __main__.*
    # e o artefato não carregaria fora daqui; usa a referência canônica do módulo
    from loteia.model.train import BaselineMedianaSetor

    df = montar_dataset_enriquecido()
    train, test = split_temporal(df)
    print(f"treino (<= {ANO_CORTE_TREINO}): {len(train):,}  |  teste: {len(test):,}")

    # log_alvo=False: o experimento mostrou que, com geo+zona+coords, o modelo
    # de loss absoluta em escala real supera o log-alvo (MAE/MAPE/R²).
    modelo = treinar_modelo(
        train,
        features_num=FEATURES_FINAIS_NUM,
        features_cat=FEATURES_FINAIS_CAT,
        log_alvo=False,
    )
    baseline = BaselineMedianaSetor().fit(train[FEATURES], train[ALVO])
    baseline.features_ = FEATURES
    met_modelo = avaliar(modelo, test)
    met_baseline = avaliar(baseline, test)
    print(f"modelo (geo+zona): {met_modelo}")
    print(f"baseline         : {met_baseline}")

    salvar_artefato(
        MODELS / "preco_m2.joblib",
        modelo=modelo,
        baseline=baseline,
        metricas={"modelo": met_modelo, "baseline": met_baseline},
        features=modelo.features_,
        ano_corte=ANO_CORTE_TREINO,
    )
    print(f"artefato salvo em {MODELS / 'preco_m2.joblib'}")
