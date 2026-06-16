"""Camada 1 - incerteza no preco: regressao quantilica / conformal (MAPIE).
Retorna intervalo, nao so ponto."""
import numpy as np
import pandas as pd
from mapie.regression import ConformalizedQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from loteia.config import SEED
from loteia.model.train import (
    ALVO,
    FEATURES_CAT,
    FEATURES_NUM,
    montar_preprocessador,
)


class ModeloQuantilico:
    """CQR: quantis por gradient boosting + calibração conformal (MAPIE).

    O conjunto de calibração sai de um split aleatório DENTRO do treino —
    a regra de validação temporal vale para treino × teste, não aqui.
    """

    def __init__(
        self,
        cobertura: float = 0.8,
        features_num: list[str] | None = None,
        features_cat: list[str] | None = None,
        log_alvo: bool = False,
    ):
        self.cobertura = cobertura
        self.features_num = features_num or list(FEATURES_NUM)
        self.features_cat = features_cat or list(FEATURES_CAT)
        self.features_ = self.features_num + self.features_cat
        self.log_alvo = log_alvo

    def fit(self, train_df: pd.DataFrame) -> "ModeloQuantilico":
        X, y = train_df[self.features_], train_df[ALVO]
        if self.log_alvo:
            y = np.log1p(y)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=0.25, random_state=SEED
        )
        self.prep_ = montar_preprocessador(
            self.features_num, self.features_cat
        ).fit(X_fit)
        self.cqr_ = ConformalizedQuantileRegressor(
            estimator=HistGradientBoostingRegressor(
                loss="quantile", random_state=SEED
            ),
            confidence_level=self.cobertura,
        )
        self.cqr_.fit(self.prep_.transform(X_fit), y_fit)
        self.cqr_.conformalize(self.prep_.transform(X_cal), y_cal)
        return self

    def predict_intervalo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Colunas preco_m2_inf / preco_m2_med / preco_m2_sup por linha de df."""
        Xt = self.prep_.transform(df[self.features_])
        med, intervalo = self.cqr_.predict_interval(Xt)
        inf, sup = intervalo[:, 0, 0], intervalo[:, 1, 0]
        med = np.clip(med, inf, sup)
        if getattr(self, "log_alvo", False):
            # expm1 é monotônico → preserva a ordem e a cobertura do intervalo
            inf, med, sup = np.expm1(inf), np.expm1(med), np.expm1(sup)
        return pd.DataFrame(
            {"preco_m2_inf": inf, "preco_m2_med": med, "preco_m2_sup": sup},
            index=df.index,
        )


def cobertura_empirica(modelo: ModeloQuantilico, df: pd.DataFrame) -> float:
    """Fração das observações cujo preço real cai dentro do intervalo previsto."""
    pred = modelo.predict_intervalo(df)
    dentro = df[ALVO].between(pred["preco_m2_inf"], pred["preco_m2_sup"])
    return float(dentro.mean())


if __name__ == "__main__":
    from loteia.config import MODELS
    from loteia.model.experimento_geo import (
        FEATURES_FINAIS_CAT,
        FEATURES_FINAIS_NUM,
        montar_dataset_enriquecido,
    )
    from loteia.model.train import salvar_artefato, split_temporal

    # rodando como script, a classe local seria picklada como __main__.*
    # e o artefato não carregaria na API; usa a referência canônica do módulo
    from loteia.model.uncertainty import ModeloQuantilico

    df = montar_dataset_enriquecido()
    train, test = split_temporal(df)

    modelo = ModeloQuantilico(
        cobertura=0.8,
        features_num=FEATURES_FINAIS_NUM,
        features_cat=FEATURES_FINAIS_CAT,
        log_alvo=False,
    ).fit(train)
    cob = cobertura_empirica(modelo, test)
    pred = modelo.predict_intervalo(test)
    largura = (pred["preco_m2_sup"] - pred["preco_m2_inf"]).median()
    anos_teste = "+".join(str(a) for a in sorted(test["ano"].unique()))
    print(f"cobertura nominal 80% | empírica no teste ({anos_teste}): {cob:.1%}")
    print(f"largura mediana do intervalo: R$ {largura:,.0f}/m²")

    salvar_artefato(
        MODELS / "preco_m2_quantilico.joblib",
        modelo_quantilico=modelo,
        cobertura_nominal=0.8,
        cobertura_empirica_teste=cob,
    )
    print(f"artefato salvo em {MODELS / 'preco_m2_quantilico.joblib'}")
