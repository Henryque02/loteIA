"""Camada 5 - experimento controlado: o enriquecimento geoespacial melhora o
modelo de preço?

CRITÉRIO DE SUCESSO (definido ANTES de rodar): o modelo enriquecido deve reduzir
o MAE no teste temporal (2024) em pelo menos 5% frente ao modelo de 3 features.
Resultado negativo bem documentado também é entrega válida.
"""
import pandas as pd

from loteia.config import DATA_PROCESSED
from loteia.data.features import (
    FEATURES_GEO_CAT,
    FEATURES_GEO_NUM,
    montar_features_geo,
)
from loteia.model.explain import explicar_global
from loteia.model.train import (
    ALVO,
    FEATURES,
    FEATURES_NUM,
    BaselineMedianaSetor,
    avaliar,
    split_temporal,
    treinar_modelo,
)


def montar_dataset_enriquecido(anos: tuple[int, ...] = (2023, 2024)) -> pd.DataFrame:
    """Terrenos limpos + features geoespaciais, cacheado em data/processed."""
    from loteia.data.download import ler_ano
    from loteia.data.geo import (
        baixar_centroides_quadras,
        baixar_pontos_osm,
        baixar_renda_ibge,
        baixar_zoneamento,
    )
    from loteia.data.join import calcular_preco_m2, filtrar_terrenos

    cache = DATA_PROCESSED / "terrenos_geo.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    df = pd.concat([ler_ano(a) for a in anos], ignore_index=True)
    df = calcular_preco_m2(filtrar_terrenos(df))
    osm = baixar_pontos_osm()
    df = montar_features_geo(
        df,
        centroides=baixar_centroides_quadras(),
        estacoes=osm["estacoes"],
        comercios=osm["comercios"],
        escolas=osm["escolas"],
        gdf_renda=baixar_renda_ibge(),
        gdf_zona=baixar_zoneamento(),
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# Feature sets do modelo final enriquecido (geo + zona, sem o setor ordinal).
FEATURES_FINAIS_NUM = FEATURES_NUM + FEATURES_GEO_NUM
FEATURES_FINAIS_CAT = FEATURES_GEO_CAT


if __name__ == "__main__":
    df = montar_dataset_enriquecido()
    train, test = split_temporal(df)
    print(f"treino: {len(train):,} | teste: {len(test):,}\n")

    baseline = BaselineMedianaSetor().fit(train[FEATURES], train[ALVO])
    baseline.features_ = FEATURES
    modelo_3f = treinar_modelo(train)  # área+testada+setor
    modelo_geo = treinar_modelo(  # +geo +zona, sem setor ordinal
        train, features_num=FEATURES_FINAIS_NUM, features_cat=FEATURES_FINAIS_CAT
    )
    modelo_log = treinar_modelo(  # idem, com log-alvo
        train, features_num=FEATURES_FINAIS_NUM, features_cat=FEATURES_FINAIS_CAT,
        log_alvo=True,
    )

    resultados = pd.DataFrame(
        {
            "baseline mediana/setor": avaliar(baseline, test),
            "GBM 3 features": avaliar(modelo_3f, test),
            "GBM geo+zona": avaliar(modelo_geo, test),
            "GBM geo+zona (log)": avaliar(modelo_log, test),
        }
    ).T
    print(resultados.round(3).to_string(), "\n")

    mae_3f = resultados.loc["GBM 3 features", "mae"]
    mae_final = resultados.loc["GBM geo+zona", "mae"]
    ganho = 1 - mae_final / mae_3f
    veredito = "SUCESSO" if ganho >= 0.05 else "NEGATIVO"
    print(f"ganho de MAE (geo+zona vs 3f): {ganho:+.1%} (critério: ≥ +5%) → {veredito}\n")

    # modelo final = geo+zona sem log (melhor MAE/MAPE/R² no teste)
    print("importância global (média |SHAP|, R$/m²) do modelo final:")
    print(explicar_global(modelo_geo, train).round(1).to_string())
