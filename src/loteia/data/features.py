"""Engenharia de atributos, incluindo enriquecimento geoespacial (Camada 5).
Geocodificação por centroide de quadra fiscal (precisão ~1 quarteirão) +
distâncias/contagens em EPSG:31983 (UTM, metros)."""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Praça da Sé (marco zero de SP) em EPSG:31983
MARCO_ZERO_SP: tuple[float, float] = (333_287.0, 7_394_586.0)

FEATURES_GEO_NUM: list[str] = [
    "dist_centro", "dist_estacao", "n_comercio_1km", "n_escola_1km", "renda_setor",
    "x", "y",
]
FEATURES_GEO_CAT: list[str] = ["zona"]
# Compat: lista única de features geo (numéricas + categóricas).
FEATURES_GEO: list[str] = FEATURES_GEO_NUM + FEATURES_GEO_CAT
SEM_ZONA = "SEM_ZONA"


def montar_centroides(gdf_quadras: gpd.GeoDataFrame) -> pd.DataFrame:
    """Centroide por (setor, quadra). Subquadras são agregadas pela média."""
    c = gdf_quadras.geometry.centroid
    base = pd.DataFrame(
        {
            "setor": gdf_quadras["cd_setor_fiscal"].astype(str).str.zfill(3),
            "quadra": gdf_quadras["cd_quadra_fiscal"].astype(str).str.zfill(3),
            "x": c.x,
            "y": c.y,
        }
    )
    return base.groupby(["setor", "quadra"], as_index=False)[["x", "y"]].mean()


def geocodificar(df: pd.DataFrame, centroides: pd.DataFrame) -> pd.DataFrame:
    """Anexa x, y pelo par (setor, quadra) extraído do sql10. Sem par → NaN."""
    out = df.copy()
    chaves = pd.DataFrame(
        {"setor": out["sql10"].str[:3], "quadra": out["sql10"].str[3:6]}
    )
    xy = chaves.merge(centroides, on=["setor", "quadra"], how="left")
    out["x"] = xy["x"].to_numpy()
    out["y"] = xy["y"].to_numpy()
    return out


def _validos(pontos: np.ndarray) -> np.ndarray:
    return np.isfinite(pontos).all(axis=1)


def dist_mais_proximo(pontos: np.ndarray, alvos: np.ndarray) -> np.ndarray:
    """Distância euclidiana de cada ponto ao alvo mais próximo (NaN propaga)."""
    out = np.full(len(pontos), np.nan)
    ok = _validos(pontos)
    if ok.any() and len(alvos):
        d, _ = cKDTree(alvos).query(pontos[ok])
        out[ok] = d
    return out


def contar_no_raio(pontos: np.ndarray, alvos: np.ndarray, raio: float) -> np.ndarray:
    """Quantos alvos caem dentro do raio de cada ponto (ponto inválido → 0)."""
    out = np.zeros(len(pontos), dtype=int)
    ok = _validos(pontos)
    if ok.any() and len(alvos):
        vizinhos = cKDTree(alvos).query_ball_point(pontos[ok], r=raio)
        out[ok] = [len(v) for v in vizinhos]
    return out


def renda_por_ponto(
    pontos: np.ndarray, gdf_renda: gpd.GeoDataFrame, coluna: str = "renda"
) -> np.ndarray:
    """Renda do polígono (setor censitário) que contém cada ponto; fora → NaN."""
    out = np.full(len(pontos), np.nan)
    ok = _validos(pontos)
    if not ok.any():
        return out
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pontos[ok, 0], pontos[ok, 1]),
        crs=gdf_renda.crs,
    )
    juncao = gpd.sjoin(
        pts, gdf_renda[[coluna, "geometry"]], how="left", predicate="within"
    )
    # ponto na fronteira de 2 polígonos: fica com o primeiro
    valores = juncao.groupby(level=0)[coluna].first()
    out[ok] = valores.reindex(range(len(pts))).to_numpy()
    return out


def zona_por_ponto(
    pontos: np.ndarray, gdf_zona: gpd.GeoDataFrame, coluna: str = "zona"
) -> np.ndarray:
    """Sigla da zona (zoneamento) do polígono que contém cada ponto.

    Categórica: ponto fora de qualquer polígono (ou inválido) recebe o
    sentinela SEM_ZONA, não NaN.
    """
    out = np.full(len(pontos), SEM_ZONA, dtype=object)
    ok = _validos(pontos)
    if not ok.any():
        return out
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pontos[ok, 0], pontos[ok, 1]),
        crs=gdf_zona.crs,
    )
    juncao = gpd.sjoin(
        pts, gdf_zona[[coluna, "geometry"]], how="left", predicate="within"
    )
    valores = juncao.groupby(level=0)[coluna].first().reindex(range(len(pts)))
    out[ok] = valores.fillna(SEM_ZONA).to_numpy()
    return out


def montar_features_geo(
    df: pd.DataFrame,
    *,
    centroides: pd.DataFrame,
    estacoes: np.ndarray,
    comercios: np.ndarray,
    escolas: np.ndarray,
    gdf_renda: gpd.GeoDataFrame,
    gdf_zona: gpd.GeoDataFrame | None = None,
    marco_zero: tuple[float, float] = MARCO_ZERO_SP,
) -> pd.DataFrame:
    """Geocodifica por quadra e anexa as FEATURES_GEO ao DataFrame.

    x, y (centroide da quadra) ficam como features contínuas; `zona` é
    categórica (SEM_ZONA quando o ponto não cai em nenhum perímetro).
    """
    out = geocodificar(df, centroides)
    xy = out[["x", "y"]].to_numpy(dtype=float)
    out["dist_centro"] = np.hypot(xy[:, 0] - marco_zero[0], xy[:, 1] - marco_zero[1])
    out["dist_estacao"] = dist_mais_proximo(xy, estacoes)
    out["n_comercio_1km"] = contar_no_raio(xy, comercios, raio=1_000.0)
    out["n_escola_1km"] = contar_no_raio(xy, escolas, raio=1_000.0)
    out["renda_setor"] = renda_por_ponto(xy, gdf_renda)
    out["zona"] = zona_por_ponto(xy, gdf_zona) if gdf_zona is not None else SEM_ZONA
    return out
