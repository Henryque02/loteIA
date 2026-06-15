"""Downloads geoespaciais da Camada 5, com cache em data/interim.
Quadras fiscais (GeoSampa WFS) -> centroides; OSM (Overpass) -> estações,
comércio, escolas; IBGE Censo 2010 -> renda por setor censitário."""
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

from loteia.config import DATA_INTERIM, DATA_RAW

WFS_URL = "https://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
IBGE_AGREGADOS = (
    "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2010/Resultados_do_Universo/"
    "Agregados_por_Setores_Censitarios/SP_Capital_20231030.zip"
)
IBGE_MALHA = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
    "malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2010/"
    "setores_censitarios_shp/sp/sp_setores_censitarios.zip"
)
# Overpass e FTP do IBGE devolvem 406/403 sem User-Agent
_HEADERS = {"User-Agent": "LoteIA-academico/0.1 (projeto de disciplina)"}
_CRS_METRICO = "EPSG:31983"


def _wfs_paginado(typename: str, pagina: int) -> "gpd.GeoDataFrame":
    """Baixa todas as features de uma camada WFS, paginado e com retry.

    O GeoServer do GeoSampa estrangula payloads grandes de geometria → read
    timeout generoso (connect 30s, read 600s) e até 4 tentativas por página.
    """
    frames = []
    inicio = 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": typename, "outputFormat": "application/json",
            "count": pagina, "startIndex": inicio, "sortBy": "cd_identificador",
        }
        for tentativa in range(4):
            try:
                r = requests.get(
                    WFS_URL, params=params, headers=_HEADERS, timeout=(30, 600)
                )
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if tentativa == 3:
                    raise
                print(f"    retry {typename} @ {inicio} ({e.__class__.__name__})",
                      flush=True)
        gdf = gpd.read_file(io.BytesIO(r.content))
        if len(gdf) == 0:
            break
        frames.append(gdf)
        print(f"  {typename}: +{len(gdf):,} (total {inicio + len(gdf):,})", flush=True)
        if len(gdf) < pagina:
            break
        inicio += pagina
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry").set_crs(
        _CRS_METRICO, allow_override=True
    )


def baixar_centroides_quadras(pagina: int = 5_000) -> pd.DataFrame:
    """Baixa todas as quadras fiscais (paginado) e cacheia só os centroides."""
    from loteia.data.features import montar_centroides

    cache = DATA_INTERIM / "centroides_quadras.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    frames = []
    inicio = 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": "geoportal:quadra_fiscal",
            "outputFormat": "application/json",
            "count": pagina, "startIndex": inicio,
            "sortBy": "cd_identificador",
        }
        r = requests.get(WFS_URL, params=params, headers=_HEADERS, timeout=300)
        r.raise_for_status()
        gdf = gpd.read_file(io.BytesIO(r.content))
        if len(gdf) == 0:
            break
        frames.append(montar_centroides(gdf.set_crs(_CRS_METRICO, allow_override=True)))
        print(f"  quadras: +{len(gdf):,} (total {inicio + len(gdf):,})", flush=True)
        if len(gdf) < pagina:
            break
        inicio += pagina

    centroides = (
        pd.concat(frames, ignore_index=True)
        .groupby(["setor", "quadra"], as_index=False)[["x", "y"]]
        .mean()
    )
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    centroides.to_parquet(cache)
    return centroides


def _overpass(consulta: str) -> np.ndarray:
    """Roda a consulta e devolve pontos (x, y) em EPSG:31983."""
    corpo = f'[out:json][timeout:300];area[name="São Paulo"][admin_level=8]->.sp;{consulta}'
    r = requests.post(OVERPASS_URL, data={"data": corpo}, headers=_HEADERS, timeout=360)
    r.raise_for_status()
    elementos = r.json()["elements"]
    lons, lats = [], []
    for e in elementos:
        centro = e if "lat" in e else e.get("center", {})
        if "lat" in centro:
            lons.append(centro["lon"])
            lats.append(centro["lat"])
    t = Transformer.from_crs(4326, _CRS_METRICO, always_xy=True)
    x, y = t.transform(lons, lats)
    return np.column_stack([x, y])


def baixar_pontos_osm() -> dict[str, np.ndarray]:
    """Estações de metrô/trem, comércio e escolas de SP via Overpass (cacheado)."""
    consultas = {
        "estacoes": "(node(area.sp)[railway=station];node(area.sp)[station=subway];);out;",
        "comercios": "node(area.sp)[shop];out;",
        "escolas": "(node(area.sp)[amenity=school];way(area.sp)[amenity=school];);out center;",
    }
    pontos = {}
    for nome, consulta in consultas.items():
        cache = DATA_INTERIM / f"osm_{nome}.parquet"
        if cache.exists():
            pontos[nome] = pd.read_parquet(cache)[["x", "y"]].to_numpy()
            continue
        xy = _overpass(consulta)
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(xy, columns=["x", "y"]).to_parquet(cache)
        print(f"  osm {nome}: {len(xy):,} pontos", flush=True)
        pontos[nome] = xy
    return pontos


def baixar_zoneamento(pagina: int = 2_000) -> "gpd.GeoDataFrame":
    """Perímetros de zoneamento vigente (WFS zoneamento_2016_map1) em EPSG:31983.

    Coluna `zona` = sigla do perímetro (ex.: ZM, ZEU, ZER, ZC). Cobertura
    citywide; cacheado em parquet.
    """
    cache = DATA_INTERIM / "zoneamento.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)

    gdf = _wfs_paginado("geoportal:zoneamento_2016_map1", pagina)
    gdf = gdf.rename(columns={"cd_zoneamento_perimetro": "zona"})[["zona", "geometry"]]
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    return gdf


def _baixar_zip(url: str, dest: Path) -> Path:
    if not dest.exists():
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, headers=_HEADERS, timeout=600)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def baixar_renda_ibge() -> gpd.GeoDataFrame:
    """Malha de setores censitários da capital + renda média (V005, Censo 2010)."""
    cache = DATA_INTERIM / "renda_setores_censitarios.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)

    zip_agregados = _baixar_zip(IBGE_AGREGADOS, DATA_RAW / "ibge_sp_capital_2010.zip")
    with zipfile.ZipFile(zip_agregados) as z:
        nome = next(n for n in z.namelist() if n.lower().endswith("basico_sp1.csv"))
        with z.open(nome) as f:
            basico = pd.read_csv(
                f, sep=";", decimal=",", encoding="latin1", dtype={"Cod_setor": str}
            )
    renda = basico[["Cod_setor", "V005"]].rename(
        columns={"Cod_setor": "cd_setor", "V005": "renda"}
    )
    renda["renda"] = pd.to_numeric(renda["renda"], errors="coerce")

    zip_malha = _baixar_zip(IBGE_MALHA, DATA_RAW / "ibge_sp_malha_setores.zip")
    malha = gpd.read_file(f"zip://{zip_malha}")
    malha = malha[malha["CD_GEOCODI"].astype(str).str.startswith("3550308")]
    malha = malha.rename(columns={"CD_GEOCODI": "cd_setor"})[["cd_setor", "geometry"]]
    malha["cd_setor"] = malha["cd_setor"].astype(str)

    gdf = malha.merge(renda, on="cd_setor", how="left").to_crs(_CRS_METRICO)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    return gdf


if __name__ == "__main__":
    print("centroides de quadras fiscais...")
    centroides = baixar_centroides_quadras()
    print(f"  {len(centroides):,} quadras")
    print("pontos OSM...")
    osm = baixar_pontos_osm()
    for nome, xy in osm.items():
        print(f"  {nome}: {len(xy):,}")
    print("renda IBGE...")
    renda = baixar_renda_ibge()
    print(f"  {len(renda):,} setores censitários | renda mediana R$ {renda['renda'].median():,.0f}")
    print("zoneamento...")
    zona = baixar_zoneamento()
    print(f"  {len(zona):,} perímetros | {zona['zona'].nunique()} siglas distintas")
