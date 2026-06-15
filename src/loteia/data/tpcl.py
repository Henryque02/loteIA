"""Cadastro TPCL (IPTU) ano-alinhado — junta a cada transação de ITBI o cadastro
do **ano da transação**, não o snapshot atual.

Motivação (verificado empiricamente): os arquivos anuais de ITBI embutem sempre
o cadastro ATUAL; usar área/uso atuais para uma venda de 2019 enviesa (lote
vendido vazio e construído depois seria descartado do alvo de terrenos). O join
ano-alinhado corrige isso.

ACESSO À FONTE (passo manual): o cadastro TPCL/IPTU histórico por ano de exercício
não é baixável de forma automatizada. Obtenha cada ano e salve em
``data/raw/tpcl_<ano>.csv`` (ou .parquet) com as colunas:
``sql, area_terreno, area_construida, uso_desc``. Fontes:
- Base dos Dados (BigQuery): base de IPTU de São Paulo particionada por ano.
- GeoSampa "Acervo / Camadas para Download" (cadastro) — porém o portal entrega o
  snapshot atual; para anos antigos use a Base dos Dados ou pedido via LAI/e-SIC.
"""
import pandas as pd

from loteia.config import DATA_RAW
from loteia.data.join import norm_sql10

# Colunas mínimas que o cadastro de cada ano precisa expor após o carregamento.
COLS_CADASTRO = ["sql10", "area_terreno", "area_construida", "uso_desc"]


def carregar_cadastro_tpcl(ano: int, caminho: "str | None" = None) -> pd.DataFrame:
    """Lê o cadastro TPCL de um ano e normaliza para COLS_CADASTRO.

    Procura ``data/raw/tpcl_<ano>.{parquet,csv}`` se `caminho` não for dado.
    Espera uma coluna de SQL (``sql`` ou ``sql10``) + área/uso; normaliza o SQL
    para 10 dígitos via :func:`norm_sql10`. Sem arquivo, ergue FileNotFoundError
    com instrução (nunca inventa dados).
    """
    candidatos = (
        [caminho]
        if caminho is not None
        else [DATA_RAW / f"tpcl_{ano}.parquet", DATA_RAW / f"tpcl_{ano}.csv"]
    )
    origem = next((c for c in candidatos if c is not None and pd.io.common.file_exists(c)), None)
    if origem is None:
        raise FileNotFoundError(
            f"Cadastro TPCL de {ano} ausente. Salve data/raw/tpcl_{ano}.csv com as "
            "colunas sql, area_terreno, area_construida, uso_desc (ver docstring do "
            "módulo loteia.data.tpcl para as fontes)."
        )

    origem = str(origem)
    raw = pd.read_parquet(origem) if origem.endswith(".parquet") else pd.read_csv(origem)
    col_sql = "sql10" if "sql10" in raw.columns else "sql"
    cad = pd.DataFrame({
        "sql10": norm_sql10(raw[col_sql]),
        "area_terreno": pd.to_numeric(raw["area_terreno"], errors="coerce"),
        "area_construida": pd.to_numeric(raw["area_construida"], errors="coerce"),
        "uso_desc": raw["uso_desc"].astype("string"),
    })
    return cad.dropna(subset=["sql10"])


def juntar_cadastro_ano_alinhado(
    itbi: pd.DataFrame, cadastros: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """Anexa a cada transação o cadastro do ANO da transação (não o atual).

    `cadastros` mapeia ano → DataFrame com COLS_CADASTRO. As colunas resultantes
    seguem os nomes usados a jusante (area_terreno_itbi, area_construida_itbi,
    uso_desc), para que ``filtrar_terrenos`` e ``calcular_preco_m2`` funcionem
    sem alteração. Transações cujo ano não tem cadastro ficam com área NaN (são
    naturalmente descartadas pelos filtros).
    """
    partes = []
    for ano, grupo in itbi.groupby("ano"):
        cad = cadastros.get(int(ano))
        g = grupo.copy()
        if cad is None:
            g["area_terreno_itbi"] = pd.NA
            g["area_construida_itbi"] = pd.NA
            g["uso_desc"] = pd.NA
        else:
            cad = cad[COLS_CADASTRO].rename(
                columns={
                    "area_terreno": "area_terreno_itbi",
                    "area_construida": "area_construida_itbi",
                }
            )
            g = g.drop(
                columns=["area_terreno_itbi", "area_construida_itbi", "uso_desc"],
                errors="ignore",
            ).merge(cad, on="sql10", how="left")
        partes.append(g)
    return pd.concat(partes, ignore_index=True)
