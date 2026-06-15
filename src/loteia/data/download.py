"""Baixa os arquivos públicos de ITBI (Fazenda/SP) e o cadastro do IPTU (GeoSampa)."""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from loteia.config import DATA_INTERIM, DATA_RAW

ITBI_URLS: dict[int, str] = {
    2023: "https://www.prefeitura.sp.gov.br/cidade/secretarias/upload/fazenda/arquivos/XLSX/GUIAS-DE-ITBI-PAGAS-2023.xlsx",
    2024: "https://prefeitura.sp.gov.br/cidade/secretarias/upload/fazenda/arquivos/itbi/GUIAS-DE-ITBI-PAGAS-2024.xlsx",
}

# Layout posicional das abas mensais (sem header) — fonte: aba EXPLICAÇÕES do xlsx.
ITBI_COLS: list[str] = [
    "sql_raw", "logradouro", "numero", "complemento", "bairro", "referencia",
    "cep", "natureza", "valor_transacao", "data_transacao", "vvr",
    "proporcao_transmitida", "vvr_proporcional", "base_calculo",
    "tipo_financiamento", "valor_financiado", "cartorio", "matricula",
    "situacao_sql", "area_terreno_itbi", "testada", "fracao_ideal",
    "area_construida_itbi", "uso_cod", "uso_desc", "padrao_cod",
    "padrao_desc", "acc",
]

_NUM_COLS: list[str] = [
    "valor_transacao", "vvr", "proporcao_transmitida", "vvr_proporcional",
    "base_calculo", "valor_financiado", "area_terreno_itbi", "testada",
    "fracao_ideal", "area_construida_itbi",
]

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def baixar_itbi(ano: int) -> Path:
    """Baixa o xlsx do ano para data/raw/. Pula se o arquivo já existe e parece completo."""
    dest = DATA_RAW / f"GUIAS-DE-ITBI-PAGAS-{ano}.xlsx"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    if ano not in ITBI_URLS:
        raise ValueError(f"URL não mapeada para o ano {ano}. Adicione em ITBI_URLS.")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl", "-L", "--retry", "10", "--retry-delay", "8",
            "--retry-all-errors", "--connect-timeout", "30",
            "--speed-time", "45", "--speed-limit", "3000",
            "-A", _UA, "-o", str(dest), ITBI_URLS[ano],
        ],
        check=True,
    )
    return dest


def ler_ano(ano: int) -> pd.DataFrame:
    """Lê todas as abas mensais do ano. Cacheia em parquet (1ª vez ~1min, depois instantâneo)."""
    from loteia.data.join import norm_sql10  # importação local para evitar ciclo

    cache = DATA_INTERIM / f"itbi_{ano}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    xls = pd.ExcelFile(baixar_itbi(ano))
    meses = [s for s in xls.sheet_names if re.fullmatch(r"[A-Z]{3}-\d{4}", str(s))]
    frames = []
    for m in meses:
        d = pd.read_excel(xls, sheet_name=m, header=None, names=ITBI_COLS)
        d["mes_aba"] = m
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["sql_raw"].notna()].copy()
    df["sql_raw"] = pd.to_numeric(df["sql_raw"], errors="coerce")
    df = df[df["sql_raw"].notna()].copy()
    for c in _NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["data_transacao"] = pd.to_datetime(df["data_transacao"], errors="coerce")
    df["ano"] = ano
    df["sql10"] = norm_sql10(df["sql_raw"])
    df["setor"] = df["sql10"].str[:3]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


if __name__ == "__main__":
    anos = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(ITBI_URLS)
    for ano in sorted(anos):
        print(f"ITBI {ano}...", end=" ", flush=True)
        df = ler_ano(ano)
        print(f"{len(df):,} transações")
