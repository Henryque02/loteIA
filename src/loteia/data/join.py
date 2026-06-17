"""Cruza ITBI x IPTU pela chave SQL (Setor-Quadra-Lote).
Anexa area_terreno, area_construida e zoneamento a cada transacao.
Usa a área já embutida no próprio ITBI (cobertura ~100%)."""
import sys

import pandas as pd


def norm_sql10(serie: pd.Series) -> pd.Series:
    """Normaliza a chave SQL do ITBI para 10 dígitos (setor+quadra+lote, sem verificador).

    O ITBI armazena o SQL como número de 11 dígitos (inclui dígito de controle).
    Ao ser lido pelo pandas, perde zeros à esquerda. Esta função:
      1. converte para inteiro (remove decimais de float)
      2. zfill(11) para restaurar zeros
      3. pega os 10 primeiros dígitos (descarta o verificador)
    """
    s = pd.to_numeric(serie, errors="coerce")
    out = pd.Series(pd.NA, index=serie.index, dtype="object")
    ok = s.notna()
    out[ok] = s[ok].astype("int64").astype(str).str.zfill(11).str[:10]
    return out.astype("string")


def filtrar_terrenos(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas transações de terrenos (uso=TERRENO e área construída < 1 m²)."""
    return df[
        (df["uso_desc"] == "TERRENO")
        & (df["area_construida_itbi"].fillna(0) < 1)
    ].copy()


def calcular_preco_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra transações limpas e calcula preço/m² do terreno.

    Limpo = compra e venda (natureza 1.*), totalidade do imóvel (proporção 100%,
    fração ideal 1), valor plausível (> R$1.000) e área do terreno > 0.
    Retorna o DataFrame filtrado com a coluna 'preco_m2' adicionada.
    """
    clean = df[
        df["natureza"].astype(str).str.startswith("1.")
        & df["proporcao_transmitida"].eq(100)
        & df["fracao_ideal"].eq(1)
        & df["valor_transacao"].gt(1000)
        & df["area_terreno_itbi"].gt(0)
    ].copy()
    clean["preco_m2"] = clean["valor_transacao"] / clean["area_terreno_itbi"]
    return clean


if __name__ == "__main__":
    from loteia.data.download import ler_ano

    anos = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [2023, 2024, 2025]
    df = pd.concat([ler_ano(a) for a in anos], ignore_index=True)
    terrenos = filtrar_terrenos(df)
    preco = calcular_preco_m2(terrenos)
    print(f"terrenos: {len(terrenos):,}  |  limpos p/ alvo: {len(preco):,}")
    print(preco["preco_m2"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(0))
