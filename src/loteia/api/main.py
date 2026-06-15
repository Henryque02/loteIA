"""FastAPI. POST /viabilidade -> probabilidade de retorno, faixa de preco,
distribuicao de TIR, fatores SHAP, sensibilidade. Carrega artefatos na subida."""
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from loteia.config import MODELS, SEED
from loteia.data.features import FEATURES_GEO, montar_features_geo
from loteia.finance.cashflow import Custos
from loteia.finance.montecarlo import Incerteza, simular_viabilidade
from loteia.model.train import carregar_artefato

ARTEFATO_PADRAO = MODELS / "preco_m2_quantilico.joblib"
ARTEFATO_PONTUAL_PADRAO = MODELS / "preco_m2.joblib"


class TerrenoBase(BaseModel):
    setor: str
    quadra: str | None = None  # obrigatória quando o modelo usa features geo
    testada: float = Field(gt=0)


class Terreno(TerrenoBase):
    area_lote_m2: float = Field(gt=0)


class FaixaIncerteza(BaseModel):
    minimo: float
    moda: float
    maximo: float

    def como_incerteza(self) -> Incerteza:
        return Incerteza(self.minimo, self.moda, self.maximo)


class CustosOperacionais(BaseModel):
    """Custos sobre o VGV (%) + admin mensal + licenciamento. Todos opcionais."""

    comissao_pct: float = Field(default=0.0, ge=0, le=1)
    impostos_pct: float = Field(default=0.0, ge=0, le=1)
    marketing_pct: float = Field(default=0.0, ge=0, le=1)
    admin_mensal: float = Field(default=0.0, ge=0)
    licenciamento: float = Field(default=0.0, ge=0)

    def como_custos(self) -> Custos:
        return Custos(
            comissao_pct=self.comissao_pct,
            impostos_pct=self.impostos_pct,
            marketing_pct=self.marketing_pct,
            admin_mensal=self.admin_mensal,
            licenciamento=self.licenciamento,
        )


class Projeto(BaseModel):
    n_lotes: int = Field(gt=0)
    custo_gleba: float = Field(ge=0)
    custo_infra: FaixaIncerteza
    meses_obra: int = Field(gt=0)
    meses_vendas: FaixaIncerteza
    n_parcelas: int = Field(default=1, gt=0)
    taxa_alvo_anual: float = Field(gt=-1)
    custos: CustosOperacionais = CustosOperacionais()
    mes_inicio_vendas: int = Field(default=1, gt=0)


class PedidoViabilidade(BaseModel):
    terreno: Terreno
    projeto: Projeto
    n_sims: int = Field(default=10_000, gt=0, le=100_000)
    seed: int = SEED
    incluir_distribuicao: bool = False
    # correlação preço↔absorção (mercado quente: preço alto + venda rápida).
    # 0.5 = moderada (default do produto); 0 = inputs independentes.
    rho_mercado: float = Field(default=0.5, ge=0, lt=1)


class Gleba(BaseModel):
    area_vendavel_m2: float = Field(gt=0)
    candidatos_area_lote: list[float] = Field(min_length=1)
    custo_gleba: float = Field(ge=0)
    custo_infra: FaixaIncerteza
    meses_obra: int = Field(gt=0)
    absorcao_lotes_mes: FaixaIncerteza
    taxa_alvo_anual: float = Field(gt=-1)
    n_parcelas: int = Field(default=1, gt=0)
    custos: CustosOperacionais = CustosOperacionais()
    mes_inicio_vendas: int = Field(default=1, gt=0)


class PedidoOtimizar(BaseModel):
    terreno: TerrenoBase
    gleba: Gleba
    n_sims: int = Field(default=5_000, gt=0, le=100_000)
    seed: int = SEED
    rho_mercado: float = Field(default=0.5, ge=0, lt=1)


def _percentis(x: np.ndarray) -> dict[str, float | None]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def criar_app(
    caminho_artefato: Path | None = None,
    caminho_pontual: Path | None = None,
) -> FastAPI:
    caminho = caminho_artefato or ARTEFATO_PADRAO
    caminho_p = caminho_pontual or ARTEFATO_PONTUAL_PADRAO

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        modelo = carregar_artefato(caminho)["modelo_quantilico"]
        app.state.modelo = modelo
        # modelo pontual é opcional: sem ele a resposta omite os fatores SHAP
        app.state.modelo_pontual = (
            carregar_artefato(caminho_p)["modelo"] if caminho_p.exists() else None
        )
        features = getattr(modelo, "features_", [])
        app.state.usa_geo = bool(set(FEATURES_GEO) & set(features))
        if app.state.usa_geo:
            # lê os caches locais (data/interim); baixa só se não existirem
            from loteia.data.geo import (
                baixar_centroides_quadras,
                baixar_pontos_osm,
                baixar_renda_ibge,
                baixar_zoneamento,
            )

            osm = baixar_pontos_osm()
            app.state.geo = {
                "centroides": baixar_centroides_quadras(),
                "estacoes": osm["estacoes"],
                "comercios": osm["comercios"],
                "escolas": osm["escolas"],
                "gdf_renda": baixar_renda_ibge(),
                "gdf_zona": baixar_zoneamento(),
            }
        yield

    app = FastAPI(title="LoteIA", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def _linha_base(t: TerrenoBase) -> pd.DataFrame:
        """Linha de features do terreno (sem a área, que varia por uso)."""
        X = pd.DataFrame([{"setor": t.setor, "testada": t.testada}])
        if app.state.usa_geo:
            if not t.quadra:
                raise HTTPException(
                    status_code=422,
                    detail="o modelo usa features geoespaciais: informe a quadra "
                    "fiscal (3 dígitos) junto com o setor",
                )
            X["sql10"] = f"{t.setor:0>3}{t.quadra:0>3}0000"
            X = montar_features_geo(X, **app.state.geo)
        return X

    @app.post("/viabilidade")
    def viabilidade(pedido: PedidoViabilidade):
        t, p = pedido.terreno, pedido.projeto
        X = _linha_base(t)
        X["area_terreno_itbi"] = t.area_lote_m2
        faixa = app.state.modelo.predict_intervalo(X).iloc[0]
        # intervalo de 80% do preço/m² → triangular do preço do lote
        preco_lote = Incerteza(
            faixa["preco_m2_inf"] * t.area_lote_m2,
            faixa["preco_m2_med"] * t.area_lote_m2,
            faixa["preco_m2_sup"] * t.area_lote_m2,
        )
        custos = p.custos.como_custos()
        r = simular_viabilidade(
            n_lotes=p.n_lotes,
            custo_gleba=p.custo_gleba,
            meses_obra=p.meses_obra,
            preco_lote=preco_lote,
            custo_infra=p.custo_infra.como_incerteza(),
            meses_vendas=p.meses_vendas.como_incerteza(),
            taxa_alvo_anual=p.taxa_alvo_anual,
            n_parcelas=p.n_parcelas,
            custos=custos,
            mes_inicio_vendas=p.mes_inicio_vendas,
            rho_mercado=pedido.rho_mercado,
            n_sims=pedido.n_sims,
            seed=pedido.seed,
        )

        from loteia.finance.sensitivity import sensibilidade_tornado

        sens = sensibilidade_tornado(
            n_lotes=p.n_lotes,
            custo_gleba=p.custo_gleba,
            meses_obra=p.meses_obra,
            preco_lote=preco_lote,
            custo_infra=p.custo_infra.como_incerteza(),
            meses_vendas=p.meses_vendas.como_incerteza(),
            taxa_alvo_anual=p.taxa_alvo_anual,
            n_parcelas=p.n_parcelas,
            custos=custos,
            mes_inicio_vendas=p.mes_inicio_vendas,
        )

        fatores_shap = None
        if app.state.modelo_pontual is not None:
            from loteia.model.explain import explicar_local

            contrib, base = explicar_local(app.state.modelo_pontual, X)
            fatores_shap = {
                "contribuicoes": {k: float(v) for k, v in contrib.items()},
                "base": base,
            }

        distribuicoes = None
        if pedido.incluir_distribuicao:
            tirs = r["tir_anual"]
            distribuicoes = {
                "vpl": [float(v) for v in r["vpl"][:2_000]],
                "tir_anual": [
                    float(v) for v in tirs[np.isfinite(tirs)][:2_000]
                ],
            }

        return {
            "prob_viavel": r["prob_viavel"],
            "faixa_preco_m2": {
                "inf": float(faixa["preco_m2_inf"]),
                "med": float(faixa["preco_m2_med"]),
                "sup": float(faixa["preco_m2_sup"]),
            },
            "tir_anual": _percentis(r["tir_anual"]),
            "vpl": _percentis(r["vpl"]),
            "sensibilidade": sens.to_dict(orient="records"),
            "fatores_shap": fatores_shap,
            "distribuicoes": distribuicoes,
            "taxa_alvo_anual": p.taxa_alvo_anual,
            "n_sims": pedido.n_sims,
        }

    @app.get("/quadra/{setor}/{quadra}")
    def quadra(setor: str, quadra: str):
        if not app.state.usa_geo:
            raise HTTPException(
                status_code=404, detail="recursos geoespaciais não carregados"
            )
        c = app.state.geo["centroides"]
        achado = c[(c["setor"] == f"{setor:0>3}") & (c["quadra"] == f"{quadra:0>3}")]
        if achado.empty:
            raise HTTPException(status_code=404, detail="quadra não encontrada")
        from pyproj import Transformer

        x, y = float(achado["x"].iloc[0]), float(achado["y"].iloc[0])
        lon, lat = Transformer.from_crs(31983, 4326, always_xy=True).transform(x, y)
        return {"x": x, "y": y, "lat": lat, "lon": lon}

    @app.post("/otimizar")
    def otimizar(pedido: PedidoOtimizar):
        from loteia.optimize.config_search import (
            otimizar_configuracao,
            precificador_do_modelo,
        )

        g = pedido.gleba
        resultado = otimizar_configuracao(
            area_vendavel_m2=g.area_vendavel_m2,
            candidatos_area_lote=g.candidatos_area_lote,
            precificador=precificador_do_modelo(
                app.state.modelo, _linha_base(pedido.terreno)
            ),
            custo_gleba=g.custo_gleba,
            custo_infra=g.custo_infra.como_incerteza(),
            meses_obra=g.meses_obra,
            absorcao_lotes_mes=g.absorcao_lotes_mes.como_incerteza(),
            taxa_alvo_anual=g.taxa_alvo_anual,
            n_parcelas=g.n_parcelas,
            custos=g.custos.como_custos(),
            mes_inicio_vendas=g.mes_inicio_vendas,
            rho_mercado=pedido.rho_mercado,
            n_sims=pedido.n_sims,
            seed=pedido.seed,
        )
        # NaN (ex.: TIR indefinida) não é JSON válido
        resultado = resultado.astype(object).where(resultado.notna(), None)
        return {
            "configuracoes": resultado.to_dict(orient="records"),
            "taxa_alvo_anual": g.taxa_alvo_anual,
            "n_sims": pedido.n_sims,
        }

    return app


app = criar_app()
