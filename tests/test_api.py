"""Testes da API: POST /viabilidade liga modelo de preço → Monte Carlo."""
import pytest
from fastapi.testclient import TestClient

from loteia.api.main import criar_app
from loteia.model.train import salvar_artefato, split_temporal
from loteia.model.uncertainty import ModeloQuantilico
from tests.test_model import _df_sintetico


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from loteia.model.train import treinar_modelo

    train, _ = split_temporal(_df_sintetico(n=400, seed=1), ano_corte=2023)
    pasta = tmp_path_factory.mktemp("models")
    caminho = pasta / "preco_m2_quantilico.joblib"
    salvar_artefato(caminho, modelo_quantilico=ModeloQuantilico(0.8).fit(train))
    caminho_pontual = pasta / "preco_m2.joblib"
    salvar_artefato(caminho_pontual, modelo=treinar_modelo(train))
    with TestClient(criar_app(caminho, caminho_pontual)) as c:
        yield c


PAYLOAD = {
    "terreno": {"setor": "085", "area_lote_m2": 300.0, "testada": 10.0},
    "projeto": {
        "n_lotes": 50,
        "custo_gleba": 2_000_000.0,
        "custo_infra": {"minimo": 1_500_000, "moda": 2_000_000, "maximo": 2_500_000},
        "meses_obra": 12,
        "meses_vendas": {"minimo": 12, "moda": 24, "maximo": 48},
        "n_parcelas": 1,
        "taxa_alvo_anual": 0.15,
    },
    "n_sims": 500,
    "seed": 42,
}


class TestHealth:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}


class TestCustosNaViabilidade:
    def test_custos_operacionais_reduzem_vpl(self, client):
        sem = client.post(
            "/viabilidade", json={**PAYLOAD, "incluir_distribuicao": True}
        ).json()
        payload = {
            **PAYLOAD,
            "incluir_distribuicao": True,
            "projeto": {
                **PAYLOAD["projeto"],
                "custos": {
                    "comissao_pct": 0.06,
                    "impostos_pct": 0.02,
                    "marketing_pct": 0.03,
                    "admin_mensal": 50_000.0,
                    "licenciamento": 1_000_000.0,
                },
            },
        }
        com = client.post("/viabilidade", json=payload).json()
        # custos pesados precisam mover o VPL mediano para baixo de verdade
        import numpy as np

        assert np.median(com["distribuicoes"]["vpl"]) < np.median(
            sem["distribuicoes"]["vpl"]
        )

    def test_campos_de_custo_sao_opcionais(self, client):
        # sem os campos novos, segue funcionando (default 0)
        assert client.post("/viabilidade", json=PAYLOAD).status_code == 200


class TestViabilidade:
    def test_retorna_200_com_campos_principais(self, client):
        r = client.post("/viabilidade", json=PAYLOAD)
        assert r.status_code == 200
        corpo = r.json()
        assert {"prob_viavel", "faixa_preco_m2", "tir_anual", "vpl"} <= set(corpo)

    def test_prob_entre_0_e_1(self, client):
        corpo = client.post("/viabilidade", json=PAYLOAD).json()
        assert 0.0 <= corpo["prob_viavel"] <= 1.0

    def test_faixa_de_preco_ordenada(self, client):
        faixa = client.post("/viabilidade", json=PAYLOAD).json()["faixa_preco_m2"]
        assert faixa["inf"] <= faixa["med"] <= faixa["sup"]

    def test_distribuicoes_resumidas_em_percentis(self, client):
        corpo = client.post("/viabilidade", json=PAYLOAD).json()
        for chave in ("tir_anual", "vpl"):
            assert {"p10", "p50", "p90"} <= set(corpo[chave])
            assert corpo[chave]["p10"] <= corpo[chave]["p90"]

    def test_mesma_seed_mesmo_resultado(self, client):
        r1 = client.post("/viabilidade", json=PAYLOAD).json()
        r2 = client.post("/viabilidade", json=PAYLOAD).json()
        assert r1["prob_viavel"] == r2["prob_viavel"]

    def test_payload_invalido_da_422(self, client):
        assert client.post("/viabilidade", json={"terreno": {}}).status_code == 422


class TestCamposExtras:
    def test_sensibilidade_na_resposta(self, client):
        corpo = client.post("/viabilidade", json=PAYLOAD).json()
        sens = corpo["sensibilidade"]
        assert {linha["variavel"] for linha in sens} == {
            "preco_lote", "custo_infra", "meses_vendas",
        }
        assert all("amplitude" in linha for linha in sens)

    def test_fatores_shap_na_resposta(self, client):
        corpo = client.post("/viabilidade", json=PAYLOAD).json()
        fatores = corpo["fatores_shap"]
        assert fatores is not None
        assert {"setor", "area_terreno_itbi", "testada"} <= set(fatores["contribuicoes"])
        assert "base" in fatores

    def test_distribuicoes_so_quando_pedidas(self, client):
        sem = client.post("/viabilidade", json=PAYLOAD).json()
        assert sem["distribuicoes"] is None
        com = client.post(
            "/viabilidade", json={**PAYLOAD, "incluir_distribuicao": True}
        ).json()
        assert 0 < len(com["distribuicoes"]["vpl"]) <= 2000
        assert 0 < len(com["distribuicoes"]["tir_anual"]) <= 2000

    def test_sem_artefato_pontual_shap_e_none(self, tmp_path):
        train, _ = split_temporal(_df_sintetico(n=400, seed=1), ano_corte=2023)
        caminho = tmp_path / "q.joblib"
        salvar_artefato(caminho, modelo_quantilico=ModeloQuantilico(0.8).fit(train))
        with TestClient(criar_app(caminho, tmp_path / "inexistente.joblib")) as c:
            corpo = c.post("/viabilidade", json=PAYLOAD).json()
        assert corpo["fatores_shap"] is None


class TestQuadra:
    def test_sem_recursos_geo_da_404(self, client):
        assert client.get("/quadra/085/013").status_code == 404


PAYLOAD_OTIMIZAR = {
    "terreno": {"setor": "085", "testada": 10.0},
    "gleba": {
        "area_vendavel_m2": 30_000.0,
        "candidatos_area_lote": [150.0, 300.0, 600.0],
        "custo_gleba": 10_000_000.0,
        "custo_infra": {"minimo": 8_000_000, "moda": 10_000_000, "maximo": 12_000_000},
        "meses_obra": 12,
        "absorcao_lotes_mes": {"minimo": 2, "moda": 4, "maximo": 8},
        "taxa_alvo_anual": 0.15,
        "n_parcelas": 1,
    },
    "n_sims": 300,
    "seed": 42,
}


class TestOtimizar:
    def test_retorna_configuracoes_ordenadas(self, client):
        r = client.post("/otimizar", json=PAYLOAD_OTIMIZAR)
        assert r.status_code == 200
        configs = r.json()["configuracoes"]
        assert len(configs) == 3
        probs = [c["prob_viavel"] for c in configs]
        assert probs == sorted(probs, reverse=True)

    def test_n_lotes_respeita_area_vendavel(self, client):
        configs = client.post("/otimizar", json=PAYLOAD_OTIMIZAR).json()["configuracoes"]
        for c in configs:
            assert c["n_lotes"] == int(30_000.0 // c["area_lote_m2"])

    def test_payload_invalido_da_422(self, client):
        assert client.post("/otimizar", json={"terreno": {}}).status_code == 422

    def test_aceita_rho_mercado(self, client):
        r = client.post("/otimizar", json={**PAYLOAD_OTIMIZAR, "rho_mercado": 0.6})
        assert r.status_code == 200
        assert len(r.json()["configuracoes"]) == 3


class TestCorrelacaoMercado:
    def test_rho_altera_a_distribuicao(self, client):
        base = {**PAYLOAD, "incluir_distribuicao": True}
        sem = client.post("/viabilidade", json={**base, "rho_mercado": 0.0}).json()
        com = client.post("/viabilidade", json={**base, "rho_mercado": 0.9}).json()
        # a correlação muda o caminho de amostragem → distribuição de VPL diferente
        assert sem["distribuicoes"]["vpl"] != com["distribuicoes"]["vpl"]

    def test_default_aceito_sem_rho(self, client):
        # omitir rho_mercado usa o default do produto (correlação moderada)
        assert client.post("/viabilidade", json=PAYLOAD).status_code == 200

    def test_rho_fora_do_intervalo_da_422(self, client):
        assert client.post(
            "/viabilidade", json={**PAYLOAD, "rho_mercado": 1.5}
        ).status_code == 422


class TestViabilidadeComModeloGeo:
    """Integração com o artefato real (features geoespaciais) e caches locais."""

    @pytest.fixture(scope="class")
    def client_real(self):
        from loteia.api.main import ARTEFATO_PADRAO

        if not ARTEFATO_PADRAO.exists():
            pytest.skip("artefato real ausente (rode make train antes)")
        with TestClient(criar_app()) as c:
            yield c

    def test_com_quadra_responde_200(self, client_real):
        payload = {**PAYLOAD, "terreno": {**PAYLOAD["terreno"], "quadra": "013"}}
        r = client_real.post("/viabilidade", json=payload)
        assert r.status_code == 200
        assert 0.0 <= r.json()["prob_viavel"] <= 1.0

    def test_sem_quadra_da_422_quando_modelo_e_geo(self, client_real):
        r = client_real.post("/viabilidade", json=PAYLOAD)
        assert r.status_code == 422
        assert "quadra" in r.json()["detail"].lower()

    def test_otimizar_com_quadra_responde_200(self, client_real):
        payload = {
            **PAYLOAD_OTIMIZAR,
            "terreno": {**PAYLOAD_OTIMIZAR["terreno"], "quadra": "013"},
        }
        r = client_real.post("/otimizar", json=payload)
        assert r.status_code == 200
        assert len(r.json()["configuracoes"]) == 3

    def test_quadra_existente_devolve_lat_lon(self, client_real):
        r = client_real.get("/quadra/085/013")
        assert r.status_code == 200
        corpo = r.json()
        # quadra central de SP: lat ~ -23.5, lon ~ -46.6
        assert -24.5 < corpo["lat"] < -23.0
        assert -47.5 < corpo["lon"] < -46.0

    def test_quadra_inexistente_da_404(self, client_real):
        assert client_real.get("/quadra/999/999").status_code == 404
