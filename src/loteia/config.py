"""Caminhos e parâmetros globais do projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Corte temporal da validação (treina <= ; testa >)
ANO_CORTE_TREINO = 2023
SEED = 42
