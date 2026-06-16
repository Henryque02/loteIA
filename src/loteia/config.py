"""Caminhos e parâmetros globais do projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Corte temporal da validação (treina <= ; testa >). Walk-forward: com 2025
# disponível, treina 2023+2024 e testa no ano mais novo e fechado (2025).
ANO_CORTE_TREINO = 2024
SEED = 42
