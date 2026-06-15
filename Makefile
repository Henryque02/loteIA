.PHONY:  data geo train api front test lint
front: ; uv run streamlit run app/streamlit_app.py
data:  ; uv run python -m loteia.data.download && uv run python -m loteia.data.join
geo:   ; uv run python -m loteia.data.geo
train: ; uv run python -m loteia.model.train && uv run python -m loteia.model.uncertainty
api:   ; uv run uvicorn loteia.api.main:app --reload
test:  ; uv run pytest -q
lint:  ; uv run ruff check src tests
