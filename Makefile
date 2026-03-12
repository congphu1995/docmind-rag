.PHONY: dev test lint eval eval-custom seed seed-custom frontend backend infra clean

# Infrastructure
infra:
	docker compose up -d

infra-down:
	docker compose down

# Backend
backend:
	uv run uvicorn backend.app.main:app --reload --port 8000

worker:
	uv run celery -A backend.app.workers.celery_app worker --loglevel=info

# Frontend
frontend:
	cd frontend && npm run dev

# Full dev (requires 3 terminals or use &)
dev: infra
	@echo "Run in separate terminals:"
	@echo "  make backend"
	@echo "  make worker"
	@echo "  make frontend"

# Testing
test:
	uv run pytest tests/unit -v

test-all:
	uv run pytest -v

test-integration:
	uv run pytest tests/integration -m integration -v

# Linting
lint:
	uv run ruff check .
	uv run ruff format --check .

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

# Eval
eval:
	PYTHONPATH=. uv run python eval/run_eval.py

eval-custom:
	PYTHONPATH=. uv run python eval/run_eval.py --dataset custom

# Seed
seed:
	PYTHONPATH=. uv run python scripts/seed_demo_data.py

seed-custom:
	PYTHONPATH=. uv run python scripts/seed_custom_eval.py

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
