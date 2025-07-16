.PHONY: help install install-dev test test-cov lint format type-check clean setup-dirs

help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code (black)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  clean        Clean up generated files"
	@echo "  setup-dirs   Create necessary directories"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/

format:
	black src/ tests/

type-check:
	mypy src/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

setup-dirs:
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p models
	mkdir -p logs
	mkdir -p chroma_db