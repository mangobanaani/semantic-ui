.PHONY: help install install-dev test test-cov lint format clean run docker-build docker-up docker-down docker-logs

# Default target
help:
	@echo "Semantic Kernel UI - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-cov        Run tests with coverage report"
	@echo "  make test-watch      Run tests in watch mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run all linters (flake8, mypy)"
	@echo "  make format          Format code with black and isort"
	@echo "  make format-check    Check formatting without making changes"
	@echo ""
	@echo "Development:"
	@echo "  make run             Run the application locally"
	@echo "  make clean           Clean generated files and caches"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-up       Start Docker services"
	@echo "  make docker-dev      Start development Docker environment"
	@echo "  make docker-down     Stop Docker services"
	@echo "  make docker-logs     View Docker logs"
	@echo "  make docker-test     Run tests in Docker"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# Testing
test:
	PYTHONPATH=src pytest tests/ -v

test-cov:
	PYTHONPATH=src pytest tests/ -v \
		--cov=src/semantic_kernel_ui \
		--cov-report=html \
		--cov-report=term

test-watch:
	PYTHONPATH=src ptw tests/ -- -v

# Code quality
lint:
	@echo "Running flake8..."
	@flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,E402,W503,E501 || true
	@echo ""
	@echo "Running mypy..."
	@mypy src/ --config-file mypy.ini || true
	@echo ""
	@echo "Linting complete!"

format:
	@echo "Running black..."
	@black src/ tests/
	@echo ""
	@echo "Running isort..."
	@isort src/ tests/
	@echo ""
	@echo "Formatting complete!"

format-check:
	@echo "Checking black formatting..."
	@black --check src/ tests/
	@echo ""
	@echo "Checking isort formatting..."
	@isort --check-only src/ tests/
	@echo ""
	@echo "Format check complete!"

# Development
run:
	PYTHONPATH=src streamlit run src/semantic_kernel_ui/app.py \
		--server.port=8501 \
		--server.address=localhost

clean:
	@echo "Cleaning generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ coverage_reports/ .coverage 2>/dev/null || true
	@echo "Clean complete!"

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-dev:
	docker-compose -f docker-compose.dev.yml up

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f app

docker-test:
	docker-compose -f docker-compose.dev.yml run --rm test-runner

docker-lint:
	docker-compose -f docker-compose.dev.yml run --rm linter
