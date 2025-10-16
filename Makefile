.PHONY: help install dev serve clean test

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	pip install -r requirements.txt
	pip install -e .

dev: ## Install in development mode
	pip install -e .

serve: ## Start the web server
	python start.py

serve-cli: ## Start the server using the CLI module
	python -m app.cli serve

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

test: ## Run tests (if any)
	@echo "No tests defined yet"
