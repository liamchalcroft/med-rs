# medrs Makefile
.PHONY: install build test clean lint format docs

# Installation
install:
	cargo build --release --features python
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"
	cargo install cargo-nextest

# Building
build:
	cargo build --release --features python

# Testing
test: test-rust test-python

test-rust:
	cargo nextest run --all-features --release

test-python:
	python -m pytest tests/ -v

# Code quality
lint:
	cargo clippy --all-features -- -D warnings
	@command -v ruff >/dev/null 2>&1 && ruff check src/python/ tests/ || echo "ruff not available"

format:
	cargo fmt
	@command -v black >/dev/null 2>&1 && black src/python/ tests/ examples/ || echo "black not available"

# Documentation
docs:
	cd docs && make html

# Cleanup
clean:
	cargo clean
	rm -rf src/python/__pycache__/ tests/__pycache__ .pytest_cache

# Development helpers
check-env:
	@echo "Rust: $$(rustc --version)"
	@echo "Python: $$(python --version)"
	@echo "Cargo: $$(cargo --version)"