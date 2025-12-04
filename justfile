# Install dependencies and synchronize the environment
sync:
    uv sync

# Build the package
build:
    uv build

# Check code for linting issues
lint:
    uvx ruff check

# Fix linting issues
fix:
    uvx ruff check --fix

# Format code
format:
    uvx ruff format

# Install pre-commit hooks (only needs to be run once)
pre-commit-install:
    uvx pre-commit install --install-hooks

# Run pre-commit hooks on all files
pre-commit-run:
    uvx pre-commit run --all-files

# Run all tests
test:
    uv run pytest tests/ -v

# Run tests with coverage report
test-cov:
    uv run pytest tests/ --cov=coco_lib --cov-report=term-missing --cov-report=html

# Run doctests in source files
test-doctest:
    uv run pytest --doctest-modules src/coco_lib/ -v

# Run all tests including doctests with coverage
test-all:
    uv run pytest tests/ --doctest-modules src/coco_lib/ --cov=coco_lib --cov-report=term-missing -v