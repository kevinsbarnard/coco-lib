sync:
    uv sync

build:
    uv build

lint:
    uvx ruff check

fix:
    uvx ruff check --fix

format:
    uvx ruff format

pre-commit-install:
    uvx pre-commit install --install-hooks

pre-commit-run:
    uvx pre-commit run --all-files