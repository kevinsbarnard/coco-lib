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
