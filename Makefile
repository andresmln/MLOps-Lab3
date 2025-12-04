install:
	uv sync

format:
	uv run black .

lint:
	uv run pylint mylib api cli

test:
	uv run python -m pytest

refactor: format lint

all: install refactor test
