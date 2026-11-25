# OpenMFP Copilot

An AI copilot for the [OpenMFP](https://github.com/openmfp) libraries.

## Roadmap

| Stage | Topic | Status |
|---|---|---|
| 1 | FastAPI backend skeleton | in progress |
| 2 | LLM API: structured outputs, tool calling, streaming | planned |
| 3 | RAG with PostgreSQL + pgvector | planned |
| 4 | MCP server for OpenMFP libraries | planned |
| 5 | Angular frontend | planned |

## Repository layout

```
backend/    Python package (FastAPI), managed with uv
frontend/   Angular app (stage 5)
```

## Backend

Requirements: [uv](https://docs.astral.sh/uv/). uv installs the Python version from `backend/.python-version` if it is missing.

```bash
cd backend
uv sync                      # create .venv and install dependencies
cp .env.example .env         # optional, local overrides
uv run fastapi dev           # http://localhost:8000, API docs at /docs
```

Checks:

```bash
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src tests        # type check (strict)
```

## License

[Apache-2.0](LICENSE)
