# OpenMFP Copilot

An AI copilot for the [OpenMFP](https://github.com/openmfp) and [Platform Mesh](https://github.com/platform-mesh) libraries and portal apps.

## Roadmap

| Stage | Topic | Status |
|---|---|---|
| 1 | FastAPI backend skeleton | done |
| 2 | LLM API: structured outputs, tool calling, streaming | done |
| 3 | RAG with PostgreSQL + pgvector | in progress |
| 4 | MCP server for OpenMFP and Platform Mesh libraries | planned |
| 5 | Angular frontend | planned |

## Repository layout

```
backend/      Python package (FastAPI), managed with uv
frontend/     Angular app (stage 5)
compose.yaml  Local PostgreSQL with pgvector (stage 3)
```

## Database

Requirements: Docker with Compose. Stage 3 stores documentation embeddings in PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension.

```bash
docker compose up -d --wait   # postgres://copilot:copilot@localhost:5432/openmfp_copilot
docker compose down           # stop; add -v to delete the data volume
```

## Backend

Requirements: [uv](https://docs.astral.sh/uv/). uv installs the Python version from `backend/.python-version` if it is missing.

```bash
cd backend
uv sync                      # create .venv and install dependencies
cp .env.example .env         # then set OPENMFP_COPILOT_ANTHROPIC_API_KEY
uv run fastapi dev           # http://localhost:8000, API docs at /docs
```

`POST /ask` calls the Claude API and needs an API key from [platform.claude.com](https://platform.claude.com). Without `OPENMFP_COPILOT_ANTHROPIC_API_KEY`, the SDK falls back to `ANTHROPIC_API_KEY`.

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What is portal-ui-lib for?"}'
```

`POST /ask` and `POST /ask/stream` let Claude read a repository's README from GitHub with the `get_readme` tool before answering. Without a token, GitHub allows 60 requests per hour.

`POST /ask/stream` streams the answer as Server-Sent Events: `delta` events with `{"text": "..."}` and a `tool_call` event with `{"name": "...", "input": {...}}` before each tool runs, then `done`, or `error` with `{"detail": "..."}`.

```bash
curl -sN localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question": "What is portal-ui-lib for?"}'
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
