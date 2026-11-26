import asyncio
import logging
from typing import Literal

from alembic import context
from alembic.autogenerate.api import AutogenContext
from pgvector.sqlalchemy import Vector
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from openmfp_copilot.config import get_settings
from openmfp_copilot.documents.models import DocumentChunk

logging.basicConfig(format="%(levelname)-5.5s [%(name)s] %(message)s")
logging.getLogger("alembic").setLevel(logging.INFO)

database_url = str(get_settings().database_url)
target_metadata = DocumentChunk.metadata


def render_item(type_: str, obj: object, autogen_context: AutogenContext) -> str | Literal[False]:
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({obj.dim})"
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        render_item=render_item,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, render_item=render_item
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
