from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from openmfp_copilot.config import Settings

type SessionFactory = async_sessionmaker[AsyncSession]


@asynccontextmanager
async def database_sessions(settings: Settings) -> AsyncIterator[SessionFactory]:
    engine = create_async_engine(str(settings.database_url))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
