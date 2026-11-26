import asyncio
import logging
import sys

import httpx2
from voyageai.error import InvalidRequestError

from openmfp_copilot.config import Settings, get_settings
from openmfp_copilot.db.session import database_sessions
from openmfp_copilot.documents.embeddings import create_embedder
from openmfp_copilot.documents.ingestion import Ingestion
from openmfp_copilot.github.client import GitHubClient, ReadmeNotFoundError
from openmfp_copilot.libraries.repository import KNOWN_LIBRARIES

logger = logging.getLogger("openmfp_copilot.ingest")


async def ingest_all(settings: Settings) -> int:
    failures = 0
    async with httpx2.AsyncClient() as http, database_sessions(settings) as session_factory:
        ingestion = Ingestion(GitHubClient(http), create_embedder(settings), session_factory)
        for library in KNOWN_LIBRARIES:
            name = f"{library.organization}/{library.name}"
            try:
                chunks = await ingestion.ingest_readme(library)
            except (ReadmeNotFoundError, httpx2.HTTPStatusError, InvalidRequestError) as error:
                failures += 1
                logger.error("%s: failed: %s", name, error)
            else:
                logger.info("%s: %d chunks", name, chunks)
    logger.info(
        "Ingested %d of %d libraries", len(KNOWN_LIBRARIES) - failures, len(KNOWN_LIBRARIES)
    )
    return failures


def main() -> None:
    logging.basicConfig(format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    failures = asyncio.run(ingest_all(get_settings()))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
