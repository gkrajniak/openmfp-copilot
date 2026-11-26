import argparse
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


async def ingest_all(settings: Settings, *, force: bool) -> int:
    ingested = unchanged = failures = 0
    async with httpx2.AsyncClient() as http, database_sessions(settings) as session_factory:
        ingestion = Ingestion(GitHubClient(http), create_embedder(settings), session_factory)
        for library in KNOWN_LIBRARIES:
            name = f"{library.organization}/{library.name}"
            try:
                result = await ingestion.ingest_readme(library, force=force)
            except (ReadmeNotFoundError, httpx2.HTTPStatusError, InvalidRequestError) as error:
                failures += 1
                logger.error("%s: failed: %s", name, error)
            else:
                if result.unchanged:
                    unchanged += 1
                    logger.info("%s: unchanged, skipped", name)
                else:
                    ingested += 1
                    logger.info("%s: %d chunks", name, result.chunks)
    logger.info(
        "%d libraries: %d ingested, %d unchanged, %d failed",
        len(KNOWN_LIBRARIES),
        ingested,
        unchanged,
        failures,
    )
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m openmfp_copilot.ingest",
        description="Embed the README of every known library and store the chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-ingest READMEs even when their content and embedding model are unchanged",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    failures = asyncio.run(ingest_all(get_settings(), force=args.force))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
