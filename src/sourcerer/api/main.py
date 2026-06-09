"""FastAPI application factory.

Ensures the database schema exists on startup, then serves the query API.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sourcerer.api.routes import router
from sourcerer.config import get_settings
from sourcerer.db.session import init_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    init_schema()  # idempotent; safe if already created
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sourcerer",
        description="A hybrid RAG knowledge assistant that answers with sources.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
