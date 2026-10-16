from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api.routes import router
from .core.config import BACKEND_DIR, get_settings
from .core.db import get_engine, init_engine
from .security.uploads import UploadError
from .services.executor import executor
from .services.sessions import ServiceError

log = logging.getLogger("experimentpilot")


def run_migrations() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    init_engine()
    run_migrations()
    get_engine()
    yield
    executor.shutdown()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="ExperimentPilot AI API", version=__version__, lifespan=lifespan,
                  description="Budget-constrained, risk-aware Bayesian diagnosis of ML model failures. "
                              "Trusted-local deployment: the API has no authentication layer.",
                  docs_url="/api/docs" if s.expose_docs else None, redoc_url="/api/redoc" if s.expose_docs else None,
                  openapi_url="/api/openapi.json" if s.expose_docs else None)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_list, allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["Content-Type"],
                       expose_headers=["Content-Disposition"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message, **exc.extra}})

    @app.exception_handler(UploadError)
    async def _upload_error(_: Request, exc: UploadError):
        return JSONResponse(status_code=exc.status, content={"error": {"code": "invalid_upload", "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(x) for x in e["loc"][1:]) or str(e["loc"][-1]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error",
                            "message": "; ".join(f"{d['field']}: {d['message']}" for d in details)[:400], "details": details}})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "An unexpected error occurred."}})

    app.include_router(router)
    return app


app = create_app()
