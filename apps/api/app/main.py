import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from .core import db as dbmod
from .core.redis_client import redis_status
from .routers import router

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="ListGuard API", version="0.1.0")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # NOTE: no Content-Security-Policy yet: dashboard.html runs an inline
        # module script, so `default-src 'self'` without 'unsafe-inline' would
        # break the app while adding zero XSS value (unsafe-inline is exactly
        # what an event-handler payload needs). CSP lands with the
        # external-scripts refactor; innerHTML is neutralized by escapeHtml.
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": str(exc.errors()[:3])}},
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        db_state = "ok"
        try:
            with dbmod.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - details go to server logs, never to callers
            log.exception("readiness db check failed")
            db_state = "fail"
        redis_state = redis_status()
        ok = db_state == "ok" and redis_state["state"] in ("ok", "skipped")
        payload = {"status": "ready" if ok else "not_ready", "checks": {"db": db_state, "redis": redis_state}}
        return JSONResponse(status_code=200 if ok else 503, content=payload)

    app.include_router(router)
    web_dir = Path(__file__).resolve().parents[2] / "web"  # apps/api/app -> apps/web
    if web_dir.is_dir():  # pragma: no cover - installed-package edge; always true in checkout
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


app = create_app()
