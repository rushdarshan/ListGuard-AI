import logging

import redis

from .config import settings

log = logging.getLogger(__name__)


def redis_status() -> dict:
    """Return {'state': 'ok'|'skipped'|'fail', 'detail': str}.

    REDIS_URL empty -> 'skipped' (local/test without redis). Does not fail /ready.
    REDIS_URL set -> ping with short timeout; failure is reported, /ready returns 503.
    """
    if not settings.redis_url:
        return {"state": "skipped", "detail": "REDIS_URL not configured"}
    try:
        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        try:
            client.ping()
        finally:
            client.close()  # return pooled connection; /ready may poll frequently
        return {"state": "ok", "detail": "pong"}
    except Exception:  # noqa: BLE001 - details go to server logs, never to callers
        log.exception("readiness redis check failed")
        return {"state": "fail", "detail": "unavailable"}
