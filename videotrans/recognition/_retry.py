import random
import re
import time
from typing import Any, Callable, Optional, Tuple, TypeVar

import httpx

from videotrans.configure.config import logger, settings
from videotrans.configure.excepts import NO_RETRY_EXCEPT, StopTask

T = TypeVar("T")

RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _safe_no_retry_types() -> Tuple[type, ...]:
    return tuple(
        cls
        for cls in NO_RETRY_EXCEPT
        if isinstance(cls, type) and issubclass(cls, BaseException)
    )


def build_httpx_retry_types() -> Tuple[type, ...]:
    """Return transient timeout/read exception types safe against MagicMock httpx."""
    candidates: list[type] = [TimeoutError]
    for name in (
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
    ):
        cls = getattr(httpx, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            candidates.append(cls)
    return tuple(dict.fromkeys(candidates))


def extract_status_code(exc: BaseException) -> Optional[int]:
    """Extract numeric HTTP status code from provider SDK or HTTP exceptions."""
    for attr in ("status_code", "status", "code", "http_status"):
        val = getattr(exc, attr, None)
        if val is not None and not isinstance(val, bool):
            try:
                return int(val)
            except (TypeError, ValueError):
                m = re.match(r"^(\d{3})\b", str(val).strip())
                if m:
                    return int(m.group(1))
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if val is not None and not isinstance(val, bool):
            try:
                return int(val)
            except (TypeError, ValueError):
                m = re.match(r"^(\d{3})\b", str(val).strip())
                if m:
                    return int(m.group(1))
    return None


def is_retryable_status(code: Any) -> bool:
    """Return True for transient HTTP status codes (408, 429, 5xx)."""
    if code is None or isinstance(code, bool):
        return False
    try:
        status = int(code)
    except (TypeError, ValueError):
        return False
    return status in RETRYABLE_HTTP_STATUS


def is_retryable_error(exc: BaseException) -> bool:
    """Return True if exception represents a retryable HTTP status or quota/rate limit error."""
    code = extract_status_code(exc)
    if is_retryable_status(code):
        return True
    msg = str(getattr(exc, "message", exc) or exc).lower()
    return any(
        term in msg
        for term in (
            "resource_exhausted",
            "quota",
            "rate limit",
            "rate_limit",
            "ratelimit",
            "too many requests",
            "429",
            "overloaded",
            "unavailable",
        )
    )


def resolve_max_attempts(max_attempts: Optional[int] = None) -> int:
    """Resolve retry attempts at runtime from settings.get('retry_nums', 1)."""
    if max_attempts is not None:
        return max(1, int(max_attempts))
    try:
        retry_nums = int(float(settings.get("retry_nums", 1) or 1))
    except (TypeError, ValueError):
        retry_nums = 1
    return max(3, retry_nums + 1)


def run_with_retry(
    fn: Callable[[], T],
    *,
    should_retry: Callable[[BaseException], bool],
    is_cancelled: Optional[Callable[[], bool]] = None,
    max_attempts: Optional[int] = None,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    logger_obj=None,
    label: str = "STT",
) -> T:
    """Execute fn with runtime-configured exponential backoff + jitter retries."""
    attempts = resolve_max_attempts(max_attempts)
    log = logger_obj or logger
    no_retry_types = _safe_no_retry_types()

    for attempt in range(attempts):
        if is_cancelled is not None and is_cancelled():
            raise StopTask(f"[{label}] task cancelled")
        try:
            return fn()
        except BaseException as exc:
            if no_retry_types and isinstance(exc, no_retry_types):
                raise
            if attempt + 1 >= attempts or not should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** attempt)) * random.uniform(0.75, 1.25)
            log.warning(
                f"[{label}] attempt {attempt + 1}/{attempts} failed "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.2f}s"
            )
            if is_cancelled is not None and is_cancelled():
                raise StopTask(f"[{label}] task cancelled")
            sleep_fn(delay)
            if is_cancelled is not None and is_cancelled():
                raise StopTask(f"[{label}] task cancelled")

    raise RuntimeError(f"[{label}] unreachable retry termination")
