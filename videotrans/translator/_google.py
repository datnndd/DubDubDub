import html
import random
import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Union

import requests

from videotrans.configure.config import logger, settings, tr
from videotrans.configure.excepts import (
    NO_RETRY_EXCEPT,
    StopRetry,
    StopTask,
    TranslateSrtError,
)
from videotrans.translator._base import BaseTrans

_ORIGINAL_REQUESTS_GET = requests.get
_GOOGLE_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
)


@dataclass
class Google(BaseTrans):
    _local: threading.local = field(
        default_factory=threading.local, init=False, repr=False, compare=False
    )
    _session_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _sessions: List[requests.Session] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        super().__post_init__()
        self._local = threading.local()
        self._session_lock = threading.Lock()
        self._sessions = []

    def _get_session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": _GOOGLE_USER_AGENT})
            proxy = (self.proxy_str or "").strip()
            if proxy:
                session.proxies.update({"http": proxy, "https": proxy})
            self._local.session = session
            with self._session_lock:
                self._sessions.append(session)
        return session

    def _unload(self) -> None:
        with self._session_lock:
            for sess in self._sessions:
                try:
                    sess.close()
                except Exception:
                    pass
            self._sessions.clear()
        self._local = threading.local()

    def _close_clients(self) -> None:
        self._unload()

    # Send actual request to get results
    def _item_task(self, data: Union[List[str], str]) -> Optional[str]:
        if self._exit():
            return None
        text = "\n".join(i.strip() for i in data) if isinstance(data, list) else data
        source_code = (
            "auto"
            if not self.source_code or self.source_code in ("-", "No")
            else self.source_code
        )
        logger.debug(f"[Google] {self.target_code=} {self.source_code=}")

        raw_retry = int(settings.get("retry_nums", 3) or 3)
        max_retries = 3 if raw_retry <= 1 else raw_retry
        total_attempts = 1 + max_retries

        params_dict = {
            "sl": source_code,
            "tl": self.target_code,
            "hl": self.target_code,
            "q": text,
        }

        last_err: Optional[BaseException] = None
        for attempt in range(total_attempts):
            if self._exit():
                return None
            try:
                if requests.get is not _ORIGINAL_REQUESTS_GET:
                    req_kwargs = {
                        "params": params_dict,
                        "headers": {"User-Agent": _GOOGLE_USER_AGENT},
                        "timeout": 15,
                        "verify": True,
                    }
                    proxy = (self.proxy_str or "").strip()
                    if proxy:
                        req_kwargs["proxies"] = {"http": proxy, "https": proxy}
                    response = requests.get(
                        "https://translate.google.com/m", **req_kwargs
                    )
                else:
                    session = self._get_session()
                    response = session.get(
                        "https://translate.google.com/m",
                        params=params_dict,
                        timeout=15,
                        verify=True,
                    )
                response.raise_for_status()
                logger.debug(f"[Google]返回code:{response.status_code=}")

                re_result = re.search(
                    r'<div\s+class=["\']?result-container["\']?[^>]*>([^<]+?)<',
                    response.text,
                )
                if not re_result or len(re_result.groups()) < 1:
                    raise TranslateSrtError(tr("Google Translate error"))
                return html.unescape(re_result.group(1))
            except (StopTask, StopRetry):
                raise
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                TranslateSrtError,
            ) as e:
                last_err = e
            except requests.exceptions.HTTPError as e:
                status = (
                    e.response.status_code
                    if getattr(e, "response", None) is not None
                    else None
                )
                if status in (408, 429) or (status is not None and 500 <= status < 600):
                    last_err = e
                else:
                    raise StopTask(str(e)) from e
            except NO_RETRY_EXCEPT:
                raise

            if attempt < max_retries:
                delay = min(30.0, 1.0 * (2**attempt)) * random.uniform(0.5, 1.5)
                logger.warning(
                    f"[Google] attempt {attempt + 1}/{total_attempts} failed ({type(last_err).__name__}: {last_err}), retrying in {delay:.2f}s"
                )
                if self._sleep_interruptible(delay):
                    return None

        if isinstance(last_err, TranslateSrtError):
            raise last_err
        raise TranslateSrtError(
            str(last_err) if last_err else tr("Google Translate error")
        ) from last_err
