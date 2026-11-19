import copy
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from tenacity import RetryError

from videotrans import translator
from videotrans.configure.base import BaseCon
from videotrans.configure.config import TEMP_ROOT, logger, settings, tr
from videotrans.task.taskcfg import SrtItem
from videotrans.util.help_misc import get_md5, serial
from videotrans.util.help_srt import cleartext, get_subtitle_from_srt


_CACHE_IO_LOCK = threading.Lock()
_SRT_BLOCK_RE = re.compile(
    r"(?m)(?:^|\n)\s*\d+\s*\r?\n\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}"
)


def _encode_indexed_batch(items: Sequence[Union[SrtItem, dict, str]]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(items, 1):
        if isinstance(item, str):
            raw = item
        elif hasattr(item, "get"):
            raw = item.get("text", "")
        else:
            raw = getattr(item, "text", item)
        clean = re.sub(r"\s*\r?\n\s*", " ", str(raw or "")).strip()
        lines.append(f"[{idx}] {clean}")
    return "\n".join(lines)


def _parse_indexed_response(raw: str, expected_count: int) -> dict[int, str]:
    parsed: dict[int, str] = {}
    current_idx: Optional[int] = None
    for raw_line in (raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= expected_count:
                current_idx = idx
                text_part = m.group(2).strip()
                if idx not in parsed:
                    parsed[idx] = text_part
                elif text_part:
                    parsed[idx] = f"{parsed[idx]} {text_part}".strip()
                continue
            if idx == 0 or not (1900 <= idx <= 2100):
                current_idx = None
                continue
        if current_idx is not None:
            parsed[current_idx] = (
                f"{parsed[current_idx]} {line}".strip()
                if parsed[current_idx]
                else line
            )
    return parsed


def _parse_srt_fallback(
    raw_res: str, items: Sequence[Union[SrtItem, dict]]
) -> dict[int, str]:
    parsed: dict[int, str] = {}
    if not raw_res or not str(raw_res).strip():
        return parsed
    fallback_items = get_subtitle_from_srt(raw_res, is_file=False)
    if len(fallback_items) == len(items):
        return {
            idx: fallback_items[idx - 1]["text"].strip()
            for idx in range(1, len(items) + 1)
        }
    time_map = {
        x["time"]: x["text"].strip()
        for x in fallback_items
        if x.get("time")
    }
    line_map = {
        x["line"]: x["text"].strip()
        for x in fallback_items
        if x.get("line") is not None
    }
    for idx, src in enumerate(items, 1):
        txt = time_map.get(src.get("time", ""), "")
        if not txt:
            txt = line_map.get(src.get("line"), "")
        if txt:
            parsed[idx] = txt
    return parsed


@dataclass
class BaseTrans(BaseCon):
    # Translation channel
    translate_type: int = 0
    # List storing subtitles to be translated
    text_list: Optional[List[SrtItem]] = None
    # Unique task ID
    uuid: Optional[str] = None
    # Do not use cache during tests
    is_test: bool = False
    # Original language code
    source_code: str = ""
    # Target language code
    target_code: str = ""
    # For AI channels, natural language representation of target language; equals target_code for other channels
    target_language_name: str = ""

    # Translation API URL
    api_url: str = field(default="", init=False)
    # Model name
    model_name: str = field(default="", init=False)
    # Number of subtitle lines per batch
    trans_thread: int = 5
    # Pause seconds after translation (loaded dynamically in __post_init__)
    wait_sec: float = 0.0
    # Whether AI translation channel and send full SRT format is selected
    aisendsrt: Optional[bool] = None
    local_dir: Optional[str] = None

    aitrans_context: bool = field(default=False, init=False)
    context_window: int = field(default=5, init=False)
    compact_mode: bool = field(default=True, init=False)
    _progress_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _run_abort_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        super().__post_init__()
        Path(TEMP_ROOT + "/translate_cache").mkdir(parents=True, exist_ok=True)
        self.wait_sec = float(settings.get("translation_wait", 0) or 0)
        requested_mode = (
            settings.get("aisendsrt", False)
            if self.aisendsrt is None
            else self.aisendsrt
        )
        self.aisendsrt = (
            bool(requested_mode)
            and self.translate_type in translator.AI_TRANS_CHANNELS
        )
        self.aitrans_context = (
            bool(settings.get("aitrans_context", False))
            and self.translate_type in translator.AI_TRANS_CHANNELS
        )
        raw_cw = settings.get("aitrans_context_window", 5)
        self.context_window = max(0, int(5 if raw_cw is None else raw_cw))
        raw_compact = settings.get("aitrans_compact", True)
        self.compact_mode = (
            bool(True if raw_compact is None else raw_compact)
            and self.translate_type in translator.AI_TRANS_CHANNELS
        )
        if self.aisendsrt:
            self.trans_thread = max(1, int(settings.get("aitrans_thread", 20) or 20))
        else:
            self.trans_thread = max(1, int(settings.get("trans_thread", 5) or 5))
        self._progress_lock = threading.Lock()
        self._run_abort_event = threading.Event()

    def _exit(self) -> bool:
        if self._run_abort_event.is_set():
            return True
        return super()._exit()

    def _item_task(self, data: Union[List[str], str]):
        raise NotImplementedError()

    def _translate_batch(
        self,
        data: Union[List[str], str],
        context_lines: Optional[List[str]] = None,
    ) -> Optional[str]:
        return self._item_task(data)

    def _sleep_interruptible(self, seconds: float) -> bool:
        if seconds <= 0:
            return self._exit()
        event = getattr(self.cancellation_token, "_event", None)
        deadline = time.monotonic() + seconds
        while True:
            if self._exit():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if event is not None and hasattr(event, "wait"):
                if event.wait(min(0.05, remaining)):
                    return True
            else:
                if self._run_abort_event.wait(min(0.2, remaining)):
                    return True

    def _effective_concurrency(self, num_batches: int) -> int:
        if num_batches <= 1:
            return 1
        if self.aitrans_context and self.context_window > 0:
            return 1
        is_ai = self.translate_type in translator.AI_TRANS_CHANNELS
        cfg_key = "aitrans_concurrency" if is_ai else "trans_concurrency"
        default_c = 4 if is_ai else 1
        raw_c = settings.get(cfg_key, default_c)
        c = int(default_c if raw_c is None else raw_c)
        return max(1, min(c, num_batches))

    def _emit_batch_progress(self, completed: int, total: int) -> None:
        pct = round(completed / max(1, total) * 100.0, 1)
        self.signal(
            text=tr("starttrans") + f" {completed}/{total}",
            type="trans_progress",
            progress=pct,
            completed=completed,
            total=total,
        )

    def _build_cache_key_input(
        self, payload: Any, context_lines: Optional[List[str]] = None
    ) -> Any:
        if context_lines:
            return {"payload": payload, "context": list(context_lines)}
        return payload

    # Actual operations run -> run_text|run_srt -> _item_task
    def run(self) -> List[SrtItem]:
        self._run_abort_event.clear()
        try:
            if hasattr(self, "_download"):
                self.signal(text=tr("check or download models"))
                self._download()
                self.signal(text=tr("Transation subtitles"))
            items = self.text_list or []
            if not items:
                return []
            if not self.aisendsrt:
                source_text = [t["text"].replace("\n", " ") for t in items]
                return self._run_text(
                    [
                        source_text[i : i + self.trans_thread]
                        for i in range(0, len(source_text), self.trans_thread)
                    ]
                )
            return self._run_srt(
                [
                    items[i : i + self.trans_thread]
                    for i in range(0, len(items), self.trans_thread)
                ]
            )
        except RetryError as e:
            raise e.last_attempt.exception()
        except (OSError, FileNotFoundError) as e:
            _e = str(e)
            if self.local_dir and (
                "no file named model.safetensors" in _e
                or os.path.basename(self.local_dir) in _e
            ):
                from videotrans.configure.excepts import DownloadModelsError

                raise DownloadModelsError(
                    tr("model incomplete error", self.local_dir, tr("Help document"))
                )
            raise
        finally:
            self._run_abort_event.set()
            if hasattr(self, "_unload"):
                self._unload()
            if hasattr(self, "_close_clients"):
                self._close_clients()

    def _process_text_batch(
        self,
        it: List[str],
        context_lines: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        if self._exit():
            return None
        is_ai = self.translate_type in translator.AI_TRANS_CHANNELS
        payload: Union[List[str], str] = (
            [f"[{idx}] {line}" for idx, line in enumerate(it, 1)] if is_ai else it
        )
        cache_input = self._build_cache_key_input(payload, context_lines)
        cached = self._get_cache(cache_input)
        from_cache = cached is not None and bool(str(cached).strip())

        if from_cache:
            cleaned = str(cached)
        else:
            raw_res = self._translate_batch(payload, context_lines=context_lines)
            if self._exit() or raw_res is None:
                return None
            cleaned = cleartext(raw_res)

        if is_ai:
            if _SRT_BLOCK_RE.search(cleaned):
                srt_items = get_subtitle_from_srt(cleaned, is_file=False)
                if srt_items:
                    lines_out = [
                        item["text"].strip() for item in srt_items[: len(it)]
                    ]
                    if len(lines_out) < len(it):
                        lines_out.extend([""] * (len(it) - len(lines_out)))
                    if not from_cache and any(lines_out):
                        canonical = "\n".join(
                            f"[{idx}] {txt}" for idx, txt in enumerate(lines_out, 1)
                        )
                        self._set_cache(cache_input, canonical)
                    return lines_out

            parsed = _parse_indexed_response(cleaned, len(it))
            if not parsed and "-->" in cleaned:
                srt_items = get_subtitle_from_srt(cleaned, is_file=False)
                if srt_items:
                    lines_out = [
                        item["text"].strip() for item in srt_items[: len(it)]
                    ]
                    if len(lines_out) < len(it):
                        lines_out.extend([""] * (len(it) - len(lines_out)))
                    if not from_cache and any(lines_out):
                        canonical = "\n".join(
                            f"[{idx}] {txt}" for idx, txt in enumerate(lines_out, 1)
                        )
                        self._set_cache(cache_input, canonical)
                    return lines_out

            if parsed:
                missing_indices = [
                    idx
                    for idx, src in enumerate(it, 1)
                    if str(src).strip() and not parsed.get(idx, "").strip()
                ]
                if missing_indices and not from_cache and not self._exit():
                    retry_payload = [
                        f"[{k}] {it[idx - 1]}"
                        for k, idx in enumerate(missing_indices, 1)
                    ]
                    retry_raw = self._translate_batch(
                        retry_payload, context_lines=context_lines
                    )
                    if not self._exit() and retry_raw is not None:
                        retry_clean = cleartext(retry_raw)
                        retry_parsed = _parse_indexed_response(
                            retry_clean, len(missing_indices)
                        )
                        if retry_parsed:
                            for k, orig_idx in enumerate(missing_indices, 1):
                                if retry_parsed.get(k, "").strip():
                                    parsed[orig_idx] = retry_parsed[k].strip()
                        elif len(missing_indices) == 1 and retry_clean.strip():
                            parsed[missing_indices[0]] = re.sub(
                                r"^\[\d+\]\s*", "", retry_clean.strip()
                            ).strip()
                lines_out = [parsed.get(idx, "").strip() for idx in range(1, len(it) + 1)]
                if not from_cache and any(lines_out):
                    canonical = "\n".join(
                        f"[{idx}] {txt}" for idx, txt in enumerate(lines_out, 1)
                    )
                    self._set_cache(cache_input, canonical)
                return lines_out

            # Positional fallback when mocked _item_task omits [id] tags
            sep_res = [line.strip() for line in cleaned.split("\n")]
            if len(sep_res) > len(it) and sep_res and not sep_res[-1]:
                sep_res.pop()
            lines_out = [
                sep_res[x] if x < len(sep_res) else "" for x in range(len(it))
            ]
            if not from_cache and any(lines_out):
                self._set_cache(cache_input, cleaned)
            return lines_out

        # Non-AI channels (e.g., Google)
        if not from_cache and cleaned.strip():
            self._set_cache(cache_input, cleaned)
        sep_res = cleaned.split("\n")
        return [sep_res[x].strip() if x < len(sep_res) else "" for x in range(len(it))]

    def _run_text(self, split_source_text: List[List[str]]):
        total_batches = len(split_source_text)
        if total_batches == 0:
            return []
        concurrency = self._effective_concurrency(total_batches)
        logger.debug(
            f"以纯文本行形式翻译，每次翻译{self.trans_thread}行，并发{concurrency}，翻译后暂停{self.wait_sec}s"
        )

        batch_results: List[Optional[List[str]]] = [None] * total_batches
        if concurrency == 1:
            rolling_context: List[str] = []
            for i, it in enumerate(split_source_text):
                if self._exit():
                    return []
                self.signal(text=tr("starttrans") + f" {i} ")
                ctx = (
                    rolling_context[-self.context_window :]
                    if (
                        self.aitrans_context
                        and self.context_window > 0
                        and rolling_context
                    )
                    else None
                )
                lines_out = self._process_text_batch(it, context_lines=ctx)
                if self._exit() or lines_out is None:
                    return []
                batch_results[i] = lines_out
                for line in lines_out:
                    if line:
                        rolling_context.append(line)
                if self.context_window > 0 and len(rolling_context) > self.context_window:
                    rolling_context = rolling_context[-self.context_window :]
                for line in lines_out:
                    self.signal(text=line + "\n", type="subtitle")
                with self._progress_lock:
                    self._emit_batch_progress(i + 1, total_batches)
                if self.wait_sec > 0 and total_batches > 1 and i < total_batches - 1:
                    if self._sleep_interruptible(self.wait_sec):
                        return []
        else:
            completed_count = 0
            pool = ThreadPoolExecutor(max_workers=concurrency)
            cancelled_or_error = False
            try:
                future_to_idx = {
                    pool.submit(self._process_text_batch, it, None): idx
                    for idx, it in enumerate(split_source_text)
                }
                pending = set(future_to_idx.keys())
                while pending:
                    if self._exit():
                        cancelled_or_error = True
                        return []
                    done, pending = wait(
                        pending, timeout=0.2, return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        idx = future_to_idx[fut]
                        res = fut.result()
                        if self._exit() or res is None:
                            cancelled_or_error = True
                            self._run_abort_event.set()
                            return []
                        batch_results[idx] = res
                        with self._progress_lock:
                            completed_count += 1
                            self._emit_batch_progress(completed_count, total_batches)
            except BaseException:
                cancelled_or_error = True
                self._run_abort_event.set()
                raise
            finally:
                if cancelled_or_error:
                    self._run_abort_event.set()
                pool.shutdown(
                    wait=not cancelled_or_error, cancel_futures=cancelled_or_error
                )

            if self._exit():
                return []
            for lines_out in batch_results:
                if lines_out:
                    for line in lines_out:
                        self.signal(text=line + "\n", type="subtitle")

        target_list: List[str] = []
        for lines_out in batch_results:
            if lines_out:
                target_list.extend(lines_out)

        items = self.text_list or []
        max_i = len(target_list)
        logger.debug(f"原始行数:{len(items)},翻译后行数:{max_i}")
        _empty_line = 0
        for i, _it in enumerate(items):
            text = target_list[i].strip() if i < max_i else ""
            if not text:
                _empty_line += 1
            items[i]["text"] = text

        if not items or _empty_line >= len(items):
            from videotrans.configure.excepts import TranslateSrtError

            raise TranslateSrtError(tr("Translate result is empty") + f"\n{self.api_url}")
        return items

    def _process_srt_batch(
        self,
        it: List[SrtItem],
        context_lines: Optional[List[str]] = None,
    ) -> Optional[List[SrtItem]]:
        if self._exit():
            return None
        from videotrans.configure.excepts import TranslateSrtError

        if self.compact_mode:
            payload_str = _encode_indexed_batch(it)
            cache_input = self._build_cache_key_input(payload_str, context_lines)
            cached = self._get_cache(cache_input)
            from_cache = cached is not None and bool(str(cached).strip())
            if from_cache:
                raw_res = str(cached)
            else:
                raw_res = self._translate_batch(payload_str, context_lines=context_lines)
                if self._exit() or raw_res is None:
                    return None
                if not str(raw_res).strip():
                    raise TranslateSrtError(
                        tr("Translate result is empty") + f"\n{self.api_url}"
                    )

            if _SRT_BLOCK_RE.search(raw_res):
                parsed = _parse_srt_fallback(raw_res, it)
            else:
                parsed = _parse_indexed_response(raw_res, len(it))
                if not parsed and "-->" in raw_res:
                    parsed = _parse_srt_fallback(raw_res, it)
                elif not parsed:
                    plain = [ln.strip() for ln in raw_res.splitlines() if ln.strip()]
                    if len(plain) == len(it):
                        parsed = {idx: plain[idx - 1] for idx in range(1, len(it) + 1)}

            missing_indices = [
                idx
                for idx, item in enumerate(it, 1)
                if str(item.get("text", "")).strip() and not parsed.get(idx, "").strip()
            ]
            if missing_indices and not from_cache and not self._exit():
                missing_items = [it[idx - 1] for idx in missing_indices]
                retry_payload = _encode_indexed_batch(missing_items)
                retry_res = self._translate_batch(
                    retry_payload, context_lines=context_lines
                )
                if not self._exit() and retry_res is not None and retry_res.strip():
                    retry_parsed = _parse_indexed_response(
                        retry_res, len(missing_items)
                    )
                    if retry_parsed:
                        for k, orig_idx in enumerate(missing_indices, 1):
                            if retry_parsed.get(k, "").strip():
                                parsed[orig_idx] = retry_parsed[k].strip()
                    elif len(missing_items) == 1:
                        single_txt = re.sub(
                            r"^\[\d+\]\s*", "", retry_res.strip()
                        ).strip()
                        if single_txt:
                            parsed[missing_indices[0]] = single_txt

            batch_items: List[SrtItem] = []
            for idx, item in enumerate(it, 1):
                copied = copy.deepcopy(item)
                copied["text"] = parsed.get(idx, "").strip()
                batch_items.append(copied)

            if not from_cache and any(x["text"] for x in batch_items):
                canonical = "\n".join(
                    f"[{idx}] {x['text']}" for idx, x in enumerate(batch_items, 1)
                )
                self._set_cache(cache_input, canonical)
            return batch_items

        # Legacy full-SRT mode (aitrans_compact=False)
        srt_str = "\n\n".join(
            [
                f"{srt_dict['line']}\n{srt_dict['time']}\n{srt_dict['text'].strip()}"
                for srt_dict in it
            ]
        )
        cache_input = self._build_cache_key_input(srt_str, context_lines)
        cached = self._get_cache(cache_input)
        from_cache = cached is not None and bool(str(cached).strip())
        if from_cache:
            result = str(cached)
        else:
            result = self._translate_batch(srt_str, context_lines=context_lines)
            if self._exit() or result is None:
                return None
            if not str(result).strip():
                raise TranslateSrtError(
                    tr("Translate result is empty") + f"\n{self.api_url}"
                )
            self._set_cache(cache_input, result)

        parsed_items = get_subtitle_from_srt(result, is_file=False)
        if len(parsed_items) == len(it):
            batch_items = []
            for idx, src in enumerate(it):
                copied = copy.deepcopy(src)
                copied["text"] = parsed_items[idx]["text"]
                batch_items.append(copied)
            return batch_items

        time_map = {x["time"]: x["text"] for x in parsed_items if x.get("time")}
        line_map = {
            x["line"]: x["text"] for x in parsed_items if x.get("line") is not None
        }
        batch_items = []
        for src in it:
            copied = copy.deepcopy(src)
            copied["text"] = time_map.get(
                src.get("time", ""), line_map.get(src.get("line"), "")
            )
            batch_items.append(copied)
        return batch_items

    # Send full subtitle format content for translation
    def _run_srt(self, split_source_text: List[List[SrtItem]]):
        total_batches = len(split_source_text)
        if total_batches == 0:
            return []
        concurrency = self._effective_concurrency(total_batches)
        logger.debug(
            f"以SRT字幕块翻译，每次翻译 {self.trans_thread} 条字幕块，并发{concurrency}，翻译后暂停{self.wait_sec}s"
        )
        from videotrans.configure.excepts import TranslateSrtError

        batch_results: List[Optional[List[SrtItem]]] = [None] * total_batches
        if concurrency == 1:
            rolling_context: List[str] = []
            for i, it in enumerate(split_source_text):
                if self._exit():
                    return []
                self.signal(text=tr("starttrans") + f" {i} ")
                ctx = (
                    rolling_context[-self.context_window :]
                    if (
                        self.aitrans_context
                        and self.context_window > 0
                        and rolling_context
                    )
                    else None
                )
                batch_items = self._process_srt_batch(it, context_lines=ctx)
                if self._exit() or batch_items is None:
                    return []
                batch_results[i] = batch_items
                for item in batch_items:
                    txt = str(item.get("text", "")).strip()
                    if txt:
                        rolling_context.append(txt)
                if self.context_window > 0 and len(rolling_context) > self.context_window:
                    rolling_context = rolling_context[-self.context_window :]
                srt_preview = "\n\n".join(
                    f"{x['line']}\n{x['time']}\n{x['text']}" for x in batch_items
                )
                self.signal(text=srt_preview, type="subtitle")
                with self._progress_lock:
                    self._emit_batch_progress(i + 1, total_batches)
                if self.wait_sec > 0 and total_batches > 1 and i < total_batches - 1:
                    if self._sleep_interruptible(self.wait_sec):
                        return []
        else:
            completed_count = 0
            pool = ThreadPoolExecutor(max_workers=concurrency)
            cancelled_or_error = False
            try:
                future_to_idx = {
                    pool.submit(self._process_srt_batch, it, None): idx
                    for idx, it in enumerate(split_source_text)
                }
                pending = set(future_to_idx.keys())
                while pending:
                    if self._exit():
                        cancelled_or_error = True
                        return []
                    done, pending = wait(
                        pending, timeout=0.2, return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        idx = future_to_idx[fut]
                        res = fut.result()
                        if self._exit() or res is None:
                            cancelled_or_error = True
                            self._run_abort_event.set()
                            return []
                        batch_results[idx] = res
                        with self._progress_lock:
                            completed_count += 1
                            self._emit_batch_progress(completed_count, total_batches)
            except BaseException:
                cancelled_or_error = True
                self._run_abort_event.set()
                raise
            finally:
                if cancelled_or_error:
                    self._run_abort_event.set()
                pool.shutdown(
                    wait=not cancelled_or_error, cancel_futures=cancelled_or_error
                )

            if self._exit():
                return []
            for batch_items in batch_results:
                if batch_items:
                    srt_preview = "\n\n".join(
                        f"{x['line']}\n{x['time']}\n{x['text']}" for x in batch_items
                    )
                    self.signal(text=srt_preview, type="subtitle")

        raws_list: List[SrtItem] = []
        for batch_items in batch_results:
            if batch_items:
                raws_list.extend(batch_items)

        _empty_line = 0
        for it in raws_list:
            if not it["text"].strip():
                _empty_line += 1
        if not raws_list or _empty_line >= len(raws_list):
            raise TranslateSrtError(tr("Translate result is empty") + f"\n{self.api_url}")
        logger.debug(
            f"原始字幕行数：{len(self.text_list or [])}, 翻译后行数:{len(raws_list)}"
        )
        return raws_list

    def _set_cache(self, it, res_str):
        if not res_str or not str(res_str).strip():
            return
        cache_dir = Path(TEMP_ROOT) / "translate_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_cache = cache_dir / f"{self._get_key(it)}.txt"
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="tr_", suffix=".tmp", dir=str(cache_dir)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(res_str)
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.warning(f"Failed to write translation cache temp file {file_cache}: {e}")
            return

        with _CACHE_IO_LOCK:
            last_err: Optional[Exception] = None
            for attempt in range(4):
                try:
                    os.replace(tmp_path, file_cache)
                    return
                except (PermissionError, OSError) as e:
                    last_err = e
                    try:
                        if file_cache.exists() and file_cache.stat().st_size > 0:
                            break
                    except OSError:
                        pass
                    if attempt < 3:
                        time.sleep(0.015 * (attempt + 1))
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if last_err is not None:
                logger.warning(
                    f"Skipping translation cache write after error for {file_cache}: {last_err}"
                )

    def _get_cache(self, it) -> Union[str, None]:
        if self.is_test:
            return None
        file_cache = Path(TEMP_ROOT) / "translate_cache" / f"{self._get_key(it)}.txt"
        with _CACHE_IO_LOCK:
            for attempt in range(4):
                try:
                    if not file_cache.exists():
                        return None
                    content = file_cache.read_text(encoding="utf-8")
                    if content:
                        logger.debug("本次跳过翻译，使用缓存")
                    return content
                except (PermissionError, OSError) as e:
                    if attempt < 3:
                        time.sleep(0.015 * (attempt + 1))
                        continue
                    logger.warning(
                        f"Translation cache read error for {file_cache}, treating as cache miss: {e}"
                    )
                    return None
        return None

    def _get_key(self, it) -> str:
        if isinstance(it, dict):
            it_str = json.dumps(it, sort_keys=True, ensure_ascii=False)
        else:
            it_str = serial(it)
        key_str = f"{self.translate_type}-{self.api_url}-{self.aisendsrt}-{self.model_name}-{self.source_code}-{self.target_code}-{it_str}"
        return get_md5(key_str)
