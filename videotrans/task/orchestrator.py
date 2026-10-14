"""UI-independent orchestration for one noninteractive video task."""

from __future__ import annotations

import shutil
import threading
import traceback
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from videotrans.configure.config import TEMP_DIR, logger
from videotrans.configure.excepts import get_msg_from_except
from videotrans.task.taskcfg import TaskCfgVTT
from videotrans.task.trans_create import TransCreate
from videotrans.util._ffmpeg_misc import format_video


class EventKind(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    LOG = "log"
    PROGRESS = "progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskEvent:
    job_id: str
    kind: EventKind
    stage: str | None = None
    message: str = ""
    progress: float | None = None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskFailure:
    code: str
    stage: str
    message: str
    cause: str
    retryable: bool = False


@dataclass(frozen=True)
class TaskResult:
    job_id: str
    status: TaskStatus
    output_dir: Path
    outputs: tuple[Path, ...] = ()
    failure: TaskFailure | None = None
    segments: tuple[dict[str, Any], ...] = ()
    asr_duration: float | None = None
    transcript_options: dict[str, Any] | None = None


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class TaskRequest:
    params: Mapping[str, object]

    def normalize(self) -> dict:
        values = dict(self.params)
        if "source_code" in values and "source_language_code" not in values:
            values["source_language_code"] = values.pop("source_code")
        elif "source_code" in values:
            values.pop("source_code")
        if "target_code" in values and "target_language_code" not in values:
            values["target_language_code"] = values.pop("target_code")
        elif "target_code" in values:
            values.pop("target_code")

        unknown = set(values) - {item.name for item in fields(TaskCfgVTT)}
        if unknown:
            raise ValueError(f"Unsupported task parameters: {', '.join(sorted(unknown))}")

        raw_name = values.get("name")
        if not raw_name:
            raise ValueError("Input file is required")
        source = Path(raw_name).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Input file does not exist: {source}")

        file_info = format_video(source.as_posix())
        for key in ("name", "dirname", "basename", "noextname", "ext", "uuid"):
            values[key] = getattr(file_info, key)

        target_dir = values.get("target_dir")
        if not target_dir:
            raise ValueError("Output directory is required")
        output = Path(target_dir).expanduser().resolve()
        if output == Path(output.anchor):
            raise ValueError("Output directory cannot be a filesystem root")
        values["target_dir"] = output.as_posix()

        cache = Path(values.get("cache_folder") or Path(TEMP_DIR) / file_info.uuid).expanduser().resolve()
        cache_roots = [Path(TEMP_DIR).resolve()]
        try:
            from videotrans.configure import config as _cfg
            if hasattr(_cfg, "TEMP_DIR") and _cfg.TEMP_DIR:
                cache_roots.append(Path(_cfg.TEMP_DIR).resolve())
            from videotrans.configure._paths import TEMP_ROOT
            if TEMP_ROOT:
                cache_roots.append(Path(TEMP_ROOT).resolve())
        except Exception:
            pass

        if not any(cache == root or root in cache.parents for root in cache_roots):
            raise ValueError("Cache directory must be inside the application temp directory")
        values["cache_folder"] = cache.as_posix()
        return values


EventSink = Callable[[TaskEvent], None]


def run(
    request: TaskRequest,
    event_sink: EventSink | None = None,
    cancellation_token: CancellationToken | None = None,
    stage_limit: str | None = None,
    stop_after_stage: str | None = None,
) -> TaskResult:
    """Validate and execute one task, optionally stopping after a completed stage."""
    emit = event_sink or (lambda _event: None)
    token = cancellation_token or CancellationToken()
    stage = "validation"
    job_id = ""
    output_dir = Path(".")
    terminal_emitted = False

    def send(kind: EventKind, message: str = "", *, progress=None, details=None) -> None:
        nonlocal terminal_emitted
        if kind in {EventKind.SUCCEEDED, EventKind.FAILED, EventKind.CANCELLED}:
            if terminal_emitted:
                return
            terminal_emitted = True
        emit(TaskEvent(job_id, kind, stage, message, progress, details or {}))

    try:
        values = request.normalize()
        job_id = str(values["uuid"])
        output_dir = Path(str(values["target_dir"]))
        cache_dir = Path(str(values["cache_folder"]))
        send(EventKind.QUEUED)
        if token.is_cancelled():
            send(EventKind.CANCELLED, "Task cancelled before execution")
            return TaskResult(job_id, TaskStatus.CANCELLED, output_dir)

        if values.pop("clear_cache", False) and cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        values["clear_cache"] = False

        asr_duration_val: float | None = None

        def relay(raw: dict) -> None:
            nonlocal asr_duration_val
            raw_type = raw.get("type", "logs")
            if raw_type == "set_precent":
                text, _, percent = str(raw.get("text", "")).partition("???")
                send(EventKind.PROGRESS, text, progress=float(percent or 0))
            elif raw_type == "asr_timing":
                dur = raw.get("duration")
                if dur is not None:
                    try:
                        asr_duration_val = float(dur)
                    except (ValueError, TypeError):
                        pass
                send(EventKind.LOG, str(raw.get("text", "")), details={"source_type": raw_type, "duration": asr_duration_val})
            elif raw_type in {"logs", "subtitle", "replace_subtitle"}:
                send(EventKind.LOG, str(raw.get("text", "")), details={"source_type": raw_type})

        task = TransCreate(
            cfg=TaskCfgVTT(**values),
            event_sink=relay,
            cancellation_token=token,
        )
        if task.cfg.subtitles:
            # Stage 4 is authoritative for edited text/timing. Reuse the prepared
            # media/cache and enter the existing dubbing/assembly stages directly.
            task.cfg.clear_cache = False
            Path(task.cfg.source_sub).parent.mkdir(parents=True, exist_ok=True)
            Path(task.cfg.source_sub).write_text(task.cfg.subtitles, encoding="utf-8")
            Path(task.cfg.target_sub).write_text(task.cfg.subtitles, encoding="utf-8")
            from videotrans.util.help_srt import get_subtitle_from_srt
            task.source_srt_list = get_subtitle_from_srt(task.cfg.source_sub, is_file=True) or []
            task.target_srt_list = get_subtitle_from_srt(task.cfg.target_sub, is_file=True) or []
            task.should_recogn = False
            task.should_trans = False
        send(EventKind.RUNNING)

        if stage_limit == "asr":
            stages = [("prepare", True), ("recogn", task.should_recogn),
                      ("diariz", task.should_recogn)]
        else:
            stages = [("prepare", True), ("recogn", task.should_recogn),
                      ("diariz", task.should_recogn), ("trans", task.should_trans),
                      ("dubbing", task.should_dubbing), ("align", task.should_dubbing),
                      ("recogn2pass", task.should_recogn2),
                      ("assembling", task.should_hebing), ("task_done", True)]

        for stage_name, enabled in stages:
            if not enabled:
                continue
            stage = stage_name
            if token.is_cancelled():
                send(EventKind.CANCELLED, "Task cancelled")
                return TaskResult(job_id, TaskStatus.CANCELLED, output_dir)

            if stage_name == "recogn" and task.should_recogn and getattr(task.cfg, "project_id", None) and not task.cfg.force_recogn:
                from videotrans.core.content_hash import compute_content_hash
                from videotrans.core.project_store import find_project_by_audio_hash
                audio_path = getattr(task.cfg, "shibie_audio", None) or getattr(task.cfg, "name", None)
                if audio_path and Path(audio_path).is_file():
                    h = compute_content_hash(audio_path)
                    if h:
                        cached = find_project_by_audio_hash(h)
                        if cached and cached.get("id") != task.cfg.project_id and cached.get("state", {}).get("segments"):
                            cached_segs = cached["state"]["segments"]
                            send(EventKind.LOG, f"Reusing transcript from project {cached['id']} (content hash {h[:8]}...)")
                            task.source_srt_list = [
                                {
                                    "line": seg.get("id", i + 1),
                                    "time": f"{seg.get('startTime', '00:00:00,000')} --> {seg.get('endTime', '00:00:00,000')}",
                                    "text": seg.get("sourceText") or seg.get("text", "")
                                }
                                for i, seg in enumerate(cached_segs)
                            ]
                            task.should_recogn = False

            send(EventKind.STAGE_STARTED)
            getattr(task, stage_name)()
            if token.is_cancelled():
                send(EventKind.CANCELLED, "Task cancelled")
                return TaskResult(job_id, TaskStatus.CANCELLED, output_dir)
            send(EventKind.STAGE_COMPLETED)
            _persist_stage_artifacts(task, stage_name)
            if stage_name == stop_after_stage:
                break

        _embed_thumbnail(task.cfg)
        outputs = _collect_outputs(task.cfg)
        segments = extract_transcript_segments(task)
        transcript_options = extract_transcript_options(task)
        if stage_limit == "asr" and segments and getattr(task.cfg, "project_id", None):
            import json
            from videotrans.core.project_store import init_project_dirs
            baseline = init_project_dirs(task.cfg.project_id)["transcripts"] / "asr_baseline.json"
            baseline.write_text(json.dumps({"segments": segments, "transcriptOptions": transcript_options}, ensure_ascii=False), encoding="utf-8")
        if asr_duration_val is None:
            asr_duration_val = getattr(task, "asr_duration", None)
        detected_lang = getattr(task, "detect_language", None) or getattr(task.cfg, "source_code", None)
        send(
            EventKind.SUCCEEDED,
            details={
                "outputs": tuple(map(str, outputs)),
                "segments": list(segments),
                "transcript_options": transcript_options,
                "asr_duration": asr_duration_val,
                "detected_language": detected_lang,
                "source_language": detected_lang,
                "target_language": getattr(task.cfg, "target_code", None),
            },
        )
        return TaskResult(
            job_id,
            TaskStatus.SUCCEEDED,
            output_dir,
            outputs,
            segments=tuple(segments),
            asr_duration=asr_duration_val,
            transcript_options=transcript_options,
        )
    except Exception as exc:
        cause = "".join(traceback.format_exception(exc))
        logger.exception("Task %s failed during %s", job_id, stage, exc_info=True)
        failure = TaskFailure(
            code=f"{stage}_failed",
            stage=stage,
            message=get_msg_from_except(exc) or str(exc),
            cause=cause,
        )
        try:
            send(EventKind.FAILED, failure.message, details={"code": failure.code})
        except Exception:
            logger.exception("Failed to send terminal failure event for task %s", job_id, exc_info=True)
        return TaskResult(job_id, TaskStatus.FAILED, output_dir, failure=failure)


def _collect_outputs(cfg: TaskCfgVTT) -> tuple[Path, ...]:
    target_dir = Path(cfg.target_dir)
    candidates = [cfg.targetdir_mp4, cfg.source_sub, cfg.target_sub,
                  cfg.source_wav_output, cfg.target_wav_output,
                  target_dir / f"{cfg.target_language_code}-dubbing.wav",
                  target_dir / f"{cfg.target_language_code}-slowed.mp4",
                  target_dir / f"{cfg.target_language_code}-{cfg.noextname}.wav"]
    if cfg.only_out_mp4:
        candidates.append(target_dir.parent / Path(cfg.targetdir_mp4).name)
    output_dir = Path(cfg.target_dir).resolve()
    found = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        is_task_output = output_dir in path.parents or (
            cfg.only_out_mp4 and path.parent == output_dir.parent
        )
        if path.is_file() and is_task_output:
            found.append(path)
    return tuple(dict.fromkeys(found))


def _embed_thumbnail(cfg: TaskCfgVTT) -> None:
    """Attach the selected cover image without re-encoding the rendered streams."""
    thumbnail = Path(cfg.thumbnail).resolve() if cfg.thumbnail else None
    video = Path(cfg.targetdir_mp4).resolve() if cfg.targetdir_mp4 else None
    if not thumbnail or not thumbnail.is_file() or not video or not video.is_file():
        return
    from videotrans.util.help_ffmpeg import runffmpeg
    staged = Path(cfg.cache_folder) / f"thumbnail-{video.name}"
    runffmpeg([
        '-y', '-i', video.as_posix(), '-i', thumbnail.as_posix(),
        '-map', '0', '-map', '1', '-c', 'copy', '-c:v:1', 'mjpeg',
        '-disposition:v:1', 'attached_pic', staged.as_posix(),
    ])
    if staged.is_file():
        staged.replace(video)


def _persist_stage_artifacts(task: Any, stage_name: str) -> None:
    """Store intermediate stage outputs in output/projects/{project_id}/ and update project state."""
    pid = getattr(task.cfg, "project_id", None)
    if not pid:
        return
    try:
        from videotrans.core.project_store import init_project_dirs, update_project_state, get_project, update_project
        import shutil
        import json
        dirs = init_project_dirs(pid)

        if stage_name == "prepare":
            for attr in ("shibie_audio", "vocal_file", "novocal_file", "source_wav"):
                src = getattr(task.cfg, attr, None)
                if src and Path(src).is_file():
                    dst = dirs["media"] / Path(src).name
                    if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
                        shutil.copy2(src, dst)
            p = get_project(pid)
            if p and not p.get("audio_hash"):
                from videotrans.core.content_hash import compute_content_hash
                ref_audio = getattr(task.cfg, "shibie_audio", None) or getattr(task.cfg, "name", None)
                if ref_audio and Path(ref_audio).is_file():
                    h = compute_content_hash(ref_audio)
                    if h:
                        update_project(pid, audio_hash=h)

        elif stage_name in ("recogn", "diariz"):
            if getattr(task.cfg, "source_sub", None) and Path(task.cfg.source_sub).is_file():
                shutil.copy2(task.cfg.source_sub, dirs["transcripts"] / "source.srt")
            speaker_file = Path(task.cfg.cache_folder) / "speaker.json"
            if speaker_file.is_file():
                shutil.copy2(speaker_file, dirs["transcripts"] / "speaker.json")
            opts_file = Path(task.cfg.cache_folder) / "transcript_options.json"
            if opts_file.is_file():
                shutil.copy2(opts_file, dirs["transcripts"] / "transcript_options.json")
            segments = extract_transcript_segments(task)
            transcript_options = extract_transcript_options(task)
            if segments:
                (dirs["transcripts"] / "segments.json").write_text(json.dumps(list(segments), ensure_ascii=False, indent=2), encoding="utf-8")
                p = get_project(pid)
                cur_state = dict(p.get("state") or {}) if p else {}
                cur_state["segments"] = list(segments)
                if stage_name == "diariz":
                    cur_state["forceAsr"] = False
                if transcript_options:
                    cur_state["transcript_options"] = transcript_options
                    cur_state["selected_segment_option"] = cur_state.get("selected_segment_option", "utterances")
                src_code = getattr(task, "detect_language", None) or getattr(task.cfg, "source_code", None)
                if src_code:
                    if "languages" not in cur_state:
                        cur_state["languages"] = {}
                    cur_state["languages"]["source"] = {
                        "code": src_code,
                        "name": src_code,
                        "flag": "",
                        "autoDetected": bool(getattr(task, "detect_language", None)),
                    }
                if getattr(task.cfg, "target_code", None):
                    if "languages" not in cur_state:
                        cur_state["languages"] = {}
                    cur_state["languages"]["target"] = {
                        "code": getattr(task.cfg, "target_code"),
                        "name": getattr(task.cfg, "target_code"),
                    }
                update_project_state(pid, cur_state, stage=2, status="completed" if stage_name == "diariz" else "processing")

        elif stage_name == "trans":
            if getattr(task.cfg, "target_sub", None) and Path(task.cfg.target_sub).is_file():
                shutil.copy2(task.cfg.target_sub, dirs["transcripts"] / "target.srt")
            segments = extract_transcript_segments(task)
            if segments:
                (dirs["transcripts"] / "segments.json").write_text(json.dumps(list(segments), ensure_ascii=False, indent=2), encoding="utf-8")
                p = get_project(pid)
                cur_state = dict(p.get("state") or {}) if p else {}
                cur_state["segments"] = list(segments)
                update_project_state(pid, cur_state, stage=3, status="completed")

        elif stage_name in ("dubbing", "align"):
            queue_file = Path(task.cfg.cache_folder) / "queue_tts.json"
            if queue_file.is_file():
                shutil.copy2(queue_file, dirs["dubbing"] / "queue_tts.json")
            if getattr(task.cfg, "target_wav", None) and Path(task.cfg.target_wav).is_file():
                shutil.copy2(task.cfg.target_wav, dirs["dubbing"] / Path(task.cfg.target_wav).name)

        elif stage_name in ("assembling", "task_done"):
            p = get_project(pid)
            cur_state = dict(p.get("state") or {}) if p else {}
            parent_id = p.get("parent_id") if p else None
            if parent_id:
                from videotrans.core.project_store import get_parent_project_dir
                import re
                parent = get_project(parent_id)
                parent_name = (parent.get("name") if parent else None) or parent_id
                parent_dir = get_parent_project_dir(parent_name)
                safe_stem = re.sub(r"[^\w.-]+", "-", Path(p.get("name") or pid).stem, flags=re.UNICODE).strip("-")
                if getattr(task.cfg, "targetdir_mp4", None) and Path(task.cfg.targetdir_mp4).is_file():
                    shutil.copy2(task.cfg.targetdir_mp4, parent_dir / f"{safe_stem}.mp4")
                if getattr(task.cfg, "target_sub", None) and Path(task.cfg.target_sub).is_file():
                    shutil.copy2(task.cfg.target_sub, parent_dir / f"{safe_stem}.srt")
            update_project_state(pid, cur_state, stage=4, status="completed")

    except Exception as exc:
        logger.warning("Failed to persist stage %s artifacts for project %s: %s", stage_name, pid, exc)


def run_staged_asr(
    request: TaskRequest,
    event_sink: EventSink | None = None,
    cancellation_token: CancellationToken | None = None,
) -> TaskResult:
    """Execute audio extraction, speech recognition (ASR), and speaker diarization."""
    return run(request, event_sink, cancellation_token, stage_limit="asr")


def run_staged_translation(
    request: TaskRequest,
    event_sink: EventSink | None = None,
    cancellation_token: CancellationToken | None = None,
) -> TaskResult:
    """Execute LLM / machine translation of transcript segments with 1:1 matching."""
    from videotrans import translator
    from videotrans.util.help_srt import ms_to_time_string
    from videotrans.util.segment_ops import calculate_cps

    emit = event_sink or (lambda _event: None)
    token = cancellation_token or CancellationToken()
    stage = "trans"
    job_id = ""
    output_dir = Path(".")
    terminal_emitted = False

    def send(kind: EventKind, message: str = "", *, progress=None, details=None) -> None:
        nonlocal terminal_emitted
        if kind in {EventKind.SUCCEEDED, EventKind.FAILED, EventKind.CANCELLED}:
            if terminal_emitted:
                return
            terminal_emitted = True
        emit(TaskEvent(job_id, kind, stage, message, progress, details or {}))

    def relay(raw: dict) -> None:
        send(
            EventKind.LOG,
            str(raw.get("text", "")),
            details={"source_type": raw.get("type", "logs")},
        )

    try:
        values = request.normalize()
        job_id = str(values.get("uuid") or "trans-job")
        target_dir = values.get("target_dir")
        output_dir = Path(str(target_dir)) if target_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)

        send(EventKind.QUEUED, "Queued for translation")
        if token.is_cancelled():
            send(EventKind.CANCELLED, "Task cancelled before execution")
            return TaskResult(job_id, TaskStatus.CANCELLED, output_dir)

        send(EventKind.RUNNING, "Starting translation")
        send(EventKind.STAGE_STARTED, "Translating segments")

        raw_segments = list(values.get("segments") or request.params.get("segments") or [])
        source_code = str(values.get("source_language_code") or values.get("source_language") or "zh-cn")
        target_code = str(values.get("target_language_code") or values.get("target_language") or "en")
        translate_type = int(values.get("translate_type", 0) or 0)
        aisendsrt = values.get("aisendsrt")

        # Bypass when source and target languages are identical
        if source_code.lower() == target_code.lower():
            updated_segments = []
            for seg in raw_segments:
                seg_copy = dict(seg)
                if not seg_copy.get("targetText"):
                    seg_copy["targetText"] = seg_copy.get("sourceText") or seg_copy.get("text") or ""
                seg_copy["text"] = seg_copy["targetText"]
                dur = max(0.1, float(seg_copy.get("endSec", 0.0) or 0.0) - float(seg_copy.get("startSec", 0.0) or 0.0))
                cps, cps_status = calculate_cps(seg_copy["targetText"], dur)
                seg_copy["targetCps"] = cps
                seg_copy["cps"] = cps
                seg_copy["cpsStatus"] = cps_status
                updated_segments.append(seg_copy)

            send(EventKind.STAGE_COMPLETED, "Translation bypassed (same language)")
            send(EventKind.PROGRESS, "Complete", progress=100.0)
            send(EventKind.SUCCEEDED, "Translation complete", details={"segments": updated_segments})
            return TaskResult(job_id, TaskStatus.SUCCEEDED, output_dir, segments=tuple(updated_segments))

        if not raw_segments:
            send(EventKind.STAGE_COMPLETED, "No segments to translate")
            send(EventKind.PROGRESS, "Complete", progress=100.0)
            send(EventKind.SUCCEEDED, "Translation complete", details={"segments": []})
            return TaskResult(job_id, TaskStatus.SUCCEEDED, output_dir, segments=())

        # Construct SrtItem-compatible structures for translator.run
        text_list = []
        for idx, seg in enumerate(raw_segments, 1):
            src_text = str(seg.get("sourceText") if seg.get("sourceText") is not None else (seg.get("text") or "")).strip()
            start_sec = float(seg.get("startSec", 0.0) or 0.0)
            end_sec = float(seg.get("endSec", 0.0) or 0.0)
            start_ms = int(round(start_sec * 1000))
            end_ms = int(round(end_sec * 1000))
            start_str = ms_to_time_string(ms=start_ms)
            end_str = ms_to_time_string(ms=end_ms)
            text_list.append({
                "line": idx,
                "time": f"{start_str} --> {end_str}",
                "text": src_text,
                "start_time": start_ms,
                "end_time": end_ms,
                "startraw": start_str,
                "endraw": end_str,
            })

        if token.is_cancelled():
            send(EventKind.CANCELLED, "Task cancelled")
            return TaskResult(job_id, TaskStatus.CANCELLED, output_dir)

        send(EventKind.PROGRESS, "Translating segments with selected provider...", progress=35.0)

        translated_items = translator.run(
            translate_type=translate_type,
            text_list=text_list,
            source_code=source_code,
            target_code=target_code,
            aisendsrt=aisendsrt,
            uuid=job_id,
            event_sink=relay,
            cancellation_token=token,
        )

        if token.is_cancelled():
            send(EventKind.CANCELLED, "Task cancelled")
            return TaskResult(job_id, TaskStatus.CANCELLED, output_dir)

        # 1:1 segment matching preserving all segment metadata
        updated_segments = []
        for idx, seg in enumerate(raw_segments):
            seg_copy = dict(seg)
            translated_text = ""
            if translated_items and idx < len(translated_items):
                item = translated_items[idx]
                if isinstance(item, dict):
                    translated_text = str(item.get("text") or "").strip()
                elif hasattr(item, "text"):
                    translated_text = str(getattr(item, "text", "") or "").strip()
                elif isinstance(item, str):
                    translated_text = item.strip()

            if translated_text:
                seg_copy["targetText"] = translated_text
                seg_copy["text"] = translated_text
            elif not seg_copy.get("targetText"):
                seg_copy["targetText"] = seg_copy.get("sourceText") or seg_copy.get("text") or ""

            dur = max(0.1, float(seg_copy.get("endSec", 0.0) or 0.0) - float(seg_copy.get("startSec", 0.0) or 0.0))
            target_cps, target_cps_status = calculate_cps(seg_copy["targetText"], dur)
            seg_copy["targetCps"] = target_cps
            seg_copy["cps"] = target_cps
            seg_copy["cpsStatus"] = target_cps_status
            updated_segments.append(seg_copy)

        send(EventKind.PROGRESS, "Translation complete", progress=100.0)
        send(EventKind.STAGE_COMPLETED, "Translation stage finished")

        pid = values.get("project_id")
        if pid:
            try:
                from videotrans.core.project_store import init_project_dirs, update_project_state, get_project
                import json
                dirs = init_project_dirs(str(pid))
                (dirs["transcripts"] / "segments.json").write_text(
                    json.dumps(updated_segments, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                p = get_project(str(pid))
                cur_state = dict(p.get("state") or {}) if p else {}
                cur_state["segments"] = updated_segments
                update_project_state(str(pid), cur_state, stage=3, status="completed")
            except Exception as e:
                logger.warning("Failed to persist translation stage artifacts for project %s: %s", pid, e)

        send(EventKind.SUCCEEDED, "Translation succeeded", details={"segments": updated_segments})
        return TaskResult(job_id, TaskStatus.SUCCEEDED, output_dir, segments=tuple(updated_segments))

    except Exception as exc:
        cause = "".join(traceback.format_exception(exc))
        logger.exception("Translation task %s failed during %s", job_id, stage, exc_info=True)
        failure = TaskFailure(
            code="trans_failed",
            stage=stage,
            message=get_msg_from_except(exc) or str(exc),
            cause=cause,
        )
        try:
            send(EventKind.FAILED, failure.message, details={"code": failure.code})
        except Exception:
            pass
        return TaskResult(job_id, TaskStatus.FAILED, output_dir, failure=failure)


def _get_speaker_list(task: object) -> list[str]:
    import json
    cache_folder = getattr(getattr(task, "cfg", None), "cache_folder", None)
    target_dir = getattr(getattr(task, "cfg", None), "target_dir", None)
    speaker_list: list[str] = []
    for spk_path in (
        Path(cache_folder) / "speaker.json" if cache_folder else None,
        Path(target_dir) / "speaker.json" if target_dir else None,
    ):
        if spk_path and spk_path.is_file():
            try:
                loaded = json.loads(spk_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    speaker_list = [str(x) for x in loaded]
                    break
                elif isinstance(loaded, dict):
                    speaker_list = [
                        str(v)
                        for _, v in sorted(
                            loaded.items(),
                            key=lambda p: int(p[0]) if str(p[0]).isdigit() else p[0],
                        )
                    ]
                    break
            except Exception:
                pass
    return speaker_list


def _raw_items_to_segments(
    raw_items: list[Any],
    speaker_list: list[str] | None = None,
    enable_diariz: bool = False,
) -> list[dict[str, Any]]:
    """Convert raw subtitle items into normalized frontend Segment objects with start/end timing."""
    from typing import Any
    from videotrans.util.segment_ops import calculate_cps, format_timestamp, parse_timestamp

    if not raw_items:
        return []

    speaker_list = speaker_list or []
    raw_spk_per_item = []
    for idx, item in enumerate(raw_items):
        item_dict = item if isinstance(item, dict) else dict(item.items()) if hasattr(item, "items") else getattr(item, "__dict__", {})
        spk = item_dict.get("spk") or getattr(item, "spk", None)
        if not spk and idx < len(speaker_list) and speaker_list[idx]:
            spk = speaker_list[idx]
        raw_spk_per_item.append(str(spk) if spk else "")

    distinct_speakers = [s for s in dict.fromkeys(raw_spk_per_item) if s]

    palette = [
        {"id": "spk_1", "name": "Speaker 1", "code": "S1", "color": "amber"},
        {"id": "spk_2", "name": "Speaker 2", "code": "S2", "color": "secondary"},
        {"id": "spk_3", "name": "Speaker 3", "code": "S3", "color": "emerald"},
        {"id": "spk_4", "name": "Speaker 4", "code": "S4", "color": "rose"},
        {"id": "spk_5", "name": "Speaker 5", "code": "S5", "color": "purple"},
    ]

    spk_map: dict[str, dict[str, str]] = {}
    if enable_diariz and len(distinct_speakers) > 1:
        for i, spk_key in enumerate(distinct_speakers):
            if i < len(palette):
                spk_map[spk_key] = palette[i]
            else:
                spk_map[spk_key] = {
                    "id": f"spk_{i+1}",
                    "name": f"Speaker {i+1}",
                    "code": f"S{i+1}",
                    "color": "stone",
                }
    else:
        single = palette[0]
        for spk_key in distinct_speakers:
            spk_map[spk_key] = single

    segments = []
    for idx, item in enumerate(raw_items):
        item_dict = item if isinstance(item, dict) else dict(item.items()) if hasattr(item, "items") else getattr(item, "__dict__", {})
        text = str(item_dict.get("text", getattr(item, "text", ""))).strip()

        start_val = item_dict.get("start_time", getattr(item, "start_time", 0))
        end_val = item_dict.get("end_time", getattr(item, "end_time", 0))

        if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)) and (start_val > 0 or end_val > 0):
            start_sec = float(start_val) / 1000.0
            end_sec = float(end_val) / 1000.0
        else:
            startraw = item_dict.get("startraw", getattr(item, "startraw", ""))
            endraw = item_dict.get("endraw", getattr(item, "endraw", ""))
            start_sec = parse_timestamp(str(startraw))
            end_sec = parse_timestamp(str(endraw))

        start_time_str = format_timestamp(start_sec)
        end_time_str = format_timestamp(end_sec)
        dur = max(0.1, end_sec - start_sec)
        cps, cps_status = calculate_cps(text, dur)

        raw_spk = raw_spk_per_item[idx] if idx < len(raw_spk_per_item) else ""
        spk_info = spk_map.get(raw_spk) or palette[0]

        segments.append({
            "id": idx + 1,
            "speaker": spk_info["name"],
            "speakerId": spk_info["id"],
            "speakerName": spk_info["name"],
            "speakerLabel": spk_info["name"],
            "speakerColor": spk_info["color"],
            "speakerCode": spk_info["code"],
            "startTime": start_time_str,
            "endTime": end_time_str,
            "startSec": round(start_sec, 3),
            "endSec": round(end_sec, 3),
            "text": text,
            "sourceText": text,
            "targetText": "",
            "cps": cps,
            "cpsStatus": cps_status,
            "hasOcrDiff": False,
        })

    return segments


def extract_transcript_segments(task: object) -> list[dict[str, Any]]:
    """Extract and normalize recognized dialogue segments from a completed ASR task."""
    raw_items = getattr(task, "source_srt_list", None) or []
    if not raw_items:
        source_sub = getattr(getattr(task, "cfg", None), "source_sub", None)
        if source_sub and Path(source_sub).is_file():
            try:
                from videotrans.util.help_srt import get_subtitle_from_srt
                raw_items = get_subtitle_from_srt(str(source_sub), is_file=True) or []
            except Exception:
                raw_items = []

    if not raw_items:
        return []

    speaker_list = _get_speaker_list(task)
    enable_diariz = getattr(getattr(task, "cfg", None), "enable_diariz", False)
    return _raw_items_to_segments(raw_items, speaker_list, enable_diariz)


def extract_transcript_options(task: object) -> dict[str, list[dict[str, Any]]] | None:
    """Extract and normalize alternative transcript segmentation options (e.g. Deepgram utterances vs paragraphs)."""
    import json
    cache_folder = getattr(getattr(task, "cfg", None), "cache_folder", None)
    target_dir = getattr(getattr(task, "cfg", None), "target_dir", None)

    raw_options = getattr(task, "transcript_options", None)
    if not isinstance(raw_options, dict):
        raw_options = None
    if not raw_options:
        for opts_path in (
            Path(cache_folder) / "transcript_options.json" if cache_folder else None,
            Path(target_dir) / "transcript_options.json" if target_dir else None,
        ):
            if opts_path and opts_path.is_file():
                try:
                    loaded = json.loads(opts_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        raw_options = loaded
                        break
                except Exception:
                    pass

    if not raw_options or not isinstance(raw_options, dict):
        return None

    speaker_list = _get_speaker_list(task)
    enable_diariz = getattr(getattr(task, "cfg", None), "enable_diariz", False)

    normalized_options: dict[str, list[dict[str, Any]]] = {}
    for key, items in raw_options.items():
        if isinstance(items, (list, tuple)) and items:
            normalized_options[key] = _raw_items_to_segments(list(items), speaker_list, enable_diariz)

    return normalized_options if normalized_options else None
