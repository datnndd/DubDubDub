"""Reset one project workflow stage and its dependent results."""
from __future__ import annotations

import json
from pathlib import Path

from videotrans.core.db import db_conn
from videotrans.core.job_store import events_since
from videotrans.core import project_store
from videotrans.core.project_store import get_project, get_project_dir, update_project_state


class BaselineMissingError(ValueError):
    pass


def _asr_baseline(project_id: str) -> dict | None:
    path = get_project_dir(project_id) / "transcripts" / "asr_baseline.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("segments"), list):
                return data
        except (OSError, ValueError, AttributeError):
            pass
    with db_conn() as conn:
        jobs = conn.execute(
            "SELECT id FROM jobs WHERE project_id = ? AND type = 'asr' AND status = 'succeeded' ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    for job in jobs:
        for event in reversed(events_since(job["id"], limit=500)):
            try:
                payload = json.loads(event["payload"])
                details = payload.get("details") or {}
                if payload.get("kind") == "succeeded" and isinstance(details.get("segments"), list):
                    return {"segments": details["segments"], "transcriptOptions": details.get("transcript_options")}
            except (ValueError, TypeError, AttributeError):
                continue
    return None


def _clear_fields(items: list[dict], *fields: str) -> list[dict]:
    return [{key: value for key, value in item.items() if key not in fields} for item in items]


def _remove_files(folder: Path, names: tuple[str, ...] | None = None) -> None:
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if not path.is_file() or (names is not None and path.name not in names):
            continue
        path.unlink()


def reset_project_stage(project_id: str, stage: int) -> dict:
    if stage not in (1, 2, 3, 4):
        raise ValueError("Stage must be 1, 2, 3, or 4")
    project = get_project(project_id)
    if project is None:
        raise LookupError("Project not found")
    state = dict(project.get("state") or {})
    base = get_project_dir(project_id).resolve()
    roots = (project_store.get_projects_root().resolve(), project_store.DEFAULT_PROJECTS_DIR.resolve())
    if not any(base.is_relative_to(root) and base != root for root in roots):
        raise ValueError("Project artifact path is outside the projects directory")
    config = dict((state.get("backend") or {}).get("config") or {})

    if stage == 2:
        baseline = _asr_baseline(project_id)
        if baseline is None:
            raise BaselineMissingError("Original ASR transcript unavailable. Reset Stage 1 and run transcription again before resetting Stage 2.")
        state["segments"] = baseline["segments"]
        state["transcriptOptions"] = baseline.get("transcriptOptions")
        state["selectedSegmentOption"] = "utterances"
        config.update(translateType=0, translationMode="srt")
        state.pop("ocrCrop", None)

    if stage == 4:
        baseline = state.get("stage3Baseline")
        if isinstance(baseline, list):
            state["segments"] = baseline
        elif state.get("segments") and (project.get("stage", 1) >= 4 or state.get("currentStep", 1) >= 4):
            raise BaselineMissingError("Original Stage 3 result unavailable. Reset and rerun Stage 3 before resetting Stage 4.")
        state["annotations"] = []
        state["burnAnnotations"] = True

    if stage <= 1:
        state["segments"] = []
        state["annotations"] = []
        state["burnAnnotations"] = True
        state["transcriptOptions"] = None
        state["selectedSegmentOption"] = "utterances"
        state["forceAsr"] = True
        state["languages"] = {
            "source": {"code": "zh-cn", "name": "Simplified Chinese", "flag": "", "autoDetected": False},
            "target": {"code": "vi", "name": "Vietnamese"}, "timingMode": "voice",
        }
        state["engines"] = {"speakerDiarization": False, "speakerCount": 0, "ocrSlideEngine": True}
        config.update(recognType=1, modelName="nova-3", useCuda=False, deepgramOptions={
            "utt_split": 0.8, "diarize_model": "latest", "smart_format": True,
            "punctuate": True, "paragraphs": True, "utterances": True, "extra": "",
        })
        config.update(translateType=0, translationMode="srt")

    if stage <= 3:
        state["segments"] = _clear_fields(state.get("segments") or [], "previewAudioUrl", "previewAudioId", "previewSpeedFactor", "previewVoice", "voiceOverride")
        state["speakerVoiceMap"] = {}
        state["segmentVoiceOverrides"] = {}
        state["dubbingStatus"] = "idle"
        state["tuning"] = {"pace": 1.0, "timbreWarmth": 62, "ducking": "85/15"}
        state["autoFitVoiceSpeed"] = True
        state["maxSpeedRate"] = 1.35
        state["stage3Baseline"] = None
        config.update(ttsType=2, voiceRole="")

    state["subtitleStyles"] = None
    state["editVideo"] = None
    state["currentStep"] = stage
    state["backend"] = {**(state.get("backend") or {}), "config": config}

    if stage <= 1:
        _remove_files(base / "media")
        _remove_files(base / "transcripts")
    elif stage == 2:
        _remove_files(base / "transcripts", ("target.srt",))
    if stage <= 3:
        _remove_files(base / "dubbing")

    updated = update_project_state(project_id, state, stage=stage, status="pending", merge=False)
    if stage == 1:
        from videotrans.core.project_store import update_project
        updated = update_project(project_id, audio_hash="")
        _remove_files(base / "transcripts", ("segments.json", "source.srt", "transcript_options.json"))
    return updated
