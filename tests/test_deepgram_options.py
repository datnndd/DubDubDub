# -*- coding: utf-8 -*-
"""Verification of Deepgram ASR preset options and user parameter overrides."""

from unittest.mock import MagicMock, patch
import pytest

from videotrans.recognition._deepgram import DeepgramRecogn, parse_deepgram_options
from videotrans.api.task_params import build_task_params


def test_parse_deepgram_options_formats():
    # Dict passthrough
    assert parse_deepgram_options({"utt_split": 0.8}) == {"utt_split": 0.8}

    # JSON string
    assert parse_deepgram_options('{"diarize_model": "latest", "utt_split": 0.8, "smart_format": true}') == {
        "diarize_model": "latest",
        "utt_split": 0.8,
        "smart_format": True,
    }

    # Query string format
    parsed = parse_deepgram_options("diarize_model=latest&utt_split=0.8&smart_format=true&punctuate=false&keywords=AI")
    assert parsed == {
        "diarize_model": "latest",
        "utt_split": 0.8,
        "smart_format": True,
        "punctuate": False,
        "keywords": "AI",
    }

    # Empty or None
    assert parse_deepgram_options("") == {}
    assert parse_deepgram_options(None) == {}


def test_deepgram_recogn_preset_defaults(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"dummy-audio-bytes")

    rec = DeepgramRecogn(
        audio_file=str(audio_path),
        cache_folder=str(tmp_path),
        model_name="nova-3",
        detect_language="zh-cn",
        max_speakers=2,
    )

    captured_options = {}

    def mock_transcribe_file(payload, options, timeout=600, **kwargs):
        nonlocal captured_options
        to_dict_val = options.to_dict() if callable(getattr(options, "to_dict", None)) else {}
        captured_options = {
            "model": getattr(options, "model", None),
            "language": getattr(options, "language", None),
            "smart_format": getattr(options, "smart_format", None),
            "punctuate": getattr(options, "punctuate", None),
            "paragraphs": getattr(options, "paragraphs", None),
            "utterances": getattr(options, "utterances", None),
            "diarize": getattr(options, "diarize", None),
            "diarize_model": getattr(options, "diarize_model", None),
            "utt_split": getattr(options, "utt_split", None),
            "to_dict": to_dict_val,
        }
        mock_res = MagicMock()
        mock_res.__getitem__.side_effect = lambda key: {"utterances": []} if key == "results" else {}
        return mock_res

    with patch("videotrans.recognition._deepgram.DeepgramClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.listen.rest.v.return_value.transcribe_file = mock_transcribe_file

        rec._exec()

    assert captured_options["model"] == "nova-3"
    assert captured_options["language"] == "zh"
    assert captured_options["smart_format"] is True
    assert captured_options["punctuate"] is True
    assert captured_options["paragraphs"] is True
    assert captured_options["diarize"] is None
    assert captured_options["diarize_model"] == "latest"
    assert captured_options["utt_split"] == 0.8
    assert captured_options["to_dict"]["diarize_model"] == "latest"
    assert "diarize" not in captured_options["to_dict"]


def test_deepgram_recogn_user_overrides(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"dummy-audio-bytes")

    rec = DeepgramRecogn(
        audio_file=str(audio_path),
        cache_folder=str(tmp_path),
        model_name="nova-2",
        detect_language="en",
        max_speakers=-1,
        deepgram_options={
            "utt_split": 1.2,
            "smart_format": False,
            "extra": "keywords=OpenAI,Google&numerals=true",
        },
    )

    captured_options = {}

    def mock_transcribe_file(payload, options, timeout=600, **kwargs):
        nonlocal captured_options
        captured_options = {
            "model": getattr(options, "model", None),
            "language": getattr(options, "language", None),
            "smart_format": getattr(options, "smart_format", None),
            "utt_split": getattr(options, "utt_split", None),
            "diarize": getattr(options, "diarize", None),
            "keywords": getattr(options, "keywords", None),
            "numerals": getattr(options, "numerals", None),
        }
        mock_res = MagicMock()
        return mock_res

    with patch("videotrans.recognition._deepgram.DeepgramClient") as mock_client_cls:
        with patch("videotrans.recognition._deepgram.DeepgramConverter") as mock_conv:
            with patch("videotrans.recognition._deepgram.srt") as mock_srt:
                with patch("videotrans.recognition._deepgram.get_subtitle_from_srt", return_value=[]):
                    mock_client = MagicMock()
                    mock_client_cls.return_value = mock_client
                    mock_client.listen.rest.v.return_value.transcribe_file = mock_transcribe_file

                    rec._exec()

    assert captured_options["model"] == "nova-2"
    assert captured_options["language"] == "en"
    assert captured_options["utt_split"] == 1.2
    assert captured_options["smart_format"] is False
    assert captured_options["keywords"] == "OpenAI,Google"
    assert captured_options["numerals"] is True


def test_task_params_preserves_deepgram_options(tmp_path):
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()

    params = build_task_params(
        source,
        {
            "sourceLanguage": "zh-cn",
            "targetLanguage": "en",
            "recognType": 1,
            "deepgramOptions": {
                "utt_split": 0.8,
                "diarize_model": "latest",
                "extra": "keywords=test",
            },
        },
        job_type="asr",
        temp_dir=str(temp_dir),
    )

    assert params["deepgram_options"] == {
        "utt_split": 0.8,
        "diarize_model": "latest",
        "extra": "keywords=test",
    }


def test_deepgram_extracts_utterances_and_paragraphs_options(tmp_path):
    import json
    from videotrans.task.orchestrator import extract_transcript_options

    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"dummy-audio")

    rec = DeepgramRecogn(
        audio_file=str(audio_path),
        cache_folder=str(tmp_path),
        model_name="nova-3",
        detect_language="en",
        max_speakers=2,
    )

    class MockUtterance:
        def __init__(self, start, end, transcript, speaker=0):
            self.start = start
            self.end = end
            self.transcript = transcript
            self.speaker = speaker

    class MockSentence:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    class MockParagraph:
        def __init__(self, start, end, sentences, speaker=0):
            self.start = start
            self.end = end
            self.sentences = sentences
            self.speaker = speaker

    mock_res = MagicMock()
    mock_res.results.utterances = [
        MockUtterance(0.5, 2.5, "Hello world", 0),
        MockUtterance(2.8, 5.0, "This is second utterance", 1),
    ]
    mock_channel = MagicMock()
    mock_alt = MagicMock()
    mock_alt.paragraphs.paragraphs = [
        MockParagraph(
            0.5,
            5.0,
            [
                MockSentence(0.5, 2.5, "Hello world."),
                MockSentence(2.8, 5.0, "This is second utterance."),
            ],
            speaker=0,
        )
    ]
    mock_channel.alternatives = [mock_alt]
    mock_res.results.channels = [mock_channel]

    with patch("videotrans.recognition._deepgram.DeepgramClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.listen.rest.v.return_value.transcribe_file.return_value = mock_res

        result = rec._exec()

    # Result defaults to utterances
    assert len(result) == 2
    assert result[0]["text"] == "Hello world"
    assert result[0]["start_time"] == 500
    assert result[0]["end_time"] == 2500

    # Verify transcript_options.json exists and contains both options
    opts_file = tmp_path / "transcript_options.json"
    assert opts_file.is_file()
    opts = json.loads(opts_file.read_text(encoding="utf-8"))
    assert "utterances" in opts
    assert "paragraphs" in opts
    assert len(opts["utterances"]) == 2
    assert len(opts["paragraphs"]) == 1
    assert opts["paragraphs"][0]["text"] == "Hello world. This is second utterance."
    assert opts["paragraphs"][0]["start_time"] == 500
    assert opts["paragraphs"][0]["end_time"] == 5000

    # Verify orchestrator extract_transcript_options normalizes to Segments
    task_mock = MagicMock()
    task_mock.cfg.cache_folder = str(tmp_path)
    task_mock.cfg.enable_diariz = True
    normalized = extract_transcript_options(task_mock)
    assert normalized is not None
    assert "utterances" in normalized
    assert "paragraphs" in normalized
    assert len(normalized["utterances"]) == 2
    assert normalized["utterances"][0]["startSec"] == 0.5
    assert normalized["utterances"][0]["endSec"] == 2.5
    assert len(normalized["paragraphs"]) == 1
    assert normalized["paragraphs"][0]["startSec"] == 0.5
    assert normalized["paragraphs"][0]["endSec"] == 5.0


def test_deepgram_custom_model_build_task_params(tmp_path):
    from videotrans import recognition
    from videotrans.api.task_params import build_task_params

    dummy_media = tmp_path / "test.mp4"
    dummy_media.write_bytes(b"dummy")

    # A custom model like nova-3-general, nova-2-meeting, etc. should be accepted without raising ValueError
    params = build_task_params(
        dummy_media,
        {
            "recognType": recognition.Deepgram,
            "modelName": "nova-3-general",
            "sourceLanguage": "zh-cn",
            "targetLanguage": "vi",
        },
        job_type="asr",
        output_dir=tmp_path,
        temp_dir=tmp_path,
    )
    assert params["model_name"] == "nova-3-general"
    assert params["recogn_type"] == recognition.Deepgram


def test_build_task_params_source_language_fallback(tmp_path):
    from videotrans import recognition
    from videotrans.api.task_params import build_task_params

    dummy_media = tmp_path / "test.mp4"
    dummy_media.write_bytes(b"dummy")

    # 'auto' or invalid language code should gracefully fallback to 'zh-cn'
    params = build_task_params(
        dummy_media,
        {
            "recognType": recognition.Deepgram,
            "modelName": "nova-3",
            "sourceLanguage": "auto",
        },
        job_type="asr",
        output_dir=tmp_path,
        temp_dir=tmp_path,
    )
    assert params["source_language_code"] == "zh-cn"


@pytest.mark.asyncio
async def test_jobs_post_deepgram_custom_model_and_media_recovery(tmp_path):
    import httpx
    from videotrans.api.app import create_app
    from videotrans.core.project_store import create_project
    from videotrans import recognition

    dummy_media = tmp_path / "recovered_video.mp4"
    dummy_media.write_bytes(b"fake-video-data")

    # Create project in DB with media_path
    proj = create_project(
        name="Batch Test Video",
        media_path=str(dummy_media),
        duration=10.0,
    )

    app = create_app(
        upload_dir=tmp_path / "uploads",
        media_probe=lambda p: {"time": 10.0, "width": 1920, "height": 1080},
        asr_validator=lambda *_: None,
        translation_validator=lambda *_: None,
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Submit job with projectId pointing to project whose media is not yet in in-memory media_store
        res = await client.post(
            "/api/jobs",
            json={
                "mediaId": "unknown-memory-id",
                "projectId": proj["id"],
                "jobType": "asr",
                "options": {
                    "recognType": recognition.Deepgram,
                    "modelName": "nova-3-general",
                    "sourceLanguage": "zh-cn",
                },
            },
        )
        assert res.status_code == 202, f"Expected 202 Accepted, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["id"]
