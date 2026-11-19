# Execution Plan: UI/UX Remediation, Pipeline Optimization, OCR Paragraph Fix & Video OCR Support

Date: 2026-11-18

## Status

Complete (M1–M7 Verified)

## Outcome

Remediated all 15+ diagnosed UI/UX interaction and display defects across the Web-only React 19 + FastAPI workflow, eliminated verified correctness/latency/memory bottlenecks in STT, AI translation, and TTS/dubbing assembly, fixed multi-line paragraph extraction in PaddleOCR, and delivered resource-efficient end-to-end Video OCR support (`subtitle_source="video_ocr"`) across the core scanner, Stage 2 repair, Stage 1 WebUI, API, and CLI. All 285 frontend tests, 118 E2E tests, 146 baseline unit tests, 110 optimization tests, and 104 hardening tests pass with 0 failures.

## Context

- Authority & Workflow: `AGENTS.md`, `docs/WORKFLOW.md`
- Architecture Decisions:
  - `docs/decisions/0001-video-ocr-as-subtitle-source.md` (Hard-subtitle OCR into `source_sub` SRT; `<= 250ms` boundary precision; in-memory ROI cropping, adaptive sampling, selective OCR, no physical scene splits)
  - `docs/decisions/0002-web-only-runtime.md` (Web-only React + FastAPI + CLI runtime; no PySide6)
- Baseline Pre-Change Review & Design Proposal: `docs/reviews/2026-11-18-feature-review-and-video-ocr-proposal.md`
- Baseline Commit: `1c096aec` (`main`)

## Scope

In scope:

- **M1 (R1)**: Pre-change baseline review & design proposal Markdown report (`docs/reviews/2026-11-18-feature-review-and-video-ocr-proposal.md`) and this active repository plan.
- **M2 (R2)**: Fix all 15 UI/UX defects across Batch/Date-Parent sessions, `StatusFooter`/`Header`/`WorkflowStepper`/`VideoPlayer`/`ProviderDocs`, Stage 2 OCR & Translation modals, Stage 3 Dubbing & Voice Management & `CreateVoiceModal`, Stage 4 BGM/CapCut export/`SettingsModal`, partial `update_project_state` merge (`merge: bool = True` with `reset_project_stage` at `videotrans/core/stage_reset.py:77, 126` passing `merge=False` or clearing `ocrCrop=None`), and `tests/test_webui.py` alignment.
- **M3 (R3a)**: STT fixes & optimizations (`DeepgramRecogn` `detect_language=None`, `asr_duration` tracking, `_stage_prepare.py` single `441000_ac2_raw.wav` extraction when `is_separate=True` with `vocal_bgm` success/fallback resampling to `source_wav`, header-only duration checks, `cut_audio` inside `cache_folder`, `faster-*.srt` cleanup, `_merge_sub` guards, vectorized Ten-VAD fallback, Qwen `detect_language` forwarding, client close/timeouts/retries).
- **M4 (R3b)**: AI Translation & prompt optimizations (`_run_srt` cache key symmetry, multiline `<think>` / unclosed tag parsing, `aitrans_context` sliding window & `None` guard, `Google` URL-encode/TLS/timeout/Session, bounded `ThreadPoolExecutor` concurrency + pooled clients with `threading.Lock`-guarded `Gemini.api_keys` rotation & per-`api_key` `genai.Client` pooling, runtime retry loops / dynamic tenacity callbacks keeping transient `429`/5xx backoff active even with `"retry_nums": 1`, static `system`-prompt caching split + per-batch filtered `{GLOSSARY_DICT}` in `user` + compact indexed payload, `/api/translate` temp cleanup & batch progress).
- **M5 (R3c)**: TTS & Dubbing assembly optimizations (per-engine preview locks + `asyncio.to_thread` for cloud TTS, Stage 3 batch dubbing endpoint using `VieNeu` `max_batch_size=4` + `dubbingSlice.ts` wiring, NumPy vectorized 48 kHz mono audio assembly with explicit `sr != 48000` resampling & mono downmix (`ndim > 1`) replacing $O(N \cdot T)$ `pydub` overlay, cloud client & VieNeu engine/voice reuse, temp WAV/MP3 cleanup, header-only `_rate.py` duration checks, bounded LRU/TTL preview/dubbing caches).
- **M6 (R4 + R5)**: R4 OCR paragraph fixes (`use_angle_cls=False` default, layout-aware row-band clustering + column gutter detection + CJK/Latin `\n`/space joining, `(device, mapped_lang)` engine cache, `paddle.is_compiled_with_cuda()`-only GPU detection + `select_device("cuda")` CPU fallback + `_ensure_win_dlls()` ordering, `np.ndarray` polygon guard, `ocr_helpers.py:141, 144` frame-level joining) and R5 Video OCR support (`format=bgr24` before `crop` + explicit `(out_w, out_h)` `ffmpeg` ROI crop pipe + `PyAV`/`frame_at_fast` pre-commit binary-search refinement on `EMPTY<->ACTIVE` and `ACTIVE<->ACTIVE` within `<=250ms` bracketing `[last_coarse_ts, ts_ms]`, `_last_ocr_ts` at `t=0` & dedup-skip `_last_active_ts`/grace-gap advance, localized max-tile `8x4`/`16x4` MAD/edge-diff dedup + bounded heartbeat, `<50MB` scanner+decoder overhead, bounded LRU OCR cache, `threading.Lock` on cached `PaddleOcrProvider`, `orchestrator.py:194` `ocr_progress` relay & `:234-253` audio-hash bypass, `task_params.py:117-130` & `jobs.py:110-113` `video_ocr` validation bypass for both `asr` and `vtv`, `_stage_recogn.py:51-52` `asr_duration`, `job_id or get_current_job_id()` + `CancellationToken` + `proc_registry`, Stage 2 dedup + cached provider, Stage 1 WebUI/API/CLI `subtitle_source="video_ocr"` integration).
- **M7**: Full E2E opaque-box test pass (Tiers 1–4), Tier 5 adversarial coverage hardening, full regression verification, and move this plan to `docs/plans/completed/`.

Out of scope:

- Reintroducing PySide6/Qt desktop widgets or desktop packaging (`docs/decisions/0002-web-only-runtime.md`).
- Adding online/cloud OCR providers beyond the `BaseOcrProvider` boundary (`docs/decisions/0001-video-ocr-as-subtitle-source.md`).

## Approach

Execute in strict milestone dependency order so interface contracts remain stable:

1. **M1 (Docs-Only)**: Complete and verify baseline review report and active execution plan before touching source or test files.
2. **M2 (UI/UX & State Integrity)**: Fix `update_project_state` key-level merge and local date-parent grouping first so frontend/backend contracts match, then resolve all remaining React UI/UX interaction/display defects and `test_webui.py` drift.
3. **M3–M4 (STT & AI Translation)**: Land independent backend correctness, concurrency, caching, prompt-efficiency, and temp-cleanup improvements with focused unit + benchmark proof.
4. **M5 (TTS & Dubbing Assembly)**: Implement per-engine locking, Stage 3 batch dubbing API + `dubbingSlice.ts` integration, NumPy assembly, and client/temp hygiene.
5. **M6 (OCR Paragraph + Video OCR)**: Fix R4 paragraph extraction first (so Video OCR and Stage 2 emit coherent multi-line text), then implement R5 streaming ROI decoder, 2-stage dedup, bidirectional refinement, cancellation, Stage 2 speedup, and Stage 1 WebUI/CLI wiring.
6. **M7 (E2E & Hardening)**: Run independent E2E Tiers 1–4, execute Tier 5 adversarial checks, verify all frontend and Python regression suites pass, and promote this plan to `docs/plans/completed/`.

## Interface Contracts

- **`PUT /api/projects/{project_id}/state`** (`videotrans/core/project_store.py:update_project_state`): Add `merge: bool = True` so partial background/autosave updates shallow-merge incoming `state_dict` onto existing `project["state"]` (never erasing `parent_id`, `project`, `languages`, `backend`, or `engines`), while `reset_project_stage` (`videotrans/core/stage_reset.py:77, 126`) passes `merge=False` (or sets `ocrCrop=None`) so deleted `ocrCrop` is never resurrected across Stage 2 resets.
- **Date-Parent IDs (`parent_YYYY-MM-DD`)**: Both frontend and backend MUST compute `YYYY-MM-DD` using local calendar date.
- **Stage 3 Batch Dubbing**: `POST /api/tts/batch` / `POST /api/projects/{id}/dubbing/generate` returns per-segment `{ segmentId, ok, previewUrl, durationSec, error }` so failures are surfaced rather than masked.
- **`POST /api/projects/{id}/dubbing/assemble`** (`videotrans/api/routes/projects.py:368, 412-414`): Mixes segments into a pre-allocated `48000 Hz` mono NumPy `float32` buffer (`soundfile.write`), explicitly resampling clips whenever `sr != 48000` and downmixing multi-channel clips (`ndim > 1`) to mono while preserving clip offsets, 30ms collision clamping, and linear fades.
- **OCR (`videotrans/ocr/`)**: `OcrConfig(use_angle_cls=False, paragraph_mode="preserve_lines", cpu_threads=None, coarse_interval_ms=500, max_roi_width=1280)`; `PaddleOcrProvider.select_device("cuda")` falls back cleanly to `"cpu"` when `paddle.is_compiled_with_cuda()` is `False`, and cached `PaddleOcrProvider` engine switching/`engine.ocr()` calls are guarded by a `threading.Lock`; `scan_video_to_srt(..., stop_event=None, progress_callback=None, job_id=None)` defaults `job_id = job_id or get_current_job_id()`, honors cancellation and `proc_registry`, executes pre-commit binary-search boundary refinement bracketing `[last_coarse_ts, ts_ms]`, and emits `ocr_progress` relayed by `orchestrator.py:194`; `build_task_params` (`task_params.py:117-130`), `jobs.py:110-113` (`asr_validator` for both `asr` and `vtv`), `orchestrator.py:234-253` (audio-hash cache bypass/scoping), and `cli.py` accept/honor `subtitleSource`/`--subtitle_source`, `ocrRoi`/`--ocr_roi`, and `ocrRoiConfirmed`.

## Risks And Recovery

- **Offline Hugging Face / model download stalls in CI/local pytest**: Always run Python tests with `HF_HUB_OFFLINE=1` and `--timeout=30..120`, matching `.github/workflows/main.yml`.
- **Windows Paddle + Torch DLL load order (`shm.dll` `WinError 127`)**: Enforce `_ensure_win_dlls()` before any `import paddle` entry point (`select_device`, `_resolve_use_gpu`, `is_available`, `_require_engine`).
- **Regression during multi-milestone edits**: Keep each milestone scoped to its owned files; verify `cd frontend && bun run test`, `bun run build`, and targeted/grouped `pytest` suites after every milestone before proceeding. Roll back isolated milestone commits if any invariant breaks.

## Progress

- [x] **M1**: Pre-change baseline review (`docs/reviews/2026-11-18-feature-review-and-video-ocr-proposal.md`) and active plan created (docs-only, baseline `1c096aec`).
- [x] **M2**: R2 UI/UX fixes, partial `state_json` merge, and `test_webui.py` alignment.
- [x] **M3**: R3a STT correctness, audio prep deduplication, temp cleanup, and Ten-VAD/Qwen optimization.
- [x] **M4**: R3b AI Translation cache symmetry, `<think>`/tag parsing, concurrency, retries, and prompt/token efficiency.
- [x] **M5**: R3c TTS lock scoping, Stage 3 batch dubbing, NumPy audio assembly, client reuse, and cache bounds.
- [x] **M6**: R4 OCR paragraph extraction fixes and R5 resource-efficient Video OCR + Stage 1/CLI/Stage 2 integration.
- [x] **M7**: E2E Tiers 1–4 pass, Tier 5 hardening, full regression, and move plan to `docs/plans/completed/`.

## Decisions

- 2026-11-18: Verified that `.agents/skills/` contains only developer/harness skills (`skills-lock.json`); runtime AI translation skills/prompts live in `videotrans/prompts/`, `videotrans/util/prompts.py`, and `videotrans/translator/`.
- 2026-11-18: Confirmed via empirical matrix that `PaddleOCR` `use_angle_cls=True` (`cls=True`) causes interior horizontal paragraph lines to be rotated 180° and dropped (`score < 0.5`); `OcrConfig.use_angle_cls` will default to `False`.
- 2026-11-18: Selected single-process `format=bgr24`-before-`crop` `ffmpeg` ROI crop pipe with explicit `(out_w, out_h)` sizing (`174.6 ms` / 12 frames) + `PyAV`/`frame_at_fast` pre-commit binary-search refinement over per-frame `ffmpeg` seeking (`3208.3 ms`) to achieve `< 50 MB` scanner+decoder RSS overhead and `<= 250ms` boundary accuracy.
- 2026-11-18: Confirmed that with `enable_mkldnn=False` required on CPU `PaddlePaddle 3.3.1` (`OMP_NUM_THREADS=1` in `_paths.py:66`), `cpu_threads=4` vs `1` yields negligible speedup (`687.6 ms` vs `678.6 ms/call`), so CPU Video OCR speedups rely primarily on localized max-tile deduplication, single-pipe ROI decoding, and thread-safe cached provider reuse.

## Validation

- **Focused proof**:
  - Frontend: `cd frontend && bun run test` (baseline: 171 pass) & `cd frontend && bun run build`
  - OCR unit + empirical multi-line EN/ZH/mixed/jittered/2-column checks: `.venv\Scripts\pytest.exe tests/test_ocr_*.py` (baseline: 25 pass)
  - STT / Translation / TTS suite: `$env:HF_HUB_OFFLINE="1"; .venv\Scripts\pytest.exe tests/test_base_recogn.py tests/test_base_trans.py tests/test_base_tts.py tests/test_audio_fit.py tests/test_deepgram_options.py tests/test_deepgram_timing.py tests/test_model_cache_preview.py tests/test_vieneutts.py tests/test_vieneu_features.py tests/test_dubbing_assembly.py tests/test_voice_sync_and_truncation.py tests/test_stage2_stage3_translation.py` (baseline: 117 pass)
  - Reproducible Video OCR resource check: end-to-end sample clip measuring wall-clock time, CPU/GPU usage, peak RSS delta, text accuracy, and `<= 250ms` boundary error.
- **Integration or end-to-end proof**:
  - Web runtime no-Qt & API contract: `tests/test_web_runtime_no_qt.py`, `tests/test_webui.py` (resolve 2 baseline failures in M2)
  - E2E Tiers 1–4 + Tier 5 adversarial hardening in M7.
- **Repository-required checks**:
  - `git diff --check`, 4-file grouped `pytest` under `HF_HUB_OFFLINE=1`, `bun run build`, `bun run test`.

## Result

Verified complete across Milestones M1–M7. All 285 frontend tests, 118 E2E integration tests, 146 baseline unit tests, 110 optimization tests, and 104 hardening tests pass cleanly.
