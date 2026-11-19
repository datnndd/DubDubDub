# Feature Review, Pipeline Bottleneck Evaluation, OCR Paragraph Verification & Video OCR Design Proposal

Date: 2026-11-18  
Baseline Commit: `1c096aec` (`main`)  
Status: Pre-change R1 Review & Architecture Proposal (Docs-only baseline)

---

## Executive Summary

`pyvideotrans` operates strictly as a **Web-only React 19 + FastAPI + CLI** runtime (`docs/decisions/0002-web-only-runtime.md`, enforced by `tests/test_web_runtime_no_qt.py` — PySide6/Qt is retired and absent from `pyproject.toml` and `.venv`). Recent commits (`b08f2d3e` through `1c096aec`) introduced batch multi-video folder ingestion with date-parent grouping, Stage 2 interval OCR segment replacement, VieNeu Voice Design/Cloning/Audition with Smart Auto Speed Fit, and Stage 4 subtitle styling & CapCut export.

This pre-change R1 baseline review documents verified defects, bottlenecks, empirical OCR findings, and concrete remediation proposals across all five required areas:

1. **Newly Added UI/UX Features (R2)**: **15+ interaction, state-integrity, and information-display defects** across Batch/Date-Parent sessions, `StatusFooter`/`Header`/`WorkflowStepper`/`VideoPlayer`/`ProviderDocs`, Stage 2 OCR & Translation modals, Stage 3 Dubbing & Voice Management & `CreateVoiceModal`, Stage 4 BGM/CapCut export/`SettingsModal`, plus 2 pre-existing `tests/test_webui.py` test/runtime drift failures.
2. **STT, AI Translation Skills & TTS Evaluation (R3)**:
   - Clarifies that `.agents/skills/` holds developer/harness agent skills (`skills-lock.json`), whereas runtime AI translation skills/prompts live in `videotrans/prompts/` (`srt/`, `text/`, `resegment/`, `recogn/`), `videotrans/util/prompts.py`, and `videotrans/translator/`.
   - Identifies critical correctness/latency/resource bottlenecks: Stage 3 TTS serialization behind a single global `asyncio.Lock()`, `max_batch_size=1` discarding VieNeu batching, synchronous blocking cloud TTS inside `async def`, $O(N \cdot T)$ `pydub` overlay loops; AI translation's 0%-hit SRT cache key mismatch, sequential batch execution, `retry_nums=1` (zero retries), unflagged multiline `<think>` regex, `aitrans_context` `len(None)` crash/prompt overflow, and ~1,300-token per-batch boilerplate; STT's `DeepgramRecogn` `detect_language[:2]` `None` crash, `maxtasksperchild=1` worker respawn, redundant `ffmpeg` audio extractions, full-WAV RAM duration probes, `TEMP_DIR/clip_*` leaks, and non-vectorized Ten-VAD fallback.
3. **OCR Paragraph vs. Single-Line Verification (R4)**: Empirical tests on `PaddleOCR 2.10.0` / CPU-only `PaddlePaddle 3.3.1` prove that multi-line paragraphs drop interior lines when `use_angle_cls=True` (`cls=True`), scramble reading order on vertically jittered same-line and two-column layouts due to raw `(y, x)` sorting, collapse `\n` and corrupt CJK spacing via `" ".join`, cache the wrong language engine across calls, falsely report `"gpu"` via `torch.cuda.is_available()`, and crash with Windows `shm.dll` `WinError 127` when `select_device()` is called before `_ensure_win_dlls()`.
4. **Resource-Efficient Video OCR Proposal (R5)**: Baseline 6s 720p H.264 scanning takes **28.95s** (~4.8× slower than real-time), swells RSS by **+450.6 MB**, and misses segment end boundaries by **500ms** (exceeding the 250ms target in ADR 0001). We propose a single-process streaming `format=bgr24`-before-`crop` `ffmpeg` ROI crop pipe (`174.6 ms` vs `3208.3 ms` — **18.4× faster** decode) paired with `PyAV`/fast-seek pre-commit binary-search boundary refinement, localized max-tile (`8x4`/`16x4`) dHash+MAD+edge deduplication, `<50 MB` scanner+decoder RSS overhead, bounded LRU OCR caching, thread-safe cached provider reuse, `CancellationToken` + `proc_registry` lifecycle tracking, Stage 2 `extract_ocr_segment_text` dedup + cached provider reuse, and WebUI Stage 1 / API / CLI `subtitle_source="video_ocr"` integration.
5. **Prioritized Roadmap (P0/P1/P2 → M2–M7)**: Maps every fix and feature cleanly to Milestones M2–M7 with verifiable acceptance gates.

---

## Baseline Test & Environment Verification

All baseline checks were executed directly against commit `1c096aec` using the project's `.venv` (`Python 3.10.19`, `uv`) and `bun` (`1.3.14`), with `HF_HUB_OFFLINE=1` matching `.github/workflows/main.yml`:

| Suite / Check | Command | Verified Result | Notes |
|---|---|---|---|
| Frontend unit & component suite | `cd frontend && bun run test` | **171 pass, 0 fail** (816 `expect()` across 11 files, `897ms`) | Uses `react-dom/server` `renderToStaticMarkup` & Zustand store tests |
| Frontend TypeScript + Vite build | `cd frontend && bun run build` | **Pass** (`tsc && vite build`, 1,937 modules, `4.76s`) | Outputs `dist/assets/index-*.js` (`630.00 kB`) |
| Targeted STT / Translation / TTS suite | `$env:HF_HUB_OFFLINE="1"; .venv\Scripts\pytest.exe -q --disable-warnings --timeout=30 tests/test_base_recogn.py tests/test_base_trans.py tests/test_base_tts.py tests/test_audio_fit.py tests/test_deepgram_options.py tests/test_deepgram_timing.py tests/test_model_cache_preview.py tests/test_vieneutts.py tests/test_vieneu_features.py tests/test_dubbing_assembly.py tests/test_voice_sync_and_truncation.py tests/test_stage2_stage3_translation.py` | **117 passed**, 6 warnings (`20.81s`) | Note: running without `HF_HUB_OFFLINE=1` can stall on Hugging Face hub checks |
| Existing OCR unit suite | `$env:HF_HUB_OFFLINE="1"; .venv\Scripts\pytest.exe -q --disable-warnings --timeout=30 tests/test_ocr_paddle.py tests/test_ocr_scanner.py tests/test_ocr_stage2_repair.py` | **25 passed**, 1 warning (`15.23s`) | All 25 mock the OCR engine (`FakeEngine`/`FakeProvider`) |
| Parent store, scanner, Stage 2 OCR, Deepgram | `$env:HF_HUB_OFFLINE="1"; .venv\Scripts\pytest.exe -q --disable-warnings --timeout=30 tests/test_parent_project_store.py tests/test_folder_scanner.py tests/test_ocr_stage2_repair.py tests/test_deepgram_options.py` | **23 passed**, 6 warnings (`9.99s`) | Verified clean |
| Web-only no-Qt & WebUI API contract | `$env:HF_HUB_OFFLINE="1"; .venv\Scripts\pytest.exe -q --disable-warnings --timeout=30 tests/test_web_runtime_no_qt.py tests/test_webui.py` | **19 passed, 2 failed** (`11.36s`) | Failures at `tests/test_webui.py:178` & `tests/test_webui.py:297` (documented in Area 1) |

---

## Area 1: Newly Added Features Inventory & UI/UX Defect Diagnosis (R2)

### 1.1 Recent Commits & Architecture Summary (`1e717bf2`..`1c096aec`)

Per `docs/decisions/0002-web-only-runtime.md`, `tests/test_web_runtime_no_qt.py` guarantees `PySide6` is absent from all supported runtime paths (`webui.py`, `cli.py`, `videotrans/api/`, `videotrans/task/`, `videotrans/recognition/`, `videotrans/translator/`, `videotrans/tts/`, `videotrans/ocr/`). Recent feature waves built on **React 19 + TypeScript + Tailwind CSS v4 + Zustand 5 + Radix UI** (`frontend/src/`) and **FastAPI + Uvicorn + SQLite** (`videotrans/api/`, `videotrans/core/`):

1. **Batch Video Ingestion, Date-Parent Grouping & Multi-Video Sessions** (`1e717bf2`, `b08f2d3e`, `dbc48ad4`, `f8181149`, `2c3e5cf3`, `5c1d1b54`, `861f31ac`, `e511a324`, `36ba7b82`): `FolderScanModal.tsx`, `SessionVideosDrawer.tsx`, `ProjectsScreen.tsx`, `Header.tsx`, `projectSlice.ts`, `jobSlice.ts`, `videotrans/core/{project_store.py,proc_registry.py,secret_store.py}`, `videotrans/api/routes/{media.py,projects.py,jobs.py}`.
2. **Stage 2 Interval OCR Segment Replacement & UI Simplification** (`f6c33628`, `7ca5b7ec`): `Stage2ReviewTranscript.tsx`, `Stage2OcrDialog.tsx`, `transcriptSlice.ts`, `videotrans/api/ocr_helpers.py`, `videotrans/api/routes/stages.py`.
3. **Voice Cloning, VieNeu Voice Design, Audition, Smart Auto Speed Fit & Batch Dubbing Queue** (`9c87a1ba`, `2b4c64d5`, `9909aa26`, `7b0da57c`, `6c2d72e2`, `f6c33628`, `c824aa55`, `f8181149`, `2c3e5cf3`): `Stage3VoiceDubbing.tsx`, `VoiceManagementScreen.tsx`, `CreateVoiceModal.tsx`, `VoiceSelector.tsx`, `services/voiceAuditionManager.ts`, `dubbingSlice.ts`, `videotrans/services/{voice_preview.py,audio_fit.py}`, `videotrans/tts/_vieneutts.py`, `videotrans/core/voice_store.py`, `videotrans/api/routes/voices.py`.
4. **Stage 4 Subtitle Styling, CapCut Export & Stage Reset** (`958d0055`, `5fd5cf6e`, `f6c33628`): `Stage4EditVideo.tsx`, `CapCutExportModal.tsx`, `VideoPlayer.tsx`, `editVideoSlice.ts`, `StatusFooter.tsx`.

### 1.2 Diagnosed UI/UX Interaction & Display Defects (15+ Issues)

#### A. Batch Video & Date-Parent Session Workflow
1. **UTC vs. Local Timezone Date-Parent Mismatch (`parent_YYYY-MM-DD`)**
   - **Evidence**: `frontend/src/components/FolderScanModal.tsx:187, 432`, `frontend/src/screens/ProjectsScreen.tsx:79-81, 188, 228`, and `frontend/src/store/projectSlice.ts:361` compute today's date via `new Date().toISOString().split('T')[0]` (UTC). Backend `videotrans/core/project_store.py:110` computes `datetime.date.today().isoformat()` (local OS timezone).
   - **Impact**: In non-UTC timezones (e.g., UTC+7 between 00:00–07:00 local), frontend creates/groups under yesterday's UTC date while backend creates today's local date, splitting a single day's batch across two date-parent folders.
   - **Proposed Fix**: Introduce a shared local calendar date helper (`formatLocalDateYmd(date = new Date())` using `getFullYear()`, `getMonth() + 1`, `getDate()`) across all frontend components/stores, matching Python's `datetime.date.today().isoformat()`.

2. **`FolderScanModal` Filename-Only Deduplication, Session Overwrite & Stale Modal State**
   - **Evidence**: `frontend/src/components/FolderScanModal.tsx:123-124` deduplicates picked browser files via `new Set(prev.map((p) => p.filename))`, silently dropping identically named files from different subdirectories (e.g., `ep1/video.mp4` and `ep2/video.mp4`). Line `194` calls `setSessionSubprojects(batchRes.projects, ...)` where `videotrans/api/routes/projects.py:155-159` returns only newly created subprojects—overwriting existing videos when scanning additional folders into an open session. Lines `40-48` never reset `folderPaths`, `detectedVideos`, `searchFilter`, or `scanError` when closing/reopening the modal.
   - **Proposed Fix**: Deduplicate by normalized relative/full path (`v.path || v.filename`); merge/append non-duplicate subprojects (or refetch `listSubprojects(batchRes.parent.id)`) when adding to the active parent session; reset transient modal state on close/open.

3. **`SessionVideosDrawer` Status Mislabeling, Missing `'failed'` State & No-Op Re-Selection**
   - **Evidence**: `frontend/src/components/SessionVideosDrawer.tsx:109-126` computes `completedCount` as `s.stage >= 2 || projectJobStatuses[s.id]?.status === 'completed'` and displays `"{completedCount} of {N} videos ready for review"` even when Stage 3 dubbing or Stage 4 render is complete. Lines `83-107` lack a branch for `sub.status === 'failed'` (displaying `'Pending'` after reload, unlike `ProjectsScreen.tsx:429`) and duplicate `getSubprojectStatusInfo` from `Header.tsx:28-112`. Lines `189-190` invoke `switchSessionVideo(sub.id)` even when `sub.id === activeProjectId`, triggering redundant autosave + `fetchProject`.
   - **Proposed Fix**: Import a single shared `getSubprojectStatusInfo` helper; render `'Failed'` when `sub.status === 'failed'`; label drawer counts accurately (stage-aware readiness); short-circuit `switchSessionVideo(sub.id)` when `sub.id === activeProjectId`.

4. **`StatusFooter` Cross-Stage Busy/Progress Bleed & Single-Video `projectJobStatuses` Desync**
   - **Evidence**: `frontend/src/components/StatusFooter.tsx:30, 65-72` bundles `dubbingStatus === 'running'`, `assemblingDubbing`, and `translationModal?.status === 'translating'` into `isBusy`. If the user navigates to Stage 1 while background dubbing is running for another video, Stage 1 shows `'Processing ASR…'` and disables `Start Dub` and `Reset`. Lines `163-174` display `jobMessage`/`jobProgress` even when dubbing or translation is the active background operation. `StatusFooter.tsx:73-79` keeps `Start Batch Dub` enabled (`disabled: false`) even if subprojects lack media or are already past Stage 1. Meanwhile, `frontend/src/store/jobSlice.ts:273-357` (`startPrepareJob`) never writes to `projectJobStatuses[activeProjectId]`, so switching videos via `SessionVideosDrawer` loses single-video ASR telemetry.
   - **Proposed Fix**: Scope `StatusFooter` busy/progress telemetry by `currentStep` and active project ID; disable `Start Batch Dub` unless at least one session subproject is runnable in Stage 1 (`media_id` or `media_path` present and `stage === 1`); update `projectJobStatuses[activeProjectId]` in `startPrepareJob()`.

5. **`startBatchPrepareJobs` Double-Completion Race & Background `state_json` Overwrite**
   - **Evidence**:
     - `frontend/src/store/jobSlice.ts:194-243`: `subscribeJobStream` calls `void handleSubSuccess(...)` inside `onEvent` when `payload.status === 'succeeded' || payload.stage === 'task_done'` (lines `228-229`) **and** passes `async () => { await handleSubSuccess(); }` as the `onDone` callback (lines `241-243`). Both fire on a normal SSE completion, invoking `handleSubSuccess` twice, decrementing `runningCount` twice, and breaking the `maxConcurrency = 2` limit.
     - `frontend/src/store/jobSlice.ts:168-172`: Background subprojects (`!isCurrent`) call `updateProjectState(pid, { segments, transcriptOptions, currentStep: 2 }, 2, mediaId)`. On the backend, `videotrans/api/routes/projects.py:310-315` → `videotrans/core/project_store.py:241-246` passes `state=state_dict` directly to `update_project`, overwriting `state_json` wholesale and **erasing** `parent_id`, `project` metadata (`filename`, `duration`, `previewUrl`), `languages`, `backend`, and `engines`.
   - **Proposed Fix**: Guard `handleSubSuccess` (and `failed`/`error` handlers) with an idempotent `settled` boolean per subproject; keep `jobSlice.ts:172`'s `mediaId` 4th argument intact (matching `frontend/src/api/projects.ts:37-42`: `updateProjectState(id, state, stage, mediaId)`). In `videotrans/core/project_store.py:211-246` (`update_project_state`), add `merge: bool = True` so partial autosaves/background updates shallow-merge onto existing `project["state"]` (`{**existing_state, **state_dict}`, preserving `parent_id`, `project`, `languages`, `backend`, `engines`), while `reset_project_stage` (`videotrans/core/stage_reset.py:77, 126`) passes `merge=False` (or explicitly sets `state["ocrCrop"] = None`) so deleted `ocrCrop` keys are never resurrected across Stage 2 resets.

6. **`ProjectsScreen` Parent-Record Leakage, Unfiltered Group Totals & Table View Parity**
   - **Evidence**: `frontend/src/screens/ProjectsScreen.tsx:126-157, 678` and `frontend/src/components/Header.tsx:166` do not filter out date-parent container rows (`state?.is_parent === true` or `id.startsWith('parent_')`) from `projectsList` counts, `filteredProjects`, `sortedProjects`, or `Select All (${sortedProjects.length})`. Flat/A-Z sorting renders `parent_YYYY-MM-DD` as a project card, and empty workspaces display `1` project in the Header badge. Lines `248-254, 281-300` compute `totalDuration`, `completedCount`, and `processingCount` before applying `searchQuery`. Lines `1137-1182` (Grouped Table View) omit the date-group selection checkbox, `+ Add Video`, and `Scan Folders` actions present in Grid View (`969-1088`).
   - **Proposed Fix**: Derive a memoized `videoProjects` list excluding parent containers (`!p.state?.is_parent && !p.id.startsWith('parent_')`) for Header counts, Flat/A-Z lists, and `Select All`; compute group summary metrics after `searchQuery` filtering; add group checkbox, `+ Add Video`, and `Scan Folders` parity to Grouped Table View.

#### B. Misleading / Hardcoded Displays & Broken Bindings
7. **`ProviderDocLink` ASR Provider ID Collision (`frontend/src/config/providerDocs.ts:159-179`)**
   - **Evidence**: `videotrans/recognition/__init__.py:10-15` defines `QWENASR=0`, `Deepgram=1`, `GEMINI_SPEECH=2`, `GOOGLE_SPEECH=3`, `ElevenLabs=4`, `FASTER_WHISPER=5`. However, `getProviderDocInfo(idOrType, 'asr')` in `providerDocs.ts:159-163` maps `0 -> whisper` (instead of Qwen-ASR) and omits `2..5`, falling through to the numeric TTS `switch` at lines `166-179`—mapping Gemini STT (`2`) to VieNeu-TTS, Google STT (`3`) to Gemini TTS, ElevenLabs STT (`4`) to Edge-TTS, and Whisper (`5`) to OpenAI TTS!
   - **Proposed Fix**: Add complete `categoryHint === 'asr'` numeric mappings (`0 -> qwen_asr`, `1 -> deepgram`, `2 -> gemini_stt`, `3 -> google_stt`, `4 -> elevenlabs_stt`, `5 -> whisper`) and add missing ASR/translation entries to `PROVIDER_DOCS` with early return so `'asr'` never falls into TTS IDs.

8. **Hardcoded Telemetry & Non-Functional VideoPlayer Audio Switcher**
   - **Evidence**:
     - `frontend/src/components/Header.tsx:256-259`: Static `"Warm GPU"` badge regardless of actual CUDA/CPU availability.
     - `frontend/src/components/Header.tsx:251-254`: Renders `"Auto-saved Just now"` even on `Projects` / `Voice Management` views where no project is open.
     - `frontend/src/components/WorkflowStepper.tsx:26-44`: Hardcoded `"OCR Slide Engine: Active"`, `"Neural Voice Sync: 99.8%"`, `"Multi-Track Timeline: Synced"` badges.
     - `frontend/src/components/VideoPlayer.tsx:261-284` (`Stage2ReviewTranscript.tsx:279`): Audio channel toggle labels are hardcoded `"EN Orig"` and `"ES Dub ★"` regardless of `languages.source.code`/`languages.target.code`, and `VideoPlayer.tsx` never reads `playback.audioChannel` to mute/unmute original vs. dubbed audio tracks.
     - `frontend/src/components/VideoPlayer.tsx:41, 330, 340`: `currentSegment = segments.find((s) => s.id === activeSegmentId) || segments[0]` keeps a subtitle overlay visible during silent gaps or before `segments[0].startSec`.
   - **Proposed Fix**: Drive Header hardware pill from backend `/api/options` CUDA capability (`GPU Ready` / `CPU Mode`); hide auto-save indicator outside `dubbing` view or when `!activeProjectId`; replace static WorkflowStepper telemetry strings with real dynamic stage status; derive audio switcher labels from `languages.source.code` / `languages.target.code` and wire `playback.audioChannel` to `videoRef`/`dubAudioRef` gain/mute; display subtitle overlay only when `currentTime` falls inside an active segment (or when explicitly selected and inside/near active playhead).

#### C. Stage 2 OCR Dialog & Translation Modal UX
9. **`Stage2OcrDialog` Remounting, Bounds Guard, Scrubbing & Batch/Override Integrity**
   - **Evidence**:
     - `frontend/src/screens/Stage2ReviewTranscript.tsx:709`: `key={`${ocrSegment.id}-${ocrResult?.text ?? ''}`}` forces `Stage2OcrDialog` to unmount and remount when OCR returns a result, reloading `<video>` and resetting playback.
     - `frontend/src/components/Stage2OcrDialog.tsx:54-57`: `point(event)` divides by `bounds.width` and `bounds.height` without guarding against `0`.
     - `Stage2OcrDialog.tsx:102-106`: `<video>` lacks scrubbing/frame-step controls across `[startSec, endSec]`.
     - `Stage2ReviewTranscript.tsx:249-272` vs `Stage2OcrDialog.tsx:127-183`: Single-segment OCR previews detected entries for user confirmation (`Confirm & Replace`), whereas batch OCR auto-overwrites `sourceText` via `applyOcrText` without preview confirmation.
     - `frontend/src/store/transcriptSlice.ts:341`: `replaceSegmentWithOcrEntries` re-indexes `id: i + 1` for all segments, invalidating `segmentVoiceOverrides` keyed by `seg.id` if a user returns from Stage 3 to Stage 2.
   - **Proposed Fix**: Key `Stage2OcrDialog` stably by `ocrSegment.id`; guard `bounds.width <= 0 || bounds.height <= 0` in `point()`; add seek/frame-step controls within `[segment.startSec, segment.endSec]`; unify batch OCR preview/confirm UX; remap `segmentVoiceOverrides` (or preserve non-replaced segment IDs) during `replaceSegmentWithOcrEntries`.

10. **Translation Modal Error Detail & Export Error Handling**
    - **Evidence**: `frontend/src/store/transcriptSlice.ts:470-478` populates `translationModal.error` on failure, but `frontend/src/screens/Stage2ReviewTranscript.tsx:761` only renders `translationModal.message` (`"Translation failed"`), hiding actionable errors (e.g., invalid/missing API key). Line `362` invokes `exportTranslatedSrt(segments)` without catching thrown validation errors.
    - **Proposed Fix**: Render `translationModal.error` in an accessible error banner inside the modal when `status === 'error'`; wrap `exportTranslatedSrt` in `try/catch` and display inline UI feedback on error.

#### D. Stage 3 Dubbing, Voice Management & `CreateVoiceModal`
11. **Stage 3 Failure Masking, Missing Cancel, Hardcoded Providers & Unscoped VieNeu Chips**
    - **Evidence**:
      - `frontend/src/store/dubbingSlice.ts:473-475, 506-529`: `runFullDubbing` catches per-segment `previewTTS` failures with `console.warn` only, then unconditionally marks `dubbingStatus: 'completed'` (`'Voice completed'`). Proceeding to Stage 4 then fails with HTTP 400 in `assembleProjectDubbing`.
      - `frontend/src/screens/Stage3VoiceDubbing.tsx:297` & `StatusFooter.tsx:105-112`: No cancel button or abort controller exists once `runFullDubbing()` begins.
      - `Stage3VoiceDubbing.tsx:26-31, 366-378, 385, 595`: Hardcodes `TTS_PROVIDERS` instead of reading `backend.options.voices` and can fall back to `'No'` / `'default'` before voices hydrate.
      - `Stage3VoiceDubbing.tsx:708-735`: VieNeu Vietnamese emotion chips (`[cười]`, `[thở dài]`, `[hắng giọng]`) render regardless of TTS provider or target language.
    - **Proposed Fix**: Track failed segments in `dubbingSlice.ts` (`dubbingErrors` / `failedSegmentIds`), surface failed counts + retry action, and prevent marking `'completed'` when any/all segments failed; add a Cancel button wired to `AbortController`/backend cancellation; populate TTS provider options from `backend.options.voices` with valid default fallback; show VieNeu emotion chips only when `ttsType === VIENEU_TTS` (and Vietnamese target).

12. **`VoiceManagementScreen` Lab Picker, Gender Filter, Error Alerts & `CreateVoiceModal` Mic Leak**
    - **Evidence**:
      - `frontend/src/screens/VoiceManagementScreen.tsx:278-290, 1026-1044`: `lab-voice-picker` reads `filteredCustomVoices` / `filteredPresetVoices` instead of the unfiltered voice lists, so typing in the library search box empties the Synthesis Lab dropdown; `genderFilter` skips custom voices; VieNeu style/temperature sliders render for non-VieNeu voices; edit/delete failures (`lines 160, 177, 330`) only `console.error`.
      - `frontend/src/components/CreateVoiceModal.tsx:146-150`: Closing the modal while `isRecording` is active pauses `<audio>` players but fails to stop `mediaRecorderRef.current`, clear `recordingTimerRef.current`, or stop microphone `MediaStream` tracks.
    - **Proposed Fix**: Populate `lab-voice-picker` from full `customVoices`/`presetVoices`; apply `genderFilter` consistently; gate VieNeu controls by provider; surface edit/delete errors in UI alert state; clean up `MediaRecorder`, interval timers, and `stream.getTracks().forEach(t => t.stop())` on modal close/unmount.

#### E. Stage 4 BGM, CapCut Export, Settings Modal & `test_webui.py` Alignment
13. **Stage 4 BGM/Thumbnail Persistence & Audible Preview**
    - **Evidence**: `frontend/src/store/editVideoSlice.ts:118, 149` and `projectSlice.ts:480-483` store volatile `blob:` URLs for uploaded BGM and thumbnails that are stripped on autosave/reload; `Stage4EditVideo.tsx:409` renders `<audio id="stage4-bgm-preview" className="hidden" />` with no playback/volume synchronization with `VideoPlayer`.
    - **Proposed Fix**: Persist uploaded BGM/thumbnail server media paths/URLs (`/api/media/...` or project asset URLs) via backend upload so they survive project switches/reloads, and synchronize `#stage4-bgm-preview` play/pause/seek/volume/mute with `VideoPlayer`.

14. **CapCut `subtitles_target.srt` Wrong Text Field**
    - **Evidence**: `frontend/src/components/CapCutExportModal.tsx:74-77` exports `generateSrtContent(segments, 'sourceText')` for `subtitles_target.srt`, contradicting both its UI label (`"Raw target language subtitles"`) and backend `videotrans/api/routes/projects.py:523` (`rawTranslation` / `targetText`).
    - **Proposed Fix**: Export `rawTranslation` when present, falling back to `targetText` (never `sourceText`), matching `projects.py:523`.

15. **Settings Modal Store Sync & `tests/test_webui.py` Alignment**
    - **Evidence**: `frontend/src/components/SettingsModal.tsx:48-51` calls `initializeBackend()`, which skips updating `backend.config` once `isReady` is already `true` (`prepareSlice.ts:146-161`); `ApiProvidersTab.tsx:125-127` only syncs `deepgram` config, omitting LLM and TTS settings. On the backend, `tests/test_webui.py:178` fails because `videotrans/api/task_params.py:122-124` (changed in `2c3e5cf3`) bypasses `asr_models()` validation for `Deepgram`, and `tests/test_webui.py:297` fails because `videotrans/params.json` on disk sets `"model_name": "nova-3-general"` while the test expects `"nova-3"` (and `videotrans/cfg.json` contains a stale temp `homedir`).
    - **Proposed Fix**: Add a `refreshBackendConfig()` action or `force` flag on `initializeBackend(true)` and sync saved provider settings back into the store; restore/explicitly align Deepgram model validation/allowlist (`task_params.py` & `params.json` / `contants.DEEPGRAM_MODEL`) and clean stale `cfg.json` test artifact so `tests/test_webui.py` passes cleanly.

---

## Area 2: STT, AI-Powered Translation Skills & TTS Evaluation (R3)

### 2.1 Clarifying "AI Translation Skills" vs. `.agents/skills/`

A repository-wide search confirms `git grep -i "skill"` returns **0 matches** in `videotrans/`, `frontend/`, `docs/`, and `tests/`.
- **`.agents/skills/`** (tracked in `skills-lock.json`) contains **developer/harness agent workflow skills** (`code-review`, `diagnosing-bugs`, `ponytail`, `tdd`, `onboard-repository`, `improve-harness`, etc.) — none are runtime video-translation modules.
- **Runtime AI translation capabilities ("skills", prompt engineering, glossary injection, and LLM re-segmentation)** reside in:
  - `videotrans/prompts/srt/*.txt` (13 files: 12 identical 5,956-byte templates including active `chatgpt.txt`, `deepseek.txt`, `gemini.txt` plus legacy pruned names `ai302`, `azure`, `bailian`, `minimax`, `openrouter`, `siliconflow`, `xiaomi`, `zhipuai`, `zijie`; plus `localllm.txt` at 2,467 bytes)
  - `videotrans/prompts/text/*.txt` (13 identical 3,069-byte templates)
  - `videotrans/prompts/resegment/llm.txt` (2,348 bytes) & `llm2.txt` (2,206 bytes)
  - `videotrans/prompts/recogn/gemini_recogn.txt` (2,302 bytes)
  - `videotrans/util/prompts.py:9-46` (`get_prompt`, `qwenmt_glossary`, `get_prompt_file`)
  - `videotrans/translator/` (`_base.py`, `_openaicompat.py`, `_gemini.py`, `_deepseek.py`, `_chatgpt.py`, `_google.py`, `_registry.py`, `_runner.py`)

### 2.2 Speech-to-Text (STT) Evaluation (`videotrans/recognition/`, `videotrans/process/`, `videotrans/task/`)

Provider registry (`videotrans/recognition/__init__.py:10-27`): `QWENASR=0`, `Deepgram=1`, `GEMINI_SPEECH=2`, `GOOGLE_SPEECH=3`, `ElevenLabs=4`, `FASTER_WHISPER=5`.

#### Identified Bugs, Bottlenecks & Resource Inefficiencies
1. **`DeepgramRecogn` `detect_language=None` Crash & Missing `asr_duration`**
   - `videotrans/recognition/_deepgram.py:139, 194, 196, 271, 273, 302, 304, 336, 338` slices `self.detect_language[:2]` without null-guarding (line `83` checks `if self.detect_language`, but line `139` `logger.info(..., self.detect_language[:2])` and post-processing throw `TypeError` when `detect_language` is `None` or `'auto'`).
   - `videotrans/task/_stage_recogn.py:138-141` returns early when `speaker.json` exists, skipping `self.asr_duration = round(time.time() - _st, 2)` at line `163`.
2. **Redundant `ffmpeg` Audio Extractions (`_stage_prepare.py:181-256`)**
   - `PrepareMixin._split_audio_byraw` decodes `self.cfg.name` via `ffmpeg` to 16 kHz mono `source_wav`, then when `is_separate=True` decodes `self.cfg.name` a **2nd time** to 44.1 kHz stereo `441000_ac2_raw.wav`, runs `vocal_bgm`, and runs `ffmpeg` a **3rd time** to resample `vocal.wav` back to `source_wav`.
3. **Full-WAV RAM Decodes & Temp-File Leaks (`_whisper.py`, `_base.py`, `stt_faster.py`)**
   - `FasterAll.__post_init__` (`videotrans/recognition/_whisper.py:30`) calls `len(AudioSegment.from_wav(self.audio_file))` solely to get duration, decoding the entire PCM file into RAM.
   - `BaseRecogn.cut_audio` (`videotrans/recognition/_base.py:204-283`) reloads the full WAV in the parent process via `AudioSegment.from_wav`, converts to float32 NumPy samples, pads 500ms head/tail silence per chunk, and writes clips to `f"{config.TEMP_DIR}/clip_{time.time()}/audio_{i}.wav"`—outside `self.cache_folder`, so `BaseTask.set_end` (`shutil.rmtree(self.cfg.cache_folder)`, `videotrans/task/_base.py:128`) never deletes them.
   - `FasterAll` (`_whisper.py:62`) leaves `TEMP_ROOT/faster-{timestamp}.srt` uncleaned after normal completion (`stt_faster.py:203-204`).
4. **`BaseRecogn._merge_sub` Index & Timestamp Bugs (`_base.py:348-439`)**
   - `_phase4_redistribute_by_punct` indexes `post_srt_raws[i - 1]['text'][-1]` and `it['text'][-1]` without empty-string checks (risking `IndexError`); `_phase2_merge_first` and `_phase3_merge_last` update `start_time`/`end_time` without updating `startraw`/`endraw`/`time`.
5. **Ten-VAD Pure-Python Energy Loop & `stt_qwen.py` Language Drop**
   - `videotrans/process/vad.py:159-160` computes fallback split energies using a Python list comprehension over slices, and `get_speech_timestamp` (`lines 135-160`) re-runs `_detect_raw_segments` up to 3× on long segments.
   - `videotrans/process/stt_qwen.py:65` passes `language=[None for it in it_list]` unconditionally, discarding the user's `detect_language` hint.
6. **Subprocess Cold-Start & Cloud STT HTTP Hygiene**
   - `GlobalProcessManager` (`videotrans/process/process_manager.py:116, 131`) uses `maxtasksperchild=1` on CPU/GPU executors (paying full Python+CUDA+model reload every run) with 0.5s/1.0s polling (`process_manager.py:46`, `base.py:265`).
   - `ElevenLabsRecogn` (`_elevenlabs.py:22-46`) loads the entire WAV into RAM, creates an unclosed `httpx.Client` with no timeout, and forces `scribe_v1` for `'auto'`. `GeminiRecogn` (`_gemini.py:93-95, 107-145`) sends up to 50 padded clips sequentially, calls `StopRetry` on `429`/`500` (preventing backoff), and silently returns `''` on other `APIError`s. `GoogleRecogn` (`_google.py:32-84`) retries from chunk `0` if any single chunk fails. `DeepgramRecogn` (`_deepgram.py:66-71, 144`) reads `>50 MiB` files fully into RAM and creates a new client per call with `retry_nums=1` (0 retries).

#### Proposed STT Fixes (M3)
- Guard `detect_language` safely across `_deepgram.py` (`(self.detect_language or "")[:2]`) and set `self.asr_duration` before returning in `_stage_recogn.py`.
- Extract once in `_stage_prepare.py` when `is_separate=True`: extract `441000_ac2_raw.wav` once, derive `16kHz mono pcm_s16le` `self.cfg.source_wav` from `self.cfg.vocal` when `vocal_bgm` succeeds, **and** in the `else` / `except Exception:` fallback branch resample `16kHz mono pcm_s16le` `source_wav` from `441000_ac2_raw.wav` so `RecognMixin.recogn()` (`_stage_recogn.py:60`) never aborts if vocal separation fails.
- Add a header-only `get_audio_duration_ms()` utility (`wave` / `soundfile.info`) replacing `AudioSegment.from_wav(...)` duration checks.
- Place `cut_audio` temp directories inside `self.cache_folder` (fallback `TEMP_DIR` only if unset) and delete `faster-*.srt` in `finally`.
- Guard empty strings in `_phase4_redistribute_by_punct` and sync `startraw`/`endraw`/`time` in `_phase2_merge_first`/`_phase3_merge_last`.
- Vectorize `vad.py:159-160` energy calculation via NumPy (`np.abs(...).reshape(...).sum(axis=1)`), forward mapped `detect_language` in `stt_qwen.py:65`, close `ElevenLabs` `httpx.Client` with explicit timeout, and add backoff on `429`/5xx in Gemini/Deepgram/ElevenLabs/Google STT (while preserving per-chunk progress in `GoogleRecogn`).

### 2.3 AI-Powered Translation Evaluation (`videotrans/translator/`, `videotrans/prompts/`, `videotrans/util/prompts.py`)

Provider registry (`videotrans/translator/_registry.py:10-16`): `GOOGLE_INDEX=0`, `CHATGPT_INDEX=1`, `GEMINI_INDEX=2`, `DEEPSEEK_INDEX=3`.

#### Identified Bugs, Bottlenecks & Token Inefficiencies
1. **0%-Hit SRT Cache Key Mismatch (`videotrans/translator/_base.py:149, 154`)**: `_run_srt` looks up cache with `self._get_cache(srt_str)` (`str`) but writes cache with `self._set_cache(it, result)` (`List[SrtItem]`). `_get_key()` serializes its input with `serial()`, generating completely different MD5 keys for `str` vs `list[dict]`.
2. **Purely Sequential Execution, Import-Time `@retry` & Zero Default Retries (`_base.py:49-52, 85-167`, `_openaicompat.py:40`, `_gemini.py:28, 32-33`, `_google.py:19`)**: `self.trans_thread` only controls chunk slice length; `_run_text` and `_run_srt` execute chunks in a sequential `for` loop with `time.sleep(self.wait_sec)`. Meanwhile `@retry(..., stop=stop_after_attempt(settings.get('retry_nums')))` in `_openaicompat.py:40`, `_gemini.py:28`, and `_google.py:19` binds `settings.get('retry_nums')` **at import time** (ignoring runtime `SettingsModal` updates), and `videotrans/cfg.json:1` defaults `"retry_nums": 1` (0 retries). In addition, `Gemini._item_task` (`_gemini.py:32-33`) mutates `self.api_keys` via `pop(0)`/`append` without a lock (racing under concurrency) and raises `StopTask` on `429` and `500` (`_gemini.py:114`), aborting instead of retrying.
3. **`aitrans_context` Crash & Context Overflow (`_base.py:50`)**: When `aitrans_context=True`, `self.trans_thread = len(self.text_list)`, cramming the entire subtitle track into one LLM prompt (risking `LengthFinishReasonError`)—and if `text_list` is `None` (as when `OpenAICampat` is instantiated for `llm_segment` in `_stage_recogn.py:147`), `len(None)` raises `TypeError`. When `aitrans_context=False`, batches have zero cross-batch continuity.
4. **Multiline `<think>` & Tag Parsing Failures (`_openaicompat.py:97-100`, `_base.py:106-125, 157`)**:
   - `_openaicompat.py:97` runs `re.sub(r'<think>(.*?)</think>', '', result)` **without** `flags=re.I | re.S`, failing to strip multiline reasoning blocks; if `<TRANSLATE_TEXT>` is missing/unclosed, line `100` returns raw unstripped text (unlike `_gemini.py:105-109`).
   - `_run_text` (`_base.py:106-115`) splits blindly on `\n`, misaligning or padding `""` when line counts drift; `_run_srt` + `BaseTask.check_target_sub` silently blanks cues if the LLM alters a timestamp by 1ms.
5. **~1,300-Token Per-Batch Prompt Bloat & Missing Prefix Caching**:
   - `videotrans/prompts/srt/chatgpt.txt` is **5,956 bytes** (~1,200–1,400 tokens) duplicated across 12 files (`text/chatgpt.txt` is 3,069 bytes across 13 files).
   - `OpenAICampat._item_task` (`_openaicompat.py:48-56`) puts only `'You are a top-tier Subtitle Translation Engine.'` in `system` and sends the entire 5.9 KB instructions + `{GLOSSARY_DICT}` + `{batch_input}` in `user`—preventing OpenAI/DeepSeek/Gemini prefix caching and repeating immutable SRT timestamps on both input and output (~15–20 extra tokens/cue).
   - `get_prompt()` (`videotrans/util/prompts.py:9-20`) is uncached and injects every glossary entry regardless of whether those terms appear in the current batch.
6. **Per-Request Client Instantiation, `Google` Security/Encoding Bug & `/api/translate` Leaks**:
   - `_openaicompat.py:76, 149` and `_gemini.py:35` instantiate a fresh `OpenAI`/`genai.Client` on every batch (no TCP/TLS pooling).
   - `Google._item_task` (`_google.py:24, 30`) interpolates unescaped `?q={text}`, uses `verify=False`, sets no `timeout`, and `cleartext()` only unescapes `&#39;`/`&quot;` rather than `html.unescape`.
   - `POST /api/translate` (`videotrans/api/routes/stages.py:91-122`) leaves dummy `.mp4`, `cache-{uuid}`, and `target-{uuid}` in `TEMP_DIR`, and `run_staged_translation` emits no per-batch progress between 35% and 100%.

#### Proposed AI Translation Fixes (M4)
- Fix `_run_srt` cache key symmetry (`_get_cache(srt_str)` and `_set_cache(srt_str, result)`).
- Fix `_openaicompat.py:97` (`flags=re.I | re.S`), strip unclosed `<think>` / markdown code fences, and support `(?:</TRANSLATE_TEXT>|$)` uniformly across OpenAI/DeepSeek/Gemini.
- Guard `len(self.text_list or [])` in `_base.py:50` and replace all-in-one `aitrans_context` with a bounded sliding context window (passing last `K=3..5` translated lines as read-only context).
- Fix `Google._item_task` with `params={"sl": ..., "tl": ..., "hl": ..., "q": text}`, `timeout=15`, `verify=True`, `requests.Session` reuse, and `html.unescape`.
- Add bounded `ThreadPoolExecutor` concurrency (`aitrans_concurrency`, e.g., `min(4..6, len(chunks))`, preserving output order) + pooled `OpenAI`/`genai.Client` instances (guarding `Gemini.api_keys` rotation at `videotrans/translator/_gemini.py:32-33` with a `threading.Lock` and pooling `genai.Client` instances keyed per `api_key`). Replace import-time `@retry(..., stop=stop_after_attempt(settings.get('retry_nums')))` decorators (`_openaicompat.py:40`, `_gemini.py:28`, `_google.py:19`) with runtime retry loops or dynamic tenacity callbacks, ensuring exponential backoff with jitter on transient `429`/5xx remains active even when `videotrans/cfg.json` has `"retry_nums": 1` (removing `429`/`500` from Gemini's `StopTask` list).
- Place **only** static translation instructions in `system` (byte-identical across batches so provider prefix caching hits consistently), cache template loading (`lru_cache`), place per-batch filtered `{GLOSSARY_DICT}` + compact indexed line payload (`[1] ...`) in `user` to eliminate timestamp round-tripping while anchoring 1-to-1 cue alignment, clean up `/api/translate` temp dirs, and emit per-batch translation progress.

### 2.4 Text-to-Speech (TTS) & Dubbing Assembly Evaluation (`videotrans/tts/`, `videotrans/services/`, `videotrans/api/`, `videotrans/task/_rate.py`)

#### Identified Bugs, Bottlenecks & Resource Inefficiencies
1. **Global `_preview_lock` Serialization & Event-Loop Blocking (`videotrans/services/voice_preview.py:25, 738, 828-862`)**:
   - `synthesize_unified_tts_preview` acquires a single module-level `asyncio.Lock()` (`_preview_lock`) for **all** providers, serializing every Stage 3 segment request even across independent cloud APIs.
   - Inside `_do_synthesis`, ElevenLabs (`lines 828-846`) and Gemini TTS (`lines 849-862`) execute synchronous network calls directly on the FastAPI asyncio event loop, stalling SSE streams and other HTTP routes.
2. **Discarded VieNeu Batching (`voice_preview.py:219`, `dubbingSlice.ts:387-504`)**:
   - `runFullDubbing()` sends one HTTP `/api/tts/preview` call per segment, and `_run_vieneu_synthesis` hardcodes `max_batch_size=1` in `get_cached_vieneu_engine`, discarding VieNeu's `infer_batch(..., max_batch_size=4)` capability.
3. **$O(N \cdot T)$ `pydub` Audio Overlay Loop (`videotrans/api/routes/projects.py:340-421`)**:
   - `assemble_project_dubbing_handler` loads every WAV through `pydub.AudioSegment.from_wav`, allocates a silent full-duration track, and executes `merged = merged.overlay(clip, position=start_ms)` (`lines 413-414`) inside a loop—copying the entire track buffer $N$ times.
4. **Backend TTS Client/Model Churn, Temp Leaks & Unbounded Disk Caches**:
   - `BaseTTS` defaults (`_app_settings.py:144-145`, `_base.py:48-50`) enforce `dubbing_thread=1` and `dubbing_wait=1`s sleep per line even for cloud TTS.
   - `ElevenLabsC._run` (`_elevenlabs.py:27-78`) re-reads `elevenlabs.json` per line, constructs a new `ElevenLabs` client per line, ignores `self.proxy_str`, and leaves `{filename}.mp3` uncleaned.
   - `GEMINITTS` (`_geminitts.py:8, 25-29, 101-103`) constructs a new `genai.Client` per line, ignores `self.proxy_str`, never rotates comma-separated `gemini_key`s, catches `google.api_core.exceptions.Unauthorized` instead of `google.genai.errors.APIError`, and leaves `{filename}.wav` uncleaned.
   - `VieNeuTTS` (`_vieneutts.py:19, 35-140`) instantiates/closes a fresh `Vieneu` engine instead of reusing `model_cache`, re-queries SQLite and `.npz` files per segment rather than memoizing per voice, and reports progress only per voice group.
   - `OmniVoice` (`_omnivoice.py:39`, `omnivoice_tts.py:28, 44`) respawns `maxtasksperchild=1`, reloads `get_f5tts_role()` inside the `while 1:` loop, and leaves `-24k.wav` files behind.
   - `SpeedRate` / `TtsSpeedRate` (`videotrans/task/_rate.py:176-213, 665-688`) calls `AudioSegment.from_file` solely for duration, leaks `{input_path}-after.wav`, and creates one `silence_*.wav` file per gap via `pydub`.
   - `PREVIEWS_DIR` (`tmp/voice_previews/`) and `DUBBING_CACHE` (`TEMP_ROOT/dubbing_cache/`) grow without size or TTL bounds.

#### Proposed TTS Fixes (M5)
- Replace global `_preview_lock` with per-engine/GPU locks (`_gpu_lock` vs provider locks) and wrap ElevenLabs/Gemini preview synthesis in `asyncio.to_thread()`.
- Add a Stage 3 batch dubbing endpoint (`POST /api/projects/{id}/dubbing/generate` or `POST /api/tts/batch`) that groups segments by voice, leverages `VieNeu` `max_batch_size=4` (or bounded concurrent cloud synthesis), returns per-segment `{ segmentId, ok, previewUrl, durationSec, error }`, and integrates into `dubbingSlice.ts`.
- Replace `AudioSegment.from_wav(audio_path).set_frame_rate(48000).set_channels(1)` + `merged.overlay(...)` (`videotrans/api/routes/projects.py:368, 412-414`) with `soundfile.read` + a single pre-allocated `48000 Hz` mono `numpy.zeros(..., dtype=np.float32)` buffer using vectorized slice-add + `soundfile.write`: explicitly resample clips whenever `sr != 48000` (e.g., 24 kHz VieNeu/Gemini, 22.05/44.1 kHz ElevenLabs/Edge) and downmix multi-channel audio (`ndim > 1`) to mono before slice-adding, while preserving exact clip start offsets, 30ms collision clamping, and 20–30ms linear fades.
- Reuse `ElevenLabs`/`genai.Client` per instance with proxy & key rotation, cache `elevenlabs.json`, memoize `VieNeuTTS` voice resolution + share cached engine, clean up intermediate `.mp3`/`-24k.wav`/`-after.wav`/`silence_*.wav` files, use header-only duration checks in `_rate.py`, and add bounded LRU/TTL eviction for `PREVIEWS_DIR` and `DUBBING_CACHE`.

---

## Area 3: OCR Paragraph vs. Single-Line Extraction Verification Findings (R4)

### 3.1 Environment & Hardware Verified
- **Python**: `.venv\Scripts\python.exe` (`3.10.19`)
- **OCR Stack**: `paddleocr 2.10.0`, `paddlepaddle 3.3.1` (`paddle.is_compiled_with_cuda() == False`, CPU-only build), `torch 2.7.1+cu128` (`torch.cuda.is_available() == True`, `NVIDIA GeForce GTX 1650`), `onnxruntime 1.23.2` (`CPUExecutionProvider`), `av 16.0.1`, `cv2 4.11.0`, `numpy 1.26.0`, `pillow 12.1.1`.
- **Cached PaddleOCR weights (`$USERPROFILE\.paddleocr\whl\`)**: `ch_ppocr_mobile_v2.0_cls_infer`, `ch_PP-OCRv4_det_infer`, `en_PP-OCRv3_det_infer`, `ch_PP-OCRv4_rec_infer`, `en_PP-OCRv4_rec_infer`.

### 3.2 Empirical Verification Outputs (Verbatim Reproduced Results)

Testing real `PaddleOcrProvider` (`videotrans/ocr/_paddle.py`) against synthetic PIL-rendered images (`C:\Windows\Fonts\msyh.ttc`) revealed **six concrete defects**:

1. **Windows `shm.dll` `WinError 127` Crash on `select_device()` before `recognize()`**
   - Calling `PaddleOcrProvider(device="auto").select_device("auto")` before `recognize()` imports `paddle` without calling `_ensure_win_dlls()`, breaking subsequent `torch` loading inside `albumentations.pytorch`:
     > `RuntimeError: PaddleOCR could not load: [WinError 127] The specified procedure could not be found. Error loading "C:\Users\ddat2\Downloads\Projects\pyvideotrans\.venv\lib\site-packages\torch\lib\shm.dll" or one of its dependencies.`

2. **False GPU Reporting (`videotrans/ocr/_paddle.py:182-215`)**
   - Despite `paddle.is_compiled_with_cuda() == False`, `select_device('auto')` and `_resolve_use_gpu('auto')` fall back to `torch.cuda.is_available()` (`True`) and return `'gpu'` / `use_gpu=True`.

3. **Language Caching Defect (`videotrans/ocr/_paddle.py:96`)**
   - `_require_engine` checks only `if self._engine is not None and self._active_device == device:`. After running `language="en"`, calling `recognize(..., language="zh-cn")` on the same `PaddleOcrProvider` reuses the English engine, yielding corrupted output:
     - `single_zh`: `'AXXT JiA#e'` (`conf=0.5078`)
     - `para_zh`: `''` (`num_lines: 0`)
     - `para_mixed`: `'PaddIeOCR j##iJE Python 3.10 5 CUDA 12.8  FX;EX ...'`

4. **Multi-Line Paragraph Line Dropping (`para_en` and `para_zh`)**
   - **4-line English paragraph (`para_en`, fresh `en` engine)**:
     - Ground truth: (1) `Artificial intelligence is transforming how creators localize video content`, (2) `across multiple languages. Modern speech recognition and optical character`, (3) `recognition pipelines must handle full paragraphs accurately without dropping`, (4) `lines or scrambling the natural reading order of the text.`
     - Actual `res.text`: `'Artificial intelligence is transforming how creators localize video content. recognition pipelines must handle full paragraphs accurately without dropping lines or scrambling the natural reading order of the text..'` (`num_lines: 3` — **Line 2 completely dropped!**).
   - **3-line Chinese paragraph (`para_zh`, fresh `zh-cn` engine)**:
     - Ground truth: (1) `人工智能正在改变创作者将视频内容本地化到多种语言的方式。`, (2) `现代语音识别和光学字符识别流程必须准确处理完整段落，`, (3) `而不能丢失行内容、破坏阅读顺序或在换行处插入多余空格。`
     - Actual `res.text`: `'人工智能正在改变创作者将视频内容本地化到多种语言的方式 而不能丢失行内容、破坏阅读顺序或在换行处插入多余空格'` (`num_lines: 2` — **Line 2 completely dropped, and CJK lines joined with ASCII space `' '`!**).
   - **Isolation Matrix (`gpu` × `cls` on `para_zh`)**:
     - `PaddleOCR(lang='ch', use_gpu=True, use_angle_cls=True, cls=True)` → `count=2` (Line 2 dropped, `2567.4 ms`)
     - `PaddleOCR(lang='ch', use_gpu=False, use_angle_cls=True, cls=True)` → `count=2` (Line 2 dropped, `2278.5 ms`)
     - `PaddleOCR(lang='ch', use_gpu=False, use_angle_cls=False, cls=False)` → **`count=3` (ALL 3 lines recovered at `conf=0.9979, 0.9966, 0.9928`, `2252.1 ms`)**
     - On `en_img`, `cls=False` likewise recovered all 4 lines, and `det_db_thresh=0.2, det_db_box_thresh=0.3, det_db_unclip_ratio=2.0` improved box coverage and punctuation (`0.9929–0.9985`).

5. **Reading-Order Scrambling on Jittered Same-Line & Two-Column Layouts**
   - **`jittered_sameline`** (left chunks at `y=40/90`, right chunks at `y=36/86`):
     - Detected boxes: `Right part of line one` `box=(423.0, 44.0, 277.0, 27.0)`, `Left part of line one` `box=(43.0, 49.0, 256.0, 24.0)`.
     - Actual `res.text`: `'Right part of line one Left part of line one Right part of line two Left part of line two'` (**backwards reading order on both lines**).
   - **`two_column`** (3 lines in col 1 at `x=40`, 3 lines in col 2 at `x=520`):
     - Actual `res.text`: `'Column 2 line A Column 1 line A Column 1 line B Column 2 line B Column 2 line C Column 1 line C'` (row-interleaved + col 2 ahead of col 1).

6. **Paragraph Formatting & `ocr_helpers.py` Flattening**
   - `videotrans/ocr/_paddle.py:264` joins all units with `" ".join(combined)`, erasing `\n` line breaks and inserting spaces inside wrapped CJK text.
   - `videotrans/api/ocr_helpers.py:140-185`: if an `ocr_runner` returns a `list` of line items, `observations.extend(ocr_res)` flattens individual lines so `representative_text(normalized_obs)` returns only **one** line for top-level `result["text"]`.
   - `videotrans/ocr/_paddle.py:318` (`if not poly:`) raises `ValueError: The truth value of an array with more than one element is ambiguous` whenever a PaddleOCR 3.x/PaddleX backend yields `np.ndarray` polygons.

### 3.3 Verified Root Causes & Exact R4 Fix Design

| Defect | Exact Source Location | Root Cause | Concrete R4 Fix |
|---|---|---|---|
| Middle-line dropping | `videotrans/ocr/_paddle.py:145, 230, 240`; `paddleocr/tools/infer/predict_system.py:133-155` | `use_angle_cls=True` + `cls=True` misclassifies horizontal paragraph lines as 180° rotated, flips crops upside-down, and `predict_system.py:152` drops low-confidence (`< 0.5`) garbage. | Set `OcrConfig.use_angle_cls: bool = False` (configurable) and pass `use_angle_cls=self._use_angle_cls` & `cls=self._use_angle_cls`; expose `det_db_*` tuning options. |
| Scrambled reading order & lost newlines | `videotrans/ocr/_paddle.py:247-268, 305-313`; `videotrans/ocr/_text.py:54-84` | Strict `(y, x)` tuple sort with zero vertical tolerance + `" ".join(combined)`. | Implement layout-aware **row-band clustering** (`\|yc_a - yc_b\| <= 0.5 * min(h_a, h_b)`), left-to-right `x0` sort within bands, **column gutter detection** (persistent vertical whitespace gap `> 1.5 * median_char_width` spanning ≥ 2 rows → column-first order), and CJK/Latin-aware joining (`""` vs `" "` within row, `"\n"` across rows; optional `paragraph_mode="unwrap"`). |
| Wrong language model reuse | `videotrans/ocr/_paddle.py:96` | Engine cache checks only `self._active_device == device`, ignoring `self._current_lang`. | Key engine cache by `(device, mapped_lang)` (small bounded dict, e.g., `maxsize=2` + `_active_lang` tracking). |
| False GPU report & `shm.dll` crash | `videotrans/ocr/_paddle.py:68-84, 182-215`; `videotrans/task/_stage_recogn.py:40` | `_resolve_use_gpu`/`select_device` fall back to `torch.cuda.is_available()`, and `select_device` skips `_ensure_win_dlls()`; `_stage_recogn.py:40` passes `device="cuda"` whenever PyTorch CUDA is available (`self.cfg.is_cuda=True`). | Call `_ensure_win_dlls()` in `select_device()`, `_resolve_use_gpu()`, and `is_available()`; check **only** `paddle.is_compiled_with_cuda()` + `paddle.device.cuda.device_count() > 0`, and ensure `PaddleOcrProvider.select_device("cuda")` falls back cleanly to `"cpu"` (reporting `"cpu"`) when `paddle.is_compiled_with_cuda()` is `False`. |
| `np.ndarray` truth-value crash | `videotrans/ocr/_paddle.py:318` | `if not poly:` fails on NumPy arrays. | Replace with `if poly is None or len(poly) == 0:` and normalize `np.ndarray` points cleanly. |
| Stage 2 single-line flattening | `videotrans/api/ocr_helpers.py:140-185` (`:141` & `:144`) | `observations.extend(ocr_res)` (`:141`) flattens lines before `representative_text`, and `frame_text = " ".join(texts)` (`:144`) collapses line breaks with ASCII spaces before `builder.observe()`. | Fix **both** `:141` and `:144` to join each frame's layout-ordered lines using CJK/Latin-aware `\n`/space joining (`order_lines` / `_join_ordered_lines`) so both per-frame `observations` (`representative_text`) and `SegmentBuilder` preserve multi-line paragraph structure. |

---

## Area 4: Resource-Efficient Video OCR Design Proposal (R5)

### 4.1 Baseline Empirical Measurements (6s 720p H.264 Clip)

Measured on a synthetic 6-second 1280×720 30fps H.264 clip (`[0..2s]` 2-line subtitle, `[2..4s]` 2-line subtitle, `[4..6s]` blank):
- **`scan_video_to_srt(..., coarse_interval_ms=500)`**:
  - Elapsed time: **`28.95s`** for a **6s** video (~4.8× slower than real-time).
  - Memory (RSS via `psutil`): **`95.6 MB` → `546.2 MB`** (**+450.6 MB**).
  - Segment boundaries: `[0..2000]` (`samples=2`) and `[2000..3500]` (`samples=4`) — **500ms boundary error** (`3500ms` vs `4000ms`), exceeding the `<= 250ms` tolerance in `docs/decisions/0001-video-ocr-as-subtitle-source.md`.
- **`extract_ocr_segment_text(video, 0.0, 4.0, ...)`**: **`28.54s`** (`samples=16`, OCR'd all frames with zero deduplication; top-level `text` dropped one of the two subtitles).
- **Frame Decode Micro-Benchmark (12 frames @ 500ms over 6s 720p)**:
  - Current `sampled_frame_iter` (`videotrans/ocr/_frame_source.py:87-99`): **`3208.3 ms`** (24 `ffmpeg` subprocesses: `frame_at` + `_probe_size` per frame).
  - `av` (PyAV 16.0.1) sequential decode + ROI crop: **`479.7 ms`** (**6.7× faster**, 0 subprocesses).
  - Single `ffmpeg` rawvideo crop pipe (`-vf fps=2,crop=...`): **`174.6 ms`** (**18.4× faster**, 1 subprocess total).
- **Verified Scanner/Pipeline Gaps**:
  - `_frame_source.py:55`: `frame_at()` spawns `_probe_size()` (`ffmpeg -i`) on every single frame.
  - `_scanner.py:208, 223-238, 264-271`: `_ocr_observe` on `t=0` (`:208`) never sets `self._last_ocr_ts = ts_ms`; `_refine(from_ts, to_ts, frame, roi, duration_ms)` (`:264-271`) re-OCRs the **exact same `frame`** from `to_ts` instead of decoding frames at `t`, never runs on `EMPTY -> ACTIVE` or `ACTIVE -> EMPTY`, invokes `self._builder.observe(ts_ms, ...)` **before** `_refine(...)` (`:229-238`, causing out-of-order probes after `_segment.py:152-157` closes/opens segments), and gates refinement on `(ts_ms - prev_ocr_ts) <= coarse_interval_ms + 200` (`:236`, which breaks once dedup skips frames).
  - `_scanner.py:160, 146, 326`: `self._builder.segments` doesn't exist (uses `_segments`), and `self._stats["cpu"]` is hardcoded to `"cpu"`.
  - `_stage_recogn.py:36-52`, `orchestrator.py:194, 234-253`, `task_params.py:117-130`, `jobs.py:110-113`: `_stage_recogn.py` never passes `stop_event` (`CancellationToken`) or `job_id` to `scan_video_to_srt` and skips setting `self.asr_duration` on early `video_ocr` return (`:51-52`); `orchestrator.py:194` (`relay()`) drops `type="ocr_progress"`, and `:234-253` (`find_project_by_audio_hash`) ignores `subtitle_source`/`ocr_roi`; `task_params.py:117-130` and `jobs.py:110-113` still enforce `recognType`/`asr_models`/`asr_validator` for `video_ocr` (`asr` and `vtv` jobs); `videotrans/configure/_paths.py:66` pins `OMP_NUM_THREADS=1` process-wide; `Stage1Prepare.tsx` and `cli.py` do not expose `subtitle_source="video_ocr"` or ROI selection after PySide6 removal.

### 4.2 Proposed Resource-Efficient Video OCR Architecture

#### 1. CPU/GPU Usage & Engine Lifecycle
- **CPU Performance Reality (`cpu_threads` vs. Pipeline Optimizations)**: Empirical CPU measurements on `PaddlePaddle 3.3.1` (where `enable_mkldnn=False, FLAGS_use_onednn=0` is required on CPU) alongside `videotrans/configure/_paths.py:66` (`OMP_NUM_THREADS=1`) showed negligible difference between `cpu_threads=4` and `cpu_threads=1` (`687.6 ms/call` vs `678.6 ms/call`, ~1% slower; `OMP=4` `672.8 ms` vs `668.5 ms`). Therefore, CPU speedups derive primarily from **frame deduplication**, **single-pipe ROI decoding**, and **cached provider reuse** rather than `cpu_threads` tuning (`cpu_threads` may remain configurable on `OcrConfig`, without relying on it for CPU speedup or spawning `GlobalProcessManager` copies that multiply ~350 MB model RAM).
- **Accurate GPU Dispatch & Safe CUDA Fallback**: Enable `use_gpu=True` solely when `paddle.is_compiled_with_cuda()` and `paddle.device.cuda.device_count() > 0`. In `PaddleOcrProvider.select_device("cuda")` (invoked by `_stage_recogn.py:40` when PyTorch CUDA sets `self.cfg.is_cuda=True`), fall back cleanly to `"cpu"` when `paddle.is_compiled_with_cuda()` is `False` and report the real `"cpu"`/`"gpu"` device into `_scanner.py` stats.
- **App-Scoped Bounded Thread-Safe Engine Cache**: Cache `PaddleOcrProvider` / internal `(device, mapped_lang)` engines (`maxsize=2` with explicit `release()` / unload hook) in `videotrans/ocr/__init__.py` / `app.state` so Stage 2 `/api/ocr/extract` clicks and sequential scans skip multi-second model initialization. Guard engine switching (`self._current_lang`, `self._engine`) and `engine.ocr()` calls on cached providers with a `threading.Lock` so concurrent Stage 2 `asyncio.to_thread` handlers and video scans never race.

#### 2. Memory Consumption (`< 50 MB` Scanner + Decoder RSS Overhead Target)
- **Zero Full-Frame Accumulation**: Probe `(width, height, fps, duration_ms)` **once** via `videotrans.util._ffprobe.get_video_info()` (or single `probe_video_meta`).
- **Pre-Allocated ROI Crop Buffer**: Stream only cropped & width-clamped ROI pixels (`max_roi_width=1280`) from `ffmpeg` stdout into a reusable pre-allocated `bytearray`/`np.ndarray` (`O(out_w * out_h * 3)` ≈ `< 1 MB` even for 4K sources). Never keep a list of decoded frames in RAM.
- **Bounded `SegmentBuilder` Window**: Cap retained per-segment observations in `SegmentBuilder` (`_open["observations"]`) to a bounded window (`N=16` highest-confidence/most-recent unique samples) so long static titles or watermarks never grow memory unboundedly over 30–120 min videos.

#### 3. Processing Time, Deduplication, Boundary Precision & Accuracy
- **Single-Pass Streaming ROI Decoder + `PyAV`/`frame_at_fast` Refinement**:
  - **Coarse Pass (Odd-ROI Safe)**: Place `format=bgr24` **before** `crop` in the `ffmpeg` filtergraph (e.g., `-vf "fps={fps},format=bgr24,crop={cw}:{ch}:{cx}:{cy},scale={out_w}:{out_h}:flags=bilinear"`) and compute exact `(out_w, out_h)` (even-aligned or explicit integers) in Python—never `-2` or post-crop scaling on `yuv420p`—so `ffmpeg` never silently truncates odd ROI heights (e.g., `173 -> 172` on 720p `yuv420p`) and `proc.stdout.read(out_w * out_h * 3)` never shears frames. Register the `Popen` via `videotrans.core.proc_registry.register_proc` for instant cancellation; fall back to `PyAV` sequential decode when `ffmpeg` pipe is unavailable. Optional keyframe/scene-change hints may be used when `coarse_interval_ms >= 1000`.
  - **Refinement (`frame_at_fast`)**: Lightweight random-access `PyAV` keyframe-seek (or single `ffmpeg -ss ...` with `format=bgr24` before `crop`, without `_probe_size`) invoked **only** during boundary refinement.
- **Two-Stage Localized Frame Deduplication (`< 0.2 ms/frame` CPU)**:
  - Global `64x16` MAD/edge gating (`0.012`/`0.015`) with a dropped heartbeat empirically skips real 1-digit/1-char transitions (`'Wait 10s' -> 'Wait 20s'` measured `dhash_ham=1, mad=0.00030, edge_mad=0.00039`; `'Part A' -> 'Part B'` measured `dhash_ham=0, mad=0.00033, edge_mad=0.00059`).
  - Instead, partition the ROI into a **localized tile grid** (e.g., `8x4` or `16x4` sub-tiles over luma + horizontal edge `|grad_x|`) and skip OCR only when `dhash_hamming <= 2` AND `max(tile_mad) < tile_mad_thresh` AND `max(tile_edge_mad) < tile_edge_thresh` AND `max_pixel_diff < px_thresh`, paired with a bounded periodic confirmation heartbeat (e.g., every `N` steps) so single-character transitions are never dropped.
  - On dedup-skipped frames (`_scanner.py:223-225`), advance `SegmentBuilder._open["last_ts"]` and `_last_active_ts` (when active) and evaluate `grace_gap_ms` expiry (when an empty gap is pending) so skipping static frames never freezes segment end timestamps.
  - Maintain a **bounded LRU OCR result cache (`maxsize=64`)** keyed by quantized `(dhash64, thumb_sha1)` to short-circuit flickering/oscillating frames.
- **Pre-Commit Bidirectional Binary-Search Boundary Refinement (`<= 250 ms` Precision)**:
  - Fix `_ocr_observe` at `t=0` (`_scanner.py:208`) to set `self._last_ocr_ts = ts_ms`.
  - Track `last_coarse_ts` (the most recent coarse sample timestamp, **including** dedup-skipped frames) and bracket refinement on `[last_coarse_ts, ts_ms]` rather than `prev_ocr_ts` (`_scanner.py:236`).
  - Execute binary-search refinement (`~2` probes via `frame_at_fast(t)` for a `500ms` window → `<= 125ms` midpoint error) across **all three** transition types (`EMPTY -> ACTIVE`, `ACTIVE -> EMPTY`, `ACTIVE -> ACTIVE`) **before** committing the transition to `SegmentBuilder.observe()` (`_scanner.py:229-238` vs `_segment.py:152-157`), so probes never execute out of chronological order after segment close/open.
- **Accuracy Safeguards**:
  - Upscale tiny ROIs (`roi_h < 48px`) 2× (`cv2.INTER_CUBIC`) before OCR; preserve `\n` across distinct visual lines in `SegmentBuilder` / `OcrResult`; filter persistent static ROI watermarks/channel logos spanning `> 85%` of total video duration when subtitle cues vary (`docs/plans/completed/video-ocr-subtitles.md:246-248`).
- **Stage 2 Speedup (`extract_ocr_segment_text` in `videotrans/api/ocr_helpers.py`)**:
  - Reuse the `format=bgr24`-before-`crop` `ffmpeg` ROI stream + localized 2-stage tile dedup + thread-safe cached `PaddleOcrProvider` inside `extract_ocr_segment_text`, cutting a 4s Stage 2 repair from **~28.5s (16 OCR calls)** down to **~2–3s (1–2 OCR calls)** while preserving multi-line paragraph output across `:141` and `:144`.

#### 4. Cancellation, Progress Telemetry & WebUI/API/CLI Integration
- **Cancellation, `asr_duration` & Process Safety**: Pass `CancellationToken.is_cancelled` as `stop_event` and thread `job_id` (defaulting to `job_id = job_id or get_current_job_id()` inside `scan_video_to_srt`) from `RecognMixin.recogn()` (`_stage_recogn.py:36-45`) so `videotrans.core.proc_registry` kills the streaming `ffmpeg` `Popen` immediately on `/api/jobs/{job_id}/cancel`; record `self.asr_duration = round(time.time() - _st, 2)` before returning at `_stage_recogn.py:51-52`; emit structured `ocr_progress` telemetry (`percent`, `timestamp_ms`, `segments`, `ocr_calls`, `skipped_frames`, `device`) and forward `raw_type == "ocr_progress"` in `videotrans/task/orchestrator.py:194` (`relay()`).
- **API, Orchestrator & Task Pipeline (`videotrans/api/task_params.py`, `routes/jobs.py`, `orchestrator.py`, `_stage_prepare.py`)**:
  - Accept `subtitleSource` (`"audio_asr" | "video_ocr"`), `ocrRoi` (`[x, y, w, h]`), `ocrRoiConfirmed` (`bool`), and `ocrOptions` (`coarseIntervalMs`, `useAngleCls`, `paragraphMode`) in `build_task_params()`.
  - Bypass `recognType`/`asr_models` validation in `videotrans/api/task_params.py:117-130` **and** skip `asr_validator` in `videotrans/api/routes/jobs.py:110-113` whenever `subtitle_source == "video_ocr"` (covering **both** `job_type == "asr"` and `job_type == "vtv"`).
  - Bypass or scope `find_project_by_audio_hash` in `videotrans/task/orchestrator.py:234-253` when `subtitle_source == "video_ocr"` so prior `audio_asr` or different-ROI transcripts are never reused; allow audio-less video files in `_stage_prepare.py` when `subtitle_source == "video_ocr"`.
- **WebUI Stage 1 (`frontend/src/screens/Stage1Prepare.tsx`, `frontend/src/store/prepareSlice.ts`)**:
  - Add a **Subtitle Source** selector (`Audio Speech Recognition (ASR)` vs. `Hard-Subtitle Video OCR`) with a compact ROI drag-overlay on the Stage 1 video preview and a **"Test Frame OCR"** button calling `/api/ocr/extract` on a 1s window around `video.currentTime` to confirm `ocrRoiConfirmed` before enabling `Start Dub`.
- **CLI (`cli.py:315-486`)**:
  - Add `--subtitle_source` (`audio_asr` | `video_ocr`), `--ocr_roi` (`x,y,w,h`), and `--ocr_interval_ms` to `build_parser()`, `build_stt_params()`, and `build_vtv_params()` (auto-setting `ocr_roi_confirmed=True` when valid `--ocr_roi` is provided).

---

## Area 5: Prioritized Recommendations & Implementation Roadmap (P0/P1/P2 → M2–M7)

| Priority | Milestone | Work Package | Key Target Files | Acceptance & Verification Criteria |
|---|---|---|---|---|
| **P0** | **M2** | **R2 UI/UX Fixes & Project State Integrity** (all 15 defects + `test_webui.py` alignment) | `frontend/src/{components,screens,store,config}/*`, `videotrans/core/{project_store.py,stage_reset.py}`, `videotrans/api/{routes/projects.py,task_params.py}`, `videotrans/params.json` | Local `YYYY-MM-DD` date-parent parity; `update_project_state` (`merge=True`) key-level merge + `stage_reset.py` (`merge=False` / `ocrCrop=None`); idempotent `startBatchPrepareJobs`; accurate `ProviderDocLink`, `StatusFooter`, `Header`, `VideoPlayer`, `Stage2OcrDialog`, `Stage3VoiceDubbing`, `VoiceManagementScreen`, `CreateVoiceModal`, `Stage4EditVideo`, `CapCutExportModal`, `SettingsModal`; `bun test` + `bun run build` + `test_webui.py` pass |
| **P0 / P1** | **M3** | **R3a STT Correctness, Audio Prep & Temp Hygiene** | `videotrans/recognition/{_deepgram.py,_base.py,_whisper.py,_elevenlabs.py,_gemini.py,_google.py}`, `videotrans/task/{_stage_prepare.py,_stage_recogn.py}`, `videotrans/process/{vad.py,stt_qwen.py}` | Null-safe `detect_language`, `asr_duration` on `speaker.json`, single `441000_ac2_raw.wav` extraction when `is_separate=True` with `vocal_bgm` success/fallback resampling to `source_wav`, header-only duration check, `cut_audio` inside `cache_folder`, `faster-*.srt` cleanup, `_merge_sub` guard, vectorized Ten-VAD, Qwen lang passthrough, client close/timeouts/retries; reproducible benchmark + unit tests pass |
| **P0 / P1** | **M4** | **R3b AI Translation Cache, Parsing, Concurrency & Prompt Efficiency** | `videotrans/translator/{_base.py,_openaicompat.py,_gemini.py,_google.py}`, `videotrans/util/prompts.py`, `videotrans/prompts/`, `videotrans/api/routes/stages.py`, `videotrans/task/orchestrator.py` | Symmetric SRT cache key (verified >0% hit rate), multiline `<think>` + unclosed tag parsing, `aitrans_context` sliding window & `None` guard, `Google` URL-encode/TLS/timeout, bounded `ThreadPoolExecutor` + pooled clients (`threading.Lock`-guarded `Gemini.api_keys` + per-key `genai.Client`), runtime retry loops/dynamic callbacks keeping transient `429`/5xx backoff active even with `"retry_nums": 1`, static `system` prompt split + per-batch `{GLOSSARY_DICT}` in `user` + compact payload, `/api/translate` cleanup & progress |
| **P0 / P1 / P2** | **M5** | **R3c TTS Serialization Removal, Batch Dubbing & NumPy Assembly** | `videotrans/services/voice_preview.py`, `videotrans/api/routes/{voices.py,projects.py}`, `frontend/src/store/dubbingSlice.ts`, `videotrans/tts/{_elevenlabs.py,_geminitts.py,_vieneutts.py,_omnivoice.py,_base.py}`, `videotrans/task/_rate.py` | Per-engine locks + `asyncio.to_thread` for cloud previews; Stage 3 batch dubbing endpoint (`VieNeu` `max_batch_size=4`); NumPy vectorized 48 kHz mono audio assembly with `sr != 48000` resampling & mono downmix replacing $O(N \cdot T)$ `pydub` overlay; client/engine reuse, temp WAV/MP3 cleanup, header-only duration in `_rate.py`, bounded LRU/TTL preview & dubbing caches |
| **P0 / P1** | **M6** | **R4 OCR Paragraph Fix & R5 Resource-Efficient Video OCR** | `videotrans/ocr/{_paddle.py,_text.py,_frame_source.py,_scanner.py,_segment.py,__init__.py}`, `videotrans/api/{ocr_helpers.py,task_params.py,routes/jobs.py,routes/stages.py}`, `videotrans/task/{_stage_recogn.py,_stage_prepare.py,orchestrator.py}`, `frontend/src/{screens/Stage1Prepare.tsx,store/prepareSlice.ts}`, `cli.py` | Multi-line EN/ZH/mixed/jittered/2-column tests pass with zero dropped lines & proper `\n`/reading order (`ocr_helpers.py:141, 144`); `(device, lang)` cache, accurate CUDA check + `select_device("cuda")` CPU fallback, `threading.Lock` on cached provider; `format=bgr24`-before-`crop` pipe + `frame_at_fast` pre-commit binary-search refinement (`<=250ms`) bracketing `[last_coarse_ts, ts_ms]`; localized max-tile `8x4`/`16x4` dedup + `<50MB` RSS overhead + bounded LRU cache + `CancellationToken`/`proc_registry` (`job_id`) + `orchestrator.py` `ocr_progress`/hash bypass + `task_params.py`/`jobs.py` bypass (`asr` & `vtv`) + `_stage_recogn.py` `asr_duration`; Stage 2 speedup; Stage 1 WebUI + CLI `video_ocr` end-to-end verified |
| **Final** | **M7** | **E2E Opaque-Box Suite (Tiers 1–4) & Tier 5 Hardening** | `tests/`, `frontend/tests/`, `docs/plans/completed/` | 100% pass across E2E Tiers 1–4 + Tier 5 adversarial coverage + full regression (`bun test`, `bun run build`, `pytest`) + move active plan to `docs/plans/completed/` |
