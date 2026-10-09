# Execution Plan: Batch Video Upload, Date Grouping & Multi-Video Workflow

Date: 2026-10-09

## Status

Completed

## Outcome

Supported recursive multi-folder video scanning, date-based parent project grouping (`output/projects/YYYY-MM-DD/`), batch Stage 1 to Stage 2 processing with a managed concurrency queue (limit = 2), a persistent session videos status drawer across Stages 1–4 with non-blocking switching, and direct output routing to the parent project directory.

## Context

- `AGENTS.md` and `docs/WORKFLOW.md`: Durable plan lifecycle and authority standards.
- `videotrans/core/db.py`: Database schema for projects, jobs, events with dynamic `parent_id` migration.
- `videotrans/core/project_store.py`: Project lifecycle and directory hierarchy with parent project routing.
- `videotrans/core/media_store.py`: Media inspection and storage.
- `videotrans/core/job_manager.py`: Background job lifecycle and multi-video execution.
- `videotrans/api/task_params.py`: Output directory and target parameter mapping (`parent_id` -> parent folder).
- `frontend/src/store/projectSlice.ts` & `jobSlice.ts`: Frontend project and job state management with session subproject switching and concurrency queue.
- `frontend/src/screens/Stage1Prepare.tsx`: Batch mode banner, folder scan modal triggers, and media configuration.

## Scope

In scope:
- Recursive scanning of local folders and HTML5 directory uploads for `.mp4` video files.
- Detected video preview list with selection/deselection controls.
- Date-named parent project (`YYYY-MM-DD`) grouping and database migration for `parent_id`.
- Stage 1 batch configuration with automated execution up to Stage 2 using a managed concurrency queue (concurrency = 2).
- Collapsible session video status drawer toggleable across Stages 1–4 with non-blocking switching.
- Direct output file writing to `output/projects/YYYY-MM-DD/[video_stem]_...`.

Out of scope:
- Modifying Qt/desktop GUI components (pure WebUI focus).
- Redesigning subtitle editing tools or audio mixer beyond parent output integration.

## Approach

1. **Database & Project Store**:
   - Added `parent_id` column to `projects` table with migration in `videotrans/core/db.py`.
   - Implemented date parent project helpers (`get_or_create_date_parent_project`, `list_subprojects`) in `project_store.py`.
   - Updated `task_params.py` and export endpoints to place exports in parent project directory.
2. **Media Discovery API**:
   - Added `/api/media/scan-folders` and `/api/media/register-local` in `videotrans/api/routes/media.py`.
   - Added `/api/projects/batch-create` in `videotrans/api/routes/projects.py`.
3. **Frontend Store & Batch Queue**:
   - Updated `projectSlice.ts` to manage parent projects, session subprojects, and active video switching.
   - Updated `jobSlice.ts` to support per-project job tracking and concurrent queue orchestration.
4. **UI Components**:
   - Implemented `FolderScanModal.tsx` for folder selection and video preview/filtering.
   - Implemented `SessionVideosDrawer.tsx` for cross-stage video monitoring and switching.
   - Integrated drawer toggle into `Header.tsx` and batch mode into `Stage1Prepare.tsx`.
5. **Validation**:
   - Added backend tests for folder scanner, date project store, and batch queue.
   - Added frontend tests for scan modal, session drawer, and switching.
   - Ran full test suite and verified clean pass.

## Risks And Recovery

- **Risk**: Concurrent ASR jobs exceeding GPU VRAM for local Whisper/CUDA models.
  - **Mitigation**: Managed concurrency queue capped at 2 by default, with sequential fallback.
- **Risk**: Database schema backward compatibility with existing projects without `parent_id`.
  - **Mitigation**: Schema migration check in `init_db()` using `ALTER TABLE` if column is missing.
- **Recovery**: Existing standalone projects without `parent_id` continue operating seamlessly under their individual folders.

## Progress

- [x] Design alignment with user via interview (/grill-me).
- [x] Implementation plan and prompt draft artifacts generated.
- [x] Backend: database migration for `parent_id` and date parent project store functions.
- [x] Backend: folder scanning and local media registration endpoints.
- [x] Backend: parent project export routing in `task_params.py` and project routes.
- [x] Frontend: API client methods and store updates for multi-video tracking.
- [x] Frontend: `FolderScanModal` component with preview table and checkboxes.
- [x] Frontend: `SessionVideosDrawer` component with real-time status and non-blocking switching.
- [x] Frontend: Stage 1 batch dubbing trigger and queue orchestration.
- [x] Validation: automated tests and end-to-end verification.

## Decisions

- 2026-10-09: Support both local folder path inputs (fast local scanning) and HTML5 browser folder upload/drag-and-drop.
- 2026-10-09: Group by date `YYYY-MM-DD`; if parent project already exists for today, append new videos as subprojects.
- 2026-10-09: Use a managed concurrency queue (concurrency 2) for Stage 1 -> Stage 2 ASR, unlocking each video for review as soon as its individual ASR completes.
- 2026-10-09: Save generated files directly into `output/projects/YYYY-MM-DD/` with `[video_name]` prefix.
- 2026-10-09: Provide a collapsible side drawer available across all stages (1 to 4) for monitoring and switching videos.

## Validation

- Focused backend proof: `tests/test_folder_scanner.py`, `tests/test_parent_project_store.py` (all passed).
- Frontend proof: `bun test tests/batchWorkflow.test.tsx` (6 passed), full `bun test` (165 passed across 11 test suites).
- WebUI regression proof: `pytest tests/test_webui.py` (20 passed), `pytest tests/test_webui_projects.py tests/test_project_store.py` (all 14 passed).
- Build proof: `bun run build` completed with zero TypeScript errors.

## Result

Completed and verified.
