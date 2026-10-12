# Execution Plan: Date Parent Project Grouping and Sub-Project Hierarchy

Date: 2026-10-09

## Status

Completed

## Outcome

Automatically organize all video dubbing projects into daily parent projects named after the date (`YYYY-MM-DD`). In both the Project Management screen and workflow sessions, display parent projects as date containers that group their sub-project videos, allow batch session activation, enable adding/scanning videos directly into a date parent project, and auto-assign single and batch video uploads into today's parent project.

## Context

- `AGENTS.md` & `docs/WORKFLOW.md`: Durable plan lifecycle and authority standards.
- `videotrans/core/project_store.py`: `get_or_create_date_parent_project`, `list_subprojects`, `list_projects`, cascading `delete_project`.
- `videotrans/api/routes/projects.py`: `/api/projects`, `/api/projects/batch-create`, `/api/projects/today-parent`.
- `frontend/src/screens/ProjectsScreen.tsx`: Project Management dashboard (grid and table views with date group containers).
- `frontend/src/store/projectSlice.ts`: Project list, parent project tracking, `openDateParentSession`, `sessionSubprojects`.
- `frontend/src/store/prepareSlice.ts`: Video upload and project initialization in Stage 1.
- `frontend/src/components/SessionVideosDrawer.tsx`: Multi-video drawer across stages.

## Scope

In scope:
- Backend: Auto-assigning projects to today's date parent project if `parent_id` is not explicitly provided on creation.
- Backend: Ensuring today's parent project exists upon listing or querying `/api/projects` or `/api/projects/today-parent`.
- Backend: Cascading deletion of subprojects and their artifact directories when deleting a parent project.
- Frontend store: Automatically ensuring today's parent project is tracked, updating `activeParentProjectId` and syncing `sessionSubprojects` when switching projects.
- Frontend UI (`ProjectsScreen.tsx`): Introducing a Date Parent Project grouped view (active by default) where each day forms a folder/container with badge, video count, progress summary, and sub-project cards/rows nested inside, with a fallback toggle to flat view.
- Frontend UI: Adding "Open Day Session" and "Add Video to Date" actions to parent projects.
- Automated tests: Backend proof for auto-parenting, cascading delete, and today parent retrieval; frontend proof for date group rendering and subproject interaction.

Out of scope:
- Modifying Qt GUI components (WebUI only).
- Changing subtitle or TTS rendering engines.

## Approach

1. **Backend Auto-Parenting & Today Parent Support**:
   - In `videotrans/api/routes/projects.py`:
     - Updated `create_project_handler` so any incoming project without an explicit `parentId` (and not a parent project itself) automatically defaults to `get_or_create_date_parent_project()["id"]`.
     - Added `GET /api/projects/today-parent` so clients can easily fetch or ensure today's parent project.
     - In `delete_project`: ensured deleting a parent project cascades to its subprojects in both DB and disk.
2. **Frontend Store & Session Synchronization**:
   - In `frontend/src/store/projectSlice.ts`:
     - In `createNewProject`: ensured `parentId` defaults to today's date parent project and updates `sessionSubprojects`.
     - In `selectProject`: when opening a subproject, automatically loads all siblings from its `parent_id` into `sessionSubprojects` so the user can seamlessly switch between videos of that day.
     - Implemented `openDateParentSession` to load subprojects, activate the session drawer, and switch to dubbing view.
3. **ProjectsScreen Grouped by Date**:
   - Updated `ProjectsScreen.tsx`:
     - Added group-by-date logic that groups subprojects under their parent project date (`YYYY-MM-DD`).
     - Rendered parent project header with date, "Today" badge, subproject count, progress summary, and actions ("Open Day Session", "+ Add Video", "Scan Folders").
     - Rendered subproject cards nested within their parent project container in grid view, and nested/grouped rows in table view.
     - Provided a "Group by Day" / "Flat List" toggle for user preference.
4. **Validation**:
   - Ran backend tests with pytest.
   - Ran frontend tests with bun test.
   - Tested production build with bun run build.

## Risks And Recovery

- Risk: Existing unit tests expecting unparented projects or flat project list.
  - Mitigation: Subprojects remain accessible via direct `id` queries and in `list_projects`; flat view mode remains supported via toggle.
- Recovery: If date grouping causes issues with legacy data, ungrouped items gracefully render under their creation date or fallback section.

## Progress

- [x] Backend: auto-parenting and today parent project endpoint.
- [x] Backend: cascading delete for parent projects.
- [x] Frontend store: auto-parenting on single upload and sibling synchronization in `sessionSubprojects`.
- [x] Frontend UI: Date Parent Project grouping in `ProjectsScreen.tsx` for both grid and table views.
- [x] Validation: backend unit tests and frontend test suite.

## Decisions

- 2026-10-09: Auto-assign all single and batch video creations to today's date parent project (`parent_YYYY-MM-DD`) unless explicitly overridden.
- 2026-10-09: In `ProjectsScreen`, make Date Parent Project grouping the default view while preserving a Flat List toggle.

## Validation

- Focused proof: `pytest tests/test_parent_project_store.py` (3 passed), `pytest tests/test_webui_projects.py` (6 passed).
- Integration proof: `bun test tests/batchWorkflow.test.tsx` (8 passed), full `bun test` (167 passed across 11 files).
- Build proof: `bun run build` completed with zero TypeScript errors.

## Result

Completed and verified. All video projects are grouped into daily parent projects named after the date, with hierarchical folder containers in the UI, seamless multi-video session switching, and direct output routing to parent project directories.
