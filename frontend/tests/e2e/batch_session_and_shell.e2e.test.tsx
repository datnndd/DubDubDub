import React from 'react';
import { beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { useDubDubStore } from '../../src/store';
import { Header } from '../../src/components/Header';
import { WorkflowStepper } from '../../src/components/WorkflowStepper';
import { StatusFooter } from '../../src/components/StatusFooter';
import { VideoPlayer } from '../../src/components/VideoPlayer';
import { ProviderDocLink } from '../../src/components/ProviderDocLink';
import { getProviderDocInfo } from '../../src/config/providerDocs';
import { FolderScanModal } from '../../src/components/FolderScanModal';
import { SessionVideosDrawer } from '../../src/components/SessionVideosDrawer';
import { ProjectsScreen } from '../../src/screens/ProjectsScreen';
import { SettingsModal } from '../../src/components/settings/SettingsModal';
import { saveProviderAndLoadModels } from '../../src/api/settingsApi';
import { scanFolders, registerLocalMedia } from '../../src/api/media';
import { batchCreateProjects } from '../../src/api/projects';
import {
  flushMicrotasks,
  flushTasks,
  formatLocalYmd,
  jsonResponse,
  MockEventSource,
  resetStore,
  withE2EShims,
  withFakeUtcPlus7Clock,
} from './helpers';

beforeEach(() => {
  resetStore();
});

// ============================================================================
// F1: Batch Date-Parent Grouping & Folder Scan (Tier 1 >= 5, Tier 2 >= 5)
// ============================================================================
describe('F1 — Batch Date-Parent Grouping & Folder Scan', () => {
  // --- Tier 1 ---
  test('[Tier 1] openDateParentSession loads subprojects, opens drawer, and enters Stage 1 dubbing view', async () => {
    const today = formatLocalYmd();
    const parentId = `parent_${today}`;
    const subs = [
      { id: 'sub_1', name: 'ep1.mp4', parent_id: parentId, stage: 1, status: 'pending', media_id: 'med_1' },
      { id: 'sub_2', name: 'ep2.mp4', parent_id: parentId, stage: 2, status: 'completed', media_id: 'med_2' },
    ];

    await withE2EShims(async (url) => {
      if (url.includes(`/api/projects/${parentId}/subprojects`)) return jsonResponse({ subprojects: subs });
      if (url.includes('/api/projects/sub_1')) return jsonResponse({ project: subs[0] });
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().openDateParentSession(parentId, today);
      const s = useDubDubStore.getState();
      expect(s.activeView).toBe('dubbing');
      expect(s.currentStep).toBe(1);
      expect(s.activeParentProjectId).toBe(parentId);
      expect(s.sessionDrawerOpen).toBe(true);
      expect(s.batchModeActive).toBe(true);
      expect(s.sessionSubprojects.map((p) => p.id)).toEqual(['sub_1', 'sub_2']);
      expect(s.activeProjectId).toBe('sub_1');
    });
  });

  test('[Tier 1] batch folder scan -> registerLocalMedia -> batchCreateProjects populates sessionSubprojects', async () => {
    const today = formatLocalYmd();
    const parentId = `parent_${today}`;

    await withE2EShims(async (url, _init, call) => {
      if (url.includes('/api/media/scan-folders')) {
        return jsonResponse({
          ok: true,
          videos: [
            { path: '/videos/a.mp4', filename: 'a.mp4', sizeBytes: 1024, parentDir: '/videos' },
            { path: '/videos/b.mp4', filename: 'b.mp4', sizeBytes: 2048, parentDir: '/videos' },
          ],
        });
      }
      if (url.includes('/api/media/register-local')) {
        return jsonResponse({
          ok: true,
          media: [
            { id: 'm_a', filename: 'a.mp4', durationMs: 12000 },
            { id: 'm_b', filename: 'b.mp4', durationMs: 18000 },
          ],
        });
      }
      if (url.includes('/api/projects/batch-create')) {
        expect(call.body.date).toBe(today);
        expect(call.body.items).toHaveLength(2);
        return jsonResponse({
          ok: true,
          parent: { id: parentId, name: today, stage: 1, status: 'active', state: { is_parent: true } },
          projects: [
            { id: 'p_a', name: 'a.mp4', parent_id: parentId, media_id: 'm_a', stage: 1, status: 'pending', duration: 12 },
            { id: 'p_b', name: 'b.mp4', parent_id: parentId, media_id: 'm_b', stage: 1, status: 'pending', duration: 18 },
          ],
        });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      const scan = await scanFolders(['/videos'], true);
      expect(scan.videos).toHaveLength(2);
      const reg = await registerLocalMedia(scan.videos.map((v) => v.path));
      const batch = await batchCreateProjects({
        date: today,
        items: reg.media.map((m) => ({ name: m.filename, mediaId: m.id, duration: (m.durationMs || 0) / 1000 })),
      });
      useDubDubStore.getState().setSessionSubprojects(batch.projects, batch.parent.id, batch.parent.name);
      const s = useDubDubStore.getState();
      expect(s.sessionSubprojects).toHaveLength(2);
      expect(s.activeParentProjectId).toBe(parentId);
      expect(s.batchModeActive).toBe(true);
    });
  });

  test('[Tier 1] ProjectsScreen renders grouped date-parent containers with counts and stage badges', () => {
    const today = formatLocalYmd();
    useDubDubStore.setState({
      projectsList: [
        { id: `parent_${today}`, name: today, stage: 1, status: 'active', state: { is_parent: true } },
        { id: 'v1', name: 'Lesson 01.mp4', parent_id: `parent_${today}`, stage: 2, status: 'completed', duration: 90 },
        { id: 'v2', name: 'Lesson 02.mp4', parent_id: `parent_${today}`, stage: 1, status: 'pending', duration: 45 },
      ] as any,
    });

    const html = renderToStaticMarkup(<ProjectsScreen />);
    expect(html).toContain(`data-date-group="${today}"`);
    expect(html).toContain('2 Videos');
    expect(html).toContain('1 Ready');
    expect(html).toContain('Lesson 01.mp4');
    expect(html).toContain('Lesson 02.mp4');
    expect(html).toContain(`data-testid="open-session-${today}"`);
  });

  test('[Tier 1] SessionVideosDrawer lists subprojects with active indicator and readiness summary', () => {
    useDubDubStore.setState({
      sessionDrawerOpen: true,
      activeProjectId: 'v1',
      activeParentProjectName: '2026-11-18',
      sessionSubprojects: [
        { id: 'v1', name: 'part_1.mp4', stage: 1, status: 'pending', media_id: 'm1' },
        { id: 'v2', name: 'part_2.mp4', stage: 2, status: 'completed', media_id: 'm2' },
      ] as any,
    });

    const html = renderToStaticMarkup(<SessionVideosDrawer />);
    expect(html).toContain('Session: 2026-11-18');
    expect(html).toContain('part_1.mp4');
    expect(html).toContain('part_2.mp4');
    expect(html).toContain('Active');
    expect(html).toContain('50%');
  });

  test('[Tier 1] ProjectsScreen supports alphabetical flat sort and compact Table View', () => {
    useDubDubStore.setState({
      projectsList: [
        { id: 'p_z', name: 'Zulu Video.mp4', stage: 1, status: 'pending', duration: 10 },
        { id: 'p_a', name: 'Alpha Video.mp4', stage: 2, status: 'completed', duration: 20 },
      ] as any,
    });

    const alphaHtml = renderToStaticMarkup(<ProjectsScreen initialSortBy="alphabetical" />);
    expect(alphaHtml.indexOf('Alpha Video.mp4')).toBeLessThan(alphaHtml.indexOf('Zulu Video.mp4'));
    expect(alphaHtml).not.toContain('data-date-group=');

    const tableHtml = renderToStaticMarkup(<ProjectsScreen initialViewMode="table" />);
    expect(tableHtml).toContain('data-testid="projects-table-view"');
    expect(tableHtml).toContain('Project Name');
    expect(tableHtml).toContain('Alpha Video.mp4');
  });

  // --- Tier 2 ---
  test('[Tier 2] midnight UTC+7 boundary uses local YYYY-MM-DD (2026-11-18), never UTC yesterday (2026-11-17)', async () => {
    await withFakeUtcPlus7Clock(async () => {
      // Sanity check: simulated clock is 2026-11-18 00:15:00+07:00 == 2026-11-17T17:15:00Z
      expect(formatLocalYmd(new Date())).toBe('2026-11-18');
      expect(new Date().toISOString().split('T')[0]).toBe('2026-11-17');

      let postedParentId: string | undefined;
      await withE2EShims(async (url, init, call) => {
        if (url.endsWith('/api/projects') && init?.method === 'POST') {
          postedParentId = call.body.parentId || call.body.parent_id;
          return jsonResponse({ id: 'p_local', name: 'Midnight Clip.mp4', parent_id: postedParentId, stage: 1 }, 201);
        }
        if (url.includes('/api/projects/p_local')) {
          return jsonResponse({ id: 'p_local', name: 'Midnight Clip.mp4', parent_id: postedParentId, stage: 1 });
        }
        if (url.includes('/subprojects')) return jsonResponse({ subprojects: [] });
        return jsonResponse({ projects: [] });
      }, async () => {
        await useDubDubStore.getState().createNewProject('Midnight Clip.mp4');
        expect(postedParentId).toBe('parent_2026-11-18');
      });

      // ProjectsScreen & FolderScanModal must also display local 2026-11-18
      useDubDubStore.setState({
        scanModalOpen: true,
        projectsList: [
          { id: 'parent_2026-11-18', name: '2026-11-18', stage: 1, status: 'active', state: { is_parent: true } },
          { id: 'p_local', name: 'Midnight Clip.mp4', parent_id: 'parent_2026-11-18', stage: 1, status: 'pending', duration: 15 },
        ] as any,
      });
      const projectsHtml = renderToStaticMarkup(<ProjectsScreen />);
      expect(projectsHtml).toContain('data-date-group="2026-11-18"');
      // Should NOT seed yesterday UTC (2026-11-17) as Today
      expect(projectsHtml).not.toContain('data-date-group="2026-11-17"');
      expect(projectsHtml).toContain('>Today<');
    });
  });

  test('[Tier 2] duplicate filenames across subfolders (ep1/video.mp4 vs ep2/video.mp4) are both preserved', async () => {
    const parentId = 'parent_2026-11-18';
    await withE2EShims(async (url, _init, call) => {
      if (url.includes('/api/media/scan-folders')) {
        return jsonResponse({
          ok: true,
          videos: [
            { path: '/series/ep1/video.mp4', filename: 'video.mp4', sizeBytes: 1000, parentDir: '/series/ep1' },
            { path: '/series/ep2/video.mp4', filename: 'video.mp4', sizeBytes: 1200, parentDir: '/series/ep2' },
          ],
        });
      }
      if (url.includes('/api/media/register-local')) {
        expect(call.body.paths).toEqual(['/series/ep1/video.mp4', '/series/ep2/video.mp4']);
        return jsonResponse({
          ok: true,
          media: [
            { id: 'med_ep1', filename: 'video.mp4', path: '/series/ep1/video.mp4', durationMs: 10000 },
            { id: 'med_ep2', filename: 'video.mp4', path: '/series/ep2/video.mp4', durationMs: 12000 },
          ],
        });
      }
      if (url.includes('/api/projects/batch-create')) {
        expect(call.body.items).toHaveLength(2);
        return jsonResponse({
          ok: true,
          parent: { id: parentId, name: '2026-11-18', stage: 1 },
          projects: [
            { id: 'sub_ep1', name: 'ep1/video.mp4', media_id: 'med_ep1', media_path: '/series/ep1/video.mp4', stage: 1, status: 'pending' },
            { id: 'sub_ep2', name: 'ep2/video.mp4', media_id: 'med_ep2', media_path: '/series/ep2/video.mp4', stage: 1, status: 'pending' },
          ],
        });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      const scan = await scanFolders(['/series'], true);
      expect(scan.videos).toHaveLength(2);
      const reg = await registerLocalMedia(scan.videos.map((v) => v.path));
      const batch = await batchCreateProjects({
        date: '2026-11-18',
        items: reg.media.map((m) => ({ name: m.filename, mediaId: m.id, mediaPath: m.path })),
      });
      useDubDubStore.getState().setSessionSubprojects(batch.projects, parentId, '2026-11-18');
      const subs = useDubDubStore.getState().sessionSubprojects;
      expect(subs).toHaveLength(2);
      expect(subs.map((s) => s.media_path)).toEqual(['/series/ep1/video.mp4', '/series/ep2/video.mp4']);

      // Also check exported path-dedup helper if provided by M2 FolderScanModal
      const modalMod: any = await import('../../src/components/FolderScanModal');
      if (typeof modalMod.deduplicateScannedVideos === 'function') {
        const deduped = modalMod.deduplicateScannedVideos(
          [{ path: 'ep1/video.mp4', filename: 'video.mp4', sizeBytes: 100, parentDir: 'ep1', selected: true }],
          [{ path: 'ep2/video.mp4', filename: 'video.mp4', sizeBytes: 100, parentDir: 'ep2', selected: true }]
        );
        expect(deduped).toHaveLength(2);
      }
    });
  });

  test('[Tier 2] scanning a 2nd folder appends/retains existing sessionSubprojects rather than overwriting', async () => {
    const parentId = 'parent_2026-11-18';
    const sub1 = { id: 'sub_1', name: 'folder1_a.mp4', parent_id: parentId, stage: 1, status: 'pending', media_id: 'm1' };
    const sub2 = { id: 'sub_2', name: 'folder2_b.mp4', parent_id: parentId, stage: 1, status: 'pending', media_id: 'm2' };

    useDubDubStore.setState({
      activeParentProjectId: parentId,
      activeParentProjectName: '2026-11-18',
      sessionSubprojects: [sub1] as any,
    });

    await withE2EShims(async () => {
      return jsonResponse({ ok: true });
    }, async () => {
      useDubDubStore.getState().setSessionSubprojects([sub2] as any, parentId, '2026-11-18', { merge: true });
      const subs = useDubDubStore.getState().sessionSubprojects;
      expect(subs.map((p) => p.id)).toEqual(['sub_1', 'sub_2']);
    });
  });

  test('[Tier 2] closing and reopening FolderScanModal resets transient modal state', () => {
    useDubDubStore.setState({ scanModalOpen: true });
    const openHtml = renderToStaticMarkup(<FolderScanModal />);
    expect(openHtml).toContain('Scan &amp; Batch Import Videos');

    useDubDubStore.setState({ scanModalOpen: false });
    expect(renderToStaticMarkup(<FolderScanModal />)).toBe('');

    useDubDubStore.setState({ scanModalOpen: true });
    const reopenedHtml = renderToStaticMarkup(<FolderScanModal />);
    expect(reopenedHtml).toContain('Select folders to scan');
    expect(reopenedHtml).not.toContain('Detected MP4 Videos');
  });

  test('[Tier 2] ProjectsScreen & Header exclude parent_* / is_parent rows from counts, Select All, flat list, and include Table View actions', () => {
    const today = formatLocalYmd();
    useDubDubStore.setState({
      projectsList: [
        { id: `parent_${today}`, name: today, stage: 1, status: 'active', state: { is_parent: true } },
        { id: 'v1', name: 'Clip One.mp4', parent_id: `parent_${today}`, stage: 2, status: 'completed', duration: 60 },
        { id: 'v2', name: 'Clip Two.mp4', parent_id: `parent_${today}`, stage: 1, status: 'pending', duration: 30 },
      ] as any,
    });

    // Header badge must count only actual video projects (2), not parent container (3)
    const headerHtml = renderToStaticMarkup(<Header />);
    expect(headerHtml).toContain('>2</span>');
    expect(headerHtml).not.toContain('>3</span>');

    // ProjectsScreen Select All & Total must say 2, not 3
    const gridHtml = renderToStaticMarkup(<ProjectsScreen />);
    expect(gridHtml).toContain('Select All (2)');
    expect(gridHtml).not.toContain('Select All (3)');
    expect(gridHtml).toContain('Total: <strong>2</strong> projects');

    // Flat alphabetical view must exclude parent_* container card
    const flatHtml = renderToStaticMarkup(<ProjectsScreen initialSortBy="alphabetical" />);
    expect(flatHtml).not.toContain(`activate-project-parent_${today}`);
    expect(flatHtml).toContain('Clip One.mp4');
    expect(flatHtml).toContain('Clip Two.mp4');

    // Table View parity: grouped table includes group selection checkbox / Add Video / Scan Folders
    const tableHtml = renderToStaticMarkup(<ProjectsScreen initialViewMode="table" />);
    expect(tableHtml).toContain(`checkbox-date-${today}`);
    expect(tableHtml).toContain('Add Video');
  });
});

// ============================================================================
// F2: Batch Prepare Concurrency & State Merge/Reset (Tier 1 >= 5, Tier 2 >= 5)
// ============================================================================
describe('F2 — Batch Prepare Concurrency & State Merge/Reset', () => {
  // --- Tier 1 ---
  test('[Tier 1] startBatchPrepareJobs queues >2 subprojects respecting maxConcurrency=2', async () => {
    const subs = [
      { id: 's1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'pending' },
      { id: 's2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' },
      { id: 's3', name: 'v3.mp4', media_id: 'm3', stage: 1, status: 'pending' },
      { id: 's4', name: 'v4.mp4', media_id: 'm4', stage: 1, status: 'pending' },
    ];
    useDubDubStore.setState({
      activeProjectId: 's1',
      sessionSubprojects: subs as any,
    });

    const inFlightJobs = new Set<string>();
    let maxActiveJobs = 0;
    const startedIds: string[] = [];

    await withE2EShims(async (url, init, call) => {
      if (url === '/api/jobs' && init?.method === 'POST') {
        inFlightJobs.add(call.body.projectId);
        maxActiveJobs = Math.max(maxActiveJobs, inFlightJobs.size);
        startedIds.push(call.body.projectId);
        return jsonResponse({ id: `job_${call.body.projectId}`, status: 'running' });
      }
      if (url.startsWith('/api/jobs/job_')) {
        const pid = url.split('job_')[1];
        return jsonResponse({ id: `job_${pid}`, status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: `Text ${pid}`, targetText: '' }] });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      const startPromise = useDubDubStore.getState().startBatchPrepareJobs();
      await startPromise;
      await flushTasks();

      // Initially only 2 SSE streams should be open (maxConcurrency = 2)
      expect(maxActiveJobs).toBe(2);
      expect(startedIds).toEqual(['s1', 's2']);

      // Complete s1 via SSE -> triggers s3
      inFlightJobs.delete('s1');
      MockEventSource.findByJobId('job_s1')!.emitMessage({
        status: 'succeeded',
        progress: 100,
        details: { segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Text s1', targetText: '' }] },
      });
      await flushTasks();
      expect(startedIds).toEqual(['s1', 's2', 's3']);
      expect(maxActiveJobs).toBeLessThanOrEqual(2);

      // Complete s2 -> triggers s4
      inFlightJobs.delete('s2');
      MockEventSource.findByJobId('job_s2')!.emitMessage({
        status: 'succeeded',
        progress: 100,
        details: { segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Text s2', targetText: '' }] },
      });
      await flushTasks();
      expect(startedIds).toEqual(['s1', 's2', 's3', 's4']);
      expect(maxActiveJobs).toBeLessThanOrEqual(2);

      // Complete s3 & s4
      inFlightJobs.delete('s3');
      inFlightJobs.delete('s4');
      MockEventSource.findByJobId('job_s3')!.emitMessage({ status: 'succeeded', progress: 100, details: { segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Text s3', targetText: '' }] } });
      MockEventSource.findByJobId('job_s4')!.emitMessage({ status: 'succeeded', progress: 100, details: { segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Text s4', targetText: '' }] } });
      await flushTasks();

      const finalState = useDubDubStore.getState();
      expect(finalState.sessionSubprojects.every((p) => p.stage === 2 && p.status === 'completed')).toBe(true);
    });
  });

  test('[Tier 1] startPrepareJob tracks single-video SSE progress and advances to Stage 2', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_single',
      currentStep: 1,
      project: { ...useDubDubStore.getState().project, filename: 'solo.mp4', verified: true, hasAudio: true },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_solo' },
    });

    await withE2EShims(async (url, init) => {
      if (url === '/api/jobs' && init?.method === 'POST') return jsonResponse({ id: 'job_solo', status: 'running' });
      if (url.includes('/api/jobs/job_solo')) {
        return jsonResponse({ id: 'job_solo', status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hello', targetText: '' }] });
      }
      return jsonResponse({ ok: true, projects: [] });
    }, async () => {
      await useDubDubStore.getState().startPrepareJob();
      const es = MockEventSource.findByJobId('job_solo')!;
      expect(es).toBeDefined();

      es.emitMessage({ progress: 55, stage: 'transcribing', message: 'Transcribing audio…' });
      await flushMicrotasks();
      expect(useDubDubStore.getState().jobStatus).toBe('running');
      expect(useDubDubStore.getState().jobProgress).toBe(55);

      es.emitMessage({
        status: 'succeeded',
        progress: 100,
        details: { segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hello', targetText: '' }] },
      });
      await flushMicrotasks();

      const s = useDubDubStore.getState();
      expect(s.jobStatus).toBe('completed');
      expect(s.currentStep).toBe(2);
      expect(s.segments).toHaveLength(1);
      expect(s.segments[0].sourceText).toBe('Hello');
    });
  });

  test('[Tier 1] resetStage clears downstream state after awaiting pending autosave', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_reset',
      currentStep: 3,
      segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'S1', targetText: 'T1' }] as any,
      jobStatus: 'completed',
    });

    const callOrder: string[] = [];
    await withE2EShims(async (url, init) => {
      const method = init?.method || 'GET';
      callOrder.push(`${method} ${url}`);
      if (method === 'PUT' && url.includes('/api/projects/p_reset')) {
        await new Promise((resolve) => setTimeout(resolve, 10));
        callOrder.push(`DONE ${method} ${url}`);
        return jsonResponse({ ok: true });
      }
      if (url.includes('/stages/2/reset')) return jsonResponse({ ok: true });
      if (url.includes('/api/projects/p_reset')) {
        return jsonResponse({
          id: 'p_reset',
          name: 'Resettable.mp4',
          stage: 2,
          media_id: 'm1',
          state: { currentStep: 2, segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'S1', targetText: '' }] },
        });
      }
      return jsonResponse({ ok: true, projects: [] });
    }, async () => {
      useDubDubStore.getState().triggerAutosave();
      await useDubDubStore.getState().resetStage(2);
      const putStartIdx = callOrder.indexOf('PUT /api/projects/p_reset');
      const putDoneIdx = callOrder.indexOf('DONE PUT /api/projects/p_reset');
      const resetIdx = callOrder.indexOf('POST /api/projects/p_reset/stages/2/reset');
      expect(putStartIdx).toBeGreaterThanOrEqual(0);
      expect(putDoneIdx).toBeGreaterThan(putStartIdx);
      expect(resetIdx).toBeGreaterThan(putDoneIdx);
      const s = useDubDubStore.getState();
      expect(s.currentStep).toBe(2);
      expect(s.jobStatus).toBe('idle');
      expect(s.segments[0].targetText).toBe('');
    });
  });

  test('[Tier 1] triggerAutosave persists project snapshot with media_id and stage', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_save',
      currentStep: 2,
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_save' },
      segments: [{ id: 1, startSec: 0, endSec: 3, sourceText: 'Source', targetText: 'Target' }] as any,
    });

    await withE2EShims(async () => jsonResponse({ ok: true }), async ({ fetchCalls }) => {
      await useDubDubStore.getState().triggerAutosave(true);
      const putCall = fetchCalls.find((c) => c.method === 'PUT' && c.url.includes('/api/projects/p_save'));
      expect(putCall).toBeDefined();
      expect(putCall!.body.stage).toBe(2);
      expect(putCall!.body.media_id).toBe('m_save');
      expect(putCall!.body.state.segments[0].targetText).toBe('Target');
    });
  });

  test('[Tier 1] switchSessionVideo switches activeProjectId and restores projectJobStatuses', async () => {
    useDubDubStore.setState({
      activeProjectId: 'v_a',
      projectJobStatuses: {
        v_a: { jobId: 'j_a', status: 'running', progress: 35, stage: 'transcribing', message: 'ASR 35%' },
        v_b: { jobId: 'j_b', status: 'completed', progress: 100, stage: 'completed', message: 'Done B' },
      },
    });

    await withE2EShims(async (url) => {
      if (url.includes('/api/projects/v_b')) {
        return jsonResponse({ id: 'v_b', name: 'B.mp4', stage: 2, media_id: 'mb', state: { currentStep: 2 } });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().switchSessionVideo('v_b');
      const s = useDubDubStore.getState();
      expect(s.activeProjectId).toBe('v_b');
      expect(s.jobStatus).toBe('completed');
      expect(s.jobProgress).toBe(100);
      expect(s.jobMessage).toBe('Done B');
      expect(s.projectJobStatuses['v_a'].status).toBe('running');
    });
  });

  // --- Tier 2 ---
  test('[Tier 2] SSE task_done + done firing together does NOT double-invoke handleSubSuccess or exceed maxConcurrency=2', async () => {
    const subs = [
      { id: 's1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'pending' },
      { id: 's2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' },
      { id: 's3', name: 'v3.mp4', media_id: 'm3', stage: 1, status: 'pending' },
      { id: 's4', name: 'v4.mp4', media_id: 'm4', stage: 1, status: 'pending' },
    ];
    useDubDubStore.setState({ activeProjectId: 'other_active', sessionSubprojects: subs as any });

    const startedProjectIds: string[] = [];
    let s1FetchJobCount = 0;
    let s1UpdateCount = 0;

    await withE2EShims(async (url, init, call) => {
      if (url === '/api/jobs' && init?.method === 'POST') {
        startedProjectIds.push(call.body.projectId);
        return jsonResponse({ id: `job_${call.body.projectId}`, status: 'running' });
      }
      if (url.includes('/api/jobs/job_s1')) {
        s1FetchJobCount++;
        return jsonResponse({ id: 'job_s1', status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Done 1', targetText: '' }] });
      }
      if (url.includes('/api/projects/s1') && init?.method === 'PUT') {
        s1UpdateCount++;
        return jsonResponse({ ok: true });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().startBatchPrepareJobs();
      await flushTasks();

      expect(startedProjectIds).toEqual(['s1', 's2']);

      // Simulate backend sending BOTH `task_done` message AND `done` SSE event for s1 while s2 is still running
      const es1 = MockEventSource.findByJobId('job_s1')!;
      es1.emitMessage({ status: 'succeeded', stage: 'task_done', progress: 100 });
      es1.emitDone({ status: 'succeeded', stage: 'task_done', progress: 100 });
      await flushTasks();

      // Only s3 must have been dequeued (s2 + s3 running = 2 concurrent). s4 must NOT have started!
      expect(startedProjectIds).toEqual(['s1', 's2', 's3']);
      expect(s1FetchJobCount).toBe(1);
      expect(s1UpdateCount).toBe(1);
    });
  });

  test('[Tier 2] startPrepareJob writes running and completed status into projectJobStatuses[activeProjectId]', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_sync',
      currentStep: 1,
      project: { ...useDubDubStore.getState().project, filename: 'sync.mp4', verified: true, hasAudio: true },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_sync' },
    });

    await withE2EShims(async (url, init) => {
      if (url === '/api/jobs' && init?.method === 'POST') return jsonResponse({ id: 'job_sync', status: 'running' });
      if (url.includes('/api/jobs/job_sync')) {
        return jsonResponse({ id: 'job_sync', status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Hi', targetText: '' }] });
      }
      return jsonResponse({ ok: true, projects: [] });
    }, async () => {
      await useDubDubStore.getState().startPrepareJob();
      expect(useDubDubStore.getState().projectJobStatuses['p_sync']?.status).toBe('running');

      const es = MockEventSource.findByJobId('job_sync')!;
      es.emitMessage({ progress: 60, stage: 'asr', message: 'Halfway' });
      await flushMicrotasks();
      expect(useDubDubStore.getState().projectJobStatuses['p_sync']?.progress).toBe(60);

      es.emitMessage({ status: 'succeeded', progress: 100 });
      await flushMicrotasks();
      expect(useDubDubStore.getState().projectJobStatuses['p_sync']?.status).toBe('completed');
      expect(useDubDubStore.getState().projectJobStatuses['p_sync']?.progress).toBe(100);
    });
  });

  test('[Tier 2] background batch updateProjectState includes media_id and triggerAutosave preserves metadata', async () => {
    useDubDubStore.setState({
      activeProjectId: 's_active',
      sessionSubprojects: [
        { id: 's_bg', name: 'bg.mp4', media_id: 'm_bg', stage: 1, status: 'pending' },
      ] as any,
      languages: { source: { code: 'ja', name: 'Japanese', flag: '', autoDetected: false }, target: { code: 'en', name: 'English' }, timingMode: 'voice' },
      engines: { speakerDiarization: true, speakerCount: 3, ocrSlideEngine: true },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_active' },
    });

    await withE2EShims(async (url, init) => {
      if (url === '/api/jobs' && init?.method === 'POST') {
        return jsonResponse({ id: 'job_bg', status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'BG', targetText: '' }] });
      }
      return jsonResponse({ ok: true });
    }, async ({ fetchCalls }) => {
      await useDubDubStore.getState().startBatchPrepareJobs();
      await flushTasks();
      const bgPut = fetchCalls.find((c) => c.method === 'PUT' && c.url.includes('/api/projects/s_bg'));
      expect(bgPut).toBeDefined();
      expect(bgPut!.body.media_id).toBe('m_bg');

      await useDubDubStore.getState().triggerAutosave(true);
      const activePut = fetchCalls.find((c) => c.method === 'PUT' && c.url.includes('/api/projects/s_active'));
      expect(activePut).toBeDefined();
      expect(activePut!.body.state.languages.source.code).toBe('ja');
      expect(activePut!.body.state.engines.speakerCount).toBe(3);
    });
  });

  test('[Tier 2] failed subproject renders Failed badge in SessionVideosDrawer even without projectJobStatuses', () => {
    useDubDubStore.setState({
      sessionDrawerOpen: true,
      activeProjectId: 'sub_ok',
      sessionSubprojects: [
        { id: 'sub_ok', name: 'ok.mp4', stage: 2, status: 'completed' },
        { id: 'sub_err', name: 'corrupt.mp4', stage: 1, status: 'failed' },
      ] as any,
      projectJobStatuses: {},
    });

    const html = renderToStaticMarkup(<SessionVideosDrawer />);
    expect(html).toContain('corrupt.mp4');
    expect(html).toContain('Failed');
    expect(html).not.toContain('Pending');
  });

  test('[Tier 2] re-selecting already active video short-circuits switchSessionVideo without redundant autosave/fetch', async () => {
    useDubDubStore.setState({
      activeProjectId: 'sub_current',
      sessionSubprojects: [{ id: 'sub_current', name: 'current.mp4', stage: 2, status: 'completed' }] as any,
    });

    await withE2EShims(async () => jsonResponse({ ok: true }), async ({ fetchCalls }) => {
      await useDubDubStore.getState().switchSessionVideo('sub_current');
      expect(fetchCalls).toHaveLength(0);
    });
  });
});

// ============================================================================
// F3: Shell Navigation, Telemetry, VideoPlayer & Provider Docs (Tier 1 >= 5, Tier 2 >= 5)
// ============================================================================
describe('F3 — Shell Navigation, Telemetry, VideoPlayer & Provider Docs', () => {
  // --- Tier 1 ---
  test('[Tier 1] Header renders nav tabs and Back to Projects visibility per activeView', () => {
    useDubDubStore.setState({ activeView: 'projects' });
    const projectsHtml = renderToStaticMarkup(<Header />);
    expect(projectsHtml).toContain('data-testid="nav-tab-projects"');
    expect(projectsHtml).toContain('data-testid="nav-tab-dubbing"');
    expect(projectsHtml).toContain('data-testid="nav-tab-voices"');
    expect(projectsHtml).not.toContain('data-testid="header-back-to-projects-btn"');

    useDubDubStore.getState().setActiveView('dubbing');
    const dubHtml = renderToStaticMarkup(<Header />);
    expect(dubHtml).toContain('data-testid="header-back-to-projects-btn"');
  });

  test('[Tier 1] WorkflowStepper renders steps 1..4 and language route pill', () => {
    useDubDubStore.setState({
      currentStep: 2,
      project: { ...useDubDubStore.getState().project, filename: 'demo.mp4' },
      languages: { source: { code: 'en', name: 'English', flag: '', autoDetected: false }, target: { code: 'vi', name: 'Vietnamese' }, timingMode: 'voice' },
    });

    const html = renderToStaticMarkup(<WorkflowStepper />);
    expect(html).toContain('demo.mp4');
    expect(html).toContain('EN ➔ VI');
    expect(html).toContain('Prepare');
    expect(html).toContain('Review Transcript');
    expect(html).toContain('Voice &amp; Dubbing');
    expect(html).toContain('Edit Video');
    expect(html).toContain('✓');
  });

  test('[Tier 1] StatusFooter renders stage-appropriate primary actions across Stages 1..4', () => {
    useDubDubStore.setState({ currentStep: 1, project: { ...useDubDubStore.getState().project, verified: true } });
    expect(renderToStaticMarkup(<StatusFooter />)).toContain('Start Dub');

    useDubDubStore.setState({ currentStep: 2 });
    expect(renderToStaticMarkup(<StatusFooter />)).toContain('Proceed to Voice &amp; Dubbing');

    useDubDubStore.setState({ currentStep: 3, dubbingStatus: 'idle' });
    expect(renderToStaticMarkup(<StatusFooter />)).toContain('Generate Dubbing');

    useDubDubStore.setState({ currentStep: 3, dubbingStatus: 'completed' });
    expect(renderToStaticMarkup(<StatusFooter />)).toContain('Proceed to Edit Video');

    useDubDubStore.setState({ currentStep: 4 });
    const stage4Html = renderToStaticMarkup(<StatusFooter />);
    expect(stage4Html).toContain('Render dubbed video');
    expect(stage4Html).toContain('data-continue-in-capcut-btn="true"');
  });

  test('[Tier 1] VideoPlayer renders CC toggle, timecode, and active subtitle overlay', () => {
    useDubDubStore.setState({
      activeSegmentId: 1,
      playback: { ...useDubDubStore.getState().playback, currentTime: 1.0, formattedTime: '00:01.000' },
      project: { ...useDubDubStore.getState().project, resolution: '1280x720', fps: '30 fps', duration: '00:10.000', previewUrl: '/api/media/m1/file' },
      segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Src', targetText: 'Tgt' }] as any,
    });

    const html = renderToStaticMarkup(<VideoPlayer />);
    expect(html).toContain('data-testid="toggle-subtitles-btn"');
    expect(html).toContain('CC ON');
    expect(html).toContain('00:01.000');
    expect(html).toContain('data-canvas-subtitle="true"');
    expect(html).toContain('Tgt');
  });

  test('[Tier 1] ProviderDocLink renders secure external guide URLs for known providers and empty for unknown', () => {
    const html = renderToStaticMarkup(<ProviderDocLink providerId="deepgram" category="asr" />);
    expect(html).toContain('href="https://developers.deepgram.com');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener noreferrer"');

    const unknownHtml = renderToStaticMarkup(<ProviderDocLink providerId="totally_unknown_999" />);
    expect(unknownHtml).toBe('');
  });

  test('[Tier 1] SettingsModal renders configuration tabs and saveProviderAndLoadModels updates provider models', async () => {
    const html = renderToStaticMarkup(<SettingsModal forceOpen={true} />);
    expect(html).toContain('Settings &amp; Preferences');
    expect(html).toContain('API Providers');
    expect(html).toContain('Storage &amp; Paths');
    expect(html).toContain('General Settings');

    await withE2EShims(async (url) => {
      if (url.endsWith('/models')) return jsonResponse({ ok: true, models: ['gpt-4o', 'gpt-4o-mini'] });
      return jsonResponse({ ok: true, configured: true });
    }, async () => {
      const res = await saveProviderAndLoadModels('llm', 'openai', { apiKey: 'sk-test', model: 'gpt-4o' });
      expect(res.models).toEqual(['gpt-4o', 'gpt-4o-mini']);
    });
  });

  // --- Tier 2 ---
  test('[Tier 2] StatusFooter scopes busy state to currentStep and disables Start Batch Dub when no Stage 1 runnable video exists', () => {
    // Stage 3 dubbing running in background must NOT lock Stage 1 or display Processing ASR...
    useDubDubStore.setState({
      currentStep: 1,
      activeProjectId: 'p1',
      jobStatus: 'idle',
      dubbingStatus: 'running',
      project: { ...useDubDubStore.getState().project, verified: true },
      sessionSubprojects: [],
    });
    const stage1Html = renderToStaticMarkup(<StatusFooter />);
    expect(stage1Html).not.toContain('Processing ASR');
    expect(stage1Html).toContain('Start Dub');
    expect(stage1Html).not.toContain('data-testid="reset-stage-1-btn" disabled=""');

    // Start Batch Dub must be disabled when subprojects have no media or are already past Stage 1
    useDubDubStore.setState({
      currentStep: 1,
      dubbingStatus: 'idle',
      sessionSubprojects: [
        { id: 's1', name: 'no_media.mp4', stage: 1, status: 'pending' },
        { id: 's2', name: 'already_stage2.mp4', stage: 2, status: 'completed', media_id: 'm2' },
      ] as any,
    });
    const batchHtml = renderToStaticMarkup(<StatusFooter />);
    const btnMatch = batchHtml.match(/<button[^>]*>[^<]*(?:<[^>]+>)*\s*Start Batch Dub[^<]*<\/button>/);
    expect(batchHtml).toContain('Start Batch Dub');
    // The primary action button containing Start Batch Dub must be disabled
    expect(btnMatch?.[0] || batchHtml).toContain('disabled=""');
  });

  test('[Tier 2] ProviderDocLink ASR IDs 0..5 resolve accurately without colliding with TTS IDs', () => {
    // ASR IDs in videotrans/recognition/__init__.py: QWENASR=0, Deepgram=1, GEMINI=2, GOOGLE=3, ElevenLabs=4, FASTER_WHISPER=5
    const doc0 = getProviderDocInfo(0, 'asr');
    const doc1 = getProviderDocInfo(1, 'asr');
    const doc2 = getProviderDocInfo(2, 'asr');
    const doc3 = getProviderDocInfo(3, 'asr');
    const doc4 = getProviderDocInfo(4, 'asr');
    const doc5 = getProviderDocInfo(5, 'asr');

    expect(doc0).not.toBeNull();
    expect(doc0?.category).toBe('asr');
    expect(doc1?.docUrl).toContain('deepgram.com');
    expect(doc2?.category).toBe('asr');
    expect(doc2?.docUrl).not.toContain('vieneu');
    expect(doc3?.category).toBe('asr');
    expect(doc3?.docUrl).not.toContain('gemini-api/docs/speech');
    expect(doc4?.category).toBe('asr');
    expect(doc4?.docUrl).not.toContain('edge-tts');
    expect(doc5?.category).toBe('asr');
    expect(doc5?.docUrl).not.toContain('text-to-speech');

    const link2Html = renderToStaticMarkup(<ProviderDocLink providerId={2} category="asr" />);
    expect(link2Html).not.toContain('vieneu');
  });

  test('[Tier 2] Header hides Auto-saved outside active dubbing project and reflects real CUDA/CPU capability', () => {
    useDubDubStore.setState({
      activeView: 'projects',
      activeProjectId: null,
      backend: {
        ...useDubDubStore.getState().backend,
        options: { ...useDubDubStore.getState().backend.options, cudaAvailable: false } as any,
        config: { ...useDubDubStore.getState().backend.config, useCuda: false },
      },
    });
    const projectsHtml = renderToStaticMarkup(<Header />);
    expect(projectsHtml).not.toContain('Auto-saved');
    expect(projectsHtml).not.toContain('Warm GPU');

    useDubDubStore.setState({ activeView: 'voices', activeProjectId: 'p1' });
    const voicesHtml = renderToStaticMarkup(<Header />);
    expect(voicesHtml).not.toContain('Auto-saved');
  });

  test('[Tier 2] VideoPlayer derives audio switcher labels from languages and hides subtitle overlay in silent gaps', () => {
    useDubDubStore.setState({
      activeSegmentId: 1,
      languages: {
        source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
        target: { code: 'vi', name: 'Vietnamese' },
        timingMode: 'voice',
      },
      playback: { ...useDubDubStore.getState().playback, currentTime: 5.0, formattedTime: '00:05.000', audioChannel: 'orig' },
      segments: [
        { id: 1, startSec: 0, endSec: 2, sourceText: 'First cue source', targetText: 'First cue target' },
        { id: 2, startSec: 8, endSec: 10, sourceText: 'Second cue source', targetText: 'Second cue target' },
      ] as any,
    });

    const html = renderToStaticMarkup(<VideoPlayer showAudioSwitcher={true} />);
    expect(html).not.toContain('EN Orig');
    expect(html).not.toContain('ES Dub');
    expect(html).toContain('ZH');
    expect(html).toContain('VI');
    // At t=5.0s (gap between [0,2] and [8,10]), subtitle overlay must NOT display Segment 1
    expect(html).not.toContain('First cue target');
  });

  test('[Tier 2] SettingsModal / store refresh syncs updated backend config even when backend.ready is true', async () => {
    useDubDubStore.setState({
      backend: {
        ...useDubDubStore.getState().backend,
        ready: true,
        config: { ...useDubDubStore.getState().backend.config, modelName: 'nova-2', translateType: 0 },
      },
    });

    await withE2EShims(async (url) => {
      if (url.includes('/api/options')) {
        return jsonResponse({
          languages: [],
          asrProviders: [],
          translationProviders: [],
          translationModes: [],
          voices: [],
          defaults: { recognType: 1, modelName: 'nova-3', translateType: 2, translationMode: 'srt' },
        });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      const s: any = useDubDubStore.getState();
      if (typeof s.refreshBackendConfig === 'function') {
        await s.refreshBackendConfig();
      } else {
        await s.initializeBackend(true);
      }
      expect(useDubDubStore.getState().backend.config.modelName).toBe('nova-3');
      expect(useDubDubStore.getState().backend.config.translateType).toBe(2);
    });
  });
});

// ============================================================================
// Tier 3: Pairwise Cross-Feature Flow (F1 + F2 + F3)
// ============================================================================
describe('Tier 3 — Batch Session + Prepare Concurrency + Shell Telemetry', () => {
  test('[Tier 3] F1 + F2 + F3: openDateParentSession -> startBatchPrepareJobs (SSE) -> switchSessionVideo -> Stage 2 shell telemetry', async () => {
    const today = formatLocalYmd();
    const parentId = `parent_${today}`;
    const subs = [
      { id: 'b1', name: 'clip1.mp4', parent_id: parentId, media_id: 'm1', stage: 1, status: 'pending' },
      { id: 'b2', name: 'clip2.mp4', parent_id: parentId, media_id: 'm2', stage: 1, status: 'pending' },
      { id: 'b3', name: 'clip3.mp4', parent_id: parentId, media_id: 'm3', stage: 1, status: 'pending' },
    ];

    await withE2EShims(async (url, init, call) => {
      if (url.includes(`/api/projects/${parentId}/subprojects`)) return jsonResponse({ subprojects: subs });
      if (url.includes('/api/projects/b1')) return jsonResponse({ project: subs[0] });
      if (url.includes('/api/projects/b2')) {
        return jsonResponse({
          project: { ...subs[1], stage: 2, status: 'completed', state: { currentStep: 2, segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Clip 2 Line', targetText: '' }] } },
        });
      }
      if (url === '/api/jobs' && init?.method === 'POST') return jsonResponse({ id: `job_${call.body.projectId}`, status: 'running' });
      if (url.startsWith('/api/jobs/job_')) return jsonResponse({ id: url.split('/').pop(), status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Recognized', targetText: '' }] });
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().openDateParentSession(parentId, today);
      await useDubDubStore.getState().startBatchPrepareJobs();
      await flushTasks();

      MockEventSource.findByJobId('job_b1')!.emitMessage({ status: 'succeeded', progress: 100 });
      MockEventSource.findByJobId('job_b2')!.emitMessage({ status: 'succeeded', progress: 100 });
      await flushTasks();
      MockEventSource.findByJobId('job_b3')!.emitMessage({ status: 'succeeded', progress: 100 });
      await flushTasks();

      await useDubDubStore.getState().switchSessionVideo('b2');
      const s = useDubDubStore.getState();
      expect(s.activeProjectId).toBe('b2');
      expect(s.currentStep).toBe(2);

      const drawerHtml = renderToStaticMarkup(<SessionVideosDrawer />);
      expect(drawerHtml).toContain('clip2.mp4');
      expect(drawerHtml).toContain('100%');
      const stepperHtml = renderToStaticMarkup(<WorkflowStepper />);
      expect(stepperHtml).toContain('Review Transcript');
      const footerHtml = renderToStaticMarkup(<StatusFooter />);
      expect(footerHtml).toContain('Proceed to Voice &amp; Dubbing');
    });
  });
});
