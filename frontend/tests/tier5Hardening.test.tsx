import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { StatusFooter } from '../src/components/StatusFooter';
import { SessionVideosDrawer } from '../src/components/SessionVideosDrawer';
import { Stage1Prepare } from '../src/screens/Stage1Prepare';
import { useDubDubStore } from '../src/store';

const initialSnapshot = useDubDubStore.getInitialState();
const originalFetch = globalThis.fetch;
const originalEventSource = (globalThis as any).EventSource;
const originalCreateObjectUrl = URL.createObjectURL;

function jsonRes(body: any, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  listeners: Record<string, Array<(e: any) => void>> = {};
  onmessage: ((e: any) => void) | null = null;
  onerror: ((e: any) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, cb: (e: any) => void) {
    this.listeners[type] = this.listeners[type] || [];
    this.listeners[type].push(cb);
  }

  emitMessage(data: any) {
    const evt = { data: JSON.stringify(data) };
    this.onmessage?.(evt);
  }

  emitDone(data: any) {
    const evt = { data: JSON.stringify(data) };
    (this.listeners['done'] || []).forEach((cb) => cb(evt));
  }

  emitError(err: any = new Error('stream dropped')) {
    this.onerror?.(err);
  }

  close() {
    this.closed = true;
  }
}

async function flushMicrotasks(ms = 15) {
  await new Promise((r) => setTimeout(r, ms));
}

beforeEach(() => {
  MockEventSource.instances = [];
  (globalThis as any).EventSource = MockEventSource;
  (URL as any).createObjectURL = () => 'blob:mock-preview';
  useDubDubStore.setState(
    {
      ...initialSnapshot,
      activeProjectId: 'proj_t5',
      currentStep: 1,
      maxUnlockedStep: 4,
      sessionSubprojects: [],
      projectJobStatuses: {},
      backend: {
        ...initialSnapshot.backend,
        ready: true,
        mediaId: 'media_t5',
        status: 'ready',
      },
      project: {
        ...initialSnapshot.project,
        filename: 'demo.mp4',
        verified: true,
        durationSec: 12,
      },
    },
    true
  );
});

afterEach(() => {
  useDubDubStore.getState().clearJob();
  globalThis.fetch = originalFetch;
  (globalThis as any).EventSource = originalEventSource;
  (URL as any).createObjectURL = originalCreateObjectUrl;
});

describe('M7 Phase 2 Tier 5 Adversarial Hardening — Frontend', () => {
  test('Gap 9a: exportEditedVideo SSE succeeded keeps currentStep=4, clears editVideo.exporting, and keeps StatusFooter ready', async () => {
    useDubDubStore.setState({ currentStep: 4, activeProjectId: 'proj_render_ok' });

    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/render') {
        return jsonRes({ id: 'job_render_ok', jobType: 'render', status: 'queued' }, 202);
      }
      if (s === '/api/jobs/job_render_ok') {
        return jsonRes({ id: 'job_render_ok', jobType: 'render', status: 'succeeded', message: 'Render complete' });
      }
      if (s === '/api/projects') {
        return jsonRes({ projects: [] });
      }
      return jsonRes({ ok: true });
    }) as any;

    await useDubDubStore.getState().exportEditedVideo();
    expect(useDubDubStore.getState().editVideo.exporting).toBe(true);
    expect(useDubDubStore.getState().jobStage).toBe('render');
    expect(MockEventSource.instances.length).toBe(1);

    // Intermediate stage events from orchestrator must not clobber jobStage away from 'render'
    MockEventSource.instances[0].emitMessage({ stage: 'assembling', progress: 60, message: 'Muxing video' });
    await flushMicrotasks();
    expect(useDubDubStore.getState().jobStage).toBe('render');

    MockEventSource.instances[0].emitMessage({
      status: 'succeeded',
      kind: 'succeeded',
      stage: 'task_done',
      progress: 100,
      message: 'Render complete',
    });
    await flushMicrotasks(25);

    const st = useDubDubStore.getState();
    expect(st.currentStep).toBe(4);
    expect(st.editVideo.exporting).toBe(false);
    expect(st.editVideo.error).toBeNull();
    expect(st.jobStatus).toBe('completed');
    expect(st.jobStage).toBe('render');
    expect(st.projectJobStatuses['proj_render_ok']?.stage).toBe('render');

    const html = renderToStaticMarkup(<StatusFooter />);
    expect(html).toContain('Render dubbed video');
    expect(html).not.toContain('Rendering Video…');
  });

  test('Gap 9b: exportEditedVideo SSE failed resets editVideo.exporting, sets editVideo.error, and re-enables StatusFooter', async () => {
    useDubDubStore.setState({ currentStep: 4, activeProjectId: 'proj_render_fail' });

    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/render') {
        return jsonRes({ id: 'job_render_fail', jobType: 'render', status: 'queued' }, 202);
      }
      return jsonRes({ ok: true });
    }) as any;

    await useDubDubStore.getState().exportEditedVideo();
    expect(useDubDubStore.getState().editVideo.exporting).toBe(true);

    MockEventSource.instances[0].emitMessage({
      status: 'failed',
      error: 'NVENC encoder unavailable',
      message: 'NVENC encoder unavailable',
    });
    await flushMicrotasks(20);

    const st = useDubDubStore.getState();
    expect(st.currentStep).toBe(4);
    expect(st.editVideo.exporting).toBe(false);
    expect(st.editVideo.error).toBe('NVENC encoder unavailable');
    expect(st.jobStatus).toBe('failed');
    expect(st.jobStage).toBe('render');

    const html = renderToStaticMarkup(<StatusFooter />);
    expect(html).toContain('Render dubbed video');
    expect(html).toContain('NVENC encoder unavailable');
    expect(html).not.toContain('Rendering Video…');
  });

  test('Gap 9c: cancelActiveJob during Stage 4 render clears editVideo.exporting and re-enables render button', async () => {
    useDubDubStore.setState({ currentStep: 4, activeProjectId: 'proj_render_cancel' });
    const cancelledUrls: string[] = [];

    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/render') {
        return jsonRes({ id: 'job_render_cancel', jobType: 'render', status: 'queued' }, 202);
      }
      if (s === '/api/jobs/job_render_cancel/cancel' && init?.method === 'POST') {
        cancelledUrls.push(s);
        return jsonRes({ id: 'job_render_cancel', status: 'cancelled' });
      }
      return jsonRes({ ok: true });
    }) as any;

    await useDubDubStore.getState().exportEditedVideo();
    expect(useDubDubStore.getState().editVideo.exporting).toBe(true);

    await useDubDubStore.getState().cancelActiveJob();
    const st = useDubDubStore.getState();
    expect(cancelledUrls).toEqual(['/api/jobs/job_render_cancel/cancel']);
    expect(st.editVideo.exporting).toBe(false);
    expect(st.editVideo.error).toContain('cancelled');
    expect(st.jobStatus).toBe('cancelled');
    expect(st.jobStage).toBe('render');

    const html = renderToStaticMarkup(<StatusFooter />);
    expect(html).toContain('Render dubbed video');
    expect(html).not.toContain('Rendering Video…');
  });

  test('Gap 9d: pollJob on render job handles succeeded and failed without regressing currentStep', async () => {
    useDubDubStore.setState({
      currentStep: 4,
      activeProjectId: 'proj_render_poll',
      editVideo: { ...initialSnapshot.editVideo, exporting: true, error: null },
      activeJob: { jobId: 'job_render_poll', jobType: 'render', stage: 'render', status: 'running', progress: 50, message: 'Rendering' },
    });

    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/jobs/job_render_poll') {
        return jsonRes({ id: 'job_render_poll', jobType: 'render', status: 'succeeded', progress: 100, message: 'Done' });
      }
      return jsonRes({ projects: [] });
    }) as any;

    await useDubDubStore.getState().pollJob('job_render_poll', 'proj_render_poll', { jobType: 'render' });
    const st = useDubDubStore.getState();
    expect(st.currentStep).toBe(4);
    expect(st.editVideo.exporting).toBe(false);
    expect(st.jobStatus).toBe('completed');
    expect(st.jobStage).toBe('render');
  });

  test('Gap 10: selectMedia preserves probed metadata, playback.duration, and subtitleSource across createNewProject -> selectProject', async () => {
    useDubDubStore.setState({
      activeProjectId: null,
      subtitleSource: 'video_ocr',
      ocrCrop: { active: false, segmentId: null, roi: [0.1, 0.72, 0.8, 0.2], loading: false, error: null },
    });

    let createdBody: any = null;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/media') {
        return jsonRes({
          id: 'm_silent_ocr',
          filename: 'silent_slides.mp4',
          sizeBytes: 2048000,
          durationMs: 15000,
          resolution: '1920x1080',
          fps: 30,
          videoCodec: 'h264',
          audioCodec: null,
          container: 'mp4',
          bitrate: 2000000,
          hasAudio: false,
          hasVideo: true,
        });
      }
      if (s === '/api/projects' && init?.method === 'POST') {
        createdBody = JSON.parse(String(init.body || '{}'));
        return jsonRes({
          project: {
            id: 'proj_created_ocr',
            name: 'silent_slides.mp4',
            media_id: 'm_silent_ocr',
            duration: 15,
            stage: 1,
            state: {},
          },
        });
      }
      if (s === '/api/projects/proj_created_ocr' && (!init?.method || init.method === 'GET')) {
        // Simulate backend returning minimal state (worst-case mock)
        return jsonRes({
          project: {
            id: 'proj_created_ocr',
            name: 'silent_slides.mp4',
            media_id: 'm_silent_ocr',
            duration: 15,
            stage: 1,
            state: {},
          },
        });
      }
      return jsonRes({ ok: true, projects: [], subprojects: [] });
    }) as any;

    const fakeFile = new File(['video-bytes'], 'silent_slides.mp4', { type: 'video/mp4' });
    await useDubDubStore.getState().selectMedia(fakeFile);

    const st = useDubDubStore.getState();
    expect(createdBody?.state?.subtitleSource).toBe('video_ocr');
    expect(createdBody?.state?.project?.resolution).toBe('1920x1080');
    expect(st.activeProjectId).toBe('proj_created_ocr');
    expect(st.project.resolution).toBe('1920x1080');
    expect(st.project.durationSec).toBe(15);
    expect(st.playback.duration).toBe(15);
    expect(st.project.hasAudio).toBe(false);
    expect(st.project.hasVideo).toBe(true);
    expect(st.subtitleSource).toBe('video_ocr');
    expect(st.ocrCrop.roi).toEqual([0.1, 0.72, 0.8, 0.2]);
  });

  test('Gap 11a: startBatchPrepareJobs populates activeJob & startTime, syncs on switchSessionVideo, and cancels via cancelActiveJob', async () => {
    useDubDubStore.setState({
      activeProjectId: 'sub_1',
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'pending' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' } as any,
        { id: 'sub_3', name: 'v3.mp4', media_id: 'm3', stage: 1, status: 'pending' } as any,
      ],
      projectJobStatuses: {},
    });

    const cancelledJobs: string[] = [];
    let submitCount = 0;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/jobs' && init?.method === 'POST') {
        submitCount += 1;
        const body = JSON.parse(String(init.body || '{}'));
        return jsonRes({ id: `batch_job_${body.projectId}`, status: 'queued' }, 202);
      }
      if (s.includes('/cancel') && init?.method === 'POST') {
        cancelledJobs.push(s);
        return jsonRes({ status: 'cancelled' });
      }
      if (s.startsWith('/api/projects/sub_')) {
        const sid = s.split('/').pop()!;
        return jsonRes({ project: { id: sid, name: `${sid}.mp4`, media_id: `m_${sid}`, stage: 1, state: {} } });
      }
      return jsonRes({ ok: true, projects: [], subprojects: [] });
    }) as any;

    await useDubDubStore.getState().startBatchPrepareJobs();
    await flushMicrotasks(20);

    const runningState = useDubDubStore.getState();
    expect(submitCount).toBe(2);
    expect(runningState.startTime).not.toBeNull();
    expect(runningState.activeJob?.jobId).toBe('batch_job_sub_1');
    expect(runningState.jobStatus).toBe('running');

    // Switching to sub_2 syncs activeJob to sub_2's running batch job
    await useDubDubStore.getState().switchSessionVideo('sub_2');
    expect(useDubDubStore.getState().activeJob?.jobId).toBe('batch_job_sub_2');

    // Cancel must POST cancel for running batch jobs and cancel queued sub_3
    await useDubDubStore.getState().cancelActiveJob();
    const afterCancel = useDubDubStore.getState();
    expect(cancelledJobs.sort()).toEqual([
      '/api/jobs/batch_job_sub_1/cancel',
      '/api/jobs/batch_job_sub_2/cancel',
    ]);
    expect(afterCancel.jobStatus).toBe('cancelled');
    expect(afterCancel.projectJobStatuses['sub_1']?.status).toBe('cancelled');
    expect(afterCancel.projectJobStatuses['sub_2']?.status).toBe('cancelled');
    expect(afterCancel.projectJobStatuses['sub_3']?.status).toBe('cancelled');
  });

  test('Gap 11b: startBatchPrepareJobs onError checks fetchJob before marking failed', async () => {
    useDubDubStore.setState({
      activeProjectId: 'sub_err_1',
      sessionSubprojects: [
        { id: 'sub_err_1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'pending' } as any,
      ],
      projectJobStatuses: {},
    });

    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/jobs' && init?.method === 'POST') {
        return jsonRes({ id: 'batch_err_job', status: 'queued' }, 202);
      }
      if (s === '/api/jobs/batch_err_job') {
        return jsonRes({
          id: 'batch_err_job',
          status: 'succeeded',
          segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Recovered', targetText: '' }],
        });
      }
      return jsonRes({ ok: true, projects: [] });
    }) as any;

    await useDubDubStore.getState().startBatchPrepareJobs();
    await flushMicrotasks(20);
    expect(MockEventSource.instances.length).toBe(1);

    MockEventSource.instances[0].emitError(new Error('transient SSE drop'));
    await flushMicrotasks(25);

    const st = useDubDubStore.getState();
    expect(st.projectJobStatuses['sub_err_1']?.status).toBe('completed');
    expect(st.jobStatus).toBe('completed');
    expect(st.segments[0]?.sourceText).toBe('Recovered');
  });

  test('I2 Gap 1: cancelActiveJob preserves already-completed batch subprojects and switchSessionVideo clears activeJob for completed subprojects', async () => {
    useDubDubStore.setState({
      activeProjectId: 'sub_1',
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'pending' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' } as any,
        { id: 'sub_3', name: 'v3.mp4', media_id: 'm3', stage: 1, status: 'pending' } as any,
      ],
      projectJobStatuses: {},
    });

    const cancelledUrls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/jobs' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body || '{}'));
        return jsonRes({ id: `job_${body.projectId}`, status: 'queued' }, 202);
      }
      if (s === '/api/jobs/job_sub_1') {
        return jsonRes({
          id: 'job_sub_1',
          status: 'succeeded',
          segments: [{ id: 1, startSec: 0, endSec: 1, sourceText: 'Sub 1 done', targetText: '' }],
        });
      }
      if (s.includes('/cancel') && init?.method === 'POST') {
        cancelledUrls.push(s);
        return jsonRes({ status: 'cancelled' });
      }
      if (s.startsWith('/api/projects/sub_')) {
        const sid = s.split('/').pop()!;
        return jsonRes({ project: { id: sid, name: `${sid}.mp4`, media_id: `m_${sid}`, stage: sid === 'sub_1' ? 2 : 1, state: {} } });
      }
      return jsonRes({ ok: true, projects: [], subprojects: [] });
    }) as any;

    await useDubDubStore.getState().startBatchPrepareJobs();
    await flushMicrotasks(20);
    expect(MockEventSource.instances.length).toBe(2);

    // Complete sub_1 while sub_2 is still running, triggering sub_3 to start
    const es1 = MockEventSource.instances.find((e) => e.url.includes('job_sub_1'))!;
    es1.emitMessage({
      status: 'succeeded',
      kind: 'succeeded',
      stage: 'task_done',
      progress: 100,
    });
    await flushMicrotasks(30);

    const midState = useDubDubStore.getState();
    expect(midState.projectJobStatuses['sub_1']?.status).toBe('completed');
    expect(midState.projectJobStatuses['sub_3']?.status).toBe('running');
    expect(midState.activeJob).toBeNull();

    // Switching away and back to completed sub_1 must NOT restore a stale activeJob
    await useDubDubStore.getState().switchSessionVideo('sub_3');
    expect(useDubDubStore.getState().activeJob?.jobId).toBe('job_sub_3');
    await useDubDubStore.getState().switchSessionVideo('sub_1');
    expect(useDubDubStore.getState().activeJob).toBeNull();
    expect(useDubDubStore.getState().jobStatus).toBe('completed');

    // Cancel remaining batch while viewing completed sub_1
    await useDubDubStore.getState().cancelActiveJob();
    const afterCancel = useDubDubStore.getState();
    expect(cancelledUrls.sort()).toEqual([
      '/api/jobs/job_sub_2/cancel',
      '/api/jobs/job_sub_3/cancel',
    ]);
    expect(cancelledUrls).not.toContain('/api/jobs/job_sub_1/cancel');
    expect(afterCancel.projectJobStatuses['sub_1']?.status).toBe('completed');
    expect(afterCancel.projectJobStatuses['sub_1']?.progress).toBe(100);
    expect(afterCancel.jobStatus).toBe('completed');
    expect(afterCancel.projectJobStatuses['sub_2']?.status).toBe('cancelled');
    expect(afterCancel.projectJobStatuses['sub_3']?.status).toBe('cancelled');
  });

  test('I2 Gap 2a: StatusFooter renders Cancel Processing when viewing a queued subproject during active batch', () => {
    useDubDubStore.setState({
      currentStep: 1,
      activeProjectId: 'sub_3',
      jobStatus: 'idle',
      jobStage: null,
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1 } as any,
        { id: 'sub_3', name: 'v3.mp4', media_id: 'm3', stage: 1 } as any,
      ],
      projectJobStatuses: {
        sub_1: { jobId: 'j1', status: 'running', progress: 30, message: 'Running', stage: 'recogn' },
        sub_3: { status: 'queued', progress: 0, message: 'Queued for ASR', stage: 'prepare' },
      },
    });

    const html = renderToStaticMarkup(<StatusFooter />);
    expect(html).toContain('Cancel Processing');
    const cancelBtn = html.split('<button').find((chunk) => chunk.includes('Cancel Processing'));
    expect(cancelBtn).toBeDefined();
    expect(cancelBtn!).not.toContain('disabled=""');
    expect(cancelBtn!).toContain('bg-rose-700');
  });

  test('I2 Gap 2b: startBatchPrepareJobs skips already-completed stage >= 2 subprojects', async () => {
    useDubDubStore.setState({
      activeProjectId: 'sub_pending',
      sessionSubprojects: [
        { id: 'sub_done', name: 'done.mp4', media_id: 'm_done', stage: 2, status: 'completed' } as any,
        { id: 'sub_pending', name: 'todo.mp4', media_id: 'm_todo', stage: 1, status: 'pending' } as any,
      ],
      projectJobStatuses: {
        sub_done: { jobId: 'job_old', status: 'completed', progress: 100, message: 'Done', stage: 'completed' },
      },
    });

    const submittedPids: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/jobs' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body || '{}'));
        submittedPids.push(body.projectId);
        return jsonRes({ id: `job_${body.projectId}`, status: 'succeeded', segments: [] });
      }
      return jsonRes({ ok: true, projects: [] });
    }) as any;

    await useDubDubStore.getState().startBatchPrepareJobs();
    await flushMicrotasks(25);

    expect(submittedPids).toEqual(['sub_pending']);
    expect(useDubDubStore.getState().projectJobStatuses['sub_done']?.status).toBe('completed');
  });

  test('I3 Gap 5: selectProject into standalone project clears batch banner, sessionSubprojects, and stale activeJob', async () => {
    useDubDubStore.setState({
      currentStep: 1,
      activeProjectId: 'sub_1',
      activeParentProjectId: 'parent_2026-11-18',
      activeParentProjectName: '2026-11-18',
      batchModeActive: true,
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'processing' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' } as any,
      ],
      activeJob: {
        jobId: 'job_sub_1',
        projectId: 'sub_1',
        stage: 'recogn',
        status: 'running',
        progress: 40,
        message: 'Running ASR',
      },
      projectJobStatuses: {
        sub_1: { jobId: 'job_sub_1', status: 'running', progress: 40, message: 'Running ASR', stage: 'recogn' },
      },
    });

    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/projects/proj_standalone') {
        return jsonRes({
          id: 'proj_standalone',
          name: 'standalone.mp4',
          parent_id: null,
          media_id: 'm_standalone',
          stage: 1,
          status: 'pending',
          duration: 12,
          state: {
            currentStep: 1,
            project: { filename: 'standalone.mp4', verified: true },
          },
        });
      }
      return jsonRes({ ok: true, subprojects: [] });
    }) as any;

    await useDubDubStore.getState().selectProject('proj_standalone', true);
    const st = useDubDubStore.getState();
    expect(st.activeProjectId).toBe('proj_standalone');
    expect(st.activeParentProjectId).toBeNull();
    expect(st.activeParentProjectName).toBeNull();
    expect(st.batchModeActive).toBe(false);
    expect(st.sessionSubprojects).toEqual([]);
    expect(st.activeJob).toBeNull();
    expect(st.jobStatus).toBe('idle');

    const footerHtml = renderToStaticMarkup(<StatusFooter />);
    expect(footerHtml).toContain('Start Dub');
    expect(footerHtml).not.toContain('Cancel Processing');

    const stage1Html = renderToStaticMarkup(<Stage1Prepare />);
    expect(stage1Html).not.toContain('Batch Mode Active');
  });

  test('I3 Gap 6: runBatchTranslation and translateSingleSegment do not overwrite newly selected project on switch race', async () => {
    // 6a: runBatchTranslation project-switch race
    useDubDubStore.setState({
      activeProjectId: 'p_one',
      currentStep: 2,
      segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'One', targetText: '' } as any],
      transcriptOptions: {
        utterances: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'One', targetText: '' } as any],
      },
      selectedSegmentOption: 'utterances',
    });

    let resolveBatchTranslate!: (resp: Response) => void;
    const savedCalls: Array<{ url: string; body: any }> = [];

    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/translate') {
        return new Promise<Response>((resolve) => {
          resolveBatchTranslate = resolve;
        });
      }
      if (s.startsWith('/api/projects/') && init?.method === 'PUT') {
        savedCalls.push({ url: s, body: JSON.parse(String(init.body || '{}')) });
        return jsonRes({ ok: true });
      }
      return jsonRes({ ok: true });
    }) as any;

    const batchPromise = useDubDubStore.getState().runBatchTranslation();
    await flushMicrotasks(5);

    // Switch to p_two while /api/translate is still in-flight
    useDubDubStore.setState({
      activeProjectId: 'p_two',
      segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Two', targetText: 'KEEP_TWO' } as any],
      transcriptOptions: null,
    });

    resolveBatchTranslate(
      jsonRes({
        ok: true,
        segments: [{ id: 1, sourceText: 'One', targetText: 'TRANSLATED_ONE' }],
      })
    );
    await batchPromise;

    const afterBatch = useDubDubStore.getState();
    expect(afterBatch.activeProjectId).toBe('p_two');
    expect(afterBatch.segments[0]?.targetText).toBe('KEEP_TWO');
    expect(afterBatch.translationModal.active).toBe(false);
    expect(savedCalls.some((c) => c.url === '/api/projects/p_two')).toBe(false);
    const pOneSave = savedCalls.find((c) => c.url === '/api/projects/p_one');
    expect(pOneSave).toBeDefined();
    expect(pOneSave!.body.state.segments[0].targetText).toBe('TRANSLATED_ONE');

    // 6b: translateSingleSegment project-switch race
    savedCalls.length = 0;
    useDubDubStore.setState({
      activeProjectId: 'p_one',
      segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'One', targetText: '' } as any],
    });
    let resolveSingleTranslate!: (resp: Response) => void;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s === '/api/translate') {
        return new Promise<Response>((resolve) => {
          resolveSingleTranslate = resolve;
        });
      }
      if (s.startsWith('/api/projects/') && init?.method === 'PUT') {
        savedCalls.push({ url: s, body: JSON.parse(String(init.body || '{}')) });
        return jsonRes({ ok: true });
      }
      return jsonRes({ ok: true });
    }) as any;

    const singlePromise = useDubDubStore.getState().translateSingleSegment(1);
    await flushMicrotasks(5);

    useDubDubStore.setState({
      activeProjectId: 'p_two',
      segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Two', targetText: 'KEEP_TWO' } as any],
    });

    resolveSingleTranslate(
      jsonRes({
        ok: true,
        segments: [{ id: 1, sourceText: 'One', targetText: 'SINGLE_ONE' }],
      })
    );
    await singlePromise;

    expect(useDubDubStore.getState().segments[0]?.targetText).toBe('KEEP_TWO');
    expect(savedCalls.some((c) => c.url === '/api/projects/p_two')).toBe(false);
    const singleSave = savedCalls.find((c) => c.url === '/api/projects/p_one');
    expect(singleSave).toBeDefined();
    expect(singleSave!.body.state.segments[0].targetText).toBe('SINGLE_ONE');
  });

  test('I3 Gap 7: Stage 4 render cancel leaves background Stage 1 batch ASR untouched and footer/drawer disable completed batch', async () => {
    useDubDubStore.setState({
      currentStep: 4,
      activeProjectId: 'sub_1',
      jobStage: 'render',
      activeJob: {
        jobId: 'job_render_1',
        projectId: 'sub_1',
        jobType: 'render',
        stage: 'render',
        status: 'running',
        progress: 50,
        message: 'Rendering',
      } as any,
      editVideo: { ...initialSnapshot.editVideo, exporting: true },
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 4, status: 'completed' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'processing' } as any,
      ],
      projectJobStatuses: {
        sub_1: { jobId: 'job_render_1', status: 'running', progress: 50, message: 'Rendering', stage: 'render' },
        sub_2: { jobId: 'job_asr_2', status: 'running', progress: 30, message: 'Running ASR', stage: 'recogn' },
      },
    });

    const cancelledUrls: string[] = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/cancel') && init?.method === 'POST') {
        cancelledUrls.push(s);
        return jsonRes({ status: 'cancelled' });
      }
      return jsonRes({ ok: true });
    }) as any;

    await useDubDubStore.getState().cancelActiveJob();
    const afterRenderCancel = useDubDubStore.getState();
    expect(cancelledUrls).toEqual(['/api/jobs/job_render_1/cancel']);
    expect(afterRenderCancel.projectJobStatuses['sub_1']?.status).toBe('cancelled');
    expect(afterRenderCancel.projectJobStatuses['sub_2']?.status).toBe('running');
    expect(afterRenderCancel.projectJobStatuses['sub_2']?.stage).toBe('recogn');

    // 7b: StatusFooter & SessionVideosDrawer disable batch ASR button when all subprojects are completed
    useDubDubStore.setState({
      currentStep: 1,
      activeProjectId: 'sub_1',
      jobStatus: 'idle',
      jobStage: null,
      activeJob: null,
      sessionDrawerOpen: true,
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'completed' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'completed' } as any,
      ],
      projectJobStatuses: {},
    });

    const footerHtml = renderToStaticMarkup(<StatusFooter />);
    const batchBtn = footerHtml.split('<button').find((chunk) => chunk.includes('Start Batch Dub'));
    expect(batchBtn).toBeDefined();
    expect(batchBtn!).toContain('disabled=""');

    const drawerHtml = renderToStaticMarkup(<SessionVideosDrawer />);
    const drawerBtn = drawerHtml.split('<button').find((chunk) => chunk.includes('Process Batch ASR'));
    expect(drawerBtn).toBeDefined();
    expect(drawerBtn!).toContain('disabled=""');
  });

  test('I4 Gap 4: pollJob populates transcriptOptions, triggers showSegmentationModal, and syncs detected_language', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_poll',
      currentStep: 1,
      segments: [],
      transcriptOptions: null,
      showSegmentationModal: false,
      backend: {
        ...initialSnapshot.backend,
        ready: true,
        mediaId: 'm_poll',
        status: 'ready',
        options: {
          ...initialSnapshot.backend.options,
          languages: [
            { code: 'zh-cn', name: 'Chinese' },
            { code: 'en', name: 'English' },
            { code: 'vi', name: 'Vietnamese' },
          ],
        },
      },
      languages: {
        source: { code: 'zh-cn', name: 'Chinese' },
        target: { code: 'vi', name: 'Vietnamese' },
      },
    });

    const utterances = [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hello world' } as any];
    const paragraphs = [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hello world' } as any];

    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/jobs/job_poll_1') {
        return jsonRes({
          id: 'job_poll_1',
          status: 'succeeded',
          progress: 100,
          message: 'Done',
          segments: utterances,
          transcriptOptions: { utterances, paragraphs },
          details: { detected_language: 'en' },
        });
      }
      if (s === '/api/jobs/job_poll_2') {
        return jsonRes({
          id: 'job_poll_2',
          status: 'succeeded',
          progress: 100,
          message: 'Done',
          segments: utterances,
          details: { transcript_options: { utterances, paragraphs }, detected_language: 'en' },
        });
      }
      return jsonRes({ ok: true, projects: [] });
    }) as any;

    await useDubDubStore.getState().pollJob('job_poll_1', 'p_poll');
    const st1 = useDubDubStore.getState();
    expect(st1.transcriptOptions?.utterances?.length).toBe(1);
    expect(st1.transcriptOptions?.paragraphs?.length).toBe(1);
    expect(st1.showSegmentationModal).toBe(true);
    expect(st1.segments.length).toBe(1);
    expect(st1.languages.source.code).toBe('en');
    expect(st1.currentStep).toBe(2);

    // Also verify details.transcript_options fallback on a second polled job
    useDubDubStore.setState({
      activeProjectId: 'p_poll',
      currentStep: 1,
      segments: [],
      transcriptOptions: null,
      showSegmentationModal: false,
      languages: {
        source: { code: 'zh-cn', name: 'Chinese' },
        target: { code: 'vi', name: 'Vietnamese' },
      },
    });
    await useDubDubStore.getState().pollJob('job_poll_2', 'p_poll');
    const st2 = useDubDubStore.getState();
    expect(st2.transcriptOptions?.utterances?.length).toBe(1);
    expect(st2.transcriptOptions?.paragraphs?.length).toBe(1);
    expect(st2.showSegmentationModal).toBe(true);
    expect(st2.languages.source.code).toBe('en');
  });

  test('I4 Gap 5: deleteProjectById/deleteProjectsByIds prune sessionSubprojects and SessionVideosDrawer ignores Stage 4 render', async () => {
    globalThis.fetch = (async () => jsonRes({ ok: true, projects: [], deleted: ['sub_1', 'sub_2'] })) as any;

    // 5a-1: deleteProjectById prunes deleted subproject and updates batchModeActive / parent state
    useDubDubStore.setState({
      activeParentProjectId: 'parent_2026-11-18',
      activeParentProjectName: '2026-11-18',
      batchModeActive: true,
      activeProjectId: 'sub_1',
      activeView: 'dubbing',
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1, status: 'pending' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' } as any,
      ],
      projectJobStatuses: {
        sub_1: { status: 'pending', progress: 0, message: '', stage: 'prepare' },
        sub_2: { status: 'pending', progress: 0, message: '', stage: 'prepare' },
      },
    });

    await useDubDubStore.getState().deleteProjectById('sub_1');
    const afterOne = useDubDubStore.getState();
    expect(afterOne.sessionSubprojects.map((p) => p.id)).toEqual(['sub_2']);
    expect(afterOne.batchModeActive).toBe(true);
    expect(afterOne.activeParentProjectId).toBe('parent_2026-11-18');
    expect(afterOne.activeProjectId).toBeNull();
    expect(afterOne.projectJobStatuses['sub_1']).toBeUndefined();

    await useDubDubStore.getState().deleteProjectById('sub_2');
    const afterTwo = useDubDubStore.getState();
    expect(afterTwo.sessionSubprojects).toEqual([]);
    expect(afterTwo.batchModeActive).toBe(false);
    expect(afterTwo.activeParentProjectId).toBeNull();
    expect(afterTwo.activeParentProjectName).toBeNull();

    // 5a-2: deleteProjectsByIds prunes multiple deleted subprojects and clears batchModeActive
    useDubDubStore.setState({
      activeParentProjectId: 'parent_2026-11-18',
      activeParentProjectName: '2026-11-18',
      batchModeActive: true,
      activeProjectId: 'sub_1',
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 1 } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1 } as any,
      ],
    });
    await useDubDubStore.getState().deleteProjectsByIds(['sub_1', 'sub_2']);
    const afterBulk = useDubDubStore.getState();
    expect(afterBulk.sessionSubprojects).toEqual([]);
    expect(afterBulk.batchModeActive).toBe(false);
    expect(afterBulk.activeParentProjectId).toBeNull();
    expect(afterBulk.activeProjectId).toBeNull();

    // 5b: SessionVideosDrawer keeps 'Process Batch ASR' enabled when another subproject is running stage: 'render'
    useDubDubStore.setState({
      currentStep: 1,
      activeProjectId: 'sub_2',
      sessionDrawerOpen: true,
      sessionSubprojects: [
        { id: 'sub_1', name: 'v1.mp4', media_id: 'm1', stage: 4, status: 'completed' } as any,
        { id: 'sub_2', name: 'v2.mp4', media_id: 'm2', stage: 1, status: 'pending' } as any,
      ],
      projectJobStatuses: {
        sub_1: { jobId: 'job_render_1', status: 'running', progress: 50, message: 'Rendering', stage: 'render' },
      },
    });
    const drawerHtml = renderToStaticMarkup(<SessionVideosDrawer />);
    const drawerBtn = drawerHtml.split('<button').find((chunk) => chunk.includes('Process Batch ASR'));
    expect(drawerBtn).toBeDefined();
    expect(drawerBtn!).not.toContain('disabled=""');
  });

  test('I5 Gap 1: Stage 4 StatusFooter renders Cancel Render while busy, updateBackendConfig mutates config, blank-cue translation preserves IDs, and ApiProvidersTab allows editing fromEnv keys', async () => {
    const { ApiProvidersTab } = await import('../src/components/settings/tabs/ApiProvidersTab');

    // 1. Stage 4 busy Cancel Render button
    useDubDubStore.setState({
      currentStep: 4,
      activeProjectId: 'proj_s4_busy',
      editVideo: { ...initialSnapshot.editVideo, exporting: true, error: null },
      jobStatus: 'running',
      jobStage: 'render',
    });
    const busyFooter = renderToStaticMarkup(<StatusFooter />);
    expect(busyFooter).toContain('data-testid="cancel-render-btn"');
    expect(busyFooter).toContain('Cancel Render');
    expect(busyFooter).toContain('Rendering Video…');

    // 2. updateBackendConfig supports both (key, value) and partial object
    useDubDubStore.getState().updateBackendConfig('ttsType', 3);
    expect(useDubDubStore.getState().backend.config.ttsType).toBe(3);
    useDubDubStore.getState().updateBackendConfig({ ttsType: 1, modelName: 'nova-2' });
    expect(useDubDubStore.getState().backend.config.ttsType).toBe(1);
    expect(useDubDubStore.getState().backend.config.modelName).toBe('nova-2');

    // 3. Blank-cue translation ID alignment when backend omits or includes blank cue
    useDubDubStore.setState({
      activeProjectId: 'proj_blank_cue',
      currentStep: 2,
      segments: [
        { id: 1, startSec: 0, endSec: 1, sourceText: 'First', targetText: '' } as any,
        { id: 2, startSec: 1, endSec: 2, sourceText: '', targetText: '' } as any,
        { id: 3, startSec: 2, endSec: 3, sourceText: 'Third', targetText: '' } as any,
      ],
    });
    globalThis.fetch = (async (url: string | URL | Request) => {
      if (String(url) === '/api/translate') {
        return jsonRes({
          ok: true,
          segments: [
            { id: 1, sourceText: 'First', targetText: 'Mot' },
            { id: 3, sourceText: 'Third', targetText: 'Ba' },
          ],
        });
      }
      return jsonRes({ ok: true });
    }) as any;

    await useDubDubStore.getState().runBatchTranslation();
    const segsAfter = useDubDubStore.getState().segments;
    expect(segsAfter.map((s) => s.targetText)).toEqual(['Mot', '', 'Ba']);

    // 4. ApiProvidersTab does not disable API key input when fromEnv is true
    const providersHtml = renderToStaticMarkup(
      <ApiProvidersTab
        providers={{
          deepgram: {
            id: 'deepgram',
            name: 'Deepgram',
            category: 'asr',
            configured: true,
            fromEnv: true,
            apiKey: 'dg-secret-env',
            model: 'nova-3',
            models: ['nova-3'],
          },
        }}
        onRefresh={async () => {}}
      />
    );
    expect(providersHtml).toContain('ENV OVERRIDE');
    const keyInput = providersHtml.split('<input').find((c) => c.includes('value="dg-secret-env"'));
    expect(keyInput).toBeDefined();
    expect(keyInput!).not.toContain('disabled=""');
  });

  test('I5 Gap 2: CreateVoiceModal probeAudioElementDuration handles WebM Infinity seek & recording fallback, and resolveCloneAudioUploadSource uploads trimmed audio without cut_start/cut_end', async () => {
    const { probeAudioElementDuration, resolveCloneAudioUploadSource } = await import('../src/components/CreateVoiceModal');

    // 1. WebM Infinity duration resolved by seeking currentTime = 1e101 -> timeupdate
    class SeekResolvingAudio {
      duration = Number.POSITIVE_INFINITY;
      onloadedmetadata: (() => void) | null = null;
      onerror: (() => void) | null = null;
      private _currentTime = 0;
      private listeners: Record<string, Array<() => void>> = {};

      constructor(_src?: string) {}

      addEventListener(type: string, cb: () => void) {
        this.listeners[type] = this.listeners[type] || [];
        this.listeners[type].push(cb);
      }

      removeEventListener(type: string, cb: () => void) {
        this.listeners[type] = (this.listeners[type] || []).filter((fn) => fn !== cb);
      }

      get currentTime() {
        return this._currentTime;
      }

      set currentTime(val: number) {
        this._currentTime = val;
        if (val > 1e50) {
          this.duration = 6.4;
          (this.listeners['timeupdate'] || []).slice().forEach((cb) => cb());
        }
      }
    }

    let resolvedSeekDur = 0;
    const seekAudio = probeAudioElementDuration(
      'blob:webm-seek',
      5.0,
      (d) => {
        resolvedSeekDur = d;
      },
      SeekResolvingAudio as any
    );
    seekAudio.onloadedmetadata?.(new Event('loadedmetadata'));
    expect(resolvedSeekDur).toBe(6.4);
    expect(seekAudio.currentTime).toBe(0);

    // 2. WebM Infinity duration that remains Infinity falls back to recordingSecondsRef (including < 2.0s short recordings)
    class UnresolvableWebmAudio {
      duration = Number.POSITIVE_INFINITY;
      onloadedmetadata: (() => void) | null = null;
      onerror: (() => void) | null = null;
      currentTime = 0;
      addEventListener() {}
      removeEventListener() {}
    }

    let shortRecDur = 0;
    const shortAudio = probeAudioElementDuration(
      'blob:webm-short',
      1.0,
      (d) => {
        shortRecDur = d;
      },
      UnresolvableWebmAudio as any
    );
    shortAudio.onloadedmetadata?.(new Event('loadedmetadata'));
    await flushMicrotasks(175);
    expect(shortRecDur).toBe(1.0);

    // 3. resolveCloneAudioUploadSource uploads distinct trimmed file without cut_start/cut_end
    const rawFile = new File(['raw-45s-audio'], 'raw.wav', { type: 'audio/wav' });
    const trimmedFile = new File(['trimmed-8s-audio'], 'trimmed.wav', { type: 'audio/wav' });

    const afterApplyTrim = resolveCloneAudioUploadSource({
      isTrimmed: true,
      isTrimming: false,
      audioFile: trimmedFile,
      audioBlob: trimmedFile,
      rawAudioFile: rawFile,
      rawAudioBlob: rawFile,
    });
    expect(afterApplyTrim.sourceAudio).toBe(trimmedFile);
    expect(afterApplyTrim.useCutRange).toBe(false);

    const whileTrimmingRaw = resolveCloneAudioUploadSource({
      isTrimmed: false,
      isTrimming: true,
      audioFile: rawFile,
      audioBlob: rawFile,
      rawAudioFile: rawFile,
      rawAudioBlob: rawFile,
    });
    expect(whileTrimmingRaw.sourceAudio).toBe(rawFile);
    expect(whileTrimmingRaw.useCutRange).toBe(true);
  });
});
