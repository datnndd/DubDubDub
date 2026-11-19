import { describe, expect, test } from 'bun:test';
import { useDubDubStore } from '../src/store';

function jsonRes(body: any, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('M5 Stage 3 Batch TTS & dubbingSlice integration', () => {
  test('groups segments by provider:voice, chunks by 12, and surfaces partial failures', async () => {
    const originalFetch = globalThis.fetch;
    const batchRequests: any[] = [];

    const segments = Array.from({ length: 14 }, (_, idx) => ({
      id: idx + 1,
      startSec: idx * 2,
      endSec: idx * 2 + 1.5,
      sourceText: `Src ${idx + 1}`,
      targetText: `Target ${idx + 1}`,
      speakerId: idx < 13 ? 'spk_1' : 'spk_2',
    }));

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/api/tts/batch')) {
        const body = JSON.parse(String(init?.body || '{}'));
        batchRequests.push(body);
        return Promise.resolve(jsonRes({
          ok: true,
          results: body.segments.map((seg: any) => (
            seg.segmentId === 3
              ? { segmentId: seg.segmentId, ok: false, error: 'Segment 3 synthesis error' }
              : {
                  segmentId: seg.segmentId,
                  ok: true,
                  id: `prev_${seg.segmentId}`,
                  previewUrl: `/api/tts/preview/prev_${seg.segmentId}/audio`,
                  appliedSpeed: 1.0,
                }
          )),
        }));
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_batch_1',
        voices: [
          { id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' },
          { id: 'Thanh Bình', name: 'Thanh Bình', provider: 2, kind: 'preset' },
        ] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh', spk_2: 'Thanh Bình' },
        segmentVoiceOverrides: {},
        segments: segments as any,
      });

      await useDubDubStore.getState().runFullDubbing();

      // 13 items for Mai Anh -> 12 + 1; 1 item for Thanh Bình -> 1
      expect(batchRequests.length).toBe(3);
      expect(batchRequests[0].segments.length).toBe(12);
      expect(batchRequests[1].segments.length).toBe(1);
      expect(batchRequests[2].segments.length).toBe(1);

      const st = useDubDubStore.getState();
      expect(st.dubbingStatus).toBe('failed');
      expect(st.failedSegmentIds).toEqual([3]);
      expect(st.dubbingErrors[3]).toBe('Segment 3 synthesis error');
      expect(st.segments[0].previewAudioId).toBe('prev_1');
      expect(st.segments[13].previewAudioId).toBe('prev_14');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('does NOT fall back to previewTTS on HTTP 500 batch errors', async () => {
    const originalFetch = globalThis.fetch;
    let previewCalls = 0;

    globalThis.fetch = ((url: string | URL | Request) => {
      const s = String(url);
      if (s.includes('/api/tts/batch')) {
        return Promise.resolve(jsonRes({ detail: 'TTS backend crashed' }, 500));
      }
      if (s.includes('/api/tts/preview')) {
        previewCalls++;
        return Promise.resolve(jsonRes({ preview_url: '/audio/fallback.wav', id: 'prev_fb' }));
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_batch_500',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      await useDubDubStore.getState().runFullDubbing();

      const st = useDubDubStore.getState();
      expect(previewCalls).toBe(0);
      expect(st.dubbingStatus).toBe('failed');
      expect(st.dubbingErrors[1]).toContain('TTS backend crashed');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('falls back cleanly to previewTTS on HTTP 404 from /api/tts/batch', async () => {
    const originalFetch = globalThis.fetch;
    let previewCalls = 0;

    globalThis.fetch = ((url: string | URL | Request) => {
      const s = String(url);
      if (s.includes('/api/tts/batch')) {
        return Promise.resolve(jsonRes({ detail: 'Not Found' }, 404));
      }
      if (s.includes('/api/tts/preview')) {
        previewCalls++;
        return Promise.resolve(jsonRes({ preview_url: '/api/tts/preview/prev_404/audio', id: 'prev_404', applied_speed: 1.0 }));
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_batch_404',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      await useDubDubStore.getState().runFullDubbing();

      const st = useDubDubStore.getState();
      expect(previewCalls).toBe(1);
      expect(st.dubbingStatus).toBe('completed');
      expect(st.segments[0].previewAudioId).toBe('prev_404');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('cancelDubbing aborts in-flight batch, notifies cancel endpoint, and guards restart tokens', async () => {
    const originalFetch = globalThis.fetch;
    const batchSignals: AbortSignal[] = [];
    const cancelCalls: string[] = [];
    let resolveFirst!: (r: Response) => void;
    let resolveSecond!: (r: Response) => void;
    let batchCount = 0;

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/api/tts/batch/') && s.endsWith('/cancel')) {
        cancelCalls.push(s);
        return Promise.resolve(jsonRes({ ok: true, cancelled: true }));
      }
      if (s.includes('/api/tts/batch')) {
        batchCount++;
        if (init?.signal) batchSignals.push(init.signal);
        if (batchCount === 1) {
          return new Promise<Response>((res) => {
            resolveFirst = res;
          });
        }
        return new Promise<Response>((res) => {
          resolveSecond = res;
        });
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_batch_cancel',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const run1 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));
      useDubDubStore.getState().cancelDubbing();
      expect(batchSignals[0]?.aborted).toBe(true);
      expect(cancelCalls.length).toBe(1);

      const run2 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));
      expect(useDubDubStore.getState().dubbingStatus).toBe('running');

      resolveFirst(jsonRes({ ok: true, results: [{ segmentId: 1, ok: true, previewUrl: '/old.wav', id: 'old' }] }));
      await run1;
      expect(useDubDubStore.getState().dubbingStatus).toBe('running');

      resolveSecond(jsonRes({ ok: true, results: [{ segmentId: 1, ok: true, previewUrl: '/new.wav', id: 'new' }] }));
      await run2;
      expect(useDubDubStore.getState().dubbingStatus).toBe('completed');
      expect(useDubDubStore.getState().segments[0].previewAudioId).toBe('new');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('run1 settlement after cancel+restart does NOT delete run2 activeBatchId', async () => {
    const originalFetch = globalThis.fetch;
    const cancelCalls: string[] = [];
    const postedBatchIds: string[] = [];
    let resolveFirst!: (r: Response) => void;
    let resolveSecond!: (r: Response) => void;
    let batchCount = 0;

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/api/tts/batch/') && s.endsWith('/cancel')) {
        cancelCalls.push(s);
        return Promise.resolve(jsonRes({ ok: true, cancelled: true }));
      }
      if (s.includes('/api/tts/batch')) {
        const body = JSON.parse(String(init?.body || '{}'));
        postedBatchIds.push(body.batchId);
        batchCount++;
        if (batchCount === 1) {
          return new Promise<Response>((res) => { resolveFirst = res; });
        }
        return new Promise<Response>((res) => { resolveSecond = res; });
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_batch_race',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const run1 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));
      useDubDubStore.getState().cancelDubbing();

      const run2 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));

      // Settle run1 while run2 is active; run1 finally must NOT wipe run2's activeBatchId
      resolveFirst(jsonRes({ ok: false, results: [] }));
      await run1;

      useDubDubStore.getState().cancelDubbing();
      expect(cancelCalls.length).toBe(2);
      expect(cancelCalls[1]).toContain(postedBatchIds[1]);

      resolveSecond(jsonRes({ ok: false, results: [] }));
      await run2;
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('omitted segment result in batchRes.results never steals another segment audio', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = ((url: string | URL | Request) => {
      const s = String(url);
      if (s.includes('/api/tts/batch')) {
        // Omit segment 1 completely; only return segment 2
        return Promise.resolve(jsonRes({
          ok: false,
          results: [
            { segmentId: 2, ok: true, id: 'prev_2', previewUrl: '/api/tts/preview/prev_2/audio', appliedSpeed: 1.0 },
          ],
        }));
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_omitted_seg',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [
          { id: 1, startSec: 0, endSec: 2, sourceText: 'A', targetText: 'Một', speakerId: 'spk_1' },
          { id: 2, startSec: 2, endSec: 4, sourceText: 'B', targetText: 'Hai', speakerId: 'spk_1' },
        ] as any,
      });

      await useDubDubStore.getState().runFullDubbing();
      const st = useDubDubStore.getState();
      expect(st.dubbingStatus).toBe('failed');
      expect(st.segments[0].previewAudioId).toBeUndefined();
      expect(st.segments[0].previewAudioUrl).toBeUndefined();
      expect(st.segments[1].previewAudioId).toBe('prev_2');
      expect(st.failedSegmentIds).toEqual([1]);
      expect(st.dubbingErrors[1]).toContain('Missing batch result');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('cancelDubbing works when runFullDubbing auto-creates a project from null activeProjectId', async () => {
    const originalFetch = globalThis.fetch;
    const cancelCalls: string[] = [];
    const batchSignals: AbortSignal[] = [];
    let resolveBatch!: (r: Response) => void;

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/api/tts/batch/') && s.endsWith('/cancel')) {
        cancelCalls.push(s);
        return Promise.resolve(jsonRes({ ok: true, cancelled: true }));
      }
      if (s.includes('/api/tts/batch')) {
        if (init?.signal) batchSignals.push(init.signal);
        return new Promise<Response>((res) => { resolveBatch = res; });
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    const origCreateNewProject = useDubDubStore.getState().createNewProject;
    try {
      useDubDubStore.setState({
        activeProjectId: null,
        createNewProject: async () => {
          useDubDubStore.setState({ activeProjectId: 'proj_created_auto' });
          return 'proj_created_auto';
        },
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const run = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 10));

      useDubDubStore.getState().cancelDubbing();
      expect(batchSignals[0]?.aborted).toBe(true);
      expect(cancelCalls.length).toBe(1);

      resolveBatch(jsonRes({ ok: false, results: [] }));
      await run;
    } finally {
      useDubDubStore.setState({ createNewProject: origCreateNewProject });
      globalThis.fetch = originalFetch;
    }
  });

  test('enterStage4 halts and does NOT assemble when runFullDubbing is cancelled', async () => {
    const originalFetch = globalThis.fetch;
    let assembleCalls = 0;
    let resolveBatch!: (r: Response) => void;

    globalThis.fetch = ((url: string | URL | Request) => {
      const s = String(url);
      if (s.includes('/dubbing/assemble')) {
        assembleCalls++;
        return Promise.resolve(jsonRes({ ok: true, audio_url: '/api/projects/p/dubbing/audio' }));
      }
      if (s.includes('/api/tts/batch/') && s.endsWith('/cancel')) {
        return Promise.resolve(jsonRes({ ok: true, cancelled: true }));
      }
      if (s.includes('/api/tts/batch')) {
        return new Promise<Response>((res) => { resolveBatch = res; });
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'proj_enter4_cancel',
        dubbingStatus: 'idle',
        assemblingDubbing: false,
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const enterPromise = useDubDubStore.getState().enterStage4();
      await new Promise((r) => setTimeout(r, 10));
      useDubDubStore.getState().cancelDubbing();

      resolveBatch(jsonRes({ ok: false, results: [] }));
      await enterPromise;

      expect(assembleCalls).toBe(0);
      expect(useDubDubStore.getState().dubbingStatus).toBe('idle');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('cancelDubbing aborts run while real createNewProject is suspended after setting activeProjectId', async () => {
    const originalFetch = globalThis.fetch;
    let batchCalls = 0;
    let releaseLoadProjects!: (r: Response) => void;
    let loadProjectsPending = false;

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      const method = (init?.method || 'GET').toUpperCase();
      if (s.includes('/api/tts/batch')) {
        batchCalls++;
        return Promise.resolve(jsonRes({ ok: true, results: [] }));
      }
      if (method === 'POST' && s.endsWith('/api/projects')) {
        return Promise.resolve(jsonRes({ id: 'proj_real_auto', name: 'Auto Created' }));
      }
      if (s.includes('/api/projects/proj_real_auto')) {
        return Promise.resolve(jsonRes({
          id: 'proj_real_auto',
          name: 'Auto Created',
          state_json: JSON.stringify({
            segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }],
            speakerVoiceMap: { spk_1: 'Mai Anh' },
          }),
        }));
      }
      if (method === 'GET' && s.endsWith('/api/projects')) {
        loadProjectsPending = true;
        return new Promise<Response>((res) => {
          releaseLoadProjects = res;
        });
      }
      return Promise.resolve(jsonRes({ ok: true, projects: [] }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: null,
        activeParentProjectId: null,
        dubbingStatus: 'idle',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const run = useDubDubStore.getState().runFullDubbing();

      // Wait until selectProject() inside real createNewProject() has updated activeProjectId
      // while loadProjects() is still suspended
      for (let i = 0; i < 50; i++) {
        if (useDubDubStore.getState().activeProjectId === 'proj_real_auto' && loadProjectsPending) {
          break;
        }
        await new Promise((r) => setTimeout(r, 5));
      }
      expect(useDubDubStore.getState().activeProjectId).toBe('proj_real_auto');
      expect(useDubDubStore.getState().segments.length).toBe(1);
      expect(loadProjectsPending).toBe(true);

      useDubDubStore.getState().cancelDubbing();
      expect(useDubDubStore.getState().dubbingStatus).toBe('idle');

      releaseLoadProjects(jsonRes({ projects: [] }));
      await run;

      expect(batchCalls).toBe(0);
      expect(useDubDubStore.getState().dubbingStatus).toBe('idle');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('1c-Twist-B: cancelled Run 1 mid-createNewProject does not hijack Run 2 activeProjectId or cancel Run 2', async () => {
    const originalFetch = globalThis.fetch;
    let createCalls = 0;
    let resolveCreate1!: (r: Response) => void;
    let resolveBatch2!: (r: Response) => void;
    let getProj1Calls = 0;
    let batchCalls = 0;
    const batchSignals: AbortSignal[] = [];
    const cancelCalls: string[] = [];

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      const method = (init?.method || 'GET').toUpperCase();
      if (s.includes('/api/tts/batch/') && s.endsWith('/cancel')) {
        cancelCalls.push(s);
        return Promise.resolve(jsonRes({ ok: true, cancelled: true }));
      }
      if (s.includes('/api/tts/batch')) {
        batchCalls++;
        if (init?.signal) batchSignals.push(init.signal);
        return new Promise<Response>((res) => {
          resolveBatch2 = res;
        });
      }
      if (method === 'POST' && s.endsWith('/api/projects')) {
        createCalls++;
        if (createCalls === 1) {
          return new Promise<Response>((res) => {
            resolveCreate1 = res;
          });
        }
        return Promise.resolve(jsonRes({ id: 'proj_tb_2', name: 'Twist B 2' }));
      }
      if (s.includes('/api/projects/proj_tb_1')) {
        getProj1Calls++;
        return Promise.resolve(jsonRes({
          id: 'proj_tb_1',
          name: 'Twist B 1',
          state_json: JSON.stringify({
            segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Stale', targetText: 'Cũ', speakerId: 'spk_1' }],
            speakerVoiceMap: { spk_1: 'Mai Anh' },
          }),
        }));
      }
      if (s.includes('/api/projects/proj_tb_2')) {
        return Promise.resolve(jsonRes({
          id: 'proj_tb_2',
          name: 'Twist B 2',
          state_json: JSON.stringify({
            segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }],
            speakerVoiceMap: { spk_1: 'Mai Anh' },
          }),
        }));
      }
      return Promise.resolve(jsonRes({ ok: true, projects: [] }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: null,
        activeParentProjectId: null,
        dubbingStatus: 'idle',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const run1 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));
      expect(createCalls).toBe(1);

      useDubDubStore.getState().cancelDubbing();
      expect(useDubDubStore.getState().dubbingStatus).toBe('idle');

      const run2 = useDubDubStore.getState().runFullDubbing();
      for (let i = 0; i < 50; i++) {
        if (batchCalls === 1 && useDubDubStore.getState().activeProjectId === 'proj_tb_2') break;
        await new Promise((r) => setTimeout(r, 5));
      }
      expect(useDubDubStore.getState().activeProjectId).toBe('proj_tb_2');
      expect(useDubDubStore.getState().dubbingStatus).toBe('running');
      expect(batchCalls).toBe(1);
      expect(batchSignals[0]?.aborted).toBe(false);

      // Now Run 1's slow POST /api/projects resolves late
      resolveCreate1(jsonRes({ id: 'proj_tb_1', name: 'Twist B 1' }));
      await run1;

      expect(getProj1Calls).toBe(0);
      expect(useDubDubStore.getState().activeProjectId).toBe('proj_tb_2');
      expect(useDubDubStore.getState().dubbingStatus).toBe('running');
      expect(batchSignals[0]?.aborted).toBe(false);

      // Run 2 must still be independently cancellable
      useDubDubStore.getState().cancelDubbing();
      expect(batchSignals[0]?.aborted).toBe(true);
      expect(cancelCalls.length).toBe(1);
      expect(useDubDubStore.getState().dubbingStatus).toBe('idle');

      resolveBatch2(jsonRes({ ok: false, results: [] }));
      await run2;
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('rapid double-start while Run 1 is in loadProjects after selectProject lets Run 2 win without being aborted by Run 1', async () => {
    const originalFetch = globalThis.fetch;
    let releaseLoadProjects!: (r: Response) => void;
    let loadProjectsCalls = 0;
    let batchCalls = 0;
    const batchSignals: AbortSignal[] = [];

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      const method = (init?.method || 'GET').toUpperCase();
      if (s.includes('/api/tts/batch')) {
        batchCalls++;
        if (init?.signal) batchSignals.push(init.signal);
        return Promise.resolve(jsonRes({
          ok: true,
          results: [{ segmentId: 1, ok: true, id: 'prev_rapid_2', previewUrl: '/api/tts/preview/prev_rapid_2/audio', appliedSpeed: 1.0 }],
        }));
      }
      if (method === 'POST' && s.endsWith('/api/projects')) {
        return Promise.resolve(jsonRes({ id: 'proj_rapid_1', name: 'Rapid 1' }));
      }
      if (s.includes('/api/projects/proj_rapid_1')) {
        return Promise.resolve(jsonRes({
          id: 'proj_rapid_1',
          name: 'Rapid 1',
          state_json: JSON.stringify({
            segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }],
            speakerVoiceMap: { spk_1: 'Mai Anh' },
          }),
        }));
      }
      if (method === 'GET' && s.endsWith('/api/projects')) {
        loadProjectsCalls++;
        if (loadProjectsCalls === 1) {
          return new Promise<Response>((res) => {
            releaseLoadProjects = res;
          });
        }
        return Promise.resolve(jsonRes({ projects: [] }));
      }
      return Promise.resolve(jsonRes({ ok: true, projects: [] }));
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: null,
        activeParentProjectId: null,
        dubbingStatus: 'idle',
        voices: [{ id: 'Mai Anh', name: 'Mai Anh', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        speakerVoiceMap: { spk_1: 'Mai Anh' },
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' }] as any,
      });

      const run1 = useDubDubStore.getState().runFullDubbing();
      for (let i = 0; i < 50; i++) {
        if (useDubDubStore.getState().activeProjectId === 'proj_rapid_1' && loadProjectsCalls === 1) {
          break;
        }
        await new Promise((r) => setTimeout(r, 5));
      }
      expect(useDubDubStore.getState().activeProjectId).toBe('proj_rapid_1');
      expect(loadProjectsCalls).toBe(1);

      // Rapid double-start WITHOUT calling cancelDubbing()
      const run2 = useDubDubStore.getState().runFullDubbing();
      await run2;

      expect(batchCalls).toBe(1);
      expect(batchSignals[0]?.aborted).toBe(false);
      expect(useDubDubStore.getState().dubbingStatus).toBe('completed');
      expect(useDubDubStore.getState().segments[0].previewAudioId).toBe('prev_rapid_2');

      // Now release Run 1's loadProjects() — Run 1 must exit cleanly without aborting or restarting batchTTS
      releaseLoadProjects(jsonRes({ projects: [] }));
      await run1;

      expect(batchCalls).toBe(1);
      expect(useDubDubStore.getState().dubbingStatus).toBe('completed');
      expect(useDubDubStore.getState().segments[0].previewAudioId).toBe('prev_rapid_2');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
