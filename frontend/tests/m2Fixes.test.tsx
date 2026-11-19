import React from 'react';
import { beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { useDubDubStore } from '../src/store';
import { formatLocalDateYmd, getYesterdayLocalDateYmd, parseProjectDate } from '../src/utils/date';
import { isParentProject } from '../src/utils/project';
import { getSubprojectStatusInfo } from '../src/utils/projectStatus';
import { getVideoDedupKey } from '../src/components/FolderScanModal';
import { Header } from '../src/components/Header';
import { SessionVideosDrawer } from '../src/components/SessionVideosDrawer';
import { StatusFooter } from '../src/components/StatusFooter';
import { WorkflowStepper } from '../src/components/WorkflowStepper';
import { VideoPlayer } from '../src/components/VideoPlayer';
import { ProjectsScreen } from '../src/screens/ProjectsScreen';
import { Stage2OcrDialog, normalizePointerPoint } from '../src/components/Stage2OcrDialog';
import { Stage2ReviewTranscript } from '../src/screens/Stage2ReviewTranscript';
import { Stage3VoiceDubbing } from '../src/screens/Stage3VoiceDubbing';
import { Stage4EditVideo } from '../src/screens/Stage4EditVideo';
import { VoiceManagementScreen, matchesGenderFilter } from '../src/screens/VoiceManagementScreen';
import { generateSrtContent } from '../src/components/CapCutExportModal';
import { getProviderDocInfo } from '../src/config/providerDocs';
import { pickDefaultVoice } from '../src/store/dubbingSlice';

function jsonRes(body: any, status: number = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('M2 Fixes — Comprehensive Unit & Component Coverage', () => {
  beforeEach(() => {
    useDubDubStore.setState({
      activeProjectId: 'proj_m2',
      activeView: 'dubbing',
      currentStep: 1,
      maxUnlockedStep: 1,
      activeParentProjectId: null,
      activeParentProjectName: null,
      sessionSubprojects: [],
      projectsList: [],
      projectJobStatuses: {},
      segments: [],
      speakerVoiceMap: {},
      segmentVoiceOverrides: {},
      voices: [],
      customVoices: [],
      dubbingStatus: 'idle',
      dubbingError: null,
      dubbingErrors: {},
      failedSegmentIds: [],
      assemblingDubbing: false,
      assembledDubUrl: null,
      jobStatus: 'idle',
      jobProgress: null,
      jobStage: null,
      jobMessage: 'Ready to process',
      translationModal: {
        active: false,
        status: 'idle',
        progress: 0,
        message: 'Translating transcript...',
        error: null,
      },
      backend: {
        ...useDubDubStore.getState().backend,
        ready: true,
        status: 'idle',
        mediaId: 'med_1',
        options: {
          languages: [
            { code: 'zh-cn', name: 'Simplified Chinese' },
            { code: 'vi', name: 'Vietnamese' },
          ],
          asrProviders: [
            { id: 'deepgram', label: 'Deepgram', recognType: 1, models: ['nova-3'], requiresSettings: true, configured: true },
          ],
          translationProviders: [
            { id: 'google', label: 'Google Translate', translateType: 0, requiresSettings: false, configured: true },
          ],
          translationModes: [{ id: 'srt', label: 'Send SRT' }],
          voices: [[2, 'VieNeu-TTS'], [4, 'Edge-TTS']],
          hardware: { cudaAvailable: false, gpuCount: 0 },
        },
        config: {
          ...useDubDubStore.getState().backend.config,
          recognType: 1,
          modelName: 'nova-3',
          translateType: 0,
          translationMode: 'srt',
          ttsType: 2,
        },
      },
      languages: {
        source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
        target: { code: 'vi', name: 'Vietnamese' },
        timingMode: 'voice',
      },
      project: {
        ...useDubDubStore.getState().project,
        filename: 'sample.mp4',
        verified: true,
        durationSec: 12,
        duration: '00:12.000',
        lastSaved: '10:00',
      },
      playback: {
        currentTime: 0,
        formattedTime: '00:00.000',
        duration: 12,
        isPlaying: false,
        playbackSpeed: 1.0,
        audioChannel: 'dub',
        stopAtTime: null,
        seekRequest: null,
      },
      editVideo: {
        audioMix: { original: 0, dubbed: 100, background: 35 },
        backgroundAudio: null,
        thumbnail: null,
        exporting: false,
        error: null,
        activeTab: 'audio',
        isCapCutModalOpen: false,
      },
    });
  });

  test('Item #1: local calendar date helpers format local YYYY-MM-DD, yesterday across boundaries, and parse epoch seconds', () => {
    const marchFirst = new Date(2026, 2, 1, 0, 5, 0);
    expect(formatLocalDateYmd(marchFirst)).toBe('2026-03-01');
    expect(getYesterdayLocalDateYmd(marchFirst)).toBe('2026-02-28');

    const janFirst = new Date(2026, 0, 1, 0, 10, 0);
    expect(getYesterdayLocalDateYmd(janFirst)).toBe('2025-12-31');

    const fromEpochSec = parseProjectDate(1728450000);
    expect(fromEpochSec).not.toBeNull();
    expect(fromEpochSec!.getFullYear()).toBeGreaterThanOrEqual(2024);

    const fromIso = parseProjectDate('2026-11-18T12:30:00Z');
    expect(fromIso?.toISOString()).toBe('2026-11-18T12:30:00.000Z');
    expect(parseProjectDate(null)).toBeNull();
    expect(parseProjectDate('')).toBeNull();
  });

  test('Item #2: path dedup key distinguishes subfolder videos and setSessionSubprojects merges without wiping', async () => {
    expect(getVideoDedupKey({ path: 'ep1\\video.mp4', filename: 'video.mp4' })).toBe('ep1/video.mp4');
    expect(getVideoDedupKey({ path: 'ep2/video.mp4', filename: 'video.mp4' })).toBe('ep2/video.mp4');
    expect(getVideoDedupKey({ path: 'ep1/video.mp4', filename: 'video.mp4' })).not.toBe(
      getVideoDedupKey({ path: 'ep2/video.mp4', filename: 'video.mp4' })
    );

    const store = useDubDubStore.getState() as any;
    store.setSessionSubprojects([{ id: 's1', name: 'ep1/video.mp4', stage: 1, status: 'pending' }], 'parent_today', 'Today');
    store.setSessionSubprojects([{ id: 's2', name: 'ep2/video.mp4', stage: 1, status: 'pending' }], 'parent_today', 'Today', { merge: true });
    expect(useDubDubStore.getState().sessionSubprojects.map((s) => s.id)).toEqual(['s1', 's2']);

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string | URL | Request) => {
      if (String(url).includes('/subprojects')) {
        return jsonRes({ subprojects: [] });
      }
      return jsonRes({ id: 's1', name: 'ep1/video.mp4', parent_id: 'parent_today', stage: 1, state: { currentStep: 1 } });
    }) as any;
    try {
      await useDubDubStore.getState().selectProject('s1', false);
      expect(useDubDubStore.getState().sessionSubprojects.map((s) => s.id)).toEqual(['s1', 's2']);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('Item #3: getSubprojectStatusInfo handles failed status, SessionVideosDrawer stage labels & disabled batch ASR, Header HW & Auto-saved', async () => {
    const failedInfo = getSubprojectStatusInfo({ id: 'x', stage: 2, status: 'failed' }, {});
    expect(failedInfo.status).toBe('failed');
    expect(failedInfo.label).toBe('Failed');

    useDubDubStore.setState({
      sessionDrawerOpen: true,
      currentStep: 3,
      sessionSubprojects: [
        { id: 'v1', name: 'one.mp4', stage: 3, status: 'completed' } as any,
        { id: 'v2', name: 'two.mp4', stage: 2, status: 'failed' } as any,
      ],
    });
    const stage3Drawer = renderToStaticMarkup(<SessionVideosDrawer />);
    expect(stage3Drawer).toContain('1 of 2 videos voice-ready');
    expect(stage3Drawer).toContain('Failed');

    useDubDubStore.setState({ currentStep: 1 });
    const stage1NoMediaDrawer = renderToStaticMarkup(<SessionVideosDrawer />);
    expect(stage1NoMediaDrawer).toContain('disabled=""');

    // No-op switchSessionVideo on active project
    const originalFetch = globalThis.fetch;
    let fetchCount = 0;
    globalThis.fetch = (async () => {
      fetchCount++;
      return jsonRes({});
    }) as any;
    try {
      useDubDubStore.setState({ activeProjectId: 'v1' });
      await useDubDubStore.getState().switchSessionVideo('v1');
      expect(fetchCount).toBe(0);
    } finally {
      globalThis.fetch = originalFetch;
    }

    // Header GPU/CPU badge & Auto-saved visibility
    useDubDubStore.setState({
      activeView: 'projects',
      activeProjectId: null,
      projectsList: [{ id: 'parent_2026-11-18', name: '2026-11-18', stage: 1, status: 'active', state: { is_parent: true } }] as any,
      backend: {
        ...useDubDubStore.getState().backend,
        ready: true,
        options: { ...useDubDubStore.getState().backend.options, hardware: { cudaAvailable: true, gpuCount: 1 } },
      },
    });
    const projectsHeader = renderToStaticMarkup(<Header />);
    expect(projectsHeader).toContain('GPU Ready');
    expect(projectsHeader).not.toContain('Auto-saved');
    expect(projectsHeader).not.toContain('Warm GPU');

    useDubDubStore.setState({
      activeView: 'dubbing',
      activeProjectId: 'v1',
      backend: {
        ...useDubDubStore.getState().backend,
        ready: true,
        options: { ...useDubDubStore.getState().backend.options, hardware: { cudaAvailable: false, gpuCount: 0 } },
      },
    });
    const dubbingHeader = renderToStaticMarkup(<Header />);
    expect(dubbingHeader).toContain('CPU Mode');
    expect(dubbingHeader).toContain('Auto-saved');
  });

  test('Item #4: StatusFooter isolates cross-stage busy state, disables non-runnable batch, and exposes Cancel Dubbing', () => {
    useDubDubStore.setState({
      currentStep: 1,
      dubbingStatus: 'running',
      sessionSubprojects: [
        { id: 'a', name: 'a.mp4', stage: 2, status: 'completed' } as any,
        { id: 'b', name: 'b.mp4', stage: 1, status: 'pending' } as any, // no media
      ],
    });
    const stage1Footer = renderToStaticMarkup(<StatusFooter />);
    expect(stage1Footer).not.toContain('Processing ASR…');
    expect(stage1Footer).toContain('Start Batch Dub (2 Videos)');
    expect(stage1Footer).toContain('disabled=""');

    useDubDubStore.setState({ currentStep: 3, dubbingStatus: 'running', jobProgress: 50, jobMessage: 'Creating voice (1/2)…' });
    const stage3Footer = renderToStaticMarkup(<StatusFooter />);
    expect(stage3Footer).toContain('Cancel Dubbing');
  });

  test('Item #5: startPrepareJob populates projectJobStatuses and startBatchPrepareJobs settles idempotently', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/jobs') {
        return jsonRes({
          id: 'job_sync_1',
          status: 'succeeded',
          segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: '' }],
        });
      }
      return jsonRes({ ok: true, projects: [] });
    }) as any;
    try {
      await useDubDubStore.getState().startPrepareJob();
      const ps = useDubDubStore.getState().projectJobStatuses['proj_m2'];
      expect(ps?.status).toBe('completed');
      expect(ps?.progress).toBe(100);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('Item #6: ProjectsScreen filters parent containers, recomputes totals on search, and renders Grouped Table View parity', () => {
    const today = formatLocalDateYmd();
    useDubDubStore.setState({
      projectsList: [
        { id: `parent_${today}`, name: today, stage: 1, status: 'active', state: { is_parent: true } } as any,
      ],
    });
    const emptyHtml = renderToStaticMarkup(<ProjectsScreen />);
    expect(emptyHtml).toContain('No projects yet');
    expect(emptyHtml).toContain('Total: <strong>0</strong> projects');

    useDubDubStore.setState({
      projectsList: [
        { id: `parent_${today}`, name: today, stage: 1, status: 'active', state: { is_parent: true } } as any,
        { id: 'v1', name: 'Alpha.mp4', parent_id: `parent_${today}`, stage: 2, status: 'completed', duration: 60 } as any,
        { id: 'v2', name: 'Beta.mp4', parent_id: `parent_${today}`, stage: 1, status: 'pending', duration: 30 } as any,
      ],
    });
    expect(isParentProject({ id: `parent_${today}`, state: { is_parent: true } })).toBe(true);
    expect(isParentProject({ id: 'v1' })).toBe(false);

    const tableHtml = renderToStaticMarkup(<ProjectsScreen initialViewMode="table" />);
    expect(tableHtml).toContain(`data-testid="checkbox-date-${today}"`);
    expect(tableHtml).toContain(`data-testid="add-video-${today}"`);
    expect(tableHtml).toContain(`data-testid="scan-folders-date-${today}"`);
    expect(tableHtml).toContain('Select All (2)');
    expect(tableHtml).toContain('Total: <strong>2</strong> projects');
  });

  test('Item #7: providerDocs maps ASR 0..5 and LLM 0..3 without falling into TTS', () => {
    expect(getProviderDocInfo(0, 'asr')?.id).toBe('qwen_asr');
    expect(getProviderDocInfo(1, 'asr')?.id).toBe('deepgram');
    expect(getProviderDocInfo(2, 'asr')?.id).toBe('gemini_stt');
    expect(getProviderDocInfo(3, 'asr')?.id).toBe('google_stt');
    expect(getProviderDocInfo(4, 'asr')?.id).toBe('elevenlabs_stt');
    expect(getProviderDocInfo(5, 'asr')?.id).toBe('whisper');
    expect(getProviderDocInfo(0, 'llm')?.id).toBe('google');
    expect(getProviderDocInfo(1, 'llm')?.id).toBe('openai');
    expect(getProviderDocInfo(2, 'llm')?.id).toBe('gemini');
    expect(getProviderDocInfo(3, 'llm')?.id).toBe('deepseek');
    expect(getProviderDocInfo('elevenlabs', 'asr')?.id).toBe('elevenlabs_stt');
    expect(getProviderDocInfo('elevenlabs', 'tts')?.id).toBe('elevenlabs');
    expect(getProviderDocInfo(99, 'llm')).toBeNull();
  });

  test('Item #8: WorkflowStepper shows real dynamic telemetry and VideoPlayer unmutes orig channel & clears silent gaps', () => {
    useDubDubStore.setState({
      currentStep: 2,
      segments: [
        { id: 1, startSec: 1, endSec: 3, sourceText: 'A', targetText: 'B', speakerId: 'spk_1' } as any,
        { id: 2, startSec: 6, endSec: 8, sourceText: 'C', targetText: '', speakerId: 'spk_1' } as any,
      ],
    });
    const stepperHtml = renderToStaticMarkup(<WorkflowStepper />);
    expect(stepperHtml).toContain('2 cues • 1/2 translated');
    expect(stepperHtml).not.toContain('OCR Slide Engine: Active');
    expect(stepperHtml).not.toContain('Neural Voice Sync: 99.8%');
    expect(stepperHtml).not.toContain('Multi-Track Timeline: Synced');

    useDubDubStore.getState().seek(4.5); // silent gap between [1..3] and [6..8]
    expect(useDubDubStore.getState().activeSegmentId).toBeNull();

    useDubDubStore.setState({
      currentStep: 3,
      project: { ...useDubDubStore.getState().project, previewUrl: '/api/media/m1/file' },
      playback: { ...useDubDubStore.getState().playback, currentTime: 4.5, audioChannel: 'orig' },
    });
    const playerHtml = renderToStaticMarkup(<VideoPlayer showAudioSwitcher={true} />);
    expect(playerHtml).toContain('ZH Orig');
    expect(playerHtml).toContain('VI Dub ★');
    expect(playerHtml).not.toContain('data-canvas-subtitle="true"');
    expect(playerHtml).not.toContain('muted=""');
  });

  test('Item #9 & #10: normalizePointerPoint guards zero bounds, Stage2OcrDialog renders Confirm & Apply OCR, translation errors show inline', () => {
    expect(normalizePointerPoint(10, 10, { left: 0, top: 0, width: 0, height: 100 })).toBeNull();
    expect(normalizePointerPoint(50, 25, { left: 0, top: 0, width: 100, height: 100 })).toEqual([0.5, 0.25]);

    const seg1 = { id: 1, startSec: 1, endSec: 3, startTime: '00:01', endTime: '00:03', sourceText: 'S1', targetText: 'T1', speakerId: 'spk_1' } as any;
    const seg2 = { id: 2, startSec: 4, endSec: 6, startTime: '00:04', endTime: '00:06', sourceText: 'S2', targetText: 'T2', speakerId: 'spk_1' } as any;
    const batchHtml = renderToStaticMarkup(
      <Stage2OcrDialog
        segment={seg1}
        batchSegments={[seg1, seg2]}
        batchResults={[{ segment: seg1, text: 'OCR 1', confidence: 0.9, applied: false }]}
        currentSegments={[seg1, seg2]}
        videoUrl="/api/media/m1/file"
        roi={[0.1, 0.7, 0.8, 0.2]}
        onRoiChange={() => {}}
        onExtract={() => {}}
        onConfirmBatchApply={() => {}}
        onClose={() => {}}
        loading={false}
        error={null}
      />
    );
    expect(batchHtml).toContain('Confirm &amp; Apply OCR');
    expect(batchHtml).toContain('data-testid="ocr-frame-scrubber"');

    useDubDubStore.setState({
      translationModal: {
        active: true,
        status: 'failed',
        progress: 0,
        message: 'Translation failed',
        error: '401 Unauthorized: Missing OpenAI API key',
      },
    });
    const stage2Html = renderToStaticMarkup(<Stage2ReviewTranscript />);
    expect(stage2Html).toContain('data-testid="translation-modal-error"');
    expect(stage2Html).toContain('401 Unauthorized: Missing OpenAI API key');
  });

  test('Item #11 & #12: pickDefaultVoice skips No/clone, matchesGenderFilter avoids vietnam false positive, VoiceManagementScreen hides VieNeu controls for ElevenLabs', () => {
    expect(
      pickDefaultVoice([
        { id: 'No', name: 'No Dubbing', kind: 'preset' },
        { id: 'clone', name: 'Clone', kind: 'clone' },
        { id: 'Ngọc', name: 'Ngọc', kind: 'preset' },
      ])
    ).toBe('Ngọc');

    expect(matchesGenderFilter({ id: 'v1', name: 'Vietnam Heritage Female' }, 'male')).toBe(false);
    expect(matchesGenderFilter({ id: 'v2', name: 'Minh (nam miền Bắc)' }, 'male')).toBe(true);

    useDubDubStore.setState({
      backend: {
        ...useDubDubStore.getState().backend,
        config: { ...useDubDubStore.getState().backend.config, ttsType: 0 },
      },
      voices: [{ id: 'Rachel', name: 'Rachel', provider: 0, kind: 'preset' }] as any,
      customVoices: [{ id: 'cv_el', name: 'Eleven Custom', provider: 0, language: 'en' }] as any,
    });
    const vmHtml = renderToStaticMarkup(<VoiceManagementScreen />);
    expect(vmHtml).not.toContain('VieNeu Speaking Style');
    expect(vmHtml).not.toContain('Expressiveness (Temp)');
  });

  test('Item #13, #14, #15: Stage4 BGM stays mounted across tabs, CapCut rawTranslation never uses sourceText, initializeBackend(true) refreshes defaults', () => {
    useDubDubStore.setState({
      currentStep: 4,
      editVideo: {
        ...useDubDubStore.getState().editVideo,
        activeTab: 'audio',
        backgroundAudio: { id: 'a1', file: null, name: 'bgm.mp3', url: '/api/media/m_bgm/file' },
        error: 'Upload failed',
      },
    });
    const stage4Html = renderToStaticMarkup(<Stage4EditVideo />);
    expect(stage4Html).toContain('id="stage4-bgm-preview"');
    expect(stage4Html).toContain('/api/media/m_bgm/file');
    expect(stage4Html).toContain('Upload failed');

    const srt = generateSrtContent(
      [{ id: 1, startSec: 0, endSec: 2, sourceText: 'DO NOT EXPORT SOURCE', targetText: 'Fallback target' }],
      'rawTranslation'
    );
    expect(srt).toContain('Fallback target');
    expect(srt).not.toContain('DO NOT EXPORT SOURCE');
  });

  test('Item #15: initializeBackend(true) updates modelName from defaults when ready is true', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      jsonRes({
        languages: [],
        asrProviders: [],
        translationProviders: [],
        translationModes: [],
        voices: [],
        defaults: {
          recognType: 1,
          modelName: 'nova-3',
          translateType: 0,
          translationMode: 'srt',
          ttsType: 2,
        },
      })) as any;
    try {
      useDubDubStore.setState({
        backend: {
          ...useDubDubStore.getState().backend,
          ready: true,
          config: { ...useDubDubStore.getState().backend.config, modelName: 'nova-2' },
        },
      });
      await useDubDubStore.getState().initializeBackend(false);
      expect(useDubDubStore.getState().backend.config.modelName).toBe('nova-2');
      await useDubDubStore.getState().initializeBackend(true);
      expect(useDubDubStore.getState().backend.config.modelName).toBe('nova-3');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('Iter2 Fix #1: updateTtsProvider updates backend.config.ttsType and Stage3VoiceDubbing reflects new provider', () => {
    useDubDubStore.setState({
      currentStep: 3,
      segments: [
        { id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Xin chào', speakerId: 'spk_1' } as any,
      ],
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
    });
    const beforeHtml = renderToStaticMarkup(<Stage3VoiceDubbing />);
    expect(beforeHtml).toContain('data-testid="vieneu-emotion-chips-1"');

    useDubDubStore.getState().updateTtsProvider(0);
    expect(useDubDubStore.getState().backend.config.ttsType).toBe(0);

    useDubDubStore.setState({
      voices: [{ id: 'Rachel', name: 'Rachel', provider: 0, kind: 'preset' }] as any,
    });
    const afterHtml = renderToStaticMarkup(<Stage3VoiceDubbing />);
    expect(afterHtml).not.toContain('data-testid="vieneu-emotion-chips-1"');
    expect(afterHtml).toContain('data-action="select-tts-provider"');
  });

  test('Iter2 Fix #2a: pollJob resolving after SSE settlement does not overwrite segments or re-trigger autosave', async () => {
    const originalFetch = globalThis.fetch;
    const originalEventSource = (globalThis as any).EventSource;

    let esInstance: any = null;
    (globalThis as any).EventSource = class MockEventSource {
      onmessage: ((e: any) => void) | null = null;
      onerror: ((e: any) => void) | null = null;
      private listeners: Record<string, ((e: any) => void)[]> = {};
      constructor(_url: string) {
        esInstance = this;
      }
      addEventListener(type: string, cb: (e: any) => void) {
        this.listeners[type] = this.listeners[type] || [];
        this.listeners[type].push(cb);
      }
      emit(type: string, data: any) {
        const evt = { data: JSON.stringify(data) };
        if (type === 'message') this.onmessage?.(evt);
        else (this.listeners[type] || []).forEach((cb) => cb(evt));
      }
      close() {}
    };

    let resolveSlowPoll!: (val: Response) => void;
    let pollFetchCount = 0;
    let putAutosaveCount = 0;
    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/api/jobs/job_race_1')) {
        pollFetchCount++;
        if (pollFetchCount === 1) {
          return new Promise<Response>((res) => {
            resolveSlowPoll = res;
          });
        }
        return Promise.resolve(jsonRes({
          id: 'job_race_1',
          status: 'succeeded',
          segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'SSE Seg', targetText: '' }],
        }));
      }
      if (s.includes('/api/projects/proj_m2') && init?.method === 'PUT') {
        putAutosaveCount++;
        return Promise.resolve(jsonRes({ ok: true }));
      }
      return Promise.resolve(jsonRes({ ok: true, projects: [] }));
    }) as any;

    try {
      const store = useDubDubStore.getState();
      store.subscribeToJob('job_race_1', 'proj_m2');

      // Start an in-flight pollJob before SSE settles
      const inFlightPoll = store.pollJob('job_race_1', 'proj_m2');

      // SSE settles first
      esInstance.onmessage({
        data: JSON.stringify({
          status: 'succeeded',
          details: { segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'SSE Seg', targetText: '' }] },
        }),
      });
      await new Promise((r) => setTimeout(r, 20));
      expect(useDubDubStore.getState().segments[0]?.sourceText).toBe('SSE Seg');
      expect(useDubDubStore.getState().jobStatus).toBe('completed');
      const autosaveBeforeLatePoll = putAutosaveCount;

      // Now the slow in-flight poll resolves with stale data
      resolveSlowPoll(jsonRes({
        id: 'job_race_1',
        status: 'succeeded',
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Stale Poll', targetText: '' }],
      }));
      await inFlightPoll;
      await new Promise((r) => setTimeout(r, 20));

      expect(useDubDubStore.getState().segments[0]?.sourceText).toBe('SSE Seg');
      expect(putAutosaveCount).toBe(autosaveBeforeLatePoll);
    } finally {
      useDubDubStore.getState().clearJob();
      globalThis.fetch = originalFetch;
      (globalThis as any).EventSource = originalEventSource;
    }
  });

  test('Iter2 Fix #2b: done({}) with missing status checks fetchJob.status in both subscribeToJob and startBatchPrepareJobs', async () => {
    const originalFetch = globalThis.fetch;
    const originalEventSource = (globalThis as any).EventSource;
    const instances: any[] = [];

    (globalThis as any).EventSource = class MockEventSource {
      onmessage: ((e: any) => void) | null = null;
      onerror: ((e: any) => void) | null = null;
      listeners: Record<string, ((e: any) => void)[]> = {};
      constructor(public url: string) {
        instances.push(this);
      }
      addEventListener(type: string, cb: (e: any) => void) {
        this.listeners[type] = this.listeners[type] || [];
        this.listeners[type].push(cb);
      }
      emitDone(data: any) {
        const evt = { data: JSON.stringify(data) };
        (this.listeners['done'] || []).forEach((cb) => cb(evt));
      }
      close() {}
    };

    globalThis.fetch = (async (url: string | URL | Request) => {
      const s = String(url);
      if (s === '/api/jobs') {
        return jsonRes({ id: 'job_done_fail_batch', status: 'queued' });
      }
      if (s.includes('/api/jobs/job_done_fail_sub')) {
        return jsonRes({ id: 'job_done_fail_sub', status: 'failed', error: 'Sub job crashed' });
      }
      if (s.includes('/api/jobs/job_done_fail_batch')) {
        return jsonRes({ id: 'job_done_fail_batch', status: 'failed', error: 'Batch job crashed' });
      }
      return jsonRes({ ok: true, projects: [] });
    }) as any;

    try {
      // 1. subscribeToJob receiving done({}) when fetchJob reports failed
      useDubDubStore.getState().subscribeToJob('job_done_fail_sub', 'proj_m2');
      await instances[0].emitDone({});
      await new Promise((r) => setTimeout(r, 10));
      expect(useDubDubStore.getState().jobStatus).toBe('failed');
      expect(useDubDubStore.getState().projectJobStatuses['proj_m2']?.status).toBe('failed');
      useDubDubStore.getState().clearJob();

      // 2. startBatchPrepareJobs receiving done({}) when fetchJob reports failed
      useDubDubStore.setState({
        activeProjectId: 'sub_fail_1',
        sessionSubprojects: [{ id: 'sub_fail_1', name: 'v.mp4', media_id: 'm1', stage: 1, status: 'pending' } as any],
        projectJobStatuses: {},
      });
      await useDubDubStore.getState().startBatchPrepareJobs();
      await new Promise((r) => setTimeout(r, 15));
      expect(instances.length).toBeGreaterThanOrEqual(2);
      const batchEs = instances[instances.length - 1];
      batchEs.emitDone({});
      await new Promise((r) => setTimeout(r, 20));
      expect(useDubDubStore.getState().projectJobStatuses['sub_fail_1']?.status).toBe('failed');
      expect(useDubDubStore.getState().jobStatus).toBe('failed');
    } finally {
      useDubDubStore.getState().clearJob();
      globalThis.fetch = originalFetch;
      (globalThis as any).EventSource = originalEventSource;
    }
  });

  test('Iter2 Fix #3: cancelDubbing + immediate restart race, scoped abort, tuple parsing, customVoices filter', async () => {
    // Tuple parsing in loadVoices + customVoices filter in pickDefaultVoice
    expect(
      pickDefaultVoice([], [
        { id: 'No' },
        { id: 'clone' },
        { id: 'sys', kind: 'system' },
        { id: 'voice_real' },
      ] as any)
    ).toBe('voice_real');

    const originalFetch = globalThis.fetch;
    let callNum = 0;
    let resolveRun1!: (val: Response) => void;
    let resolveRun2!: (val: Response) => void;
    const signals: AbortSignal[] = [];

    globalThis.fetch = ((url: string | URL | Request, init?: RequestInit) => {
      const s = String(url);
      if (s.includes('/api/voices') && (!init?.method || init.method === 'GET')) {
        return Promise.resolve(jsonRes({ voices: [['v_tuple_1', 'Tuple Voice One'], ['clone', 'Clone']] }));
      }
      if (s.includes('/api/custom-voices')) {
        return Promise.resolve(jsonRes({ voices: [] }));
      }
      if (s.includes('/api/voices/preview') || s.includes('/api/tts/preview')) {
        callNum++;
        if (init?.signal) signals.push(init.signal);
        if (callNum === 1) {
          return new Promise<Response>((res) => {
            resolveRun1 = res;
          });
        }
        return new Promise<Response>((res) => {
          resolveRun2 = res;
        });
      }
      return Promise.resolve(jsonRes({ ok: true }));
    }) as any;

    try {
      await useDubDubStore.getState().loadVoices();
      expect(useDubDubStore.getState().voices[0]).toMatchObject({ id: 'v_tuple_1', name: 'Tuple Voice One' });

      // Empty/invalid voices guard: when voices are loaded with only 'No', runFullDubbing fails instead of sending 'default'
      useDubDubStore.setState({
        activeProjectId: 'proj_guard',
        voices: [{ id: 'No', name: 'No', provider: 2, kind: 'preset' }] as any,
        customVoices: [],
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' } as any],
      });
      await useDubDubStore.getState().runFullDubbing();
      expect(useDubDubStore.getState().dubbingStatus).toBe('failed');
      expect(useDubDubStore.getState().dubbingErrors[1]).toBe('No voice assigned for segment');

      // Cancel + immediate restart race & scoped abort
      useDubDubStore.setState({
        activeProjectId: 'proj_race',
        voices: [{ id: 'v_tuple_1', name: 'Tuple Voice One', provider: 2, kind: 'preset' }] as any,
        segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hi', targetText: 'Chào', speakerId: 'spk_1' } as any],
      });
      const run1 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));

      // Scoped cancel: cancelling while on another project must NOT abort proj_race
      useDubDubStore.setState({ activeProjectId: 'proj_other' });
      useDubDubStore.getState().cancelDubbing();
      expect(signals[0]?.aborted).toBe(false);

      // Now switch back to proj_race, cancel, and immediately start run2
      useDubDubStore.setState({ activeProjectId: 'proj_race' });
      useDubDubStore.getState().cancelDubbing();
      expect(signals[0]?.aborted).toBe(true);
      const run2 = useDubDubStore.getState().runFullDubbing();
      await new Promise((r) => setTimeout(r, 5));
      expect(useDubDubStore.getState().dubbingStatus).toBe('running');

      // Unblock run1 while run2 is still in-flight -> run1 must NOT overwrite dubbingStatus to 'idle'
      resolveRun1(jsonRes({ audio_url: '/audio/old.wav', id: 'old' }));
      await run1;
      expect(useDubDubStore.getState().dubbingStatus).toBe('running');

      // Finish run2
      resolveRun2(jsonRes({ audio_url: '/audio/new.wav', id: 'new' }));
      await run2;
      expect(useDubDubStore.getState().dubbingStatus).toBe('completed');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('Iter2 Fix #5: providerDocs openai+tts, CapCut whitespace rawTranslation, and invalid Date formatting', () => {
    expect(getProviderDocInfo('openai', 'tts')?.id).toBe('openaitts');
    expect(getProviderDocInfo('openai', 'llm')?.id).toBe('openai');

    const srt = generateSrtContent(
      [{ id: 1, startSec: 0, endSec: 2, sourceText: 'SRC', rawTranslation: '   ', targetText: 'EDITED' }],
      'rawTranslation'
    );
    expect(srt).toContain('EDITED');

    expect(formatLocalDateYmd(new Date('invalid'))).toBe('');
  });
});
