import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { Stage1Prepare } from '../src/screens/Stage1Prepare';
import { StatusFooter } from '../src/components/StatusFooter';
import { useDubDubStore } from '../src/store';
import { buildPrepareJobRequest } from '../src/store/jobSlice';

const initialSnapshot = useDubDubStore.getInitialState();
const originalFetch = globalThis.fetch;

function jsonRes(body: any, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

beforeEach(() => {
  globalThis.fetch = (async () => jsonRes({ ok: true })) as any;
  useDubDubStore.setState(
    {
      ...initialSnapshot,
      activeProjectId: 'proj_vocr_test',
      currentStep: 1,
      subtitleSource: 'audio_asr',
      ocrRoiConfirmed: false,
      ocrOptions: { coarseIntervalMs: 500, useAngleCls: false, paragraphMode: 'preserve_lines' },
      stage1OcrTest: { loading: false, error: null, text: null },
      ocrCrop: { active: false, segmentId: null, roi: [0.05, 0.75, 0.9, 0.2], loading: false, error: null },
      jobStatus: 'idle',
      jobProgress: null,
      jobStage: null,
      jobMessage: 'Ready to process',
      sessionSubprojects: [],
      projectJobStatuses: {},
      languages: {
        source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
        target: { code: 'vi', name: 'Vietnamese' },
        timingMode: 'voice',
      },
      project: {
        ...initialSnapshot.project,
        filename: 'silent_slides.mp4',
        previewUrl: '/api/media/m_vocr_1/file',
        verified: true,
        hasVideo: true,
        hasAudio: false,
        durationSec: 10,
      },
      backend: {
        ...initialSnapshot.backend,
        ready: true,
        mediaId: 'm_vocr_1',
        status: 'ready',
        options: {
          ...initialSnapshot.backend.options,
          languages: [
            { code: 'zh-cn', name: 'Simplified Chinese' },
            { code: 'vi', name: 'Vietnamese' },
          ],
          asrProviders: [
            {
              id: 'deepgram',
              label: 'Deepgram',
              recognType: 1,
              requiresSettings: true,
              configured: false,
              models: ['nova-3'],
            },
          ],
        },
        config: {
          ...initialSnapshot.backend.config,
          recognType: 1,
          modelName: 'nova-3',
        },
      },
    },
    true
  );
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Stage 1 Video OCR Integration', () => {
  test('renders Subtitle Source selector, keeps ASR Engine in audio_asr mode, and switches to ROI/OCR controls in video_ocr mode', () => {
    const asrHtml = renderToStaticMarkup(<Stage1Prepare />);
    expect(asrHtml).toContain('data-testid="subtitle-source-selector"');
    expect(asrHtml).toContain('ASR Engine');
    expect(asrHtml).toContain('Deepgram Parameters');
    expect(asrHtml).not.toContain('data-testid="ocr-roi-overlay"');
    expect(asrHtml).not.toContain('data-testid="test-frame-ocr-btn"');

    useDubDubStore.getState().setSubtitleSource('video_ocr');
    const ocrHtml = renderToStaticMarkup(<Stage1Prepare />);
    expect(ocrHtml).toContain('data-testid="subtitle-source-selector"');
    expect(ocrHtml).toContain('Hard-Subtitle Video OCR');
    expect(ocrHtml).toContain('data-testid="ocr-roi-overlay"');
    expect(ocrHtml).toContain('data-testid="ocr-roi-drag-layer"');
    expect(ocrHtml).toContain('data-testid="test-frame-ocr-btn"');
    expect(ocrHtml).toContain('ROI Unconfirmed');
    expect(ocrHtml).toContain('OCR ROI Left percent');
    expect(ocrHtml).toContain('OCR scan interval ms');
    expect(ocrHtml).toContain('OCR paragraph mode');
    expect(ocrHtml).not.toContain('Deepgram Parameters');
    expect(ocrHtml).toContain('Audio diarization applies to ASR mode only');
  });

  test('setOcrCropRoi resets ocrRoiConfirmed back to false and clears prior test result', () => {
    const store = useDubDubStore.getState();
    store.setSubtitleSource('video_ocr');
    useDubDubStore.setState({
      ocrRoiConfirmed: true,
      stage1OcrTest: { loading: false, error: null, text: 'Previous line', confidence: 0.95 },
    });

    store.setOcrCropRoi([0.1, 0.72, 0.8, 0.2]);
    const next = useDubDubStore.getState();
    expect(next.ocrCrop.roi).toEqual([0.1, 0.72, 0.8, 0.2]);
    expect(next.ocrRoiConfirmed).toBe(false);
    expect(next.stage1OcrTest.text).toBeNull();
  });

  test('testStage1OcrFrame clamps 1s window, confirms ROI on non-empty text, and handles errors/empty frames', async () => {
    const extractPayloads: any[] = [];
    let nextResponse: { status: number; body: any } = {
      status: 200,
      body: { ok: true, text: 'Line one\nLine two', confidence: 0.94, entries: [] },
    };

    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      if (String(url) === '/api/ocr/extract') {
        extractPayloads.push(JSON.parse(String(init?.body || '{}')));
        return jsonRes(nextResponse.body, nextResponse.status);
      }
      return jsonRes({ ok: true });
    }) as any;

    useDubDubStore.getState().setSubtitleSource('video_ocr');
    // At t=0 on a 10s video: startSec >= 0 and endSec > startSec
    const resAtStart = await useDubDubStore.getState().testStage1OcrFrame(0);
    expect(resAtStart.ok).toBe(true);
    expect(extractPayloads[0]).toEqual({
      mediaId: 'm_vocr_1',
      roi: [0.05, 0.75, 0.9, 0.2],
      startSec: 0,
      endSec: 1,
      language: 'zh-cn',
    });
    expect(useDubDubStore.getState().ocrRoiConfirmed).toBe(true);
    expect(useDubDubStore.getState().stage1OcrTest.text).toBe('Line one\nLine two');

    // Near end t=9.9 on a 10s video: clamped within [0, 10]
    await useDubDubStore.getState().testStage1OcrFrame(9.9);
    expect(extractPayloads[1].startSec).toBe(9);
    expect(extractPayloads[1].endSec).toBe(10);

    // Empty frame text keeps ocrRoiConfirmed: false and surfaces guidance
    nextResponse = { status: 200, body: { ok: true, text: '   ', confidence: 0 } };
    await useDubDubStore.getState().testStage1OcrFrame(4);
    expect(useDubDubStore.getState().ocrRoiConfirmed).toBe(false);
    expect(useDubDubStore.getState().stage1OcrTest.error).toContain('No subtitle text detected');

    // HTTP error keeps ocrRoiConfirmed: false and surfaces error
    nextResponse = { status: 400, body: { detail: 'Invalid ROI bounds' } };
    const errRes = await useDubDubStore.getState().testStage1OcrFrame(4);
    expect(errRes.ok).toBe(false);
    expect(useDubDubStore.getState().ocrRoiConfirmed).toBe(false);
    expect(useDubDubStore.getState().stage1OcrTest.error).toContain('Invalid ROI bounds');
  });

  test('StatusFooter gates Start Dub on ocrRoiConfirmed in video_ocr mode and updates busy label', () => {
    useDubDubStore.setState({ subtitleSource: 'video_ocr', ocrRoiConfirmed: false });
    const unconfirmedFooter = renderToStaticMarkup(<StatusFooter />);
    expect(unconfirmedFooter).toContain('Start Dub');
    expect(unconfirmedFooter).toContain('disabled=""');

    useDubDubStore.setState({ ocrRoiConfirmed: true });
    const confirmedFooter = renderToStaticMarkup(<StatusFooter />);
    expect(confirmedFooter).toContain('Start Dub');
    expect(confirmedFooter).not.toContain('title="Run Test Frame OCR to confirm subtitle ROI"');

    useDubDubStore.setState({ jobStatus: 'running', jobStage: 'prepare' });
    const busyFooter = renderToStaticMarkup(<StatusFooter />);
    expect(busyFooter).toContain('Cancel Processing');

    useDubDubStore.setState({
      jobStatus: 'idle',
      backend: { ...useDubDubStore.getState().backend, status: 'analyzing' },
    });
    const analyzingFooter = renderToStaticMarkup(<StatusFooter />);
    expect(analyzingFooter).toContain('Running Video OCR…');
  });

  test('startBatchPrepareJobs blocks unconfirmed video_ocr and forwards OCR options when confirmed', async () => {
    const jobCalls: any[] = [];
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      if (String(url) === '/api/jobs' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body || '{}'));
        jobCalls.push(body);
        return jsonRes({
          id: `job_${body.projectId}`,
          status: 'succeeded',
          segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Batch OCR', targetText: '', sourceType: 'ocr' }],
        });
      }
      return jsonRes({ ok: true });
    }) as any;

    useDubDubStore.setState({
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: false,
      ocrOptions: { coarseIntervalMs: 250, useAngleCls: true, paragraphMode: 'unwrap' },
      sessionSubprojects: [
        { id: 'sub_1', name: 'ep1.mp4', media_id: 'm_1', stage: 1, status: 'pending' } as any,
        { id: 'sub_2', name: 'ep2.mp4', media_id: 'm_2', stage: 1, status: 'pending' } as any,
      ],
    });

    await useDubDubStore.getState().startBatchPrepareJobs();
    expect(jobCalls).toHaveLength(0);
    expect(useDubDubStore.getState().jobStatus).toBe('failed');

    useDubDubStore.getState().setOcrRoiConfirmed(true);
    await useDubDubStore.getState().startBatchPrepareJobs();
    expect(jobCalls).toHaveLength(2);
    expect(jobCalls[0].options.subtitleSource).toBe('video_ocr');
    expect(jobCalls[0].options.ocrRoiConfirmed).toBe(true);
    expect(jobCalls[0].options.ocrOptions).toEqual({
      coarseIntervalMs: 250,
      useAngleCls: true,
      paragraphMode: 'unwrap',
    });

    const singleReq = buildPrepareJobRequest(useDubDubStore.getState());
    expect(singleReq.options.ocrOptions.coarseIntervalMs).toBe(250);
    expect(singleReq.options.ocrOptions.useAngleCls).toBe(true);
    expect(singleReq.options.ocrOptions.paragraphMode).toBe('unwrap');
  });

  test('M2 regression: initializeBackend(true) refreshes defaults and updateTtsProvider updates ttsType', async () => {
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

    useDubDubStore.getState().updateTtsProvider(0);
    expect(useDubDubStore.getState().backend.config.ttsType).toBe(0);
  });
});
