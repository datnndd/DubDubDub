import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { useDubDubStore } from '../src/store';
import { Stage4EditVideo } from '../src/screens/Stage4EditVideo';
import { VideoPlayer } from '../src/components/VideoPlayer';
import { CapCutExportModal } from '../src/components/CapCutExportModal';
import { buildRenderRequest } from '../src/store/editVideoSlice';
import {
  computeRenderedVideoRect,
  generateAnnotationsAssContent,
} from '../src/types/annotation';

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  useDubDubStore.setState({
    currentStep: 4,
    maxUnlockedStep: 4,
    activeProjectId: 'proj_ann_test',
    backend: {
      ...(useDubDubStore.getState().backend || {}),
      mediaId: 'media_ann_1',
      status: 'ready',
      config: { translateType: 2, ttsType: 2 },
    } as any,
    languages: {
      source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
      target: { code: 'vi', name: 'Vietnamese' },
      timingMode: 'voice',
    },
    project: {
      ...(useDubDubStore.getState().project || {}),
      filename: 'tutorial.mp4',
      resolution: '1920x1080',
      fps: '30 fps',
      duration: '00:12',
      durationSec: 12,
      previewUrl: '/api/media/media_ann_1/file',
      verified: true,
    } as any,
    playback: {
      ...(useDubDubStore.getState().playback || {}),
      currentTime: 2.0,
      isPlaying: false,
      playbackSpeed: 1.0,
      formattedTime: '00:02',
      seekRequest: null,
    } as any,
    segments: [
      {
        id: 1,
        startSec: 0.5,
        endSec: 3.5,
        sourceText: '大家好',
        targetText: 'Xin chào mọi người',
        speakerId: 'spk_1',
      } as any,
    ],
    annotations: [],
    activeAnnotationId: null,
    annotationDrawMode: false,
    annotationScanning: false,
    annotationTranslating: false,
    annotationStatusMessage: null,
    editVideo: {
      audioMix: { original: 0, dubbed: 100, background: 30 },
      backgroundAudio: null,
      thumbnail: null,
      exporting: false,
      error: null,
      activeTab: 'annotations',
      isCapCutModalOpen: false,
      burnAnnotations: true,
    },
  });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Stage 4 On-Screen Annotations Suite', () => {
  test('computeRenderedVideoRect accounts for pillarbox and letterbox aspect ratios', () => {
    // 9:16 vertical video inside a 16:9 (1600x900) player container -> pillarboxed horizontally
    const vertical = computeRenderedVideoRect(1600, 900, 1080, 1920);
    expect(vertical.top).toBe(0);
    expect(vertical.height).toBe(900);
    expect(Math.round(vertical.width)).toBe(506);
    expect(Math.round(vertical.left)).toBe(547);

    // 21:9 ultrawide video inside a 16:9 (1600x900) player container -> letterboxed vertically
    const wide = computeRenderedVideoRect(1600, 900, 2100, 900);
    expect(wide.left).toBe(0);
    expect(wide.width).toBe(1600);
    expect(Math.round(wide.height)).toBe(686);
    expect(Math.round(wide.top)).toBe(107);
  });

  test('scanVideoAnnotations and translateAnnotations populate Stage 4 store without touching segments', async () => {
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/ocr/annotations/scan')) {
        return new Response(
          JSON.stringify({
            ok: true,
            filteredWatermarks: 1,
            annotations: [
              {
                id: 'ann_srv_1',
                startSec: 1.0,
                endSec: 4.0,
                box: [0.12, 0.18, 0.28, 0.08],
                sourceText: '超清屏幕',
                targetText: '',
                style: { bgColor: '#FFF5B4', bgOpacity: 92, textColor: '#111111', fontSize: 30, bold: true },
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }
      if (url.includes('/api/translate')) {
        const body = JSON.parse(String(init?.body || '{}'));
        return new Response(
          JSON.stringify({
            ok: true,
            segments: body.segments.map((s: any) => ({
              ...s,
              targetText: 'Màn hình siêu nét',
            })),
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }) as any;

    await useDubDubStore.getState().scanVideoAnnotations();
    expect(useDubDubStore.getState().annotations.length).toBe(1);
    expect(useDubDubStore.getState().annotations[0].sourceText).toBe('超清屏幕');

    await useDubDubStore.getState().translateAnnotations();
    const updated = useDubDubStore.getState().annotations[0];
    expect(updated.targetText).toBe('Màn hình siêu nét');
    // Main dialogue segment remains untouched
    expect(useDubDubStore.getState().segments[0].targetText).toBe('Xin chào mọi người');

    const renderReq = buildRenderRequest(useDubDubStore.getState());
    expect(renderReq.options.annotations?.length).toBe(1);
    expect(renderReq.options.burnAnnotations).toBe(true);
  });

  test('Stage4EditVideo, VideoPlayer, and CapCutExportModal render positioned annotations and 5-file bundle', () => {
    useDubDubStore.getState().addManualAnnotation(1.5);
    const annId = useDubDubStore.getState().annotations[0].id;
    useDubDubStore.getState().updateAnnotation(annId, {
      sourceText: '防水等级 IP68',
      targetText: 'Chuẩn chống nước IP68',
    });

    const stage4Html = renderToStaticMarkup(<Stage4EditVideo />);
    expect(stage4Html).toContain('data-testid="stage4-tab-annotations"');
    expect(stage4Html).toContain('data-testid="scan-annotations-btn"');
    expect(stage4Html).toContain('data-testid="translate-annotations-btn"');
    expect(stage4Html).toContain('data-testid="timeline-annotations-track"');
    expect(stage4Html).toContain('Chuẩn chống nước IP68');

    const playerHtml = renderToStaticMarkup(<VideoPlayer subtitleVariant="capcut" />);
    expect(playerHtml).toContain(`data-canvas-annotation="${annId}"`);
    expect(playerHtml).toContain('Chuẩn chống nước IP68');

    useDubDubStore.getState().setCapCutModalOpen(true);
    const modalHtml = renderToStaticMarkup(<CapCutExportModal />);
    expect(modalHtml).toContain('Exported Project Assets (5 Files)');
    expect(modalHtml).toContain('data-download-annotations-srt="true"');
    expect(modalHtml).toContain('data-download-annotations-ass="true"');

    const assContent = generateAnnotationsAssContent(useDubDubStore.getState().annotations, 1920, 1080);
    expect(assContent).toContain('PlayResX: 1920');
    expect(assContent).toContain('Chuẩn chống nước IP68');
    expect(assContent).toContain('\\an5\\pos(');
  });
});
