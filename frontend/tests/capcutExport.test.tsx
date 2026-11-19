import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { useDubDubStore } from '../src/store';
import { Stage4EditVideo } from '../src/screens/Stage4EditVideo';
import { VideoPlayer } from '../src/components/VideoPlayer';
import { StatusFooter } from '../src/components/StatusFooter';
import { Stage2ReviewTranscript, exportTranslatedSrt, untranslatedSegmentNumbers } from '../src/screens/Stage2ReviewTranscript';
import {
  CapCutExportModal,
  formatSrtTime,
  generateSrtContent,
} from '../src/components/CapCutExportModal';
import { buildRenderRequest } from '../src/store/editVideoSlice';

const mockSegment = {
  id: 1,
  startSec: 1.5,
  endSec: 4.8,
  startTime: '00:01.500',
  endTime: '00:04.800',
  sourceText: 'Hello world',
  targetText: 'Xin chào thế giới',
  speakerId: 'spk_1',
};

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })) as any;

  useDubDubStore.setState({
    currentStep: 4,
    maxUnlockedStep: 4,
    activeProjectId: 'proj_capcut_test',
    project: {
      id: 'proj_capcut_test',
      filename: 'sample_video.mp4',
      duration: '00:10',
      durationSec: 10,
      previewUrl: 'blob:video-preview-test',
      verified: true,
    },
    playback: {
      currentTime: 2.0,
      isPlaying: false,
      playbackSpeed: 1.0,
      formattedTime: '00:02',
      seekRequest: null,
    },
    segments: [mockSegment],
    subtitleStyles: {
      preset: 'clean',
      fontFamily: 'Inter',
      fontSize: 26,
      fontWeight: 'bold',
      fontStyle: 'italic',
      opacity: 85,
      color: '#FFFFFF',
      outlineColor: '#000000',
      outlineWidth: 2,
      shadowColor: 'rgba(0,0,0,.75)',
      shadowSize: 2,
    },
    editVideo: {
      audioMix: { original: 10, dubbed: 100, background: 30 },
      backgroundAudio: null,
      thumbnail: null,
      exporting: false,
      error: null,
      activeTab: 'subtitles',
      isCapCutModalOpen: false,
    },
  });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('Stage 4 Subtitle Customization & CapCut Export Suite', () => {
  describe('SRT Formatting Utilities', () => {
    test('formatSrtTime formats millisecond timestamps accurately', () => {
      expect(formatSrtTime(0)).toBe('00:00:00,000');
      expect(formatSrtTime(1.5)).toBe('00:00:01,500');
      expect(formatSrtTime(65.123)).toBe('00:01:05,123');
      expect(formatSrtTime(3661.045)).toBe('01:01:01,045');
    });

    test('generateSrtContent formats valid SRT blocks for edited and source text', () => {
      const srtEdited = generateSrtContent([mockSegment], 'targetText');
      expect(srtEdited).toContain('1\n00:00:01,500 --> 00:00:04,800\nXin chào thế giới');

      const srtSource = generateSrtContent([mockSegment], 'sourceText');
      expect(srtSource).toContain('1\n00:00:01,500 --> 00:00:04,800\nHello world');
    });
  });

  describe('Stage 2 translated SRT', () => {
    test('downloads UTF-8 translated SRT without a voice or network request', async () => {
      let blob: Blob | null = null;
      let filename = '';
      let requests = 0;
      const originalDocument = globalThis.document;
      const originalFetch = globalThis.fetch;
      const originalCreateObjectURL = URL.createObjectURL;
      const originalRevokeObjectURL = URL.revokeObjectURL;
      (globalThis as any).document = {
        createElement: () => ({ set href(_value: string) {}, set download(value: string) { filename = value; }, click: () => {} }),
        body: { appendChild: () => {}, removeChild: () => {} },
      };
      URL.createObjectURL = (value: Blob | MediaSource) => { blob = value as Blob; return 'blob:test-srt'; };
      URL.revokeObjectURL = () => {};
      globalThis.fetch = (async () => { requests++; throw new Error('Unexpected network request'); }) as typeof fetch;
      try {
        exportTranslatedSrt([mockSegment]);
        expect(filename).toBe('translated_subtitles.srt');
        expect(blob?.type).toContain('charset=utf-8');
        expect(await blob!.text()).toBe('1\n00:00:01,500 --> 00:00:04,800\nXin chào thế giới\n\n');
        expect(requests).toBe(0);
        expect(() => exportTranslatedSrt([{ ...mockSegment, targetText: '  ' }])).toThrow('Translate every segment');
      } finally {
        (globalThis as any).document = originalDocument;
        globalThis.fetch = originalFetch;
        URL.createObjectURL = originalCreateObjectURL;
        URL.revokeObjectURL = originalRevokeObjectURL;
      }
    });

    test('requires every target segment and uses target text with original timing', () => {
      expect(untranslatedSegmentNumbers([{ targetText: 'Xin chào' }, { targetText: '  ' }])).toEqual([2]);
      const srt = generateSrtContent([mockSegment], 'targetText');
      expect(srt).toContain('1\n00:00:01,500 --> 00:00:04,800\nXin chào thế giới');
      expect(srt).not.toContain('Hello world');
    });

    test('offers export in Stage 2 and identifies untranslated segments without voice controls', () => {
      useDubDubStore.setState({ currentStep: 2, segments: [{ ...mockSegment, targetText: '' }] });
      const incomplete = renderToStaticMarkup(<Stage2ReviewTranscript />);
      expect(incomplete).toContain('data-testid="export-translated-srt-btn"');
      expect(incomplete).toContain('disabled=""');
      expect(incomplete).toContain('Translate segment 1 before exporting the SRT');
      useDubDubStore.setState({ segments: [mockSegment] });
      const complete = renderToStaticMarkup(<Stage2ReviewTranscript />);
      expect(complete).toContain('Export Translated SRT');
      expect(complete).not.toContain('Translate segment 1 before exporting');
      expect(complete).not.toContain('Generate Dubbing');
    });
  });

  describe('Subtitle Styling State & Render Request', () => {
    test('subtitleStyles holds typography settings and mutates cleanly', () => {
      const store = useDubDubStore.getState();
      expect(store.subtitleStyles.fontFamily).toBe('Inter');
      expect(store.subtitleStyles.fontWeight).toBe('bold');
      expect(store.subtitleStyles.fontStyle).toBe('italic');
      expect(store.subtitleStyles.opacity).toBe(85);

      store.updateSubtitleStyle('fontFamily', 'Roboto');
      store.updateSubtitleStyle('opacity', 90);
      expect(useDubDubStore.getState().subtitleStyles.fontFamily).toBe('Roboto');
      expect(useDubDubStore.getState().subtitleStyles.opacity).toBe(90);
    });

    test('buildRenderRequest includes full typography options in payload', () => {
      const state = useDubDubStore.getState();
      const payload = buildRenderRequest(state);

      expect(payload.jobType).toBe('render');
      expect(payload.options.subtitleStyle.fontFamily).toBe('Inter');
      expect(payload.options.subtitleStyle.fontWeight).toBe('bold');
      expect(payload.options.subtitleStyle.fontStyle).toBe('italic');
      expect(payload.options.subtitleStyle.opacity).toBe(85);
    });
  });

  describe('Stage4EditVideo Inspector UI', () => {
    test('renders font family, bold/italic toggles, font size, and opacity slider', () => {
      const html = renderToStaticMarkup(<Stage4EditVideo />);

      expect(html).toContain('data-subtitle-font-family="true"');
      expect(html).toContain('data-subtitle-bold-toggle="true"');
      expect(html).toContain('data-subtitle-italic-toggle="true"');
      expect(html).toContain('data-subtitle-font-size="true"');
      expect(html).toContain('data-subtitle-opacity="true"');
      expect(html).toContain('Opacity / Transparency');
      expect(html).toContain('85%');
    });
  });

  describe('VideoPlayer Live Preview Sync', () => {
    test('applies font-weight, font-style, and opacity to canvas subtitle preview', () => {
      const html = renderToStaticMarkup(<VideoPlayer subtitleVariant="capcut" />);

      expect(html).toContain('font-weight:bold');
      expect(html).toContain('font-style:italic');
      expect(html).toContain('opacity:0.85');
      expect(html).toContain('font-family:Inter');
    });
  });

  describe('StatusFooter CapCut Bridge Button & Tooltip', () => {
    test('renders Continue in CapCut button in Stage 4 with exact tooltip', () => {
      useDubDubStore.setState({ currentStep: 4 });
      const html = renderToStaticMarkup(<StatusFooter />);

      expect(html).toContain('data-continue-in-capcut-btn="true"');
      expect(html).toContain('Continue in CapCut');
      expect(html).toContain(
        'Export merged audio and .srt subtitles, then import them with your original video into CapCut for advanced effects and transitions.'
      );
    });

    test('does NOT render Continue in CapCut button during Stage 1, 2, or 3', () => {
      for (const step of [1, 2, 3]) {
        useDubDubStore.setState({ currentStep: step });
        const html = renderToStaticMarkup(<StatusFooter />);
        expect(html).not.toContain('data-continue-in-capcut-btn="true"');
      }
    });
  });

  describe('CapCutExportModal Interactions & Content', () => {
    test('returns null when isCapCutModalOpen is false', () => {
      useDubDubStore.setState({
        editVideo: {
          ...useDubDubStore.getState().editVideo,
          isCapCutModalOpen: false,
        },
      });
      const html = renderToStaticMarkup(<CapCutExportModal />);
      expect(html).toBe('');
    });

    test('renders 3-step guide, three generated asset downloads, and launchers when open', () => {
      useDubDubStore.setState({
        editVideo: {
          ...useDubDubStore.getState().editVideo,
          isCapCutModalOpen: true,
        },
      });
      const html = renderToStaticMarkup(<CapCutExportModal />);

      expect(html).toContain('data-capcut-modal="true"');
      expect(html).toContain('Continue Editing in CapCut');
      expect(html).toContain('3-Step CapCut Workflow Guide');

      // 3 Asset download cards
      expect(html).toContain('data-download-edited-srt="true"');
      expect(html).toContain('subtitles_edited.srt');
      expect(html).toContain('data-download-target-srt="true"');
      expect(html).toContain('subtitles_target.srt');
      expect(html).not.toContain('data-download-video="true"');
      expect(html).toContain('Exported Project Assets (3 Files)');
      expect(html).toContain('data-download-audio="true"');
      expect(html).toContain('voiceover_merged.wav');

      // Zip & Launchers
      expect(html).toContain('data-download-zip-btn="true"');
      expect(html).toContain('Download All as ZIP');
      expect(html).toContain('data-open-folder-btn="true"');
      expect(html).toContain('Open Folder in Explorer');
      expect(html).toContain('data-launch-capcut-btn="true"');
      expect(html).toContain('Launch CapCut');
    });
  });
});
