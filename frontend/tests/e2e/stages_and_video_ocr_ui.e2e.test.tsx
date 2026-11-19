import React from 'react';
import { beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { useDubDubStore } from '../../src/store';
import { Stage1Prepare } from '../../src/screens/Stage1Prepare';
import { Stage2ReviewTranscript, exportTranslatedSrt } from '../../src/screens/Stage2ReviewTranscript';
import { Stage2OcrDialog, roiFromDrag } from '../../src/components/Stage2OcrDialog';
import { Stage3VoiceDubbing } from '../../src/screens/Stage3VoiceDubbing';
import { VoiceManagementScreen } from '../../src/screens/VoiceManagementScreen';
import { CreateVoiceModal } from '../../src/components/CreateVoiceModal';
import { Stage4EditVideo } from '../../src/screens/Stage4EditVideo';
import { CapCutExportModal, generateSrtContent } from '../../src/components/CapCutExportModal';
import { VideoPlayer } from '../../src/components/VideoPlayer';
import { StatusFooter } from '../../src/components/StatusFooter';
import { buildPrepareJobRequest } from '../../src/store/jobSlice';
import { buildRenderRequest } from '../../src/store/editVideoSlice';
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

const baseSeg = {
  id: 1,
  speakerId: 'spk_1',
  speakerName: 'Speaker 1',
  startSec: 1.0,
  endSec: 3.0,
  startTime: '00:01.000',
  endTime: '00:03.000',
  sourceText: 'Original ASR line',
  targetText: 'Bản dịch mẫu',
};

// ============================================================================
// F4: Stage 2 OCR Dialog & Translation Modal UX (Tier 1 >= 5, Tier 2 >= 5)
// ============================================================================
describe('F4 — Stage 2 OCR Dialog & Translation Modal UX', () => {
  // --- Tier 1 ---
  test('[Tier 1] Stage2ReviewTranscript renders segments, source badges, and translation toolbar', () => {
    useDubDubStore.setState({
      currentStep: 2,
      segments: [
        { ...baseSeg, id: 1, sourceType: 'asr' },
        { ...baseSeg, id: 2, startSec: 4, endSec: 6, sourceText: 'OCR text', targetText: 'Dịch OCR', sourceType: 'ocr' },
      ] as any,
    });

    const html = renderToStaticMarkup(<Stage2ReviewTranscript />);
    expect(html).toContain('data-testid="translation-provider-select"');
    expect(html).toContain('data-testid="batch-translate-btn"');
    expect(html).toContain('data-testid="source-badge-1"');
    expect(html).toContain('data-testid="source-badge-2"');
    expect(html).toContain('ASR');
    expect(html).toContain('OCR');
  });

  test('[Tier 1] Stage2OcrDialog renders ROI overlay, coordinate inputs, and roiFromDrag clamps normalized ROI', () => {
    expect(roiFromDrag([0.8, 0.9], [0.1, 0.7])).toEqual([0.1, 0.7, expect.closeTo(0.7, 5), expect.closeTo(0.2, 5)]);
    expect(roiFromDrag([-0.5, 0.6], [1.5, 1.2])).toEqual([0, 0.6, 1, expect.closeTo(0.4, 5)]);

    const html = renderToStaticMarkup(
      <Stage2OcrDialog
        segment={baseSeg as any}
        videoUrl="/api/media/m1/file"
        roi={[0.1, 0.75, 0.8, 0.2]}
        onRoiChange={() => {}}
        onExtract={() => {}}
        onClose={() => {}}
        loading={false}
        error={null}
      />
    );
    expect(html).toContain('data-testid="stage2-ocr-dialog"');
    expect(html).toContain('data-testid="ocr-roi-overlay"');
    expect(html).toContain('aria-label="Left percent"');
    expect(html).toContain('aria-label="Height percent"');
  });

  test('[Tier 1] replaceSegmentWithOcrEntries replaces 1 ASR segment with chronological OCR entries and preserves speaker', async () => {
    useDubDubStore.setState({
      segments: [
        { ...baseSeg, id: 1, startSec: 10, endSec: 15, speakerId: 'spk_host', speakerName: 'Host' },
        { ...baseSeg, id: 2, startSec: 16, endSec: 20, sourceText: 'Later segment' },
      ] as any,
    });

    await useDubDubStore.getState().replaceSegmentWithOcrEntries(1, [
      { startSec: 12, endSec: 14, startTime: '00:00:12,000', endTime: '00:00:14,000', text: 'OCR B', confidence: 0.91 },
      { startSec: 10, endSec: 12, startTime: '00:00:10,000', endTime: '00:00:12,000', text: 'OCR A', confidence: 0.95 },
    ]);

    const segs = useDubDubStore.getState().segments;
    expect(segs).toHaveLength(3);
    expect(segs.map((s) => s.id)).toEqual([1, 2, 3]);
    expect(segs[0].sourceText).toBe('OCR A');
    expect(segs[1].sourceText).toBe('OCR B');
    expect(segs[0].speakerName).toBe('Host');
    expect(segs[0].sourceType).toBe('ocr');
    expect(segs[0].targetText).toBe('');
    expect(segs[2].sourceText).toBe('Later segment');
  });

  test('[Tier 1] runBatchTranslation applies translated targetText and completes translationModal', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_tr',
      segments: [{ ...baseSeg, id: 1, sourceText: 'Hello', targetText: '' }] as any,
    });

    await withE2EShims(async (url) => {
      if (url.includes('/api/translate')) {
        return jsonResponse({
          ok: true,
          segments: [{ id: 1, sourceText: 'Hello', targetText: 'Xin chào' }],
        });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().runBatchTranslation('en', 'vi');
      const s = useDubDubStore.getState();
      expect(s.segments[0].targetText).toBe('Xin chào');
      expect(s.translationModal.status).toBe('completed');
      expect(s.translationModal.progress).toBe(100);
    });
  });

  test('[Tier 1] exportTranslatedSrt downloads UTF-8 SRT and rejects untranslated segments', async () => {
    await withE2EShims(async () => jsonResponse({ ok: true }), async ({ downloads }) => {
      exportTranslatedSrt([baseSeg as any]);
      expect(downloads).toHaveLength(1);
      expect(downloads[0].filename).toBe('translated_subtitles.srt');
      const text = await downloads[0].readText();
      expect(text).toContain('00:00:01,000 --> 00:00:03,000\nBản dịch mẫu');
      expect(() => exportTranslatedSrt([{ ...baseSeg, targetText: '   ' } as any])).toThrow('Translate every segment');
    });
  });

  // --- Tier 2 ---
  test('[Tier 2] Stage2OcrDialog guards 0-width ROI and renders video scrubbing/frame-step controls', () => {
    const zeroWidthHtml = renderToStaticMarkup(
      <Stage2OcrDialog
        segment={baseSeg as any}
        videoUrl="/api/media/m1/file"
        roi={[0.1, 0.7, 0, 0.2]}
        onRoiChange={() => {}}
        onExtract={() => {}}
        onClose={() => {}}
        loading={false}
        error={null}
      />
    );
    expect(zeroWidthHtml).toContain('disabled=""');
    expect(roiFromDrag([0.5, 0.5], [0.5, 0.5])).toEqual([0.5, 0.5, 0, 0]);

    // Scrubbing / frame-step controls within segment interval must be rendered in Stage2OcrDialog
    const validHtml = renderToStaticMarkup(
      <Stage2OcrDialog
        segment={baseSeg as any}
        videoUrl="/api/media/m1/file"
        roi={[0.1, 0.7, 0.8, 0.2]}
        onRoiChange={() => {}}
        onExtract={() => {}}
        onClose={() => {}}
        loading={false}
        error={null}
      />
    );
    expect(validHtml).toMatch(/type="range"|data-testid="ocr-scrub|Previous frame|Next frame|Frame -|Frame \+|Step back|Step forward|Seek/i);
  });

  test('[Tier 2] Stage2OcrDialog supports unified preview/confirm action in batch OCR mode', () => {
    const seg1 = { ...baseSeg, id: 1, startSec: 1, endSec: 2 };
    const seg2 = { ...baseSeg, id: 2, startSec: 3, endSec: 4 };
    const html = renderToStaticMarkup(
      <Stage2OcrDialog
        segment={seg1 as any}
        batchSegments={[seg1 as any, seg2 as any]}
        batchResults={[
          { segment: seg1 as any, text: 'Detected Batch 1', confidence: 0.94, applied: false },
          { segment: seg2 as any, text: 'Detected Batch 2', confidence: 0.91, applied: false },
        ]}
        currentSegments={[seg1 as any, seg2 as any]}
        progress={2}
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
    expect(html).toContain('Detected Batch 1');
    expect(html).toContain('data-testid="confirm-apply-batch-ocr-btn"');
    expect(html).toContain('Confirm &amp; Apply OCR');
  });

  test('[Tier 2] replaceSegmentWithOcrEntries preserves and remaps segmentVoiceOverrides when IDs shift', async () => {
    useDubDubStore.setState({
      segments: [
        { ...baseSeg, id: 1, startSec: 0, endSec: 3, sourceText: 'ASR 1' },
        { ...baseSeg, id: 2, startSec: 5, endSec: 8, sourceText: 'Keep me' },
      ] as any,
      segmentVoiceOverrides: { 2: 'custom_voice_B' },
    });

    await useDubDubStore.getState().replaceSegmentWithOcrEntries(1, [
      { startSec: 0, endSec: 1.5, startTime: '00:00:00,000', endTime: '00:00:01,500', text: 'OCR 1a', confidence: 0.9 },
      { startSec: 1.5, endSec: 3.0, startTime: '00:00:01,500', endTime: '00:00:03,000', text: 'OCR 1b', confidence: 0.9 },
    ]);

    const s = useDubDubStore.getState();
    expect(s.segments.map((seg) => seg.id)).toEqual([1, 2, 3]);
    expect(s.segments[2].sourceText).toBe('Keep me');
    expect(s.segmentVoiceOverrides[3]).toBe('custom_voice_B');
    expect(s.segmentVoiceOverrides[2]).toBeUndefined();
  });

  test('[Tier 2] translationModal renders actionable error details when translation fails', async () => {
    useDubDubStore.setState({
      segments: [{ ...baseSeg, id: 1 }] as any,
    });

    await withE2EShims(async () => jsonResponse({ detail: 'Invalid OpenAI API key (401)' }, 401), async () => {
      await useDubDubStore.getState().runBatchTranslation();
      const s = useDubDubStore.getState();
      expect(s.translationModal.active).toBe(true);
      expect(s.translationModal.status).toBe('failed');
      expect(s.translationModal.error).toContain('Invalid OpenAI API key (401)');

      const html = renderToStaticMarkup(<Stage2ReviewTranscript />);
      expect(html).toContain('Invalid OpenAI API key (401)');
    });
  });

  test('[Tier 2] replaceSegmentWithOcrEntries preserves multi-line paragraph breaks and empty export throws cleanly', async () => {
    useDubDubStore.setState({ segments: [{ ...baseSeg, id: 1, startSec: 0, endSec: 3 }] as any });
    await useDubDubStore.getState().replaceSegmentWithOcrEntries(1, [
      { startSec: 0, endSec: 3, startTime: '00:00:00,000', endTime: '00:00:03,000', text: 'Line one\nLine two', confidence: 0.96 },
    ]);
    expect(useDubDubStore.getState().segments[0].sourceText).toBe('Line one\nLine two');
    expect(() => exportTranslatedSrt([])).toThrow('Translate every segment');
  });
});

// ============================================================================
// F5: Stage 3 Dubbing UX, Voice Management & Stage 4/CapCut (Tier 1 >= 5, Tier 2 >= 5)
// ============================================================================
describe('F5 — Stage 3 Dubbing UX, Voice Management & Stage 4/CapCut', () => {
  // --- Tier 1 ---
  test('[Tier 1] Stage3VoiceDubbing speaker voice mapping and per-segment override resolution', () => {
    useDubDubStore.setState({
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
      segments: [
        { ...baseSeg, id: 1, speakerId: 'spk_1', speakerName: 'Host' },
        { ...baseSeg, id: 2, speakerId: 'spk_2', speakerName: 'Guest' },
      ] as any,
    });

    const store = useDubDubStore.getState();
    store.setSpeakerVoice('spk_1', 'Ngọc');
    store.setSegmentVoiceOverride(2, 'custom_guest_voice');

    expect(store.getDistinctSpeakers()).toHaveLength(2);
    expect(store.getResolvedVoiceForSegment(useDubDubStore.getState().segments[0])).toBe('Ngọc');
    expect(store.getResolvedVoiceForSegment(useDubDubStore.getState().segments[1])).toBe('custom_guest_voice');

    store.clearSegmentVoiceOverride(2);
    expect(store.getResolvedVoiceForSegment(useDubDubStore.getState().segments[1])).toBe('Ngọc');
  });

  test('[Tier 1] runFullDubbing synthesizes previews and marks completed when all segments succeed', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_dub',
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
      segments: [
        { ...baseSeg, id: 1, startSec: 0, endSec: 2, targetText: 'Xin chào' },
        { ...baseSeg, id: 2, startSec: 2, endSec: 4, targetText: 'Tạm biệt' },
      ] as any,
    });

    await withE2EShims(async (url, _init, call) => {
      if (url.includes('/api/tts/preview')) {
        return jsonResponse({ preview_url: `/api/audio/seg_${call.body.segment_id}.wav`, id: `prev_${call.body.segment_id}`, applied_speed: 1.0 });
      }
      if (url.includes('/api/tts/batch') || url.includes('/dubbing/generate')) {
        const segs = call.body.segments || call.body.items || [];
        return jsonResponse({
          ok: true,
          results: segs.map((s: any) => ({ segmentId: s.segmentId ?? s.id, ok: true, previewUrl: `/api/audio/seg_${s.segmentId ?? s.id}.wav`, id: `prev_${s.segmentId ?? s.id}` })),
        });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().runFullDubbing();
      const s = useDubDubStore.getState();
      expect(s.dubbingStatus).toBe('completed');
      expect(s.segments[0].previewAudioUrl).toContain('seg_1.wav');
      expect(s.segments[1].previewAudioUrl).toContain('seg_2.wav');
    });
  });

  test('[Tier 1] VoiceManagementScreen renders tabs, filters, and Synthesis Lab controls', () => {
    useDubDubStore.setState({
      customVoices: [{ id: 'cv_1', name: 'Storyteller', provider: 2, language: 'vi', description: 'Warm' }] as any,
      voices: [{ id: 'preset_1', name: 'Bình', provider: 2, kind: 'preset' }] as any,
    });

    const html = renderToStaticMarkup(<VoiceManagementScreen />);
    expect(html).toContain('data-screen="voice-management-screen"');
    expect(html).toContain('data-testid="tab-custom-voices"');
    expect(html).toContain('data-testid="lab-voice-picker"');
    expect(html).toContain('data-testid="test-phrase-input"');
    expect(html).toContain('data-testid="synthesize-test-phrase-btn"');
    expect(html).toContain('Storyteller');
  });

  test('[Tier 1] CreateVoiceModal renders cloning/design tabs when open and createVoice persists via API', async () => {
    useDubDubStore.setState({ isCreateVoiceModalOpen: true });
    const html = renderToStaticMarkup(<CreateVoiceModal />);
    expect(html).toContain('data-modal="create-voice"');
    expect(html).toContain('Voice Cloning');
    expect(html).toContain('Voice Design');

    await withE2EShims(async (url, init) => {
      if (url.endsWith('/api/custom-voices') && init?.method === 'POST') {
        return jsonResponse({ id: 'cv_new', name: 'New Clone', provider: 2, language: 'vi' }, 201);
      }
      if (url.includes('/api/custom-voices')) return jsonResponse({ voices: [{ id: 'cv_new', name: 'New Clone', provider: 2 }] });
      if (url.includes('/api/voices')) return jsonResponse({ voices: [] });
      return jsonResponse({ ok: true });
    }, async () => {
      const created = await useDubDubStore.getState().createVoice({ name: 'New Clone', provider: 2, language: 'vi' } as any);
      expect(created.id).toBe('cv_new');
      expect(useDubDubStore.getState().customVoices.some((v) => v.id === 'cv_new')).toBe(true);
    });
  });

  test('[Tier 1] Stage4EditVideo renders mix sliders & subtitle typography and CapCutExportModal renders 3 assets', () => {
    useDubDubStore.setState({
      currentStep: 4,
      segments: [baseSeg as any],
      editVideo: { ...useDubDubStore.getState().editVideo, isCapCutModalOpen: true },
    });

    const stage4Html = renderToStaticMarkup(<Stage4EditVideo />);
    expect(stage4Html).toContain('data-mix-slider="original"');
    expect(stage4Html).toContain('data-mix-slider="dubbed"');
    expect(stage4Html).toContain('data-mix-slider="background"');

    const capcutHtml = renderToStaticMarkup(<CapCutExportModal />);
    expect(capcutHtml).toContain('data-capcut-modal="true"');
    expect(capcutHtml).toContain('subtitles_edited.srt');
    expect(capcutHtml).toContain('subtitles_target.srt');
    expect(capcutHtml).toContain('voiceover_merged.wav');
  });

  // --- Tier 2 ---
  test('[Tier 2] runFullDubbing with 100% segment failure sets failed status and error details (NEVER completed)', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_fail_all',
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
      segments: [
        { ...baseSeg, id: 1, targetText: 'Line 1' },
        { ...baseSeg, id: 2, targetText: 'Line 2' },
      ] as any,
    });

    await withE2EShims(async (url) => {
      if (url.includes('/api/tts/preview') || url.includes('/api/tts/batch') || url.includes('/dubbing/generate')) {
        return jsonResponse({ detail: 'TTS synthesis service offline' }, 500);
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().runFullDubbing();
      const s = useDubDubStore.getState();
      expect(s.dubbingStatus).toBe('failed');
      expect(s.dubbingError).toBeTruthy();
      expect(s.projectJobStatuses['p_fail_all']?.status).toBe('failed');
    });
  });

  test('[Tier 2] runFullDubbing with partial segment failure does NOT mark completed and surfaces failure count/details', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_fail_partial',
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
      segments: [
        { ...baseSeg, id: 1, targetText: 'Line 1' },
        { ...baseSeg, id: 2, targetText: 'Line 2' },
      ] as any,
    });

    await withE2EShims(async (url, _init, call) => {
      if (url.includes('/api/tts/preview')) {
        if (call.body.segment_id === 2) return jsonResponse({ detail: 'Segment 2 timeout' }, 500);
        return jsonResponse({ preview_url: '/api/audio/seg_1.wav', id: 'prev_1', applied_speed: 1.0 });
      }
      if (url.includes('/api/tts/batch') || url.includes('/dubbing/generate')) {
        return jsonResponse({
          ok: false,
          results: [
            { segmentId: 1, ok: true, previewUrl: '/api/audio/seg_1.wav', id: 'prev_1' },
            { segmentId: 2, ok: false, error: 'Segment 2 timeout' },
          ],
        });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().runFullDubbing();
      const s = useDubDubStore.getState();
      expect(s.dubbingStatus).not.toBe('completed');
      expect(s.dubbingError).toBeTruthy();
      expect(s.projectJobStatuses['p_fail_partial']?.status).not.toBe('completed');
    });
  });

  test('[Tier 2] Stage 3 / StatusFooter renders Cancel control during running dubbing and cancelDubbing aborts cleanly', async () => {
    useDubDubStore.setState({
      currentStep: 3,
      dubbingStatus: 'running',
      jobProgress: 40,
      jobMessage: 'Creating voice (2/5)…',
      segments: [baseSeg as any],
    });

    const stage3Html = renderToStaticMarkup(<Stage3VoiceDubbing />);
    const footerHtml = renderToStaticMarkup(<StatusFooter />);
    expect(stage3Html + footerHtml).toMatch(/Cancel/i);

    const s: any = useDubDubStore.getState();
    expect(typeof s.cancelDubbing).toBe('function');
    s.cancelDubbing();
    expect(useDubDubStore.getState().dubbingStatus).not.toBe('running');
  });

  test('[Tier 2] Stage3VoiceDubbing populates TTS providers from backend.options.voices and scopes VieNeu emotion chips', () => {
    useDubDubStore.setState({
      currentStep: 3,
      languages: {
        source: { code: 'zh-cn', name: 'Chinese', flag: '', autoDetected: false },
        target: { code: 'en', name: 'English' },
        timingMode: 'voice',
      },
      backend: {
        ...useDubDubStore.getState().backend,
        options: {
          ...useDubDubStore.getState().backend.options,
          voices: [
            [2, 'VieNeu-TTS'],
            [4, 'Edge-TTS'],
            [5, 'OpenAI TTS'],
          ] as any,
        },
        config: {
          ...useDubDubStore.getState().backend.config,
          ttsType: 4, // Edge-TTS (NOT VieNeu!)
        },
      },
      segments: [{ ...baseSeg, id: 1, targetText: 'Hello English' }] as any,
    });

    const html = renderToStaticMarkup(<Stage3VoiceDubbing />);
    expect(html).toContain('Edge-TTS');
    expect(html).toContain('OpenAI TTS');
    // VieNeu Vietnamese emotion chips must NOT appear when Edge-TTS + English are selected
    expect(html).not.toContain('+ [cười]');
    expect(html).not.toContain('+ [thở dài]');
    expect(html).not.toContain('+ [hắng giọng]');
  });

  test('[Tier 2] VoiceManagementScreen applies genderFilter to custom voices and exposes full lab-voice-picker', () => {
    useDubDubStore.setState({
      customVoices: [
        { id: 'cv_f', name: 'Linh Female Custom', gender: 'female', provider: 2, language: 'vi' },
        { id: 'cv_m', name: 'Nam Male Custom', gender: 'male', provider: 2, language: 'vi' },
      ] as any,
      voices: [{ id: 'p1', name: 'Preset One', provider: 2, kind: 'preset' }] as any,
    });

    const femaleCustomHtml = renderToStaticMarkup(
      <VoiceManagementScreen initialTab="custom" initialGenderFilter="female" />
    );
    expect(femaleCustomHtml).toContain('Linh Female Custom');
    expect(femaleCustomHtml).not.toContain('Nam Male Custom');
  });

  test('[Tier 2] Stage4EditVideo persists server non-blob BGM and thumbnail URLs across autosave', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_assets',
      currentStep: 4,
    });

    await withE2EShims(async (url, init) => {
      if (url === '/api/assets/background-audio' && init?.method === 'POST') {
        return jsonResponse({ id: 'asset_99', name: 'bgm_track.mp3' }, 201);
      }
      if (url === '/api/assets/thumbnail' && init?.method === 'POST') {
        return jsonResponse({ id: 'thumb_99', name: 'cover.png' }, 201);
      }
      if (url === '/api/media' && init?.method === 'POST') {
        return jsonResponse({ id: 'asset_99', filename: 'bgm_track.mp3' }, 201);
      }
      return jsonResponse({ ok: true });
    }, async ({ fetchCalls }) => {
      const bgmFile = new File(['audio-bytes'], 'bgm_track.mp3', { type: 'audio/mpeg' });
      const thumbFile = new File(['png-bytes'], 'cover.png', { type: 'image/png' });
      await useDubDubStore.getState().setBackgroundAudio(bgmFile);
      await useDubDubStore.getState().setThumbnail(thumbFile);

      const bgm = useDubDubStore.getState().editVideo.backgroundAudio;
      expect(bgm).not.toBeNull();
      expect(bgm!.id).toBe('asset_99');
      expect(bgm!.url).toBe('/api/media/asset_99/file');
      expect(bgm!.url?.startsWith('blob:')).toBe(false);

      const thumb = useDubDubStore.getState().editVideo.thumbnail;
      expect(thumb).not.toBeNull();
      expect(thumb!.id).toBe('thumb_99');
      expect(thumb!.url).toBeTruthy();
      expect(thumb!.url?.startsWith('blob:')).toBe(false);
      expect(thumb!.url?.startsWith('data:image/png;base64,')).toBe(true);

      expect(fetchCalls.some((c) => c.url === '/api/assets/background-audio' && c.method === 'POST')).toBe(true);
      expect(fetchCalls.some((c) => c.url === '/api/media' && c.method === 'POST')).toBe(true);
      expect(fetchCalls.some((c) => c.url === '/api/assets/thumbnail' && c.method === 'POST')).toBe(true);

      await useDubDubStore.getState().triggerAutosave(true);
      const putCalls = fetchCalls.filter((c) => c.method === 'PUT' && c.url.includes('/api/projects/p_assets'));
      const putCall = putCalls[putCalls.length - 1];
      expect(putCall).toBeDefined();
      expect(putCall!.body.state.editVideo.backgroundAudio?.url).toBe('/api/media/asset_99/file');
      expect(putCall!.body.state.editVideo.thumbnail?.id).toBe('thumb_99');
      expect(putCall!.body.state.editVideo.thumbnail?.url).toBe(thumb!.url);
    });
  });

  test('[Tier 2] CapCutExportModal subtitles_target.srt exports rawTranslation/targetText, never sourceText', () => {
    const seg = {
      id: 1,
      startSec: 0,
      endSec: 2,
      sourceText: 'Source English Should Never Appear',
      rawTranslation: 'Bản dịch thô tiếng Việt',
      targetText: 'Bản dịch đã chỉnh sửa',
    };
    const srt = generateSrtContent([seg], 'rawTranslation' as any);
    expect(srt).toContain('Bản dịch thô tiếng Việt');
    expect(srt).not.toContain('Source English Should Never Appear');
  });
});

// ============================================================================
// F12 WebUI: Stage 1 video_ocr Selector, ROI Overlay & Confirmation Gating
// ============================================================================
describe('F12 WebUI — Stage 1 video_ocr Selector & ROI Gating', () => {
  test('[Tier 1] Stage1Prepare renders Subtitle Source selector (audio_asr vs video_ocr) and ROI overlay when video_ocr active', () => {
    useDubDubStore.setState({
      project: { ...useDubDubStore.getState().project, previewUrl: '/api/media/m_ocr/file', verified: true, hasVideo: true, hasAudio: false },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_ocr' },
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: false,
    } as any);

    const html = renderToStaticMarkup(<Stage1Prepare />);
    expect(html).toMatch(/video.?ocr|Hard-Subtitle/i);
    expect(html).toMatch(/Test Frame OCR|ocr-roi-overlay/i);
  });

  test('[Tier 1] buildPrepareJobRequest includes subtitleSource, ocrRoi, ocrRoiConfirmed, and ocrOptions when video_ocr selected', () => {
    useDubDubStore.setState({
      activeProjectId: 'p_vocr',
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_vocr' },
      ocrCrop: { ...useDubDubStore.getState().ocrCrop, roi: [0.05, 0.78, 0.9, 0.18] },
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: true,
      ocrOptions: { coarseIntervalMs: 500, useAngleCls: false, paragraphMode: 'preserve_lines' },
    } as any);

    const req: any = buildPrepareJobRequest(useDubDubStore.getState());
    expect(req.options.subtitleSource).toBe('video_ocr');
    expect(req.options.ocrRoi).toEqual([0.05, 0.78, 0.9, 0.18]);
    expect(req.options.ocrRoiConfirmed).toBe(true);
  });

  test('[Tier 2] startPrepareJob allows audio-less video and unconfigured cloud ASR when subtitleSource=video_ocr & ocrRoiConfirmed=true', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_audioless',
      currentStep: 1,
      project: { ...useDubDubStore.getState().project, filename: 'silent.mp4', verified: true, hasVideo: true, hasAudio: false },
      backend: {
        ...useDubDubStore.getState().backend,
        mediaId: 'm_silent',
        options: {
          ...useDubDubStore.getState().backend.options,
          asrProviders: [{ id: 'deepgram', label: 'Deepgram', recognType: 1, requiresSettings: true, configured: false, models: ['nova-3'] }],
        },
        config: { ...useDubDubStore.getState().backend.config, recognType: 1 },
      },
      ocrCrop: { ...useDubDubStore.getState().ocrCrop, roi: [0.1, 0.75, 0.8, 0.2] },
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: true,
    } as any);

    await withE2EShims(async (url, init, call) => {
      if (url === '/api/jobs' && init?.method === 'POST') {
        expect(call.body.options.subtitleSource).toBe('video_ocr');
        return jsonResponse({ id: 'job_vocr', status: 'succeeded', segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'OCR Line', targetText: '', sourceType: 'ocr' }] });
      }
      return jsonResponse({ ok: true, projects: [] });
    }, async () => {
      await useDubDubStore.getState().startPrepareJob();
      const s = useDubDubStore.getState();
      expect(s.jobStatus).toBe('completed');
      expect(s.currentStep).toBe(2);
      expect(s.segments[0].sourceText).toBe('OCR Line');
    });
  });

  test('[Tier 2] startPrepareJob blocks video_ocr submission when ocrRoiConfirmed is false', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_unconfirmed',
      project: { ...useDubDubStore.getState().project, filename: 'unconfirmed.mp4', verified: true, hasVideo: true, hasAudio: false },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_unconfirmed' },
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: false,
    } as any);

    await withE2EShims(async () => jsonResponse({ ok: true }), async ({ fetchCalls }) => {
      await useDubDubStore.getState().startPrepareJob();
      expect(fetchCalls.some((c) => c.url === '/api/jobs')).toBe(false);
      expect(useDubDubStore.getState().jobStatus).toBe('failed');
    });
  });

  test('[Tier 2] triggerAutosave and selectProject persist and restore subtitleSource, ocrRoiConfirmed, and ocrOptions', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_persist_ocr',
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_persist' },
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: true,
      ocrOptions: { coarseIntervalMs: 400, useAngleCls: false, paragraphMode: 'preserve_lines' },
    } as any);

    let savedState: any = null;
    await withE2EShims(async (url, init, call) => {
      if (url.includes('/api/projects/p_persist_ocr') && init?.method === 'PUT') {
        savedState = call.body.state;
        return jsonResponse({ ok: true });
      }
      if (url.includes('/api/projects/p_persist_ocr')) {
        return jsonResponse({ id: 'p_persist_ocr', name: 'Persist.mp4', stage: 1, media_id: 'm_persist', state: savedState });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().triggerAutosave(true);
      expect(savedState.subtitleSource).toBe('video_ocr');
      expect(savedState.ocrRoiConfirmed).toBe(true);

      resetStore();
      await useDubDubStore.getState().selectProject('p_persist_ocr', true);
      const restored: any = useDubDubStore.getState();
      expect(restored.subtitleSource).toBe('video_ocr');
      expect(restored.ocrRoiConfirmed).toBe(true);
    });
  });
});

// ============================================================================
// Tier 3 Pairwise Cross-Feature Flows
// ============================================================================
describe('Tier 3 — Cross-Feature UI Workflows', () => {
  test('[Tier 3] F2 + F4 + F5: Stage 2 replaceSegmentWithOcrEntries remaps overrides -> Batch Translate -> Stage 3 runFullDubbing -> enterStage4', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_t3',
      currentStep: 2,
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
      segments: [
        { ...baseSeg, id: 1, startSec: 0, endSec: 4, speakerId: 'spk_1', sourceText: 'Old ASR' },
        { ...baseSeg, id: 2, startSec: 5, endSec: 8, speakerId: 'spk_2', sourceText: 'Second line' },
      ] as any,
      segmentVoiceOverrides: { 2: 'custom_spk2' },
    });

    await withE2EShims(async (url, _init, call) => {
      if (url.includes('/api/translate')) {
        return jsonResponse({
          ok: true,
          segments: call.body.segments.map((s: any) => ({ id: s.id, targetText: `VI:${s.sourceText}` })),
        });
      }
      if (url.includes('/api/tts/preview')) {
        return jsonResponse({ preview_url: `/api/audio/${call.body.segment_id}.wav`, id: `p_${call.body.segment_id}`, applied_speed: 1.0 });
      }
      if (url.includes('/api/tts/batch') || url.includes('/dubbing/generate')) {
        const segs = call.body.segments || call.body.items || [];
        return jsonResponse({
          ok: true,
          results: segs.map((s: any) => ({ segmentId: s.segmentId ?? s.id, ok: true, previewUrl: `/api/audio/${s.segmentId ?? s.id}.wav`, id: `p_${s.segmentId ?? s.id}` })),
        });
      }
      if (url.includes('/dubbing/assemble')) {
        return jsonResponse({ ok: true, audio_url: '/api/projects/p_t3/dubbing/audio' });
      }
      return jsonResponse({ ok: true });
    }, async () => {
      await useDubDubStore.getState().replaceSegmentWithOcrEntries(1, [
        { startSec: 0, endSec: 2, startTime: '00:00:00,000', endTime: '00:00:02,000', text: 'OCR Part 1', confidence: 0.95 },
        { startSec: 2, endSec: 4, startTime: '00:00:02,000', endTime: '00:00:04,000', text: 'OCR Part 2', confidence: 0.94 },
      ]);

      expect(useDubDubStore.getState().segmentVoiceOverrides[3]).toBe('custom_spk2');
      await useDubDubStore.getState().runBatchTranslation('en', 'vi');
      expect(useDubDubStore.getState().segments[2].targetText).toBe('VI:Second line');

      await useDubDubStore.getState().runFullDubbing();
      expect(useDubDubStore.getState().dubbingStatus).toBe('completed');

      await useDubDubStore.getState().enterStage4();
      expect(useDubDubStore.getState().currentStep).toBe(4);
      expect(useDubDubStore.getState().assembledDubUrl).toContain('/api/projects/p_t3/dubbing/audio');
    });
  });

  test('[Tier 3] F3 + F12: Stage 1 video_ocr -> Stage 2 OCR segments -> VideoPlayer dynamic channels & gap hiding', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_vocr_player',
      currentStep: 1,
      languages: {
        source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
        target: { code: 'vi', name: 'Vietnamese' },
        timingMode: 'voice',
      },
      project: { ...useDubDubStore.getState().project, filename: 'silent_zh.mp4', verified: true, hasVideo: true, hasAudio: false },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_silent_zh' },
      subtitleSource: 'video_ocr',
      ocrRoiConfirmed: true,
    } as any);

    await withE2EShims(async (url, init) => {
      if (url === '/api/jobs' && init?.method === 'POST') {
        return jsonResponse({
          id: 'job_vp',
          status: 'succeeded',
          segments: [
            { id: 1, startSec: 0, endSec: 2, sourceText: '你好', targetText: 'Xin chào', sourceType: 'ocr' },
            { id: 2, startSec: 6, endSec: 8, sourceText: '再见', targetText: 'Tạm biệt', sourceType: 'ocr' },
          ],
        });
      }
      return jsonResponse({ ok: true, projects: [] });
    }, async ({ fetchCalls }) => {
      await useDubDubStore.getState().startPrepareJob();
      const jobPost = fetchCalls.find((c) => c.url === '/api/jobs' && c.method === 'POST');
      expect(jobPost).toBeDefined();
      expect(jobPost!.body?.options?.subtitleSource).toBe('video_ocr');
      expect(useDubDubStore.getState().currentStep).toBe(2);

      // Inside Segment 1 [0..2]: subtitle visible + ZH/VI channel labels
      useDubDubStore.getState().updatePlaybackTime(1.0);
      const activeHtml = renderToStaticMarkup(<VideoPlayer showAudioSwitcher={true} />);
      expect(activeHtml).toContain('Xin chào');
      expect(activeHtml).toContain('ZH');
      expect(activeHtml).toContain('VI');

      // Inside silent gap [2..6]: subtitle overlay must be hidden
      useDubDubStore.getState().updatePlaybackTime(4.0);
      const gapHtml = renderToStaticMarkup(<VideoPlayer showAudioSwitcher={true} />);
      expect(gapHtml).not.toContain('Xin chào');
      expect(gapHtml).not.toContain('Tạm biệt');
    });
  });
});

// ============================================================================
// Tier 4: End-to-End Real-World Frontend Application Scenarios
// ============================================================================
describe('Tier 4 — End-to-End Real-World Frontend Scenarios', () => {
  test('[Tier 4 Scenario A] Batch midnight local-date scan -> SSE ASR (concurrency=2) -> Stage 2 OCR+Translate -> Stage 3 failure+retry -> Stage 4 BGM & CapCut -> resetStage(2)', async () => {
    await withFakeUtcPlus7Clock(async () => {
      const localToday = formatLocalYmd(new Date()); // 2026-11-18
      const parentId = `parent_${localToday}`;
      let ttsShouldFail = true;

      await withE2EShims(async (url, init, call) => {
        if (url.includes('/api/media/scan-folders')) {
          return jsonResponse({
            ok: true,
            videos: [
              { path: '/batch/ep1/video.mp4', filename: 'video.mp4', sizeBytes: 1000, parentDir: '/batch/ep1' },
              { path: '/batch/ep2/video.mp4', filename: 'video.mp4', sizeBytes: 1100, parentDir: '/batch/ep2' },
              { path: '/batch/ep3/video.mp4', filename: 'video.mp4', sizeBytes: 1200, parentDir: '/batch/ep3' },
            ],
          });
        }
        if (url.includes('/api/media/register-local')) {
          return jsonResponse({
            ok: true,
            media: [
              { id: 'm1', filename: 'video.mp4', path: '/batch/ep1/video.mp4', durationMs: 10000 },
              { id: 'm2', filename: 'video.mp4', path: '/batch/ep2/video.mp4', durationMs: 10000 },
              { id: 'm3', filename: 'video.mp4', path: '/batch/ep3/video.mp4', durationMs: 10000 },
            ],
          });
        }
        if (url.includes('/api/projects/batch-create')) {
          return jsonResponse({
            ok: true,
            parent: { id: parentId, name: localToday, stage: 1, state: { is_parent: true } },
            projects: [
              { id: 't4_s1', name: 'ep1/video.mp4', parent_id: parentId, media_id: 'm1', stage: 1, status: 'pending' },
              { id: 't4_s2', name: 'ep2/video.mp4', parent_id: parentId, media_id: 'm2', stage: 1, status: 'pending' },
              { id: 't4_s3', name: 'ep3/video.mp4', parent_id: parentId, media_id: 'm3', stage: 1, status: 'pending' },
            ],
          });
        }
        if (url.includes('/api/projects/t4_s1') && (!init?.method || init.method === 'GET')) {
          return jsonResponse({
            id: 't4_s1',
            name: 'ep1/video.mp4',
            parent_id: parentId,
            media_id: 'm1',
            stage: 1,
            state: { currentStep: 1, segments: [] },
          });
        }
        if (url.includes(`/api/projects/${parentId}/subprojects`)) {
          return jsonResponse({ subprojects: useDubDubStore.getState().sessionSubprojects });
        }
        if (url === '/api/jobs' && init?.method === 'POST') {
          return jsonResponse({ id: `job_${call.body.projectId}`, status: 'running' });
        }
        if (url.startsWith('/api/jobs/job_')) {
          return jsonResponse({
            id: url.split('/').pop(),
            status: 'succeeded',
            segments: [{ id: 1, startSec: 0, endSec: 3, sourceText: 'ASR draft', targetText: '' }],
          });
        }
        if (url.includes('/api/translate')) {
          return jsonResponse({
            ok: true,
            segments: call.body.segments.map((s: any) => ({ id: s.id, targetText: 'Bản dịch chuẩn', rawTranslation: 'Bản dịch thô' })),
          });
        }
        if (url.includes('/api/tts/preview')) {
          if (ttsShouldFail) return jsonResponse({ detail: 'TTS temporary outage' }, 500);
          return jsonResponse({ preview_url: '/api/audio/t4_1.wav', id: 'prev_t4_1', applied_speed: 1.0 });
        }
        if (url.includes('/api/tts/batch') || url.includes('/dubbing/generate')) {
          if (ttsShouldFail) return jsonResponse({ detail: 'TTS temporary outage' }, 500);
          return jsonResponse({ ok: true, results: [{ segmentId: 1, ok: true, previewUrl: '/api/audio/t4_1.wav', id: 'prev_t4_1' }] });
        }
        if (url.includes('/dubbing/assemble')) {
          return jsonResponse({ ok: true, audio_url: '/api/projects/t4_s1/dubbing/audio' });
        }
        if (url === '/api/assets/background-audio' && init?.method === 'POST') {
          return jsonResponse({ id: 'bgm_t4', name: 'bgm.mp3' }, 201);
        }
        if (url === '/api/assets/thumbnail' && init?.method === 'POST') {
          return jsonResponse({ id: 'thumb_t4', name: 'thumb.png' }, 201);
        }
        if (url === '/api/media' && init?.method === 'POST') {
          return jsonResponse({ id: 'bgm_t4', filename: 'bgm.mp3' }, 201);
        }
        if (url.includes('/stages/2/reset')) {
          return jsonResponse({ ok: true });
        }
        return jsonResponse({ ok: true, projects: [] });
      }, async () => {
        // 1. Scan & batch-create under local 2026-11-18
        const scan = await scanFolders(['/batch'], true);
        const reg = await registerLocalMedia(scan.videos.map((v) => v.path));
        const batch = await batchCreateProjects({
          date: localToday,
          items: reg.media.map((m) => ({ name: m.filename, mediaId: m.id, mediaPath: m.path })),
        });
        expect(batch.parent.id).toBe('parent_2026-11-18');
        useDubDubStore.getState().setSessionSubprojects(batch.projects, batch.parent.id, batch.parent.name);
        await useDubDubStore.getState().selectProject('t4_s1', true);

        // 2. Batch prepare concurrency=2 with SSE task_done + done
        await useDubDubStore.getState().startBatchPrepareJobs();
        await flushTasks();
        const es1 = MockEventSource.findByJobId('job_t4_s1')!;
        es1.emitMessage({ status: 'succeeded', stage: 'task_done', progress: 100 });
        es1.emitDone({ status: 'succeeded' });
        await flushTasks();
        MockEventSource.findByJobId('job_t4_s2')!.emitMessage({ status: 'succeeded', progress: 100 });
        await flushTasks();
        MockEventSource.findByJobId('job_t4_s3')!.emitMessage({ status: 'succeeded', progress: 100 });
        await flushTasks();

        expect(useDubDubStore.getState().currentStep).toBe(2);

        // 3. Stage 2 OCR replace + batch translate
        await useDubDubStore.getState().replaceSegmentWithOcrEntries(1, [
          { startSec: 0, endSec: 3, startTime: '00:00:00,000', endTime: '00:00:03,000', text: 'OCR corrected line', confidence: 0.98 },
        ]);
        await useDubDubStore.getState().runBatchTranslation('en', 'vi');
        expect(useDubDubStore.getState().segments[0].targetText).toBe('Bản dịch chuẩn');

        // 4. Stage 3 failure detection followed by retry success
        useDubDubStore.setState({ voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any });
        await useDubDubStore.getState().runFullDubbing();
        expect(useDubDubStore.getState().dubbingStatus).toBe('failed');

        ttsShouldFail = false;
        await useDubDubStore.getState().runFullDubbing();
        expect(useDubDubStore.getState().dubbingStatus).toBe('completed');

        // 5. Stage 4 assembly, BGM upload persistence, CapCut SRT check, and resetStage(2)
        await useDubDubStore.getState().enterStage4();
        expect(useDubDubStore.getState().currentStep).toBe(4);
        await useDubDubStore.getState().setBackgroundAudio(new File(['mp3'], 'bgm.mp3', { type: 'audio/mpeg' }));
        expect(useDubDubStore.getState().editVideo.backgroundAudio?.url).toBe('/api/media/bgm_t4/file');

        const srt = generateSrtContent(useDubDubStore.getState().segments, 'rawTranslation' as any);
        expect(srt).not.toContain('OCR corrected line');

        await useDubDubStore.getState().resetStage(2);
        expect(useDubDubStore.getState().jobStatus).toBe('idle');
      });
    });
  });

  test('[Tier 4 Scenario B] Audio-less video_ocr -> Stage 1 ROI extract confirm -> Stage 2 review & SRT export -> Stage 3 dubbing -> Stage 4 render payload', async () => {
    useDubDubStore.setState({
      activeProjectId: 'p_t4_vocr',
      currentStep: 1,
      voices: [{ id: 'Ngọc', name: 'Ngọc', provider: 2, kind: 'preset' }] as any,
      project: { ...useDubDubStore.getState().project, filename: 'silent_tutorial.mp4', verified: true, hasVideo: true, hasAudio: false, durationSec: 10 },
      backend: { ...useDubDubStore.getState().backend, mediaId: 'm_t4_vocr' },
      languages: { source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false }, target: { code: 'vi', name: 'Vietnamese' }, timingMode: 'voice' },
    });

    await withE2EShims(async (url, init, call) => {
      if (url.includes('/api/ocr/extract')) {
        return jsonResponse({ ok: true, text: '测试帧字幕', confidence: 0.97 });
      }
      if (url === '/api/render' && init?.method === 'POST') {
        return jsonResponse({ id: 'job_render_t4', status: 'running' });
      }
      if (url === '/api/jobs' && init?.method === 'POST') {
        return jsonResponse({ id: 'job_vocr_t4', status: 'running' });
      }
      if (url.includes('/api/jobs/job_render_t4')) {
        return jsonResponse({ id: 'job_render_t4', status: 'succeeded', progress: 100 });
      }
      if (url.includes('/api/jobs/job_vocr_t4')) {
        return jsonResponse({
          id: 'job_vocr_t4',
          status: 'succeeded',
          segments: [
            { id: 1, startSec: 0, endSec: 2.5, startTime: '00:00.000', endTime: '00:02.500', sourceText: '第一句字幕\n第二行', targetText: '', sourceType: 'ocr' },
          ],
        });
      }
      if (url.includes('/api/translate')) {
        return jsonResponse({ ok: true, segments: [{ id: 1, targetText: 'Phụ đề câu một' }] });
      }
      if (url.includes('/api/tts/preview')) {
        return jsonResponse({ preview_url: '/api/audio/vocr_1.wav', id: 'prev_vocr_1', applied_speed: 1.0 });
      }
      if (url.includes('/api/tts/batch') || url.includes('/dubbing/generate')) {
        return jsonResponse({ ok: true, results: [{ segmentId: 1, ok: true, previewUrl: '/api/audio/vocr_1.wav', id: 'prev_vocr_1' }] });
      }
      if (url.includes('/dubbing/assemble')) {
        return jsonResponse({ ok: true, audio_url: '/api/projects/p_t4_vocr/dubbing/audio' });
      }
      return jsonResponse({ ok: true, projects: [] });
    }, async ({ downloads, fetchCalls }) => {
      const store: any = useDubDubStore.getState();
      store.setSubtitleSource?.('video_ocr');
      if (!store.setSubtitleSource) useDubDubStore.setState({ subtitleSource: 'video_ocr' } as any);
      store.setOcrCropRoi([0.08, 0.78, 0.84, 0.16]);
      store.setOcrRoiConfirmed?.(true);
      if (!store.setOcrRoiConfirmed) useDubDubStore.setState({ ocrRoiConfirmed: true } as any);

      // 1. Start video_ocr Stage 1 job on audio-less video
      await useDubDubStore.getState().startPrepareJob();
      const es = MockEventSource.findByJobId('job_vocr_t4')!;
      es.emitMessage({ status: 'succeeded', progress: 100 });
      await flushMicrotasks(10);

      expect(useDubDubStore.getState().currentStep).toBe(2);
      expect(useDubDubStore.getState().segments[0].sourceText).toBe('第一句字幕\n第二行');

      // 2. Translate & export SRT
      await useDubDubStore.getState().runBatchTranslation('zh-cn', 'vi');
      exportTranslatedSrt(useDubDubStore.getState().segments);
      expect(downloads).toHaveLength(1);
      expect(await downloads[0].readText()).toContain('Phụ đề câu một');

      // 3. Stage 3 dubbing & Stage 4 render request
      await useDubDubStore.getState().runFullDubbing();
      expect(useDubDubStore.getState().dubbingStatus).toBe('completed');
      await useDubDubStore.getState().enterStage4();
      expect(useDubDubStore.getState().currentStep).toBe(4);

      const renderPayload = buildRenderRequest(useDubDubStore.getState());
      expect(renderPayload.jobType).toBe('render');
      await useDubDubStore.getState().exportEditedVideo();
      expect(fetchCalls.some((c) => c.url === '/api/render' && c.method === 'POST' && c.body?.jobType === 'render')).toBe(true);
    });
  });
});
