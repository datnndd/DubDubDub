import { StateCreator } from 'zustand';
import type { ProjectMetadata, BackendOptions } from '../types/project';
import type { OcrCropState, OcrOptions, OcrSubtitleEntry, Stage1OcrTestState, SubtitleSource } from '../types/segment';
import { uploadMedia } from '../api/media';
import { fetchOptions } from '../api/settings';
import { requestOcrExtract } from '../api/stages';
import { formatTimecode } from './playbackSlice';

export function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

export interface DeepgramConfigOptions {
  utt_split?: number;
  diarize_model?: string;
  smart_format?: boolean;
  punctuate?: boolean;
  paragraphs?: boolean;
  utterances?: boolean;
  extra?: string;
}

export interface PrepareSlice {
  selectedFile: File | null;
  mediaSelectionVersion: number;
  project: ProjectMetadata & { previewUrl?: string };
  languages: {
    source: { code: string; name: string; flag: string; autoDetected: boolean };
    target: { code: string; name: string };
    timingMode: string;
  };
  engines: {
    speakerDiarization: boolean;
    speakerCount: number;
    ocrSlideEngine: boolean;
  };
  backend: {
    ready: boolean;
    error: string | null;
    mediaId: string | null;
    status: 'idle' | 'analyzing' | 'ready' | 'submitting' | 'queued' | 'running' | 'completed' | 'failed';
    message: string;
    options: BackendOptions;
    config: {
      recognType: number;
      translateType: number;
      translationMode: string;
      ttsType: number;
      modelName: string;
      voiceRole: string;
      useCuda: boolean;
      deepgramOptions: DeepgramConfigOptions;
    };
  };
  ocrCrop: OcrCropState;
  subtitleSource: SubtitleSource;
  ocrRoiConfirmed: boolean;
  ocrOptions: OcrOptions;
  stage1OcrTest: Stage1OcrTestState;

  initializeBackend: (forceRefreshConfig?: boolean) => Promise<void>;
  selectMedia: (file: File) => Promise<void>;
  updateSourceLanguage: (code: string, name: string) => void;
  updateTargetLanguage: (code: string, name: string) => void;
  updateAsrProvider: (recognType: number, modelName: string) => void;
  updateTranslationProvider: (translateType: number) => void;
  updateTranslationMode: (mode: string) => void;
  updateTtsProvider: (ttsType: number) => void;
  updateBackendConfig: (keyOrPartial: string | Record<string, any>, value?: any) => void;
  updateDeepgramOptions: (opts: Partial<DeepgramConfigOptions>) => void;
  updateUseCuda: (useCuda: boolean) => void;
  updateTimingMode: (timingMode: string) => void;
  updateEngineConfig: (key: 'speakerDiarization' | 'speakerCount' | 'ocrSlideEngine', value: any) => void;
  setOcrCropRoi: (roi: [number, number, number, number]) => void;
  setOcrCropActive: (active: boolean, segmentId?: number | null) => void;
  setSubtitleSource: (source: SubtitleSource) => void;
  setOcrRoiConfirmed: (confirmed: boolean) => void;
  updateOcrOptions: (opts: Partial<OcrOptions>) => void;
  testStage1OcrFrame: (timeSec?: number) => Promise<{ ok: boolean; text: string; confidence?: number; entries?: OcrSubtitleEntry[]; error?: string }>;
}

export const createPrepareSlice: StateCreator<any, [], [], PrepareSlice> = (set, get) => ({
  selectedFile: null,
  mediaSelectionVersion: 0,
  project: {
    filename: 'Select a video to begin',
    format: '—',
    resolution: '—',
    fps: '—',
    duration: '00:00.000',
    durationSec: 0,
    fileSize: '—',
    videoCodec: '—',
    audioCodec: '—',
    bitrate: 'Not reported',
    hasAudio: false,
    hasVideo: false,
    verified: false,
    lastSaved: 'Just now',
  },
  languages: {
    source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
    target: { code: 'vi', name: 'Vietnamese' },
    timingMode: 'voice',
  },
  engines: {
    speakerDiarization: false,
    speakerCount: 0,
    ocrSlideEngine: true,
  },
  backend: {
    ready: false,
    error: null,
    mediaId: null,
    status: 'idle',
    message: 'Choose a video to start',
    options: {
      languages: [],
      asrProviders: [],
      translationProviders: [],
      translationModes: [],
      voices: [],
    },
    config: {
      recognType: 1,
      translateType: 0,
      translationMode: 'srt',
      ttsType: 2,
      modelName: 'nova-3',
      voiceRole: '',
      useCuda: false,
      deepgramOptions: {
        utt_split: 0.8,
        diarize_model: 'latest',
        smart_format: true,
        punctuate: true,
        paragraphs: true,
        utterances: true,
        extra: '',
      },
    },
  },
  ocrCrop: {
    active: false,
    segmentId: null,
    roi: [0.05, 0.75, 0.9, 0.2],
    loading: false,
    error: null,
  },
  subtitleSource: 'audio_asr',
  ocrRoiConfirmed: false,
  ocrOptions: {
    coarseIntervalMs: 500,
    useAngleCls: false,
    paragraphMode: 'preserve_lines',
  },
  stage1OcrTest: {
    loading: false,
    error: null,
    text: null,
  },

  initializeBackend: async (forceRefreshConfig: boolean = false) => {
    try {
      const opts = await fetchOptions();
      set((state: any) => {
        const shouldPreserve = Boolean(state.backend?.ready) && !forceRefreshConfig;
        return {
          backend: {
            ...state.backend,
            ready: true,
            options: opts,
            config: {
              ...state.backend.config,
              recognType: shouldPreserve ? state.backend.config.recognType : (opts.defaults?.recognType ?? state.backend.config.recognType),
              modelName: shouldPreserve ? state.backend.config.modelName : (opts.defaults?.modelName ?? state.backend.config.modelName),
              translateType: shouldPreserve ? state.backend.config.translateType : (opts.defaults?.translateType ?? state.backend.config.translateType),
              translationMode: shouldPreserve ? state.backend.config.translationMode : (opts.defaults?.translationMode ?? state.backend.config.translationMode),
              ttsType: shouldPreserve ? state.backend.config.ttsType : (opts.defaults?.ttsType ?? state.backend.config.ttsType),
            },
          },
        };
      });
    } catch (err: any) {
      set((state: any) => ({
        backend: {
          ...state.backend,
          error: `Backend unavailable: ${err.message}`,
        },
      }));
    }
  },

  selectMedia: async (file: File) => {
    if (!file) return;
    const version = (get().mediaSelectionVersion || 0) + 1;
    const previewUrl = URL.createObjectURL(file);

    set((state: any) => ({
      selectedFile: file,
      mediaSelectionVersion: version,
      ocrRoiConfirmed: false,
      stage1OcrTest: { loading: false, error: null, text: null },
      backend: {
        ...state.backend,
        status: 'analyzing',
        message: 'Uploading and inspecting source media…',
        mediaId: null,
        error: null,
      },
      project: {
        ...state.project,
        filename: file.name,
        fileSize: formatBytes(file.size),
        format: file.name.includes('.') ? file.name.split('.').pop()!.toUpperCase() : 'Media',
        previewUrl,
        verified: false,
      },
      playback: {
        ...state.playback,
        currentTime: 0,
        formattedTime: '00:00.000',
        isPlaying: false,
      },
    }));

    try {
      const media = await uploadMedia(file);
      if (get().mediaSelectionVersion !== version) return;

      const durSec = media.durationMs ? media.durationMs / 1000 : 0;
      const persistentPreviewUrl = `/api/media/${media.id}/file`;
      set((state: any) => ({
        backend: {
          ...state.backend,
          status: 'ready',
          message: 'Media verified — ready to process',
          mediaId: media.id,
        },
        project: {
          ...state.project,
          filename: media.filename,
          fileSize: formatBytes(media.sizeBytes),
          durationSec: durSec,
          duration: formatTimecode(durSec),
          resolution: media.resolution || 'Audio only',
          fps: media.fps ? `${Number(media.fps).toFixed(2)} fps` : '—',
          videoCodec: media.videoCodec || '—',
          format: media.container || state.project.format,
          bitrate: media.bitrate ? `${(Number(media.bitrate) / 1_000_000).toFixed(2)} Mbps` : 'Not reported',
          audioCodec: media.audioCodec || '—',
          hasAudio: media.hasAudio,
          hasVideo: media.hasVideo,
          verified: true,
          previewUrl: persistentPreviewUrl,
        },
        playback: {
          ...state.playback,
          duration: durSec,
        },
      }));

      const probedProject = { ...get().project };
      const curSubtitleSource = get().subtitleSource || 'audio_asr';
      const curOcrCrop = get().ocrCrop ? { ...get().ocrCrop } : undefined;
      const curOcrOptions = get().ocrOptions ? { ...get().ocrOptions } : undefined;

      // If no active project, or if current active project already has segments / was past step 1, create a new one
      const curProjId = get().activeProjectId;
      const hasExistingProgress = (get().segments?.length || 0) > 0 || get().currentStep > 1;
      if (!curProjId || hasExistingProgress) {
        await get().createNewProject(media.filename, media.id, durSec, undefined, {
          initialState: {
            project: probedProject,
            subtitleSource: curSubtitleSource,
            ocrCrop: curOcrCrop ? { roi: curOcrCrop.roi } : undefined,
            ocrRoiConfirmed: false,
            ocrOptions: curOcrOptions,
            languages: get().languages,
            backend: { mediaId: media.id, config: get().backend?.config },
          },
        });
        if (get().mediaSelectionVersion !== version) return;
        set((state: any) => ({
          subtitleSource: curSubtitleSource,
          ocrCrop: curOcrCrop ? { ...state.ocrCrop, roi: curOcrCrop.roi } : state.ocrCrop,
          ocrOptions: curOcrOptions || state.ocrOptions,
          project: {
            ...state.project,
            ...probedProject,
            previewUrl: persistentPreviewUrl,
            verified: true,
          },
          playback: {
            ...state.playback,
            duration: durSec,
          },
          backend: {
            ...state.backend,
            mediaId: media.id,
            status: 'ready',
            message: 'Media verified — ready to process',
          },
        }));
      }
      get().triggerAutosave();
    } catch (err: any) {
      if (get().mediaSelectionVersion !== version) return;
      set((state: any) => ({
        backend: {
          ...state.backend,
          status: 'failed',
          error: err.message,
          message: 'Media inspection failed',
        },
      }));
    }
  },

  updateSourceLanguage: (code: string, name: string) => {
    set((state: any) => ({
      languages: {
        ...state.languages,
        source: { code, name, flag: '', autoDetected: false },
      },
    }));
    get().triggerAutosave();
  },

  updateTargetLanguage: (code: string, name: string) => {
    set((state: any) => ({
      languages: {
        ...state.languages,
        target: { code, name },
      },
    }));
    get().loadVoices?.();
    get().triggerAutosave();
  },

  updateAsrProvider: (recognType: number, modelName: string) => {
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          recognType,
          modelName,
        },
      },
    }));
    get().triggerAutosave();
  },

  updateTranslationProvider: (translateType: number) => {
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          translateType,
        },
      },
    }));
    get().triggerAutosave?.();
  },

  updateTranslationMode: (translationMode: string) => {
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          translationMode,
        },
      },
    }));
    get().triggerAutosave?.();
  },

  updateTtsProvider: (ttsType: number) => {
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          ttsType,
        },
      },
    }));
    get().triggerAutosave?.();
  },

  updateBackendConfig: (keyOrPartial: string | Record<string, any>, value?: any) => {
    const patch =
      typeof keyOrPartial === 'string'
        ? { [keyOrPartial]: value }
        : keyOrPartial && typeof keyOrPartial === 'object'
        ? keyOrPartial
        : {};
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          ...patch,
        },
      },
    }));
    get().triggerAutosave?.();
  },

  updateDeepgramOptions: (opts: Partial<DeepgramConfigOptions>) => {
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          deepgramOptions: {
            ...state.backend.config.deepgramOptions,
            ...opts,
          },
        },
      },
    }));
    get().triggerAutosave?.();
  },

  updateUseCuda: (useCuda: boolean) => {
    set((state: any) => ({
      backend: {
        ...state.backend,
        config: {
          ...state.backend.config,
          useCuda,
        },
      },
    }));
    get().triggerAutosave?.();
  },

  updateTimingMode: (timingMode: string) => {
    set((state: any) => ({
      languages: {
        ...state.languages,
        timingMode,
      },
    }));
    get().triggerAutosave?.();
  },

  updateEngineConfig: (key: 'speakerDiarization' | 'speakerCount' | 'ocrSlideEngine', value: any) => {
    set((state: any) => ({
      engines: {
        ...state.engines,
        [key]: value,
      },
    }));
    get().triggerAutosave();
  },

  setOcrCropRoi: (roi: [number, number, number, number]) => {
    set((state: any) => ({
      ocrCrop: {
        ...state.ocrCrop,
        roi,
      },
      ocrRoiConfirmed: false,
      stage1OcrTest: {
        loading: false,
        error: null,
        text: null,
      },
    }));
    get().triggerAutosave?.();
  },

  setOcrCropActive: (active: boolean, segmentId: number | null = null) => {
    set((state: any) => ({
      ocrCrop: {
        ...state.ocrCrop,
        active,
        segmentId,
      },
    }));
  },

  setSubtitleSource: (source: SubtitleSource) => {
    set((state: any) => ({
      subtitleSource: source,
      stage1OcrTest: {
        ...(state.stage1OcrTest || { loading: false, text: null }),
        error: null,
      },
    }));
    get().triggerAutosave?.();
  },

  setOcrRoiConfirmed: (confirmed: boolean) => {
    set({ ocrRoiConfirmed: Boolean(confirmed) });
    get().triggerAutosave?.();
  },

  updateOcrOptions: (opts: Partial<OcrOptions>) => {
    set((state: any) => ({
      ocrOptions: {
        ...(state.ocrOptions || { coarseIntervalMs: 500, useAngleCls: false, paragraphMode: 'preserve_lines' }),
        ...opts,
      },
    }));
    get().triggerAutosave?.();
  },

  testStage1OcrFrame: async (timeSec?: number) => {
    const state = get();
    const mediaId = state.backend?.mediaId;
    const roi: [number, number, number, number] = state.ocrCrop?.roi || [0.05, 0.75, 0.9, 0.2];

    if (!mediaId) {
      const msg = 'Upload and verify a video before testing frame OCR.';
      set({
        ocrRoiConfirmed: false,
        stage1OcrTest: { loading: false, error: msg, text: null },
      });
      return { ok: false, text: '', error: msg };
    }

    if (!Array.isArray(roi) || roi.length !== 4 || roi[2] <= 0 || roi[3] <= 0) {
      const msg = 'Subtitle ROI width and height must be greater than 0%.';
      set({
        ocrRoiConfirmed: false,
        stage1OcrTest: { loading: false, error: msg, text: null },
      });
      return { ok: false, text: '', error: msg };
    }

    const dur = Number(state.project?.durationSec || 0);
    const rawCur = timeSec !== undefined && Number.isFinite(timeSec) ? timeSec : Number(state.playback?.currentTime || 0);
    const cur = Math.max(0, rawCur);

    let startSec: number;
    let endSec: number;
    if (dur > 0) {
      if (dur <= 1.0) {
        startSec = 0;
        endSec = dur;
      } else {
        startSec = Math.max(0, Math.min(cur - 0.5, dur - 1.0));
        endSec = Math.min(dur, startSec + 1.0);
      }
    } else {
      startSec = Math.max(0, cur - 0.5);
      endSec = startSec + 1.0;
    }
    startSec = Number(startSec.toFixed(3));
    endSec = Number(Math.max(startSec + 0.05, endSec).toFixed(3));

    set({
      stage1OcrTest: { loading: true, error: null, text: null, startSec, endSec },
    });

    try {
      const res = await requestOcrExtract({
        mediaId,
        roi,
        startSec,
        endSec,
        language: state.languages?.source?.code,
      });
      const extractedText = String(res?.text ?? '');
      const hasText = extractedText.trim().length > 0;
      const entries = Array.isArray(res?.entries) ? res.entries : [];

      set({
        ocrRoiConfirmed: hasText,
        stage1OcrTest: {
          loading: false,
          error: hasText ? null : 'No subtitle text detected in this 1s window. Scrub to a frame with subtitles, adjust ROI, or confirm manually if intentional.',
          text: extractedText,
          confidence: res?.confidence,
          startSec,
          endSec,
          entries,
        },
      });
      get().triggerAutosave?.();
      return {
        ok: true,
        text: extractedText,
        confidence: res?.confidence,
        entries,
      };
    } catch (err: any) {
      const msg = err?.message || 'Frame OCR test failed';
      set({
        ocrRoiConfirmed: false,
        stage1OcrTest: {
          loading: false,
          error: msg,
          text: null,
          startSec,
          endSec,
        },
      });
      return { ok: false, text: '', error: msg };
    }
  },
});
