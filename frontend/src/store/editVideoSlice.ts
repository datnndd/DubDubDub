import { StateCreator } from 'zustand';
import type { AudioMixSettings, SubtitleStyleSettings, EditVideoState } from '../types/editor';
import type { AnnotationItem, AnnotationStyle } from '../types/annotation';
import { uploadEditAsset, uploadMedia } from '../api/media';
import {
  requestAnnotationRegion,
  requestAnnotationScan,
  requestRender,
  requestTranslate,
} from '../api/stages';

async function readFileAsDataUrl(file: File): Promise<string> {
  if (typeof FileReader === 'undefined') {
    return typeof URL !== 'undefined' && URL.createObjectURL ? URL.createObjectURL(file) : '';
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
    reader.onerror = () => reject(reader.error || new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

const DEFAULT_AUDIO_MIX: AudioMixSettings = { original: 0, dubbed: 100, background: 35 };

export const DEFAULT_ANNOTATION_EXCLUDE_ROI: [number, number, number, number] = [0.0, 0.78, 1.0, 0.22];

export const DEFAULT_SUBTITLE_STYLES: SubtitleStyleSettings = {
  preset: 'clean', fontFamily: 'Arial', fontSize: 22, color: '#FFFFFF',
  outlineColor: '#000000', outlineWidth: 2, shadowColor: 'rgba(0,0,0,.75)',
  shadowSize: 2, fontWeight: 'normal', fontStyle: 'normal', opacity: 100,
  aiLipSync: true, deReverb: true, faceRetouch: false, superRes4K: true,
};

export const DEFAULT_EDIT_VIDEO: EditVideoState = {
  audioMix: { ...DEFAULT_AUDIO_MIX }, backgroundAudio: null, thumbnail: null,
  exporting: false, error: null, activeTab: 'audio', isCapCutModalOpen: false,
  burnAnnotations: true,
};

export function normalizeAnnotationRecord(raw: any, index: number): AnnotationItem {
  const rawBox = Array.isArray(raw?.box) && raw.box.length === 4 ? raw.box : [0.1, 0.12, 0.32, 0.08];
  const bx = Math.max(0, Math.min(0.98, Number(rawBox[0]) || 0));
  const by = Math.max(0, Math.min(0.98, Number(rawBox[1]) || 0));
  const bw = Math.max(0.02, Math.min(1 - bx, Number(rawBox[2]) || 0.3));
  const bh = Math.max(0.02, Math.min(1 - by, Number(rawBox[3]) || 0.08));
  const startSec = Math.max(0, Number(raw?.startSec) || 0);
  const endSec = Math.max(startSec + 0.2, Number(raw?.endSec) || startSec + 2.5);
  const styleRaw = raw?.style && typeof raw.style === 'object' ? raw.style : {};
  return {
    id: String(raw?.id || `ann_${index + 1}`),
    startSec: Number(startSec.toFixed(3)),
    endSec: Number(endSec.toFixed(3)),
    ...(raw?.startTime ? { startTime: String(raw.startTime) } : {}),
    ...(raw?.endTime ? { endTime: String(raw.endTime) } : {}),
    box: [bx, by, bw, bh],
    sourceText: String(raw?.sourceText ?? ''),
    targetText: String(raw?.targetText ?? ''),
    ...(raw?.confidence != null ? { confidence: Number(raw.confidence) } : {}),
    style: {
      bgColor: String(styleRaw.bgColor || '#18181B'),
      bgOpacity: styleRaw.bgOpacity != null ? Math.max(0, Math.min(100, Number(styleRaw.bgOpacity))) : 92,
      textColor: String(styleRaw.textColor || '#FFFFFF'),
      fontSize: styleRaw.fontSize != null ? Math.max(10, Math.min(120, Number(styleRaw.fontSize))) : 28,
      bold: styleRaw.bold !== false,
      ...(styleRaw.fontFamily ? { fontFamily: String(styleRaw.fontFamily) } : {}),
    },
  };
}

export function buildRenderRequest(state: any) {
  const hasAnnotations = Array.isArray(state.annotations) && state.annotations.length > 0;
  return {
    mediaId: state.backend.mediaId,
    projectId: state.activeProjectId,
    jobType: 'render',
    options: {
      ...state.backend.config,
      audioMix: state.editVideo.audioMix,
      originalAudioVolume: state.editVideo.audioMix.original / 100,
      backgroundAudioVolume: state.editVideo.audioMix.background / 100,
      subtitleStyle: state.subtitleStyles,
      segments: state.segments,
      speakerVoiceMap: state.speakerVoiceMap,
      segmentVoiceOverrides: state.segmentVoiceOverrides,
      backgroundAudioId: state.editVideo.backgroundAudio?.id,
      thumbnailId: state.editVideo.thumbnail?.id,
      ...(state.autoFitVoiceSpeed !== undefined ? { autoFitVoiceSpeed: state.autoFitVoiceSpeed } : {}),
      ...(state.maxSpeedRate !== undefined ? { maxSpeedRate: state.maxSpeedRate } : {}),
      ...(hasAnnotations ? { annotations: state.annotations } : {}),
      ...(hasAnnotations || state.editVideo?.burnAnnotations !== undefined
        ? (hasAnnotations
            ? { burnAnnotations: state.editVideo?.burnAnnotations ?? true }
            : state.editVideo?.burnAnnotations !== undefined && state.annotations !== undefined
            ? { burnAnnotations: Boolean(state.editVideo.burnAnnotations) }
            : {})
        : {}),
    },
  };
}

export interface EditVideoSlice {
  subtitleStyles: SubtitleStyleSettings;
  editVideo: EditVideoState;
  muteCache: Record<string, number>;
  annotations: AnnotationItem[];
  activeAnnotationId: string | null;
  annotationDrawMode: boolean;
  annotationScanning: boolean;
  annotationTranslating: boolean;
  annotationStatusMessage: string | null;
  annotationExcludeRoi: [number, number, number, number];
  filterStaticWatermarks: boolean;

  setAudioMix: (channel: 'original' | 'dubbed' | 'background', val: any) => void;
  toggleAudioMute: (channel: 'original' | 'dubbed' | 'background') => void;
  setBackgroundAudio: (file: File | null) => Promise<void>;
  removeBackgroundAudio: () => void;
  setThumbnail: (file: File | null) => Promise<void>;
  removeThumbnail: () => void;
  updateSubtitleStyle: (key: keyof SubtitleStyleSettings, val: any) => void;
  setActiveInspectorTab: (tab: 'audio' | 'subtitles' | 'annotations' | 'assets') => void;
  setCapCutModalOpen: (open: boolean) => void;
  exportEditedVideo: () => Promise<void>;

  setActiveAnnotationId: (id: string | null) => void;
  setAnnotationDrawMode: (enabled: boolean) => void;
  setFilterStaticWatermarks: (enabled: boolean) => void;
  setAnnotationExcludeRoi: (roi: [number, number, number, number]) => void;
  setBurnAnnotations: (enabled: boolean) => void;
  scanVideoAnnotations: () => Promise<void>;
  extractAnnotationFromRoi: (roi: [number, number, number, number], anchorSec?: number) => Promise<void>;
  addManualAnnotation: (atSec?: number) => void;
  updateAnnotation: (id: string, patch: Partial<AnnotationItem>) => void;
  updateAnnotationStyle: (id: string, stylePatch: Partial<AnnotationStyle>) => void;
  deleteAnnotation: (id: string) => void;
  clearAnnotations: () => void;
  translateAnnotations: (ids?: string[]) => Promise<void>;
}

export const createEditVideoSlice: StateCreator<any, [], [], EditVideoSlice> = (set, get) => ({
  subtitleStyles: { ...DEFAULT_SUBTITLE_STYLES },
  editVideo: { ...DEFAULT_EDIT_VIDEO },
  muteCache: {},
  annotations: [],
  activeAnnotationId: null,
  annotationDrawMode: false,
  annotationScanning: false,
  annotationTranslating: false,
  annotationStatusMessage: null,
  annotationExcludeRoi: [...DEFAULT_ANNOTATION_EXCLUDE_ROI],
  filterStaticWatermarks: true,

  setAudioMix: (channel: 'original' | 'dubbed' | 'background', val: any) => {
    if (!['original', 'dubbed', 'background'].includes(channel)) return;
    const num = Number(val);
    const clamped = isNaN(num) ? 0 : Math.max(0, Math.min(150, num));
    set((state: any) => ({
      editVideo: {
        ...state.editVideo,
        audioMix: {
          ...state.editVideo.audioMix,
          [channel]: clamped,
        },
      },
    }));
    get().triggerAutosave();
  },

  toggleAudioMute: (channel: 'original' | 'dubbed' | 'background') => {
    if (!['original', 'dubbed', 'background'].includes(channel)) return;
    const curVal = get().editVideo.audioMix[channel];
    const cache = { ...(get().muteCache || {}) };

    if (curVal > 0) {
      cache[channel] = curVal;
      set((state: any) => ({
        muteCache: cache,
        editVideo: {
          ...state.editVideo,
          audioMix: {
            ...state.editVideo.audioMix,
            [channel]: 0,
          },
        },
      }));
    } else {
      const restored = cache[channel] && cache[channel] > 0
        ? cache[channel]
        : DEFAULT_AUDIO_MIX[channel] || 100;
      set((state: any) => ({
        editVideo: {
          ...state.editVideo,
          audioMix: {
            ...state.editVideo.audioMix,
            [channel]: restored,
          },
        },
      }));
    }
    get().triggerAutosave();
  },

  setBackgroundAudio: async (file: File | null) => {
    if (!file) return;
    set((state: any) => ({
      editVideo: { ...state.editVideo, error: null },
    }));
    try {
      const resp = await uploadEditAsset('background-audio', file);
      const respUrl = (resp as any)?.url;
      let url = typeof respUrl === 'string' && respUrl && !respUrl.startsWith('blob:') ? respUrl : '';
      if (!url) {
        try {
          const mediaResp = await uploadMedia(file);
          if (mediaResp?.id) {
            url = `/api/media/${mediaResp.id}/file`;
          }
        } catch {
          // Fallback when /api/media is unavailable in mocked test environments
        }
      }
      if (!url && typeof URL !== 'undefined' && URL.createObjectURL) {
        url = URL.createObjectURL(file);
      }
      set((state: any) => ({
        editVideo: {
          ...state.editVideo,
          error: null,
          backgroundAudio: { id: resp.id, file, name: resp.name, url },
        },
      }));
      get().triggerAutosave();
    } catch (err: any) {
      console.warn('Failed to upload background audio:', err);
      set((state: any) => ({
        editVideo: {
          ...state.editVideo,
          error: err?.message || 'Failed to upload background audio',
        },
      }));
    }
  },

  removeBackgroundAudio: () => {
    const cur = get().editVideo.backgroundAudio;
    if (cur?.url && cur.url.startsWith('blob:') && typeof URL !== 'undefined') {
      try { URL.revokeObjectURL(cur.url); } catch (_) {}
    }
    set((state: any) => ({
      editVideo: {
        ...state.editVideo,
        backgroundAudio: null,
      },
    }));
    get().triggerAutosave();
  },

  setThumbnail: async (file: File | null) => {
    if (!file) return;
    set((state: any) => ({
      editVideo: { ...state.editVideo, error: null },
    }));
    try {
      const resp = await uploadEditAsset('thumbnail', file);
      const respUrl = (resp as any)?.url;
      let url = typeof respUrl === 'string' && respUrl && !respUrl.startsWith('blob:') ? respUrl : '';
      if (!url) {
        try {
          url = await readFileAsDataUrl(file);
        } catch {
          if (typeof URL !== 'undefined' && URL.createObjectURL) {
            url = URL.createObjectURL(file);
          }
        }
      }
      set((state: any) => ({
        editVideo: {
          ...state.editVideo,
          error: null,
          thumbnail: { id: resp.id, file, name: resp.name, url },
        },
      }));
      get().triggerAutosave();
    } catch (err: any) {
      console.warn('Failed to upload thumbnail:', err);
      set((state: any) => ({
        editVideo: {
          ...state.editVideo,
          error: err?.message || 'Failed to upload thumbnail',
        },
      }));
    }
  },

  removeThumbnail: () => {
    const cur = get().editVideo.thumbnail;
    if (cur?.url && cur.url.startsWith('blob:') && typeof URL !== 'undefined') {
      try { URL.revokeObjectURL(cur.url); } catch (_) {}
    }
    set((state: any) => ({
      editVideo: {
        ...state.editVideo,
        thumbnail: null,
      },
    }));
    get().triggerAutosave();
  },

  updateSubtitleStyle: (key: keyof SubtitleStyleSettings, val: any) => {
    set((state: any) => ({
      subtitleStyles: {
        ...state.subtitleStyles,
        [key]: val,
      },
    }));
    get().triggerAutosave();
  },

  setActiveInspectorTab: (tab: 'audio' | 'subtitles' | 'annotations' | 'assets') => {
    set((state: any) => ({
      editVideo: {
        ...state.editVideo,
        activeTab: tab,
      },
    }));
  },

  setCapCutModalOpen: (open: boolean) => {
    set((state: any) => ({
      editVideo: {
        ...state.editVideo,
        isCapCutModalOpen: open,
      },
    }));
  },

  setActiveAnnotationId: (id: string | null) => {
    set({ activeAnnotationId: id });
  },

  setAnnotationDrawMode: (enabled: boolean) => {
    set((state: any) => ({
      annotationDrawMode: enabled,
      editVideo: {
        ...state.editVideo,
        ...(enabled ? { activeTab: 'annotations' as const } : {}),
      },
    }));
  },

  setFilterStaticWatermarks: (enabled: boolean) => {
    set({ filterStaticWatermarks: Boolean(enabled) });
  },

  setAnnotationExcludeRoi: (roi: [number, number, number, number]) => {
    set({ annotationExcludeRoi: roi });
  },

  setBurnAnnotations: (enabled: boolean) => {
    set((state: any) => ({
      editVideo: {
        ...state.editVideo,
        burnAnnotations: Boolean(enabled),
      },
    }));
    get().triggerAutosave();
  },

  scanVideoAnnotations: async () => {
    const state = get();
    const mediaId = state.backend?.mediaId;
    if (!mediaId) {
      set((s: any) => ({
        editVideo: { ...s.editVideo, error: 'Upload and inspect a video before scanning annotations' },
      }));
      return;
    }

    set((s: any) => ({
      annotationScanning: true,
      annotationStatusMessage: 'Scanning frames for on-screen callouts…',
      editVideo: { ...s.editVideo, error: null },
    }));

    try {
      const sourceLang = state.languages?.source?.code || 'zh-cn';
      const resp = await requestAnnotationScan({
        mediaId,
        language: sourceLang,
        excludeRoi: state.annotationExcludeRoi,
        sampleIntervalMs: 1000,
        filterStatic: state.filterStaticWatermarks,
      });

      const incoming = Array.isArray(resp?.annotations) ? resp.annotations : [];
      const existing: AnnotationItem[] = state.annotations || [];
      // Keep manual annotations that do not overlap newly scanned items
      const merged: AnnotationItem[] = [...existing];
      let addedCount = 0;
      incoming.forEach((rawItem, idx) => {
        const norm = normalizeAnnotationRecord(
          { ...rawItem, id: `ann_${Date.now().toString(36)}_${idx + 1}` },
          merged.length
        );
        const isDuplicate = merged.some(
          (ex) =>
            Math.abs(ex.startSec - norm.startSec) < 0.75 &&
            Math.abs(ex.box[0] - norm.box[0]) < 0.08 &&
            Math.abs(ex.box[1] - norm.box[1]) < 0.08 &&
            ex.sourceText.trim().toLowerCase() === norm.sourceText.trim().toLowerCase()
        );
        if (!isDuplicate) {
          merged.push(norm);
          addedCount += 1;
        }
      });

      merged.sort((a, b) => a.startSec - b.startSec || a.box[1] - b.box[1]);
      const wmNote = resp?.filteredWatermarks ? ` (${resp.filteredWatermarks} static watermark(s) filtered)` : '';
      set({
        annotations: merged,
        activeAnnotationId: merged[0]?.id || null,
        annotationScanning: false,
        annotationStatusMessage:
          addedCount > 0
            ? `Detected ${addedCount} on-screen annotation${addedCount === 1 ? '' : 's'}${wmNote}.`
            : `No new on-screen annotations found${wmNote}.`,
      });
      get().triggerAutosave();
    } catch (err: any) {
      const msg = err?.message || 'Annotation scan failed';
      set((s: any) => ({
        annotationScanning: false,
        annotationStatusMessage: null,
        editVideo: { ...s.editVideo, error: msg },
      }));
    }
  },

  extractAnnotationFromRoi: async (roi: [number, number, number, number], anchorSec?: number) => {
    const state = get();
    const mediaId = state.backend?.mediaId;
    const playhead = anchorSec ?? state.playback?.currentTime ?? 0;

    if (!mediaId) {
      set((s: any) => ({
        editVideo: { ...s.editVideo, error: 'Upload and inspect a video before drawing an annotation box' },
      }));
      return;
    }

    set((s: any) => ({
      annotationScanning: true,
      annotationStatusMessage: 'Reading drawn region & detecting time span…',
      editVideo: { ...s.editVideo, error: null, activeTab: 'annotations' },
    }));

    try {
      const sourceLang = state.languages?.source?.code || 'zh-cn';
      const resp = await requestAnnotationRegion({
        mediaId,
        anchorSec: Math.max(0, Number(playhead) || 0),
        roi,
        language: sourceLang,
        maxSpanSec: 15.0,
      });

      const currentList: AnnotationItem[] = get().annotations || [];
      const uniqueId = `ann_${Date.now().toString(36)}_${currentList.length + 1}`;
      const newAnn = normalizeAnnotationRecord(
        {
          ...(resp?.annotation || {}),
          id: uniqueId,
          box: resp?.annotation?.box || roi,
        },
        currentList.length
      );
      const nextList = [...currentList, newAnn].sort(
        (a, b) => a.startSec - b.startSec || a.box[1] - b.box[1]
      );

      set({
        annotations: nextList,
        activeAnnotationId: newAnn.id,
        annotationScanning: false,
        annotationDrawMode: false,
        annotationStatusMessage: newAnn.sourceText
          ? `Captured "${newAnn.sourceText.slice(0, 36)}" (${newAnn.startSec.toFixed(1)}s–${newAnn.endSec.toFixed(1)}s)`
          : 'Added region box — enter text in the card below.',
      });
      get().triggerAutosave();
    } catch (err: any) {
      const msg = err?.message || 'Region OCR failed';
      set((s: any) => ({
        annotationScanning: false,
        annotationStatusMessage: null,
        editVideo: { ...s.editVideo, error: msg },
      }));
    }
  },

  addManualAnnotation: (atSec?: number) => {
    const state = get();
    const duration = Math.max(1, Number(state.project?.durationSec || 10));
    const startSec = Math.max(0, Math.min(duration - 0.5, Number(atSec ?? state.playback?.currentTime ?? 0)));
    const endSec = Math.min(duration, startSec + 3.0);
    const list: AnnotationItem[] = state.annotations || [];
    const id = `ann_${Date.now().toString(36)}_${list.length + 1}`;
    const created: AnnotationItem = {
      id,
      startSec: Number(startSec.toFixed(2)),
      endSec: Number(Math.max(startSec + 0.5, endSec).toFixed(2)),
      box: [0.15, 0.15, 0.35, 0.09],
      sourceText: '',
      targetText: '',
      style: {
        bgColor: '#18181B',
        bgOpacity: 92,
        textColor: '#FFFFFF',
        fontSize: 28,
        bold: true,
      },
    };
    const next = [...list, created].sort((a, b) => a.startSec - b.startSec);
    set((s: any) => ({
      annotations: next,
      activeAnnotationId: id,
      editVideo: { ...s.editVideo, activeTab: 'annotations' },
    }));
    get().triggerAutosave();
  },

  updateAnnotation: (id: string, patch: Partial<AnnotationItem>) => {
    set((state: any) => ({
      annotations: (state.annotations || []).map((item: AnnotationItem) => {
        if (item.id !== id) return item;
        const nextStart = patch.startSec !== undefined ? Math.max(0, Number(patch.startSec) || 0) : item.startSec;
        const nextEnd =
          patch.endSec !== undefined
            ? Math.max(nextStart + 0.1, Number(patch.endSec) || nextStart + 0.5)
            : Math.max(nextStart + 0.1, item.endSec);
        return {
          ...item,
          ...patch,
          startSec: Number(nextStart.toFixed(3)),
          endSec: Number(nextEnd.toFixed(3)),
        };
      }),
    }));
    get().triggerAutosave();
  },

  updateAnnotationStyle: (id: string, stylePatch: Partial<AnnotationStyle>) => {
    set((state: any) => ({
      annotations: (state.annotations || []).map((item: AnnotationItem) =>
        item.id === id
          ? {
              ...item,
              style: {
                ...item.style,
                ...stylePatch,
              },
            }
          : item
      ),
    }));
    get().triggerAutosave();
  },

  deleteAnnotation: (id: string) => {
    set((state: any) => {
      const next = (state.annotations || []).filter((item: AnnotationItem) => item.id !== id);
      return {
        annotations: next,
        activeAnnotationId: state.activeAnnotationId === id ? (next[0]?.id || null) : state.activeAnnotationId,
      };
    });
    get().triggerAutosave();
  },

  clearAnnotations: () => {
    set({
      annotations: [],
      activeAnnotationId: null,
      annotationStatusMessage: null,
    });
    get().triggerAutosave();
  },

  translateAnnotations: async (ids?: string[]) => {
    const state = get();
    const list: AnnotationItem[] = state.annotations || [];
    const idSet = ids && ids.length ? new Set(ids) : null;
    const targets = list.filter(
      (a) => (!idSet || idSet.has(a.id)) && Boolean((a.sourceText || '').trim())
    );
    if (!targets.length) {
      set({ annotationStatusMessage: 'Add or scan source text before translating annotations.' });
      return;
    }

    set((s: any) => ({
      annotationTranslating: true,
      annotationStatusMessage: `Translating ${targets.length} annotation${targets.length === 1 ? '' : 's'}…`,
      editVideo: { ...s.editVideo, error: null },
    }));

    try {
      const sourceLanguage = state.languages?.source?.code || 'zh-cn';
      const targetLanguage = state.languages?.target?.code || 'vi';
      const translateType = Number(state.backend?.config?.translateType ?? 0);
      const resp = await requestTranslate({
        segments: targets.map((t, i) => ({
          id: i + 1,
          annotationId: t.id,
          startSec: t.startSec,
          endSec: t.endSec,
          sourceText: t.sourceText,
        })),
        sourceLanguage,
        targetLanguage,
        translateType,
        translationMode: 'text',
        projectId: state.activeProjectId || undefined,
        mediaId: state.backend?.mediaId || undefined,
      });

      const returned = Array.isArray(resp?.segments) ? resp.segments : [];
      const byAnnId = new Map<string, string>();
      targets.forEach((orig, idx) => {
        const match =
          returned.find((r: any) => r?.annotationId === orig.id) || returned[idx];
        const translated = String(match?.targetText ?? match?.text ?? '').trim();
        if (translated) {
          byAnnId.set(orig.id, translated);
        }
      });

      set((s: any) => ({
        annotations: (s.annotations || []).map((item: AnnotationItem) =>
          byAnnId.has(item.id) ? { ...item, targetText: byAnnId.get(item.id)! } : item
        ),
        annotationTranslating: false,
        annotationStatusMessage: `Translated ${byAnnId.size} of ${targets.length} annotation${targets.length === 1 ? '' : 's'}.`,
      }));
      get().triggerAutosave();
    } catch (err: any) {
      const msg = err?.message || 'Annotation translation failed';
      set((s: any) => ({
        annotationTranslating: false,
        annotationStatusMessage: null,
        editVideo: { ...s.editVideo, error: msg },
      }));
    }
  },

  exportEditedVideo: async () => {
    const state = get();
    const mediaId = state.backend?.mediaId;
    if (!mediaId) return;

    set((s: any) => ({
      editVideo: { ...s.editVideo, exporting: true, error: null },
      jobStatus: 'running',
      jobStage: 'render',
      jobProgress: 0,
      jobMessage: 'Rendering dubbed video…',
      startTime: Date.now(),
      elapsedSeconds: 0,
    }));

    try {
      const payload = buildRenderRequest(state);
      const res: any = await requestRender(payload);
      const jobId = res?.id || res?.jobId;
      if (!jobId) {
        throw new Error('Server did not return a render job ID');
      }
      if (res?.status === 'succeeded') {
        set((s: any) => ({
          editVideo: { ...s.editVideo, exporting: false, error: null },
          jobStatus: 'completed',
          jobStage: 'render',
          jobProgress: 100,
          jobMessage: res.message || 'Video rendered successfully',
        }));
        get().loadProjects?.();
        return;
      }
      state.subscribeToJob?.(jobId, state.activeProjectId, { jobType: 'render' });
    } catch (err: any) {
      const errMsg = err?.message || 'Video export failed';
      set((s: any) => ({
        editVideo: { ...s.editVideo, exporting: false, error: errMsg },
        jobStatus: 'failed',
        jobStage: 'render',
        jobMessage: errMsg,
      }));
    }
  },
});
