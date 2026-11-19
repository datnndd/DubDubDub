import { StateCreator } from 'zustand';
import type { Segment, OcrSubtitleEntry } from '../types/segment';
import { formatTimecode } from './playbackSlice';
import { requestTranslate, requestOcrExtract } from '../api/stages';
import { updateProjectState } from '../api/projects';

const DEFAULT_SEGMENTS: Segment[] = [];

export interface TranscriptSlice {
  segments: Segment[];
  activeSegmentId: number | null;
  searchQuery: string;
  lockedTerms: Array<{ id: number; source: string; target: string }>;
  translationModal: {
    active: boolean;
    status: string;
    progress: number;
    message: string;
    error: string | null;
  };
  transcriptOptions: {
    utterances?: Segment[];
    paragraphs?: Segment[];
  } | null;
  selectedSegmentOption: 'utterances' | 'paragraphs';
  showSegmentationModal: boolean;

  setSegments: (segments: Segment[]) => void;
  setTranscriptOptions: (options: { utterances?: Segment[]; paragraphs?: Segment[] } | null) => void;
  selectSegmentationOption: (option: 'utterances' | 'paragraphs') => void;
  setShowSegmentationModal: (show: boolean) => void;
  setActiveSegmentId: (id: number | null) => void;
  setSearchQuery: (query: string) => void;
  updateSegmentText: (id: number, text: string, isTarget?: boolean) => void;
  updateSegmentTiming: (id: number, startSec: number, endSec: number) => void;
  splitSegment: (id: number, splitTime?: number) => void;
  deleteSegment: (id: number) => void;
  mergeWithNextSegment: (id: number) => void;
  extractOcrForSegment: (segmentId: number, roi: [number, number, number, number], interval?: { startSec: number; endSec: number }) => Promise<{ text: string; confidence?: number; entries?: OcrSubtitleEntry[] }>;
  applyOcrText: (segmentId: number, text: string, confidence?: number) => Promise<void>;
  replaceSegmentWithOcrEntries: (segmentId: number, ocrEntries: OcrSubtitleEntry[]) => Promise<void>;
  runBatchTranslation: (sourceLang?: string, targetLang?: string) => Promise<void>;
  translateSingleSegment: (segmentId: number) => Promise<void>;
  closeTranslationModal: () => void;
}

export function buildTranslationRequest(state: any) {
  return {
    segments: state.segments,
    sourceLanguage: state.languages?.source?.code || 'zh-cn',
    targetLanguage: state.languages?.target?.code || 'vi',
    translateType: state.backend?.config?.translateType ?? 0,
    translationMode: state.backend?.config?.translationMode || 'srt',
  };
}

export const createTranscriptSlice: StateCreator<any, [], [], TranscriptSlice> = (set, get) => ({
  segments: DEFAULT_SEGMENTS,
  activeSegmentId: 1,
  searchQuery: '',
  lockedTerms: [
    { id: 1, source: "AI", target: "inteligencia artificial" },
    { id: 2, source: "vocoder", target: "vocoder" },
    { id: 3, source: "neural mesh", target: "malla neuronal" },
  ],
  translationModal: {
    active: false,
    status: 'idle',
    progress: 0,
    message: 'Translating transcript...',
    error: null,
  },
  transcriptOptions: null,
  selectedSegmentOption: 'utterances',
  showSegmentationModal: false,

  setSegments: (segments: Segment[]) => {
    set({ segments });
    get().triggerAutosave();
  },

  setTranscriptOptions: (options) => {
    if (!options) {
      set({ transcriptOptions: null, showSegmentationModal: false });
      return;
    }
    const hasBoth = Boolean(options.utterances?.length && options.paragraphs?.length);
    const curSelected = get().selectedSegmentOption || 'utterances';
    const chosen = curSelected === 'paragraphs' && options.paragraphs?.length
      ? options.paragraphs
      : (options.utterances?.length ? options.utterances : options.paragraphs || []);

    set({
      transcriptOptions: options,
      showSegmentationModal: hasBoth,
      ...(chosen.length > 0 && get().segments.length === 0 ? { segments: chosen, activeSegmentId: chosen[0]?.id || 1 } : {}),
    });
    get().triggerAutosave();
  },

  selectSegmentationOption: (option) => {
    const opts = get().transcriptOptions;
    if (!opts) return;
    const chosen = option === 'paragraphs' ? opts.paragraphs : opts.utterances;
    if (Array.isArray(chosen) && chosen.length > 0) {
      set({
        selectedSegmentOption: option,
        segments: chosen,
        activeSegmentId: chosen[0]?.id || 1,
        showSegmentationModal: false,
      });
      get().triggerAutosave();
    }
  },

  setShowSegmentationModal: (show: boolean) => set({ showSegmentationModal: show }),

  setActiveSegmentId: (id: number | null) => set({ activeSegmentId: id }),

  setSearchQuery: (searchQuery: string) => set({ searchQuery }),

  updateSegmentText: (id: number, text: string, isTarget: boolean = false) => {
    set((state: any) => {
      const next = state.segments.map((s: Segment) => {
        if (s.id !== id) return s;
        const dur = Math.max(0.1, (s.endSec || 0) - (s.startSec || 0));
        const cps = Number((text.trim().length / dur).toFixed(1));
        const cpsStatus = cps <= 14.5 ? 'Optimal' : cps <= 18.0 ? 'Good' : 'Warning';
        if (isTarget) {
          return { ...s, targetText: text, cps, cpsStatus,
            previewAudioUrl: undefined, previewAudioId: undefined, previewSpeedFactor: undefined };
        } else {
          return { ...s, sourceText: text, text, cps, cpsStatus };
        }
      });
      const nextOptions = state.transcriptOptions ? {
        ...state.transcriptOptions,
        [state.selectedSegmentOption]: next,
      } : state.transcriptOptions;
      return {
        segments: next,
        transcriptOptions: nextOptions,
        dubbingStatus: isTarget && state.dubbingStatus === 'completed' ? 'idle' : state.dubbingStatus,
      };
    });
    get().triggerAutosave();
  },

  updateSegmentTiming: (id: number, startSec: number, endSec: number) => {
    if (isNaN(startSec) || isNaN(endSec) || startSec >= endSec) return;
    set((state: any) => {
      const next = state.segments.map((s: Segment) => {
        if (s.id !== id) return s;
        const dur = Math.max(0.1, endSec - startSec);
        const text = s.targetText || s.sourceText || '';
        const cps = Number((text.trim().length / dur).toFixed(1));
        const cpsStatus = cps <= 14.5 ? 'Optimal' : cps <= 18.0 ? 'Good' : 'Warning';
        return {
          ...s,
          startSec,
          endSec,
          startTime: formatTimecode(startSec),
          endTime: formatTimecode(endSec),
          cps,
          cpsStatus,
        };
      });
      const nextOptions = state.transcriptOptions ? {
        ...state.transcriptOptions,
        [state.selectedSegmentOption]: next,
      } : state.transcriptOptions;
      return { segments: next, transcriptOptions: nextOptions };
    });
    get().triggerAutosave();
  },

  splitSegment: (id: number, splitTime?: number) => {
    set((state: any) => {
      const idx = state.segments.findIndex((s: Segment) => s.id === id);
      if (idx === -1) return state;
      const seg = state.segments[idx];
      const t = splitTime !== undefined ? splitTime : (seg.startSec + seg.endSec) / 2;
      const clamped = Math.max(seg.startSec + 0.1, Math.min(seg.endSec - 0.1, t));

      const newId = Math.max(0, ...state.segments.map((s: Segment) => s.id)) + 1;
      const words = (seg.sourceText || '').split(' ');
      const mid = Math.max(1, Math.floor(words.length / 2));
      const text1 = words.slice(0, mid).join(' ');
      const text2 = words.slice(mid).join(' ');

      const targetWords = (seg.targetText || '').split(' ');
      const targetMid = Math.max(1, Math.floor(targetWords.length / 2));
      const target1 = targetWords.slice(0, targetMid).join(' ');
      const target2 = targetWords.slice(targetMid).join(' ');

      const s1: Segment = {
        ...seg,
        endSec: clamped,
        endTime: formatTimecode(clamped),
        sourceText: text1,
        targetText: target1,
      };

      const s2: Segment = {
        ...seg,
        id: newId,
        startSec: clamped,
        startTime: formatTimecode(clamped),
        sourceText: text2,
        targetText: target2,
      };

      const next = [...state.segments];
      next.splice(idx, 1, s1, s2);
      return { segments: next };
    });
    get().triggerAutosave();
  },

  deleteSegment: (id: number) => {
    set((state: any) => ({
      segments: state.segments.filter((s: Segment) => s.id !== id),
    }));
    get().triggerAutosave();
  },

  mergeWithNextSegment: (id: number) => {
    set((state: any) => {
      const idx = state.segments.findIndex((s: Segment) => s.id === id);
      if (idx === -1 || idx === state.segments.length - 1) return state;
      const cur = state.segments[idx];
      const next = state.segments[idx + 1];

      const merged: Segment = {
        ...cur,
        endSec: next.endSec,
        endTime: next.endTime,
        sourceText: `${cur.sourceText} ${next.sourceText}`.trim(),
        targetText: `${cur.targetText} ${next.targetText}`.trim(),
      };

      const newSegs = [...state.segments];
      newSegs.splice(idx, 2, merged);
      return { segments: newSegs };
    });
    get().triggerAutosave();
  },

  extractOcrForSegment: async (segmentId: number, roi: [number, number, number, number], interval?: { startSec: number; endSec: number }) => {
    const mediaId = get().backend?.mediaId;
    if (!mediaId) throw new Error('Upload a video before using OCR');
    const seg = get().segments.find((s: Segment) => s.id === segmentId);
    if (!seg) throw new Error('This segment is no longer available');
    const result = await requestOcrExtract({
      mediaId,
      roi,
      startSec: interval?.startSec ?? seg.startSec,
      endSec: interval?.endSec ?? seg.endSec,
      segmentId,
      language: get().languages?.source?.code,
    });
    if (!result.ok) throw new Error('OCR extraction failed');
    return { text: result.text || '', confidence: result.confidence, entries: result.entries || [] };
  },

  applyOcrText: async (segmentId: number, text: string, confidence?: number) => {
    const recognized = text.trim();
    if (!recognized) return;
    if (get().segments.find((segment: Segment) => segment.id === segmentId)?.sourceText === recognized) return;
    set((state: any) => {
      const segments = state.segments.map((segment: Segment) => {
        if (segment.id !== segmentId || segment.sourceText === recognized) return segment;
        return {
          ...segment, sourceText: recognized, text: recognized, targetText: '',
          sourceType: 'ocr',
          hasOcrDiff: true, ocrSlideText: recognized, ocrConfidence: confidence,
          previewAudioUrl: undefined, previewAudioId: undefined,
          previewSpeedFactor: undefined, previewVoice: undefined,
        };
      });
      return {
        segments,
        dubbingStatus: 'idle',
        assembledDubUrl: null,
        stage3Baseline: null,
        transcriptOptions: state.transcriptOptions ? {
          ...state.transcriptOptions,
          [state.selectedSegmentOption]: segments,
        } : null,
      };
    });
    await get().triggerAutosave(true);
  },

  replaceSegmentWithOcrEntries: async (segmentId: number, ocrEntries: OcrSubtitleEntry[]) => {
    if (!ocrEntries || ocrEntries.length === 0) return;
    const currentState = get();
    const idx = currentState.segments.findIndex((s: Segment) => s.id === segmentId);
    if (idx === -1) return;
    const original = currentState.segments[idx];

    const newSegments: Segment[] = ocrEntries.map((entry, entryIdx) => {
      const dur = Math.max(0.1, entry.endSec - entry.startSec);
      const text = entry.text.trim();
      const cps = Number((text.length / dur).toFixed(1));
      const cpsStatus = cps <= 14.5 ? 'Optimal' : cps <= 18.0 ? 'Good' : 'Warning';
      const startTime = entry.startTime || formatTimecode(entry.startSec);
      const endTime = entry.endTime || formatTimecode(entry.endSec);

      return {
        ...original,
        id: original.id + entryIdx,
        speakerId: original.speakerId || 'speaker-1',
        speakerName: original.speakerName || 'Speaker 1',
        speakerCode: original.speakerCode,
        speakerLabel: original.speakerLabel || original.speakerName,
        speakerColor: original.speakerColor,
        startTime,
        endTime,
        startSec: entry.startSec,
        endSec: entry.endSec,
        sourceText: text,
        text,
        targetText: '',
        sourceType: 'ocr',
        hasOcrDiff: true,
        ocrSlideText: text,
        ocrConfidence: entry.confidence,
        cps,
        cpsStatus,
        previewAudioUrl: undefined,
        previewAudioId: undefined,
        previewSpeedFactor: undefined,
        previewVoice: undefined,
      };
    });

    const prevIdBySeg = new Map<Segment, number>();
    currentState.segments.forEach((s: Segment) => prevIdBySeg.set(s, s.id));
    const origOverride = currentState.segmentVoiceOverrides?.[segmentId] ?? original.voiceOverride;
    const newSegSet = new Set<Segment>(newSegments);

    const next = [...currentState.segments];
    next.splice(idx, 1, ...newSegments);

    next.sort((a, b) => a.startSec - b.startSec);
    const nextOverrides: Record<number, string> = {};
    const reindexed = next.map((seg, i) => {
      const newId = i + 1;
      let overrideForSeg: string | undefined;
      if (newSegSet.has(seg)) {
        overrideForSeg = origOverride;
      } else {
        const oldId = prevIdBySeg.get(seg);
        overrideForSeg = (oldId != null ? currentState.segmentVoiceOverrides?.[oldId] : undefined) ?? seg.voiceOverride;
      }
      if (overrideForSeg) {
        nextOverrides[newId] = overrideForSeg;
        return { ...seg, id: newId, voiceOverride: overrideForSeg };
      }
      const { voiceOverride: _removed, ...rest } = seg;
      return { ...rest, id: newId };
    });

    const activeSeg = reindexed.find(
      (s) => s.startSec === newSegments[0].startSec && s.sourceText === newSegments[0].sourceText
    ) || reindexed[Math.min(idx, reindexed.length - 1)];

    set((state: any) => ({
      segments: reindexed,
      segmentVoiceOverrides: nextOverrides,
      activeSegmentId: activeSeg?.id || 1,
      dubbingStatus: 'idle',
      assembledDubUrl: null,
      stage3Baseline: null,
      transcriptOptions: state.transcriptOptions ? {
        ...state.transcriptOptions,
        [state.selectedSegmentOption]: reindexed,
      } : null,
    }));

    await get().triggerAutosave(true);
  },

  runBatchTranslation: async (sourceLang?: string, targetLang?: string) => {
    const targetProjectId = get().activeProjectId;
    const baseSegments: Segment[] = get().segments || [];
    const baseOptions = get().transcriptOptions;
    const baseOptionKey = get().selectedSegmentOption;

    if (sourceLang && get().languages?.source?.code !== sourceLang) {
      const avail = get().backend?.options?.languages || [];
      const opt = avail.find((l: any) => l.code === sourceLang);
      get().updateSourceLanguage(sourceLang, opt?.name || sourceLang);
    }
    if (targetLang && get().languages?.target?.code !== targetLang) {
      const avail = get().backend?.options?.languages || [];
      const opt = avail.find((l: any) => l.code === targetLang);
      get().updateTargetLanguage(targetLang, opt?.name || targetLang);
    }

    const curSource = sourceLang || get().languages?.source?.code || 'zh-cn';
    const curTarget = targetLang || get().languages?.target?.code || 'vi';

    set({
      translationModal: {
        active: true,
        status: 'translating',
        progress: 25,
        message: `Translating transcript from ${curSource} to ${curTarget}…`,
        error: null,
      },
    });

    try {
      const req: any = buildTranslationRequest(get());
      if (sourceLang) req.sourceLanguage = sourceLang;
      if (targetLang) req.targetLanguage = targetLang;
      if (targetProjectId) req.projectId = targetProjectId;
      if (get().backend?.mediaId) req.mediaId = get().backend.mediaId;
      const resp = await requestTranslate(req);

      if (resp.ok && resp.segments) {
        const segMap = new Map<number, string>();
        const respById = new Map<number, any>();
        resp.segments.forEach((s: any, idx: number) => {
          let trans = '';
          if (typeof s.targetText === 'string' && s.targetText.trim()) {
            trans = s.targetText;
          } else if (typeof s.text === 'string' && s.text.trim()) {
            if (!s.sourceText || s.text !== s.sourceText) {
              trans = s.text;
            }
          }
          if (!trans && (s.targetText || s.text)) {
            trans = s.targetText || s.text;
          }

          const hasExplicitId = (s.id !== undefined && s.id !== null) || (s.line !== undefined && s.line !== null);
          if (s.id !== undefined && s.id !== null) {
            segMap.set(Number(s.id), trans);
            respById.set(Number(s.id), s);
          }
          if (s.line !== undefined && s.line !== null) {
            segMap.set(Number(s.line), trans);
            if (!respById.has(Number(s.line))) {
              respById.set(Number(s.line), s);
            }
          }
          if (!hasExplicitId) {
            if (!segMap.has(idx + 1)) segMap.set(idx + 1, trans);
            if (!segMap.has(idx)) segMap.set(idx, trans);
          }
        });

        const sameProject = get().activeProjectId === targetProjectId;
        const sourceSegments: Segment[] = sameProject ? (get().segments || []) : baseSegments;
        const lengthsMatch = resp.segments.length === sourceSegments.length;
        const nextSegments = sourceSegments.map((s: Segment, index: number) => {
          const srcText = String(s.sourceText ?? s.text ?? '').trim();
          const posSeg = resp.segments[index];
          const posMatchesId =
            posSeg &&
            (posSeg.id === undefined || posSeg.id === null || Number(posSeg.id) === Number(s.id)) &&
            (posSeg.line === undefined || posSeg.line === null || Number(posSeg.line) === Number(s.id));
          const respSeg = respById.get(Number(s.id)) || (lengthsMatch && posMatchesId ? posSeg : undefined);
          let translated = '';
          if (!srcText) {
            translated = (respSeg && typeof respSeg.targetText === 'string' ? respSeg.targetText.trim() : '') || (s.targetText || '').trim();
          } else if (respSeg && typeof respSeg.targetText === 'string' && respSeg.targetText.trim()) {
            translated = respSeg.targetText;
          } else if (respSeg && typeof respSeg.text === 'string' && respSeg.text.trim() && (!respSeg.sourceText || respSeg.text !== respSeg.sourceText)) {
            translated = respSeg.text;
          } else {
            translated = segMap.get(Number(s.id)) ?? (lengthsMatch ? segMap.get(index + 1) : undefined) ?? s.targetText ?? '';
          }

          const dur = Math.max(0.1, (s.endSec || 0) - (s.startSec || 0));
          const cps = Number(((translated || '').trim().length / dur).toFixed(1));
          const cpsStatus = cps <= 14.5 ? 'Optimal' : cps <= 18.0 ? 'Good' : 'Warning';

          return {
            ...s,
            targetText: translated,
            rawTranslation: translated,
            cps,
            cpsStatus,
          };
        });

        const curOptions = sameProject ? get().transcriptOptions : baseOptions;
        const curOptionKey = sameProject ? get().selectedSegmentOption : baseOptionKey;
        const nextOptions = curOptions ? {
          ...curOptions,
          [curOptionKey]: nextSegments,
        } : curOptions;

        if (sameProject) {
          set({
            segments: nextSegments,
            transcriptOptions: nextOptions,
            translationModal: {
              active: true,
              status: 'completed',
              progress: 100,
              message: 'Translation completed successfully',
              error: null,
            },
          });
          get().triggerAutosave();

          setTimeout(() => {
            if (get().translationModal.status === 'completed' && get().activeProjectId === targetProjectId) {
              get().closeTranslationModal();
            }
          }, 1200);
        } else {
          if (targetProjectId) {
            try {
              await updateProjectState(targetProjectId, {
                segments: nextSegments,
                ...(nextOptions ? { transcriptOptions: nextOptions } : {}),
              });
            } catch (saveErr) {
              console.warn('Failed to persist batch translation for previous project:', saveErr);
            }
          }
          if (get().translationModal?.status === 'translating') {
            get().closeTranslationModal();
          }
        }
      } else {
        throw new Error((resp as any)?.detail || (resp as any)?.error || (resp as any)?.message || 'Translation failed');
      }
    } catch (err: any) {
      if (get().activeProjectId === targetProjectId) {
        set({
          translationModal: {
            active: true,
            status: 'failed',
            progress: 0,
            message: 'Translation failed',
            error: err.message,
          },
        });
      } else if (get().translationModal?.status === 'translating') {
        get().closeTranslationModal();
      }
    }
  },

  translateSingleSegment: async (segmentId: number) => {
    const targetProjectId = get().activeProjectId;
    const baseSegments: Segment[] = get().segments || [];
    const baseOptions = get().transcriptOptions;
    const baseOptionKey = get().selectedSegmentOption;
    const seg = baseSegments.find((s: Segment) => s.id === segmentId);
    if (!seg) return;
    const text = seg.sourceText || seg.text || '';
    if (!text.trim()) return;

    try {
      const curSource = get().languages?.source?.code || 'zh-cn';
      const curTarget = get().languages?.target?.code || 'vi';
      const req: any = {
        segments: [{ ...seg, text, sourceText: text }],
        sourceLanguage: curSource,
        targetLanguage: curTarget,
        translateType: get().backend?.config?.translateType ?? 0,
        translationMode: get().backend?.config?.translationMode || 'srt',
      };
      if (targetProjectId) {
        req.projectId = targetProjectId;
      }
      const resp = await requestTranslate(req);
      if (resp.ok && resp.segments && resp.segments.length > 0) {
        const item = resp.segments[0];
        const trans = (item.targetText && item.targetText.trim())
          ? item.targetText
          : (item.text && (!item.sourceText || item.text !== item.sourceText))
            ? item.text
            : (item.targetText || item.text || '');
        if (trans) {
          if (get().activeProjectId === targetProjectId) {
            get().updateSegmentText(segmentId, trans, true);
          } else if (targetProjectId) {
            const nextSegments = baseSegments.map((s: Segment) => {
              if (s.id !== segmentId) return s;
              const dur = Math.max(0.1, (s.endSec || 0) - (s.startSec || 0));
              const cps = Number((trans.trim().length / dur).toFixed(1));
              const cpsStatus = cps <= 14.5 ? 'Optimal' : cps <= 18.0 ? 'Good' : 'Warning';
              return {
                ...s,
                targetText: trans,
                cps,
                cpsStatus,
                previewAudioUrl: undefined,
                previewAudioId: undefined,
                previewSpeedFactor: undefined,
              };
            });
            const nextOptions = baseOptions ? {
              ...baseOptions,
              [baseOptionKey]: nextSegments,
            } : baseOptions;
            try {
              await updateProjectState(targetProjectId, {
                segments: nextSegments,
                ...(nextOptions ? { transcriptOptions: nextOptions } : {}),
              });
            } catch (saveErr) {
              console.warn('Failed to persist single segment translation for previous project:', saveErr);
            }
          }
        }
      }
    } catch (err) {
      console.warn('Single segment translation failed:', err);
    }
  },

  closeTranslationModal: () => {
    set((state: any) => ({
      translationModal: { ...state.translationModal, active: false },
    }));
  },
});
