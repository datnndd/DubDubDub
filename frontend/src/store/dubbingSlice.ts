import { StateCreator } from 'zustand';
import type { Speaker, VoiceOption, DubbingTuning } from '../types/dubbing';
import type { Segment } from '../types/segment';
import { fetchVoices } from '../api/settings';
import { assembleProjectDubbing, fetchProject, updateProjectState } from '../api/projects';
import {
  type CustomVoice,
  type CreateVoicePayload,
  fetchCustomVoices,
  createCustomVoice as apiCreateCustomVoice,
  updateCustomVoice as apiUpdateCustomVoice,
  deleteCustomVoice as apiDeleteCustomVoice,
  bulkDeleteCustomVoices as apiBulkDeleteCustomVoices,
  previewTTS,
  batchTTS,
  cancelBatchTTS,
} from '../api/voices';

const SPEAKER_COLORS = ['amber', 'secondary', 'emerald', 'sky', 'indigo', 'purple', 'rose'];
const activeDubbingControllers = new Map<string, AbortController>();
const activeDubbingRunTokens = new Map<string, symbol>();
const activeBatchIds = new Map<string, string>();

export function pickDefaultVoice(
  voices: VoiceOption[] = [],
  customVoices: Array<{ id: string; kind?: string }> = []
): string {
  const isUsable = (v: { id?: string; kind?: string } | undefined | null) => {
    const id = String(v?.id || '').trim();
    if (!id) return false;
    if (id === 'No' || id.toLowerCase() === 'clone') return false;
    if (v?.kind === 'system') return false;
    return true;
  };
  const validPreset = voices.find(isUsable);
  if (validPreset) return validPreset.id;
  const validCustom = customVoices.find(isUsable);
  return validCustom?.id || '';
}

export interface DubbingSlice {
  speakers: Speaker[];
  speakerVoiceMap: Record<string, string>;
  segmentVoiceOverrides: Record<number, string>;
  dubbingStatus: 'idle' | 'running' | 'completed' | 'failed';
  dubbingError: string | null;
  dubbingErrors: Record<number, string>;
  failedSegmentIds: number[];
  voicesLoading: boolean;
  assemblingDubbing: boolean;
  assembledDubUrl: string | null;
  tuning: DubbingTuning;
  voices: VoiceOption[];
  customVoices: CustomVoice[];
  activePreviewVoiceId: string | null;
  isCreateVoiceModalOpen: boolean;
  isVoiceManagerDrawerOpen: boolean;

  loadVoices: () => Promise<void>;
  loadCustomVoices: (provider?: number) => Promise<void>;
  createVoice: (payload: CreateVoicePayload) => Promise<CustomVoice>;
  updateVoice: (id: string, updates: Partial<CustomVoice>) => Promise<void>;
  deleteVoice: (id: string, hard?: boolean) => Promise<void>;
  deleteVoices: (ids: string[], hard?: boolean) => Promise<void>;
  setCreateVoiceModalOpen: (open: boolean) => void;
  setVoiceManagerDrawerOpen: (open: boolean) => void;
  setActivePreviewVoiceId: (id: string | null) => void;
  setDubbingStatus: (status: 'idle' | 'running' | 'completed' | 'failed') => void;

  setSpeakerVoice: (speakerId: string, voiceId: string) => void;
  setSegmentVoiceOverride: (segmentId: number, voiceId: string) => void;
  clearSegmentVoiceOverride: (segmentId: number) => void;
  updateTuning: (key: keyof DubbingTuning, value: any) => void;
  getDistinctSpeakers: (customSegments?: Segment[], customSpeakers?: Speaker[]) => Speaker[];
  getResolvedVoiceForSegment: (segment: Segment) => string;
  updateSegmentVoicePreview: (segmentId: number, previewUrl: string, previewId?: string, voice?: string, appliedSpeed?: number) => void;
  autoFitVoiceSpeed: boolean;
  maxSpeedRate: number;
  setAutoFitVoiceSpeed: (enabled: boolean) => void;
  setMaxSpeedRate: (rate: number) => void;
  runFullDubbing: () => Promise<void>;
  cancelDubbing: () => void;
  enterStage4: () => Promise<void>;
}

export const createDubbingSlice: StateCreator<any, [], [], DubbingSlice> = (set, get) => ({
  autoFitVoiceSpeed: true,
  dubbingError: null,
  dubbingErrors: {},
  failedSegmentIds: [],
  voicesLoading: false,
  assemblingDubbing: false,
  assembledDubUrl: null,
  maxSpeedRate: 1.35,
  setAutoFitVoiceSpeed: (enabled: boolean) => {
    set({ autoFitVoiceSpeed: enabled });
    get().triggerAutosave?.();
  },
  setMaxSpeedRate: (rate: number) => {
    set({ maxSpeedRate: rate });
    get().triggerAutosave?.();
  },
  speakers: [
    {
      id: "spk_1",
      code: "AC",
      name: "Alex Carter",
      role: "Keynote Speaker",
      preset: "Studio Warmth v4.2",
      clarity: "98%",
      coverage: "86%",
      color: "amber",
    },
    {
      id: "spk_2",
      code: "ER",
      name: "Elena Rostova",
      role: "Guest / Interviewer",
      preset: "Crisp Broadcast",
      clarity: "99%",
      coverage: "14%",
      color: "secondary",
    },
  ],
  speakerVoiceMap: {},
  segmentVoiceOverrides: {},
  dubbingStatus: 'idle',
  tuning: {
    pace: 1.0,
    timbreWarmth: 62,
    ducking: "85/15",
  },
  voices: [],
  customVoices: [],
  activePreviewVoiceId: null,
  isCreateVoiceModalOpen: false,
  isVoiceManagerDrawerOpen: false,

  setDubbingStatus: (status: 'idle' | 'running' | 'completed' | 'failed') => set({ dubbingStatus: status }),

  loadVoices: async () => {
    set({ voicesLoading: true });
    try {
      const ttsType = get().backend?.config?.ttsType ?? 2;
      const [vList, cList] = await Promise.all([
        fetchVoices(ttsType),
        fetchCustomVoices(ttsType).catch(() => []),
      ]);

      const normalizedVoices: VoiceOption[] = (vList || [])
        .map((item: any): VoiceOption | null => {
          if (typeof item === 'string') {
            return {
              id: item,
              name: item,
              provider: ttsType,
              kind: item.toLowerCase() === 'clone' ? 'clone' : 'preset',
            };
          }
          if (Array.isArray(item) && item.length >= 1) {
            const id = String(item[0] ?? '').trim();
            const name = String((item.length >= 2 ? item[1] : item[0]) ?? id).trim() || id;
            if (!id) return null;
            return {
              id,
              name,
              provider: ttsType,
              kind: id.toLowerCase() === 'clone' ? 'clone' : 'preset',
            };
          }
          if (item && typeof item === 'object') {
            const id = String(item.id ?? item.name ?? '').trim();
            const name = String(item.name ?? item.id ?? '').trim();
            if (!id && !name) return null;
            return {
              id: id || name,
              name: name || id,
              provider: item.provider ?? ttsType,
              kind: item.kind ?? 'preset',
              sampleUrl: item.sampleUrl,
            };
          }
          return null;
        })
        .filter((v): v is VoiceOption => Boolean(v));

      set({ voices: normalizedVoices, customVoices: cList || [] });
    } catch (err) {
      console.warn('Failed to load voices:', err);
    } finally {
      set({ voicesLoading: false });
    }
  },

  loadCustomVoices: async (provider?: number) => {
    try {
      const cList = await fetchCustomVoices(provider);
      set({ customVoices: cList || [] });
    } catch (err) {
      console.warn('Failed to load custom voices:', err);
    }
  },

  createVoice: async (payload: CreateVoicePayload) => {
    const newVoice = await apiCreateCustomVoice(payload);
    await get().loadVoices();
    await get().loadCustomVoices();
    return newVoice;
  },

  updateVoice: async (id: string, updates: Partial<CustomVoice>) => {
    await apiUpdateCustomVoice(id, updates);
    await get().loadVoices();
  },

  deleteVoice: async (id: string, hard: boolean = false) => {
    await apiDeleteCustomVoice(id, hard);
    await get().loadVoices();
  },

  deleteVoices: async (ids: string[], hard: boolean = false) => {
    if (!ids || ids.length === 0) return;
    await apiBulkDeleteCustomVoices(ids, hard);
    await get().loadVoices();
  },

  setCreateVoiceModalOpen: (open: boolean) => set({ isCreateVoiceModalOpen: open }),
  setVoiceManagerDrawerOpen: (open: boolean) => set({ isVoiceManagerDrawerOpen: open }),
  setActivePreviewVoiceId: (id: string | null) => set({ activePreviewVoiceId: id }),

  setSpeakerVoice: (speakerId: string, voiceId: string) => {
    const idStr = String(speakerId ?? '').trim();
    if (!idStr || idStr === '__proto__' || idStr === 'constructor' || idStr === 'prototype') return;
    set((state: any) => {
      const map = { ...(state.speakerVoiceMap || {}) };
      const oldVoice = map[idStr];
      map[idStr] = voiceId;
      if (oldVoice === voiceId) {
        return { speakerVoiceMap: map };
      }
      return {
        speakerVoiceMap: map,
        dubbingStatus: 'idle',
        assembledDubUrl: null,
      };
    });
    get().triggerAutosave();
  },

  setSegmentVoiceOverride: (segmentId: number, voiceId: string) => {
    if (isNaN(segmentId)) return;
    set((state: any) => {
      const overrides = { ...(state.segmentVoiceOverrides || {}) };
      const oldOverride = overrides[segmentId];
      if (!voiceId) {
        delete overrides[segmentId];
      } else {
        overrides[segmentId] = voiceId;
      }
      const segs = (state.segments || []).map((seg: Segment) => {
        if (seg.id === segmentId) {
          return { ...seg, voiceOverride: voiceId || undefined };
        }
        return seg;
      });
      if (oldOverride === voiceId) {
        return { segmentVoiceOverrides: overrides, segments: segs };
      }
      return {
        segmentVoiceOverrides: overrides,
        segments: segs,
        dubbingStatus: 'idle',
        assembledDubUrl: null,
      };
    });
    get().triggerAutosave();
  },

  clearSegmentVoiceOverride: (segmentId: number) => {
    set((state: any) => {
      const overrides = { ...(state.segmentVoiceOverrides || {}) };
      delete overrides[segmentId];
      const segs = (state.segments || []).map((seg: Segment) => {
        if (seg.id === segmentId) {
          const updated = { ...seg };
          delete updated.voiceOverride;
          return updated;
        }
        return seg;
      });
      return {
        segmentVoiceOverrides: overrides,
        segments: segs,
        dubbingStatus: 'idle',
        assembledDubUrl: null,
      };
    });
    get().triggerAutosave();
  },

  updateTuning: (key: keyof DubbingTuning, value: any) => {
    set((state: any) => ({
      tuning: {
        ...state.tuning,
        [key]: value,
      },
    }));
    get().triggerAutosave();
  },

  getDistinctSpeakers: (customSegments?: Segment[], customSpeakers?: Speaker[]): Speaker[] => {
    const segments: Segment[] = customSegments ?? get().segments ?? [];
    if (!segments || segments.length === 0) {
      return customSpeakers ?? get().speakers ?? [];
    }

    const seen = new Map<string, Speaker>();
    const metaLookup = new Map<string, Speaker>();
    (customSpeakers ?? get().speakers ?? []).forEach((s: any) => {
      if (s?.id) metaLookup.set(String(s.id), s);
    });

    let colorIdx = 0;

    for (const seg of segments) {
      const rawSpkId = seg.speakerId ?? (seg as any).speakerLabel ?? (seg as any).speakerName ?? 'spk_1';
      const spkId = String(rawSpkId).trim();
      if (!spkId) continue;
      if (!seen.has(spkId)) {
        const meta = metaLookup.get(spkId);
        const name = seg.speakerName || meta?.name || (seg.speakerLabel ? `Speaker ${seg.speakerLabel}` : `Speaker ${seen.size + 1}`);
        const words = spkId.split(/[\s_]+/);
        const code = seg.speakerCode || meta?.code || (words.length > 1
          ? `${words[0][0]}${words[1][0]}`.toUpperCase()
          : spkId.slice(0, 2).toUpperCase());
        const color = seg.speakerColor || meta?.color || SPEAKER_COLORS[colorIdx % SPEAKER_COLORS.length];

        seen.set(spkId, {
          id: spkId,
          code,
          name,
          role: meta?.role || 'Speaker',
          color,
        });
        colorIdx++;
      }
    }

    return Array.from(seen.values());
  },

  getResolvedVoiceForSegment: (segment: Segment): string => {
    if (!segment) return '';
    const overrides = get().segmentVoiceOverrides || {};
    if (overrides[segment.id]) return overrides[segment.id];
    if (segment.voiceOverride) return segment.voiceOverride;
    const map = get().speakerVoiceMap || {};
    const rawSpkId = segment.speakerId ?? (segment as any).speakerLabel ?? (segment as any).speakerName ?? 'spk_1';
    const spkId = String(rawSpkId).trim();
    if (spkId && map[spkId]) return map[spkId];
    if (segment.speakerId != null && map[String(segment.speakerId)]) return map[String(segment.speakerId)];
    return pickDefaultVoice(get().voices || [], get().customVoices || []);
  },

  updateSegmentVoicePreview: (segmentId: number, previewUrl: string, previewId?: string, voice?: string, appliedSpeed?: number) => {
    set((state: any) => {
      const segs = (state.segments || []).map((seg: Segment) => {
        if (seg.id === segmentId) {
          return {
            ...seg,
            previewAudioUrl: previewUrl,
            previewAudioId: previewId,
            previewSpeedFactor: appliedSpeed,
            previewVoice: voice || seg.previewVoice,
          };
        }
        return seg;
      });
      const nextErrors = { ...(state.dubbingErrors || {}) };
      delete nextErrors[segmentId];
      const nextFailedIds = (state.failedSegmentIds || []).filter((id: number) => id !== segmentId);
      return { segments: segs, dubbingErrors: nextErrors, failedSegmentIds: nextFailedIds };
    });
    get().triggerAutosave();
  },

  cancelDubbing: () => {
    const activeId = get().activeProjectId || '__default__';
    const batchId = activeBatchIds.get(activeId);
    if (batchId) {
      activeBatchIds.delete(activeId);
      void cancelBatchTTS(batchId).catch(() => {});
    }
    const controller = activeDubbingControllers.get(activeId);
    if (controller) {
      controller.abort();
      activeDubbingControllers.delete(activeId);
      activeDubbingRunTokens.delete(activeId);
    }
    if (activeId !== '__default__' && !controller) {
      const defaultBatchId = activeBatchIds.get('__default__');
      if (defaultBatchId) {
        activeBatchIds.delete('__default__');
        void cancelBatchTTS(defaultBatchId).catch(() => {});
      }
      const defaultController = activeDubbingControllers.get('__default__');
      if (defaultController) {
        defaultController.abort();
        activeDubbingControllers.delete('__default__');
      }
      activeDubbingRunTokens.delete('__default__');
    }
    set((s: any) => ({
      dubbingStatus: 'idle',
      jobStatus: s.jobStage === 'voice' ? 'idle' : s.jobStatus,
      jobMessage: s.jobStage === 'voice' ? 'Voice generation cancelled' : s.jobMessage,
      ...(activeId !== '__default__' && s.projectJobStatuses?.[activeId]?.stage === 'voice' ? {
        projectJobStatuses: {
          ...s.projectJobStatuses,
          [activeId]: {
            ...s.projectJobStatuses[activeId],
            status: 'cancelled',
            message: 'Voice generation cancelled',
          },
        },
      } : {}),
    }));
  },

  runFullDubbing: async () => {
    let targetProjectId = get().activeProjectId;
    const initialKey = targetProjectId || '__default__';
    let controllerKey = initialKey;
    const existingController = activeDubbingControllers.get(controllerKey);
    if (existingController) {
      existingController.abort();
    }
    if (controllerKey !== '__default__') {
      const pendingDefaultController = activeDubbingControllers.get('__default__');
      if (pendingDefaultController) {
        pendingDefaultController.abort();
        activeDubbingControllers.delete('__default__');
        activeDubbingRunTokens.delete('__default__');
      }
      const pendingDefaultBatchId = activeBatchIds.get('__default__');
      if (pendingDefaultBatchId) {
        activeBatchIds.delete('__default__');
        void cancelBatchTTS(pendingDefaultBatchId).catch(() => {});
      }
    }
    const controller = new AbortController();
    const runToken = Symbol('dubbing-run');
    activeDubbingControllers.set(controllerKey, controller);
    activeDubbingRunTokens.set(controllerKey, runToken);
    let currentChunkBatchId: string | null = null;

    try {
      if (!targetProjectId && get().createNewProject) {
        const projName = get().project?.filename || 'Untitled Video Project';
        const mediaId = get().backend?.mediaId;
        const dur = get().project?.durationSec;
        targetProjectId = await get().createNewProject(projName, mediaId, dur, undefined, { signal: controller.signal });
      }
      if (!targetProjectId) {
        targetProjectId = get().activeProjectId;
      }
      if (controller.signal.aborted || activeDubbingRunTokens.get(initialKey) !== runToken) {
        return;
      }
      if (targetProjectId && targetProjectId !== controllerKey) {
        const prevTargetController = activeDubbingControllers.get(targetProjectId);
        if (prevTargetController && prevTargetController !== controller) {
          prevTargetController.abort();
        }
        if (activeDubbingControllers.get(initialKey) === controller) {
          activeDubbingControllers.delete(initialKey);
        }
        if (activeDubbingRunTokens.get(initialKey) === runToken) {
          activeDubbingRunTokens.delete(initialKey);
        }
        controllerKey = targetProjectId;
        activeDubbingControllers.set(controllerKey, controller);
        activeDubbingRunTokens.set(controllerKey, runToken);
      }
      if (activeDubbingRunTokens.get(controllerKey) !== runToken) {
        return;
      }

      if (get().activeProjectId === targetProjectId) {
        set({
          dubbingStatus: 'running',
          dubbingError: null,
          dubbingErrors: {},
          failedSegmentIds: [],
          jobStatus: 'running',
          jobProgress: 0,
          jobMessage: 'Creating voice…',
          jobStage: 'voice',
        });
      }

      if (targetProjectId) {
        set((s: any) => ({
          projectJobStatuses: {
            ...s.projectJobStatuses,
            [targetProjectId!]: {
              status: 'running',
              progress: 0,
              message: 'Creating voice…',
              stage: 'voice',
            },
          },
        }));
      }

      const rawSegments: Segment[] = get().segments || [];
      const segments: Segment[] = rawSegments.map((s) => ({ ...s }));
      const ttsType = get().backend?.config?.ttsType ?? 2;
      const lang = get().languages?.target?.code || 'vi';
      const speed = get().tuning?.pace;
      const autoFit = get().autoFitVoiceSpeed ?? true;
      const maxRate = get().maxSpeedRate ?? 1.35;
      const totalDur = get().project?.durationSec || 0;

      let completedCount = 0;
      let synthesizableCount = 0;
      let succeededCount = 0;
      const totalCount = segments.length;
      const updated: Segment[] = [...segments];
      const failedMap: Record<number, string> = {};
      const failedIds: number[] = [];

      const reportProgress = () => {
        if (controller.signal.aborted || activeDubbingRunTokens.get(controllerKey) !== runToken) return;
        const pct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 100;
        const msg = `Creating voice (${completedCount}/${totalCount})…`;
        if (targetProjectId) {
          set((s: any) => ({
            projectJobStatuses: {
              ...s.projectJobStatuses,
              [targetProjectId!]: {
                status: 'running',
                progress: pct,
                message: msg,
                stage: 'voice',
              },
            },
          }));
        }
        if (get().activeProjectId === targetProjectId) {
          set({ jobProgress: pct, jobMessage: msg });
        }
      };

      // Batch synthesis loop (groups by provider:voice, chunks of 12, clean fallback on 404/non-batch)
      const synthesizeSegments = async () => {
        interface PendingTask {
          idx: number;
          seg: Segment;
          activeVoice: string;
          text: string;
          slotDur: number | undefined;
          targetProvider: number;
          resolvedVoice: string;
        }

        const pendingTasks: PendingTask[] = [];
        const customVoices = get().customVoices || [];
        const hasLoadedVoices = (get().voices?.length || 0) > 0 || customVoices.length > 0;

        for (let idx = 0; idx < segments.length; idx++) {
          if (controller.signal.aborted) return;
          const seg = segments[idx];
          const activeVoice = get().getResolvedVoiceForSegment(seg);
          const text = (seg.targetText || seg.sourceText || '').trim();

          if (!text) {
            completedCount++;
            reportProgress();
            continue;
          }

          synthesizableCount++;

          // Reuse existing valid preview for the same voice
          if (seg.previewAudioUrl && seg.previewAudioId && seg.previewVoice === activeVoice) {
            succeededCount++;
            completedCount++;
            reportProgress();
            continue;
          }

          const trimmedVoice = String(activeVoice || '').trim();
          if (
            trimmedVoice === 'No' ||
            trimmedVoice.toLowerCase() === 'clone' ||
            (!trimmedVoice && hasLoadedVoices)
          ) {
            failedIds.push(seg.id);
            failedMap[seg.id] = 'No voice assigned for segment';
            completedCount++;
            reportProgress();
            continue;
          }

          const nextSeg = segments[idx + 1];
          let slotDur: number | undefined = undefined;
          if (autoFit && typeof seg.startSec === 'number' && typeof seg.endSec === 'number') {
            const rawSlot = Math.max(0.001, seg.endSec - seg.startSec);
            if (nextSeg && typeof nextSeg.startSec === 'number') {
              const slackEnd = Math.max(seg.endSec, nextSeg.startSec - 0.15);
              const clampedEnd = Math.min(Math.max(slackEnd, seg.startSec), Math.max(nextSeg.startSec, seg.endSec));
              slotDur = Math.max(0.001, clampedEnd - seg.startSec);
            } else if (totalDur > 0) {
              slotDur = Math.max(0.001, Math.max(seg.endSec, totalDur) - seg.startSec);
            } else {
              slotDur = rawSlot;
            }
          }

          const matchedCustom = customVoices.find(
            (cv: any) => cv.id === activeVoice || cv.name === activeVoice || `Custom: ${cv.name}` === activeVoice
          );
          const targetProvider = matchedCustom?.provider ?? ttsType;
          const resolvedVoice = (matchedCustom ? matchedCustom.id : trimmedVoice) || 'default';

          pendingTasks.push({
            idx,
            seg,
            activeVoice,
            text,
            slotDur,
            targetProvider,
            resolvedVoice,
          });
        }

        if (pendingTasks.length === 0) return;

        const runPerSegmentFallback = async (tasksToRun: PendingTask[]) => {
          const isCloud = tasksToRun.some((t) => t.targetProvider === 0 || t.targetProvider === 3);
          const concurrency = isCloud ? 4 : 2;
          let cursor = 0;

          const worker = async () => {
            while (cursor < tasksToRun.length) {
              if (controller.signal.aborted || activeDubbingRunTokens.get(controllerKey) !== runToken) return;
              const task = tasksToRun[cursor++];
              const { idx, seg, activeVoice, text, slotDur, targetProvider, resolvedVoice } = task;
              try {
                const res = await previewTTS({
                  text,
                  voice: resolvedVoice,
                  provider: targetProvider,
                  language: lang,
                  speed,
                  segment_id: seg.id,
                  force_refresh: false,
                  auto_speed: autoFit,
                  slot_duration_s: slotDur,
                  max_speed_rate: maxRate,
                }, { signal: controller.signal });

                if (controller.signal.aborted || activeDubbingRunTokens.get(controllerKey) !== runToken) return;

                if (res?.audio_url || res?.preview_url) {
                  const segDone: Segment = {
                    ...seg,
                    previewAudioUrl: res.preview_url || res.audio_url,
                    previewAudioId: res.id || res.preview_id,
                    previewSpeedFactor: res.applied_speed,
                    previewVoice: activeVoice,
                  };
                  updated[idx] = segDone;
                  succeededCount++;

                  if (get().activeProjectId === targetProjectId) {
                    const currentSegs = (get().segments || []).map((s: Segment) => s.id === seg.id ? segDone : s);
                    set({ segments: currentSegs });
                  }
                } else {
                  failedIds.push(seg.id);
                  failedMap[seg.id] = 'Preview response did not return an audio URL';
                }
              } catch (segErr: any) {
                if (controller.signal.aborted || segErr?.name === 'AbortError') return;
                const errMsg = segErr instanceof Error ? segErr.message : 'TTS preview failed';
                failedIds.push(seg.id);
                failedMap[seg.id] = errMsg;
                console.warn(`Failed to synthesize preview for segment ${seg.id}:`, segErr);
              }

              completedCount++;
              reportProgress();
            }
          };

          const workerCount = Math.min(concurrency, tasksToRun.length);
          await Promise.all(Array.from({ length: Math.max(1, workerCount) }, () => worker()));
        };

        // Group by provider:voice so VieNeu sub-batches maximize max_batch_size=4 utilization
        const grouped = new Map<string, PendingTask[]>();
        for (const t of pendingTasks) {
          const gk = `${t.targetProvider}:${t.resolvedVoice}`;
          const list = grouped.get(gk);
          if (list) list.push(t);
          else grouped.set(gk, [t]);
        }

        const BATCH_CHUNK_SIZE = 12;
        const chunks: PendingTask[][] = [];
        for (const groupTasks of grouped.values()) {
          for (let i = 0; i < groupTasks.length; i += BATCH_CHUNK_SIZE) {
            chunks.push(groupTasks.slice(i, i + BATCH_CHUNK_SIZE));
          }
        }

        for (const chunk of chunks) {
          if (controller.signal.aborted || activeDubbingRunTokens.get(controllerKey) !== runToken) return;
          const chunkBatchId = `batch_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
          currentChunkBatchId = chunkBatchId;
          activeBatchIds.set(controllerKey, chunkBatchId);

          try {
            const batchRes = await batchTTS({
              batchId: chunkBatchId,
              language: lang,
              speed,
              autoSpeed: autoFit,
              maxSpeedRate: maxRate,
              segments: chunk.map((t) => ({
                segmentId: t.seg.id,
                text: t.text,
                voice: t.resolvedVoice,
                provider: t.targetProvider,
                language: lang,
                speed,
                forceRefresh: false,
                autoSpeed: autoFit,
                slotDuration: t.slotDur,
                maxSpeedRate: maxRate,
              })),
            }, { signal: controller.signal });

            if (activeBatchIds.get(controllerKey) === chunkBatchId) {
              activeBatchIds.delete(controllerKey);
            }
            if (currentChunkBatchId === chunkBatchId) {
              currentChunkBatchId = null;
            }
            if (controller.signal.aborted || activeDubbingRunTokens.get(controllerKey) !== runToken) return;

            if (!batchRes || !Array.isArray(batchRes.results)) {
              // Endpoint returned non-batch shape (e.g. legacy/generic mock) -> clean per-segment fallback
              await runPerSegmentFallback(chunk);
              continue;
            }

            const resultById = new Map<string, any>();
            for (const r of batchRes.results) {
              if (r && r.segmentId !== undefined && r.segmentId !== null) {
                resultById.set(String(r.segmentId), r);
              }
            }

            const chunkUpdates = new Map<number, Segment>();
            chunk.forEach((t, posIdx) => {
              let r = resultById.get(String(t.seg.id));
              if (!r) {
                const candidate = batchRes.results[posIdx];
                if (candidate && (candidate.segmentId === undefined || candidate.segmentId === null)) {
                  r = candidate;
                }
              }
              const audioUrl = r?.previewUrl || r?.preview_url || r?.audio_url;
              if (r && r.ok && audioUrl) {
                const segDone: Segment = {
                  ...t.seg,
                  previewAudioUrl: audioUrl,
                  previewAudioId: r.id || r.preview_id || t.seg.previewAudioId,
                  previewSpeedFactor: r.appliedSpeed ?? r.applied_speed,
                  previewVoice: t.activeVoice,
                };
                updated[t.idx] = segDone;
                chunkUpdates.set(t.seg.id, segDone);
                succeededCount++;
              } else {
                failedIds.push(t.seg.id);
                failedMap[t.seg.id] = r?.error || (r ? 'Batch TTS synthesis failed' : 'Missing batch result for segment');
              }
              completedCount++;
            });

            if (chunkUpdates.size > 0 && get().activeProjectId === targetProjectId) {
              const currentSegs = (get().segments || []).map((s: Segment) => chunkUpdates.get(s.id) ?? s);
              set({ segments: currentSegs });
            }
            reportProgress();
          } catch (batchErr: any) {
            if (activeBatchIds.get(controllerKey) === chunkBatchId) {
              activeBatchIds.delete(controllerKey);
            }
            if (currentChunkBatchId === chunkBatchId) {
              currentChunkBatchId = null;
            }
            if (controller.signal.aborted || batchErr?.name === 'AbortError' || activeDubbingRunTokens.get(controllerKey) !== runToken) {
              return;
            }
            const status = Number(batchErr?.status || 0);
            if (status === 404 || status === 405 || status === 501) {
              await runPerSegmentFallback(chunk);
              continue;
            }
            const errMsg = batchErr instanceof Error ? batchErr.message : 'Batch TTS synthesis failed';
            for (const t of chunk) {
              failedIds.push(t.seg.id);
              failedMap[t.seg.id] = errMsg;
              completedCount++;
            }
            reportProgress();
          }
        }
      };

      await synthesizeSegments();

      if (activeDubbingRunTokens.get(controllerKey) !== runToken) {
        return;
      }

      if (controller.signal.aborted) {
        if (targetProjectId) {
          set((s: any) => ({
            projectJobStatuses: {
              ...s.projectJobStatuses,
              [targetProjectId!]: {
                status: 'cancelled',
                progress: totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0,
                message: 'Voice generation cancelled',
                stage: 'voice',
              },
            },
          }));
        }
        if (get().activeProjectId === targetProjectId) {
          set({
            segments: updated,
            dubbingStatus: 'idle',
            jobStatus: 'idle',
            jobMessage: 'Voice generation cancelled',
          });
        }
        return;
      }

      if (failedIds.length > 0) {
        const errSummary = succeededCount === 0
          ? `Voice generation failed for all ${failedIds.length} segment(s).`
          : `Voice generation failed for ${failedIds.length} of ${synthesizableCount} segment(s). Retry failed segments or check provider settings.`;

        if (targetProjectId) {
          set((s: any) => ({
            projectJobStatuses: {
              ...s.projectJobStatuses,
              [targetProjectId!]: {
                status: 'failed',
                progress: totalCount > 0 ? Math.round((succeededCount / totalCount) * 100) : 0,
                message: errSummary,
                stage: 'voice',
              },
            },
          }));
        }

        if (get().activeProjectId === targetProjectId) {
          set({
            segments: updated,
            dubbingStatus: 'failed',
            jobStatus: 'failed',
            jobMessage: errSummary,
            dubbingError: errSummary,
            dubbingErrors: failedMap,
            failedSegmentIds: failedIds,
          });
          get().triggerAutosave();
        } else if (targetProjectId) {
          try {
            const bgProj = await fetchProject(targetProjectId);
            let bgState: any = (bgProj as any)?.state || null;
            if (!bgState && bgProj?.state_json) {
              try {
                bgState = typeof bgProj.state_json === 'string' ? JSON.parse(bgProj.state_json) : bgProj.state_json;
              } catch (_) {}
            }
            const mergedState = {
              ...(bgState || {}),
              segments: updated,
              dubbingStatus: 'failed',
            };
            await updateProjectState(targetProjectId, mergedState, Math.max(bgProj?.stage || 1, 3));
          } catch (_) {}
        }
        return;
      }

      if (targetProjectId) {
        set((s: any) => ({
          projectJobStatuses: {
            ...s.projectJobStatuses,
            [targetProjectId!]: {
              status: 'completed',
              progress: 100,
              message: 'Voice completed',
              stage: 'voice',
            },
          },
        }));
      }

      if (get().activeProjectId === targetProjectId) {
        set({
          segments: updated,
          stage3Baseline: updated.map((segment) => ({ ...segment })),
          dubbingStatus: 'completed',
          dubbingError: null,
          dubbingErrors: {},
          failedSegmentIds: [],
          jobStatus: 'completed',
          jobProgress: 100,
          jobMessage: 'Voice completed',
          maxUnlockedStep: Math.max(get().maxUnlockedStep || 1, 4),
        });
        get().triggerAutosave();
      } else if (targetProjectId) {
        // User switched to another video while voice generation was running in background.
        // Update background project in backend without corrupting the newly selected video's store state!
        try {
          const bgProj = await fetchProject(targetProjectId);
          let bgState: any = (bgProj as any)?.state || null;
          if (!bgState && bgProj?.state_json) {
            try {
              bgState = typeof bgProj.state_json === 'string' ? JSON.parse(bgProj.state_json) : bgProj.state_json;
            } catch (_) {}
          }
          const mergedState = {
            ...(bgState || {}),
            segments: updated,
            stage3Baseline: updated.map((segment) => ({ ...segment })),
            dubbingStatus: 'completed',
          };
          await updateProjectState(targetProjectId, mergedState, Math.max(bgProj?.stage || 1, 3));
        } catch (saveErr) {
          console.warn('Failed to update background project state after voice generation:', saveErr);
        }
      }
    } catch (err: any) {
      if (controller.signal.aborted || err?.name === 'AbortError' || activeDubbingRunTokens.get(controllerKey) !== runToken) return;
      console.error('Full dubbing failed:', err);
      if (targetProjectId) {
        set((s: any) => ({
          projectJobStatuses: {
            ...s.projectJobStatuses,
            [targetProjectId!]: {
              status: 'failed',
              progress: 0,
              message: err instanceof Error ? err.message : 'Voice dubbing failed',
              stage: 'voice',
            },
          },
        }));
      }
      if (get().activeProjectId === targetProjectId) {
        set({ dubbingStatus: 'failed', dubbingError: err instanceof Error ? err.message : 'Voice dubbing failed' });
      }
    } finally {
      if (activeDubbingControllers.get(controllerKey) === controller) {
        activeDubbingControllers.delete(controllerKey);
      }
      if (activeDubbingRunTokens.get(controllerKey) === runToken) {
        activeDubbingRunTokens.delete(controllerKey);
      }
      if (currentChunkBatchId !== null && activeBatchIds.get(controllerKey) === currentChunkBatchId) {
        activeBatchIds.delete(controllerKey);
      }
    }
  },

  enterStage4: async () => {
    const id = get().activeProjectId;
    if (get().assemblingDubbing) return;
    if (get().dubbingStatus === 'running') {
      set({ dubbingError: 'Wait for voice generation to finish before opening Stage 4' });
      return;
    }
    if (!id) {
      set({ dubbingError: 'Save this project before opening Stage 4' });
      return;
    }

    const segments: Segment[] = get().segments || [];
    const hasMismatchedOrMissingVoice = get().dubbingStatus !== 'completed' && segments.some((seg) => {
      const text = (seg.targetText || seg.sourceText || '').trim();
      if (!text) return false;
      const activeVoice = get().getResolvedVoiceForSegment(seg);
      return !seg.previewAudioId || seg.previewVoice !== activeVoice;
    });

    if (hasMismatchedOrMissingVoice) {
      await get().runFullDubbing();
      if (get().dubbingStatus !== 'completed') {
        return;
      }
    }

    set({ assemblingDubbing: true, dubbingError: null });
    try {
      const result = await assembleProjectDubbing(id, get().segments || []);
      if (get().activeProjectId === id) {
        set({ assembledDubUrl: `${result.audio_url}?v=${Date.now()}` });
        await get().setStep(4);
      }
    } catch (error) {
      if (get().activeProjectId === id) {
        set({ dubbingError: error instanceof Error ? error.message : 'Could not assemble dubbing audio' });
      }
    } finally {
      if (get().activeProjectId === id) {
        set({ assemblingDubbing: false });
      }
    }
  },
});
