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
} from '../api/voices';

const SPEAKER_COLORS = ['amber', 'secondary', 'emerald', 'sky', 'indigo', 'purple', 'rose'];

export interface DubbingSlice {
  speakers: Speaker[];
  speakerVoiceMap: Record<string, string>;
  segmentVoiceOverrides: Record<number, string>;
  dubbingStatus: 'idle' | 'running' | 'completed' | 'failed';
  dubbingError: string | null;
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
  enterStage4: () => Promise<void>;
}

export const createDubbingSlice: StateCreator<any, [], [], DubbingSlice> = (set, get) => ({
  autoFitVoiceSpeed: true,
  dubbingError: null,
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
    try {
      const ttsType = get().backend?.config?.ttsType ?? 2;
      const [vList, cList] = await Promise.all([
        fetchVoices(ttsType),
        fetchCustomVoices(ttsType).catch(() => []),
      ]);

      const normalizedVoices: VoiceOption[] = (vList || []).map((item: any) => {
        if (typeof item === 'string') {
          return {
            id: item,
            name: item,
            provider: ttsType,
            kind: item.toLowerCase() === 'clone' ? 'clone' : (item === 'No' ? 'preset' : 'preset'),
          };
        }
        return {
          id: item.id || item.name,
          name: item.name || item.id,
          provider: item.provider ?? ttsType,
          kind: item.kind ?? 'preset',
          sampleUrl: item.sampleUrl,
        };
      });

      set({ voices: normalizedVoices, customVoices: cList || [] });
    } catch (err) {
      console.warn('Failed to load voices:', err);
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
      map[idStr] = voiceId;
      return {
        speakerVoiceMap: map,
        dubbingStatus: state.dubbingStatus === 'completed' ? 'idle' : state.dubbingStatus,
      };
    });
    get().triggerAutosave();
  },

  setSegmentVoiceOverride: (segmentId: number, voiceId: string) => {
    if (isNaN(segmentId)) return;
    set((state: any) => {
      const overrides = { ...(state.segmentVoiceOverrides || {}) };
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
      return {
        segmentVoiceOverrides: overrides,
        segments: segs,
        dubbingStatus: state.dubbingStatus === 'completed' ? 'idle' : state.dubbingStatus,
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
        dubbingStatus: state.dubbingStatus === 'completed' ? 'idle' : state.dubbingStatus,
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
    if (!segment) return 'default';
    const overrides = get().segmentVoiceOverrides || {};
    if (overrides[segment.id]) return overrides[segment.id];
    if (segment.voiceOverride) return segment.voiceOverride;
    const map = get().speakerVoiceMap || {};
    const rawSpkId = segment.speakerId ?? (segment as any).speakerLabel ?? (segment as any).speakerName ?? 'spk_1';
    const spkId = String(rawSpkId).trim();
    if (spkId && map[spkId]) return map[spkId];
    if (segment.speakerId != null && map[String(segment.speakerId)]) return map[String(segment.speakerId)];
    const voices = get().voices || [];
    return voices[0]?.id || 'default';
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
      return { segments: segs };
    });
    get().triggerAutosave();
  },

  runFullDubbing: async () => {
    let targetProjectId = get().activeProjectId;
    try {
      if (!targetProjectId && get().createNewProject) {
        const projName = get().project?.filename || 'Untitled Video Project';
        const mediaId = get().backend?.mediaId;
        const dur = get().project?.durationSec;
        targetProjectId = await get().createNewProject(projName, mediaId, dur);
      }
      if (!targetProjectId) {
        targetProjectId = get().activeProjectId;
      }

      if (get().activeProjectId === targetProjectId) {
        set({ dubbingStatus: 'running', dubbingError: null });
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

      const segments: Segment[] = get().segments || [];
      const ttsType = get().backend?.config?.ttsType ?? 2;
      const lang = get().languages?.target?.code || 'vi';
      const speed = get().tuning?.pace;
      const autoFit = get().autoFitVoiceSpeed ?? true;
      const maxRate = get().maxSpeedRate ?? 1.35;
      const totalDur = get().project?.durationSec || 0;

      let completedCount = 0;
      const totalCount = segments.length;

      const updated = await Promise.all(
        segments.map(async (seg, idx) => {
          const activeVoice = get().getResolvedVoiceForSegment(seg);
          const text = seg.targetText || seg.sourceText || '';
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

          const res = await previewTTS({
              text,
              voice: activeVoice,
              provider: ttsType,
              language: lang,
              speed,
              segment_id: seg.id,
              force_refresh: true,
              auto_speed: autoFit,
              slot_duration_s: slotDur,
              max_speed_rate: maxRate,
            });
          if (!(res.audio_url || res.preview_url)) {
            throw new Error(`No preview audio was returned for segment ${seg.id}`);
          }

          completedCount++;
          if (targetProjectId) {
            const pct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 100;
            set((s: any) => ({
              projectJobStatuses: {
                ...s.projectJobStatuses,
                [targetProjectId!]: {
                  status: 'running',
                  progress: pct,
                  message: `Creating voice (${completedCount}/${totalCount})…`,
                  stage: 'voice',
                },
              },
            }));
          }

          return {
              ...seg,
              previewAudioUrl: res.preview_url || res.audio_url,
              previewAudioId: res.id || res.preview_id,
              previewSpeedFactor: res.applied_speed,
              previewVoice: activeVoice,
          };
        })
      );

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
