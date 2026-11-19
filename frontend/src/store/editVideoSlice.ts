import { StateCreator } from 'zustand';
import type { AudioMixSettings, SubtitleStyleSettings, EditVideoState } from '../types/editor';
import { uploadEditAsset, uploadMedia } from '../api/media';
import { requestRender } from '../api/stages';

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

export const DEFAULT_SUBTITLE_STYLES: SubtitleStyleSettings = {
  preset: 'clean', fontFamily: 'Arial', fontSize: 22, color: '#FFFFFF',
  outlineColor: '#000000', outlineWidth: 2, shadowColor: 'rgba(0,0,0,.75)',
  shadowSize: 2, fontWeight: 'normal', fontStyle: 'normal', opacity: 100,
  aiLipSync: true, deReverb: true, faceRetouch: false, superRes4K: true,
};

export const DEFAULT_EDIT_VIDEO: EditVideoState = {
  audioMix: { ...DEFAULT_AUDIO_MIX }, backgroundAudio: null, thumbnail: null,
  exporting: false, error: null, activeTab: 'audio', isCapCutModalOpen: false,
};

export function buildRenderRequest(state: any) {
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
    },
  };
}

export interface EditVideoSlice {
  subtitleStyles: SubtitleStyleSettings;
  editVideo: EditVideoState;
  muteCache: Record<string, number>;

  setAudioMix: (channel: 'original' | 'dubbed' | 'background', val: any) => void;
  toggleAudioMute: (channel: 'original' | 'dubbed' | 'background') => void;
  setBackgroundAudio: (file: File | null) => Promise<void>;
  removeBackgroundAudio: () => void;
  setThumbnail: (file: File | null) => Promise<void>;
  removeThumbnail: () => void;
  updateSubtitleStyle: (key: keyof SubtitleStyleSettings, val: any) => void;
  setActiveInspectorTab: (tab: 'audio' | 'subtitles' | 'assets') => void;
  setCapCutModalOpen: (open: boolean) => void;
  exportEditedVideo: () => Promise<void>;
}

export const createEditVideoSlice: StateCreator<any, [], [], EditVideoSlice> = (set, get) => ({
  subtitleStyles: { ...DEFAULT_SUBTITLE_STYLES },
  editVideo: { ...DEFAULT_EDIT_VIDEO },
  muteCache: {},

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

  setActiveInspectorTab: (tab: 'audio' | 'subtitles' | 'assets') => {
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
