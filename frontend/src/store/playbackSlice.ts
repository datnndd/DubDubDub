import { StateCreator } from 'zustand';

export function formatTimecode(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00.000';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

export interface PlaybackSlice {
  playback: {
    currentTime: number;
    formattedTime: string;
    duration: number;
    isPlaying: boolean;
    playbackSpeed: number;
    audioChannel: 'orig' | 'dub';
    stopAtTime: number | null;
    seekRequest: { time: number; play?: boolean; stopAt?: number | null; timestamp: number } | null;
  };
  seek: (seconds: number) => void;
  seekAndPlay: (startSec: number, endSec?: number, segmentId?: number | string) => void;
  setPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setAudioChannel: (channel: 'orig' | 'dub') => void;
  setStopAtTime: (stopAtTime: number | null) => void;
  updatePlaybackTime: (currentTime: number, duration?: number) => void;
}

export const createPlaybackSlice: StateCreator<any, [], [], PlaybackSlice> = (set, get) => ({
  playback: {
    currentTime: 0,
    formattedTime: '00:00.000',
    duration: 0,
    isPlaying: false,
    playbackSpeed: 1.0,
    audioChannel: 'dub',
    stopAtTime: null,
    seekRequest: null,
  },

  seek: (seconds: number) => {
    const dur = get().playback.duration || get().project?.durationSec || 0;
    const clamped = Math.max(0, dur > 0 ? Math.min(seconds, dur) : seconds);
    set((state: any) => ({
      playback: {
        ...state.playback,
        currentTime: clamped,
        formattedTime: formatTimecode(clamped),
        stopAtTime: null,
        seekRequest: {
          time: clamped,
          play: false,
          stopAt: null,
          timestamp: Date.now(),
        },
      },
    }));

    // Find and activate segment corresponding to clamped time (clear in silent gaps)
    const segments = get().segments || [];
    const active = segments.find((s: any) => clamped >= s.startSec && clamped <= s.endSec);
    const nextActiveId = active ? active.id : null;
    if (nextActiveId !== get().activeSegmentId) {
      set({ activeSegmentId: nextActiveId });
    }
  },

  seekAndPlay: (startSec: number, endSec?: number, segmentId?: number | string) => {
    const dur = get().playback.duration || get().project?.durationSec || 0;
    const clampedStart = Math.max(0, dur > 0 ? Math.min(startSec, dur) : startSec);
    const stopAt = typeof endSec === 'number' && endSec > clampedStart ? endSec : null;

    set((state: any) => ({
      activeSegmentId: segmentId !== undefined ? segmentId : state.activeSegmentId,
      playback: {
        ...state.playback,
        currentTime: clampedStart,
        formattedTime: formatTimecode(clampedStart),
        isPlaying: true,
        stopAtTime: stopAt,
        seekRequest: {
          time: clampedStart,
          play: true,
          stopAt: stopAt,
          timestamp: Date.now(),
        },
      },
    }));
  },

  setPlaying: (isPlaying: boolean) => {
    set((state: any) => ({
      playback: { ...state.playback, isPlaying },
    }));
  },

  setPlaybackSpeed: (playbackSpeed: number) => {
    set((state: any) => ({
      playback: { ...state.playback, playbackSpeed },
    }));
  },

  setAudioChannel: (audioChannel: 'orig' | 'dub') => {
    set((state: any) => ({
      playback: { ...state.playback, audioChannel },
    }));
  },

  setStopAtTime: (stopAtTime: number | null) => {
    set((state: any) => ({
      playback: {
        ...state.playback,
        stopAtTime,
      },
    }));
  },

  updatePlaybackTime: (currentTime: number, duration?: number) => {
    const stopAt = get().playback.stopAtTime;
    const shouldStop = stopAt !== null && stopAt !== undefined && currentTime >= stopAt;

    set((state: any) => {
      const dur = duration !== undefined ? duration : state.playback.duration;
      return {
        playback: {
          ...state.playback,
          currentTime,
          formattedTime: formatTimecode(currentTime),
          duration: dur,
          isPlaying: shouldStop ? false : state.playback.isPlaying,
          stopAtTime: shouldStop ? null : state.playback.stopAtTime,
        },
      };
    });

    const segments = get().segments || [];
    const active = segments.find((s: any) => currentTime >= s.startSec && currentTime <= s.endSec);
    const nextActiveId = active ? active.id : null;
    if (nextActiveId !== get().activeSegmentId) {
      set({ activeSegmentId: nextActiveId });
    }
  },
});
