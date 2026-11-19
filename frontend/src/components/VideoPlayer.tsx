import React, { useRef, useEffect, useState } from 'react';
import { useDubDubStore } from '../store';
import { Tv as VideoIcon, Subtitles } from 'lucide-react';
import { voiceAuditionManager } from '../services/voiceAuditionManager';

interface VideoPlayerProps {
  title?: string;
  subtitleVariant?: 'capcut' | 'dual' | 'none';
  showAudioSwitcher?: boolean;
  showSubtitleToggle?: boolean;
  initialSubtitlesVisible?: boolean;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  title = 'Synchronized Video Player',
  subtitleVariant = 'capcut',
  showAudioSwitcher = false,
  showSubtitleToggle = true,
  initialSubtitlesVisible = true,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const dubAudioRef = useRef<HTMLAudioElement | null>(null);
  const [subtitlesVisible, setSubtitlesVisible] = useState(initialSubtitlesVisible);
  const [dubPlaybackError, setDubPlaybackError] = useState<string | null>(null);

  const project = useDubDubStore((s) => s.project);
  const languages = useDubDubStore((s) => s.languages);
  const playback = useDubDubStore((s) => s.playback);
  const segments = useDubDubStore((s) => s.segments);
  const subtitleStyles = useDubDubStore((s) => s.subtitleStyles);
  const editVideo = useDubDubStore((s) => s.editVideo);
  const assembledDubUrl = useDubDubStore((s) => s.assembledDubUrl);
  const currentStep = useDubDubStore((s) => s.currentStep);
  const updatePlaybackTime = useDubDubStore((s) => s.updatePlaybackTime);
  const setAudioChannel = useDubDubStore((s) => s.setAudioChannel);
  const setPlaying = useDubDubStore((s) => s.setPlaying);
  const setStopAtTime = useDubDubStore((s) => s.setStopAtTime);
  const seekRequest = useDubDubStore((s) => s.playback.seekRequest);
  const playbackSpeed = useDubDubStore((s) => s.playback.playbackSpeed);

  const hasDubPreviews = segments.some((s) => Boolean(s.previewAudioUrl));
  const currentSegment =
    segments.find((s) => playback.currentTime >= s.startSec && playback.currentTime <= s.endSec) ?? null;
  const previewUrl = project.previewUrl || '';
  const sourceCode = (languages?.source?.code ? String(languages.source.code).split('-')[0] : 'SRC').toUpperCase();
  const targetCode = (languages?.target?.code ? String(languages.target.code).split('-')[0] : 'DUB').toUpperCase();
  const isOrigChannel = playback.audioChannel === 'orig';

  // Synchronize audio mix and channel selection
  useEffect(() => {
    if (!videoRef.current) return;
    if (currentStep === 3) {
      videoRef.current.muted = !isOrigChannel;
      videoRef.current.volume = isOrigChannel ? 1 : 0;
    } else if (currentStep === 2) {
      const muteForDub = !isOrigChannel && hasDubPreviews;
      videoRef.current.muted = muteForDub;
      videoRef.current.volume = muteForDub ? 0 : 1;
    } else if (currentStep === 4 && editVideo?.audioMix) {
      if (isOrigChannel) {
        videoRef.current.muted = false;
        videoRef.current.volume = 1;
      } else {
        const origVol = Number(editVideo.audioMix.original) || 0;
        videoRef.current.volume = Math.max(0, Math.min(1, origVol / 100));
        videoRef.current.muted = origVol === 0;
      }
    }
  }, [currentStep, editVideo?.audioMix, isOrigChannel, hasDubPreviews]);

  useEffect(() => {
    const shouldMountDubAudio = currentStep === 3 || currentStep === 4 || (currentStep === 2 && hasDubPreviews);
    if (!shouldMountDubAudio || typeof Audio === 'undefined') return;
    dubAudioRef.current = new Audio();
    dubAudioRef.current.onerror = () => setDubPlaybackError('Generated dubbing audio could not be loaded. Regenerate its preview.');
    return () => {
      dubAudioRef.current?.pause();
      if (dubAudioRef.current) dubAudioRef.current.src = '';
      dubAudioRef.current = null;
      setDubPlaybackError(null);
    };
  }, [currentStep, assembledDubUrl, hasDubPreviews]);

  useEffect(() => {
    if (!dubAudioRef.current) return;
    if (isOrigChannel) {
      dubAudioRef.current.volume = 0;
      dubAudioRef.current.pause();
    } else if (currentStep === 4) {
      dubAudioRef.current.volume = Math.max(0, Math.min(1, Number(editVideo?.audioMix?.dubbed || 0) / 100));
    } else {
      dubAudioRef.current.volume = 1;
    }
  }, [currentStep, editVideo?.audioMix?.dubbed, isOrigChannel]);

  useEffect(() => {
    if (dubAudioRef.current) dubAudioRef.current.playbackRate = playbackSpeed || 1.0;
  }, [currentStep, playbackSpeed]);

  const syncDubAudio = () => {
    const video = videoRef.current;
    const audio = dubAudioRef.current;
    if (!video || !audio) return;
    const shouldPlayDub = currentStep === 3 || currentStep === 4 || (currentStep === 2 && hasDubPreviews);
    if (!shouldPlayDub || isOrigChannel) {
      audio.pause();
      return;
    }
    if (currentStep === 4) {
      if (!assembledDubUrl || video.paused || Number(editVideo?.audioMix?.dubbed || 0) <= 0) {
        audio.pause();
        return;
      }
      const url = new URL(assembledDubUrl, window.location.href).href;
      if (audio.src !== url) audio.src = url;
      if (Math.abs(audio.currentTime - video.currentTime) > 0.3) audio.currentTime = video.currentTime;
      if (audio.paused) audio.play().catch((error) => setDubPlaybackError(
        `Could not play assembled dubbing audio: ${error instanceof Error ? error.message : 'Playback was blocked'}`
      ));
      return;
    }
    const curSegments = useDubDubStore.getState().segments || [];
    const segment = curSegments.find((item, index) => {
      if (!item.previewAudioUrl) return false;
      const nextSeg = curSegments.slice(index + 1).find((s) => typeof s.startSec === 'number' && s.startSec > item.startSec);
      const effectiveEnd = nextSeg && typeof nextSeg.startSec === 'number'
        ? Math.min(Math.max(item.endSec, nextSeg.startSec - 0.03), Math.max(item.startSec, nextSeg.startSec - 0.03))
        : (item.endSec + 3.0);
      return video.currentTime >= item.startSec && video.currentTime < effectiveEnd;
    });
    if (!segment || video.paused || voiceAuditionManager.isPlayingState()) {
      audio.pause();
      return;
    }
    const url = new URL(segment.previewAudioUrl!, window.location.href).href;
    if (audio.src !== url) {
      audio.src = url;
      audio.currentTime = Math.max(0, video.currentTime - segment.startSec);
    } else if (Math.abs(audio.currentTime - (video.currentTime - segment.startSec)) > 0.3) {
      audio.currentTime = Math.max(0, video.currentTime - segment.startSec);
    }
    if (audio.paused) {
      audio.play().catch((error) => setDubPlaybackError(
        `Could not play generated dubbing audio: ${error instanceof Error ? error.message : 'Playback was blocked'}`
      ));
    }
  };

  useEffect(() => {
    syncDubAudio();
  }, [playback.audioChannel]);

  // Synchronize seek & play commands from store
  useEffect(() => {
    if (!videoRef.current || !seekRequest) return;
    const video = videoRef.current;
    video.currentTime = seekRequest.time;
    if (seekRequest.play) {
      const playPromise = video.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          console.warn('Playback error or user gesture required:', err);
        });
      }
    }
  }, [seekRequest]);

  // Synchronize playback speed
  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.playbackRate = playbackSpeed || 1.0;
  }, [playbackSpeed]);

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const curTime = videoRef.current.currentTime;
    const dur = videoRef.current.duration;

    // Check if playback should stop at the block's end boundary
    const currentStopAt = useDubDubStore.getState().playback.stopAtTime;
    if (currentStopAt !== null && currentStopAt !== undefined && curTime >= currentStopAt) {
      videoRef.current.pause();
      setPlaying(false);
      setStopAtTime(null);
    }

    updatePlaybackTime(curTime, dur);
    syncDubAudio();
  };

  const handlePlay = () => {
    setPlaying(true);
    syncDubAudio();
  };

  const handlePause = () => {
    setPlaying(false);
    dubAudioRef.current?.pause();
  };

  // Subtitle dynamic styles for CapCut canvas
  const outlineWidth = subtitleStyles.outlineWidth ?? 2;
  const outlineColor = subtitleStyles.outlineColor || '#000000';
  const textColor = subtitleStyles.color || '#FFFFFF';
  const shadowColor = subtitleStyles.shadowColor || 'rgba(0,0,0,0.85)';
  const blurBase = Math.max(4, (subtitleStyles.shadowSize || 2) * 2.5);

  const capcutStyle: React.CSSProperties = {
    fontFamily: subtitleStyles.fontFamily || 'Arial, sans-serif',
    fontSize: `${subtitleStyles.fontSize || 22}px`,
    fontWeight: subtitleStyles.fontWeight || 'normal',
    fontStyle: subtitleStyles.fontStyle || 'normal',
    opacity: (subtitleStyles.opacity ?? 100) / 100,
    color: textColor,
    WebkitTextStroke: `${outlineWidth}px ${outlineColor}`,
    paintOrder: 'stroke fill',
    textShadow: [
      // Solid black outline around text (8-directional crisp stroke)
      `-${outlineWidth}px -${outlineWidth}px 0 ${outlineColor}`,
      ` ${outlineWidth}px -${outlineWidth}px 0 ${outlineColor}`,
      `-${outlineWidth}px  ${outlineWidth}px 0 ${outlineColor}`,
      ` ${outlineWidth}px  ${outlineWidth}px 0 ${outlineColor}`,
      ` 0 -${outlineWidth}px 0 ${outlineColor}`,
      ` 0  ${outlineWidth}px 0 ${outlineColor}`,
      `-${outlineWidth}px 0 0 ${outlineColor}`,
      ` ${outlineWidth}px 0 0 ${outlineColor}`,
      // Blurred outline around the subtitle (diffused multi-layer Gaussian halo)
      `0 0 ${blurBase}px ${shadowColor}`,
      `0 0 ${blurBase * 2}px ${shadowColor}`,
      `0 0 ${blurBase * 3}px rgba(0,0,0,0.65)`,
    ].join(', '),
    filter: `drop-shadow(0 0 ${blurBase}px rgba(0,0,0,0.85))`,
  };

  // Subtitle dynamic styles for dual teleprompter deck (Stage 2 & Stage 3)
  const dualSubtitleStyle: React.CSSProperties = {
    fontFamily: subtitleStyles.fontFamily || 'Arial, sans-serif',
    fontSize: `${subtitleStyles.fontSize || 20}px`,
    color: '#FFFFFF',
    WebkitTextStroke: '1.5px #000000',
    paintOrder: 'stroke fill',
    textShadow: [
      // Black outline around the text
      '-1.5px -1.5px 0 #000000',
      ' 1.5px -1.5px 0 #000000',
      '-1.5px  1.5px 0 #000000',
      ' 1.5px  1.5px 0 #000000',
      ' 0 -1.5px 0 #000000',
      ' 0  1.5px 0 #000000',
      '-1.5px 0 0 #000000',
      ' 1.5px 0 0 #000000',
      // Blurred outline around the subtitle
      '0 0 6px rgba(0,0,0,0.95)',
      '0 0 12px rgba(0,0,0,0.85)',
      '0 0 20px rgba(0,0,0,0.65)',
    ].join(', '),
    filter: 'drop-shadow(0 0 6px rgba(0,0,0,0.85))',
  };

  return (
    <div className="h-full w-full bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col min-h-0 overflow-hidden">
      {(currentStep === 3 || currentStep === 4) && dubPlaybackError && <div role="alert" className="bg-red-50 px-3 py-1 text-xs text-red-800">{dubPlaybackError}</div>}
      {/* Player Header Strip */}
      <div className="h-7.5 px-3 border-b border-[#E7E4DC] flex items-center justify-between bg-[#FAF9F6] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[#8D4B00] text-sm font-bold">▶</span>
          <span className="text-xs font-bold text-stone-900">{title}</span>
          <span className="px-1.5 py-0.2 rounded text-[8px] font-mono font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
            {project.resolution} • {project.fps}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {showSubtitleToggle && subtitleVariant !== 'none' && (
            <button
              type="button"
              data-testid="toggle-subtitles-btn"
              onClick={() => setSubtitlesVisible(!subtitlesVisible)}
              className={`px-2 py-0.5 rounded text-[10px] font-semibold flex items-center gap-1 transition-colors border cursor-pointer ${
                subtitlesVisible
                  ? 'bg-amber-50 text-[#8D4B00] border-amber-200 hover:bg-amber-100'
                  : 'bg-stone-100 text-stone-400 border-stone-200 hover:text-stone-600'
              }`}
              title={subtitlesVisible ? 'Remove subtitles overlay from video' : 'Show subtitles overlay on video'}
            >
              <Subtitles className="w-3 h-3" />
              <span>CC {subtitlesVisible ? 'ON' : 'OFF'}</span>
            </button>
          )}

          {showAudioSwitcher && (
            <div className="flex items-center p-0.5 bg-stone-100 rounded-lg border border-stone-200 text-[10px]">
              <button
                className={`px-2 py-0.5 rounded transition-colors ${
                  playback.audioChannel === 'orig'
                    ? 'font-bold bg-white text-[#8D4B00] shadow-2xs border border-amber-200'
                    : 'text-stone-500 hover:text-stone-800'
                }`}
                onClick={() => setAudioChannel('orig')}
              >
                {sourceCode} Orig
              </button>
              <button
                className={`px-2 py-0.5 rounded transition-colors ${
                  playback.audioChannel === 'dub'
                    ? 'font-bold bg-white text-[#8D4B00] shadow-2xs border border-amber-200'
                    : 'text-stone-500 hover:text-stone-800'
                }`}
                onClick={() => setAudioChannel('dub')}
              >
                {targetCode} Dub ★
              </button>
            </div>
          )}

          <span className="px-2 py-0.5 rounded bg-stone-100 text-stone-700 font-mono text-[10px] border border-stone-200 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 warm-pulse" />
            <span>
              <span data-playhead-timecode>{playback.formattedTime}</span> / {project.duration}
            </span>
          </span>
        </div>
      </div>

      {/* Widescreen Visual Preview Container */}
      <div className="relative flex-1 min-h-0 bg-neutral-950 flex items-center justify-center overflow-hidden group">
        {previewUrl ? (
          <video
            ref={videoRef}
            data-source-preview="true"
            className="w-full h-full object-contain bg-black"
            src={previewUrl}
            muted={currentStep === 3 && playback.audioChannel !== 'orig'}
            controls
            preload="metadata"
            onTimeUpdate={handleTimeUpdate}
            onSeeked={() => {
              if (videoRef.current) {
                updatePlaybackTime(videoRef.current.currentTime, videoRef.current.duration);
              }
              syncDubAudio();
            }}
            onLoadedMetadata={handleTimeUpdate}
            onPlay={handlePlay}
            onPause={handlePause}
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center bg-stone-900 text-stone-400 p-6 text-center">
            <span className="text-3xl mb-2">🎬</span>
            <p className="text-xs font-semibold text-stone-300">No media loaded</p>
            <p className="text-[11px] text-stone-500 mt-1">Upload a video to preview subtitles and playback</p>
          </div>
        )}

        {/* Subtitles Overlay */}
        {subtitleVariant !== 'none' && subtitlesVisible && currentSegment && (currentSegment.targetText || currentSegment.sourceText) && (
          <div className="absolute inset-x-3 bottom-4 z-20 flex justify-center text-center pointer-events-none">
            {subtitleVariant === 'capcut' ? (
              <div className="w-full max-w-2xl px-4 py-1 flex flex-col items-center">
                <p
                  data-canvas-subtitle="true"
                  className="font-semibold text-center leading-snug tracking-wide select-none"
                  style={capcutStyle}
                >
                  {currentSegment?.targetText || currentSegment?.sourceText || ''}
                </p>
              </div>
            ) : (
              <div className="w-full max-w-2xl px-4 py-1 flex flex-col items-center">
                <p
                  data-canvas-subtitle="true"
                  className="font-bold text-base leading-snug tracking-wide text-center select-none"
                  style={dualSubtitleStyle}
                >
                  {currentSegment?.targetText || currentSegment?.sourceText || ''}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
