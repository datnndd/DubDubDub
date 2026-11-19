import React, { useRef, useEffect } from 'react';
import { useDubDubStore } from '../store';
import { VideoPlayer } from '../components/VideoPlayer';
import {
  Sliders,
  Type,
  Image as ImageIcon,
  Volume2,
  VolumeX,
  Music,
  Trash2,
  Upload,
  Video,
  Mic,
  Subtitles,
  Bold,
  Italic,
} from 'lucide-react';

export const Stage4EditVideo: React.FC = () => {
  const bgmInputRef = useRef<HTMLInputElement | null>(null);
  const thumbInputRef = useRef<HTMLInputElement | null>(null);
  const bgmAudioRef = useRef<HTMLAudioElement | null>(null);

  const segments = useDubDubStore((s) => s.segments);
  const project = useDubDubStore((s) => s.project);
  const playback = useDubDubStore((s) => s.playback);
  const seek = useDubDubStore((s) => s.seek);
  const activeSegmentId = useDubDubStore((s) => s.activeSegmentId);
  const setActiveSegmentId = useDubDubStore((s) => s.setActiveSegmentId);
  const subtitleStyles = useDubDubStore((s) => s.subtitleStyles);
  const editVideo = useDubDubStore((s) => s.editVideo);
  const assembledDubUrl = useDubDubStore((s) => s.assembledDubUrl);
  const setAudioMix = useDubDubStore((s) => s.setAudioMix);
  const toggleAudioMute = useDubDubStore((s) => s.toggleAudioMute);
  const setBackgroundAudio = useDubDubStore((s) => s.setBackgroundAudio);
  const removeBackgroundAudio = useDubDubStore((s) => s.removeBackgroundAudio);
  const setThumbnail = useDubDubStore((s) => s.setThumbnail);
  const removeThumbnail = useDubDubStore((s) => s.removeThumbnail);
  const updateSubtitleStyle = useDubDubStore((s) => s.updateSubtitleStyle);
  const setActiveInspectorTab = useDubDubStore((s) => s.setActiveInspectorTab);
  const updateSegmentText = useDubDubStore((s) => s.updateSegmentText);

  const activeTab = editVideo.activeTab || 'audio';
  const duration = project.durationSec || 1;
  const playheadPercent = Math.min(100, Math.max(0, (playback.currentTime / duration) * 100));
  const bgmUrl = editVideo.backgroundAudio?.url || '';
  const bgmMixPercent = Number(editVideo.audioMix?.background ?? 0);

  useEffect(() => {
    const audio = bgmAudioRef.current;
    if (!audio) return;
    if (!bgmUrl) {
      audio.pause();
      return;
    }
    const clampedVol = Math.max(0, Math.min(1, bgmMixPercent / 100));
    audio.volume = clampedVol;
    audio.muted = clampedVol <= 0;
    audio.playbackRate = playback.playbackSpeed || 1.0;

    if (playback.isPlaying && clampedVol > 0) {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        const expectedTime = playback.currentTime % audio.duration;
        if (Math.abs(audio.currentTime - expectedTime) > 0.35) {
          audio.currentTime = expectedTime;
        }
      }
      if (audio.paused) {
        audio.play().catch(() => {});
      }
    } else {
      audio.pause();
    }
  }, [bgmUrl, bgmMixPercent, playback.isPlaying, playback.playbackSpeed, playback.currentTime]);

  useEffect(() => {
    const audio = bgmAudioRef.current;
    if (!audio || !bgmUrl || !playback.seekRequest) return;
    if (Number.isFinite(audio.duration) && audio.duration > 0) {
      audio.currentTime = playback.seekRequest.time % audio.duration;
    } else {
      audio.currentTime = playback.seekRequest.time;
    }
  }, [bgmUrl, playback.seekRequest]);

  useEffect(() => {
    return () => {
      bgmAudioRef.current?.pause();
    };
  }, []);

  const handleBgmChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setBackgroundAudio(file);
  };

  const handleThumbChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setThumbnail(file);
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    seek(pct * duration);
  };

  return (
    <div className="flex-1 min-h-0 w-full p-2.5 flex flex-col gap-2.5 overflow-hidden">
      {/* Hidden file inputs for BGM and Thumbnail */}
      <input
        ref={bgmInputRef}
        id="stage4-background-input"
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleBgmChange}
      />
      <input
        ref={thumbInputRef}
        id="stage4-thumbnail-input"
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleThumbChange}
      />
      <audio
        id="stage4-bgm-preview"
        ref={bgmAudioRef}
        src={bgmUrl || undefined}
        loop
        className="hidden"
      />
      {editVideo.error && (
        <div role="alert" data-testid="stage4-error-alert" className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs text-rose-800 shrink-0">
          {editVideo.error}
        </div>
      )}

      {/* UPPER DECK: Preview (7-8 cols) and Inspector (4-5 cols) */}
      <section className="flex-1 min-h-0 w-full grid grid-cols-12 gap-2.5 overflow-hidden">
        {/* UPPER LEFT: Video Preview Deck */}
        <div className="col-span-12 lg:col-span-7 xl:col-span-8 h-full min-h-0 overflow-hidden">
          <VideoPlayer title="CapCut Studio Preview" subtitleVariant="capcut" />
        </div>

        {/* UPPER RIGHT: 3-Tab Inspector */}
        <div className="col-span-12 lg:col-span-5 xl:col-span-4 h-full bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col min-h-0 overflow-hidden">
          {/* Tabs Navigation */}
          <div className="h-9 px-2 border-b border-[#E7E4DC] flex items-center justify-between bg-[#FAF9F6] shrink-0">
            <div className="flex items-center gap-1 p-0.5 bg-stone-100 rounded-lg text-xs font-semibold">
              <button
                onClick={() => setActiveInspectorTab('audio')}
                className={`px-2.5 py-1 rounded-md transition-colors flex items-center gap-1 cursor-pointer ${
                  activeTab === 'audio'
                    ? 'bg-white text-[#8D4B00] shadow-2xs'
                    : 'text-stone-500 hover:text-stone-800'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Audio Mix</span>
              </button>
              <button
                onClick={() => setActiveInspectorTab('subtitles')}
                className={`px-2.5 py-1 rounded-md transition-colors flex items-center gap-1 cursor-pointer ${
                  activeTab === 'subtitles'
                    ? 'bg-white text-[#8D4B00] shadow-2xs'
                    : 'text-stone-500 hover:text-stone-800'
                }`}
              >
                <Type className="w-3.5 h-3.5" />
                <span>Subtitles</span>
              </button>
              <button
                onClick={() => setActiveInspectorTab('assets')}
                className={`px-2.5 py-1 rounded-md transition-colors flex items-center gap-1 cursor-pointer ${
                  activeTab === 'assets'
                    ? 'bg-white text-[#8D4B00] shadow-2xs'
                    : 'text-stone-500 hover:text-stone-800'
                }`}
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Assets</span>
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-y-auto p-3 space-y-4">
            {activeTab === 'audio' && (
              <div className="space-y-3.5">
                <h4 className="text-[11px] font-bold uppercase tracking-wider text-stone-400">Audio Channels</h4>

                {/* Original Channel */}
                <div className="p-2.5 rounded-xl border border-stone-200 bg-stone-50/50 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-stone-800">Original Audio</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-[#8D4B00]">{editVideo.audioMix.original}%</span>
                      <button
                        onClick={() => toggleAudioMute('original')}
                        className="p-1 rounded hover:bg-stone-200 text-stone-600 cursor-pointer"
                        title="Toggle mute"
                      >
                        {editVideo.audioMix.original === 0 ? (
                          <VolumeX className="w-3.5 h-3.5 text-rose-600" />
                        ) : (
                          <Volume2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>
                  <input
                    type="range"
                    data-mix-slider="original"
                    min="0"
                    max="150"
                    value={editVideo.audioMix.original}
                    onChange={(e) => setAudioMix('original', e.target.value)}
                    className="w-full"
                  />
                </div>

                {/* Dubbed Channel */}
                <div className="p-2.5 rounded-xl border border-stone-200 bg-stone-50/50 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-stone-800">Dubbed Voiceover</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-[#8D4B00]">{editVideo.audioMix.dubbed}%</span>
                      <button
                        onClick={() => toggleAudioMute('dubbed')}
                        className="p-1 rounded hover:bg-stone-200 text-stone-600 cursor-pointer"
                        title="Toggle mute"
                      >
                        {editVideo.audioMix.dubbed === 0 ? (
                          <VolumeX className="w-3.5 h-3.5 text-rose-600" />
                        ) : (
                          <Volume2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>
                  <p className="text-[10px] text-stone-500">
                    {assembledDubUrl ? 'Stitched Stage 3 voiceover loaded' : 'No stitched voiceover loaded'}
                  </p>
                  <input
                    type="range"
                    data-mix-slider="dubbed"
                    min="0"
                    max="150"
                    value={editVideo.audioMix.dubbed}
                    onChange={(e) => setAudioMix('dubbed', e.target.value)}
                    className="w-full"
                  />
                </div>

                {/* Background Channel */}
                <div className="p-2.5 rounded-xl border border-stone-200 bg-stone-50/50 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-stone-800">Background Music</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-[#8D4B00]">{editVideo.audioMix.background}%</span>
                      <button
                        onClick={() => toggleAudioMute('background')}
                        className="p-1 rounded hover:bg-stone-200 text-stone-600 cursor-pointer"
                        title="Toggle mute"
                      >
                        {editVideo.audioMix.background === 0 ? (
                          <VolumeX className="w-3.5 h-3.5 text-rose-600" />
                        ) : (
                          <Volume2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>
                  <input
                    type="range"
                    data-mix-slider="background"
                    min="0"
                    max="150"
                    value={editVideo.audioMix.background}
                    onChange={(e) => setAudioMix('background', e.target.value)}
                    className="w-full"
                  />
                </div>
              </div>
            )}

            {activeTab === 'subtitles' && (
              <div className="space-y-3.5">
                <h4 className="text-[11px] font-bold uppercase tracking-wider text-stone-400">Typography &amp; Style</h4>

                <div className="space-y-3">
                  {/* Font Family */}
                  <div>
                    <label className="text-[10px] text-stone-400 font-medium block mb-1">Font Family</label>
                    <select
                      data-subtitle-font-family="true"
                      value={subtitleStyles.fontFamily || 'Arial'}
                      onChange={(e) => updateSubtitleStyle('fontFamily', e.target.value)}
                      className="w-full text-xs font-semibold px-2.5 py-1.5 rounded-lg border border-stone-200 bg-stone-50 text-stone-800 focus:outline-none focus:border-amber-400 cursor-pointer"
                    >
                      <option value="Arial">Arial</option>
                      <option value="Inter">Inter</option>
                      <option value="Impact">Impact</option>
                      <option value="Roboto">Roboto</option>
                      <option value="Georgia">Georgia</option>
                      <option value="sans-serif">System Default</option>
                    </select>
                  </div>

                  {/* Font Style Toggles (Bold, Italic) */}
                  <div>
                    <label className="text-[10px] text-stone-400 font-medium block mb-1">Font Style</label>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        data-subtitle-bold-toggle="true"
                        onClick={() => updateSubtitleStyle('fontWeight', subtitleStyles.fontWeight === 'bold' ? 'normal' : 'bold')}
                        className={`flex-1 py-1 px-2 rounded-lg border text-xs font-bold flex items-center justify-center gap-1 transition-colors cursor-pointer ${
                          subtitleStyles.fontWeight === 'bold'
                            ? 'bg-[#8D4B00] text-white border-amber-900 shadow-2xs'
                            : 'bg-stone-50 border-stone-200 text-stone-700 hover:bg-stone-100'
                        }`}
                        title="Toggle Bold"
                      >
                        <Bold className="w-3.5 h-3.5" />
                        <span>Bold</span>
                      </button>
                      <button
                        type="button"
                        data-subtitle-italic-toggle="true"
                        onClick={() => updateSubtitleStyle('fontStyle', subtitleStyles.fontStyle === 'italic' ? 'normal' : 'italic')}
                        className={`flex-1 py-1 px-2 rounded-lg border text-xs font-semibold italic flex items-center justify-center gap-1 transition-colors cursor-pointer ${
                          subtitleStyles.fontStyle === 'italic'
                            ? 'bg-[#8D4B00] text-white border-amber-900 shadow-2xs'
                            : 'bg-stone-50 border-stone-200 text-stone-700 hover:bg-stone-100'
                        }`}
                        title="Toggle Italic"
                      >
                        <Italic className="w-3.5 h-3.5" />
                        <span>Italic</span>
                      </button>
                    </div>
                  </div>

                  {/* Font Size */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-stone-700">Font Size</span>
                      <span className="font-mono font-bold text-[#8D4B00]">{subtitleStyles.fontSize}px</span>
                    </div>
                    <input
                      type="range"
                      data-subtitle-font-size="true"
                      min="12"
                      max="64"
                      value={subtitleStyles.fontSize}
                      onChange={(e) => updateSubtitleStyle('fontSize', parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  {/* Opacity / Transparency */}
                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-stone-700">Opacity / Transparency</span>
                      <span className="font-mono font-bold text-[#8D4B00]">{subtitleStyles.opacity ?? 100}%</span>
                    </div>
                    <input
                      type="range"
                      data-subtitle-opacity="true"
                      min="0"
                      max="100"
                      value={subtitleStyles.opacity ?? 100}
                      onChange={(e) => updateSubtitleStyle('opacity', parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-stone-700">Outline Width</span>
                      <span className="font-mono font-bold text-[#8D4B00]">{subtitleStyles.outlineWidth}px</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="6"
                      value={subtitleStyles.outlineWidth}
                      onChange={(e) => updateSubtitleStyle('outlineWidth', parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-stone-700">Blurred Outline (Shadow / Glow)</span>
                      <span className="font-mono font-bold text-[#8D4B00]">{subtitleStyles.shadowSize}px</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="12"
                      value={subtitleStyles.shadowSize}
                      onChange={(e) => updateSubtitleStyle('shadowSize', parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2 pt-2">
                    <div>
                      <label className="text-[10px] text-stone-400 font-medium block mb-0.5">Text Color</label>
                      <input
                        type="color"
                        value={subtitleStyles.color}
                        onChange={(e) => updateSubtitleStyle('color', e.target.value)}
                        className="w-full h-8 rounded border border-stone-200 cursor-pointer"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-stone-400 font-medium block mb-0.5">Outline Color</label>
                      <input
                        type="color"
                        value={subtitleStyles.outlineColor}
                        onChange={(e) => updateSubtitleStyle('outlineColor', e.target.value)}
                        className="w-full h-8 rounded border border-stone-200 cursor-pointer"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'assets' && (
              <div className="space-y-4">
                {/* BGM Upload & Preview */}
                <div className="space-y-2">
                  <h4 className="text-[11px] font-bold uppercase tracking-wider text-stone-400">Background Music</h4>
                  {editVideo.backgroundAudio ? (
                    <div className="p-3 rounded-xl border border-stone-200 bg-stone-50 flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <Music className="w-4 h-4 text-[#8D4B00] shrink-0" />
                        <span className="text-xs font-semibold text-stone-800 truncate">
                          {editVideo.backgroundAudio.name}
                        </span>
                      </div>
                      <button
                        onClick={() => removeBackgroundAudio()}
                        className="p-1 rounded hover:bg-rose-50 text-stone-400 hover:text-rose-600"
                        title="Remove BGM"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => bgmInputRef.current?.click()}
                      className="w-full py-4 px-3 rounded-xl border-2 border-dashed border-stone-200 hover:border-amber-400 bg-stone-50/50 flex flex-col items-center justify-center gap-1 text-stone-500 hover:text-stone-800 transition-colors cursor-pointer"
                    >
                      <Upload className="w-4 h-4 text-stone-400" />
                      <span className="text-xs font-semibold">Upload Background Music</span>
                    </button>
                  )}
                </div>

                {/* Thumbnail Upload & Preview */}
                <div className="space-y-2">
                  <h4 className="text-[11px] font-bold uppercase tracking-wider text-stone-400">Video Thumbnail</h4>
                  {editVideo.thumbnail ? (
                    <div className="relative aspect-video rounded-xl overflow-hidden border border-stone-200 group">
                      <img
                        src={editVideo.thumbnail.url}
                        alt="Video thumbnail"
                        className="w-full h-full object-cover"
                      />
                      <button
                        onClick={() => removeThumbnail()}
                        className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/60 hover:bg-rose-600 text-white transition-colors"
                        title="Remove Thumbnail"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => thumbInputRef.current?.click()}
                      className="w-full aspect-video rounded-xl border-2 border-dashed border-stone-200 hover:border-amber-400 bg-stone-50/50 flex flex-col items-center justify-center gap-1 text-stone-500 hover:text-stone-800 transition-colors cursor-pointer"
                    >
                      <ImageIcon className="w-5 h-5 text-stone-400" />
                      <span className="text-xs font-semibold">Upload Custom Thumbnail</span>
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* LOWER DECK: Multi-Track Timeline */}
      <section className="h-[210px] shrink-0 w-full bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col overflow-hidden">
        {/* Timeline Header & Zoom Controls */}
        <div className="h-7 px-3 border-b border-[#E7E4DC] flex items-center justify-between bg-[#FAF9F6] shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold tracking-wider uppercase text-stone-700">TIMELINE</span>
            <span className="text-[10px] text-stone-400 font-mono">
              {playback.formattedTime} / {project.duration}
            </span>
          </div>
        </div>

        {/* Tracks Deck */}
        <div
          className="relative flex-1 overflow-x-auto overflow-y-hidden p-2 flex flex-col justify-between select-none cursor-pointer"
          onClick={handleTimelineClick}
        >
          {/* Playhead Marker */}
          <div
            data-timeline-playhead="true"
            className="absolute top-0 bottom-0 w-0.5 bg-[#8D4B00] z-30 pointer-events-none transition-all duration-75"
            style={{ left: `${playheadPercent}%` }}
          >
            <div className="w-2.5 h-2.5 -ml-1 -top-0.5 bg-[#8D4B00] rounded-full shadow-xs" />
          </div>

          {/* Track 1: Video Track */}
          <div className="h-8 flex items-center gap-2 bg-stone-100/70 rounded-lg px-2 border border-stone-200/80">
            <div className="w-20 shrink-0 flex items-center gap-1 text-[10px] font-bold text-stone-700">
              <Video className="w-3 h-3 text-stone-500" />
              <span>Video</span>
            </div>
            <div className="flex-1 h-5 bg-stone-200/80 rounded flex items-center px-2 text-[10px] font-mono text-stone-600 truncate">
              {project.filename}
            </div>
          </div>

          {/* Track 2: Subtitles Track */}
          <div className="h-8 flex items-center gap-2 bg-stone-100/70 rounded-lg px-2 border border-stone-200/80">
            <div className="w-20 shrink-0 flex items-center gap-1 text-[10px] font-bold text-stone-700">
              <Subtitles className="w-3 h-3 text-stone-500" />
              <span>Subtitles</span>
            </div>
            <div className="relative flex-1 h-5 bg-amber-100/50 rounded overflow-hidden">
              {segments.map((seg) => {
                const segLeft = (seg.startSec / duration) * 100;
                const segWidth = Math.max(0.8, ((seg.endSec - seg.startSec) / duration) * 100);
                const isSelected = seg.id === activeSegmentId;

                return (
                  <div
                    key={seg.id}
                    data-segment-card={seg.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      setActiveSegmentId(seg.id);
                      seek(seg.startSec);
                    }}
                    className={`absolute top-0 bottom-0 rounded border text-[9px] truncate px-1 flex items-center transition-all ${
                      isSelected
                        ? 'bg-[#8D4B00] text-white border-amber-900 shadow-xs z-10'
                        : 'bg-amber-200/80 text-amber-950 border-amber-300 hover:bg-amber-300'
                    }`}
                    style={{ left: `${segLeft}%`, width: `${segWidth}%` }}
                    title={seg.targetText || seg.sourceText}
                  >
                    {seg.targetText || seg.sourceText}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Track 3: Dubbed Audio Track */}
          <div className="h-8 flex items-center gap-2 bg-stone-100/70 rounded-lg px-2 border border-stone-200/80">
            <div className="w-20 shrink-0 flex items-center gap-1 text-[10px] font-bold text-stone-700">
              <Mic className="w-3 h-3 text-stone-500" />
              <span>Dubbed</span>
            </div>
            <div className="relative flex-1 h-5 bg-emerald-100/50 rounded overflow-hidden">
              {segments.map((seg) => {
                const segLeft = (seg.startSec / duration) * 100;
                const segWidth = Math.max(0.8, ((seg.endSec - seg.startSec) / duration) * 100);
                return (
                  <div
                    key={seg.id}
                    className="absolute top-0 bottom-0 bg-emerald-200/80 border border-emerald-300 rounded text-[9px] px-1 flex items-center text-emerald-950 truncate"
                    style={{ left: `${segLeft}%`, width: `${segWidth}%` }}
                  >
                    TTS #{seg.id}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Track 4: BGM Track */}
          <div className="h-8 flex items-center gap-2 bg-stone-100/70 rounded-lg px-2 border border-stone-200/80">
            <div className="w-20 shrink-0 flex items-center gap-1 text-[10px] font-bold text-stone-700">
              <Music className="w-3 h-3 text-stone-500" />
              <span>BGM</span>
            </div>
            <div className="flex-1 h-5 bg-stone-200/80 rounded flex items-center px-2 text-[10px] font-mono text-stone-600 truncate">
              {editVideo.backgroundAudio?.name || 'No BGM track loaded'}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
