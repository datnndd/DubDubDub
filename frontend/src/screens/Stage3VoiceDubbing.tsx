import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useDubDubStore } from '../store';
import { VideoPlayer } from '../components/VideoPlayer';
import { VoiceSelector } from '../components/VoiceSelector';
import {
  Volume2,
  Play,
  Pause,
  Sliders,
  Mic,
  RefreshCw,
  Plus,
  ArrowRight,
  Sparkles,
  Loader2,
  Users,
  Filter,
  X,
  Gauge,
} from 'lucide-react';
import { previewTTS } from '../api/voices';
import { useVoiceAudition, auditionVoice } from '../services/voiceAuditionManager';
import { ProviderDocLink } from '../components/ProviderDocLink';
import type { Segment } from '../types/segment';

const TTS_PROVIDERS = [
  { id: 0, name: 'ElevenLabs' },
  { id: 1, name: 'OmniVoice' },
  { id: 2, name: 'VieNeu-TTS' },
  { id: 3, name: 'Gemini TTS' },
];

const getSpeakerBadgeClasses = (color?: string) => {
  switch (color) {
    case 'secondary':
    case 'purple':
      return 'bg-purple-700 text-white';
    case 'emerald':
    case 'green':
      return 'bg-emerald-700 text-white';
    case 'sky':
    case 'blue':
      return 'bg-sky-700 text-white';
    case 'indigo':
      return 'bg-indigo-700 text-white';
    case 'rose':
    case 'red':
      return 'bg-rose-700 text-white';
    case 'amber':
    default:
      return 'bg-[#8D4B00] text-white';
  }
};

export const Stage3VoiceDubbing: React.FC = () => {
  const segments = useDubDubStore((s) => s.segments);
  const rawSpeakers = useDubDubStore((s) => s.speakers);
  const getDistinctSpeakers = useDubDubStore((s) => s.getDistinctSpeakers);
  const speakerVoiceMap = useDubDubStore((s) => s.speakerVoiceMap);
  const segmentVoiceOverrides = useDubDubStore((s) => s.segmentVoiceOverrides);
  const voices = useDubDubStore((s) => s.voices);
  const customVoices = useDubDubStore((s) => s.customVoices);
  const tuning = useDubDubStore((s) => s.tuning);
  const backend = useDubDubStore((s) => s.backend);
  const languages = useDubDubStore((s) => s.languages);
  const seek = useDubDubStore((s) => s.seek);
  const seekAndPlay = useDubDubStore((s) => s.seekAndPlay);
  const activeSegmentId = useDubDubStore((s) => s.activeSegmentId);
  const enterStage4 = useDubDubStore((s) => s.enterStage4);
  const assemblingDubbing = useDubDubStore((s) => s.assemblingDubbing);
  const dubbingStatus = useDubDubStore((s) => s.dubbingStatus);
  const triggerAutosave = useDubDubStore((s) => s.triggerAutosave);
  const setSpeakerVoice = useDubDubStore((s) => s.setSpeakerVoice);
  const setSegmentVoiceOverride = useDubDubStore((s) => s.setSegmentVoiceOverride);
  const clearSegmentVoiceOverride = useDubDubStore((s) => s.clearSegmentVoiceOverride);
  const updateSegmentVoicePreview = useDubDubStore((s) => s.updateSegmentVoicePreview);
  const updateSegmentText = useDubDubStore((s) => s.updateSegmentText);
  const setCreateVoiceModalOpen = useDubDubStore((s) => s.setCreateVoiceModalOpen);
  const setActiveView = useDubDubStore((s) => s.setActiveView);
  const loadVoices = useDubDubStore((s) => s.loadVoices);
  const updateBackendConfig = useDubDubStore((s) => (s as any).updateBackendConfig);
  const project = useDubDubStore((s) => s.project);
  const autoFitVoiceSpeed = useDubDubStore((s) => s.autoFitVoiceSpeed);
  const maxSpeedRate = useDubDubStore((s) => s.maxSpeedRate);
  const dubbingError = useDubDubStore((s) => s.dubbingError);
  const jobProgress = useDubDubStore((s) => s.jobProgress);
  const jobMessage = useDubDubStore((s) => s.jobMessage);
  const setAutoFitVoiceSpeed = useDubDubStore((s) => s.setAutoFitVoiceSpeed);
  const setMaxSpeedRate = useDubDubStore((s) => s.setMaxSpeedRate);

  const [loadingPreviewMap, setLoadingPreviewMap] = useState<Record<number, boolean>>({});
  const [selectedSpeakerFilter, setSelectedSpeakerFilter] = useState<string>('all');
  const [loadingSpeakerAudition, setLoadingSpeakerAudition] = useState<Record<string, boolean>>({});
  const [previewError, setPreviewError] = useState<string | null>(null);

  const { isKeyPlaying, isKeyLoading, stop, stopIfKey, play, error: auditionError } = useVoiceAudition();

  const speakers = useMemo(() => {
    return getDistinctSpeakers(segments, rawSpeakers);
  }, [segments, rawSpeakers, getDistinctSpeakers]);

  const speakerCueCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    segments.forEach((seg) => {
      const rawSpkId = seg.speakerId ?? (seg as any).speakerLabel ?? (seg as any).speakerName ?? 'spk_1';
      const segSpkId = String(rawSpkId).trim();
      const matchedSpeaker = speakers.find(
        (s) => s.id === segSpkId || (seg.speakerId != null && s.id === String(seg.speakerId))
      );
      const key = matchedSpeaker?.id || segSpkId;
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  }, [segments, speakers]);

  const filteredSegments = useMemo(() => {
    if (selectedSpeakerFilter === 'all') return segments;
    return segments.filter((seg) => {
      const rawSpkId = seg.speakerId ?? (seg as any).speakerLabel ?? (seg as any).speakerName ?? 'spk_1';
      const segSpkId = String(rawSpkId).trim();
      const matchedSpeaker = speakers.find(
        (s) => s.id === segSpkId || (seg.speakerId != null && s.id === String(seg.speakerId))
      );
      return segSpkId === selectedSpeakerFilter || matchedSpeaker?.id === selectedSpeakerFilter;
    });
  }, [segments, speakers, selectedSpeakerFilter]);

  const currentProvider = Number(backend?.config?.ttsType ?? 2);

  // Ensure voices are loaded for current provider
  useEffect(() => {
    loadVoices();
  }, [currentProvider, loadVoices]);

  // Stop preview audio playback on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  const handleProviderChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newProvider = parseInt(e.target.value, 10);
    if (updateBackendConfig) {
      updateBackendConfig('ttsType', newProvider);
    }
    await loadVoices();
    triggerAutosave();
  };

  const handleAuditionSpeakerVoice = async (speakerId: string, voiceId: string) => {
    if (!voiceId || !voiceId.trim() || voiceId === 'No' || voiceId.trim().toLowerCase() === 'clone') return;
    const targetLanguage = languages?.target?.code || 'vi';
    const matchedCustom = customVoices.find(
      (cv) => cv.id === voiceId || cv.name === voiceId || `Custom: ${cv.name}` === voiceId
    );
    const matchedPreset = voices.find((v) => v.id === voiceId || v.name === voiceId);
    const resolvedVoice = matchedCustom ? matchedCustom.id : (matchedPreset ? matchedPreset.id : voiceId);
    const targetProvider = matchedCustom?.provider ?? matchedPreset?.provider ?? currentProvider;
    const isMatchedPresetClone =
      matchedPreset &&
      matchedPreset.id.toLowerCase() !== 'clone' &&
      (matchedPreset.kind === 'custom' || matchedPreset.kind === 'clone' || matchedPreset.id.startsWith('voice_'));
    const staticSample = matchedCustom
      ? `/api/custom-voices/${matchedCustom.id}/audio`
      : matchedPreset?.sampleUrl || (isMatchedPresetClone ? `/api/custom-voices/${matchedPreset.id}/audio` : undefined);

    await auditionVoice({
      key: `speaker-${speakerId}`,
      voice: resolvedVoice,
      provider: targetProvider,
      language: targetLanguage,
      staticSampleUrl: staticSample,
      onLoadingChange: (loading) => {
        setLoadingSpeakerAudition((prev) => ({ ...prev, [speakerId]: loading }));
      },
    });
  };

  const handleTogglePlay = (segmentId: number, audioUrl: string) => {
    const segKey = `segment-${segmentId}`;
    if (isKeyPlaying(segKey)) {
      stop();
    } else {
      play(segKey, audioUrl);
    }
  };

  const handleGeneratePreview = async (seg: Segment, activeVoice: string) => {
    setPreviewError(null);
    setLoadingPreviewMap((prev) => ({ ...prev, [seg.id]: true }));
    try {
      const textToSynthesize = seg.targetText || seg.sourceText || '';
      const matchedCustom = customVoices.find(
        (cv) => cv.id === activeVoice || cv.name === activeVoice || `Custom: ${cv.name}` === activeVoice
      );
      const matchedPreset = voices.find((v) => v.id === activeVoice || v.name === activeVoice);
      const resolvedVoice = matchedCustom ? matchedCustom.id : (matchedPreset ? matchedPreset.id : activeVoice);
      const targetProvider = matchedCustom?.provider ?? matchedPreset?.provider ?? currentProvider;

      let slotDur: number | undefined = undefined;
      if (autoFitVoiceSpeed && typeof seg.startSec === 'number' && typeof seg.endSec === 'number') {
        const segIdx = segments.findIndex((s) => s.id === seg.id);
        const nextSeg = segIdx >= 0 ? segments[segIdx + 1] : undefined;
        const totalDur = project?.durationSec || 0;
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
        text: textToSynthesize,
        voice: resolvedVoice,
        provider: targetProvider,
        language: languages?.target?.code || 'vi',
        speed: tuning?.pace,
        segment_id: seg.id,
        force_refresh: true,
        auto_speed: autoFitVoiceSpeed,
        slot_duration_s: slotDur,
        max_speed_rate: maxSpeedRate,
      });

      const audioUrl = res.audio_url || res.preview_url;
      if (!audioUrl) throw new Error('The preview response did not contain an audio URL');
      updateSegmentVoicePreview(seg.id, audioUrl, res.id || res.preview_id, activeVoice, res.applied_speed);
    } catch (err) {
      console.error(`Preview generation failed for segment ${seg.id}:`, err);
      setPreviewError(err instanceof Error ? err.message : 'Voice preview generation failed');
    } finally {
      setLoadingPreviewMap((prev) => ({ ...prev, [seg.id]: false }));
    }
  };

  return (
    <div className="flex-1 min-h-0 w-full p-2.5 grid grid-cols-12 gap-2.5 overflow-hidden">
      {/* LEFT: Synchronized Video Preview (6 cols) */}
      <div className="col-span-12 lg:col-span-6 xl:col-span-6 h-full min-h-0 overflow-hidden">
        <VideoPlayer title="Neural Voice Synthesis Deck" subtitleVariant="dual" />
      </div>

      {/* RIGHT: Speaker Voice Matrix & Per-Segment Overrides (6 cols) */}
      <div className="col-span-12 lg:col-span-6 xl:col-span-6 h-full bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col min-h-0 overflow-hidden">
        {/* Header Strip */}
        <div className="h-11 px-3 border-b border-[#E7E4DC] flex items-center justify-between bg-[#FAF9F6] shrink-0 gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Mic className="w-4 h-4 text-[#8D4B00] shrink-0" />
            <span className="font-bold text-xs text-stone-900 truncate">Multi-Speaker Voice Matrix</span>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-[#8D4B00] shrink-0">
              {speakers.length} Speakers
            </span>
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {/* Auto Speed Fit Controls */}
            <div className="flex items-center gap-1 bg-stone-100/90 p-0.5 rounded-lg border border-stone-200 shadow-2xs">
              <button
                type="button"
                data-testid="auto-speed-toggle"
                onClick={() => setAutoFitVoiceSpeed(!autoFitVoiceSpeed)}
                className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-semibold transition-colors cursor-pointer ${
                  autoFitVoiceSpeed
                    ? 'bg-amber-100 text-[#8D4B00] border border-amber-300 shadow-2xs font-bold'
                    : 'text-stone-500 hover:text-stone-800'
                }`}
                title="Automatically adjust voice speed (pitch-preserving) to fit subtitle duration"
              >
                <Gauge className="w-3.5 h-3.5 text-[#8D4B00]" />
                <span>Auto Speed</span>
              </button>
              {autoFitVoiceSpeed && (
                <select
                  data-testid="max-speed-selector"
                  value={maxSpeedRate}
                  onChange={(e) => setMaxSpeedRate(parseFloat(e.target.value))}
                  className="text-[10px] font-mono font-semibold bg-white border border-stone-200 text-stone-700 rounded px-1.5 py-0.5 cursor-pointer focus:outline-none focus:border-[#8D4B00]"
                  title="Maximum speech speedup ceiling (Smart Fit)"
                >
                  <option value={1.15}>Max 1.15×</option>
                  <option value={1.25}>Max 1.25×</option>
                  <option value={1.35}>Max 1.35×</option>
                  <option value={1.50}>Max 1.50×</option>
                </select>
              )}
            </div>

            <button
              data-action="proceed-to-edit-video"
              onClick={() => void enterStage4()}
              disabled={assemblingDubbing || dubbingStatus === 'running'}
              className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold bg-[#8D4B00] text-white hover:bg-[#723c00] transition-colors cursor-pointer shadow-2xs"
              title="Proceed to Edit Video (Stage 4)"
            >
              <span>{assemblingDubbing ? 'Assembling Voice…' : 'Proceed to Edit Video'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setCreateVoiceModalOpen(true)}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-semibold bg-white border border-stone-200 text-stone-700 hover:bg-stone-50 hover:border-amber-400 transition-colors cursor-pointer shadow-2xs"
              title="Create New Custom Cloned Voice"
            >
              <Plus className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span>Create Voice</span>
            </button>
            <button
              onClick={() => setActiveView('voices')}
              className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium text-stone-700 bg-white border border-stone-200 hover:bg-stone-50 hover:border-amber-400 transition-colors cursor-pointer shadow-2xs"
              title="Open Voice Management Studio"
            >
              <Sliders className="w-3.5 h-3.5 text-stone-500" />
              <span>Library</span>
            </button>
          </div>
        </div>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          {dubbingStatus === 'running' && (
            <div
              data-testid="dubbing-running-banner"
              className="rounded-xl border border-amber-300 bg-amber-50 p-3 space-y-2 text-xs text-amber-950 shadow-xs"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 font-bold text-amber-900">
                  <Loader2 className="w-4 h-4 animate-spin text-[#8D4B00]" />
                  <span>Synthesizing Voice Dubbing in Background…</span>
                </div>
                <span className="text-xs font-mono font-bold text-[#8D4B00]">
                  {jobProgress != null ? `${Math.round(jobProgress)}%` : 'Processing…'}
                </span>
              </div>
              <p className="text-[11px] text-amber-800">
                {jobMessage || 'Generating voice audio for dialogue segments…'}
              </p>
              <div className="w-full bg-amber-200/80 rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-[#8D4B00] h-full transition-all duration-300 rounded-full"
                  style={{ width: `${Math.min(100, Math.max(0, jobProgress ?? 0))}%` }}
                />
              </div>
              <p className="text-[10px] text-stone-500 italic">
                Voice generation runs in the background. You can switch to other videos in the session bar to edit or review them at any time.
              </p>
            </div>
          )}
          {(previewError || auditionError || dubbingError) && (
            <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
              {previewError || auditionError || dubbingError}
            </div>
          )}
          {/* Upper Voice Casting Console */}
          <div className="space-y-2.5 bg-stone-50/70 p-3 rounded-xl border border-stone-200">
            <div className="flex items-center justify-between gap-2">
              <h4 className="text-[11px] font-bold uppercase tracking-wider text-stone-500">
                Upper Voice Casting Console
              </h4>
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] font-semibold text-stone-600">Provider:</label>
                <select
                  data-action="select-tts-provider"
                  value={currentProvider}
                  onChange={handleProviderChange}
                  className="text-xs bg-white border border-stone-200 rounded px-2 py-1 font-semibold text-stone-800 cursor-pointer focus:outline-none focus:border-[#8D4B00]"
                >
                  {TTS_PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <ProviderDocLink providerId={currentProvider} category="tts" />
              </div>
            </div>

            {/* Speaker Cards */}
            <div className="space-y-2">
              {speakers.map((spk) => {
                const currentVoice = (speakerVoiceMap && speakerVoiceMap[spk.id]) || (voices[0]?.id ?? 'default');
                const badgeClass = getSpeakerBadgeClasses(spk.color);
                return (
                  <div
                    key={spk.id}
                    data-speaker-card={spk.id}
                    className="p-2.5 rounded-lg border border-stone-200 bg-white flex items-center justify-between gap-3 shadow-2xs"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div
                        className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs shrink-0 shadow-2xs ${badgeClass}`}
                      >
                        {spk.code || (spk.id != null ? String(spk.id).slice(0, 2).toUpperCase() : 'SP')}
                      </div>
                      <div className="min-w-0">
                        <div className="font-bold text-xs text-stone-900 truncate">{spk.name}</div>
                        <div className="text-[10px] text-stone-500 truncate">{spk.role || 'Primary Speaker'}</div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0" data-speaker-voice-select={spk.id}>
                      <VoiceSelector
                        value={currentVoice}
                        onChange={(voiceId) => {
                          stopIfKey(`speaker-${spk.id}`);
                          setSpeakerVoice(spk.id, voiceId);
                        }}
                        size="md"
                        provider={currentProvider}
                        language={languages?.target?.code || 'vi'}
                      />
                      {(() => {
                        const hasValidVoice = Boolean(
                          currentVoice &&
                          currentVoice.trim() &&
                          currentVoice !== 'No' &&
                          currentVoice.trim().toLowerCase() !== 'clone'
                        );
                        const isThisLoading = isKeyLoading(`speaker-${spk.id}`) || Boolean(loadingSpeakerAudition[spk.id]);
                        const isThisPlaying = isKeyPlaying(`speaker-${spk.id}`);

                        return (
                          <button
                            type="button"
                            data-action="audition-speaker-voice"
                            data-speaker-id={spk.id}
                            onClick={() => handleAuditionSpeakerVoice(spk.id, currentVoice)}
                            disabled={!hasValidVoice || isThisLoading}
                            className={`p-2 rounded-lg border transition-colors flex items-center justify-center cursor-pointer shadow-2xs ${
                              isThisPlaying
                                ? 'bg-[#8D4B00] text-white border-[#8D4B00]'
                                : 'bg-white text-stone-600 border-stone-200 hover:border-amber-400 hover:text-[#8D4B00]'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                            title={
                              !hasValidVoice
                                ? 'No voice assigned'
                                : isThisPlaying
                                ? 'Stop Audition'
                                : isThisLoading
                                ? 'Generating voice preview…'
                                : `Audition ${currentVoice}`
                            }
                            aria-label={`Audition voice for ${spk.name}`}
                          >
                            {isThisLoading ? (
                              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#8D4B00]" />
                            ) : isThisPlaying ? (
                              <Pause className="w-3.5 h-3.5" />
                            ) : (
                              <Play className="w-3.5 h-3.5" />
                            )}
                          </button>
                        );
                      })()}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Per-Segment Dialogue Blocks */}
          <div className="space-y-3">
            {/* Header with Title, Cue Count, and Speaker Filter */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <h4 className="text-[11px] font-bold uppercase tracking-wider text-stone-400">
                  Dialogue Cue Overrides
                </h4>
                {selectedSpeakerFilter !== 'all' && (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-[#8D4B00]">
                    Filtered
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* Speaker Filter Dropdown */}
                <div className="flex items-center gap-1.5 bg-stone-100/90 px-2.5 py-1 rounded-lg border border-stone-200 shadow-2xs">
                  <Filter className="w-3 h-3 text-[#8D4B00]" />
                  <span className="text-[11px] font-semibold text-stone-600">Speaker:</span>
                  <select
                    data-testid="stage3-speaker-filter"
                    data-action="filter-speaker"
                    value={selectedSpeakerFilter}
                    onChange={(e) => setSelectedSpeakerFilter(e.target.value)}
                    className="text-xs font-semibold bg-white border border-stone-200 rounded px-2 py-0.5 text-stone-800 focus:outline-none focus:border-[#8D4B00] cursor-pointer"
                  >
                    <option value="all">All Speakers ({segments.length})</option>
                    {speakers.map((spk) => (
                      <option key={spk.id} value={spk.id}>
                        {spk.name} ({speakerCueCounts[spk.id] || 0})
                      </option>
                    ))}
                  </select>
                  {selectedSpeakerFilter !== 'all' && (
                    <button
                      type="button"
                      data-testid="clear-speaker-filter-btn"
                      onClick={() => setSelectedSpeakerFilter('all')}
                      className="p-0.5 rounded hover:bg-stone-200 text-stone-400 hover:text-stone-700 cursor-pointer"
                      title="Clear speaker filter"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>

                <span className="text-[10px] text-stone-500 font-mono">
                  {filteredSegments.length} of {segments.length} {segments.length === 1 ? 'cue' : 'cues'}
                </span>
              </div>
            </div>

            {/* Quick Interactive Speaker Pills */}
            {speakers.length > 1 && (
              <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 text-xs">
                <button
                  type="button"
                  data-testid="filter-pill-all"
                  onClick={() => setSelectedSpeakerFilter('all')}
                  className={`px-2.5 py-1 rounded-full text-[11px] font-semibold transition-colors cursor-pointer shrink-0 ${
                    selectedSpeakerFilter === 'all'
                      ? 'bg-[#8D4B00] text-white shadow-2xs'
                      : 'bg-stone-100 hover:bg-stone-200 text-stone-600'
                  }`}
                >
                  All ({segments.length})
                </button>
                {speakers.map((spk) => {
                  const isSelected = selectedSpeakerFilter === spk.id;
                  const badgeClass = getSpeakerBadgeClasses(spk.color);
                  const count = speakerCueCounts[spk.id] || 0;
                  return (
                    <button
                      key={spk.id}
                      type="button"
                      data-testid={`filter-pill-${spk.id}`}
                      onClick={() => setSelectedSpeakerFilter(isSelected ? 'all' : spk.id)}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold transition-all cursor-pointer shrink-0 border ${
                        isSelected
                          ? 'bg-amber-50 border-[#8D4B00] text-[#8D4B00] ring-1 ring-[#8D4B00]/30 shadow-2xs font-bold'
                          : 'bg-white border-stone-200 hover:border-stone-300 text-stone-700'
                      }`}
                    >
                      <span className={`w-3.5 h-3.5 rounded-full text-[8px] flex items-center justify-center font-bold ${badgeClass}`}>
                        {spk.code || (spk.id != null ? String(spk.id).slice(0, 1).toUpperCase() : 'S')}
                      </span>
                      <span>{spk.name}</span>
                      <span className="text-[10px] opacity-70">({count})</span>
                    </button>
                  );
                })}
              </div>
            )}

            {filteredSegments.length === 0 ? (
              <div
                data-testid="no-cues-for-speaker"
                className="p-8 text-center bg-stone-50 rounded-xl border border-stone-200 text-stone-500 space-y-2"
              >
                <Users className="w-8 h-8 mx-auto text-stone-300" />
                <p className="font-semibold text-xs text-stone-700">No dialogue cues found for this speaker</p>
                <p className="text-[11px] text-stone-400">Try switching to another speaker or resetting the filter.</p>
                <button
                  type="button"
                  data-testid="reset-empty-speaker-filter-btn"
                  onClick={() => setSelectedSpeakerFilter('all')}
                  className="mt-2 px-3 py-1 bg-white border border-stone-200 rounded-lg text-xs font-semibold text-stone-700 hover:bg-stone-50 cursor-pointer shadow-2xs"
                >
                  Show All Cues
                </button>
              </div>
            ) : (
              filteredSegments.map((seg) => {
              const rawSpkId = seg.speakerId ?? (seg as any).speakerLabel ?? (seg as any).speakerName ?? 'spk_1';
              const segSpkId = String(rawSpkId).trim();
              const matchedSpeaker = speakers.find((s) => s.id === segSpkId || (seg.speakerId != null && s.id === String(seg.speakerId)));
              const spkCode = seg.speakerCode || matchedSpeaker?.code || (segSpkId.length > 1 ? segSpkId.slice(0, 2).toUpperCase() : 'S1');
              const spkName = seg.speakerName || matchedSpeaker?.name || 'Speaker 1';
              const spkColor = seg.speakerColor || matchedSpeaker?.color || 'amber';
              const badgeClass = getSpeakerBadgeClasses(spkColor);

              const hasOverride = Boolean(
                (segmentVoiceOverrides && segmentVoiceOverrides[seg.id]) || seg.voiceOverride
              );
              const activeVoice =
                (segmentVoiceOverrides && segmentVoiceOverrides[seg.id]) ||
                seg.voiceOverride ||
                (speakerVoiceMap && (speakerVoiceMap[segSpkId] || (seg.speakerId != null ? speakerVoiceMap[String(seg.speakerId)] : undefined))) ||
                (voices[0]?.id ?? 'Default');

              const isLoadingPreview = Boolean(loadingPreviewMap[seg.id]);
              const isPlayingThis = isKeyPlaying(`segment-${seg.id}`);

              return (
                <div
                  key={seg.id}
                  data-segment-card={`stage3-${seg.id}`}
                  onClick={(e) => {
                    const target = e.target as HTMLElement;
                    if (target.closest('button, select, textarea, input, label, a')) return;
                    seekAndPlay(seg.startSec, seg.endSec, seg.id);
                  }}
                  className={`p-3 rounded-xl border transition-all cursor-pointer ${
                    activeSegmentId === seg.id
                      ? 'border-[#8D4B00] bg-amber-50/20 ring-1 ring-[#8D4B00]/20 shadow-xs'
                      : 'border-stone-200 bg-white hover:border-stone-300 shadow-2xs'
                  } space-y-2.5`}
                >
                  {/* Block Header: Speaker Badge, Timestamp, Voice Selector */}
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0 flex-wrap">
                      <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-stone-100 text-stone-700">
                        #{seg.id}
                      </span>

                      {/* Original Speaker Badge with Stage 2 styling */}
                      <span
                        data-speaker-badge={seg.speakerId || segSpkId}
                        className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wide inline-flex items-center gap-1 shadow-2xs ${badgeClass}`}
                      >
                        <span className="opacity-90 font-mono">{spkCode}</span>
                        <span>{spkName}</span>
                      </span>

                      {/* Seekable Timestamp */}
                      <button
                        data-action="seek-segment"
                        onClick={(e) => {
                          e.stopPropagation();
                          seekAndPlay(seg.startSec, seg.endSec, seg.id);
                        }}
                        className="font-mono text-[10px] text-stone-500 font-semibold hover:text-[#8D4B00] transition-colors flex items-center gap-1 cursor-pointer"
                        title="Play video for this dialogue cue"
                      >
                        <Play className="w-2.5 h-2.5 fill-current text-[#8D4B00]" />
                        <span>{seg.startTime}</span>
                        {seg.endTime && <span className="opacity-60">➔ {seg.endTime}</span>}
                      </button>
                    </div>

                    {/* Dual-Level Voice Selector */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      {!hasOverride && (
                        <span className="text-[9px] font-bold text-stone-400 bg-stone-100 px-1.5 py-0.5 rounded uppercase tracking-wide">
                          (Default)
                        </span>
                      )}

                      <VoiceSelector
                        value={activeVoice}
                        onChange={(voiceId) => {
                          stopIfKey(`segment-${seg.id}`);
                          setSegmentVoiceOverride(seg.id, voiceId);
                        }}
                        size="sm"
                        provider={currentProvider}
                        language={languages?.target?.code || 'vi'}
                      />

                      {hasOverride && (
                        <button
                          data-action="reset-segment-voice"
                          onClick={() => clearSegmentVoiceOverride(seg.id)}
                          className="p-1 rounded hover:bg-stone-100 text-stone-400 hover:text-stone-700 cursor-pointer"
                          title="Reset to Speaker Default"
                        >
                          <RefreshCw className="w-3 h-3" />
                        </button>
                      )}

                      <button
                        data-action="set-speaker-default"
                        onClick={() => {
                          const targetSpkId = matchedSpeaker?.id || seg.speakerId || segSpkId;
                          setSpeakerVoice(targetSpkId, activeVoice);
                          clearSegmentVoiceOverride(seg.id);
                        }}
                        className="p-1 rounded hover:bg-stone-100 text-stone-400 hover:text-[#8D4B00] cursor-pointer"
                        title={`Apply "${activeVoice}" as default for ${spkName}`}
                      >
                        <Users className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  {/* Texts: Source Text & Editable Target Dubbing Script */}
                  <div className="space-y-2">
                    <div>
                      <label className="text-[10px] text-stone-400 font-medium block mb-0.5">
                        Source Text
                      </label>
                      <div className="w-full text-xs p-2 rounded-lg bg-stone-50 border border-stone-100 text-stone-600 select-text">
                        {seg.sourceText || seg.text || '—'}
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center justify-between mb-0.5">
                        <label className="text-[10px] font-semibold text-stone-700 block">
                          Target Dubbing Text
                        </label>
                        {/* Emotion tag quick insertion chips for VieNeu */}
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => updateSegmentText(seg.id, `${(seg.targetText || '').trim()} [cười] `, true)}
                            className="px-1.5 py-0.2 rounded text-[9px] font-medium bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200 cursor-pointer"
                            title="Insert chuckle emotion tag"
                          >
                            + [cười]
                          </button>
                          <button
                            type="button"
                            onClick={() => updateSegmentText(seg.id, `${(seg.targetText || '').trim()} [thở dài] `, true)}
                            className="px-1.5 py-0.2 rounded text-[9px] font-medium bg-sky-50 text-sky-700 hover:bg-sky-100 border border-sky-200 cursor-pointer"
                            title="Insert sigh emotion tag"
                          >
                            + [thở dài]
                          </button>
                          <button
                            type="button"
                            onClick={() => updateSegmentText(seg.id, `${(seg.targetText || '').trim()} [hắng giọng] `, true)}
                            className="px-1.5 py-0.2 rounded text-[9px] font-medium bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-200 cursor-pointer"
                            title="Insert throat clear emotion tag"
                          >
                            + [hắng giọng]
                          </button>
                        </div>
                      </div>
                      <textarea
                        data-segment-input={`stage3-${seg.id}`}
                        value={seg.targetText ?? ''}
                        onChange={(e) => updateSegmentText(seg.id, e.target.value, true)}
                        rows={2}
                        className="w-full text-xs p-2 rounded-lg bg-white border border-stone-200 focus:border-[#8D4B00] focus:ring-1 focus:ring-[#8D4B00] outline-none transition-all"
                        placeholder="Enter translated target script for voice synthesis…"
                      />
                    </div>
                  </div>

                  {/* Voice Preview Controls for each block */}
                  <div className="pt-2 border-t border-stone-100 flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <button
                        data-action="generate-voice-preview"
                        onClick={() => handleGeneratePreview(seg, activeVoice)}
                        disabled={isLoadingPreview}
                        className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold bg-amber-50 text-[#8D4B00] border border-amber-200/80 hover:bg-amber-100 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed shadow-2xs"
                        title="Generate voice preview for this text block"
                      >
                        {isLoadingPreview ? (
                          <>
                            <Loader2 className="w-3 h-3 animate-spin" />
                            <span>Generating Preview…</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="w-3 h-3 text-[#8D4B00]" />
                            <span>{seg.previewAudioUrl && (!seg.previewVoice || seg.previewVoice === activeVoice) ? 'Regenerate Voice Preview' : 'Generate Voice Preview'}</span>
                          </>
                        )}
                      </button>

                      {seg.previewAudioUrl && (
                        <div className="flex items-center gap-1.5 bg-stone-100 px-2 py-0.5 rounded-md border border-stone-200">
                          <button
                            data-action="play-voice-preview"
                            onClick={() => handleTogglePlay(seg.id, seg.previewAudioUrl!)}
                            className="p-1 rounded hover:bg-stone-200 text-stone-700 cursor-pointer"
                            title={isPlayingThis ? 'Pause preview' : 'Play audition preview'}
                          >
                            {isPlayingThis ? (
                              <Pause className="w-3 h-3 fill-current text-[#8D4B00]" />
                            ) : (
                              <Play className="w-3 h-3 fill-current text-[#8D4B00]" />
                            )}
                          </button>
                          <span className="text-[10px] font-mono text-stone-600 font-medium">
                            {seg.previewVoice && seg.previewVoice !== activeVoice ? `Audition (${seg.previewVoice})` : 'Audition'}
                          </span>
                        </div>
                      )}
                      {seg.previewAudioUrl && typeof seg.previewSpeedFactor === 'number' && (
                        <span className="text-[10px] font-mono text-stone-600" title="Speed applied to this generated voice preview">
                          Applied speed: {seg.previewSpeedFactor.toFixed(2)}x
                        </span>
                      )}
                    </div>

                    <div className="text-[10px] text-stone-400 font-mono truncate">
                      Voice: <span className="font-semibold text-stone-600">{activeVoice}</span>
                    </div>
                  </div>
                </div>
              );
            }))}
          </div>
        </div>
      </div>
    </div>
  );
};
