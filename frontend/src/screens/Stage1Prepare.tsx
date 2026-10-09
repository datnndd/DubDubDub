import React, { useRef, useState } from 'react';
import { useDubDubStore } from '../store';
import {
  Upload,
  Film,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Settings2,
  Sliders,
  Volume2,
  Mic,
  Key,
  X,
  Globe,
  Cpu,
  Users,
  Sparkles,
  StopCircle,
  FileVideo,
  FolderTree,
  ListFilter,
} from 'lucide-react';
import { ProviderDocLink } from '../components/ProviderDocLink';

export const Stage1Prepare: React.FC = () => {
  const help = {
    source: 'Language spoken in the source media. Choose it to improve transcription accuracy.',
    target: 'Language for translated subtitles and dubbed speech.',
    timing: 'Choose whether speech, video speed, or subtitles handle timing differences.',
    provider: 'Speech recognition service that turns the source audio into text.',
    model: 'Recognition model used by the selected provider. Choose one suited to your audio.',
    deepgram: 'Adjust Deepgram transcription and speaker grouping settings for this video.',
    smartFormat: 'Format recognized dates, numbers, and similar speech for easier reading.',
    punctuate: 'Add punctuation to recognized speech for more readable subtitles.',
    diarization: 'Identify who is speaking. Enable for conversations with multiple voices.',
    speakerCount: 'Expected number of speakers. Leave on Auto-detect if unsure.',
    cuda: 'Use a supported NVIDIA GPU for local processing when available.',
    uttSplit: 'Silence threshold for splitting speech into subtitle segments.',
    diarizeModel: 'Deepgram speaker identification model. Keep the default unless you need a specific model.',
    paragraphs: 'Group recognized speech into paragraphs for longer passages.',
    utterances: 'Return individual speech turns for more precise speaker timing.',
    extra: 'Optional advanced Deepgram settings. Leave blank for the standard behavior.',
  };
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isDeepgramModalOpen, setIsDeepgramModalOpen] = useState(false);

  const project = useDubDubStore((s) => s.project);
  const languages = useDubDubStore((s) => s.languages);
  const engines = useDubDubStore((s) => s.engines);
  const backend = useDubDubStore((s) => s.backend);
  const jobStatus = useDubDubStore((s) => s.jobStatus);
  const jobProgress = useDubDubStore((s) => s.jobProgress);
  const jobStage = useDubDubStore((s) => s.jobStage);
  const jobMessage = useDubDubStore((s) => s.jobMessage);
  const elapsedSeconds = useDubDubStore((s) => s.elapsedSeconds);
  const selectMedia = useDubDubStore((s) => s.selectMedia);
  const updateSourceLanguage = useDubDubStore((s) => s.updateSourceLanguage);
  const updateTargetLanguage = useDubDubStore((s) => s.updateTargetLanguage);
  const updateTimingMode = useDubDubStore((s) => s.updateTimingMode);
  const updateAsrProvider = useDubDubStore((s) => s.updateAsrProvider);
  const updateUseCuda = useDubDubStore((s) => s.updateUseCuda);
  const updateEngineConfig = useDubDubStore((s) => s.updateEngineConfig);
  const updateDeepgramOptions = useDubDubStore((s) => s.updateDeepgramOptions);
  const cancelActiveJob = useDubDubStore((s) => s.cancelActiveJob);
  const openSettings = useDubDubStore((s) => s.openSettings);
  const sessionSubprojects = useDubDubStore((s) => s.sessionSubprojects || []);
  const activeParentProjectName = useDubDubStore((s) => s.activeParentProjectName);
  const setScanModalOpen = useDubDubStore((s) => s.setScanModalOpen);
  const setSessionDrawerOpen = useDubDubStore((s) => s.setSessionDrawerOpen);

  const options = backend.options;
  const isAnalyzing = backend.status === 'analyzing';
  const isRunning = jobStatus === 'running';
  const isFailed = jobStatus === 'failed';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      selectMedia(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      selectMedia(file);
    }
  };

  const selectedProvider = options.asrProviders?.find(
    (item: any) => item.recognType === Number(backend.config.recognType)
  ) || options.asrProviders?.[0] || { id: 'deepgram', label: 'Deepgram', models: ['nova-3'], recognType: 1 };

  const isDeepgram = selectedProvider.id === 'deepgram' || Number(backend.config.recognType) === 1;

  return (
    <div className="flex-1 min-h-0 w-full p-3 flex flex-col gap-3 overflow-y-auto bg-[#F9F8F5]">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,audio/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* BATCH MODE BANNER (when session subprojects exist) */}
      {sessionSubprojects.length > 0 && (
        <div
          data-batch-banner="true"
          className="bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-2.5 flex items-center justify-between text-xs text-amber-950 shrink-0 shadow-2xs"
        >
          <div className="flex items-center gap-2">
            <FolderTree className="w-4 h-4 text-amber-600 shrink-0" />
            <div>
              <span className="font-bold">Batch Mode Active: </span>
              <span>{sessionSubprojects.length} videos loaded in Parent Project </span>
              <span className="font-mono font-bold text-amber-800">
                [{activeParentProjectName || 'Today'}]
              </span>
              {project.filename && (
                <span className="ml-2 text-stone-600 hidden md:inline">
                  • Currently editing: <strong className="text-stone-900">{project.filename}</strong>
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSessionDrawerOpen(true)}
              className="px-2.5 py-1 rounded-md bg-[#8D4B00] text-white font-semibold hover:bg-[#743D00] transition cursor-pointer flex items-center gap-1.5 shadow-2xs"
            >
              <ListFilter className="w-3.5 h-3.5" />
              <span>Session Videos ({sessionSubprojects.length})</span>
            </button>
            <button
              type="button"
              onClick={() => setScanModalOpen(true)}
              className="px-2.5 py-1 rounded-md bg-white border border-amber-400 text-amber-900 font-medium hover:bg-amber-50 transition cursor-pointer"
            >
              Scan More
            </button>
          </div>
        </div>
      )}

      {/* TOP ZONE: Studio Video Monitor & Ingest Telemetry */}
      <section className="min-h-[300px] flex-1 w-full grid grid-cols-12 gap-3">
        {/* LEFT: Video Canvas & Viewport (8 cols) */}
        <div className="col-span-12 lg:col-span-8 bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col overflow-hidden">
          {/* Header Bar */}
          <div className="h-9 px-3.5 border-b border-[#E7E4DC] flex items-center justify-between bg-[#FAF9F6] shrink-0">
            <div className="flex items-center gap-2">
              <Film className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span className="text-xs font-bold text-stone-900">Source Video &amp; Audio Monitor</span>
              <span
                onClick={() => fileInputRef.current?.click()}
                className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wide cursor-pointer transition-colors ${
                  project.verified
                    ? 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200'
                    : 'bg-amber-100 text-[#8D4B00] hover:bg-amber-200 animate-pulse'
                }`}
              >
                {project.verified ? 'Stream Verified' : 'Awaiting Media'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 font-mono text-[10px] text-stone-600 mr-1">
                <span className={`w-2 h-2 rounded-full ${project.hasAudio ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                <span>{project.hasAudio ? `Audio: ${project.audioCodec}` : 'No verified audio stream'}</span>
              </div>
              <button
                type="button"
                onClick={() => setScanModalOpen(true)}
                className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-50 hover:bg-amber-100 text-[#8D4B00] border border-amber-300 transition-colors cursor-pointer flex items-center gap-1"
                title="Scan folders recursively for MP4 videos"
              >
                <FolderTree className="w-3 h-3" />
                <span>Scan Folders</span>
              </button>
              {project.verified && (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="px-2 py-0.5 rounded text-[10px] font-semibold bg-stone-100 hover:bg-stone-200 text-stone-700 transition-colors cursor-pointer"
                >
                  Replace Video
                </button>
              )}
            </div>
          </div>

          {/* Viewport Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative flex-1 min-h-[250px] bg-neutral-950 flex items-center justify-center overflow-hidden transition-colors ${
              project.previewUrl ? '' : 'cursor-pointer'
            } ${isDragOver ? 'bg-neutral-900 ring-2 ring-amber-500/80 ring-inset' : ''}`}
            onClick={() => {
              if (!project.previewUrl) fileInputRef.current?.click();
            }}
          >
            {project.previewUrl ? (
              <video
                data-source-preview="true"
                className="w-full h-full object-contain bg-black max-h-[500px]"
                src={project.previewUrl}
                controls
                preload="metadata"
              />
            ) : (
              <div className="flex flex-col items-center justify-center gap-3 p-8 text-center select-none max-w-sm">
                <div
                  className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-all shadow-md ${
                    isDragOver
                      ? 'bg-amber-500 text-white scale-110'
                      : 'bg-stone-800/90 border border-stone-700/80 text-amber-400 group-hover:scale-105'
                  }`}
                >
                  <Upload className="w-7 h-7" />
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span className="text-sm font-bold text-stone-100">
                    {isDragOver ? 'Drop video or audio file here' : 'Drop video here, or click to browse'}
                  </span>
                  <p className="text-[11px] text-stone-400 leading-relaxed">
                    Supports MP4, MKV, MOV, WebM, MP3, WAV up to 4K. Audio will be automatically extracted for transcription.
                  </p>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="px-3.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-xs font-semibold cursor-pointer transition-all"
                  >
                    Select Media File
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setScanModalOpen(true);
                    }}
                    className="px-3.5 py-1.5 rounded-lg bg-stone-800 hover:bg-stone-700 text-stone-200 border border-stone-600 text-xs font-semibold cursor-pointer transition-all flex items-center gap-1.5"
                  >
                    <FolderTree className="w-3.5 h-3.5 text-amber-400" />
                    <span>Scan Folders (Batch)</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT: Ingest Inspector & Live Telemetry (4 cols) */}
        <div className="col-span-12 lg:col-span-4 bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col overflow-hidden">
          <div className="h-9 px-3.5 border-b border-[#E7E4DC] flex items-center justify-between bg-[#FAF9F6] shrink-0">
            <div className="flex items-center gap-1.5 font-bold text-xs text-stone-900">
              <Sliders className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span>Media Stream Diagnostics</span>
            </div>
            <span className="font-mono text-[10px] text-stone-500 font-semibold">
              {project.fileSize !== '—' ? project.fileSize : 'No File'}
            </span>
          </div>

          <div className="p-3.5 flex-1 flex flex-col gap-3 overflow-y-auto">
            {/* Live ASR Running Telemetry Card */}
            {isRunning && (
              <div
                data-asr-progress="true"
                className="p-3 rounded-xl bg-amber-50/95 border border-amber-300 text-xs text-amber-950 flex flex-col gap-2 shadow-xs animate-in fade-in duration-200"
              >
                <div className="flex items-center justify-between font-bold text-xs text-[#8D4B00]">
                  <span className="flex items-center gap-1.5">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Stage: {jobStage || 'Transcribing Speech'}</span>
                  </span>
                  <div className="flex items-center gap-1.5 font-mono text-[11px]">
                    <span className="px-1.5 py-0.5 rounded bg-amber-200/70 text-[#8D4B00] font-bold">
                      ⏱️ {elapsedSeconds}s
                    </span>
                    <span className="font-bold">{jobProgress != null ? `${jobProgress.toFixed(0)}%` : 'Running'}</span>
                  </div>
                </div>

                <div className="w-full bg-amber-200/70 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="bg-[#8D4B00] h-full transition-all duration-300 rounded-full"
                    style={{ width: `${jobProgress != null ? Math.min(100, Math.max(5, jobProgress)) : 25}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] text-stone-600 font-mono">
                  <span className="truncate pr-2">{jobMessage || 'Processing audio stream with Deepgram ASR...'}</span>
                  <button
                    type="button"
                    onClick={() => cancelActiveJob()}
                    className="shrink-0 flex items-center gap-1 text-rose-700 hover:text-rose-800 font-bold hover:underline cursor-pointer"
                  >
                    <StopCircle className="w-3 h-3" />
                    <span>Cancel</span>
                  </button>
                </div>
              </div>
            )}

            {/* Error Banner */}
            {isFailed && (
              <div className="p-3 rounded-xl bg-rose-50 border border-rose-300 text-xs text-rose-900 flex items-start gap-2.5 shadow-xs">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="font-bold text-xs text-rose-800">Processing Failed</div>
                  <div className="text-[11px] text-rose-700 mt-0.5 leading-snug">{jobMessage}</div>
                  {selectedProvider.requiresSettings && !selectedProvider.configured && (
                    <button
                      type="button"
                      onClick={() => openSettings('providers', String(selectedProvider.id || ''))}
                      className="mt-2 px-2.5 py-1 bg-rose-700 text-white rounded-md text-[10px] font-bold hover:bg-rose-800 cursor-pointer shadow-2xs"
                    >
                      Configure {selectedProvider.label} API Key in Settings
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Ingest Analyzing State */}
            {isAnalyzing && (
              <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-center gap-2.5">
                <Loader2 className="w-4 h-4 animate-spin text-[#8D4B00]" />
                <span className="font-medium">Uploading and extracting audio streams…</span>
              </div>
            )}

            {/* Technical Metadata Specs 2x2 */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-200/90 flex flex-col justify-between">
                <span className="text-[10px] text-stone-500 font-medium">Resolution</span>
                <span className="font-mono font-bold text-stone-900 mt-0.5 text-xs">{project.resolution}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-200/90 flex flex-col justify-between">
                <span className="text-[10px] text-stone-500 font-medium">Frame Rate</span>
                <span className="font-mono font-bold text-stone-900 mt-0.5 text-xs">{project.fps}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-200/90 flex flex-col justify-between">
                <span className="text-[10px] text-stone-500 font-medium">Duration</span>
                <span className="font-mono font-bold text-stone-900 mt-0.5 text-xs">{project.duration}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-200/90 flex flex-col justify-between">
                <span className="text-[10px] text-stone-500 font-medium">Container Format</span>
                <span className="font-mono font-bold text-stone-900 mt-0.5 text-xs">{project.format}</span>
              </div>
            </div>

            {/* Stream Codec Info Card */}
            <div className="p-3 rounded-lg bg-stone-50 border border-stone-200/90 flex flex-col gap-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-stone-500 font-medium">Video Codec:</span>
                <span className="font-mono font-bold text-stone-800">{project.videoCodec}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-stone-500 font-medium">Audio Codec:</span>
                <span className="font-mono font-bold text-stone-800">{project.audioCodec}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-stone-500 font-medium">File Size:</span>
                <span className="font-mono font-bold text-stone-800">{project.fileSize}</span>
              </div>
            </div>

            {/* File info notice */}
            <div className="mt-auto p-2 rounded-lg bg-amber-50/50 border border-amber-200/50 text-[11px] text-stone-600 flex items-start gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#8D4B00] shrink-0 mt-0.5" />
              <span>
                {project.verified
                  ? `Ready for automated speech recognition with ${selectedProvider.label}.`
                  : 'Upload video to verify audio tracks and enable transcription.'}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* BOTTOM ZONE: Engineering Console (Languages, ASR Engine, Audio Enhancements) */}
      <section className="w-full grid grid-cols-12 gap-3">
        {/* 1. Language & Timing Route (4 cols) */}
        <div className="col-span-12 md:col-span-4 bg-white rounded-xl border border-[#E7E4DC] p-3.5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-1.5 font-bold text-xs text-stone-900 mb-2.5">
            <Globe className="w-3.5 h-3.5 text-[#8D4B00]" />
            <span>Language &amp; Timing</span>
          </div>

          <div className="space-y-2.5 flex-1">
            <div>
              <label title={help.source} className="text-[10px] text-stone-500 font-semibold block mb-1">
                Source Spoken Language
              </label>
              <select
                title={help.source}
                value={languages?.source?.code || 'zh-cn'}
                onChange={(e) => {
                  const opt = options.languages?.find((l: any) => l.code === e.target.value);
                  updateSourceLanguage(e.target.value, opt?.name || e.target.value);
                }}
                className="w-full text-xs font-semibold p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
              >
                {(options.languages || []).map((l: any) => (
                  <option key={l.code} value={l.code}>
                    {l.name} ({l.code})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label title={help.target} className="text-[10px] text-stone-500 font-semibold block mb-1">
                Target Dubbing Language
              </label>
              <select
                title={help.target}
                value={languages?.target?.code || 'vi'}
                onChange={(e) => {
                  const opt = options.languages?.find((l: any) => l.code === e.target.value);
                  updateTargetLanguage(e.target.value, opt?.name || e.target.value);
                }}
                className="w-full text-xs font-semibold p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
              >
                {(options.languages || []).map((l: any) => (
                  <option key={l.code} value={l.code}>
                    {l.name} ({l.code})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label title={help.timing} className="text-[10px] text-stone-500 font-semibold block mb-1">
                Timing Synchronization Mode
              </label>
              <select
                title={help.timing}
                value={languages.timingMode || 'voice'}
                onChange={(e) => updateTimingMode?.(e.target.value)}
                className="w-full text-xs font-semibold p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
              >
                <option value="voice">Voice Alignment (Fit speech to video)</option>
                <option value="video">Video Auto-speed (Adjust video speed)</option>
                <option value="none">Subtitle Only (No timing stretch)</option>
              </select>
            </div>
          </div>
        </div>

        {/* 2. ASR Speech Recognition Engine (4 cols) */}
        <div className="col-span-12 md:col-span-4 bg-white rounded-xl border border-[#E7E4DC] p-3.5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2.5">
            <div className="flex items-center gap-1.5 font-bold text-xs text-stone-900">
              <Mic className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span>ASR Engine</span>
            </div>
            <div className="flex items-center gap-1.5">
              {selectedProvider.requiresSettings && (
                <button
                  type="button"
                  onClick={() => openSettings('providers', String(selectedProvider.id || ''))}
                  className={`px-2 py-0.5 rounded text-[9px] font-bold flex items-center gap-1 cursor-pointer transition-colors ${
                    selectedProvider.configured
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'bg-amber-100 text-[#8D4B00] border border-amber-300 animate-pulse'
                  }`}
                >
                  <Key className="w-2.5 h-2.5" />
                  <span>{selectedProvider.configured ? 'Key Set' : 'Key Needed'}</span>
                </button>
              )}
              <button
                type="button"
                onClick={() => openSettings('providers', String(selectedProvider.id || ''))}
                className="p-1 rounded text-stone-400 hover:text-stone-700 hover:bg-stone-100 transition-colors cursor-pointer"
                title="Configure Provider Settings"
              >
                <Settings2 className="w-3.5 h-3.5 text-[#8D4B00]" />
              </button>
            </div>
          </div>

          <div className="space-y-2.5 flex-1">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label title={help.provider} className="text-[10px] text-stone-500 font-semibold">
                    Engine Provider
                  </label>
                  <ProviderDocLink providerId={backend.config.recognType} category="asr" />
                </div>
                <select
                  title={help.provider}
                  value={backend.config.recognType}
                  onChange={(e) => {
                    const p = options.asrProviders?.find((x: any) => x.recognType === Number(e.target.value));
                    updateAsrProvider(Number(e.target.value), p?.models?.[0] || 'default');
                  }}
                  className="w-full text-xs font-semibold p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
                >
                  {(options.asrProviders || []).map((p: any) => (
                    <option key={p.recognType} value={p.recognType}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label title={help.model} className="text-[10px] text-stone-500 font-semibold block mb-1">
                  Model Variant
                </label>
                <select
                  title={help.model}
                  value={backend.config.modelName}
                  onChange={(e) => updateAsrProvider(backend.config.recognType, e.target.value)}
                  className="w-full text-xs font-semibold p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
                >
                  {(selectedProvider.models || []).map((m: string) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Deepgram Dedicated Parameters Bar */}
            {isDeepgram && (
              <div className="flex flex-col gap-2 pt-0.5">
                <button
                  type="button"
                  data-testid="deepgram-options-btn"
                  title={help.deepgram}
                  onClick={() => setIsDeepgramModalOpen(true)}
                  className="w-full py-1.5 px-2.5 rounded-lg border border-amber-300 bg-amber-50/70 hover:bg-amber-100 text-[#8D4B00] text-xs font-semibold flex items-center justify-between transition-colors cursor-pointer shadow-2xs"
                >
                  <div className="flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5" />
                    <span>Deepgram Parameters</span>
                  </div>
                  <span className="font-mono text-[10px] bg-amber-200/80 px-1.5 py-0.5 rounded font-bold">
                    utt_split={backend.config.deepgramOptions?.utt_split ?? 0.8}s
                  </span>
                </button>

                {/* Inline Quick Switches for Deepgram */}
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <label title={help.smartFormat} className="flex items-center justify-between p-1.5 rounded-lg bg-stone-50 border border-stone-200 cursor-pointer">
                    <span className="text-stone-700 font-medium">Smart Format</span>
                    <input
                      type="checkbox"
                      checked={backend.config.deepgramOptions?.smart_format ?? true}
                      onChange={(e) => updateDeepgramOptions({ smart_format: e.target.checked })}
                      className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                    />
                  </label>
                  <label title={help.punctuate} className="flex items-center justify-between p-1.5 rounded-lg bg-stone-50 border border-stone-200 cursor-pointer">
                    <span className="text-stone-700 font-medium">Punctuate</span>
                    <input
                      type="checkbox"
                      checked={backend.config.deepgramOptions?.punctuate ?? true}
                      onChange={(e) => updateDeepgramOptions({ punctuate: e.target.checked })}
                      className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                    />
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 3. Audio Enhancements, Diarization & Hardware (4 cols) */}
        <div className="col-span-12 md:col-span-4 bg-white rounded-xl border border-[#E7E4DC] p-3.5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-1.5 font-bold text-xs text-stone-900 mb-2.5">
            <Volume2 className="w-3.5 h-3.5 text-[#8D4B00]" />
            <span>Audio Processing &amp; Hardware</span>
          </div>

          <div className="space-y-2.5 flex-1">
            {/* Speaker Diarization with Speaker Count */}
            <div className="p-2 rounded-lg bg-stone-50 border border-stone-200 flex flex-col gap-2">
              <label title={help.diarization} className="flex items-center justify-between text-xs cursor-pointer">
                <div className="flex items-center gap-1.5">
                  <Users className="w-3.5 h-3.5 text-[#8D4B00]" />
                  <span className="font-semibold text-stone-800">Speaker Diarization</span>
                </div>
                <input
                  type="checkbox"
                  checked={engines.speakerDiarization}
                  onChange={(e) => updateEngineConfig('speakerDiarization', e.target.checked)}
                  className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                />
              </label>

              {engines.speakerDiarization && (
                <div title={help.speakerCount} className="flex items-center justify-between pt-1 border-t border-stone-200/80 text-[11px]">
                  <span className="text-stone-600 font-medium">Expected Speakers:</span>
                  <select
                    title={help.speakerCount}
                    value={engines.speakerCount ?? 0}
                    onChange={(e) => updateEngineConfig('speakerCount', Number(e.target.value))}
                    className="font-mono text-xs p-1 bg-white rounded border border-stone-300 font-bold"
                  >
                    <option value={0}>Auto-detect (0)</option>
                    <option value={1}>1 Speaker</option>
                    <option value={2}>2 Speakers</option>
                    <option value={3}>3 Speakers</option>
                    <option value={4}>4 Speakers</option>
                    <option value={5}>5+ Speakers</option>
                  </select>
                </div>
              )}
            </div>

            {/* GPU / CUDA Acceleration */}
            <label title={help.cuda} className="flex items-center justify-between text-xs cursor-pointer p-2 rounded-lg bg-stone-50 border border-stone-200">
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-[#8D4B00]" />
                <span className="font-semibold text-stone-800">Hardware CUDA Acceleration</span>
              </div>
              <input
                type="checkbox"
                checked={backend.config.useCuda ?? false}
                onChange={(e) => updateUseCuda?.(e.target.checked)}
                className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
              />
            </label>
          </div>
        </div>
      </section>

      {/* Deepgram Parameters Comprehensive Modal */}
      {isDeepgramModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4 animate-in fade-in duration-150">
          <div className="bg-white rounded-2xl shadow-2xl border border-stone-200 max-w-md w-full overflow-hidden flex flex-col">
            <div className="px-5 py-3 border-b border-stone-200 flex items-center justify-between bg-[#FAF9F6]">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#8D4B00]" />
                <h3 className="font-bold text-xs text-stone-900">Deepgram ASR Parameters</h3>
              </div>
              <button
                type="button"
                onClick={() => setIsDeepgramModalOpen(false)}
                className="p-1 rounded-md text-stone-400 hover:text-stone-700 hover:bg-stone-100 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-3.5 text-xs max-h-[75vh] overflow-y-auto">
              <p className="text-[11px] text-stone-500 leading-relaxed">
                Configure Deepgram ASR prerecorded audio transcription parameters. Blank fields will use Deepgram preset defaults.
              </p>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label title={help.uttSplit} className="text-[10px] font-semibold text-stone-600 block mb-1">
                    utt_split (split threshold in s)
                  </label>
                  <input
                    title={help.uttSplit}
                    type="number"
                    step="0.1"
                    min="0"
                    placeholder="0.8"
                    value={backend.config.deepgramOptions?.utt_split ?? 0.8}
                    onChange={(e) =>
                      updateDeepgramOptions({
                        utt_split: e.target.value === '' ? undefined : parseFloat(e.target.value),
                      })
                    }
                    className="w-full text-xs font-mono p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400 font-bold"
                  />
                </div>

                <div>
                  <label title={help.diarizeModel} className="text-[10px] font-semibold text-stone-600 block mb-1">
                    diarize_model
                  </label>
                  <input
                    title={help.diarizeModel}
                    type="text"
                    placeholder="latest"
                    value={backend.config.deepgramOptions?.diarize_model ?? 'latest'}
                    onChange={(e) => updateDeepgramOptions({ diarize_model: e.target.value })}
                    className="w-full text-xs font-mono p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400 font-bold"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <label title={help.smartFormat} className="flex items-center gap-2 p-2 rounded-lg bg-stone-50 border border-stone-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={backend.config.deepgramOptions?.smart_format ?? true}
                    onChange={(e) => updateDeepgramOptions({ smart_format: e.target.checked })}
                    className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                  />
                  <span className="text-[11px] font-medium text-stone-700">smart_format</span>
                </label>

                <label title={help.punctuate} className="flex items-center gap-2 p-2 rounded-lg bg-stone-50 border border-stone-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={backend.config.deepgramOptions?.punctuate ?? true}
                    onChange={(e) => updateDeepgramOptions({ punctuate: e.target.checked })}
                    className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                  />
                  <span className="text-[11px] font-medium text-stone-700">punctuate</span>
                </label>

                <label title={help.paragraphs} className="flex items-center gap-2 p-2 rounded-lg bg-stone-50 border border-stone-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={backend.config.deepgramOptions?.paragraphs ?? true}
                    onChange={(e) => updateDeepgramOptions({ paragraphs: e.target.checked })}
                    className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                  />
                  <span className="text-[11px] font-medium text-stone-700">paragraphs</span>
                </label>

                <label title={help.utterances} className="flex items-center gap-2 p-2 rounded-lg bg-stone-50 border border-stone-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={backend.config.deepgramOptions?.utterances ?? true}
                    onChange={(e) => updateDeepgramOptions({ utterances: e.target.checked })}
                    className="rounded text-[#8D4B00] focus:ring-[#8D4B00]"
                  />
                  <span className="text-[11px] font-medium text-stone-700">utterances</span>
                </label>
              </div>

              <div>
                <label title={help.extra} className="text-[10px] font-semibold text-stone-600 block mb-1">
                  Extra Deepgram Parameters (URL Query or JSON)
                </label>
                <input
                  title={help.extra}
                  type="text"
                  placeholder="e.g. keywords=term1,term2&numerals=true"
                  value={backend.config.deepgramOptions?.extra || ''}
                  onChange={(e) => updateDeepgramOptions({ extra: e.target.value })}
                  className="w-full text-xs font-mono p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
                />
                <span className="text-[10px] text-stone-400 block mt-1">
                  Pass additional Deepgram query options (e.g. keywords, numerals, profanity_filter).
                </span>
              </div>
            </div>

            <div className="px-5 py-3 border-t border-stone-200 flex justify-between items-center bg-[#FAF9F6]">
              <button
                type="button"
                onClick={() =>
                  updateDeepgramOptions({
                    utt_split: 0.8,
                    diarize_model: 'latest',
                    smart_format: true,
                    punctuate: true,
                    paragraphs: true,
                    utterances: true,
                    extra: '',
                  })
                }
                className="text-xs text-stone-500 hover:text-stone-800 underline cursor-pointer"
              >
                Reset Defaults
              </button>
              <button
                type="button"
                onClick={() => setIsDeepgramModalOpen(false)}
                className="px-4 py-1.5 rounded-lg bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold cursor-pointer shadow-2xs"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
