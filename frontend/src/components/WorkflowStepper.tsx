import React from 'react';
import { useDubDubStore } from '../store';
import { Film, Sliders, FileText, AudioWaveform, Layers, ArrowLeft } from 'lucide-react';

export const WorkflowStepper: React.FC = () => {
  const setActiveView = useDubDubStore((s) => s.setActiveView);
  const currentStep = useDubDubStore((s) => s.currentStep);
  const setStep = useDubDubStore((s) => s.setStep);
  const project = useDubDubStore((s) => s.project);
  const languages = useDubDubStore((s) => s.languages);
  const backend = useDubDubStore((s) => s.backend);
  const segments = useDubDubStore((s) => s.segments || []);
  const dubbingStatus = useDubDubStore((s) => s.dubbingStatus);
  const assembledDubUrl = useDubDubStore((s) => s.assembledDubUrl);
  const editVideo = useDubDubStore((s) => s.editVideo);

  const steps = [
    { num: 1, label: 'Prepare' },
    { num: 2, label: 'Review Transcript' },
    { num: 3, label: 'Voice & Dubbing' },
    { num: 4, label: 'Edit Video' },
  ];

  const asrProvider = backend?.options?.asrProviders?.find(
    (p: any) => p.recognType === Number(backend?.config?.recognType)
  );
  const asrLabel = asrProvider?.label || 'ASR Engine';
  const translatedCount = segments.filter((seg) => Boolean(seg?.targetText?.trim())).length;
  const voicedCount = segments.filter((seg) => Boolean(seg?.previewAudioUrl)).length;

  const stageTelemetry: Record<number, React.ReactNode> = {
    1: (
      <span className="inline-flex items-center gap-1 text-stone-600 bg-stone-100 px-2.5 py-0.5 rounded-md border border-stone-200 text-[10px] font-medium">
        <Sliders className="w-3 h-3 text-[#8D4B00]" />
        <span>{asrLabel} • {project?.verified ? 'Media Ready' : 'Awaiting Media'}</span>
      </span>
    ),
    2: (
      <span className="inline-flex items-center gap-1 text-amber-800 bg-amber-50 px-2.5 py-0.5 rounded-md border border-amber-200 text-[10px] font-semibold">
        <FileText className="w-3 h-3" />
        <span>{segments.length} cues • {translatedCount}/{segments.length} translated</span>
      </span>
    ),
    3: (
      <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md border text-[10px] font-semibold ${
        dubbingStatus === 'failed'
          ? 'text-rose-800 bg-rose-50 border-rose-200'
          : 'text-emerald-800 bg-emerald-50 border-emerald-200'
      }`}>
        <AudioWaveform className="w-3 h-3" />
        <span>
          {dubbingStatus === 'running'
            ? `Synthesizing (${voicedCount}/${segments.length})`
            : dubbingStatus === 'failed'
            ? `Dubbing Failed (${voicedCount}/${segments.length})`
            : `Voice ${voicedCount}/${segments.length} cues`}
        </span>
      </span>
    ),
    4: (
      <span className="inline-flex items-center gap-1 text-emerald-800 bg-emerald-50 px-2.5 py-0.5 rounded-md border border-emerald-200 text-[10px] font-semibold">
        <Layers className="w-3 h-3" />
        <span>
          {editVideo?.exporting
            ? 'Rendering Video…'
            : assembledDubUrl
            ? `Dub Track Ready${editVideo?.backgroundAudio ? ' + BGM' : ''}`
            : 'Dub Track Pending'}
        </span>
      </span>
    ),
  };

  const sourceCode = (languages?.source?.code ? String(languages.source.code).split('-')[0] : 'ZH').toUpperCase();
  const targetCode = (languages?.target?.code ? String(languages.target.code).split('-')[0] : 'VI').toUpperCase();

  return (
    <div className="h-9 flex-shrink-0 bg-[#FAF8F5] border-b border-[#E7E4DC] px-4 flex items-center justify-between z-20">
      {/* Media Identity Pill & Back to Projects Button */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          data-testid="workflow-back-to-projects-btn"
          onClick={() => setActiveView('projects')}
          className="flex items-center gap-1 text-[11px] font-semibold text-stone-600 hover:text-[#8D4B00] bg-white px-2 py-0.5 rounded-lg border border-stone-200 hover:border-amber-300 shadow-2xs transition-colors cursor-pointer"
          title="Back to Projects"
        >
          <ArrowLeft className="w-3.5 h-3.5 text-stone-400" />
          <span>Projects</span>
        </button>
        <div className="flex items-center gap-1.5 bg-white px-2.5 py-0.5 rounded-lg border border-stone-200 text-stone-700 shadow-2xs">
          <Film className="w-3.5 h-3.5 text-stone-400" />
          <span className="font-mono text-[11px] font-medium truncate max-w-[190px]">
            {project?.filename || 'Untitled Video Project'}
          </span>
          <span className="px-1.5 py-0.2 bg-amber-50 text-amber-900 border border-amber-200/80 rounded font-mono font-bold text-[9px] tracking-wide uppercase leading-none">
            {sourceCode} ➔ {targetCode}
          </span>
        </div>
      </div>

      {/* Synchronized 4-Step Interactive Navigation */}
      <div className="flex items-center gap-1.5 bg-white px-2.5 py-0.5 rounded-full border border-[#E7E4DC] shadow-2xs">
        {steps.map((s, idx) => {
          const isDone = s.num < currentStep;
          const isActive = s.num === currentStep;

          return (
            <React.Fragment key={s.num}>
              {isActive ? (
                <div
                  className="flex items-center gap-1 text-[#8D4B00] text-[11px] font-bold bg-amber-50 px-2.5 py-0.5 rounded-full border border-amber-200 shadow-2xs cursor-pointer"
                  onClick={() => setStep(s.num)}
                >
                  <span className="w-3.5 h-3.5 rounded-full bg-[#8D4B00] text-white flex items-center justify-center text-[9px] font-bold">
                    {s.num}
                  </span>
                  <span>{s.label}</span>
                </div>
              ) : isDone ? (
                <div
                  className="flex items-center gap-1 text-stone-600 hover:text-stone-900 text-[11px] font-semibold cursor-pointer transition-colors px-1"
                  onClick={() => setStep(s.num)}
                >
                  <span className="w-3.5 h-3.5 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center text-[9px] font-bold">
                    ✓
                  </span>
                  <span>{s.label}</span>
                </div>
              ) : (
                <div
                  className="flex items-center gap-1 text-stone-400 hover:text-stone-600 text-[11px] font-medium cursor-pointer transition-colors px-1"
                  onClick={() => setStep(s.num)}
                >
                  <span className="w-3.5 h-3.5 rounded-full bg-stone-200 text-stone-500 flex items-center justify-center text-[9px] font-semibold">
                    {s.num}
                  </span>
                  <span>{s.label}</span>
                </div>
              )}

              {idx < steps.length - 1 && <span className="text-stone-300 text-[10px]">•</span>}
            </React.Fragment>
          );
        })}
      </div>

      {/* Right Telemetry Badge */}
      <div className="hidden xl:flex items-center gap-2">
        {stageTelemetry[currentStep] || null}
      </div>
    </div>
  );
};
