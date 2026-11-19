import React, { useState } from 'react';
import { useDubDubStore } from '../store';
import { Play, ArrowRight, Video, CheckCircle2, AlertCircle, Loader2, StopCircle, Sparkles, Layers, RotateCcw } from 'lucide-react';

export const StatusFooter: React.FC = () => {
  const [showCapCutTooltip, setShowCapCutTooltip] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const currentStep = useDubDubStore((s) => s.currentStep);
  const setStep = useDubDubStore((s) => s.setStep);
  const backend = useDubDubStore((s) => s.backend);
  const project = useDubDubStore((s) => s.project);
  const jobStatus = useDubDubStore((s) => s.jobStatus);
  const jobStage = useDubDubStore((s) => s.jobStage);
  const jobProgress = useDubDubStore((s) => s.jobProgress);
  const jobMessage = useDubDubStore((s) => s.jobMessage);
  const segments = useDubDubStore((s) => s.segments || []);
  const startPrepareJob = useDubDubStore((s) => s.startPrepareJob);
  const cancelActiveJob = useDubDubStore((s) => s.cancelActiveJob);
  const exportEditedVideo = useDubDubStore((s) => s.exportEditedVideo);
  const setCapCutModalOpen = useDubDubStore((s) => s.setCapCutModalOpen);
  const dubbingStatus = useDubDubStore((s) => s.dubbingStatus);
  const dubbingError = useDubDubStore((s) => s.dubbingError);
  const runFullDubbing = useDubDubStore((s) => s.runFullDubbing);
  const cancelDubbing = useDubDubStore((s) => s.cancelDubbing);
  const enterStage4 = useDubDubStore((s) => s.enterStage4);
  const assemblingDubbing = useDubDubStore((s) => s.assemblingDubbing);
  const activeProjectId = useDubDubStore((s) => s.activeProjectId);
  const resetStage = useDubDubStore((s) => s.resetStage);
  const resettingStage = useDubDubStore((s) => s.resettingStage);
  const sessionSubprojects = useDubDubStore((s) => s.sessionSubprojects || []);
  const projectJobStatuses = useDubDubStore((s) => s.projectJobStatuses || {});
  const startBatchPrepareJobs = useDubDubStore((s) => s.startBatchPrepareJobs);
  const translationModal = useDubDubStore((s) => s.translationModal);
  const editVideo = useDubDubStore((s) => s.editVideo);
  const subtitleSource = useDubDubStore((s) => s.subtitleSource || 'audio_asr');
  const ocrRoiConfirmed = useDubDubStore((s) => Boolean(s.ocrRoiConfirmed));

  const activeSubStatus = activeProjectId ? projectJobStatuses[activeProjectId] : undefined;
  const isStage1ActiveRunning =
    (jobStatus === 'running' && jobStage !== 'voice' && jobStage !== 'render') ||
    (activeSubStatus?.status === 'running' && activeSubStatus?.stage !== 'voice' && activeSubStatus?.stage !== 'render');
  const isStage1BatchRunning = sessionSubprojects.some((s) => {
    const st = projectJobStatuses[s.id];
    return st && (st.status === 'running' || st.status === 'queued') && st.stage !== 'voice' && st.stage !== 'render';
  });
  const isStage1Busy = isStage1ActiveRunning || backend.status === 'analyzing' || isStage1BatchRunning;
  const isStage2Busy = translationModal?.status === 'translating';
  const isStage3Busy = dubbingStatus === 'running' || assemblingDubbing;
  const isStage4Busy = Boolean(editVideo?.exporting) || (jobStatus === 'running' && jobStage === 'render');
  const isCurrentStageBusy =
    resettingStage ||
    (currentStep === 1 ? isStage1Busy : currentStep === 2 ? isStage2Busy : currentStep === 3 ? isStage3Busy : isStage4Busy);

  const isStage1Cancellable = isStage1ActiveRunning || isStage1BatchRunning;
  const runnableBatchCount = sessionSubprojects.filter(
    (s) =>
      (s.stage ?? 1) <= 1 &&
      s.status !== 'completed' &&
      projectJobStatuses[s.id]?.status !== 'completed' &&
      Boolean((s as any).media_id || (s as any).mediaId || s.media_path)
  ).length;

  const handleAction = () => {
    if (currentStep === 1) {
      if (isStage1Cancellable) {
        cancelActiveJob();
      } else if (sessionSubprojects.length > 1) {
        startBatchPrepareJobs();
      } else {
        startPrepareJob();
      }
    } else if (currentStep === 2) {
      setStep(3);
    } else if (currentStep === 3) {
      if (dubbingStatus === 'running') {
        cancelDubbing();
      } else if (dubbingStatus === 'completed') {
        void enterStage4();
      } else {
        runFullDubbing();
      }
    } else if (currentStep === 4) {
      if (isStage4Busy) {
        cancelActiveJob();
      } else {
        exportEditedVideo();
      }
    }
  };

  const getActionConfig = () => {
    if (currentStep === 1) {
      const isVideoOcr = subtitleSource === 'video_ocr';
      const needsOcrConfirm = isVideoOcr && !ocrRoiConfirmed;
      if (isStage1Cancellable) {
        return {
          text: 'Cancel Processing',
          icon: <StopCircle className="w-3.5 h-3.5" />,
          disabled: false,
          color: 'bg-rose-700 hover:bg-rose-800 text-white',
        };
      }
      if (isStage1Busy) {
        return {
          text: isVideoOcr ? 'Running Video OCR…' : 'Processing ASR…',
          icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
          disabled: true,
          color: 'bg-stone-300 text-stone-600',
        };
      }
      if (sessionSubprojects.length > 1) {
        return {
          text: `Start Batch Dub (${sessionSubprojects.length} Videos)`,
          icon: <Play className="w-3.5 h-3.5 fill-current" />,
          disabled: runnableBatchCount === 0 || needsOcrConfirm,
          title: needsOcrConfirm ? 'Run Test Frame OCR to confirm subtitle ROI' : undefined,
          color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
        };
      }
      return {
        text: 'Start Dub',
        icon: <Play className="w-3.5 h-3.5 fill-current" />,
        disabled: !project?.verified || needsOcrConfirm,
        title: needsOcrConfirm ? 'Run Test Frame OCR to confirm subtitle ROI' : undefined,
        color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
      };
    }
    if (currentStep === 2) {
      return {
        text: 'Proceed to Voice & Dubbing',
        icon: <ArrowRight className="w-3.5 h-3.5" />,
        disabled: isStage2Busy,
        color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
      };
    }
    if (currentStep === 3) {
      if (assemblingDubbing) {
        return {
          text: 'Assembling Dubbing…',
          icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
          disabled: true,
          color: 'bg-stone-300 text-stone-600',
        };
      }
      if (dubbingStatus === 'running') {
        return {
          text: 'Cancel Dubbing',
          icon: <StopCircle className="w-3.5 h-3.5" />,
          disabled: false,
          color: 'bg-rose-700 hover:bg-rose-800 text-white',
        };
      }
      if (dubbingStatus === 'completed') {
        return {
          text: 'Proceed to Edit Video',
          icon: <ArrowRight className="w-3.5 h-3.5" />,
          disabled: false,
          color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
        };
      }
      return {
        text: 'Generate Dubbing',
        icon: <Sparkles className="w-3.5 h-3.5" />,
        disabled: false,
        color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
      };
    }
    return {
      text: isStage4Busy ? 'Rendering Video…' : 'Render dubbed video',
      icon: isStage4Busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Video className="w-3.5 h-3.5" />,
      disabled: isStage4Busy,
      color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
    };
  };

  const actionConfig = getActionConfig();

  // Stage-scoped telemetry icon, title, and subtitle
  let telemetryFailed = false;
  let telemetryTitle = '';
  let telemetrySub = 'All engines ready';
  if (currentStep === 1) {
    telemetryFailed = jobStatus === 'failed' && jobStage !== 'voice' && jobStage !== 'render';
    if (isStage1Busy || telemetryFailed || (jobStatus === 'completed' && jobStage !== 'voice' && jobStage !== 'render')) {
      telemetryTitle = jobMessage || (project?.verified ? 'Media verified — ready to process' : 'Choose a video to start');
    } else {
      telemetryTitle = project?.verified ? 'Media verified — ready to process' : 'Choose a video to start';
    }
    telemetrySub =
      isStage1Busy && jobProgress != null
        ? `Progress: ${jobProgress.toFixed(1)}%`
        : !project?.verified
        ? 'Upload a video, then choose languages and backend engines.'
        : 'All engines ready';
  } else if (currentStep === 2) {
    telemetryFailed = translationModal?.status === 'failed';
    if (isStage2Busy) {
      telemetryTitle = translationModal?.message || 'Translating subtitles…';
      telemetrySub = `Progress: ${Number(translationModal?.progress || 0).toFixed(1)}%`;
    } else if (telemetryFailed) {
      telemetryTitle = translationModal?.error || translationModal?.message || 'Translation failed';
      telemetrySub = 'Check translation provider settings and retry';
    } else {
      telemetryTitle = `${segments.length} segments ready`;
      telemetrySub = 'Review and translate subtitles';
    }
  } else if (currentStep === 3) {
    telemetryFailed = dubbingStatus === 'failed' || Boolean(dubbingError);
    if (assemblingDubbing) {
      telemetryTitle = 'Assembling dubbing audio…';
      telemetrySub = 'Mixing generated voice clips';
    } else if (dubbingStatus === 'running') {
      telemetryTitle = jobMessage || 'Generating voice dubbing…';
      telemetrySub = jobProgress != null ? `Progress: ${jobProgress.toFixed(1)}%` : 'Synthesizing segments…';
    } else if (telemetryFailed) {
      telemetryTitle = dubbingError || 'Voice dubbing failed';
      telemetrySub = 'Retry failed segments or check voice settings';
    } else if (dubbingStatus === 'completed') {
      telemetryTitle = 'Voice completed';
      telemetrySub = 'Ready to assemble and edit video';
    } else {
      telemetryTitle = `${segments.length} cues ready for voice synthesis`;
      telemetrySub = 'Assign voices and generate dubbing';
    }
  } else {
    telemetryFailed = Boolean(editVideo?.error) || (jobStatus === 'failed' && jobStage === 'render');
    if (isStage4Busy) {
      telemetryTitle = jobMessage || 'Rendering dubbed video…';
      telemetrySub = jobProgress != null ? `Progress: ${jobProgress.toFixed(1)}%` : 'Encoding video & audio tracks…';
    } else if (telemetryFailed) {
      telemetryTitle = editVideo?.error || jobMessage || 'Video export failed';
      telemetrySub = 'Review audio/subtitle settings and try again';
    } else {
      telemetryTitle = 'Studio timeline ready';
      telemetrySub = 'Adjust audio mix, subtitles, and render';
    }
  }

  return (
    <footer
      data-status-footer="true"
      className="h-[52px] flex-shrink-0 bg-white border-t border-[#E7E4DC] px-4 flex items-center justify-between gap-4 z-30 shadow-xs"
    >
      {/* Left: Live status telemetry */}
      <div className="flex items-center gap-3 min-w-0">
        <div
          className={`w-7 h-7 rounded-lg flex items-center justify-center ${
            telemetryFailed
              ? 'bg-rose-100 text-rose-700'
              : isCurrentStageBusy
              ? 'bg-amber-100 text-[#8D4B00]'
              : 'bg-emerald-100 text-emerald-700'
          }`}
        >
          {telemetryFailed ? (
            <AlertCircle className="w-4 h-4" />
          ) : isCurrentStageBusy ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <CheckCircle2 className="w-4 h-4" />
          )}
        </div>

        <div className="flex flex-col min-w-0">
          <div className="font-bold text-xs text-stone-900 truncate">
            {telemetryTitle}
          </div>
          <div className="text-[10px] text-stone-500 font-medium">
            {telemetrySub}
          </div>
        </div>
      </div>

      {/* Right: Primary Action Button & CapCut Bridge */}
      <div className="flex items-center gap-3 flex-shrink-0">
        {resetError && <span role="alert" className="text-[10px] text-rose-700 max-w-72">{resetError}</span>}
        <button
          type="button"
          data-testid={`reset-stage-${currentStep}-btn`}
          disabled={!activeProjectId || isCurrentStageBusy}
          onClick={async () => {
            setResetError(null);
            try { await resetStage(currentStep); }
            catch (error: any) { setResetError(error?.message || 'Could not reset stage'); }
          }}
          title={`Clear Stage ${currentStep} and dependent results. Uploaded media and exported files are kept.`}
          className="px-3 py-2 rounded-lg border border-stone-200 text-stone-700 font-bold text-xs flex items-center gap-1.5 disabled:opacity-50"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </button>
        {currentStep === 4 && (
          <div className="relative flex items-center">
            <button
              type="button"
              data-continue-in-capcut-btn="true"
              onClick={() => setCapCutModalOpen(true)}
              onMouseEnter={() => setShowCapCutTooltip(true)}
              onMouseLeave={() => setShowCapCutTooltip(false)}
              className="px-3.5 py-2 rounded-lg font-bold text-xs flex items-center gap-1.5 border border-stone-200 bg-white hover:bg-stone-50 text-stone-800 shadow-2xs transition-all cursor-pointer"
              title="Export merged audio and .srt subtitles, then import them with your original video into CapCut for advanced effects and transitions."
            >
              <Layers className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span>Continue in CapCut</span>
            </button>
            {showCapCutTooltip && (
              <div
                data-capcut-tooltip="true"
                className="absolute bottom-full mb-2 right-0 w-64 p-2.5 bg-stone-900 text-white text-[11px] rounded-lg shadow-xl z-50 pointer-events-none"
              >
                <div className="font-semibold text-amber-300 mb-0.5">Edit in CapCut</div>
                Export merged audio and .srt subtitles, then import them with your original video into CapCut for advanced effects and transitions.
              </div>
            )}
          </div>
        )}

        {currentStep === 4 && isStage4Busy && (
          <button
            type="button"
            data-testid="cancel-render-btn"
            onClick={() => cancelActiveJob()}
            className="px-3.5 py-2 rounded-lg font-bold text-xs flex items-center gap-1.5 bg-rose-700 hover:bg-rose-800 text-white shadow-2xs transition-all cursor-pointer"
          >
            <StopCircle className="w-3.5 h-3.5" />
            <span>Cancel Render</span>
          </button>
        )}

        <button
          onClick={handleAction}
          disabled={actionConfig.disabled || resettingStage}
          title={(actionConfig as any).title}
          className={`px-4 py-2 rounded-lg font-bold text-xs flex items-center gap-1.5 shadow-2xs transition-all cursor-pointer ${
            actionConfig.color
          } ${actionConfig.disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {actionConfig.icon}
          <span>{actionConfig.text}</span>
        </button>
      </div>
    </footer>
  );
};
