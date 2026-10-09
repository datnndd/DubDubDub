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
  const jobProgress = useDubDubStore((s) => s.jobProgress);
  const jobMessage = useDubDubStore((s) => s.jobMessage);
  const startPrepareJob = useDubDubStore((s) => s.startPrepareJob);
  const cancelActiveJob = useDubDubStore((s) => s.cancelActiveJob);
  const exportEditedVideo = useDubDubStore((s) => s.exportEditedVideo);
  const setCapCutModalOpen = useDubDubStore((s) => s.setCapCutModalOpen);
  const dubbingStatus = useDubDubStore((s) => s.dubbingStatus);
  const runFullDubbing = useDubDubStore((s) => s.runFullDubbing);
  const enterStage4 = useDubDubStore((s) => s.enterStage4);
  const assemblingDubbing = useDubDubStore((s) => s.assemblingDubbing);
  const activeProjectId = useDubDubStore((s) => s.activeProjectId);
  const resetStage = useDubDubStore((s) => s.resetStage);
  const resettingStage = useDubDubStore((s) => s.resettingStage);
  const sessionSubprojects = useDubDubStore((s) => s.sessionSubprojects || []);
  const startBatchPrepareJobs = useDubDubStore((s) => s.startBatchPrepareJobs);
  const translationModal = useDubDubStore((s) => s.translationModal);

  const isBusy = jobStatus === 'running' || backend.status === 'analyzing' || dubbingStatus === 'running' || assemblingDubbing || translationModal?.status === 'translating' || resettingStage;
  const isCancellable = jobStatus === 'running';

  const handleAction = () => {
    if (currentStep === 1) {
      if (isCancellable) {
        cancelActiveJob();
      } else if (sessionSubprojects.length > 1) {
        startBatchPrepareJobs();
      } else {
        startPrepareJob();
      }
    } else if (currentStep === 2) {
      setStep(3);
    } else if (currentStep === 3) {
      if (dubbingStatus === 'completed') {
        void enterStage4();
      } else {
        runFullDubbing();
      }
    } else if (currentStep === 4) {
      exportEditedVideo();
    }
  };

  const getActionConfig = () => {
    if (currentStep === 1) {
      if (isCancellable) {
        return {
          text: 'Cancel Processing',
          icon: <StopCircle className="w-3.5 h-3.5" />,
          disabled: false,
          color: 'bg-rose-700 hover:bg-rose-800 text-white',
        };
      }
      if (isBusy) {
        return {
          text: 'Processing ASR…',
          icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
          disabled: true,
          color: 'bg-stone-300 text-stone-600',
        };
      }
      if (sessionSubprojects.length > 1) {
        return {
          text: `Start Batch Dub (${sessionSubprojects.length} Videos)`,
          icon: <Play className="w-3.5 h-3.5 fill-current" />,
          disabled: false,
          color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
        };
      }
      return {
        text: 'Start Dub',
        icon: <Play className="w-3.5 h-3.5 fill-current" />,
        disabled: !project?.verified,
        color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
      };
    }
    if (currentStep === 2) {
      return {
        text: 'Proceed to Voice & Dubbing',
        icon: <ArrowRight className="w-3.5 h-3.5" />,
        disabled: false,
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
          text: 'Generating Dubbing…',
          icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
          disabled: true,
          color: 'bg-stone-300 text-stone-600',
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
      text: 'Render dubbed video',
      icon: <Video className="w-3.5 h-3.5" />,
      disabled: false,
      color: 'bg-[#8D4B00] hover:bg-[#743D00] text-white',
    };
  };

  const actionConfig = getActionConfig();

  return (
    <footer
      data-status-footer="true"
      className="h-[52px] flex-shrink-0 bg-white border-t border-[#E7E4DC] px-4 flex items-center justify-between gap-4 z-30 shadow-xs"
    >
      {/* Left: Live status telemetry */}
      <div className="flex items-center gap-3 min-w-0">
        <div
          className={`w-7 h-7 rounded-lg flex items-center justify-center ${
            jobStatus === 'failed'
              ? 'bg-rose-100 text-rose-700'
              : isBusy
              ? 'bg-amber-100 text-[#8D4B00]'
              : 'bg-emerald-100 text-emerald-700'
          }`}
        >
          {jobStatus === 'failed' ? (
            <AlertCircle className="w-4 h-4" />
          ) : isBusy ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <CheckCircle2 className="w-4 h-4" />
          )}
        </div>

        <div className="flex flex-col min-w-0">
          <div className="font-bold text-xs text-stone-900 truncate">
            {jobMessage || (project?.verified ? 'Media verified — ready to process' : 'Choose a video to start')}
          </div>
          <div className="text-[10px] text-stone-500 font-medium">
            {isBusy && jobProgress != null
              ? `Progress: ${jobProgress.toFixed(1)}%`
              : currentStep === 1 && !project?.verified
              ? 'Upload a video, then choose languages and backend engines.'
              : 'All engines ready'}
          </div>
        </div>
      </div>

      {/* Right: Primary Action Button & CapCut Bridge */}
      <div className="flex items-center gap-3 flex-shrink-0">
        {resetError && <span role="alert" className="text-[10px] text-rose-700 max-w-72">{resetError}</span>}
        <button
          type="button"
          data-testid={`reset-stage-${currentStep}-btn`}
          disabled={!activeProjectId || isBusy}
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

        <button
          onClick={handleAction}
          disabled={actionConfig.disabled || resettingStage}
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
