import React from 'react';
import { useDubDubStore } from '../store';
import {
  X,
  Layers,
  Film,
  Play,
  ArrowRight,
  Plus,
  FolderOpen,
} from 'lucide-react';
import { getSubprojectStatusInfo } from '../utils/projectStatus';

export const SessionVideosDrawer: React.FC = () => {
  const isOpen = useDubDubStore((s) => s.sessionDrawerOpen);
  const setSessionDrawerOpen = useDubDubStore((s) => s.setSessionDrawerOpen);
  const sessionSubprojects = useDubDubStore((s) => s.sessionSubprojects);
  const activeParentProjectId = useDubDubStore((s) => s.activeParentProjectId);
  const activeParentProjectName = useDubDubStore((s) => s.activeParentProjectName);
  const activeProjectId = useDubDubStore((s) => s.activeProjectId);
  const switchSessionVideo = useDubDubStore((s) => s.switchSessionVideo);
  const setScanModalOpen = useDubDubStore((s) => s.setScanModalOpen);
  const projectJobStatuses = useDubDubStore((s) => s.projectJobStatuses || {});
  const startBatchPrepareJobs = useDubDubStore((s) => s.startBatchPrepareJobs);
  const currentStep = useDubDubStore((s) => s.currentStep);

  if (!isOpen) return null;

  const stageLabels: Record<number, string> = {
    1: '1. Prepare & ASR',
    2: '2. Review Transcript',
    3: '3. Voice Dubbing',
    4: '4. Video Edit & Export',
  };

  const totalCount = sessionSubprojects.length;
  const isNotFailed = (s: { id: string; status?: string }) =>
    s.status !== 'failed' && projectJobStatuses[s.id]?.status !== 'failed';

  let readyCount = 0;
  let readinessLabel = '';
  if (currentStep === 4) {
    readyCount = sessionSubprojects.filter(
      (s) => isNotFailed(s) && (s.stage >= 4 || (s.status === 'completed' && s.stage >= 4))
    ).length;
    readinessLabel = `${readyCount} of ${totalCount} videos completed`;
  } else if (currentStep === 3) {
    readyCount = sessionSubprojects.filter(
      (s) =>
        isNotFailed(s) &&
        (s.stage >= 3 ||
          (projectJobStatuses[s.id]?.stage === 'voice' && projectJobStatuses[s.id]?.status === 'completed'))
    ).length;
    readinessLabel = `${readyCount} of ${totalCount} videos voice-ready`;
  } else {
    readyCount = sessionSubprojects.filter(
      (s) =>
        isNotFailed(s) &&
        (s.stage >= 2 ||
          (projectJobStatuses[s.id]?.status === 'completed' && projectJobStatuses[s.id]?.stage !== 'voice'))
    ).length;
    readinessLabel = `${readyCount} of ${totalCount} videos ready for review`;
  }

  const hasRunnableStage1 = sessionSubprojects.some(
    (s) =>
      (s.stage ?? 1) <= 1 &&
      s.status !== 'completed' &&
      projectJobStatuses[s.id]?.status !== 'completed' &&
      Boolean((s as any).media_id || (s as any).mediaId || s.media_path)
  );
  const isBatchAsrRunning = sessionSubprojects.some((s) => {
    const st = projectJobStatuses[s.id];
    return st && (st.status === 'running' || st.status === 'queued') && st.stage !== 'voice' && st.stage !== 'render';
  });

  return (
    <div className="fixed inset-y-0 right-0 z-40 w-96 max-w-full bg-white shadow-2xl border-l border-stone-200 flex flex-col animate-in slide-in-from-right duration-200 select-none">
      {/* Header */}
      <div className="h-14 px-4 border-b border-stone-200 flex items-center justify-between bg-stone-50/90 shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-amber-500/15 text-[#8D4B00] flex items-center justify-center border border-amber-500/25 shrink-0">
            <Layers className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h3 className="text-xs font-bold text-stone-900 truncate">
              Session: {activeParentProjectName || 'Daily Batch'}
            </h3>
            <p className="text-[10px] text-stone-500">
              {readinessLabel}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setScanModalOpen(true)}
            className="p-1.5 rounded-lg text-stone-500 hover:text-stone-900 hover:bg-stone-200/60 transition-colors cursor-pointer"
            title="Scan & Add More Folders"
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={() => setSessionDrawerOpen(false)}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-800 hover:bg-stone-200/60 transition-colors cursor-pointer"
            title="Close Panel"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Progress Bar Header */}
      {sessionSubprojects.length > 0 && (
        <div className="px-4 py-2 bg-stone-100/70 border-b border-stone-200 shrink-0">
          <div className="flex items-center justify-between text-[10px] font-semibold text-stone-600 mb-1">
            <span>Batch Readiness</span>
            <span>{Math.round((readyCount / (totalCount || 1)) * 100)}%</span>
          </div>
          <div className="w-full bg-stone-200 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-[#8D4B00] h-full transition-all duration-300"
              style={{
                width: `${Math.round((readyCount / (totalCount || 1)) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Video Items List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {sessionSubprojects.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center p-6 text-center text-stone-400">
            <Film className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-xs font-semibold text-stone-600">No videos in current batch</p>
            <p className="text-[11px] mt-1">Scan a root folder to detect and queue videos.</p>
            <button
              onClick={() => setScanModalOpen(true)}
              className="mt-4 px-3.5 py-1.5 rounded-xl bg-[#8D4B00] text-white text-xs font-bold shadow-xs hover:bg-[#743D00] flex items-center gap-1.5 cursor-pointer"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>Scan Folders</span>
            </button>
          </div>
        ) : (
          sessionSubprojects.map((sub, idx) => {
            const isActive = sub.id === activeProjectId;
            const status = getSubprojectStatusInfo(sub, projectJobStatuses, 'w-3 h-3');

            return (
              <div
                key={sub.id}
                onClick={() => {
                  if (!isActive) void switchSessionVideo(sub.id);
                }}
                className={`p-3 rounded-xl border transition-all relative ${
                  isActive
                    ? 'bg-amber-50/60 border-amber-300 shadow-xs ring-1 ring-amber-300/60 cursor-default'
                    : 'bg-white border-stone-200 hover:border-stone-300 hover:bg-stone-50/60 cursor-pointer'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2 min-w-0">
                    <span className="text-[10px] font-mono font-bold text-stone-400 mt-0.5">
                      #{idx + 1}
                    </span>
                    <div className="min-w-0">
                      <div className="font-bold text-xs text-stone-900 truncate" title={sub.name}>
                        {sub.name}
                      </div>
                      <div className="text-[10px] text-stone-500 flex items-center gap-2 mt-0.5">
                        <span>{stageLabels[sub.stage] || `Stage ${sub.stage}`}</span>
                        {sub.duration ? <span>• {Math.round(sub.duration)}s</span> : null}
                      </div>
                    </div>
                  </div>

                  {isActive && (
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#8D4B00] text-white uppercase tracking-wider shrink-0">
                      Active
                    </span>
                  )}
                </div>

                {/* Status & Progress bar */}
                <div className="mt-2.5 flex items-center justify-between">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold border ${status.pillClasses}`}
                  >
                    {status.icon}
                    <span>{status.label}</span>
                  </span>

                  <span className="text-[10px] text-stone-400 font-medium hover:text-[#8D4B00] flex items-center gap-0.5">
                    {isActive ? 'Editing' : 'Switch'} <ArrowRight className="w-2.5 h-2.5" />
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer Controls */}
      <div className="p-3 border-t border-stone-200 bg-stone-50/80 shrink-0 space-y-2">
        {currentStep === 1 && sessionSubprojects.length > 0 && (
          <button
            type="button"
            disabled={!hasRunnableStage1 || isBatchAsrRunning}
            onClick={() => startBatchPrepareJobs()}
            className="w-full py-2 rounded-xl bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center justify-center gap-1.5 shadow-xs cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Process Batch ASR (All Videos)</span>
          </button>
        )}
        <button
          type="button"
          onClick={() => setScanModalOpen(true)}
          className="w-full py-1.5 rounded-xl border border-stone-200 bg-white hover:bg-stone-50 text-stone-700 text-xs font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
        >
          <FolderOpen className="w-3.5 h-3.5 text-stone-500" />
          <span>Add More Videos / Folders</span>
        </button>
      </div>
    </div>
  );
};
