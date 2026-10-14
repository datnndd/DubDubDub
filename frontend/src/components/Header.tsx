import React from 'react';
import { useDubDubStore } from '../store';
import {
  Waves,
  Video,
  Mic,
  FolderOpen,
  Sparkles,
  Settings as SettingsIcon,
  ArrowLeft,
  Layers,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Clock,
} from 'lucide-react';
import type { ProjectRecord } from '../types/project';

export interface SubprojectStatusInfo {
  label: string;
  badgeText: string;
  pillClasses: string;
  icon: React.ReactNode;
  status: 'running' | 'completed' | 'failed' | 'queued' | 'pending';
  isVoice?: boolean;
}

export function getSubprojectStatusInfo(
  sub: { id: string; stage?: number; status?: string; name?: string },
  projectJobStatuses: Record<string, any> = {}
): SubprojectStatusInfo {
  const jobState = projectJobStatuses[sub.id];

  if (jobState) {
    if (jobState.status === 'running') {
      const isVoice = jobState.stage === 'voice';
      const label = isVoice
        ? (jobState.progress != null && jobState.progress > 0
            ? `Voice: ${jobState.progress}%`
            : 'Creating Voice…')
        : `ASR: ${jobState.progress != null ? jobState.progress.toFixed(0) : '0'}%`;
      return {
        label,
        badgeText: isVoice
          ? (jobState.progress != null && jobState.progress > 0 ? `${jobState.progress}%` : 'Voice…')
          : (jobState.progress != null ? `${jobState.progress.toFixed(0)}%` : 'ASR…'),
        pillClasses: 'bg-amber-100 text-[#8D4B00] border-amber-300 animate-pulse',
        icon: <Loader2 className="w-2.5 h-2.5 animate-spin text-amber-600" />,
        status: 'running',
        isVoice,
      };
    }
    if (jobState.status === 'completed') {
      const isVoice = jobState.stage === 'voice';
      return {
        label: isVoice ? 'Voice Ready' : 'Ready for Review',
        badgeText: isVoice ? 'Voice Ready' : 'Ready',
        pillClasses: 'bg-emerald-100 text-emerald-800 border-emerald-300',
        icon: <CheckCircle2 className="w-2.5 h-2.5 text-emerald-600" />,
        status: 'completed',
        isVoice,
      };
    }
    if (jobState.status === 'failed') {
      return {
        label: jobState.stage === 'voice' ? 'Voice Failed' : 'Failed',
        badgeText: 'Failed',
        pillClasses: 'bg-rose-100 text-rose-800 border-rose-300',
        icon: <AlertCircle className="w-2.5 h-2.5 text-rose-600" />,
        status: 'failed',
      };
    }
    if (jobState.status === 'queued') {
      return {
        label: 'Queued',
        badgeText: 'Queued',
        pillClasses: 'bg-stone-100 text-stone-700 border-stone-300',
        icon: <Clock className="w-2.5 h-2.5 text-stone-500" />,
        status: 'queued',
      };
    }
  }

  const stage = sub.stage || 1;
  if (stage >= 2 || sub.status === 'completed') {
    const isStage3or4 = stage >= 3;
    return {
      label: stage >= 4 ? 'Completed' : isStage3or4 ? 'Voice Ready' : 'Ready for Review',
      badgeText: stage >= 4 ? 'Done' : isStage3or4 ? 'Voice Ready' : 'Ready',
      pillClasses: 'bg-emerald-100 text-emerald-800 border-emerald-300',
      icon: <CheckCircle2 className="w-2.5 h-2.5 text-emerald-600" />,
      status: 'completed',
    };
  }
  if (sub.status === 'processing') {
    return {
      label: 'Processing…',
      badgeText: 'Working…',
      pillClasses: 'bg-amber-100 text-[#8D4B00] border-amber-300 animate-pulse',
      icon: <Loader2 className="w-2.5 h-2.5 animate-spin text-amber-600" />,
      status: 'running',
    };
  }

  return {
    label: 'Pending',
    badgeText: 'Pending',
    pillClasses: 'bg-stone-100 text-stone-600 border-stone-200',
    icon: <Clock className="w-2.5 h-2.5 text-stone-400" />,
    status: 'pending',
  };
}

export const Header: React.FC = () => {
  const activeView = useDubDubStore((s) => s.activeView);
  const setActiveView = useDubDubStore((s) => s.setActiveView);
  const setStep = useDubDubStore((s) => s.setStep);
  const project = useDubDubStore((s) => s.project);
  const projectsList = useDubDubStore((s) => s.projectsList);
  const openSettings = useDubDubStore((s) => s.openSettings);
  const sessionSubprojects = useDubDubStore((s) => s.sessionSubprojects || []);
  const sessionDrawerOpen = useDubDubStore((s) => s.sessionDrawerOpen);
  const setSessionDrawerOpen = useDubDubStore((s) => s.setSessionDrawerOpen);

  return (
    <header className="h-[50px] flex-shrink-0 bg-white border-b border-[#E7E4DC] px-4 flex items-center justify-between gap-4 z-30 shadow-2xs">
      {/* Left: DubDub Branding & Primary Navigation Tabs */}
      <div className="flex items-center gap-3 min-w-0">
        <div
          className="flex items-center gap-2 cursor-pointer"
          onClick={() => {
            setActiveView('dubbing');
            setStep(1);
          }}
          title="DubDub Home - Video Dubbing"
        >
          <div className="w-8 h-8 rounded-lg bg-amber-500/15 text-[#8D4B00] flex items-center justify-center border border-amber-500/25 shadow-2xs">
            <Waves className="w-4 h-4" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="font-bold text-xs tracking-tight text-stone-900">DubDub</span>
            <span className="text-[10px] text-stone-400 font-medium mt-0.5">Voice &amp; Subtitles Studio</span>
          </div>
        </div>

        <div className="h-4 w-px bg-stone-200" />

        {/* Top-Level Navigation Tabs */}
        <nav className="flex items-center gap-1 p-0.5 bg-stone-100/80 rounded-lg border border-stone-200/80 text-[11px]" aria-label="Main Navigation">
          <button
            data-tab="projects"
            data-testid="nav-tab-projects"
            onClick={() => setActiveView('projects')}
            className={`px-2.5 py-1 rounded-md font-semibold flex items-center gap-1.5 leading-none transition-colors cursor-pointer ${
              activeView === 'projects'
                ? 'bg-white text-[#8D4B00] font-bold shadow-2xs border border-amber-200/80'
                : 'text-stone-600 hover:text-stone-900 hover:bg-stone-200/50'
            }`}
          >
            <FolderOpen className={`w-3.5 h-3.5 ${activeView === 'projects' ? 'text-[#8D4B00]' : 'text-stone-400'}`} />
            <span>Projects</span>
            {projectsList?.length > 0 && (
              <span className={`px-1.5 py-0.2 rounded-full text-[9px] font-bold ${
                activeView === 'projects' ? 'bg-amber-100 text-[#8D4B00]' : 'bg-stone-200 text-stone-600'
              }`}>
                {projectsList.length}
              </span>
            )}
          </button>

          <button
            data-tab="dubbing"
            data-testid="nav-tab-dubbing"
            onClick={() => setActiveView('dubbing')}
            className={`px-2.5 py-1 rounded-md font-semibold flex items-center gap-1.5 leading-none transition-colors cursor-pointer ${
              activeView === 'dubbing'
                ? 'bg-white text-[#8D4B00] font-bold shadow-2xs border border-amber-200/80'
                : 'text-stone-600 hover:text-stone-900 hover:bg-stone-200/50'
            }`}
          >
            <Video className={`w-3.5 h-3.5 ${activeView === 'dubbing' ? 'text-[#8D4B00]' : 'text-stone-400'}`} />
            <span>Video Dub</span>
          </button>

          <button
            data-tab="voices"
            data-testid="nav-tab-voices"
            onClick={() => setActiveView('voices')}
            className={`px-2.5 py-1 rounded-md font-semibold flex items-center gap-1.5 leading-none transition-colors cursor-pointer ${
              activeView === 'voices'
                ? 'bg-white text-[#8D4B00] font-bold shadow-2xs border border-amber-200/80'
                : 'text-stone-600 hover:text-stone-900 hover:bg-stone-200/50'
            }`}
          >
            <Mic className={`w-3.5 h-3.5 ${activeView === 'voices' ? 'text-[#8D4B00]' : 'text-stone-400'}`} />
            <span>Voice Management</span>
          </button>
        </nav>
      </div>

      {/* Center: Clean Project Indicator */}
      <div className="hidden lg:flex items-center gap-2 text-[11px] text-stone-600 min-w-0">
        <span className="w-2 h-2 rounded-full bg-[#8D4B00] shrink-0" />
        <span className="font-medium text-stone-700 truncate max-w-xs">
          Project: <strong className="text-stone-900">{project?.filename || 'Untitled'}</strong>
        </span>
      </div>

      {/* Right: Back to Projects, Settings, Auto-save status */}
      <div className="flex items-center gap-2.5 flex-shrink-0">
        {activeView !== 'projects' && (
          <button
            onClick={() => setActiveView('projects')}
            data-testid="header-back-to-projects-btn"
            className="px-2.5 py-1 text-xs font-semibold text-stone-700 hover:text-[#8D4B00] bg-stone-100/90 hover:bg-amber-50 border border-stone-200 hover:border-amber-300 rounded-md transition-colors flex items-center gap-1.5 leading-none cursor-pointer shadow-2xs"
            title="Back to Projects"
          >
            <ArrowLeft className="w-3.5 h-3.5 text-stone-500" />
            <span>Back to Projects</span>
          </button>
        )}

        <button
          onClick={() => setSessionDrawerOpen(!sessionDrawerOpen)}
          data-testid="header-session-drawer-toggle"
          className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-colors flex items-center gap-1.5 leading-none cursor-pointer border shadow-2xs ${
            sessionDrawerOpen
              ? 'bg-amber-100 text-[#8D4B00] border-amber-300 font-bold'
              : 'text-stone-700 hover:text-[#8D4B00] bg-stone-100/90 hover:bg-amber-50 border-stone-200'
          }`}
          title="Toggle Session Videos Drawer"
        >
          <Layers className="w-3.5 h-3.5 text-[#8D4B00]" />
          <span>Session Videos</span>
          {sessionSubprojects.length > 0 && (
            <span className="px-1.5 py-0.2 rounded-full text-[9px] font-bold bg-[#8D4B00] text-white">
              {sessionSubprojects.length}
            </span>
          )}
        </button>

        <button
          onClick={() => openSettings()}
          className="px-2.5 py-1 rounded-md font-semibold text-stone-700 hover:text-stone-900 bg-stone-100/90 hover:bg-stone-200/80 border border-stone-200 transition-colors flex items-center gap-1.5 leading-none cursor-pointer shadow-2xs"
          title="Open Settings & Preferences"
        >
          <SettingsIcon className="w-3.5 h-3.5 text-[#8D4B00]" />
          <span>Settings</span>
        </button>

        <div className="hidden sm:flex items-center gap-1 text-[11px] text-stone-500 font-medium">
          <Sparkles className="w-3 h-3 text-stone-400" />
          <span>Auto-saved {project?.lastSaved || 'Just now'}</span>
        </div>

        <div className="flex items-center gap-2 px-2.5 py-1 bg-stone-50 rounded-lg border border-stone-200 shadow-2xs">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 warm-pulse" />
          <span className="font-mono text-[10px] font-semibold text-stone-700">Warm GPU</span>
        </div>
      </div>
    </header>
  );
};
