import React from 'react';
import {
  CheckCircle2,
  Loader2,
  AlertCircle,
  Clock,
} from 'lucide-react';

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
  projectJobStatuses: Record<string, any> = {},
  iconClass: string = 'w-2.5 h-2.5'
): SubprojectStatusInfo {
  const jobState = projectJobStatuses[sub.id];

  if (jobState) {
    if (jobState.status === 'running') {
      const isVoice = jobState.stage === 'voice';
      const label = isVoice
        ? (jobState.progress != null && jobState.progress > 0
            ? `Voice: ${jobState.progress}%`
            : 'Creating Voice…')
        : `ASR: ${jobState.progress != null ? Number(jobState.progress).toFixed(0) : '0'}%`;
      return {
        label,
        badgeText: isVoice
          ? (jobState.progress != null && jobState.progress > 0 ? `${jobState.progress}%` : 'Voice…')
          : (jobState.progress != null ? `${Number(jobState.progress).toFixed(0)}%` : 'ASR…'),
        pillClasses: 'bg-amber-100 text-[#8D4B00] border-amber-300 animate-pulse',
        icon: <Loader2 className={`${iconClass} animate-spin text-amber-600`} />,
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
        icon: <CheckCircle2 className={`${iconClass} text-emerald-600`} />,
        status: 'completed',
        isVoice,
      };
    }
    if (jobState.status === 'failed') {
      return {
        label: jobState.stage === 'voice' ? 'Voice Failed' : 'Failed',
        badgeText: 'Failed',
        pillClasses: 'bg-rose-100 text-rose-800 border-rose-300',
        icon: <AlertCircle className={`${iconClass} text-rose-600`} />,
        status: 'failed',
      };
    }
    if (jobState.status === 'queued') {
      return {
        label: 'Queued',
        badgeText: 'Queued',
        pillClasses: 'bg-stone-100 text-stone-700 border-stone-300',
        icon: <Clock className={`${iconClass} text-stone-500`} />,
        status: 'queued',
      };
    }
  }

  if (sub.status === 'failed') {
    return {
      label: 'Failed',
      badgeText: 'Failed',
      pillClasses: 'bg-rose-100 text-rose-800 border-rose-300',
      icon: <AlertCircle className={`${iconClass} text-rose-600`} />,
      status: 'failed',
    };
  }

  const stage = sub.stage || 1;
  if (stage >= 2 || sub.status === 'completed') {
    const isStage3or4 = stage >= 3;
    return {
      label: stage >= 4 ? 'Completed' : isStage3or4 ? 'Voice Ready' : 'Ready for Review',
      badgeText: stage >= 4 ? 'Done' : isStage3or4 ? 'Voice Ready' : 'Ready',
      pillClasses: 'bg-emerald-100 text-emerald-800 border-emerald-300',
      icon: <CheckCircle2 className={`${iconClass} text-emerald-600`} />,
      status: 'completed',
    };
  }
  if (sub.status === 'processing') {
    return {
      label: 'Processing…',
      badgeText: 'Working…',
      pillClasses: 'bg-amber-100 text-[#8D4B00] border-amber-300 animate-pulse',
      icon: <Loader2 className={`${iconClass} animate-spin text-amber-600`} />,
      status: 'running',
    };
  }

  return {
    label: 'Pending',
    badgeText: 'Pending',
    pillClasses: 'bg-stone-100 text-stone-600 border-stone-200',
    icon: <Clock className={`${iconClass} text-stone-400`} />,
    status: 'pending',
  };
}
