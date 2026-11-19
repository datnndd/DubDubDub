import React, { useState } from 'react';
import { useDubDubStore } from '../store';
import {
  X,
  Download,
  FileText,
  Music,
  FolderOpen,
  ExternalLink,
  CheckCircle2,
  Sparkles,
  ArrowRight,
  Layers,
} from 'lucide-react';
import { apiRequest } from '../api/client';

export function formatSrtTime(sec: number): string {
  const totalMs = Math.max(0, Math.round(Number(sec || 0) * 1000));
  const ms = totalMs % 1000;
  const totalSec = Math.floor(totalMs / 1000);
  const s = totalSec % 60;
  const totalMin = Math.floor(totalSec / 60);
  const m = totalMin % 60;
  const h = Math.floor(totalMin / 60);
  const pad = (n: number, z = 2) => String(n).padStart(z, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
}

export function generateSrtContent(
  segments: any[],
  field: 'targetText' | 'rawTranslation' | 'sourceText' = 'targetText'
): string {
  if (!Array.isArray(segments) || segments.length === 0) return '';
  return (
    segments
      .map((seg, idx) => {
        const text =
          field === 'rawTranslation'
            ? String(seg.rawTranslation ?? '').trim() || String(seg.targetText ?? '').trim() || ''
            : field === 'targetText'
            ? String(seg.targetText ?? '').trim() || String(seg.sourceText ?? '').trim() || ''
            : String(seg.sourceText ?? '').trim() || String(seg.targetText ?? '').trim() || '';
        const start = formatSrtTime(seg.startSec ?? 0);
        const end = formatSrtTime(seg.endSec ?? (seg.startSec ?? 0) + 1);
        return `${idx + 1}\n${start} --> ${end}\n${text.trim()}\n`;
      })
      .join('\n') + '\n'
  );
}

export function triggerBlobDownload(content: string, filename: string, mimeType: string = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const CapCutExportModal: React.FC = () => {
  const isOpen = useDubDubStore((s) => s.editVideo?.isCapCutModalOpen);
  const setCapCutModalOpen = useDubDubStore((s) => s.setCapCutModalOpen);
  const segments = useDubDubStore((s) => s.segments || []);
  const activeProjectId = useDubDubStore((s) => s.activeProjectId);

  const [downloadingZip, setDownloadingZip] = useState(false);
  const [folderStatus, setFolderStatus] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleDownloadEditedSrt = () => {
    const srt = generateSrtContent(segments, 'targetText');
    triggerBlobDownload(srt, 'subtitles_edited.srt');
  };

  const handleDownloadTargetSrt = () => {
    const srt = generateSrtContent(segments, 'rawTranslation');
    triggerBlobDownload(srt, 'subtitles_target.srt');
  };

  const handleDownloadAudio = () => {
    if (activeProjectId) {
      window.open(`/api/projects/${activeProjectId}/export-capcut/voiceover_merged.wav`, '_blank');
    } else {
      const firstAudio = segments.find((s) => s.previewAudioUrl)?.previewAudioUrl;
      if (firstAudio) {
        window.open(firstAudio, '_blank');
      } else {
        triggerBlobDownload('', 'voiceover_merged.wav', 'audio/wav');
      }
    }
  };

  const handleDownloadZip = async () => {
    setDownloadingZip(true);
    try {
      if (activeProjectId) {
        window.location.href = `/api/projects/${activeProjectId}/export-capcut/bundle.zip`;
      } else {
        // Fallback: download individual assets sequentially
        handleDownloadEditedSrt();
        setTimeout(handleDownloadTargetSrt, 300);
        setTimeout(handleDownloadAudio, 600);
      }
    } catch (err) {
      console.warn('ZIP download failed, downloading individual files:', err);
      handleDownloadEditedSrt();
      handleDownloadTargetSrt();
    } finally {
      setTimeout(() => setDownloadingZip(false), 1500);
    }
  };

  const handleLaunchCapCut = () => {
    // Try launching desktop protocol with fallback to web editor
    const timeout = setTimeout(() => {
      window.open('https://www.capcut.com/editor', '_blank');
    }, 1200);

    const handleBlur = () => {
      clearTimeout(timeout);
      window.removeEventListener('blur', handleBlur);
    };

    window.addEventListener('blur', handleBlur);
    window.location.href = 'capcut://';
  };

  const handleOpenFolder = async () => {
    if (!activeProjectId) {
      setFolderStatus('No active project saved on disk yet.');
      return;
    }
    try {
      setFolderStatus('Opening folder…');
      const resp = await apiRequest(`/api/projects/${activeProjectId}/open-folder`, { method: 'POST' });
      setFolderStatus(resp.path ? `Folder opened: ${resp.path}` : 'Folder opened');
      setTimeout(() => setFolderStatus(null), 4000);
    } catch {
      setFolderStatus('Unable to open folder automatically.');
      setTimeout(() => setFolderStatus(null), 3000);
    }
  };

  return (
    <div
      data-capcut-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-150"
    >
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-2xl border border-stone-200 overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-stone-200 flex items-center justify-between bg-stone-50/75 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#8D4B00] flex items-center justify-center text-white shadow-2xs">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-stone-900 flex items-center gap-2">
                <span>Continue Editing in CapCut</span>
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-amber-100 text-[#8D4B00]">
                  Studio Bridge
                </span>
              </h2>
              <p className="text-xs text-stone-500">
                Export your dubbed assets and seamless captions into CapCut for advanced transitions and effects.
              </p>
            </div>
          </div>
          <button
            data-capcut-close-btn="true"
            onClick={() => setCapCutModalOpen(false)}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-200/60 transition-colors cursor-pointer"
            title="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1">
          {/* Workflow Steps Card */}
          <div className="p-4 rounded-xl border border-amber-200/80 bg-amber-50/40 space-y-3">
            <h3 className="text-xs font-bold text-stone-900 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span>3-Step CapCut Workflow Guide</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-white border border-stone-200/80 space-y-1">
                <div className="font-bold text-[#8D4B00] flex items-center gap-1">
                  <span className="w-4 h-4 rounded-full bg-amber-100 text-[10px] flex items-center justify-center font-mono">1</span>
                  <span>Export Assets</span>
                </div>
                <p className="text-[11px] text-stone-600 leading-relaxed">
                  Download the bundle or grab the individual <span className="font-mono">.srt</span> and audio files below.
                </p>
              </div>

              <div className="p-2.5 rounded-lg bg-white border border-stone-200/80 space-y-1">
                <div className="font-bold text-[#8D4B00] flex items-center gap-1">
                  <span className="w-4 h-4 rounded-full bg-amber-100 text-[10px] flex items-center justify-center font-mono">2</span>
                  <span>Open CapCut</span>
                </div>
                <p className="text-[11px] text-stone-600 leading-relaxed">
                  Click Launch CapCut to open your desktop application or continue via CapCut Web editor.
                </p>
              </div>

              <div className="p-2.5 rounded-lg bg-white border border-stone-200/80 space-y-1">
                <div className="font-bold text-[#8D4B00] flex items-center gap-1">
                  <span className="w-4 h-4 rounded-full bg-amber-100 text-[10px] flex items-center justify-center font-mono">3</span>
                  <span>Import &amp; Style</span>
                </div>
                <p className="text-[11px] text-stone-600 leading-relaxed">
                  Import your original video and the downloaded voiceover. Go to <span className="font-semibold">Text &gt; Local Captions</span> and select <span className="font-mono">subtitles_edited.srt</span>.
                </p>
              </div>
            </div>
          </div>

          {/* Asset Downloads Grid */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-stone-800 uppercase tracking-wider">
                Exported Project Assets (3 Files)
              </h4>
              <button
                data-download-zip-btn="true"
                onClick={handleDownloadZip}
                disabled={downloadingZip}
                className="px-3 py-1.5 rounded-lg bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center gap-1.5 shadow-2xs transition-colors cursor-pointer disabled:opacity-50"
              >
                <Download className="w-3.5 h-3.5" />
                <span>{downloadingZip ? 'Packaging ZIP…' : 'Download All as ZIP'}</span>
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {/* Asset 1: Edited SRT */}
              <div className="p-3 rounded-xl border border-stone-200 bg-stone-50/50 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-amber-100 text-[#8D4B00] flex items-center justify-center shrink-0">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-stone-900 truncate">subtitles_edited.srt</div>
                    <div className="text-[10px] text-stone-500">Edited target subtitles</div>
                  </div>
                </div>
                <button
                  data-download-edited-srt="true"
                  onClick={handleDownloadEditedSrt}
                  className="px-2.5 py-1 rounded-lg border border-stone-200 bg-white hover:bg-stone-50 text-stone-700 text-xs font-semibold flex items-center gap-1 transition-colors cursor-pointer shrink-0"
                >
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </button>
              </div>

              {/* Asset 2: Target SRT */}
              <div className="p-3 rounded-xl border border-stone-200 bg-stone-50/50 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center shrink-0">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-stone-900 truncate">subtitles_target.srt</div>
                    <div className="text-[10px] text-stone-500">Raw target language subtitles</div>
                  </div>
                </div>
                <button
                  data-download-target-srt="true"
                  onClick={handleDownloadTargetSrt}
                  className="px-2.5 py-1 rounded-lg border border-stone-200 bg-white hover:bg-stone-50 text-stone-700 text-xs font-semibold flex items-center gap-1 transition-colors cursor-pointer shrink-0"
                >
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </button>
              </div>

              {/* Asset 3: Merged Audio */}
              <div className="p-3 rounded-xl border border-stone-200 bg-stone-50/50 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-700 flex items-center justify-center shrink-0">
                    <Music className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-stone-900 truncate">voiceover_merged.wav</div>
                    <div className="text-[10px] text-stone-500">Merged dubbed voice segments</div>
                  </div>
                </div>
                <button
                  data-download-audio="true"
                  onClick={handleDownloadAudio}
                  className="px-2.5 py-1 rounded-lg border border-stone-200 bg-white hover:bg-stone-50 text-stone-700 text-xs font-semibold flex items-center gap-1 transition-colors cursor-pointer shrink-0"
                >
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3.5 border-t border-stone-200 bg-stone-50 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <button
              data-open-folder-btn="true"
              onClick={handleOpenFolder}
              className="px-3 py-1.5 rounded-lg border border-stone-200 bg-white hover:bg-stone-100 text-stone-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <FolderOpen className="w-3.5 h-3.5 text-stone-500" />
              <span>Open Folder in Explorer</span>
            </button>
            {folderStatus && (
              <span className="text-[11px] text-stone-600 font-mono">{folderStatus}</span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              data-launch-capcut-btn="true"
              onClick={handleLaunchCapCut}
              className="px-4 py-1.5 rounded-lg bg-stone-900 hover:bg-black text-white text-xs font-bold flex items-center gap-1.5 shadow-2xs transition-colors cursor-pointer"
            >
              <span>Launch CapCut</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
