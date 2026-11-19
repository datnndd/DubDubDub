import React, { useRef, useState } from 'react';
import type { Segment, OcrSubtitleEntry } from '../types/segment';

type Roi = [number, number, number, number];

export function roiFromDrag(start: [number, number], end: [number, number]): Roi {
  const clamp = (value: number) => Math.max(0, Math.min(1, value));
  const x1 = clamp(Math.min(start[0], end[0]));
  const y1 = clamp(Math.min(start[1], end[1]));
  const x2 = clamp(Math.max(start[0], end[0]));
  const y2 = clamp(Math.max(start[1], end[1]));
  return [x1, y1, x2 - x1, y2 - y1];
}

export interface BatchOcrResult {
  segment: Segment;
  text?: string;
  confidence?: number;
  error?: string;
  applied?: boolean;
}

interface Props {
  segment: Segment;
  batchSegments?: Segment[];
  batchResults?: BatchOcrResult[];
  currentSegments?: Segment[];
  progress?: number;
  videoUrl: string;
  roi: Roi;
  onRoiChange: (roi: Roi) => void;
  onExtract: () => void;
  onConfirmReplace?: (entries: OcrSubtitleEntry[]) => void;
  onConfirmBatchApply?: () => void;
  onClose: () => void;
  loading: boolean;
  entries?: OcrSubtitleEntry[] | null;
  result?: { text: string; confidence?: number; applied?: boolean } | null;
  error: string | null;
}

export function normalizePointerPoint(
  clientX: number,
  clientY: number,
  bounds: { left: number; top: number; width: number; height: number } | null | undefined
): [number, number] | null {
  if (!bounds || bounds.width <= 0 || bounds.height <= 0) return null;
  const x = Math.max(0, Math.min(1, (clientX - bounds.left) / bounds.width));
  const y = Math.max(0, Math.min(1, (clientY - bounds.top) / bounds.height));
  return [x, y];
}

export const Stage2OcrDialog: React.FC<Props> = ({
  segment, batchSegments, batchResults = [], currentSegments = [], progress = 0,
  videoUrl, roi, onRoiChange, onExtract, onConfirmReplace, onConfirmBatchApply, onClose, loading, entries, result, error,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const dragStart = useRef<[number, number] | null>(null);
  const [dragRoi, setDragRoi] = useState<Roi | null>(null);
  const [previewError, setPreviewError] = useState(false);
  const isBatch = (batchSegments?.length || 0) > 1;
  const activePreviewSeg = segment;
  const minPreviewSec = Number(activePreviewSeg.startSec || 0);
  const maxPreviewSec = Math.max(minPreviewSec + 0.04, Number(activePreviewSeg.endSec || minPreviewSec + 1));
  const [previewTime, setPreviewTime] = useState<number>(() => (minPreviewSec + maxPreviewSec) / 2);

  const isCurrent = (entry: BatchOcrResult) => currentSegments.some((current) =>
    current.id === entry.segment.id && current.startSec === entry.segment.startSec && current.endSec === entry.segment.endSec);
  const visibleRoi = dragRoi ?? roi;

  const point = (event: React.PointerEvent<HTMLDivElement>): [number, number] | null => {
    if (!videoRef.current) return null;
    const bounds = videoRef.current.getBoundingClientRect();
    return normalizePointerPoint(event.clientX, event.clientY, bounds);
  };

  const seekPreviewTo = (nextTime: number) => {
    const clamped = Math.max(minPreviewSec, Math.min(maxPreviewSec, nextTime));
    setPreviewTime(clamped);
    if (videoRef.current) {
      videoRef.current.currentTime = clamped;
    }
  };

  const hasPendingBatchApply =
    isBatch &&
    Boolean(onConfirmBatchApply) &&
    batchResults.some((entry) => !entry.applied && !entry.error && Boolean(entry.text?.trim()));

  const changeCoordinate = (index: number, percent: number) => {
    if (!Number.isFinite(percent)) return;
    const next = [...roi] as Roi;
    next[index] = Math.max(0, Math.min(1, percent / 100));
    if (next[0] + next[2] > 1) next[index === 0 ? 2 : 0] = 1 - next[index === 0 ? 0 : 2];
    if (next[1] + next[3] > 1) next[index === 1 ? 3 : 1] = 1 - next[index === 1 ? 1 : 3];
    onRoiChange(next);
  };

  return (
    <div role="dialog" aria-modal="true" aria-label="Extract text from video" data-testid="stage2-ocr-dialog"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl bg-white p-5 shadow-2xl space-y-3">
        <div className="flex justify-between gap-3">
          <div>
            <h2 className="font-bold text-stone-900">{isBatch ? `OCR Extract · ${batchSegments?.length} selected segments` : `OCR Extract · Segment ${segment.id}`}</h2>
            <p className="text-xs text-stone-600">Select the subtitle region, then run OCR. Recognized text replaces each matching source subtitle automatically.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close OCR dialog" className="text-stone-600 hover:text-stone-900">✕</button>
        </div>
        {videoUrl ? (
          <div className="space-y-2">
            <div className="flex justify-center bg-black rounded-lg overflow-hidden">
              <div className="relative inline-block touch-none select-none"
                onPointerDown={(event) => {
                  const pt = point(event);
                  if (!pt) return;
                  dragStart.current = pt;
                  event.currentTarget.setPointerCapture?.(event.pointerId);
                }}
                onPointerMove={(event) => {
                  if (!dragStart.current) return;
                  const pt = point(event);
                  if (!pt) return;
                  const next = roiFromDrag(dragStart.current, pt);
                  if (next[2] > 0.01 && next[3] > 0.01) setDragRoi(next);
                }}
                onPointerUp={(event) => {
                  if (dragStart.current) {
                    const pt = point(event);
                    if (pt) {
                      const next = roiFromDrag(dragStart.current, pt);
                      if (next[2] > 0.01 && next[3] > 0.01) onRoiChange(next);
                    }
                  }
                  dragStart.current = null;
                  setDragRoi(null);
                }}
                onPointerCancel={() => { dragStart.current = null; setDragRoi(null); }}
              >
                <video ref={videoRef} src={videoUrl} className="block max-w-full max-h-[45vh]"
                  muted playsInline preload="auto"
                  onLoadedMetadata={(event) => {
                    const mid = (minPreviewSec + maxPreviewSec) / 2;
                    setPreviewTime(mid);
                    event.currentTarget.currentTime = mid;
                  }}
                  onTimeUpdate={(event) => {
                    setPreviewTime(event.currentTarget.currentTime);
                  }}
                  onError={() => setPreviewError(true)}
                />
                <div data-testid="ocr-roi-overlay" className="absolute border-2 border-amber-400 bg-amber-300/20 pointer-events-none"
                  style={{ left: `${visibleRoi[0] * 100}%`, top: `${visibleRoi[1] * 100}%`, width: `${visibleRoi[2] * 100}%`, height: `${visibleRoi[3] * 100}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg bg-stone-50 border border-stone-200 px-2.5 py-1.5 text-[11px]" data-testid="ocr-scrubber-controls">
              <button
                type="button"
                data-testid="ocr-frame-prev-btn"
                onClick={() => seekPreviewTo(previewTime - 0.04)}
                className="px-2 py-0.5 rounded bg-white border border-stone-300 text-stone-700 hover:bg-stone-100 font-semibold cursor-pointer"
                title="Step back 1 frame (-0.04s)"
              >
                -1f
              </button>
              <input
                type="range"
                data-testid="ocr-frame-scrubber"
                aria-label="OCR preview time scrubber"
                min={minPreviewSec}
                max={maxPreviewSec}
                step={0.04}
                value={Math.max(minPreviewSec, Math.min(maxPreviewSec, previewTime))}
                onChange={(e) => seekPreviewTo(Number(e.target.value))}
                className="flex-1 accent-[#8D4B00] cursor-pointer"
              />
              <button
                type="button"
                data-testid="ocr-frame-next-btn"
                onClick={() => seekPreviewTo(previewTime + 0.04)}
                className="px-2 py-0.5 rounded bg-white border border-stone-300 text-stone-700 hover:bg-stone-100 font-semibold cursor-pointer"
                title="Step forward 1 frame (+0.04s)"
              >
                +1f
              </button>
              <span className="font-mono text-[10px] text-stone-600 shrink-0">
                {previewTime.toFixed(2)}s ({minPreviewSec.toFixed(2)}–{maxPreviewSec.toFixed(2)}s)
              </span>
            </div>
          </div>
        ) : <p role="alert" className="text-sm text-rose-700">No video preview is available for OCR.</p>}
        {previewError && <p role="alert" className="text-xs text-rose-700">The video frame could not be displayed. Check that the uploaded video is still available.</p>}
        <div className="grid grid-cols-4 gap-2">
          {(['Left', 'Top', 'Width', 'Height'] as const).map((label, index) => (
            <label key={label} className="text-xs text-stone-700">{label} %
              <input type="number" min="0" max="100" step="1" aria-label={`${label} percent`}
                value={Math.round(roi[index] * 100)} onChange={(event) => changeCoordinate(index, Number(event.target.value))}
                className="w-full rounded border border-stone-300 p-1" />
            </label>
          ))}
        </div>
        <button type="button" onClick={onExtract} disabled={loading || !videoUrl || roi[2] <= 0 || roi[3] <= 0}
          className="rounded bg-[#8D4B00] px-3 py-1.5 text-xs font-bold text-white disabled:opacity-50">
          {loading ? `Reading video frames…${isBatch ? ` ${progress}/${batchSegments?.length}` : ''}` : isBatch ? 'Extract Selected Text' : 'Extract Text'}
        </button>
        {error && <p role="alert" className="text-xs text-rose-700">{error}</p>}
        {isBatch && batchResults.length > 0 && (
          <div className="space-y-3 border-t border-stone-200 pt-3" data-testid="ocr-batch-results">
            {hasPendingBatchApply && (
              <div className="flex items-center justify-between gap-2 rounded-lg border border-amber-200 bg-amber-50/70 px-3 py-2">
                <span className="text-xs font-semibold text-amber-950">
                  Review extracted batch text before replacing source subtitles.
                </span>
                <button
                  type="button"
                  data-testid="confirm-apply-batch-ocr-btn"
                  onClick={onConfirmBatchApply}
                  className="rounded bg-emerald-700 hover:bg-emerald-800 px-3 py-1.5 text-xs font-bold text-white shadow-2xs cursor-pointer shrink-0"
                >
                  Confirm &amp; Apply OCR
                </button>
              </div>
            )}
            {batchResults.map((entry) => {
              const current = isCurrent(entry);
              return <div key={entry.segment.id} data-testid={`ocr-result-${entry.segment.id}`} className="rounded border border-stone-200 p-2 space-y-1">
                <p className="text-xs font-semibold text-stone-700">Segment {entry.segment.id} · {entry.segment.startSec.toFixed(2)}s–{entry.segment.endSec.toFixed(2)}s
                  {entry.confidence != null && !entry.error ? ` · ${Math.round(entry.confidence * 100)}% confidence` : ''}</p>
                {!current && <p role="alert" className="text-xs text-amber-800">This segment or its timing changed. Run OCR again for this segment.</p>}
                {entry.error && <p role="alert" className="text-xs text-rose-700">{entry.error}</p>}
                {entry.text ? <><p className="text-sm text-stone-800 whitespace-pre-wrap">{entry.text}</p>
                  {!entry.error && (
                    <p className={`text-xs ${entry.applied ? 'text-emerald-700' : 'text-amber-800'}`}>
                      {entry.applied
                        ? 'Source subtitle updated and saved.'
                        : entry.applied === false
                        ? 'Pending confirmation — click Confirm & Apply OCR to apply.'
                        : 'Source subtitle unchanged.'}
                    </p>
                  )}</>
                  : !entry.error && <p className="text-xs text-stone-600">No text was found in this time range.</p>}
              </div>;
            })}
          </div>
        )}
        {!isBatch && entries && entries.length > 0 && (
          <div className="space-y-3 border-t border-stone-200 pt-3" data-testid="ocr-detected-entries">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <div>
                <p className="text-xs font-bold text-stone-900">
                  Detected Subtitles ({entries.length} {entries.length === 1 ? 'entry' : 'entries'})
                </p>
                <p className="text-[11px] text-stone-500">
                  Recognized subtitles automatically align with the timeline. Click below to replace this ASR segment.
                </p>
              </div>
              {onConfirmReplace && (
                <button
                  type="button"
                  data-testid="confirm-replace-ocr-btn"
                  onClick={() => onConfirmReplace(entries)}
                  className="rounded bg-emerald-700 hover:bg-emerald-800 px-3 py-1.5 text-xs font-bold text-white shadow-2xs cursor-pointer flex items-center gap-1.5"
                >
                  <span>Confirm &amp; Replace</span>
                </button>
              )}
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {entries.map((entry, idx) => (
                <div key={idx} className="rounded-lg border border-amber-200 bg-amber-50/50 p-2.5 text-xs space-y-1">
                  <div className="flex items-center justify-between font-mono text-[10px] text-stone-500">
                    <span className="font-semibold text-[#8D4B00]">
                      #{idx + 1} · {entry.startTime} ➔ {entry.endTime}
                    </span>
                    <span>
                      {(entry.endSec - entry.startSec).toFixed(2)}s
                      {entry.confidence != null ? ` · ${Math.round(entry.confidence * 100)}% confidence` : ''}
                    </span>
                  </div>
                  <p className="text-stone-900 font-medium text-xs whitespace-pre-wrap">{entry.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {!isBatch && entries && entries.length === 0 && !loading && (
          <div className="space-y-2 border-t border-stone-200 pt-3">
            <p className="text-xs text-stone-600">No text was found in this region. Adjust the crop and try again.</p>
            <p className="text-[11px] text-amber-800">The original ASR segment is preserved in the timeline automatically.</p>
          </div>
        )}

        {!isBatch && !entries && result && (
          <div className="space-y-2 border-t border-stone-200 pt-3">
            <p className="text-xs text-stone-600">Recognized text{result.confidence != null ? ` · ${Math.round(result.confidence * 100)}% confidence` : ''}</p>
            {result.text ? (
              <>
                <p className="text-sm text-stone-800 whitespace-pre-wrap">{result.text}</p>
                {!error && <p className="text-xs text-emerald-700">{result.applied ? 'Source subtitle updated and saved.' : 'Source subtitle unchanged.'}</p>}
              </>
            ) : <p className="text-xs text-stone-600">No text was found in this region. Adjust the crop and try again.</p>}
          </div>
        )}
      </div>
    </div>
  );
};
