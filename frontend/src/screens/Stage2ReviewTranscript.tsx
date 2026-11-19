import React, { useEffect, useRef, useState } from 'react';
import { useDubDubStore } from '../store';
import { VideoPlayer } from '../components/VideoPlayer';
import {
  Search,
  Split,
  Trash2,
  Merge,
  Sparkles,
  Languages,
  Check,
  X,
  Loader2,
  Settings2,
  Key,
  Layers,
  Clock,
  AlignLeft,
  Globe,
  ArrowRight,
  Download,
} from 'lucide-react';
import { ProviderDocLink } from '../components/ProviderDocLink';
import { generateSrtContent, triggerBlobDownload } from '../components/CapCutExportModal';
import { Stage2OcrDialog, type BatchOcrResult } from '../components/Stage2OcrDialog';
import type { Segment, OcrSubtitleEntry } from '../types/segment';

export function untranslatedSegmentNumbers(segments: Array<{ targetText?: string }>): number[] {
  return segments.flatMap((segment, index) => segment.targetText?.trim() ? [] : [index + 1]);
}

export function exportTranslatedSrt(segments: Array<{ targetText?: string; startSec: number; endSec: number }>): void {
  if (!segments.length || untranslatedSegmentNumbers(segments).length) {
    throw new Error('Translate every segment before exporting the SRT');
  }
  triggerBlobDownload(generateSrtContent(segments, 'targetText'), 'translated_subtitles.srt');
}

export function selectedOcrSegments(segments: Segment[], selectedIds: number[]): Segment[] {
  const selected = new Set(selectedIds);
  return segments.filter((segment) => selected.has(segment.id));
}

export async function applyExtractedOcrResult(
  entry: BatchOcrResult,
  currentSegments: () => Segment[],
  apply: (id: number, text: string, confidence?: number) => Promise<void>,
): Promise<BatchOcrResult> {
  if (entry.error || !entry.text?.trim()) return { ...entry, applied: false };
  const current = currentSegments().find((segment) => segment.id === entry.segment.id);
  if (!current || current.startSec !== entry.segment.startSec || current.endSec !== entry.segment.endSec) {
    return { ...entry, applied: false, error: 'This segment or its timing changed. Run OCR again.' };
  }
  if (current.sourceText === entry.text.trim()) return { ...entry, applied: false };
  try {
    await apply(entry.segment.id, entry.text, entry.confidence);
    return { ...entry, applied: true };
  } catch (error) {
    return { ...entry, applied: false, error: `Source text changed, but the project could not be saved: ${error instanceof Error ? error.message : 'Unknown error'}` };
  }
}

export async function extractSelectedOcrSegments(
  snapshot: Segment[],
  roi: [number, number, number, number],
  extract: (id: number, roi: [number, number, number, number], interval: Segment) => Promise<{ text: string; confidence?: number }>,
  onResult: (result: BatchOcrResult) => Promise<void> | void,
  stillActive: () => boolean,
): Promise<void> {
  for (const segment of snapshot) {
    if (!stillActive()) return;
    let entry: BatchOcrResult;
    try {
      entry = { segment, ...await extract(segment.id, roi, segment) };
    } catch (error) {
      entry = { segment, error: error instanceof Error ? error.message : 'OCR extraction failed' };
    }
    if (!stillActive()) return;
    await onResult(entry);
  }
}

const FALLBACK_TRANSLATION_PROVIDERS = [
  { id: 'google', label: 'Google Translate', translateType: 0, requiresSettings: false, configured: true },
  { id: 'openai', label: 'OpenAI ChatGPT', translateType: 1, requiresSettings: true, configured: false },
  { id: 'gemini', label: 'Gemini', translateType: 2, requiresSettings: true, configured: false },
  { id: 'deepseek', label: 'DeepSeek', translateType: 3, requiresSettings: true, configured: false },
];

const FALLBACK_TRANSLATION_MODES = [
  { id: 'srt', label: 'Send SRT', description: 'Send subtitle blocks with timestamps and structure.' },
  { id: 'text', label: 'Batch Text', description: 'Send plain subtitle text in batches.' },
];

const FALLBACK_LANGUAGES = [
  { code: 'zh-cn', name: 'Simplified Chinese' },
  { code: 'zh-tw', name: 'Traditional Chinese' },
  { code: 'en', name: 'English' },
  { code: 'vi', name: 'Vietnamese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'es', name: 'Spanish' },
  { code: 'ru', name: 'Russian' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'th', name: 'Thai' },
  { code: 'id', name: 'Indonesian' },
];

export const Stage2ReviewTranscript: React.FC = () => {
  const segments = useDubDubStore((s) => s.segments);
  const activeSegmentId = useDubDubStore((s) => s.activeSegmentId);
  const setActiveSegmentId = useDubDubStore((s) => s.setActiveSegmentId);
  const seekAndPlay = useDubDubStore((s) => s.seekAndPlay);
  const searchQuery = useDubDubStore((s) => s.searchQuery);
  const setSearchQuery = useDubDubStore((s) => s.setSearchQuery);
  const updateSegmentText = useDubDubStore((s) => s.updateSegmentText);
  const splitSegment = useDubDubStore((s) => s.splitSegment);
  const deleteSegment = useDubDubStore((s) => s.deleteSegment);
  const mergeWithNextSegment = useDubDubStore((s) => s.mergeWithNextSegment);
  const extractOcrForSegment = useDubDubStore((s) => s.extractOcrForSegment);
  const applyOcrText = useDubDubStore((s) => s.applyOcrText);
  const replaceSegmentWithOcrEntries = useDubDubStore((s) => s.replaceSegmentWithOcrEntries);
  const ocrRoi = useDubDubStore((s) => s.ocrCrop.roi);
  const setOcrCropRoi = useDubDubStore((s) => s.setOcrCropRoi);
  const project = useDubDubStore((s) => s.project);
  const activeProjectId = useDubDubStore((s) => s.activeProjectId);
  const [selectedOcrIds, setSelectedOcrIds] = useState<number[]>([]);
  const [ocrSelectionHint, setOcrSelectionHint] = useState(false);
  const [ocrDialogSegments, setOcrDialogSegments] = useState<Segment[] | null>(null);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrError, setOcrError] = useState<string | null>(null);
  const [ocrResult, setOcrResult] = useState<{ text: string; confidence?: number; applied?: boolean } | null>(null);
  const [ocrDetectedEntries, setOcrDetectedEntries] = useState<OcrSubtitleEntry[] | null>(null);
  const [ocrBatchResults, setOcrBatchResults] = useState<BatchOcrResult[]>([]);
  const [ocrProgress, setOcrProgress] = useState(0);
  const [exportError, setExportError] = useState<string | null>(null);
  const ocrRequestId = useRef(0);
  const runBatchTranslation = useDubDubStore((s) => s.runBatchTranslation);
  const translateSingleSegment = useDubDubStore((s) => s.translateSingleSegment);
  const translationModal = useDubDubStore((s) => s.translationModal);
  const closeTranslationModal = useDubDubStore((s) => s.closeTranslationModal);
  const backend = useDubDubStore((s) => s.backend);
  const languages = useDubDubStore((s) => s.languages);
  const updateSourceLanguage = useDubDubStore((s) => s.updateSourceLanguage);
  const updateTargetLanguage = useDubDubStore((s) => s.updateTargetLanguage);
  const updateTranslationProvider = useDubDubStore((s) => s.updateTranslationProvider);
  const updateTranslationMode = useDubDubStore((s) => s.updateTranslationMode);
  const openSettings = useDubDubStore((s) => s.openSettings);
  const transcriptOptions = useDubDubStore((s) => s.transcriptOptions);
  const selectedSegmentOption = useDubDubStore((s) => s.selectedSegmentOption);
  const showSegmentationModal = useDubDubStore((s) => s.showSegmentationModal);
  const selectSegmentationOption = useDubDubStore((s) => s.selectSegmentationOption);
  const setShowSegmentationModal = useDubDubStore((s) => s.setShowSegmentationModal);

  const translationProviders =
    backend.options?.translationProviders?.length
      ? backend.options.translationProviders
      : FALLBACK_TRANSLATION_PROVIDERS;

  const translationModes =
    backend.options?.translationModes?.length
      ? backend.options.translationModes
      : FALLBACK_TRANSLATION_MODES;

  const selectedProvider =
    translationProviders.find(
      (p: any) => p.translateType === Number(backend?.config?.translateType)
    ) || translationProviders[0];

  const selectedMode = backend?.config?.translationMode || 'srt';

  const availableLanguages =
    backend?.options?.languages?.length
      ? backend.options.languages
      : FALLBACK_LANGUAGES;

  const filteredSegments = (segments || []).filter((seg) => {
    if (!seg) return false;
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      (seg.sourceText || '').toLowerCase().includes(q) ||
      (seg.targetText || '').toLowerCase().includes(q) ||
      (seg.speakerName && String(seg.speakerName).toLowerCase().includes(q))
    );
  });
  const segmentIds = segments.map((segment) => segment.id).join(',');
  useEffect(() => {
    setSelectedOcrIds([]);
    setOcrSelectionHint(false);
    ocrRequestId.current += 1;
    setOcrDialogSegments(null);
  }, [activeProjectId, segmentIds]);
  const untranslated = untranslatedSegmentNumbers(segments || []);
  const canExportSrt = segments.length > 0 && untranslated.length === 0;
  const ocrSegment = ocrDialogSegments?.[0];

  const openOcr = (chosen: Segment[]) => {
    if (!chosen.length) return;
    ocrRequestId.current += 1;
    setOcrDialogSegments(chosen);
    setOcrResult(null);
    setOcrDetectedEntries(null);
    setOcrBatchResults([]);
    setOcrProgress(0);
    setOcrError(null);
  };

  const closeOcr = () => {
    ocrRequestId.current += 1;
    setOcrDialogSegments(null);
    setOcrLoading(false);
    setOcrError(null);
    setOcrResult(null);
    setOcrDetectedEntries(null);
    setOcrBatchResults([]);
    setOcrProgress(0);
  };

  const handleConfirmReplace = async (entries: OcrSubtitleEntry[]) => {
    if (!ocrSegment || !entries || entries.length === 0) return;
    try {
      await replaceSegmentWithOcrEntries(ocrSegment.id, entries);
      closeOcr();
    } catch (err: any) {
      setOcrError(err?.message || 'Failed to replace segment with OCR');
    }
  };

  const runOcr = async () => {
    if (!ocrDialogSegments?.length) return;
    const chosenIds = ocrDialogSegments.map((segment) => segment.id);
    const snapshot = selectedOcrSegments(useDubDubStore.getState().segments, chosenIds);
    if (snapshot.length !== chosenIds.length) {
      setOcrError('A selected segment is no longer available. Close this dialog and select the segments again.');
      return;
    }
    const requestId = ++ocrRequestId.current;
    setOcrLoading(true);
    setOcrError(null);
    setOcrResult(null);
    setOcrDetectedEntries(null);
    setOcrBatchResults([]);
    setOcrProgress(0);
    setOcrDialogSegments(snapshot);

    if (snapshot.length === 1) {
      const seg = snapshot[0];
      try {
        const res = await extractOcrForSegment(seg.id, ocrRoi, seg);
        if (requestId !== ocrRequestId.current) return;
        const entries = res.entries || [];
        setOcrDetectedEntries(entries);
        setOcrResult({ text: res.text || '', confidence: res.confidence, applied: false });
      } catch (err: any) {
        if (requestId !== ocrRequestId.current) return;
        setOcrError(err instanceof Error ? err.message : 'OCR extraction failed');
      } finally {
        if (requestId === ocrRequestId.current) setOcrLoading(false);
      }
      return;
    }

    await extractSelectedOcrSegments(snapshot, ocrRoi, extractOcrForSegment, async (entry) => {
      if (requestId !== ocrRequestId.current) return;
      const previewEntry: BatchOcrResult = { ...entry, applied: entry.error ? undefined : false };
      setOcrBatchResults((results) => [...results, previewEntry]);
      setOcrProgress((progress) => progress + 1);
    }, () => requestId === ocrRequestId.current);
    if (requestId === ocrRequestId.current) setOcrLoading(false);
  };

  const handleConfirmBatchApply = async () => {
    if (!ocrBatchResults.length) return;
    setOcrLoading(true);
    try {
      const updatedResults: BatchOcrResult[] = [];
      for (const entry of ocrBatchResults) {
        if (entry.applied || entry.error || !entry.text?.trim()) {
          updatedResults.push(entry);
          continue;
        }
        const applied = await applyExtractedOcrResult(
          entry,
          () => useDubDubStore.getState().segments,
          applyOcrText,
        );
        updatedResults.push(applied);
      }
      setOcrBatchResults(updatedResults);
    } finally {
      setOcrLoading(false);
    }
  };

  return (
    <div className="flex-1 min-h-0 w-full p-2.5 grid grid-cols-12 gap-2.5 overflow-hidden">
      {/* LEFT: Synchronized Video Player (6-7 cols) */}
      <div className="col-span-12 lg:col-span-6 xl:col-span-7 h-full min-h-0 overflow-hidden">
        <VideoPlayer title="Transcript Teleprompter Deck" subtitleVariant="dual" showAudioSwitcher={true} />
      </div>

      {/* RIGHT: Segment Cue Cards (5-6 cols) */}
      <div className="col-span-12 lg:col-span-6 xl:col-span-5 h-full bg-white rounded-xl border border-[#E7E4DC] shadow-xs flex flex-col min-h-0 overflow-hidden">
        {/* Tier 1: Translation Toolbar & Language Routing */}
        <div className="px-3 py-2 border-b border-[#E7E4DC] flex flex-col gap-2 bg-[#FAF9F6] shrink-0">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-1 font-bold text-xs text-stone-900">
                <Languages className="w-3.5 h-3.5 text-[#8D4B00]" />
                <span>LLM Translation:</span>
              </div>

              <select
                data-testid="translation-provider-select"
                value={backend.config.translateType}
                onChange={(e) => updateTranslationProvider(Number(e.target.value))}
                className="text-xs font-semibold py-1 px-2 bg-white rounded-md border border-stone-200 text-stone-800 focus:outline-none focus:border-amber-400"
              >
                {translationProviders.map((p: any) => (
                  <option key={p.translateType ?? p.id} value={p.translateType}>
                    {p.label}
                  </option>
                ))}
              </select>

              {selectedProvider.requiresSettings && (
                <button
                  type="button"
                  onClick={() => openSettings('providers', String(selectedProvider.id || ''))}
                  className={`px-2 py-0.5 rounded text-[9px] font-bold flex items-center gap-1 cursor-pointer transition-colors ${
                    selectedProvider.configured
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'bg-amber-100 text-[#8D4B00] border border-amber-300 animate-pulse'
                  }`}
                  title={selectedProvider.configured ? 'API Key configured' : 'API Key required'}
                >
                  <Key className="w-2.5 h-2.5" />
                  <span>{selectedProvider.configured ? 'Key Set' : 'Key Needed'}</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => openSettings('providers', String(selectedProvider.id || ''))}
                className="p-1 rounded text-stone-400 hover:text-stone-700 hover:bg-stone-100 transition-colors cursor-pointer"
                title="Configure Translation Provider Settings"
              >
                <Settings2 className="w-3.5 h-3.5 text-[#8D4B00]" />
              </button>

              <ProviderDocLink providerId={selectedProvider.id} category="llm" />

              <div className="flex items-center gap-1 ml-1">
                <span className="text-[10px] text-stone-500 font-semibold">Method:</span>
                <select
                  data-testid="translation-method-select"
                  value={selectedMode}
                  onChange={(e) => updateTranslationMode(e.target.value)}
                  className="text-xs font-semibold py-1 px-2 bg-white rounded-md border border-stone-200 text-stone-800 focus:outline-none focus:border-amber-400"
                >
                  {translationModes.map((m: any) => (
                    <option key={m.id} value={m.id} title={m.description}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              data-testid="batch-translate-btn"
              onClick={() => runBatchTranslation()}
              className="px-3 py-1.5 rounded-md bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center gap-1.5 shadow-2xs cursor-pointer transition-colors shrink-0"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Batch Translate</span>
            </button>
            <button
              type="button"
              data-testid="export-translated-srt-btn"
              disabled={!canExportSrt}
              onClick={() => {
                setExportError(null);
                try {
                  exportTranslatedSrt(segments);
                } catch (err: any) {
                  setExportError(err instanceof Error ? err.message : 'Failed to export translated SRT');
                }
              }}
              title={canExportSrt ? 'Download the current translated subtitles as an SRT file' : 'Translate every segment before exporting'}
              className="px-3 py-1.5 rounded-md border border-[#8D4B00] text-[#8D4B00] bg-white text-xs font-bold flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="w-3.5 h-3.5" />
              Export Translated SRT
            </button>
          </div>
          {exportError && (
            <p role="alert" data-testid="export-srt-error" className="text-[11px] font-semibold text-rose-700">
              {exportError}
            </p>
          )}
          {untranslated.length > 0 && (
            <p data-testid="untranslated-segments" className="text-[11px] text-amber-800">
              Translate segment{untranslated.length === 1 ? '' : 's'} {untranslated.join(', ')} before exporting the SRT.
            </p>
          )}

          {/* Language Pair Routing Strip */}
          <div className="flex items-center gap-2 pt-1 border-t border-stone-200/70 flex-wrap text-xs">
            <div className="flex items-center gap-1 font-semibold text-stone-600">
              <Globe className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span className="text-[11px] font-bold text-stone-700">Language Route:</span>
            </div>

            <div className="flex items-center gap-1.5 bg-white px-2 py-0.5 rounded-md border border-stone-200 shadow-2xs">
              <span className="text-[10px] text-stone-400 font-bold uppercase">From</span>
              <select
                data-testid="translation-source-lang-select"
                value={languages?.source?.code || 'zh-cn'}
                onChange={(e) => {
                  const opt = availableLanguages.find((l: any) => l.code === e.target.value);
                  updateSourceLanguage(e.target.value, opt?.name || e.target.value);
                }}
                className="text-xs font-semibold py-0.5 bg-transparent border-0 text-stone-800 focus:outline-none cursor-pointer"
              >
                {availableLanguages.map((l: any) => (
                  <option key={l.code} value={l.code}>
                    {l.name} ({l.code})
                  </option>
                ))}
              </select>
            </div>

            <ArrowRight className="w-3.5 h-3.5 text-stone-400 shrink-0" />

            <div className="flex items-center gap-1.5 bg-white px-2 py-0.5 rounded-md border border-stone-200 shadow-2xs">
              <span className="text-[10px] text-stone-400 font-bold uppercase">To</span>
              <select
                data-testid="translation-target-lang-select"
                value={languages?.target?.code || 'vi'}
                onChange={(e) => {
                  const opt = availableLanguages.find((l: any) => l.code === e.target.value);
                  updateTargetLanguage(e.target.value, opt?.name || e.target.value);
                }}
                className="text-xs font-semibold py-0.5 bg-transparent border-0 text-stone-800 focus:outline-none cursor-pointer"
              >
                {availableLanguages.map((l: any) => (
                  <option key={l.code} value={l.code}>
                    {l.name} ({l.code})
                  </option>
                ))}
              </select>
            </div>

            <span className="text-[10px] text-stone-400 font-medium ml-auto hidden sm:inline">
              Source &amp; target languages applied on translate
            </span>
          </div>
        </div>

        {/* Tier 2: Search, Segmentation Switcher, & Filter Strip */}
        <div className="min-h-8.5 px-3 py-1 border-b border-[#E7E4DC] flex flex-wrap items-center justify-between bg-white shrink-0 gap-2">
          <div className="relative flex-1 max-w-sm min-w-[140px]">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-stone-400" />
            <input
              type="text"
              placeholder="Search transcript..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-2.5 py-1 text-xs bg-stone-50 rounded-md border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
            />
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {transcriptOptions?.utterances?.length && transcriptOptions?.paragraphs?.length ? (
              <div className="flex items-center gap-1 bg-stone-100 p-0.5 rounded-lg border border-stone-200">
                <span className="text-[10px] text-stone-500 font-bold px-1 hidden sm:inline">Segmentation:</span>
                <button
                  data-testid="select-utterances-btn"
                  onClick={() => selectSegmentationOption('utterances')}
                  className={`px-2 py-0.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                    selectedSegmentOption === 'utterances'
                      ? 'bg-[#8D4B00] text-white shadow-2xs font-bold'
                      : 'text-stone-600 hover:text-stone-900 hover:bg-stone-200/50'
                  }`}
                >
                  Utterances ({transcriptOptions.utterances.length})
                </button>
                <button
                  data-testid="select-paragraphs-btn"
                  onClick={() => selectSegmentationOption('paragraphs')}
                  className={`px-2 py-0.5 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                    selectedSegmentOption === 'paragraphs'
                      ? 'bg-[#8D4B00] text-white shadow-2xs font-bold'
                      : 'text-stone-600 hover:text-stone-900 hover:bg-stone-200/50'
                  }`}
                >
                  Paragraphs ({transcriptOptions.paragraphs.length})
                </button>
                <button
                  data-testid="open-segmentation-modal-btn"
                  onClick={() => setShowSegmentationModal(true)}
                  title="Compare Utterances vs Paragraphs"
                  className="p-1 text-stone-400 hover:text-stone-700 hover:bg-stone-200 rounded cursor-pointer"
                >
                  <Layers className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : null}

            <span className="text-[10px] text-stone-400 font-mono font-medium whitespace-nowrap">
              {filteredSegments.length} {filteredSegments.length === 1 ? 'cue' : 'cues'}
            </span>
          </div>
        </div>

        <div className="px-3 py-1.5 border-b border-stone-200 bg-stone-50/90 shrink-0 flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-stone-800 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#8D4B00]" />
              <span>Video text OCR</span>
            </span>
            <span className="text-[11px] text-stone-500 hidden sm:inline">
              Select segment checkboxes below to read video text only within their time ranges.
            </span>
          </div>

          <div className="flex items-center gap-2">
            {selectedOcrIds.length > 0 && (
              <span className="text-[11px] font-semibold text-[#8D4B00]">
                {selectedOcrIds.length} selected
              </span>
            )}
            <button
              type="button"
              data-testid="ocr-selected-btn"
              onClick={() =>
                selectedOcrIds.length
                  ? openOcr(selectedOcrSegments(segments, selectedOcrIds))
                  : setOcrSelectionHint(true)
              }
              className="px-2.5 py-1 rounded bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center gap-1.5 shadow-2xs cursor-pointer transition-colors"
            >
              <Sparkles className="w-3 h-3" />
              <span>Run OCR on Selected Segments</span>
            </button>
          </div>
          {ocrSelectionHint && (
            <p role="alert" className="w-full text-[11px] font-semibold text-amber-900 mt-0.5">
              Check one or more segments below, then run OCR.
            </p>
          )}
        </div>

        {/* Segment Cards List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {filteredSegments.map((seg) => {
            const isActive = seg.id === activeSegmentId;
            return (
              <div
                key={seg.id}
                data-segment-card={seg.id}
                onClick={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.closest('button, select, textarea, input, label, a')) return;
                  seekAndPlay(seg.startSec, seg.endSec, seg.id);
                }}
                className={`p-3 rounded-xl border transition-all cursor-pointer ${
                  isActive
                    ? 'border-[#8D4B00] bg-amber-50/25 ring-1 ring-[#8D4B00]/20 shadow-xs'
                    : 'border-stone-200 hover:border-stone-300 bg-white'
                }`}
              >
                {/* Card Header: Checkbox, Source Badge, Speaker, Timecode, CPS, and Actions */}
                <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
                  <div className="flex items-center gap-2 flex-wrap">
                    <label
                      className="flex items-center gap-1 text-[11px] font-semibold text-[#8D4B00] cursor-pointer shrink-0"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <input
                        type="checkbox"
                        aria-label={`Select segment ${seg.id} for OCR`}
                        data-testid={`ocr-select-${seg.id}`}
                        checked={selectedOcrIds.includes(seg.id)}
                        onChange={(event) => {
                          setOcrSelectionHint(false);
                          setSelectedOcrIds((ids) =>
                            event.target.checked
                              ? [...ids, seg.id]
                              : ids.filter((id) => id !== seg.id)
                          );
                        }}
                        className="h-3.5 w-3.5 accent-[#8D4B00]"
                      />
                      <span className="text-[10px] text-stone-500 font-mono">#{seg.id}</span>
                    </label>

                    {seg.sourceType === 'ocr' ? (
                      <span
                        data-testid={`source-badge-${seg.id}`}
                        className="px-1.5 py-0.2 rounded text-[9px] font-bold uppercase tracking-wider bg-amber-100 text-amber-900 border border-amber-300 flex items-center gap-0.5"
                        title="Transcription Source: Hard-Subtitle Video OCR"
                      >
                        <Sparkles className="w-2.5 h-2.5 text-[#8D4B00]" />
                        <span>OCR</span>
                      </span>
                    ) : (
                      <span
                        data-testid={`source-badge-${seg.id}`}
                        className="px-1.5 py-0.2 rounded text-[9px] font-bold uppercase tracking-wider bg-stone-100 text-stone-700 border border-stone-200"
                        title="Transcription Source: Audio Speech Recognition (ASR)"
                      >
                        ASR
                      </span>
                    )}

                    <span className="px-2 py-0.5 rounded-full bg-[#8D4B00] text-white text-[9px] font-bold uppercase tracking-wide">
                      {seg.speakerName || seg.speakerId || 'Speaker 1'}
                    </span>
                    <span className="font-mono text-[10px] text-stone-500 font-semibold">
                      {seg.startTime} ➔ {seg.endTime}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      data-testid={`replace-ocr-btn-${seg.id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        openOcr([seg]);
                      }}
                      className="text-[10px] font-bold text-[#8D4B00] hover:text-white hover:bg-[#8D4B00] px-2 py-0.5 rounded border border-[#8D4B00]/40 bg-amber-50/60 flex items-center gap-1 cursor-pointer transition-colors shadow-2xs mr-1"
                      title="Replace this ASR segment with video OCR subtitles"
                    >
                      <Sparkles className="w-2.5 h-2.5" />
                      <span>Replace with OCR</span>
                    </button>

                    <span
                      data-cps-badge="true"
                      className={`px-1.5 py-0.2 rounded text-[9px] font-bold font-mono border ${
                        seg.cps && seg.cps <= 14.5
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : 'bg-amber-50 text-[#8D4B00] border-amber-200'
                      }`}
                    >
                      {seg.cps || 14} CPS • {seg.cpsStatus || 'Optimal'}
                    </span>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        splitSegment(seg.id);
                      }}
                      className="p-1 rounded hover:bg-stone-100 text-stone-500 hover:text-stone-800"
                      title="Split segment"
                    >
                      <Split className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        mergeWithNextSegment(seg.id);
                      }}
                      className="p-1 rounded hover:bg-stone-100 text-stone-500 hover:text-stone-800"
                      title="Merge with next"
                    >
                      <Merge className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSegment(seg.id);
                      }}
                      className="p-1 rounded hover:bg-rose-50 text-stone-400 hover:text-rose-600"
                      title="Delete segment"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>

                {/* Source Textarea */}
                <div className="mb-2">
                  <label className="text-[10px] text-stone-400 font-medium block mb-0.5">Source Text</label>
                  <textarea
                    data-segment-input={seg.id}
                    value={seg.sourceText ?? seg.text ?? ''}
                    onChange={(e) => updateSegmentText(seg.id, e.target.value, false)}
                    rows={2}
                    className="w-full text-xs p-2 rounded-lg bg-stone-50 border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
                  />
                </div>

                {/* Target Translated Text */}
                <div>
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="text-[10px] text-stone-400 font-medium">Translated Text</label>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          translateSingleSegment(seg.id);
                        }}
                        className="text-[9px] text-[#8D4B00] font-semibold hover:underline flex items-center gap-0.5 cursor-pointer"
                        title="Translate only this segment"
                      >
                        <Languages className="w-2.5 h-2.5" />
                        <span>Translate</span>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          openOcr([seg]);
                        }}
                        className="text-[9px] text-stone-500 hover:text-stone-800 font-semibold hover:underline flex items-center gap-0.5 cursor-pointer"
                      >
                        <Sparkles className="w-2.5 h-2.5" />
                        <span>OCR Extract</span>
                      </button>
                    </div>
                  </div>
                  <textarea
                    data-segment-input={`stage2-target-${seg.id}`}
                    value={seg.targetText ?? ''}
                    onChange={(e) => updateSegmentText(seg.id, e.target.value, true)}
                    rows={2}
                    className="w-full text-xs p-2 rounded-lg bg-amber-50/40 border border-amber-200/60 focus:bg-white focus:outline-none focus:border-amber-400 font-medium text-stone-900"
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {ocrSegment && (
        <Stage2OcrDialog
          key={ocrSegment.id}
          segment={ocrSegment}
          videoUrl={project.previewUrl || ''}
          batchSegments={ocrDialogSegments || undefined}
          batchResults={ocrBatchResults}
          currentSegments={segments}
          progress={ocrProgress}
          roi={ocrRoi}
          entries={ocrDetectedEntries}
          onRoiChange={(roi) => {
            ocrRequestId.current += 1;
            setOcrCropRoi(roi);
            setOcrLoading(false);
            setOcrError(null);
            setOcrResult(null);
            setOcrDetectedEntries(null);
            setOcrBatchResults([]);
            setOcrProgress(0);
          }}
          onExtract={() => void runOcr()}
          onConfirmReplace={handleConfirmReplace}
          onConfirmBatchApply={() => void handleConfirmBatchApply()}
          onClose={closeOcr}
          loading={ocrLoading}
          result={ocrResult}
          error={ocrError}
        />
      )}

      {/* Translation Progress Modal */}
      {translationModal.active && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs">
          <div className="w-full max-w-sm bg-white rounded-2xl p-5 shadow-2xl border border-stone-200 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-bold text-sm text-stone-900">
                <Languages className="w-4 h-4 text-[#8D4B00]" />
                <span>Batch Translation</span>
              </div>
              <button
                onClick={() => closeTranslationModal()}
                className="w-6 h-6 rounded-full hover:bg-stone-100 flex items-center justify-center text-stone-400 hover:text-stone-700"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center gap-1.5 text-xs text-stone-600 bg-stone-50 p-2 rounded-lg border border-stone-200">
              <span className="font-semibold text-stone-800">{languages.source.name || languages.source.code}</span>
              <ArrowRight className="w-3 h-3 text-stone-400 shrink-0" />
              <span className="font-semibold text-[#8D4B00]">{languages.target.name || languages.target.code}</span>
              <span className="text-[10px] text-stone-400 ml-auto font-mono">({selectedProvider.label})</span>
            </div>

            <p className="text-xs text-stone-600">{translationModal.message}</p>

            {translationModal.error && (
              <div
                role="alert"
                data-testid="translation-modal-error"
                className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 whitespace-pre-wrap break-words"
              >
                {translationModal.error}
              </div>
            )}

            {translationModal.status === 'translating' && (
              <div className="w-full h-1.5 bg-stone-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#8D4B00] transition-all duration-300 rounded-full"
                  style={{ width: `${translationModal.progress}%` }}
                />
              </div>
            )}

            <div className="flex justify-end gap-2 mt-2">
              <button
                onClick={() => closeTranslationModal()}
                className="px-3 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-xs font-semibold text-stone-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Segmentation Choice Modal */}
      {showSegmentationModal && transcriptOptions && (
        <div data-testid="segmentation-modal" className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-xs p-4">
          <div className="w-full max-w-2xl bg-white rounded-2xl p-5 shadow-2xl border border-stone-200 flex flex-col gap-4 max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between pb-3 border-b border-stone-100">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-amber-50 text-[#8D4B00]">
                  <Layers className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-sm text-stone-900">Choose Transcript Segmentation</h3>
                  <p className="text-xs text-stone-500">
                    Deepgram returned two subtitle segmentation options with start and end times.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowSegmentationModal(false)}
                className="w-7 h-7 rounded-full hover:bg-stone-100 flex items-center justify-center text-stone-400 hover:text-stone-700 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 overflow-y-auto pr-1">
              {/* Option 1: Utterances */}
              {transcriptOptions.utterances && (
                <div
                  className={`flex flex-col justify-between p-4 rounded-xl border transition-all ${
                    selectedSegmentOption === 'utterances'
                      ? 'border-[#8D4B00] bg-amber-50/20 ring-2 ring-[#8D4B00]/20'
                      : 'border-stone-200 hover:border-stone-300 bg-white'
                  }`}
                >
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-stone-900 flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-[#8D4B00]" />
                        <span>Utterances</span>
                      </span>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-900">
                        {transcriptOptions.utterances.length} cues
                      </span>
                    </div>

                    <p className="text-xs text-stone-500">
                      Fine-grained speech pauses and conversational turns. Best for rhythmic video subtitles.
                    </p>

                    <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-100 space-y-2 text-[11px] font-mono">
                      <div className="text-[10px] font-sans font-semibold text-stone-400 uppercase tracking-wider">
                        Sample Segments
                      </div>
                      {transcriptOptions.utterances.slice(0, 2).map((u, i) => (
                        <div key={i} className="border-b border-stone-200/60 pb-1.5 last:border-0 last:pb-0">
                          <span className="text-[#8D4B00] font-semibold">{u.startTime} ➔ {u.endTime}</span>
                          <p className="text-stone-700 font-sans text-xs mt-0.5 line-clamp-2">
                            {u.sourceText || u.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <button
                    data-testid="choose-utterances-btn"
                    onClick={() => selectSegmentationOption('utterances')}
                    className={`mt-4 w-full py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                      selectedSegmentOption === 'utterances'
                        ? 'bg-[#8D4B00] text-white shadow-xs'
                        : 'bg-stone-100 hover:bg-stone-200 text-stone-800'
                    }`}
                  >
                    {selectedSegmentOption === 'utterances' ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span>Active Option</span>
                      </>
                    ) : (
                      <span>Use Utterances</span>
                    )}
                  </button>
                </div>
              )}

              {/* Option 2: Paragraphs */}
              {transcriptOptions.paragraphs && (
                <div
                  className={`flex flex-col justify-between p-4 rounded-xl border transition-all ${
                    selectedSegmentOption === 'paragraphs'
                      ? 'border-[#8D4B00] bg-amber-50/20 ring-2 ring-[#8D4B00]/20'
                      : 'border-stone-200 hover:border-stone-300 bg-white'
                  }`}
                >
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-stone-900 flex items-center gap-1.5">
                        <AlignLeft className="w-3.5 h-3.5 text-[#8D4B00]" />
                        <span>Paragraphs (paragraph.transcript)</span>
                      </span>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-900">
                        {transcriptOptions.paragraphs.length} cues
                      </span>
                    </div>

                    <p className="text-xs text-stone-500">
                      Cohesive paragraph blocks. Best for reading, teleprompter scripts, and document flow.
                    </p>

                    <div className="p-2.5 rounded-lg bg-stone-50 border border-stone-100 space-y-2 text-[11px] font-mono">
                      <div className="text-[10px] font-sans font-semibold text-stone-400 uppercase tracking-wider">
                        Sample Segments
                      </div>
                      {transcriptOptions.paragraphs.slice(0, 2).map((p, i) => (
                        <div key={i} className="border-b border-stone-200/60 pb-1.5 last:border-0 last:pb-0">
                          <span className="text-[#8D4B00] font-semibold">{p.startTime} ➔ {p.endTime}</span>
                          <p className="text-stone-700 font-sans text-xs mt-0.5 line-clamp-2">
                            {p.sourceText || p.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <button
                    data-testid="choose-paragraphs-btn"
                    onClick={() => selectSegmentationOption('paragraphs')}
                    className={`mt-4 w-full py-2 px-3 rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                      selectedSegmentOption === 'paragraphs'
                        ? 'bg-[#8D4B00] text-white shadow-xs'
                        : 'bg-stone-100 hover:bg-stone-200 text-stone-800'
                    }`}
                  >
                    {selectedSegmentOption === 'paragraphs' ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span>Active Option</span>
                      </>
                    ) : (
                      <span>Use Paragraphs</span>
                    )}
                  </button>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-stone-100 text-xs">
              <span className="text-stone-500 text-[11px]">
                ✓ Progress is automatically saved to your project. You can switch options anytime from the toolbar.
              </span>
              <button
                onClick={() => setShowSegmentationModal(false)}
                className="px-3.5 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-xs font-semibold text-stone-700 cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
