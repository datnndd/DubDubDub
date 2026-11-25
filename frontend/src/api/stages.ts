import { apiRequest } from './client';
import type { OcrSubtitleEntry } from '../types/segment';
import type { AnnotationItem } from '../types/annotation';

export async function requestTranslate(payload: {
  segments: any[];
  sourceLanguage: string;
  targetLanguage: string;
  translateType: number;
  translationMode?: string;
  projectId?: string;
  mediaId?: string;
}): Promise<{ ok: boolean; segments: any[] }> {
  return apiRequest('/api/translate', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function requestOcrExtract(payload: {
  mediaId: string;
  roi: [number, number, number, number];
  startSec: number;
  endSec: number;
  segmentId?: number;
  language?: string;
}): Promise<{ ok: boolean; text: string; confidence?: number; boxNumber?: string; entries?: OcrSubtitleEntry[] }> {
  return apiRequest('/api/ocr/extract', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function requestSplitSegment(payload: {
  segment: any;
  splitSec: number;
}): Promise<{ ok: boolean; segments: any[] }> {
  return apiRequest('/api/segments/split', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function requestRender(payload: any): Promise<{ id?: string; jobId?: string }> {
  return apiRequest('/api/render', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function requestAnnotationScan(payload: {
  mediaId: string;
  language?: string;
  excludeRoi?: [number, number, number, number] | null;
  sampleIntervalMs?: number;
  filterStatic?: boolean;
}): Promise<{
  ok: boolean;
  annotations: AnnotationItem[];
  sampledFrames?: number;
  ocrCalls?: number;
  filteredWatermarks?: number;
  resolution?: { width: number; height: number };
}> {
  return apiRequest('/api/ocr/annotations/scan', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function requestAnnotationRegion(payload: {
  mediaId: string;
  anchorSec: number;
  roi: [number, number, number, number];
  language?: string;
  maxSpanSec?: number;
}): Promise<{ ok: boolean; annotation: AnnotationItem }> {
  return apiRequest('/api/ocr/annotations/region', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
