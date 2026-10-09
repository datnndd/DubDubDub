import { apiRequest } from './client';

export async function uploadMedia(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  return apiRequest('/api/media', {
    method: 'POST',
    body: formData,
  });
}

export async function scanFolders(paths: string[], recursive: boolean = true): Promise<{ ok: boolean; videos: Array<{ path: string; filename: string; sizeBytes: number; parentDir: string }>; count: number }> {
  return apiRequest('/api/media/scan-folders', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths, recursive }),
  });
}

export async function registerLocalMedia(paths: string[]): Promise<{ ok: boolean; media: any[]; count: number }> {
  return apiRequest('/api/media/register-local', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  });
}

export async function uploadBatchMedia(files: File[]): Promise<{ ok: boolean; media: any[]; count: number }> {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  return apiRequest('/api/media/batch-upload', {
    method: 'POST',
    body: formData,
  });
}

export async function uploadEditAsset(kind: 'background-audio' | 'thumbnail', file: File): Promise<{ id: string; name: string }> {
  const formData = new FormData();
  formData.append('file', file);
  return apiRequest(`/api/assets/${kind}`, {
    method: 'POST',
    body: formData,
  });
}

