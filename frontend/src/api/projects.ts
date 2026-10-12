import { apiRequest } from './client';
import type { ProjectRecord } from '../types/project';
import type { Segment } from '../types/segment';

export async function fetchProjects(): Promise<ProjectRecord[]> {
  const data = await apiRequest<any>('/api/projects');
  return data?.projects || (Array.isArray(data) ? data : []);
}

export async function fetchProject(id: string): Promise<ProjectRecord> {
  const data = await apiRequest<any>(`/api/projects/${encodeURIComponent(id)}`);
  return data?.project || data;
}

export async function createProject(payload: {
  name?: string;
  media_path?: string;
  media_id?: string;
  mediaId?: string;
  duration?: number;
  stage?: number;
  state?: any;
  parentId?: string;
  parent_id?: string;
}): Promise<ProjectRecord> {
  const data = await apiRequest<any>('/api/projects', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  return data?.project || data;
}

export async function fetchTodayParentProject(): Promise<{ parent: ProjectRecord; subprojects: ProjectRecord[] }> {
  return apiRequest('/api/projects/today-parent');
}

export async function updateProjectState(id: string, state: any, stage?: number, mediaId?: string): Promise<void> {
  await apiRequest(`/api/projects/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify({ state, stage, media_id: mediaId }),
  });
}

export async function resetProjectStage(id: string, stage: number): Promise<ProjectRecord> {
  return apiRequest(`/api/projects/${encodeURIComponent(id)}/stages/${stage}/reset`, { method: 'POST' });
}

export async function assembleProjectDubbing(id: string, segments: Segment[]): Promise<{ ok: boolean; audio_url: string }> {
  return apiRequest(`/api/projects/${encodeURIComponent(id)}/dubbing/assemble`, {
    method: 'POST',
    body: JSON.stringify({ segments }),
  });
}

export async function deleteProject(id: string): Promise<void> {
  await apiRequest(`/api/projects/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
}

export async function bulkDeleteProjects(ids: string[]): Promise<{ ok: boolean; deleted: string[]; count: number }> {
  return apiRequest('/api/projects/bulk-delete', {
    method: 'POST',
    body: JSON.stringify({ ids }),
  });
}

export async function resumeProject(id: string): Promise<{ job_id: string }> {
  return apiRequest(`/api/projects/${encodeURIComponent(id)}/resume`, {
    method: 'POST',
  });
}

export async function batchCreateProjects(payload: {
  items: Array<{ name: string; mediaId?: string; mediaPath?: string; duration?: number }>;
  date?: string;
  parentId?: string;
}): Promise<{ ok: boolean; parent: ProjectRecord; projects: ProjectRecord[] }> {
  return apiRequest('/api/projects/batch-create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function fetchSubprojects(parentId: string): Promise<ProjectRecord[]> {
  const data = await apiRequest<any>(`/api/projects/${encodeURIComponent(parentId)}/subprojects`);
  return data?.subprojects || [];
}

