import type { ProjectRecord } from '../types/project';

/**
 * Returns true if a project record is a date-parent container rather than a video project.
 */
export function isParentProject(p?: Partial<ProjectRecord> | Record<string, any> | null): boolean {
  if (!p) return false;
  if (p.state?.is_parent || (p as any).is_parent) return true;
  if (typeof p.state_json === 'string' && p.state_json.trim()) {
    try {
      const parsed = JSON.parse(p.state_json);
      if (parsed?.is_parent) return true;
    } catch {
      // ignore malformed json
    }
  } else if (p.state_json && typeof p.state_json === 'object' && (p.state_json as any).is_parent) {
    return true;
  }
  return typeof p.id === 'string' && p.id.startsWith('parent_');
}
