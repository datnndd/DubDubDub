/**
 * Shared local-calendar date helpers so frontend date-parent grouping
 * matches backend `datetime.date.today().isoformat()` (local OS timezone).
 */

export function formatLocalDateYmd(date: Date = new Date()): string {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
    return '';
  }
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function getYesterdayLocalDateYmd(now: Date = new Date()): string {
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
  return formatLocalDateYmd(yesterday);
}

/**
 * Parses ISO date strings as well as SQLite Unix epoch seconds (`time.time()`).
 * Numeric values with magnitude < 1e12 are treated as seconds and multiplied by 1000.
 */
export function parseProjectDate(raw?: string | number | null): Date | null {
  if (raw === undefined || raw === null || raw === '') return null;
  if (typeof raw === 'number') {
    if (!Number.isFinite(raw)) return null;
    const ms = Math.abs(raw) < 1e12 ? raw * 1000 : raw;
    const d = new Date(ms);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  const str = String(raw).trim();
  if (!str) return null;
  if (/^-?\d+(\.\d+)?$/.test(str)) {
    const num = Number(str);
    if (!Number.isFinite(num)) return null;
    const ms = Math.abs(num) < 1e12 ? num * 1000 : num;
    const d = new Date(ms);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  const d = new Date(str);
  return Number.isNaN(d.getTime()) ? null : d;
}
