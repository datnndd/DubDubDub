import { useDubDubStore } from '../../src/store';

export interface FetchCallRecord {
  url: string;
  method: string;
  body: any;
  rawBody: string | undefined;
}

export type MockFetchHandler = (
  url: string,
  init: RequestInit | undefined,
  call: FetchCallRecord
) => Response | Promise<Response>;

export class MockEventSource {
  static instances: MockEventSource[] = [];

  readonly url: string;
  readyState: number = 1; // OPEN
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: any) => void) | null = null;
  private listeners: Map<string, Set<(ev: any) => void>> = new Map();

  constructor(url: string) {
    this.url = String(url);
    MockEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: (ev: any) => void): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener);
  }

  removeEventListener(type: string, listener: (ev: any) => void): void {
    this.listeners.get(type)?.delete(listener);
  }

  emitMessage(payload: any): void {
    if (this.readyState === 2) return;
    const ev = { data: JSON.stringify(payload) } as MessageEvent;
    this.onmessage?.(ev);
    const set = this.listeners.get('message');
    if (set) {
      for (const fn of set) fn(ev);
    }
  }

  emitDone(payload: any = { status: 'succeeded' }): void {
    if (this.readyState === 2) return;
    const ev = { data: JSON.stringify(payload) } as MessageEvent;
    const set = this.listeners.get('done');
    if (set) {
      for (const fn of Array.from(set)) fn(ev);
    }
  }

  emitError(err: any = new Error('SSE connection error')): void {
    this.onerror?.(err);
    const set = this.listeners.get('error');
    if (set) {
      for (const fn of set) fn(err);
    }
  }

  close(): void {
    this.readyState = 2; // CLOSED
  }

  static findByJobId(jobId: string): MockEventSource | undefined {
    return MockEventSource.instances.find((es) => es.url.includes(`/api/jobs/${encodeURIComponent(jobId)}`) || es.url.includes(jobId));
  }

  static reset(): void {
    MockEventSource.instances = [];
  }
}

export class MockAudio {
  static instances: MockAudio[] = [];

  src: string = '';
  volume: number = 1;
  muted: boolean = false;
  playbackRate: number = 1;
  currentTime: number = 0;
  duration: number = 5;
  paused: boolean = true;
  onloadedmetadata: (() => void) | null = null;
  onerror: ((err?: any) => void) | null = null;
  onended: (() => void) | null = null;
  onplay: (() => void) | null = null;
  onpause: (() => void) | null = null;

  constructor(src?: string) {
    if (src) this.src = src;
    MockAudio.instances.push(this);
  }

  load(): void {}

  async play(): Promise<void> {
    this.paused = false;
    this.onplay?.();
  }

  pause(): void {
    this.paused = true;
    this.onpause?.();
  }

  static reset(): void {
    MockAudio.instances = [];
  }
}

export class MockFileReader {
  result: string | ArrayBuffer | null = null;
  error: Error | null = null;
  onload: ((ev?: any) => void) | null = null;
  onerror: ((ev?: any) => void) | null = null;

  readAsDataURL(blob: Blob): void {
    Promise.resolve()
      .then(async () => {
        const buf = await blob.arrayBuffer();
        const b64 = Buffer.from(buf).toString('base64');
        const mime = blob.type || 'application/octet-stream';
        this.result = `data:${mime};base64,${b64}`;
        this.onload?.({ target: this });
      })
      .catch((err) => {
        this.error = err instanceof Error ? err : new Error(String(err));
        this.onerror?.({ target: this });
      });
  }
}

const pendingTestTimers = new Set<ReturnType<typeof setTimeout>>();

export function clearPendingTestTimers(): void {
  for (const id of Array.from(pendingTestTimers)) {
    clearTimeout(id);
    pendingTestTimers.delete(id);
  }
}

export class MemoryStorage implements Storage {
  private map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }

  getItem(key: string): string | null {
    return this.map.has(key) ? this.map.get(key)! : null;
  }

  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  setItem(key: string, value: string): void {
    this.map.set(key, String(value));
  }
}

export interface CapturedDownload {
  filename: string;
  href: string;
  blob?: Blob;
  readText: () => Promise<string>;
}

export interface E2EShimContext {
  fetchCalls: FetchCallRecord[];
  downloads: CapturedDownload[];
  revokedUrls: string[];
  storage: MemoryStorage;
  eventSources: typeof MockEventSource;
  audioInstances: MockAudio[];
}

export function formatLocalYmd(date: Date = new Date()): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

/**
 * Deterministic UTC+7 clock simulator:
 * When `nowLocal` is e.g. `2026-11-18T00:15:00` in UTC+7:
 * - Local calendar getters (`getFullYear`, `getMonth`, `getDate`, `getHours`, ...) return `2026-11-18 00:15:00`
 * - `toISOString()` returns `2026-11-17T17:15:00.000Z` (previous UTC calendar day!)
 */
export async function withFakeUtcPlus7Clock<T>(
  fn: () => Promise<T> | T,
  fixedUtcMs: number = Date.UTC(2026, 10, 17, 17, 15, 0) // 2026-11-18T00:15:00+07:00
): Promise<T> {
  const RealDate = globalThis.Date;
  const OFFSET_MS = 7 * 60 * 60 * 1000;

  class UtcPlus7Date extends RealDate {
    constructor(...args: any[]) {
      if (args.length === 0) {
        super(fixedUtcMs);
      } else if (args.length === 1) {
        super(args[0]);
      } else {
        super(args[0], args[1], args[2] ?? 1, args[3] ?? 0, args[4] ?? 0, args[5] ?? 0, args[6] ?? 0);
      }
    }
    static now(): number {
      return fixedUtcMs;
    }
    private shifted(): Date {
      return new RealDate(this.getTime() + OFFSET_MS);
    }
    override getFullYear(): number {
      return this.shifted().getUTCFullYear();
    }
    override getMonth(): number {
      return this.shifted().getUTCMonth();
    }
    override getDate(): number {
      return this.shifted().getUTCDate();
    }
    override getDay(): number {
      return this.shifted().getUTCDay();
    }
    override getHours(): number {
      return this.shifted().getUTCHours();
    }
    override getMinutes(): number {
      return this.shifted().getUTCMinutes();
    }
    override getSeconds(): number {
      return this.shifted().getUTCSeconds();
    }
    override getTimezoneOffset(): number {
      return -420;
    }
  }

  (globalThis as any).Date = UtcPlus7Date;
  try {
    return await fn();
  } finally {
    globalThis.Date = RealDate;
  }
}

export function resetStore(overrides: Record<string, any> = {}): void {
  clearPendingTestTimers();
  const initial = useDubDubStore.getInitialState();
  useDubDubStore.setState(initial, true);

  // Ensure clean nested defaults and clear any persisted localStorage id
  const cur = useDubDubStore.getState();
  useDubDubStore.setState({
    activeProjectId: null,
    activeView: 'projects',
    drawerOpen: false,
    projectsList: [],
    currentStep: 1,
    maxUnlockedStep: 4,
    forceAsr: false,
    stage3Baseline: null,
    resettingStage: false,
    activeParentProjectId: null,
    activeParentProjectName: null,
    sessionSubprojects: [],
    sessionDrawerOpen: false,
    scanModalOpen: false,
    batchModeActive: false,
    activeJob: null,
    jobProgress: null,
    jobStage: null,
    jobMessage: 'Ready to process',
    jobStatus: 'idle',
    projectJobStatuses: {},
    sseUnsubscribe: null,
    pollTimer: null,
    startTime: null,
    elapsedSeconds: 0,
    segments: [],
    activeSegmentId: 1,
    searchQuery: '',
    transcriptOptions: null,
    selectedSegmentOption: 'utterances',
    showSegmentationModal: false,
    translationModal: {
      active: false,
      status: 'idle',
      progress: 0,
      message: 'Translating transcript...',
      error: null,
    },
    speakerVoiceMap: {},
    segmentVoiceOverrides: {},
    dubbingStatus: 'idle',
    dubbingError: null,
    assemblingDubbing: false,
    assembledDubUrl: null,
    voices: [],
    customVoices: [],
    activePreviewVoiceId: null,
    isCreateVoiceModalOpen: false,
    isVoiceManagerDrawerOpen: false,
    isSettingsOpen: false,
    settingsActiveTab: 'providers',
    settingsTargetProvider: null,
    backend: {
      ...cur.backend,
      ready: false,
      error: null,
      mediaId: null,
      status: 'idle',
      message: 'Choose a video to start',
      options: {
        languages: [
          { code: 'zh-cn', name: 'Simplified Chinese' },
          { code: 'en', name: 'English' },
          { code: 'vi', name: 'Vietnamese' },
        ],
        asrProviders: [
          { id: 'qwen-asr', label: 'Qwen-ASR', recognType: 0, models: ['1.7B'], requiresSettings: false, configured: true },
          { id: 'deepgram', label: 'Deepgram', recognType: 1, models: ['nova-3', 'nova-2'], requiresSettings: true, configured: true },
        ],
        translationProviders: [
          { id: 'google', label: 'Google Translate', translateType: 0, requiresSettings: false, configured: true },
          { id: 'openai', label: 'OpenAI ChatGPT', translateType: 1, requiresSettings: true, configured: true },
        ],
        translationModes: [
          { id: 'srt', label: 'Send SRT', description: 'SRT blocks' },
          { id: 'text', label: 'Batch Text', description: 'Plain text' },
        ],
        voices: [],
      },
      config: {
        recognType: 1,
        translateType: 0,
        translationMode: 'srt',
        ttsType: 2,
        modelName: 'nova-3',
        voiceRole: '',
        useCuda: false,
        deepgramOptions: {
          utt_split: 0.8,
          diarize_model: 'latest',
          smart_format: true,
          punctuate: true,
          paragraphs: true,
          utterances: true,
          extra: '',
        },
      },
    },
    ocrCrop: {
      active: false,
      segmentId: null,
      roi: [0.05, 0.75, 0.9, 0.2],
      loading: false,
      error: null,
    },
    ...overrides,
  });
}

export async function flushMicrotasks(ticks: number = 8): Promise<void> {
  for (let i = 0; i < ticks; i++) {
    await Promise.resolve();
  }
}

export async function flushTasks(ms: number = 25): Promise<void> {
  await flushMicrotasks(6);
  await new Promise((resolve) => setTimeout(resolve, ms));
  await flushMicrotasks(6);
}

export async function withE2EShims<T>(
  fetchHandler: MockFetchHandler,
  fn: (ctx: E2EShimContext) => Promise<T>
): Promise<T> {
  const originalFetch = globalThis.fetch;
  const originalEventSource = (globalThis as any).EventSource;
  const originalAudio = (globalThis as any).Audio;
  const originalFileReader = (globalThis as any).FileReader;
  const originalLocalStorage = (globalThis as any).localStorage;
  const originalDocument = (globalThis as any).document;
  const originalSetTimeout = globalThis.setTimeout;
  const originalClearTimeout = globalThis.clearTimeout;
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;

  MockEventSource.reset();
  MockAudio.reset();

  const fetchCalls: FetchCallRecord[] = [];
  const downloads: CapturedDownload[] = [];
  const revokedUrls: string[] = [];
  const blobRegistry = new Map<string, Blob>();
  const storage = new MemoryStorage();
  const trackedFetches = new Set<Promise<any>>();
  const trackedTimeouts = new Set<ReturnType<typeof setTimeout>>();
  let blobSeq = 0;

  (globalThis as any).setTimeout = ((cb: (...args: any[]) => void, ms?: number, ...args: any[]) => {
    const id = originalSetTimeout((...cbArgs: any[]) => {
      trackedTimeouts.delete(id);
      pendingTestTimers.delete(id);
      cb(...cbArgs);
    }, ms, ...args);
    trackedTimeouts.add(id);
    pendingTestTimers.add(id);
    return id;
  }) as typeof setTimeout;

  (globalThis as any).clearTimeout = ((id: any) => {
    if (id !== undefined) {
      trackedTimeouts.delete(id);
      pendingTestTimers.delete(id);
    }
    return originalClearTimeout(id);
  }) as typeof clearTimeout;

  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const p = (async () => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const method = (init?.method || 'GET').toUpperCase();
      const rawBody = init?.body !== undefined ? String(init.body) : undefined;
      let parsedBody: any = undefined;
      if (rawBody && !(init?.body instanceof FormData)) {
        try {
          parsedBody = JSON.parse(rawBody);
        } catch {
          parsedBody = rawBody;
        }
      }
      const record: FetchCallRecord = { url, method, body: parsedBody, rawBody };
      fetchCalls.push(record);
      return await fetchHandler(url, init, record);
    })();
    const tracked = p.then(
      () => undefined,
      () => undefined
    ).finally(() => {
      trackedFetches.delete(tracked);
    });
    trackedFetches.add(tracked);
    return p;
  }) as typeof fetch;

  (globalThis as any).EventSource = MockEventSource;
  (globalThis as any).Audio = MockAudio;
  (globalThis as any).FileReader = MockFileReader;
  (globalThis as any).localStorage = storage;

  URL.createObjectURL = ((obj: Blob | MediaSource) => {
    const url = `blob:e2e-${++blobSeq}`;
    if (obj instanceof Blob) {
      blobRegistry.set(url, obj);
    }
    return url;
  }) as typeof URL.createObjectURL;

  URL.revokeObjectURL = ((url: string) => {
    revokedUrls.push(url);
    blobRegistry.delete(url);
  }) as typeof URL.revokeObjectURL;

  (globalThis as any).document = {
    createElement: (tag: string) => {
      if (tag.toLowerCase() === 'a') {
        const anchor: any = {
          href: '',
          download: '',
          click: () => {
            const blob = blobRegistry.get(anchor.href);
            downloads.push({
              filename: anchor.download,
              href: anchor.href,
              blob,
              readText: async () => (blob ? await blob.text() : ''),
            });
          },
        };
        return anchor;
      }
      return { style: {}, setAttribute: () => {}, appendChild: () => {} };
    },
    body: {
      appendChild: () => {},
      removeChild: () => {},
    },
  };

  try {
    return await fn({
      fetchCalls,
      downloads,
      revokedUrls,
      storage,
      eventSources: MockEventSource,
      audioInstances: MockAudio.instances,
    });
  } finally {
    // Ensure any active SSE/poll timers in store are cleaned up before restoring globals
    try {
      const s = useDubDubStore.getState();
      s.sseUnsubscribe?.();
      if (s.pollTimer) clearInterval(s.pollTimer);
    } catch (_) {}

    // Drain in-flight mocked fetch calls & pendingProjectSave microtask chains BEFORE restoring globalThis.fetch
    let idlePasses = 0;
    for (let pass = 0; pass < 20; pass++) {
      if (trackedFetches.size > 0) {
        idlePasses = 0;
        await Promise.allSettled(Array.from(trackedFetches));
      }
      await flushMicrotasks(16);
      if (trackedFetches.size === 0) {
        idlePasses++;
        if (idlePasses >= 2) break;
        await new Promise<void>((resolve) => originalSetTimeout(resolve, 0));
        await flushMicrotasks(12);
      }
    }

    // Clear any test-scoped timers (e.g. runBatchTranslation 1200ms auto-close) before restoring timers
    for (const id of Array.from(trackedTimeouts)) {
      originalClearTimeout(id);
      trackedTimeouts.delete(id);
      pendingTestTimers.delete(id);
    }

    globalThis.setTimeout = originalSetTimeout;
    globalThis.clearTimeout = originalClearTimeout;
    globalThis.fetch = originalFetch;
    (globalThis as any).EventSource = originalEventSource;
    (globalThis as any).Audio = originalAudio;
    (globalThis as any).FileReader = originalFileReader;
    (globalThis as any).localStorage = originalLocalStorage;
    (globalThis as any).document = originalDocument;
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
    MockEventSource.reset();
    MockAudio.reset();
  }
}

export function jsonResponse(data: any, status: number = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
