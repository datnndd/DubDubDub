import { StateCreator } from 'zustand';
import type { JobRecord } from '../types/job';
import { startJob, cancelJob, fetchJob, subscribeJobStream } from '../api/jobs';

const settledJobIds = new Set<string>();
const activeJobSubTokens = new Map<string, symbol>();
const activePollTokens = new Map<string, symbol>();

interface BatchControl {
  cancelled: boolean;
  unsubs: Map<string, () => void>;
  pollTimers: Map<string, any>;
  cancelHandlers: Map<string, () => void>;
}
let activeBatchControl: BatchControl | null = null;

export interface JobSlice {
  activeJob: JobRecord | null;
  jobProgress: number | null;
  jobStage: string | null;
  jobMessage: string;
  jobStatus: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  projectJobStatuses: Record<string, { jobId?: string; status: string; progress: number; message: string; stage?: string }>;
  sseUnsubscribe: (() => void) | null;
  pollTimer: any;
  startTime: number | null;
  elapsedSeconds: number;

  startPrepareJob: () => Promise<void>;
  startBatchPrepareJobs: () => Promise<void>;
  subscribeToJob: (jobId: string, targetProjectId?: string | null, options?: { jobType?: 'asr' | 'render' | string }) => void;
  pollJob: (jobId: string, targetProjectId?: string | null, options?: { jobType?: 'asr' | 'render' | string }) => Promise<void>;
  cancelActiveJob: () => Promise<void>;
  clearJob: () => void;
}

export function buildPrepareJobRequest(state: any) {
  const config = state.backend?.config || {};
  return {
    mediaId: state.backend?.mediaId,
    projectId: state.activeProjectId,
    jobType: 'asr',
    options: {
      ...config,
      recognType: Number(config.recognType ?? 1),
      modelName: String(config.modelName || 'nova-3'),
      projectId: state.activeProjectId,
      sourceLanguage: state.languages?.source?.code || 'zh-cn',
      targetLanguage: state.languages?.target?.code || 'vi',
      timingMode: state.languages?.timingMode || 'voice',
      forceAsr: Boolean(state.forceAsr),
      speakerDiarization: Boolean(state.engines?.speakerDiarization),
      speakerCount: Number(state.engines?.speakerCount || 0),
      subtitleSource: state.subtitleSource || 'audio_asr',
      ocrRoi: state.ocrCrop?.roi || [0.05, 0.75, 0.9, 0.2],
      ocrRoiConfirmed: Boolean(state.ocrRoiConfirmed),
      ocrOptions: state.ocrOptions || {
        coarseIntervalMs: 500,
        useAngleCls: false,
        paragraphMode: 'preserve_lines',
      },
    },
  };
}

export const createJobSlice: StateCreator<any, [], [], JobSlice> = (set, get) => ({
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

  startBatchPrepareJobs: async () => {
    const subprojects = get().sessionSubprojects || [];
    if (!subprojects || subprojects.length === 0) {
      return get().startPrepareJob();
    }

    const backend = get().backend;
    const config = backend?.config || {};
    const languages = get().languages;
    const engines = get().engines;
    const subtitleSource = get().subtitleSource || 'audio_asr';
    const isVideoOcr = subtitleSource === 'video_ocr';
    const ocrRoi = get().ocrCrop?.roi || [0.05, 0.75, 0.9, 0.2];
    const ocrRoiConfirmed = Boolean(get().ocrRoiConfirmed);
    const ocrOptions = get().ocrOptions || {
      coarseIntervalMs: 500,
      useAngleCls: false,
      paragraphMode: 'preserve_lines',
    };

    if (isVideoOcr && !ocrRoiConfirmed) {
      set({
        jobStatus: 'failed',
        jobMessage: 'Please test and confirm the subtitle ROI before starting Video OCR.',
      });
      return;
    }

    const currentStatuses = get().projectJobStatuses || {};
    const queue = subprojects.filter(
      (p: any) =>
        (p.stage ?? 1) <= 1 &&
        currentStatuses[p.id]?.status !== 'completed' &&
        p.status !== 'completed'
    );
    if (queue.length === 0) return;
    const maxConcurrency = 2;
    let runningCount = 0;

    if (activeBatchControl) {
      activeBatchControl.cancelled = true;
      activeBatchControl.unsubs.forEach((fn) => { try { fn(); } catch (_) {} });
      activeBatchControl.pollTimers.forEach((t) => { try { clearInterval(t); } catch (_) {} });
    }
    const batchCtrl: BatchControl = {
      cancelled: false,
      unsubs: new Map(),
      pollTimers: new Map(),
      cancelHandlers: new Map(),
    };
    activeBatchControl = batchCtrl;

    const batchStartTime = Date.now();
    set((state: any) => {
      const initialMap = { ...state.projectJobStatuses };
      queue.forEach((p) => {
        if (!initialMap[p.id] || initialMap[p.id].status !== 'completed') {
          initialMap[p.id] = {
            status: 'queued',
            progress: 0,
            message: isVideoOcr ? 'Queued for Video OCR' : 'Queued for ASR',
            stage: 'prepare',
          };
        }
      });
      return {
        projectJobStatuses: initialMap,
        startTime: batchStartTime,
        elapsedSeconds: 0,
      };
    });

    const processNext = async () => {
      if (batchCtrl.cancelled) {
        while (queue.length > 0) {
          const rem = queue.shift()!;
          set((state: any) => ({
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [rem.id]: {
                ...(state.projectJobStatuses[rem.id] || {}),
                status: 'cancelled',
                progress: 0,
                message: 'Job cancelled by user',
                stage: 'cancelled',
              },
            },
          }));
        }
        return;
      }
      if (queue.length === 0) return;
      const sub = queue.shift()!;
      const pid = sub.id;
      const mediaId = sub.media_id || (sub as any).mediaId;

      if (!mediaId) {
        set((state: any) => ({
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [pid]: { status: 'failed', progress: 0, message: 'Missing media', stage: 'failed' },
          },
        }));
        void processNext();
        return;
      }

      runningCount++;
      const runningLabel = isVideoOcr ? 'Running Video OCR…' : 'Processing ASR…';
      set((state: any) => {
        const isCurrent = state.activeProjectId === pid;
        return {
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [pid]: { status: 'running', progress: 0, message: runningLabel, stage: 'prepare' },
          },
          ...(isCurrent ? {
            jobStatus: 'running',
            jobStage: 'prepare',
            jobProgress: 0,
            jobMessage: runningLabel,
            startTime: state.startTime ?? batchStartTime,
          } : {}),
        };
      });

      const body = {
        mediaId,
        projectId: pid,
        jobType: 'asr',
        options: {
          ...config,
          recognType: Number(config.recognType ?? 1),
          modelName: String(config.modelName || 'nova-3'),
          projectId: pid,
          sourceLanguage: languages?.source?.code || 'zh-cn',
          targetLanguage: languages?.target?.code || 'vi',
          timingMode: languages?.timingMode || 'voice',
          forceAsr: Boolean(get().forceAsr),
          speakerDiarization: Boolean(engines?.speakerDiarization),
          speakerCount: Number(engines?.speakerCount || 0),
          subtitleSource,
          ocrRoi,
          ocrRoiConfirmed,
          ocrOptions,
        },
      };

      try {
        const res = await startJob(body);
        const jobId = res.id || res.jobId;
        if (!jobId) {
          throw new Error('Server did not return a job ID');
        }
        if (batchCtrl.cancelled) {
          runningCount = Math.max(0, runningCount - 1);
          void cancelJob(jobId).catch(() => {});
          set((state: any) => ({
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [pid]: { jobId, status: 'cancelled', progress: 0, message: 'Job cancelled by user', stage: 'cancelled' },
            },
            ...(state.activeProjectId === pid ? {
              jobStatus: 'cancelled',
              jobMessage: 'Job cancelled by user',
            } : {}),
          }));
          void processNext();
          return;
        }
        settledJobIds.delete(jobId);

        set((state: any) => {
          const isCurrent = state.activeProjectId === pid;
          return {
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [pid]: {
                ...(state.projectJobStatuses[pid] || {}),
                jobId,
                status: 'running',
                progress: state.projectJobStatuses[pid]?.progress ?? 0,
                message: state.projectJobStatuses[pid]?.message || runningLabel,
                stage: state.projectJobStatuses[pid]?.stage || 'prepare',
              },
            },
            ...(isCurrent ? {
              activeJob: {
                jobId,
                projectId: pid,
                jobType: 'asr',
                stage: 'running',
                status: 'running',
                progress: 0,
                message: runningLabel,
              },
              jobStatus: 'running',
              jobStage: 'prepare',
              startTime: state.startTime ?? batchStartTime,
            } : {}),
          };
        });

        let settled = false;
        let unsub: (() => void) | null = null;
        let fallbackTimer: any = null;

        const cleanupJobTrackers = () => {
          if (fallbackTimer) {
            clearInterval(fallbackTimer);
            fallbackTimer = null;
          }
          batchCtrl.pollTimers.delete(jobId);
          batchCtrl.unsubs.delete(jobId);
          batchCtrl.cancelHandlers.delete(jobId);
          unsub?.();
        };

        const applyFailure = (message: string, status: 'failed' | 'cancelled' = 'failed') => {
          set((state: any) => ({
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [pid]: { jobId, status, progress: 0, message, stage: status },
            },
            ...(state.activeJob?.jobId === jobId ? {
              activeJob: null,
            } : {}),
            ...(state.activeProjectId === pid ? {
              jobStatus: status,
              jobMessage: message,
            } : {}),
            ...(status === 'failed' ? {
              sessionSubprojects: state.sessionSubprojects.map((p: any) =>
                p.id === pid ? { ...p, status: 'failed' } : p
              ),
            } : {}),
          }));
          void processNext();
        };

        const settleSuccess = async (segments?: any[], transOpts?: any) => {
          if (settled) return;
          settled = true;
          settledJobIds.add(jobId);
          runningCount = Math.max(0, runningCount - 1);
          cleanupJobTrackers();
          try {
            if (!segments || segments.length === 0) {
              const fullJob = await fetchJob(jobId);
              if (fullJob?.status === 'failed' || fullJob?.error) {
                applyFailure(fullJob.error || fullJob.message || 'ASR failed', 'failed');
                return;
              }
              if (fullJob?.status === 'cancelled') {
                applyFailure(fullJob.message || 'ASR cancelled', 'cancelled');
                return;
              }
              segments = fullJob?.segments || [];
              transOpts = fullJob?.transcriptOptions || transOpts;
            }

            const isCurrent = get().activeProjectId === pid;
            if (isCurrent) {
              if (segments && segments.length > 0) {
                get().setSegments(segments);
              }
              if (transOpts) {
                get().setTranscriptOptions(transOpts);
              }
              set({
                jobStatus: 'completed',
                jobProgress: 100,
                jobMessage: 'Speech recognition complete',
                currentStep: 2,
                maxUnlockedStep: Math.max(get().maxUnlockedStep || 1, 2),
              });
              get().triggerAutosave();
            } else {
              const { updateProjectState } = await import('../api/projects');
              await updateProjectState(pid, {
                segments: segments || [],
                transcriptOptions: transOpts,
                currentStep: 2,
              }, 2, mediaId);
            }
          } catch (_) {}

          set((state: any) => ({
            ...(state.activeJob?.jobId === jobId ? {
              activeJob: null,
            } : {}),
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [pid]: { jobId, status: 'completed', progress: 100, message: 'Ready for Review (Stage 2)', stage: 'completed' },
            },
            sessionSubprojects: state.sessionSubprojects.map((p: any) =>
              p.id === pid ? { ...p, stage: 2, status: 'completed' } : p
            ),
          }));

          void processNext();
        };

        const settleFailure = (message: string, status: 'failed' | 'cancelled' = 'failed') => {
          if (settled) return;
          settled = true;
          settledJobIds.add(jobId);
          runningCount = Math.max(0, runningCount - 1);
          cleanupJobTrackers();
          applyFailure(message, status);
        };

        batchCtrl.cancelHandlers.set(jobId, () => {
          settleFailure('Job cancelled by user', 'cancelled');
        });

        if (res.status === 'succeeded') {
          await settleSuccess(res.segments, res.transcriptOptions);
          return;
        }

        const startFallbackPoll = () => {
          if (settled || fallbackTimer) return;
          fallbackTimer = setInterval(async () => {
            if (settled) {
              cleanupJobTrackers();
              return;
            }
            try {
              const j = await fetchJob(jobId);
              if (settled || !j) return;
              if (j.status === 'succeeded') {
                await settleSuccess(j.segments, j.transcriptOptions);
              } else if (j.status === 'failed' || j.error) {
                settleFailure(j.error || j.message || 'ASR failed', 'failed');
              } else if (j.status === 'cancelled') {
                settleFailure(j.message || 'ASR cancelled', 'cancelled');
              }
            } catch (_) {}
          }, 1000);
          batchCtrl.pollTimers.set(jobId, fallbackTimer);
        };

        unsub = subscribeJobStream(
          jobId,
          0,
          (payload) => {
            if (settled) return;

            const isSuccess =
              payload.status === 'succeeded' ||
              payload.kind === 'succeeded' ||
              payload.stage === 'task_done';
            const isFailed = payload.status === 'failed' || Boolean(payload.error);
            const isCancelled = payload.status === 'cancelled';

            if (isSuccess) {
              void settleSuccess(payload.details?.segments, payload.details?.transcript_options);
              return;
            }
            if (isFailed) {
              settleFailure(payload.error || payload.message || 'ASR failed', 'failed');
              return;
            }
            if (isCancelled) {
              settleFailure('ASR cancelled', 'cancelled');
              return;
            }

            const prog = typeof payload.progress === 'number' ? payload.progress : undefined;
            const msg = payload.message || '';
            const stg = payload.stage || '';

            set((state: any) => {
              const prev = state.projectJobStatuses[pid] || {};
              const updated = {
                ...prev,
                jobId,
                status: 'running',
                progress: prog !== undefined ? prog : prev.progress,
                message: msg || prev.message,
                stage: stg || prev.stage,
              };
              const isCurrent = state.activeProjectId === pid;
              return {
                projectJobStatuses: {
                  ...state.projectJobStatuses,
                  [pid]: updated,
                },
                ...(isCurrent ? {
                  jobStatus: 'running',
                  jobProgress: prog !== undefined ? prog : state.jobProgress,
                  jobMessage: msg || state.jobMessage,
                  jobStage: stg || state.jobStage,
                  ...(state.startTime ? { elapsedSeconds: Math.round((Date.now() - state.startTime) / 1000) } : {}),
                } : {}),
              };
            });
          },
          async (donePayload) => {
            if (settled) return;
            if (donePayload?.status === 'failed') {
              settleFailure((donePayload as any)?.error || donePayload?.message || 'ASR failed', 'failed');
            } else if (donePayload?.status === 'cancelled') {
              settleFailure('ASR cancelled', 'cancelled');
            } else {
              await settleSuccess();
            }
          },
          async (err) => {
            if (settled) return;
            try {
              const fullJob = await fetchJob(jobId);
              if (settled) return;
              if (fullJob?.status === 'succeeded') {
                await settleSuccess(fullJob.segments, fullJob.transcriptOptions);
                return;
              }
              if (fullJob?.status === 'cancelled') {
                settleFailure(fullJob.message || 'ASR cancelled', 'cancelled');
                return;
              }
              if (fullJob?.status === 'failed' || fullJob?.error) {
                settleFailure(fullJob.error || fullJob.message || err?.message || 'Stream error', 'failed');
                return;
              }
              if (fullJob && (fullJob.status === 'running' || fullJob.status === 'queued')) {
                startFallbackPoll();
                return;
              }
            } catch (_) {}
            if (!settled) {
              settleFailure(err?.message || 'Stream error', 'failed');
            }
          }
        );
        if (settled) {
          cleanupJobTrackers();
        } else {
          batchCtrl.unsubs.set(jobId, unsub);
        }
      } catch (err: any) {
        runningCount = Math.max(0, runningCount - 1);
        set((state: any) => ({
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [pid]: { status: 'failed', progress: 0, message: err.message || 'Job submission failed', stage: 'failed' },
          },
          ...(state.activeProjectId === pid ? {
            jobStatus: 'failed',
            jobMessage: err.message || 'Job submission failed',
          } : {}),
        }));
        void processNext();
      }
    };

    const toStart = Math.min(maxConcurrency, queue.length);
    for (let i = 0; i < toStart; i++) {
      void processNext();
    }
  },

  startPrepareJob: async () => {
    const backend = get().backend;
    const mediaId = backend?.mediaId;
    const project = get().project;
    const subtitleSource = get().subtitleSource || 'audio_asr';
    const isVideoOcr = subtitleSource === 'video_ocr';

    if (!mediaId || !project.verified) {
      set({
        jobStatus: 'failed',
        jobMessage: 'Please upload and verify a video file first.',
      });
      return;
    }

    if (isVideoOcr && project.hasVideo === false) {
      const msg = 'Video OCR requires a video stream.';
      const curProjId = get().activeProjectId;
      set((state: any) => ({
        jobStatus: 'failed',
        jobMessage: msg,
        ...(curProjId ? {
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [curProjId]: { status: 'failed', progress: 0, message: msg, stage: 'failed' },
          },
        } : {}),
      }));
      return;
    }

    if (isVideoOcr && !get().ocrRoiConfirmed) {
      const msg = 'Please test and confirm the subtitle ROI before starting Video OCR.';
      const curProjId = get().activeProjectId;
      set((state: any) => ({
        jobStatus: 'failed',
        jobMessage: msg,
        ...(curProjId ? {
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [curProjId]: { status: 'failed', progress: 0, message: msg, stage: 'failed' },
          },
        } : {}),
      }));
      return;
    }

    const provider = backend.options?.asrProviders?.find(
      (item: any) => item.recognType === Number(backend.config.recognType)
    );
    if (!isVideoOcr && provider?.requiresSettings && !provider.configured) {
      set({
        jobStatus: 'failed',
        jobMessage: `Please configure ${provider.label} API key first.`,
      });
      return;
    }

    // Ensure active project exists before starting job
    let activeProjectId = get().activeProjectId;
    if (!activeProjectId) {
      activeProjectId = await get().createNewProject(project.filename || 'Untitled Video Project', mediaId, project.durationSec);
    }

    const initMsg = isVideoOcr ? 'Initiating hard-subtitle Video OCR…' : 'Initiating speech recognition & diarization…';
    set((state: any) => ({
      jobStatus: 'running',
      jobProgress: 0,
      jobStage: 'prepare',
      jobMessage: initMsg,
      startTime: Date.now(),
      elapsedSeconds: 0,
      ...(activeProjectId ? {
        projectJobStatuses: {
          ...state.projectJobStatuses,
          [activeProjectId]: {
            status: 'running',
            progress: 0,
            message: initMsg,
            stage: 'prepare',
          },
        },
      } : {}),
    }));

    const body = buildPrepareJobRequest(get());

    try {
      const res = await startJob(body);
      const jobId = res.id || res.jobId;
      if (!jobId) {
        throw new Error('Server did not return a job ID');
      }
      settledJobIds.delete(jobId);

      if (res.status === 'succeeded') {
        const transOpts = res.transcriptOptions || (res as any)?.details?.transcript_options;
        const isCurrent = !activeProjectId || get().activeProjectId === activeProjectId;
        if (isCurrent) {
          if (transOpts) {
            get().setTranscriptOptions(transOpts);
          }
          if (Array.isArray(res.segments) && res.segments.length > 0) {
            get().setSegments(res.segments);
          }
          const detectedLang = res.details?.detected_language || (res as any)?.detected_language;
          if (detectedLang) {
            const avail = get().backend?.options?.languages || [];
            const opt = avail.find((l: any) => l.code === detectedLang || l.code.startsWith(detectedLang));
            if (opt && get().languages?.source?.code !== opt.code) {
              get().updateSourceLanguage(opt.code, opt.name);
            }
          }
          set({
            forceAsr: false,
            jobStatus: 'completed',
            jobProgress: 100,
            jobMessage: 'Job completed successfully',
            currentStep: 2,
            maxUnlockedStep: Math.max(get().maxUnlockedStep || 1, 2),
          });
          get().triggerAutosave();
        } else if (activeProjectId) {
          const { updateProjectState } = await import('../api/projects');
          await updateProjectState(activeProjectId, {
            segments: res.segments || [],
            transcriptOptions: transOpts,
            currentStep: 2,
          }, 2);
        }
        if (activeProjectId) {
          set((state: any) => ({
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [activeProjectId!]: { jobId, status: 'completed', progress: 100, message: 'Ready for Review (Stage 2)', stage: 'completed' },
            },
            sessionSubprojects: state.sessionSubprojects.map((p: any) =>
              p.id === activeProjectId ? { ...p, stage: 2, status: 'completed' } : p
            ),
          }));
        }
        get().loadProjects();
        return;
      }

      get().subscribeToJob(jobId, activeProjectId);
    } catch (err: any) {
      const errMsg = err.message || 'Job submission failed';
      set((state: any) => ({
        ...((!activeProjectId || state.activeProjectId === activeProjectId) ? {
          jobStatus: 'failed',
          jobMessage: errMsg,
        } : {}),
        ...(activeProjectId ? {
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [activeProjectId]: { status: 'failed', progress: 0, message: errMsg, stage: 'failed' },
          },
        } : {}),
      }));
    }
  },

  subscribeToJob: (jobId: string, targetProjectIdParam?: string | null, options?: { jobType?: 'asr' | 'render' | string }) => {
    const targetProjectId = targetProjectIdParam !== undefined ? targetProjectIdParam : get().activeProjectId;
    const initialJobType = options?.jobType || (get().editVideo?.exporting ? 'render' : 'asr');
    const isInitialRender = initialJobType === 'render';

    // Clear previous subscription and poll timer
    const prevJobId = get().activeJob?.jobId;
    if (prevJobId && prevJobId !== jobId) {
      activeJobSubTokens.delete(prevJobId);
      activePollTokens.delete(prevJobId);
    }
    if (get().sseUnsubscribe) {
      get().sseUnsubscribe();
    }
    if (get().pollTimer) {
      clearInterval(get().pollTimer);
    }

    settledJobIds.delete(jobId);
    const subToken = Symbol(jobId);
    activeJobSubTokens.set(jobId, subToken);

    set((state: any) => ({
      activeJob: {
        jobId,
        projectId: targetProjectId || undefined,
        jobType: initialJobType,
        stage: isInitialRender ? 'render' : 'running',
        status: 'running',
        progress: 0,
        message: isInitialRender ? 'Rendering dubbed video…' : 'Processing job…',
      },
      ...((!targetProjectId || state.activeProjectId === targetProjectId) ? {
        jobStatus: 'running',
        ...(isInitialRender ? { jobStage: 'render', jobProgress: 0, jobMessage: 'Rendering dubbed video…' } : {}),
      } : {}),
      ...(targetProjectId ? {
        projectJobStatuses: {
          ...state.projectJobStatuses,
          [targetProjectId]: {
            jobId,
            status: 'running',
            progress: 0,
            message: isInitialRender ? 'Rendering dubbed video…' : 'Processing job…',
            stage: isInitialRender ? 'render' : 'prepare',
          },
        },
      } : {}),
    }));

    let settled = false;
    let unsub: (() => void) | null = null;

    // Start 1s polling fallback
    const timer = setInterval(() => {
      get().pollJob(jobId, targetProjectId, { jobType: initialJobType });
    }, 1000);
    set({ pollTimer: timer });

    const stopTimers = () => {
      if (get().pollTimer) {
        clearInterval(get().pollTimer);
        set({ pollTimer: null });
      }
      unsub?.();
      if (get().sseUnsubscribe === unsub) {
        set({ sseUnsubscribe: null });
      }
    };

    const applyTerminalFailure = (status: 'failed' | 'cancelled', message: string, isRenderJob: boolean) => {
      set((state: any) => {
        const isCurrent = !targetProjectId || state.activeProjectId === targetProjectId;
        return {
          activeJob: state.activeJob?.jobId === jobId ? { ...state.activeJob, status } : state.activeJob,
          ...(isCurrent ? {
            jobStatus: status,
            jobMessage: message,
            ...(isRenderJob ? {
              jobStage: 'render',
              editVideo: {
                ...state.editVideo,
                exporting: false,
                error: message || (status === 'cancelled' ? 'Render cancelled' : 'Video export failed'),
              },
            } : {}),
          } : {}),
          ...(targetProjectId ? {
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [targetProjectId]: { jobId, status, progress: 0, message, stage: isRenderJob ? 'render' : status },
            },
            ...(status === 'failed' && !isRenderJob ? {
              sessionSubprojects: state.sessionSubprojects.map((p: any) =>
                p.id === targetProjectId ? { ...p, status: 'failed' } : p
              ),
            } : {}),
          } : {}),
        };
      });
    };

    const handleSuccess = async (message?: string, eventSegments?: any[], eventTransOpts?: any, eventJobType?: string) => {
      if (settled || settledJobIds.has(jobId) || activeJobSubTokens.get(jobId) !== subToken) return;
      settled = true;
      settledJobIds.add(jobId);
      if (activeJobSubTokens.get(jobId) === subToken) {
        activeJobSubTokens.delete(jobId);
      }
      stopTimers();

      let isRender = eventJobType === 'render' || isInitialRender || get().activeJob?.jobType === 'render';
      let segments = eventSegments;
      let transOpts = eventTransOpts;
      try {
        const fullJob = await fetchJob(jobId);
        if (fullJob?.jobType === 'render') {
          isRender = true;
        }
        if (fullJob?.status === 'failed' || fullJob?.error) {
          applyTerminalFailure('failed', fullJob.error || fullJob.message || 'Processing failed', isRender);
          return;
        }
        if (fullJob?.status === 'cancelled') {
          applyTerminalFailure('cancelled', fullJob.message || 'Processing cancelled', isRender);
          return;
        }
        if (!isRender) {
          transOpts = fullJob?.transcriptOptions || (fullJob as any)?.details?.transcript_options || transOpts;
          if (fullJob?.segments && Array.isArray(fullJob.segments) && fullJob.segments.length > 0) {
            segments = fullJob.segments;
          }
        }
      } catch (_) {}

      const isCurrent = !targetProjectId || get().activeProjectId === targetProjectId;
      if (isRender) {
        const doneMsg = message || 'Video rendered successfully';
        if (isCurrent) {
          set((state: any) => ({
            activeJob: state.activeJob?.jobId === jobId ? { ...state.activeJob, status: 'completed', progress: 100, message: doneMsg } : state.activeJob,
            editVideo: { ...state.editVideo, exporting: false, error: null },
            jobStatus: 'completed',
            jobStage: 'render',
            jobProgress: 100,
            jobMessage: doneMsg,
            currentStep: Math.max(state.currentStep || 4, 4),
            maxUnlockedStep: Math.max(state.maxUnlockedStep || 4, 4),
          }));
        }
        if (targetProjectId) {
          set((state: any) => ({
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [targetProjectId]: { jobId, status: 'completed', progress: 100, message: doneMsg, stage: 'render' },
            },
          }));
        }
        get().loadProjects();
        return;
      }

      if (isCurrent) {
        if (transOpts) {
          get().setTranscriptOptions(transOpts);
        }
        if (segments && Array.isArray(segments) && segments.length > 0) {
          get().setSegments(segments);
        }
        set({
          forceAsr: false,
          jobStatus: 'completed',
          jobProgress: 100,
          jobMessage: message || 'Speech recognition complete',
          currentStep: 2,
          maxUnlockedStep: Math.max(get().maxUnlockedStep || 1, 2),
        });
        get().triggerAutosave();
      } else if (targetProjectId) {
        try {
          const { updateProjectState } = await import('../api/projects');
          await updateProjectState(targetProjectId, {
            segments: segments || [],
            transcriptOptions: transOpts,
            currentStep: 2,
          }, 2);
        } catch (_) {}
      }

      if (targetProjectId) {
        set((state: any) => ({
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [targetProjectId]: { jobId, status: 'completed', progress: 100, message: message || 'Ready for Review (Stage 2)', stage: 'completed' },
          },
          sessionSubprojects: state.sessionSubprojects.map((p: any) =>
            p.id === targetProjectId ? { ...p, stage: 2, status: 'completed' } : p
          ),
        }));
      }
      get().loadProjects();
    };

    const handleTerminalFailure = (status: 'failed' | 'cancelled', message: string, isRenderJob: boolean) => {
      if (settled || settledJobIds.has(jobId) || activeJobSubTokens.get(jobId) !== subToken) return;
      settled = true;
      settledJobIds.add(jobId);
      if (activeJobSubTokens.get(jobId) === subToken) {
        activeJobSubTokens.delete(jobId);
      }
      stopTimers();
      applyTerminalFailure(status, message, isRenderJob);
    };

    unsub = subscribeJobStream(
      jobId,
      0,
      async (payload) => {
        if (settled || settledJobIds.has(jobId) || activeJobSubTokens.get(jobId) !== subToken) return;
        const isCurrent = !targetProjectId || get().activeProjectId === targetProjectId;
        const isRenderEvent = isInitialRender || payload.jobType === 'render' || get().activeJob?.jobType === 'render';
        const st = get().startTime;
        if (st && isCurrent) {
          set({ elapsedSeconds: Math.round((Date.now() - st) / 1000) });
        }

        if (isCurrent) {
          if (typeof payload.progress === 'number') {
            set({ jobProgress: payload.progress });
          }
          if (isRenderEvent) {
            set({ jobStage: 'render' });
          } else if (payload.stage) {
            set({ jobStage: payload.stage });
          }
          if (payload.message) {
            set({ jobMessage: payload.message });
          }
        }

        if (targetProjectId) {
          set((state: any) => {
            const prev = state.projectJobStatuses[targetProjectId] || {};
            return {
              projectJobStatuses: {
                ...state.projectJobStatuses,
                [targetProjectId]: {
                  ...prev,
                  jobId,
                  status: 'running',
                  progress: typeof payload.progress === 'number' ? payload.progress : prev.progress,
                  stage: isRenderEvent ? 'render' : (payload.stage || prev.stage),
                  message: payload.message || prev.message,
                },
              },
            };
          });
        }

        const streamTransOpts = payload.details?.transcript_options || payload.details?.transcriptOptions;
        if (isCurrent && !isRenderEvent) {
          if (streamTransOpts) {
            get().setTranscriptOptions(streamTransOpts);
          }
          if (payload.details?.segments && Array.isArray(payload.details.segments)) {
            get().setSegments(payload.details.segments);
          }

          const detectedLang = payload.details?.detected_language || (payload.details as any)?.source_language;
          if (detectedLang) {
            const avail = get().backend?.options?.languages || [];
            const opt = avail.find((l: any) => l.code === detectedLang || l.code.startsWith(detectedLang));
            if (opt && get().languages?.source?.code !== opt.code) {
              get().updateSourceLanguage(opt.code, opt.name);
            }
          }
        }

        const isSuccess =
          payload.status === 'succeeded' ||
          payload.kind === 'succeeded' ||
          payload.stage === 'task_done';
        const isFailed = payload.status === 'failed' || Boolean(payload.error);
        const isCancelled = payload.status === 'cancelled';

        if (isSuccess) {
          await handleSuccess(payload.message, payload.details?.segments, streamTransOpts, payload.jobType);
        } else if (isFailed) {
          handleTerminalFailure('failed', payload.error || payload.message || 'Processing failed', isRenderEvent);
        } else if (isCancelled) {
          handleTerminalFailure('cancelled', payload.message || 'Processing cancelled', isRenderEvent);
        }
      },
      async (donePayload) => {
        if (settled || settledJobIds.has(jobId) || activeJobSubTokens.get(jobId) !== subToken) return;
        const isRenderDone = isInitialRender || donePayload?.jobType === 'render' || get().activeJob?.jobType === 'render';
        if (donePayload?.status === 'failed') {
          handleTerminalFailure('failed', (donePayload as any)?.error || donePayload?.message || 'Processing failed', isRenderDone);
        } else if (donePayload?.status === 'cancelled') {
          handleTerminalFailure('cancelled', donePayload?.message || 'Processing cancelled', isRenderDone);
        } else {
          await handleSuccess(donePayload?.message, undefined, undefined, donePayload?.jobType);
        }
      },
      (err) => {
        console.warn('SSE stream notice:', err);
      }
    );

    if (settled) {
      unsub();
    } else {
      set({ sseUnsubscribe: unsub });
    }
  },

  pollJob: async (jobId: string, targetProjectIdParam?: string | null, options?: { jobType?: 'asr' | 'render' | string }) => {
    if (settledJobIds.has(jobId)) return;
    const subToken = activeJobSubTokens.get(jobId);
    const pollToken = Symbol('poll');
    activePollTokens.set(jobId, pollToken);
    const targetProjectId = targetProjectIdParam !== undefined ? targetProjectIdParam : get().activeProjectId;
    try {
      const job = await fetchJob(jobId);
      if (!job) return;
      if (settledJobIds.has(jobId)) return;
      if (subToken !== undefined && activeJobSubTokens.get(jobId) !== subToken) return;
      if (activePollTokens.get(jobId) !== pollToken) return;

      const isRender = options?.jobType === 'render' || job.jobType === 'render' || (get().activeJob?.jobId === jobId && get().activeJob?.jobType === 'render');
      const isCurrent = !targetProjectId || get().activeProjectId === targetProjectId;
      const st = get().startTime;
      if (st && isCurrent) {
        set({ elapsedSeconds: Math.round((Date.now() - st) / 1000) });
      }

      if (isCurrent) {
        if (typeof job.progress === 'number') {
          set({ jobProgress: job.progress });
        }
        if (isRender) {
          set({ jobStage: 'render' });
        } else if (job.stage) {
          set({ jobStage: job.stage });
        }
        if (job.message) {
          set({ jobMessage: job.message });
        }
      }

      if (job.status === 'succeeded') {
        settledJobIds.add(jobId);
        if (subToken !== undefined && activeJobSubTokens.get(jobId) === subToken) {
          activeJobSubTokens.delete(jobId);
        }
        if (get().pollTimer) {
          clearInterval(get().pollTimer);
          set({ pollTimer: null });
        }
        if (get().sseUnsubscribe) {
          get().sseUnsubscribe();
          set({ sseUnsubscribe: null });
        }
        if (isRender) {
          const doneMsg = job.message || 'Video rendered successfully';
          if (isCurrent) {
            set((state: any) => ({
              activeJob: state.activeJob?.jobId === jobId ? { ...state.activeJob, status: 'completed', progress: 100, message: doneMsg } : state.activeJob,
              editVideo: { ...state.editVideo, exporting: false, error: null },
              jobStatus: 'completed',
              jobStage: 'render',
              jobProgress: 100,
              jobMessage: doneMsg,
              currentStep: Math.max(state.currentStep || 4, 4),
              maxUnlockedStep: Math.max(state.maxUnlockedStep || 4, 4),
            }));
          }
          if (targetProjectId) {
            set((state: any) => ({
              projectJobStatuses: {
                ...state.projectJobStatuses,
                [targetProjectId]: { jobId, status: 'completed', progress: 100, message: doneMsg, stage: 'render' },
              },
            }));
          }
          get().loadProjects();
          return;
        }
        const transOpts = job.transcriptOptions ?? (job as any)?.details?.transcript_options ?? (job as any)?.details?.transcriptOptions;
        if (isCurrent) {
          if (transOpts) {
            get().setTranscriptOptions(transOpts);
          }
          if (job.segments && Array.isArray(job.segments) && job.segments.length > 0) {
            get().setSegments(job.segments);
          }
          const detectedLang = (job as any)?.details?.detected_language || (job as any)?.details?.source_language || (job as any)?.detected_language;
          if (detectedLang) {
            const avail = get().backend?.options?.languages || [];
            const opt = avail.find((l: any) => l.code === detectedLang || l.code.startsWith(detectedLang));
            if (opt && get().languages?.source?.code !== opt.code) {
              get().updateSourceLanguage(opt.code, opt.name);
            }
          }
          set({
            forceAsr: false,
            jobStatus: 'completed',
            jobProgress: 100,
            jobMessage: job.message || 'Speech recognition complete',
            currentStep: 2,
            maxUnlockedStep: Math.max(get().maxUnlockedStep || 1, 2),
          });
          get().triggerAutosave();
        } else if (targetProjectId) {
          try {
            const { updateProjectState } = await import('../api/projects');
            await updateProjectState(targetProjectId, {
              segments: job.segments || [],
              transcriptOptions: transOpts,
              currentStep: 2,
            }, 2);
          } catch (_) {}
        }
        if (targetProjectId) {
          set((state: any) => ({
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [targetProjectId]: { jobId, status: 'completed', progress: 100, message: job.message || 'Ready for Review (Stage 2)', stage: 'completed' },
            },
            sessionSubprojects: state.sessionSubprojects.map((p: any) =>
              p.id === targetProjectId ? { ...p, stage: 2, status: 'completed' } : p
            ),
          }));
        }
        get().loadProjects();
      } else if (job.status === 'failed' || job.error) {
        settledJobIds.add(jobId);
        if (subToken !== undefined && activeJobSubTokens.get(jobId) === subToken) {
          activeJobSubTokens.delete(jobId);
        }
        if (get().pollTimer) {
          clearInterval(get().pollTimer);
          set({ pollTimer: null });
        }
        if (get().sseUnsubscribe) {
          get().sseUnsubscribe();
          set({ sseUnsubscribe: null });
        }
        const errMsg = job.error || job.message || 'Processing failed';
        set((state: any) => ({
          activeJob: state.activeJob?.jobId === jobId ? { ...state.activeJob, status: 'failed' } : state.activeJob,
          ...(isCurrent ? {
            jobStatus: 'failed',
            jobMessage: errMsg,
            ...(isRender ? {
              jobStage: 'render',
              editVideo: { ...state.editVideo, exporting: false, error: errMsg },
            } : {}),
          } : {}),
          ...(targetProjectId ? {
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [targetProjectId]: { jobId, status: 'failed', progress: 0, message: errMsg, stage: isRender ? 'render' : 'failed' },
            },
            ...(!isRender ? {
              sessionSubprojects: state.sessionSubprojects.map((p: any) =>
                p.id === targetProjectId ? { ...p, status: 'failed' } : p
              ),
            } : {}),
          } : {}),
        }));
      } else if (job.status === 'cancelled') {
        settledJobIds.add(jobId);
        if (subToken !== undefined && activeJobSubTokens.get(jobId) === subToken) {
          activeJobSubTokens.delete(jobId);
        }
        if (get().pollTimer) {
          clearInterval(get().pollTimer);
          set({ pollTimer: null });
        }
        if (get().sseUnsubscribe) {
          get().sseUnsubscribe();
          set({ sseUnsubscribe: null });
        }
        const cancelMsg = job.message || 'Processing cancelled';
        set((state: any) => ({
          activeJob: state.activeJob?.jobId === jobId ? { ...state.activeJob, status: 'cancelled' } : state.activeJob,
          ...(isCurrent ? {
            jobStatus: 'cancelled',
            jobMessage: cancelMsg,
            ...(isRender ? {
              jobStage: 'render',
              editVideo: { ...state.editVideo, exporting: false, error: cancelMsg },
            } : {}),
          } : {}),
          ...(targetProjectId ? {
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [targetProjectId]: { jobId, status: 'cancelled', progress: 0, message: cancelMsg, stage: isRender ? 'render' : 'cancelled' },
            },
          } : {}),
        }));
      } else if (targetProjectId) {
        set((state: any) => {
          const prev = state.projectJobStatuses[targetProjectId] || {};
          return {
            projectJobStatuses: {
              ...state.projectJobStatuses,
              [targetProjectId]: {
                ...prev,
                jobId,
                status: 'running',
                progress: typeof job.progress === 'number' ? job.progress : prev.progress,
                stage: isRender ? 'render' : (job.stage || prev.stage),
                message: job.message || prev.message,
              },
            },
          };
        });
      }
    } catch (err) {
      console.warn('Poll job notice:', err);
    }
  },

  cancelActiveJob: async () => {
    const job = get().activeJob;
    const activeProjectId = get().activeProjectId;
    const pStatuses = get().projectJobStatuses || {};
    const sessionSubs: any[] = get().sessionSubprojects || [];
    const sessionSubIds = new Set(sessionSubs.map((s) => s.id));
    const isRender = job?.jobType === 'render' || get().jobStage === 'render' || Boolean(get().editVideo?.exporting);

    const isActiveStatus = (st?: string | null) => st === 'running' || st === 'queued';
    const jobIdsToCancel = new Set<string>();
    const projectIdsToCancel = new Set<string>();
    if (job?.jobId) {
      const targetPid = job.projectId || activeProjectId;
      const psStatus = targetPid ? pStatuses[targetPid]?.status : undefined;
      const effectiveStatus = psStatus ?? job.status;
      if (isActiveStatus(effectiveStatus) || (isRender && Boolean(get().editVideo?.exporting))) {
        jobIdsToCancel.add(job.jobId);
        if (job.projectId) projectIdsToCancel.add(job.projectId);
      }
    }
    if (activeProjectId && pStatuses[activeProjectId]?.jobId && isActiveStatus(pStatuses[activeProjectId].status)) {
      jobIdsToCancel.add(pStatuses[activeProjectId].jobId!);
      projectIdsToCancel.add(activeProjectId);
    }
    if (!isRender && (activeBatchControl || sessionSubs.length > 0)) {
      for (const [pid, st] of Object.entries(pStatuses as Record<string, any>)) {
        if ((pid === activeProjectId || sessionSubIds.has(pid)) && st && isActiveStatus(st.status)) {
          projectIdsToCancel.add(pid);
          if (st.jobId) jobIdsToCancel.add(st.jobId);
        }
      }
    }

    if (jobIdsToCancel.size === 0 && projectIdsToCancel.size === 0 && (isRender || !activeBatchControl)) return;

    if (!isRender && activeBatchControl) {
      activeBatchControl.cancelled = true;
      for (const jid of jobIdsToCancel) {
        const handler = activeBatchControl.cancelHandlers.get(jid);
        if (handler) {
          try { handler(); } catch (_) {}
        }
      }
      activeBatchControl.unsubs.forEach((fn) => { try { fn(); } catch (_) {} });
      activeBatchControl.pollTimers.forEach((t) => { try { clearInterval(t); } catch (_) {} });
    }

    for (const jid of jobIdsToCancel) {
      settledJobIds.add(jid);
      activeJobSubTokens.delete(jid);
      activePollTokens.delete(jid);
    }
    if (get().pollTimer) {
      clearInterval(get().pollTimer);
      set({ pollTimer: null });
    }
    if (get().sseUnsubscribe) {
      get().sseUnsubscribe();
      set({ sseUnsubscribe: null });
    }
    const cancelMsg = isRender ? 'Render cancelled by user' : 'Job cancelled by user';
    try {
      await Promise.allSettled(Array.from(jobIdsToCancel).map((jid) => cancelJob(jid)));
      set((state: any) => {
        const nextStatuses = { ...state.projectJobStatuses };
        const shouldCancelActiveProject = Boolean(
          activeProjectId &&
          (projectIdsToCancel.has(activeProjectId) ||
            (job?.jobId &&
              jobIdsToCancel.has(job.jobId) &&
              (!job.projectId || job.projectId === activeProjectId)))
        );
        if (activeProjectId && shouldCancelActiveProject && !projectIdsToCancel.has(activeProjectId)) {
          projectIdsToCancel.add(activeProjectId);
        }
        for (const pid of projectIdsToCancel) {
          const prev = nextStatuses[pid] || {};
          nextStatuses[pid] = {
            ...prev,
            ...(prev.jobId || (pid === activeProjectId && job?.jobId) ? { jobId: prev.jobId || job?.jobId } : {}),
            status: 'cancelled',
            progress: 0,
            message: cancelMsg,
            stage: isRender ? 'render' : 'cancelled',
          };
        }
        const cancelActiveJobObj = Boolean(state.activeJob?.jobId && jobIdsToCancel.has(state.activeJob.jobId));
        return {
          activeJob: cancelActiveJobObj ? { ...state.activeJob, status: 'cancelled' } : state.activeJob,
          ...(shouldCancelActiveProject || !activeProjectId ? {
            jobStatus: 'cancelled',
            jobMessage: cancelMsg,
          } : {}),
          ...(isRender ? {
            jobStage: 'render',
            editVideo: { ...state.editVideo, exporting: false, error: cancelMsg },
          } : {}),
          projectJobStatuses: nextStatuses,
        };
      });
    } catch (err) {
      console.warn('Failed to cancel job:', err);
    }
  },

  clearJob: () => {
    const curId = get().activeJob?.jobId;
    if (curId) {
      settledJobIds.delete(curId);
      activeJobSubTokens.delete(curId);
      activePollTokens.delete(curId);
    }
    if (get().sseUnsubscribe) {
      get().sseUnsubscribe();
    }
    if (get().pollTimer) {
      clearInterval(get().pollTimer);
    }
    set({
      activeJob: null,
      jobProgress: null,
      jobStage: null,
      jobMessage: 'Ready',
      jobStatus: 'idle',
      sseUnsubscribe: null,
      pollTimer: null,
      startTime: null,
      elapsedSeconds: 0,
    });
  },
});
