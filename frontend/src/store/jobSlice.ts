import { StateCreator } from 'zustand';
import type { JobRecord } from '../types/job';
import { startJob, cancelJob, fetchJob, subscribeJobStream } from '../api/jobs';

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
  subscribeToJob: (jobId: string) => void;
  pollJob: (jobId: string) => Promise<void>;
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

    const queue = [...subprojects];
    const maxConcurrency = 2;
    let runningCount = 0;

    set((state: any) => {
      const initialMap = { ...state.projectJobStatuses };
      queue.forEach((p) => {
        if (!initialMap[p.id] || initialMap[p.id].status !== 'completed') {
          initialMap[p.id] = { status: 'queued', progress: 0, message: 'Queued for ASR', stage: 'prepare' };
        }
      });
      return { projectJobStatuses: initialMap };
    });

    const processNext = async () => {
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
      set((state: any) => {
        const isCurrent = state.activeProjectId === pid;
        return {
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [pid]: { status: 'running', progress: 0, message: 'Processing ASR…', stage: 'prepare' },
          },
          ...(isCurrent ? {
            jobStatus: 'running',
            jobProgress: 0,
            jobMessage: 'Processing ASR…',
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
        },
      };

      try {
        const res = await startJob(body);
        const jobId = res.id || res.jobId;
        if (!jobId) {
          throw new Error('Server did not return a job ID');
        }

        const handleSubSuccess = async (segments?: any[], transOpts?: any) => {
          runningCount--;
          try {
            if (!segments || segments.length === 0) {
              const fullJob = await fetchJob(jobId);
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

        if (res.status === 'succeeded') {
          await handleSubSuccess(res.segments, res.transcriptOptions);
          return;
        }

        subscribeJobStream(
          jobId,
          0,
          (payload) => {
            const prog = typeof payload.progress === 'number' ? payload.progress : undefined;
            const msg = payload.message || '';
            const stg = payload.stage || '';
            const status = payload.status === 'succeeded' ? 'completed' : (payload.status === 'failed' ? 'failed' : 'running');

            set((state: any) => {
              const prev = state.projectJobStatuses[pid] || {};
              const updated = {
                ...prev,
                jobId,
                status,
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
                  jobStatus: status,
                  jobProgress: prog !== undefined ? prog : state.jobProgress,
                  jobMessage: msg || state.jobMessage,
                  jobStage: stg || state.jobStage,
                } : {}),
              };
            });

            if (payload.status === 'succeeded' || payload.stage === 'task_done') {
              void handleSubSuccess(payload.details?.segments, payload.details?.transcript_options);
            } else if (payload.status === 'failed') {
              runningCount--;
              set((state: any) => ({
                projectJobStatuses: {
                  ...state.projectJobStatuses,
                  [pid]: { jobId, status: 'failed', progress: 0, message: payload.error || 'ASR failed', stage: 'failed' },
                },
              }));
              void processNext();
            }
          },
          async () => {
            await handleSubSuccess();
          },
          (err) => {
            runningCount--;
            set((state: any) => ({
              projectJobStatuses: {
                ...state.projectJobStatuses,
                [pid]: { jobId, status: 'failed', progress: 0, message: err?.message || 'Stream error', stage: 'failed' },
              },
            }));
            void processNext();
          }
        );
      } catch (err: any) {
        runningCount--;
        set((state: any) => ({
          projectJobStatuses: {
            ...state.projectJobStatuses,
            [pid]: { status: 'failed', progress: 0, message: err.message || 'Job submission failed', stage: 'failed' },
          },
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

    if (!mediaId || !project.verified) {
      set({
        jobStatus: 'failed',
        jobMessage: 'Please upload and verify a video file first.',
      });
      return;
    }

    const provider = backend.options?.asrProviders?.find(
      (item: any) => item.recognType === Number(backend.config.recognType)
    );
    if (provider?.requiresSettings && !provider.configured) {
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

    set({
      jobStatus: 'running',
      jobProgress: 0,
      jobStage: 'prepare',
      jobMessage: 'Initiating speech recognition & diarization…',
      startTime: Date.now(),
      elapsedSeconds: 0,
    });

    const body = buildPrepareJobRequest(get());

    try {
      const res = await startJob(body);
      const jobId = res.id || res.jobId;
      if (!jobId) {
        throw new Error('Server did not return a job ID');
      }

      if (res.status === 'succeeded') {
        const transOpts = res.transcriptOptions || (res as any)?.details?.transcript_options;
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
        get().loadProjects();
        return;
      }

      get().subscribeToJob(jobId);
    } catch (err: any) {
      set({
        jobStatus: 'failed',
        jobMessage: err.message || 'Job submission failed',
      });
    }
  },

  subscribeToJob: (jobId: string) => {
    // Clear previous subscription and poll timer
    if (get().sseUnsubscribe) {
      get().sseUnsubscribe();
    }
    if (get().pollTimer) {
      clearInterval(get().pollTimer);
    }

    set({
      activeJob: {
        jobId,
        stage: 'running',
        status: 'running',
        progress: 0,
        message: 'Processing job…',
      },
      jobStatus: 'running',
    });

    // Start 1s polling fallback
    const timer = setInterval(() => {
      get().pollJob(jobId);
    }, 1000);
    set({ pollTimer: timer });

    const handleSuccess = async (message?: string) => {
      if (get().pollTimer) {
        clearInterval(get().pollTimer);
        set({ pollTimer: null });
      }
      try {
        const fullJob = await fetchJob(jobId);
        const transOpts = fullJob?.transcriptOptions || (fullJob as any)?.details?.transcript_options;
        if (transOpts) {
          get().setTranscriptOptions(transOpts);
        }
        if (fullJob?.segments && Array.isArray(fullJob.segments) && fullJob.segments.length > 0) {
          get().setSegments(fullJob.segments);
        }
      } catch (_) {}

      set({
        forceAsr: false,
        jobStatus: 'completed',
        jobProgress: 100,
        jobMessage: message || 'Speech recognition complete',
        currentStep: 2,
        maxUnlockedStep: Math.max(get().maxUnlockedStep || 1, 2),
      });
      get().triggerAutosave();
      get().loadProjects();
    };

    const unsub = subscribeJobStream(
      jobId,
      0,
      async (payload) => {
        const st = get().startTime;
        if (st) {
          set({ elapsedSeconds: Math.round((Date.now() - st) / 1000) });
        }

        if (typeof payload.progress === 'number') {
          set({ jobProgress: payload.progress });
        }
        if (payload.stage) {
          set({ jobStage: payload.stage });
        }
        if (payload.message) {
          set({ jobMessage: payload.message });
        }

        const streamTransOpts = payload.details?.transcript_options || payload.details?.transcriptOptions;
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

        const isSuccess =
          payload.status === 'succeeded' ||
          payload.kind === 'succeeded' ||
          payload.stage === 'task_done';
        const isFailed = payload.status === 'failed' || Boolean(payload.error);
        const isCancelled = payload.status === 'cancelled';

        if (isSuccess) {
          await handleSuccess(payload.message);
        } else if (isFailed) {
          if (get().pollTimer) {
            clearInterval(get().pollTimer);
            set({ pollTimer: null });
          }
          set({
            jobStatus: 'failed',
            jobMessage: payload.error || payload.message || 'Processing failed',
          });
        } else if (isCancelled) {
          if (get().pollTimer) {
            clearInterval(get().pollTimer);
            set({ pollTimer: null });
          }
          set({
            jobStatus: 'cancelled',
            jobMessage: 'Processing cancelled',
          });
        }
      },
      async (donePayload) => {
        await handleSuccess(donePayload?.message);
      },
      (err) => {
        console.warn('SSE stream notice:', err);
      }
    );

    set({ sseUnsubscribe: unsub });
  },

  pollJob: async (jobId: string) => {
    try {
      const st = get().startTime;
      if (st) {
        set({ elapsedSeconds: Math.round((Date.now() - st) / 1000) });
      }

      const job = await fetchJob(jobId);
      if (!job) return;

      if (typeof job.progress === 'number') {
        set({ jobProgress: job.progress });
      }
      if (job.stage) {
        set({ jobStage: job.stage });
      }
      if (job.message) {
        set({ jobMessage: job.message });
      }

      if (job.status === 'succeeded') {
        if (get().pollTimer) {
          clearInterval(get().pollTimer);
          set({ pollTimer: null });
        }
        if (get().sseUnsubscribe) {
          get().sseUnsubscribe();
          set({ sseUnsubscribe: null });
        }
        if (job.segments && Array.isArray(job.segments) && job.segments.length > 0) {
          get().setSegments(job.segments);
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
        get().loadProjects();
      } else if (job.status === 'failed' || job.error) {
        if (get().pollTimer) {
          clearInterval(get().pollTimer);
          set({ pollTimer: null });
        }
        set({
          jobStatus: 'failed',
          jobMessage: job.error || job.message || 'Processing failed',
        });
      } else if (job.status === 'cancelled') {
        if (get().pollTimer) {
          clearInterval(get().pollTimer);
          set({ pollTimer: null });
        }
        set({
          jobStatus: 'cancelled',
          jobMessage: 'Processing cancelled',
        });
      }
    } catch (err) {
      console.warn('Poll job notice:', err);
    }
  },

  cancelActiveJob: async () => {
    const job = get().activeJob;
    if (!job?.jobId) return;
    if (get().pollTimer) {
      clearInterval(get().pollTimer);
      set({ pollTimer: null });
    }
    if (get().sseUnsubscribe) {
      get().sseUnsubscribe();
      set({ sseUnsubscribe: null });
    }
    try {
      await cancelJob(job.jobId);
      set({
        jobStatus: 'cancelled',
        jobMessage: 'Job cancelled by user',
      });
    } catch (err) {
      console.warn('Failed to cancel job:', err);
    }
  },

  clearJob: () => {
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
