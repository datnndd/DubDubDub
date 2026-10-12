import { StateCreator } from 'zustand';
import type { ProjectRecord } from '../types/project';
import type { Segment } from '../types/segment';
import { fetchProjects, fetchProject, createProject, deleteProject as apiDeleteProject, bulkDeleteProjects, updateProjectState, resetProjectStage, fetchSubprojects } from '../api/projects';
import { DEFAULT_EDIT_VIDEO, DEFAULT_SUBTITLE_STYLES } from './editVideoSlice';

let pendingProjectSave: Promise<void> = Promise.resolve();

export type MainView = 'projects' | 'dubbing' | 'voices';

export interface ProjectSlice {
  activeProjectId: string | null;
  activeView: MainView;
  drawerOpen: boolean;
  projectsList: ProjectRecord[];
  currentStep: number;
  maxUnlockedStep: number;
  forceAsr: boolean;
  stage3Baseline: Segment[] | null;
  resettingStage: boolean;
  activeParentProjectId: string | null;
  activeParentProjectName: string | null;
  sessionSubprojects: ProjectRecord[];
  sessionDrawerOpen: boolean;
  scanModalOpen: boolean;
  batchModeActive: boolean;

  setActiveView: (view: MainView) => void;
  setStep: (step: number) => Promise<void>;
  setDrawerOpen: (open: boolean) => void;
  setSessionDrawerOpen: (open: boolean) => void;
  setScanModalOpen: (open: boolean) => void;
  setBatchModeActive: (active: boolean) => void;
  setSessionSubprojects: (projects: ProjectRecord[], parentId?: string, parentName?: string) => void;
  loadSubprojects: (parentId: string) => Promise<void>;
  switchSessionVideo: (projectId: string) => Promise<void>;
  loadProjects: () => Promise<void>;
  selectProject: (id: string, autoNavigate?: boolean) => Promise<void>;
  createNewProject: (name?: string, mediaId?: string, duration?: number, parentId?: string) => Promise<string>;
  openDateParentSession: (parentId: string, parentName?: string) => Promise<void>;
  deleteProjectById: (id: string) => Promise<void>;
  deleteProjectsByIds: (ids: string[]) => Promise<void>;
  triggerAutosave: (reportFailure?: boolean) => Promise<void>;
  resetStage: (stage: number) => Promise<void>;
}

export function normalizeLanguages(incoming: any, fallback?: any) {
  const defaultFallback = {
    source: { code: 'zh-cn', name: 'Simplified Chinese', flag: '', autoDetected: false },
    target: { code: 'vi', name: 'Vietnamese' },
    timingMode: 'voice',
  };
  const base = fallback || defaultFallback;
  if (!incoming || typeof incoming !== 'object') {
    return {
      source: {
        code: base?.source?.code || 'zh-cn',
        name: base?.source?.name || 'Simplified Chinese',
        flag: base?.source?.flag || '',
        autoDetected: Boolean(base?.source?.autoDetected),
      },
      target: {
        code: base?.target?.code || 'vi',
        name: base?.target?.name || 'Vietnamese',
      },
      timingMode: base?.timingMode || 'voice',
    };
  }

  let sourceCode = base?.source?.code || 'zh-cn';
  let sourceName = base?.source?.name || 'Simplified Chinese';
  let sourceFlag = base?.source?.flag || '';
  let autoDetected = Boolean(base?.source?.autoDetected);

  if (typeof incoming.source === 'string') {
    sourceCode = incoming.source;
    sourceName = incoming.source;
  } else if (incoming.source && typeof incoming.source === 'object') {
    sourceCode = incoming.source.code || incoming.source_code || sourceCode;
    sourceName = incoming.source.name || sourceName;
    sourceFlag = incoming.source.flag || '';
    autoDetected = Boolean(incoming.source.autoDetected);
  } else if (incoming.source_code) {
    sourceCode = incoming.source_code;
  }

  let targetCode = base?.target?.code || 'vi';
  let targetName = base?.target?.name || 'Vietnamese';

  if (typeof incoming.target === 'string') {
    targetCode = incoming.target;
    targetName = incoming.target;
  } else if (incoming.target && typeof incoming.target === 'object') {
    targetCode = incoming.target.code || incoming.target_code || targetCode;
    targetName = incoming.target.name || targetName;
  } else if (incoming.target_code) {
    targetCode = incoming.target_code;
  }

  return {
    source: {
      code: sourceCode,
      name: sourceName,
      flag: sourceFlag,
      autoDetected,
    },
    target: {
      code: targetCode,
      name: targetName,
    },
    timingMode: incoming.timingMode || base?.timingMode || 'voice',
  };
}

export const createProjectSlice: StateCreator<any, [], [], ProjectSlice> = (set, get) => ({
  activeProjectId: typeof localStorage !== 'undefined' ? localStorage.getItem('dubdub_active_project_id') : null,
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

  setActiveView: (view: MainView) => set({ activeView: view }),

  setStep: (step: number) => {
    if (step < 1 || step > 4) return Promise.resolve();
    set((state: any) => ({
      currentStep: step,
      maxUnlockedStep: Math.max(state.maxUnlockedStep || 1, step),
    }));
    return get().triggerAutosave();
  },

  setDrawerOpen: (open: boolean) => set({ drawerOpen: open }),
  setSessionDrawerOpen: (open: boolean) => set({ sessionDrawerOpen: open }),
  setScanModalOpen: (open: boolean) => set({ scanModalOpen: open }),
  setBatchModeActive: (active: boolean) => set({ batchModeActive: active }),
  setSessionSubprojects: (projects: ProjectRecord[], parentId?: string, parentName?: string) => {
    set((state: any) => ({
      sessionSubprojects: projects,
      activeParentProjectId: parentId !== undefined ? parentId : state.activeParentProjectId,
      activeParentProjectName: parentName !== undefined ? parentName : state.activeParentProjectName,
      batchModeActive: projects.length > 0,
    }));
  },
  loadSubprojects: async (parentId: string) => {
    try {
      const subs = await fetchSubprojects(parentId);
      set({ sessionSubprojects: subs, activeParentProjectId: parentId });
    } catch (e) {
      console.warn('Failed to load subprojects:', e);
    }
  },
  switchSessionVideo: async (projectId: string) => {
    if (get().isCreateVoiceModalOpen) {
      set({ isCreateVoiceModalOpen: false });
    }
    await get().triggerAutosave();
    await get().selectProject(projectId, false);
    const pStatuses = get().projectJobStatuses || {};
    if (pStatuses[projectId]) {
      const ps = pStatuses[projectId];
      set({
        jobStatus: ps.status || 'idle',
        jobProgress: ps.progress ?? null,
        jobMessage: ps.message || '',
        jobStage: ps.stage || null,
      });
    } else {
      set({
        jobStatus: 'idle',
        jobProgress: null,
        jobMessage: 'Ready to process',
        jobStage: null,
      });
    }
  },

  loadProjects: async () => {
    try {
      const list = await fetchProjects();
      set({ projectsList: list });
    } catch (err) {
      console.warn('Failed to load projects list:', err);
    }
  },

  selectProject: async (id: string, autoNavigate: boolean = true) => {
    try {
      const project = await fetchProject(id);
      if (!project) {
        if (typeof localStorage !== 'undefined') {
          localStorage.removeItem('dubdub_active_project_id');
        }
        set((state: any) => ({ ...state, activeProjectId: null }));
        return;
      }
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('dubdub_active_project_id', id);
      }
      let stateData: any = (project as any).state || null;
      if (!stateData && project.state_json) {
        try {
          stateData = typeof project.state_json === 'string' ? JSON.parse(project.state_json) : project.state_json;
        } catch (e) {
          console.warn('Failed to parse project.state_json:', e);
          stateData = null;
        }
      }

      const projId = String(project?.id || id || '');
      if (stateData?.is_parent || (projId && projId.startsWith('parent_'))) {
        set({ activeParentProjectId: projId, activeParentProjectName: project?.name || projId });
        try {
          const subs = await fetchSubprojects(projId);
          set({ sessionSubprojects: subs, batchModeActive: subs.length > 0 });
          if (subs.length > 0) {
            return await get().selectProject(subs[0].id, autoNavigate);
          }
        } catch (_) {}
      }

      const targetStep = stateData?.currentStep || project.stage || 1;
      const mediaId = (project as any).media_id || stateData?.backend?.mediaId || null;
      let resolvedPreviewUrl: string | undefined = undefined;
      if (mediaId) {
        const candidateUrl = stateData?.project?.previewUrl;
        resolvedPreviewUrl = (candidateUrl && !candidateUrl.startsWith('blob:'))
          ? candidateUrl
          : `/api/media/${mediaId}/file`;
      }
      const isVerified = Boolean(mediaId && (stateData?.project?.verified ?? true));

      const rawSegments = Array.isArray(stateData?.segments) ? stateData.segments : [];
      const normalizedSegments = rawSegments.map((seg: any, idx: number) => ({
        ...seg,
        id: seg?.id ?? idx + 1,
        sourceText: String(seg?.sourceText ?? seg?.text ?? ''),
        targetText: String(seg?.targetText ?? ''),
        speakerId: String(seg?.speakerId ?? seg?.speakerLabel ?? seg?.speaker ?? 'spk_1'),
        speakerName: String(seg?.speakerName ?? (seg?.speakerLabel ? `Speaker ${seg.speakerLabel}` : 'Speaker 1')),
      }));

      const rawSpeakers = Array.isArray(stateData?.speakers) ? stateData.speakers : [];
      const normalizedSpeakers = rawSpeakers.map((spk: any, idx: number) => ({
        ...spk,
        id: String(spk?.id ?? `spk_${idx + 1}`),
        name: String(spk?.name ?? `Speaker ${idx + 1}`),
        code: String(spk?.code || spk?.id || `S${idx + 1}`).slice(0, 2).toUpperCase(),
      }));

      const rawOverrides = stateData?.segmentVoiceOverrides || {};
      const mergedOverrides: Record<number, string> = { ...rawOverrides };
      normalizedSegments.forEach((seg: any) => {
        if (seg.id && seg.voiceOverride && !mergedOverrides[seg.id]) {
          mergedOverrides[seg.id] = seg.voiceOverride;
        }
      });

      const pStatuses = get().projectJobStatuses || {};
      const jobForThisProj = pStatuses[id];
      const isVoiceRunning = jobForThisProj?.stage === 'voice' && jobForThisProj?.status === 'running';
      const isVoiceCompleted = jobForThisProj?.stage === 'voice' && jobForThisProj?.status === 'completed';
      const resolvedDubbingStatus = isVoiceRunning ? 'running' : isVoiceCompleted ? 'completed' : (stateData?.dubbingStatus || 'idle');

      set((state: any) => ({
        ...state,
        selectedFile: null,
        activeProjectId: id,
        ...(autoNavigate ? { activeView: 'dubbing' as MainView } : {}),
        currentStep: targetStep,
        maxUnlockedStep: Math.max(project.stage || 1, targetStep),
        drawerOpen: false,
        segments: normalizedSegments,
        ocrCrop: { active: false, segmentId: null, roi: stateData?.ocrCrop?.roi || [0.05, 0.75, 0.9, 0.2], loading: false, error: null },
        transcriptOptions: stateData?.transcriptOptions || null,
        selectedSegmentOption: stateData?.selectedSegmentOption || 'utterances',
        speakers: normalizedSpeakers,
        speakerVoiceMap: stateData?.speakerVoiceMap || {},
        segmentVoiceOverrides: mergedOverrides,
        dubbingStatus: resolvedDubbingStatus,
        dubbingError: null,
        assemblingDubbing: false,
        assembledDubUrl: targetStep >= 4 ? `/api/projects/${encodeURIComponent(id)}/dubbing/audio` : null,
        tuning: stateData?.tuning || { pace: 1.0, timbreWarmth: 62, ducking: '85/15' },
        autoFitVoiceSpeed: stateData?.autoFitVoiceSpeed ?? true,
        maxSpeedRate: stateData?.maxSpeedRate ?? 1.35,
        forceAsr: Boolean(stateData?.forceAsr),
        stage3Baseline: stateData?.stage3Baseline || null,
        subtitleStyles: stateData?.subtitleStyles || { ...DEFAULT_SUBTITLE_STYLES },
        editVideo: stateData?.editVideo ? { ...DEFAULT_EDIT_VIDEO, ...stateData.editVideo } : { ...DEFAULT_EDIT_VIDEO },
        jobStatus: 'idle',
        jobProgress: null,
        jobStage: null,
        jobMessage: 'Ready to process',
        backend: {
          ...state.backend,
          mediaId: mediaId,
          status: mediaId ? 'ready' : 'idle',
          error: null,
          message: '',
          config: {
            ...state.backend?.config,
            ...(stateData?.backend?.config || {}),
          },
        },
        languages: normalizeLanguages(stateData?.languages, state.languages),
        engines: {
          speakerDiarization: Boolean(stateData?.engines?.speakerDiarization),
          speakerCount: Number(stateData?.engines?.speakerCount || 0),
          ocrSlideEngine: stateData?.engines?.ocrSlideEngine ?? true,
        },
        project: {
          filename: project.name || 'Untitled Video Project',
          format: '—',
          resolution: '—',
          fps: '—',
          duration: '00:00.000',
          durationSec: 0,
          fileSize: '—',
          videoCodec: '—',
          audioCodec: '—',
          bitrate: 'Not reported',
          hasAudio: isVerified,
          hasVideo: isVerified,
          lastSaved: 'Just now',
          ...(stateData?.project || {}),
          previewUrl: resolvedPreviewUrl,
          verified: isVerified,
        },
      }));

      const parentId = project?.parent_id || project?.parentId;
      if (parentId) {
        set({ activeParentProjectId: parentId });
        try {
          const subs = await fetchSubprojects(parentId);
          set({ sessionSubprojects: subs, batchModeActive: subs.length > 0 });
        } catch (_) {}
      }
    } catch (err) {
      console.error('Failed to select project:', err);
      if (typeof localStorage !== 'undefined') {
        localStorage.removeItem('dubdub_active_project_id');
      }
      set((state: any) => ({ ...state, activeProjectId: null }));
    }
  },

  createNewProject: async (name: string = 'Untitled Video Project', mediaId?: string, duration?: number, parentId?: string) => {
    try {
      const today = new Date().toISOString().split('T')[0];
      const effectiveParentId = parentId || get().activeParentProjectId || `parent_${today}`;
      const proj = await createProject({ name, mediaId, duration, parentId: effectiveParentId });
      const newId = proj?.id;
      if (newId) {
        await get().selectProject(newId, true);
      }
      set((state: any) => ({
        ...state,
        activeView: 'dubbing' as MainView,
        currentStep: 1,
      }));
      await get().loadProjects();
      if (effectiveParentId) {
        try {
          const subs = await fetchSubprojects(effectiveParentId);
          set({ sessionSubprojects: subs, activeParentProjectId: effectiveParentId, batchModeActive: subs.length > 0 });
        } catch (_) {}
      }
      return newId || '';
    } catch (err) {
      console.error('Failed to create new project:', err);
      return '';
    }
  },

  openDateParentSession: async (parentId: string, parentName?: string) => {
    try {
      const subs = await fetchSubprojects(parentId);
      set({
        activeParentProjectId: parentId,
        activeParentProjectName: parentName || parentId,
        sessionSubprojects: subs,
        batchModeActive: subs.length > 0,
        activeView: 'dubbing' as MainView,
        currentStep: 1,
        sessionDrawerOpen: true,
      });
      if (subs.length > 0) {
        await get().selectProject(subs[0].id, false);
      }
    } catch (err) {
      console.error('Failed to open date parent session:', err);
    }
  },

  deleteProjectById: async (id: string) => {
    try {
      await apiDeleteProject(id);
      if (get().activeProjectId === id) {
        set({ activeProjectId: null, activeView: 'projects' as MainView });
        if (typeof localStorage !== 'undefined') {
          localStorage.removeItem('dubdub_active_project_id');
        }
      }
      await get().loadProjects();
    } catch (err) {
      console.error('Failed to delete project:', err);
    }
  },

  deleteProjectsByIds: async (ids: string[]) => {
    if (!ids || ids.length === 0) return;
    try {
      await bulkDeleteProjects(ids);
      const activeId = get().activeProjectId;
      if (activeId && ids.includes(activeId)) {
        set({ activeProjectId: null, activeView: 'projects' as MainView });
        if (typeof localStorage !== 'undefined') {
          localStorage.removeItem('dubdub_active_project_id');
        }
      }
      await get().loadProjects();
    } catch (err) {
      console.error('Failed to bulk delete projects:', err);
      for (const id of ids) {
        try {
          await apiDeleteProject(id);
        } catch (_) {}
      }
      await get().loadProjects();
    }
  },

  triggerAutosave: (reportFailure = false) => {
    const id = get().activeProjectId;
    if (!id || get().resettingStage) return Promise.resolve();
    const currentState = get();
    const mediaId = currentState.backend?.mediaId || null;
    // Exclude transient file objects and dead blob URLs from autosave JSON
    const snapshot = {
      backend: {
        mediaId,
        config: currentState.backend?.config,
      },
      project: {
        ...currentState.project,
        previewUrl: mediaId ? `/api/media/${mediaId}/file` : undefined,
      },
      languages: currentState.languages,
      engines: {
        speakerDiarization: currentState.engines?.speakerDiarization,
        speakerCount: currentState.engines?.speakerCount,
        ocrSlideEngine: currentState.engines?.ocrSlideEngine,
      },
      forceAsr: currentState.forceAsr,
      ocrCrop: { roi: currentState.ocrCrop?.roi },
      stage3Baseline: currentState.stage3Baseline,
      autoFitVoiceSpeed: currentState.autoFitVoiceSpeed,
      maxSpeedRate: currentState.maxSpeedRate,
      speakers: currentState.speakers,
      speakerVoiceMap: currentState.speakerVoiceMap,
      segmentVoiceOverrides: currentState.segmentVoiceOverrides,
      dubbingStatus: currentState.dubbingStatus,
      tuning: currentState.tuning,
      segments: currentState.segments,
      transcriptOptions: currentState.transcriptOptions,
      selectedSegmentOption: currentState.selectedSegmentOption,
      subtitleStyles: currentState.subtitleStyles,
      editVideo: {
        audioMix: currentState.editVideo?.audioMix,
        activeTab: currentState.editVideo?.activeTab,
      },
      currentStep: currentState.currentStep,
    };
    const save = pendingProjectSave.then(() => updateProjectState(id, snapshot, currentState.currentStep, mediaId || undefined))
      .then(() => {
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        set((s: any) => ({
          project: {
            ...s.project,
            lastSaved: timeStr,
          },
        }));
      });
    pendingProjectSave = save.catch((err) => {
        console.warn('Autosave failed:', err);
      });
    return reportFailure ? save : pendingProjectSave;
  },

  resetStage: async (stage: number) => {
    const id = get().activeProjectId;
    if (!id || stage < 1 || stage > 4) return;
    set({ resettingStage: true });
    try {
      await pendingProjectSave;
      await resetProjectStage(id, stage);
      await get().selectProject(id, false);
      get().clearJob();
      await get().loadProjects();
    } finally {
      set({ resettingStage: false });
    }
  },
});
