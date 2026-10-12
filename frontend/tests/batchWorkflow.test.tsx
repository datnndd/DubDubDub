import { beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';

import { useDubDubStore } from '../src/store';
import { FolderScanModal } from '../src/components/FolderScanModal';
import { SessionVideosDrawer } from '../src/components/SessionVideosDrawer';
import { Stage1Prepare } from '../src/screens/Stage1Prepare';
import { StatusFooter } from '../src/components/StatusFooter';
import { ProjectsScreen } from '../src/screens/ProjectsScreen';
import { Header } from '../src/components/Header';
import { Stage3VoiceDubbing } from '../src/screens/Stage3VoiceDubbing';

describe('Batch Video Workflow & Multi-Video Session', () => {
  beforeEach(() => {
    useDubDubStore.setState({
      currentStep: 1,
      maxUnlockedStep: 1,
      activeProjectId: 'proj_vid_1',
      activeParentProjectId: 'proj_parent_2026_10_09',
      activeParentProjectName: '2026-10-09',
      scanModalOpen: false,
      sessionDrawerOpen: false,
      batchModeActive: false,
      sessionSubprojects: [],
      projectJobStatuses: {},
      project: {
        id: 'proj_vid_1',
        filename: 'video1.mp4',
        duration: '00:30',
        durationSec: 30,
        previewUrl: '/api/media/med_1/file',
        verified: true,
      },
    });
  });

  test('FolderScanModal does not render when scanModalOpen is false', () => {
    useDubDubStore.setState({ scanModalOpen: false });
    const html = renderToStaticMarkup(<FolderScanModal />);
    expect(html).toBe('');
  });

  test('FolderScanModal renders scan input and options when open', () => {
    useDubDubStore.setState({ scanModalOpen: true });
    const html = renderToStaticMarkup(<FolderScanModal />);
    expect(html).toContain('Scan &amp; Batch Import Videos');
    expect(html).toContain('Scan subfolders recursively');
    expect(html).toContain('Scan Folders');
    expect(html).toContain('Folder Paths to Scan (Host Machine)');
  });

  test('SessionVideosDrawer renders video list and badges when open', () => {
    useDubDubStore.setState({
      sessionDrawerOpen: true,
      activeProjectId: 'proj_vid_1',
      activeParentProjectName: '2026-10-09',
      sessionSubprojects: [
        {
          id: 'proj_vid_1',
          name: 'clip_intro.mp4',
          stage: 1,
          status: 'idle',
          media_id: 'med_1',
          created_at: 1728450000,
          updated_at: 1728450000,
        },
        {
          id: 'proj_vid_2',
          name: 'clip_outro.mp4',
          stage: 2,
          status: 'completed',
          media_id: 'med_2',
          created_at: 1728450010,
          updated_at: 1728450010,
        },
      ],
      projectJobStatuses: {
        proj_vid_2: {
          jobId: 'job_2',
          status: 'completed',
          progress: 100,
          stage: 'completed',
          message: 'Ready for Review',
        },
      },
    });

    const html = renderToStaticMarkup(<SessionVideosDrawer />);
    expect(html).toContain('Session: 2026-10-09');
    expect(html).toContain('videos ready for review');
    expect(html).toContain('clip_intro.mp4');
    expect(html).toContain('clip_outro.mp4');
    expect(html).toContain('Active');
    expect(html).toContain('1. Prepare &amp; ASR');
    expect(html).toContain('2. Review Transcript');
    expect(html).toContain('Ready for Review');
  });

  test('Stage1Prepare displays Batch Mode Banner when sessionSubprojects exist', () => {
    useDubDubStore.setState({
      sessionSubprojects: [
        {
          id: 'proj_1',
          name: 'lesson_01.mp4',
          stage: 1,
          status: 'idle',
          media_id: 'm1',
          created_at: 1,
          updated_at: 1,
        },
        {
          id: 'proj_2',
          name: 'lesson_02.mp4',
          stage: 1,
          status: 'idle',
          media_id: 'm2',
          created_at: 2,
          updated_at: 2,
        },
      ],
      activeParentProjectName: '2026-10-09',
    });

    const html = renderToStaticMarkup(<Stage1Prepare />);
    expect(html).toContain('Batch Mode Active:');
    expect(html).toContain('2 videos loaded in Parent Project');
    expect(html).toContain('[2026-10-09]');
    expect(html).toContain('Session Videos (2)');
    expect(html).toContain('Scan More');
  });

  test('StatusFooter shows Start Batch Dub button when multiple session subprojects are present', () => {
    useDubDubStore.setState({
      currentStep: 1,
      sessionSubprojects: [
        {
          id: 'sub_1',
          name: 'video_a.mp4',
          stage: 1,
          status: 'idle',
          media_id: 'm_a',
          created_at: 1,
          updated_at: 1,
        },
        {
          id: 'sub_2',
          name: 'video_b.mp4',
          stage: 1,
          status: 'idle',
          media_id: 'm_b',
          created_at: 2,
          updated_at: 2,
        },
        {
          id: 'sub_3',
          name: 'video_c.mp4',
          stage: 1,
          status: 'idle',
          media_id: 'm_c',
          created_at: 3,
          updated_at: 3,
        },
      ],
    });

    const html = renderToStaticMarkup(<StatusFooter />);
    expect(html).toContain('Start Batch Dub (3 Videos)');
  });

  test('switchSessionVideo updates active project without resetting running background jobs', async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = ((url: string) => {
        if (url.includes('/api/projects/sub_2')) {
          return new Response(
            JSON.stringify({
              id: 'sub_2',
              name: 'video_b.mp4',
              stage: 2,
              media_id: 'm_b',
              state: {
                currentStep: 2,
                project: { filename: 'video_b.mp4', verified: true },
                segments: [{ id: 1, startSec: 0, endSec: 2, sourceText: 'Hello', targetText: 'Xin chào' }],
              },
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          );
        }
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }) as any;

      useDubDubStore.setState({
        activeProjectId: 'sub_1',
        projectJobStatuses: {
          sub_1: { jobId: 'job_1', status: 'running', progress: 45, stage: 'transcribing', message: 'Working on sub_1' },
          sub_2: { jobId: 'job_2', status: 'completed', progress: 100, stage: 'completed', message: 'Done' },
        },
      });

      await useDubDubStore.getState().switchSessionVideo('sub_2');

      const state = useDubDubStore.getState();
      expect(state.activeProjectId).toBe('sub_2');
      expect(state.jobStatus).toBe('completed');
      expect(state.jobProgress).toBe(100);
      // Background job status for sub_1 is preserved in projectJobStatuses
      expect(state.projectJobStatuses['sub_1'].status).toBe('running');
      expect(state.projectJobStatuses['sub_1'].progress).toBe(45);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('ProjectsScreen renders Date Parent Project containers with subprojects grouped inside', () => {
    const today = new Date().toISOString().split('T')[0];
    useDubDubStore.setState({
      projectsList: [
        {
          id: `parent_${today}`,
          name: today,
          stage: 1,
          status: 'active',
          state: { is_parent: true, date: today },
        },
        {
          id: 'sub_video_1',
          name: 'Morning Lecture.mp4',
          parent_id: `parent_${today}`,
          stage: 2,
          status: 'completed',
          duration: 180.5,
        },
        {
          id: 'sub_video_2',
          name: 'Afternoon Lab.mp4',
          parent_id: `parent_${today}`,
          stage: 1,
          status: 'processing',
          duration: 95.0,
        },
      ],
    });

    const html = renderToStaticMarkup(<ProjectsScreen />);
    // Check parent project date container and badges
    expect(html).toContain(`data-date-group="${today}"`);
    expect(html).toContain(today);
    expect(html).toContain('Today');
    expect(html).toContain('2 Videos');
    expect(html).toContain('1 Ready');
    expect(html).toContain(`open-session-${today}`);

    // Check subproject video cards rendered inside
    expect(html).toContain('Morning Lecture.mp4');
    expect(html).toContain('Afternoon Lab.mp4');
    expect(html).toContain('Stage 2: Review Transcript');
    expect(html).toContain('Stage 1: Prepare &amp; ASR');
    expect(html).toContain('data-testid="toggle-group-by-date"');
    expect(html).toContain('Grouped by Day');
  });

  test('openDateParentSession loads subprojects into sessionSubprojects and transitions to dubbing view', async () => {
    const originalFetch = globalThis.fetch;
    const today = new Date().toISOString().split('T')[0];
    const parentId = `parent_${today}`;
    const subprojects = [
      { id: 'sub_1', name: 'clip_a.mp4', parent_id: parentId, stage: 1, status: 'pending' },
      { id: 'sub_2', name: 'clip_b.mp4', parent_id: parentId, stage: 2, status: 'completed' },
    ];

    globalThis.fetch = (async (url: string | URL | Request) => {
      const urlStr = String(url);
      if (urlStr.includes('/subprojects')) {
        return new Response(JSON.stringify({ subprojects }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      if (urlStr.includes('/api/projects/sub_1')) {
        return new Response(JSON.stringify(subprojects[0]), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return new Response(JSON.stringify({ projects: [] }), { status: 200 });
    }) as any;

    try {
      useDubDubStore.setState({ activeView: 'projects', sessionSubprojects: [], activeParentProjectId: null });
      await useDubDubStore.getState().openDateParentSession(parentId, today);

      const state = useDubDubStore.getState();
      expect(state.activeView).toBe('dubbing');
      expect(state.currentStep).toBe(1);
      expect(state.activeParentProjectId).toBe(parentId);
      expect(state.activeParentProjectName).toBe(today);
      expect(state.sessionSubprojects.length).toBe(2);
      expect(state.sessionDrawerOpen).toBe(true);
      expect(state.batchModeActive).toBe(true);
      expect(state.activeProjectId).toBe('sub_1');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('Navbar Session Bar in Header displays all session videos with running and completed voice badges', () => {
    useDubDubStore.setState({
      activeProjectId: 'sub_1',
      sessionSubprojects: [
        {
          id: 'sub_1',
          name: 'intro_lesson.mp4',
          stage: 3,
          status: 'idle',
        },
        {
          id: 'sub_2',
          name: 'deep_dive.mp4',
          stage: 3,
          status: 'idle',
        },
        {
          id: 'sub_3',
          name: 'summary.mp4',
          stage: 2,
          status: 'idle',
        },
      ],
      projectJobStatuses: {
        sub_1: {
          status: 'running',
          stage: 'voice',
          progress: 50,
          message: 'Creating voice…',
        },
        sub_2: {
          status: 'completed',
          stage: 'voice',
          progress: 100,
          message: 'Voice completed',
        },
      },
    });

    const html = renderToStaticMarkup(<Header />);
    expect(html).toContain('data-testid="navbar-session-bar"');
    expect(html).toContain('data-testid="session-bar-item-sub_1"');
    expect(html).toContain('data-testid="session-bar-item-sub_2"');
    expect(html).toContain('data-testid="session-bar-item-sub_3"');
    expect(html).toContain('intro_lesson.mp4');
    expect(html).toContain('deep_dive.mp4');
    expect(html).toContain('summary.mp4');

    // Video 1 shows running voice status badge
    expect(html).toContain('data-testid="session-bar-badge-sub_1"');
    expect(html).toContain('50%');

    // Video 2 shows completed voice status badge
    expect(html).toContain('data-testid="session-bar-badge-sub_2"');
    expect(html).toContain('Voice Ready');

    // Video 3 shows ready for review status
    expect(html).toContain('data-testid="session-bar-badge-sub_3"');
    expect(html).toContain('Ready');
  });

  test('switchSessionVideo allows switching to another video while voice dubbing is running without disrupting background voice state', async () => {
    const originalFetch = globalThis.fetch;
    const subprojects = [
      { id: 'sub_1', name: 'video_1.mp4', stage: 3, status: 'processing', state: { dubbingStatus: 'running' } },
      { id: 'sub_2', name: 'video_2.mp4', stage: 2, status: 'completed', state: { dubbingStatus: 'idle', currentStep: 2 } },
    ];

    globalThis.fetch = (async (url: string | URL | Request) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/projects/sub_2')) {
        return new Response(JSON.stringify({ project: subprojects[1] }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      if (urlStr.includes('/api/projects/sub_1')) {
        return new Response(JSON.stringify({ project: subprojects[0] }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'sub_1',
        dubbingStatus: 'running',
        sessionSubprojects: subprojects as any,
        projectJobStatuses: {
          sub_1: {
            status: 'running',
            stage: 'voice',
            progress: 30,
            message: 'Creating voice (1/3)…',
          },
        },
      });

      // User clicks video 2 in Navbar Session Bar
      await useDubDubStore.getState().switchSessionVideo('sub_2');

      const state = useDubDubStore.getState();
      // Active project successfully switches to sub_2
      expect(state.activeProjectId).toBe('sub_2');
      // dubbingStatus on sub_2 is idle
      expect(state.dubbingStatus).toBe('idle');
      // Background voice job on sub_1 continues running in projectJobStatuses
      expect(state.projectJobStatuses['sub_1'].status).toBe('running');
      expect(state.projectJobStatuses['sub_1'].stage).toBe('voice');
      expect(state.projectJobStatuses['sub_1'].progress).toBe(30);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('runFullDubbing updates projectJobStatuses and isolates background updates when user switches project', async () => {
    const originalFetch = globalThis.fetch;
    let savedBackgroundPayload: any = null;

    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      const urlStr = String(url);
      if (urlStr.includes('/api/projects/sub_1') && init?.method === 'PUT') {
        savedBackgroundPayload = JSON.parse(String(init?.body || '{}'));
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }
      if (urlStr.includes('/api/projects/sub_1')) {
        return new Response(JSON.stringify({ project: { id: 'sub_1', stage: 3, state: { segments: [] } } }), { status: 200 });
      }
      if (urlStr.includes('/api/tts/preview') || urlStr.includes('/api/voices/preview')) {
        return new Response(
          JSON.stringify({
            audio_url: '/audio/preview_1.mp3',
            id: 'prev_1',
            applied_speed: 1.0,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }) as any;

    try {
      useDubDubStore.setState({
        activeProjectId: 'sub_1',
        segments: [
          { id: 1, sourceText: 'Hello', targetText: 'Xin chào', startSec: 0, endSec: 2 },
        ] as any,
        sessionSubprojects: [
          { id: 'sub_1', name: 'v1.mp4', stage: 3, status: 'idle' },
          { id: 'sub_2', name: 'v2.mp4', stage: 2, status: 'idle' },
        ] as any,
        projectJobStatuses: {},
      });

      // Start runFullDubbing on sub_1
      const dubbingPromise = useDubDubStore.getState().runFullDubbing();

      // Verify that sub_1 immediately registered in projectJobStatuses
      expect(useDubDubStore.getState().projectJobStatuses['sub_1']?.status).toBe('running');
      expect(useDubDubStore.getState().projectJobStatuses['sub_1']?.stage).toBe('voice');

      // User immediately switches to sub_2 before dubbing finishes
      useDubDubStore.setState({
        activeProjectId: 'sub_2',
        segments: [
          { id: 10, sourceText: 'Video 2 segment', targetText: 'Đoạn video 2', startSec: 0, endSec: 3 },
        ] as any,
      });

      // Wait for background dubbing of sub_1 to complete
      await dubbingPromise;

      const state = useDubDubStore.getState();
      // Active video (sub_2) segments were NOT clobbered by sub_1's dubbing
      expect(state.activeProjectId).toBe('sub_2');
      expect(state.segments[0].id).toBe(10);
      expect(state.segments[0].targetText).toBe('Đoạn video 2');

      // sub_1 in projectJobStatuses is marked as completed
      expect(state.projectJobStatuses['sub_1']?.status).toBe('completed');
      expect(state.projectJobStatuses['sub_1']?.stage).toBe('voice');

      // sub_1's state was saved to backend
      expect(savedBackgroundPayload).not.toBeNull();
      expect(savedBackgroundPayload.state?.dubbingStatus).toBe('completed');
      expect(savedBackgroundPayload.state?.segments[0].previewAudioUrl).toBe('/audio/preview_1.mp3');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('Stage3VoiceDubbing renders dubbing-running-banner with progress and message when dubbingStatus is running', () => {
    useDubDubStore.setState({
      currentStep: 3,
      dubbingStatus: 'running',
      jobProgress: 42.5,
      jobMessage: 'Creating voice (5/12)…',
      segments: [
        { id: 1, sourceText: 'Hello', targetText: 'Xin chào', startSec: 0, endSec: 2 },
      ] as any,
    });

    const html = renderToStaticMarkup(<Stage3VoiceDubbing />);
    expect(html).toContain('data-testid="dubbing-running-banner"');
    expect(html).toContain('Synthesizing Voice Dubbing in Background');
    expect(html).toContain('43%');
    expect(html).toContain('Creating voice (5/12)');
  });
});

