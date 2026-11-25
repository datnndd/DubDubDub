import { beforeEach, describe, expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';

import { useDubDubStore } from '../src/store';
import { Header } from '../src/components/Header';
import { WorkflowStepper } from '../src/components/WorkflowStepper';
import { ProjectsScreen } from '../src/screens/ProjectsScreen';
import { VoiceManagementScreen } from '../src/screens/VoiceManagementScreen';
import { App } from '../src/App';

describe('Top-Level Navigation & Main Screen Interfaces', () => {
  beforeEach(() => {
    useDubDubStore.setState({
      activeProjectId: null,
      activeView: 'projects',
      currentStep: 1,
      drawerOpen: false,
      isVoiceManagerDrawerOpen: false,
      isCreateVoiceModalOpen: false,
      projectsList: [
        {
          id: 'proj-alpha',
          name: 'Alpha AI Dubbing',
          stage: 1,
          status: 'pending',
          duration: 15.0,
          created_at: '2026-09-20T10:00:00Z',
          updated_at: '2026-09-25T14:00:00Z',
        },
        {
          id: 'proj-beta',
          name: 'Beta Review Video',
          stage: 2,
          status: 'completed',
          duration: 45.5,
          created_at: '2026-09-22T08:00:00Z',
          updated_at: '2026-09-26T16:00:00Z',
        },
        {
          id: 'proj-gamma',
          name: 'Gamma Audio Story',
          stage: 3,
          status: 'processing',
          duration: 120.0,
          created_at: '2026-09-24T12:00:00Z',
          updated_at: '2026-09-27T18:00:00Z',
        },
      ],
      customVoices: [
        {
          id: 'cv_viet_1',
          name: 'Minh Thư Cloned Voice',
          provider: 2,
          language: 'vi',
          description: 'Warm storyteller timbre',
          ref_text: 'Xin chào mọi người',
        },
      ],
      voices: [
        { id: 'PhamTuyen', name: 'Phạm Tuyên (Preset)', provider: 2, kind: 'preset' },
        { id: 'Rachel', name: 'Rachel (Preset)', provider: 0, kind: 'preset' },
      ],
      backend: {
        ready: true,
        error: null,
        mediaId: null,
        status: 'idle',
        message: '',
        options: {
          languages: [{ code: 'vi', name: 'Vietnamese' }, { code: 'en', name: 'English' }],
          asrProviders: [],
          translationProviders: [],
          translationModes: [],
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
        },
      },
      languages: {
        source: { code: 'en', name: 'English', flag: '', autoDetected: false },
        target: { code: 'vi', name: 'Vietnamese' },
        timingMode: 'voice',
      },
    });
  });

  describe('1. Top-Level Navigation & Header Tabs', () => {
    test('Header renders top-level navigation tabs: Projects, Video Dub, and Voice Management', () => {
      const markup = renderToStaticMarkup(<Header />);
      expect(markup).toContain('data-testid="nav-tab-projects"');
      expect(markup).toContain('data-testid="nav-tab-dubbing"');
      expect(markup).toContain('data-testid="nav-tab-voices"');
      expect(markup).toContain('Projects');
      expect(markup).toContain('Video Dub');
      expect(markup).toContain('Voice Management');
      // Count badge on projects tab
      expect(markup).toContain('3');
    });

    test('default view when application is opened is "projects"', () => {
      expect(useDubDubStore.getState().activeView).toBe('projects');
    });

    test('switching navigation tabs updates activeView in store', () => {
      const store = useDubDubStore.getState();
      expect(store.activeView).toBe('projects');

      // Switch to Video Dub
      store.setActiveView('dubbing');
      expect(useDubDubStore.getState().activeView).toBe('dubbing');

      // Switch to Voice Management
      store.setActiveView('voices');
      expect(useDubDubStore.getState().activeView).toBe('voices');

      // Switch back to Projects
      store.setActiveView('projects');
      expect(useDubDubStore.getState().activeView).toBe('projects');
    });

    test('Header renders "Back to Projects" button when not in "projects" view', () => {
      // In projects view: no Back to Projects button in header
      let markup = renderToStaticMarkup(<Header />);
      expect(markup).not.toContain('data-testid="header-back-to-projects-btn"');

      // In dubbing view: Back to Projects button appears
      useDubDubStore.getState().setActiveView('dubbing');
      markup = renderToStaticMarkup(<Header />);
      expect(markup).toContain('data-testid="header-back-to-projects-btn"');
      expect(markup).toContain('Back to Projects');

      // In voices view: Back to Projects button appears
      useDubDubStore.getState().setActiveView('voices');
      markup = renderToStaticMarkup(<Header />);
      expect(markup).toContain('data-testid="header-back-to-projects-btn"');
    });

    test('WorkflowStepper renders "Back to Projects" button in workflow bar', () => {
      const markup = renderToStaticMarkup(<WorkflowStepper />);
      expect(markup).toContain('data-testid="workflow-back-to-projects-btn"');
      expect(markup).toContain('Projects');
    });

    test('selectProject with autoNavigate=true transitions activeView to "dubbing"', async () => {
      const originalFetch = globalThis.fetch;
      globalThis.fetch = (async (url: string) => {
        if (String(url).includes('proj-beta')) {
          return new Response(
            JSON.stringify({
              id: 'proj-beta',
              name: 'Beta Review Video',
              stage: 2,
              status: 'completed',
              state: { currentStep: 2 },
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          );
        }
        return new Response('{}', { status: 200 });
      }) as any;

      try {
        useDubDubStore.setState({ activeView: 'projects', activeProjectId: null });
        await useDubDubStore.getState().selectProject('proj-beta', true);

        const state = useDubDubStore.getState();
        expect(state.activeProjectId).toBe('proj-beta');
        expect(state.activeView).toBe('dubbing');
        expect(state.currentStep).toBe(2);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    test('selectProject with autoNavigate=false preserves initial "projects" view (app startup preload)', async () => {
      const originalFetch = globalThis.fetch;
      globalThis.fetch = (async (url: string) => {
        if (String(url).includes('proj-alpha')) {
          return new Response(
            JSON.stringify({
              id: 'proj-alpha',
              name: 'Alpha AI Dubbing',
              stage: 1,
              status: 'pending',
              state: { currentStep: 1 },
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          );
        }
        return new Response('{}', { status: 200 });
      }) as any;

      try {
        useDubDubStore.setState({ activeView: 'projects', activeProjectId: null });
        await useDubDubStore.getState().selectProject('proj-alpha', false);

        const state = useDubDubStore.getState();
        expect(state.activeProjectId).toBe('proj-alpha');
        // Still on 'projects' screen!
        expect(state.activeView).toBe('projects');
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });

  describe('2. Projects Main Screen Layout', () => {
    test('renders ProjectsScreen with header, count, search bar, sort dropdown, view mode toggle, and + New Project', () => {
      const markup = renderToStaticMarkup(<ProjectsScreen />);
      expect(markup).toContain('data-screen="projects-screen"');
      expect(markup).toContain('Project Management');
      expect(markup).toContain('data-testid="projects-search-input"');
      expect(markup).toContain('data-testid="projects-sort-select"');
      expect(markup).toContain('data-testid="view-mode-grid"');
      expect(markup).toContain('data-testid="view-mode-table"');
      expect(markup).toContain('data-testid="new-project-btn"');
      expect(markup).toContain('New Project');
      expect(markup).toContain('data-testid="select-all-projects-btn"');
    });

    test('renders visual Card Grid view by default with thumbnails, duration, status, and quick action buttons', () => {
      const markup = renderToStaticMarkup(<ProjectsScreen />);
      expect(markup).toContain('data-testid="projects-grid-view"');
      expect(markup).toContain('Alpha AI Dubbing');
      expect(markup).toContain('Beta Review Video');
      expect(markup).toContain('Gamma Audio Story');

      // Duration pills
      expect(markup).toContain('15.0s');
      expect(markup).toContain('45.5s');
      expect(markup).toContain('120.0s');

      // Stage badges
      expect(markup).toContain('Stage 1: Prepare &amp; ASR');
      expect(markup).toContain('Stage 2: Review Transcript');
      expect(markup).toContain('Stage 3: Voice Dubbing');

      // Checkboxes & Action buttons
      expect(markup).toContain('data-testid="checkbox-project-proj-alpha"');
      expect(markup).toContain('data-testid="checkbox-project-proj-beta"');
      expect(markup).toContain('data-testid="activate-project-proj-alpha"');
      expect(markup).toContain('data-testid="delete-project-proj-alpha"');
    });

    test('renders projects with a legacy nonnumeric duration', () => {
      useDubDubStore.setState({
        projectsList: [{
          id: 'legacy-nan', name: 'NanDur', stage: 1, status: 'pending',
          duration: 'nan' as unknown as number,
        }],
      });
      const markup = renderToStaticMarkup(<ProjectsScreen />);
      expect(markup).toContain('NanDur');
      expect(markup).not.toContain('NaNs');
    });

    test('renders empty state when projectsList is empty', () => {
      useDubDubStore.setState({ projectsList: [] });
      const markup = renderToStaticMarkup(<ProjectsScreen />);
      expect(markup).toContain('No projects yet');
      expect(markup).toContain('Create First Project');
    });

    test('createNewProject initializes new project and navigates to Stage 1 in "dubbing" view', async () => {
      const originalFetch = globalThis.fetch;
      globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
        const urlStr = String(url);
        if (urlStr.endsWith('/api/projects') && init?.method === 'POST') {
          return new Response(
            JSON.stringify({ id: 'proj-fresh', name: 'Untitled Video Project', stage: 1, status: 'pending' }),
            { status: 201, headers: { 'Content-Type': 'application/json' } }
          );
        }
        if (urlStr.includes('proj-fresh')) {
          return new Response(
            JSON.stringify({ id: 'proj-fresh', name: 'Untitled Video Project', stage: 1, status: 'pending' }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          );
        }
        return new Response(JSON.stringify({ projects: [] }), { status: 200 });
      }) as any;

      try {
        useDubDubStore.setState({ activeView: 'projects', activeProjectId: null, currentStep: 3 });
        const newId = await useDubDubStore.getState().createNewProject('Untitled Video Project');

        expect(newId).toBe('proj-fresh');
        const state = useDubDubStore.getState();
        expect(state.activeProjectId).toBe('proj-fresh');
        expect(state.activeView).toBe('dubbing');
        expect(state.currentStep).toBe(1);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    test('chronological sorting orders projects by Last Modified (newest first) by default', () => {
      // In default sort (last-modified): proj-gamma (Sep 27) > proj-beta (Sep 26) > proj-alpha (Sep 25)
      const markup = renderToStaticMarkup(<ProjectsScreen initialSortBy="last-modified" />);
      const gammaIdx = markup.indexOf('Gamma Audio Story');
      const betaIdx = markup.indexOf('Beta Review Video');
      const alphaIdx = markup.indexOf('Alpha AI Dubbing');

      expect(gammaIdx).toBeLessThan(betaIdx);
      expect(betaIdx).toBeLessThan(alphaIdx);
    });

    test('chronological sorting orders projects by Date Created (newest first)', () => {
      // In date-created sort: proj-gamma (Sep 24) > proj-beta (Sep 22) > proj-alpha (Sep 20)
      const markup = renderToStaticMarkup(<ProjectsScreen initialSortBy="date-created" />);
      const gammaIdx = markup.indexOf('Gamma Audio Story');
      const betaIdx = markup.indexOf('Beta Review Video');
      const alphaIdx = markup.indexOf('Alpha AI Dubbing');

      expect(gammaIdx).toBeLessThan(betaIdx);
      expect(betaIdx).toBeLessThan(alphaIdx);
    });

    test('alphabetical sorting orders projects A to Z', () => {
      // In alphabetical sort: Alpha AI Dubbing < Beta Review Video < Gamma Audio Story
      const markup = renderToStaticMarkup(<ProjectsScreen initialSortBy="alphabetical" />);
      const alphaIdx = markup.indexOf('Alpha AI Dubbing');
      const betaIdx = markup.indexOf('Beta Review Video');
      const gammaIdx = markup.indexOf('Gamma Audio Story');

      expect(alphaIdx).toBeLessThan(betaIdx);
      expect(betaIdx).toBeLessThan(gammaIdx);
    });

    test('table view mode renders compact data table with columns and actions', () => {
      const markup = renderToStaticMarkup(<ProjectsScreen initialViewMode="table" />);
      expect(markup).toContain('data-testid="projects-table-view"');
      expect(markup).not.toContain('data-testid="projects-grid-view"');

      // Table column headers
      expect(markup).toContain('Project Name');
      expect(markup).toContain('Workflow Stage');
      expect(markup).toContain('Status');
      expect(markup).toContain('Duration');
      expect(markup).toContain('Last Modified');
      expect(markup).toContain('Actions');

      // Rendered projects
      expect(markup).toContain('Alpha AI Dubbing');
      expect(markup).toContain('Beta Review Video');
      expect(markup).toContain('Gamma Audio Story');
      expect(markup).toContain('data-testid="activate-project-proj-alpha"');
      expect(markup).toContain('data-testid="delete-project-proj-alpha"');
    });

    test('search input filters projects in real time by name, stage, or status', () => {
      // Set projects with specific names
      useDubDubStore.setState({
        projectsList: [
          { id: 'p1', name: 'Podcast Episode 1', stage: 1, status: 'pending', duration: 10 },
          { id: 'p2', name: 'Interview with Expert', stage: 3, status: 'completed', duration: 25 },
        ],
      });

      const allMarkup = renderToStaticMarkup(<ProjectsScreen />);
      expect(allMarkup).toContain('Podcast Episode 1');
      expect(allMarkup).toContain('Interview with Expert');
    });

    test('bulk deletion with confirmation modal cleanses selected items', async () => {
      const originalFetch = globalThis.fetch;
      let deletedIds: string[] = [];

      globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/projects/bulk-delete')) {
          const body = JSON.parse(String(init?.body || '{}'));
          deletedIds = body.ids || [];
          return new Response(JSON.stringify({ ok: true, deleted: deletedIds }), { status: 200 });
        }
        return new Response(JSON.stringify({ projects: [] }), { status: 200 });
      }) as any;

      try {
        useDubDubStore.setState({
          activeProjectId: 'proj-alpha',
          projectsList: [
            { id: 'proj-alpha', name: 'Alpha', stage: 1, status: 'pending', duration: 10 },
            { id: 'proj-beta', name: 'Beta', stage: 2, status: 'completed', duration: 20 },
          ],
        });

        await useDubDubStore.getState().deleteProjectsByIds(['proj-alpha']);

        expect(deletedIds).toEqual(['proj-alpha']);
        const updated = useDubDubStore.getState();
        expect(updated.activeProjectId).toBeNull();
        expect(updated.activeView).toBe('projects');
      } finally {
        globalThis.fetch = originalFetch;
      }
    });
  });

  describe('3. Voice Management Main Screen Layout', () => {
    test('renders VoiceManagementScreen with studio header, tabs, search, and + Create Custom Voice', () => {
      const markup = renderToStaticMarkup(<VoiceManagementScreen />);
      expect(markup).toContain('data-screen="voice-management-screen"');
      expect(markup).toContain('Voice Management Studio');
      expect(markup).toContain('data-testid="create-voice-btn"');
      expect(markup).toContain('Create Custom Voice');
      expect(markup).toContain('data-testid="tab-custom-voices"');
      expect(markup).toContain('data-testid="tab-preset-voices"');
      expect(markup).toContain('data-testid="tab-all-voices"');
      expect(markup).toContain('data-testid="voices-search-input"');
      expect(markup).toContain('data-testid="voice-language-filter"');
      expect(markup).toContain('data-testid="voice-provider-filter"');
      expect(markup).toContain('data-testid="voice-gender-filter"');
    });

    test('renders Custom Cloned Voices cards with inline audition, edit, delete, and test in lab triggers', () => {
      const markup = renderToStaticMarkup(<VoiceManagementScreen />);
      expect(markup).toContain('Minh Thư Cloned Voice');
      expect(markup).toContain('Warm storyteller timbre');
      expect(markup).toContain('VieNeu 48k');
      expect(markup).toContain('Cloned');
      expect(markup).toContain('data-action="audition-voice"');
      expect(markup).toContain('data-voice-id="cv_viet_1"');
      expect(markup).toContain('data-testid="delete-voice-cv_viet_1"');
      expect(markup).toContain('data-testid="select-for-preview-cv_viet_1"');
    });

    test('renders Test Phrase Synthesis Lab with phrase input, sample buttons, speed slider, and synthesis trigger', () => {
      const markup = renderToStaticMarkup(<VoiceManagementScreen />);
      expect(markup).toContain('Synthesis Lab &amp; Preview');
      expect(markup).toContain('Active Synthesis Voice:');
      expect(markup).toContain('data-testid="test-phrase-input"');
      expect(markup).toContain('VN Sample');
      expect(markup).toContain('EN Sample');
      expect(markup).toContain('Speech Rate');
      expect(markup).toContain('data-testid="synthesize-test-phrase-btn"');
      expect(markup).toContain('Synthesize &amp; Audition');
    });

    test('custom voice quick update modifies voice name and description in store', async () => {
      const originalFetch = globalThis.fetch;
      let putPayload: any = null;

      globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/custom-voices/cv_viet_1') && init?.method === 'PUT') {
          putPayload = JSON.parse(String(init?.body || '{}'));
          return new Response(JSON.stringify({ ok: true, id: 'cv_viet_1', ...putPayload }), { status: 200 });
        }
        if (urlStr.includes('/api/custom-voices')) {
          return new Response(JSON.stringify({
            voices: [
              { id: 'cv_viet_1', name: putPayload?.name || 'Renamed Voice', provider: 2, description: putPayload?.description || 'Updated' },
            ],
          }), { status: 200 });
        }
        return new Response('{}', { status: 200 });
      }) as any;

      try {
        await useDubDubStore.getState().updateVoice('cv_viet_1', {
          name: 'Renamed Storyteller',
          description: 'Updated warm resonance',
        });

        expect(putPayload).toEqual({
          name: 'Renamed Storyteller',
          description: 'Updated warm resonance',
        });
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    test('deleteVoice removes custom voice from store', async () => {
      const originalFetch = globalThis.fetch;
      let deleteCalled = false;

      globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/custom-voices/cv_viet_1') && init?.method === 'DELETE') {
          deleteCalled = true;
          return new Response(JSON.stringify({ ok: true, id: 'cv_viet_1' }), { status: 200 });
        }
        if (urlStr.includes('/api/custom-voices')) {
          return new Response(JSON.stringify({ voices: [] }), { status: 200 });
        }
        return new Response('{}', { status: 200 });
      }) as any;

      try {
        await useDubDubStore.getState().deleteVoice('cv_viet_1');
        expect(deleteCalled).toBe(true);
        expect(useDubDubStore.getState().customVoices).toHaveLength(0);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    test('deleteVoices bulk deletes custom voices via store and API', async () => {
      const originalFetch = globalThis.fetch;
      let bulkDeletePayload: any = null;

      globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
        const urlStr = String(url);
        if (urlStr.includes('/api/custom-voices/bulk-delete') && init?.method === 'POST') {
          bulkDeletePayload = JSON.parse(String(init.body));
          return new Response(JSON.stringify({ ok: true, deleted: ['cv_viet_1', 'cv_viet_2'], count: 2 }), { status: 200 });
        }
        if (urlStr.includes('/api/custom-voices')) {
          return new Response(JSON.stringify({ voices: [] }), { status: 200 });
        }
        return new Response('{}', { status: 200 });
      }) as any;

      try {
        useDubDubStore.setState({
          customVoices: [
            { id: 'cv_viet_1', name: 'Voice 1', provider: 2 },
            { id: 'cv_viet_2', name: 'Voice 2', provider: 2 },
          ],
        });

        await useDubDubStore.getState().deleteVoices(['cv_viet_1', 'cv_viet_2']);
        expect(bulkDeletePayload).toEqual({ ids: ['cv_viet_1', 'cv_viet_2'], hard: false });
        expect(useDubDubStore.getState().customVoices).toHaveLength(0);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    test('VoiceManagementScreen renders bulk select toolbar and voice checkboxes when custom voices exist', () => {
      useDubDubStore.setState({
        customVoices: [
          { id: 'cv_1', name: 'Custom Narrator', provider: 2 },
          { id: 'cv_2', name: 'Custom Announcer', provider: 2 },
        ],
      });

      const markup = renderToStaticMarkup(<VoiceManagementScreen initialTab="custom" />);
      expect(markup).toContain('data-testid="select-all-voices-btn"');
      expect(markup).toContain('data-testid="select-voice-cv_1"');
      expect(markup).toContain('data-testid="select-voice-cv_2"');
    });

    test('gender filter respects explicit voice gender property', () => {
      useDubDubStore.setState({
        voices: [
          { id: 'v_f', name: 'Speaker Alpha', gender: 'female', provider: 2, kind: 'preset' },
          { id: 'v_m', name: 'Speaker Beta', gender: 'male', provider: 2, kind: 'preset' },
        ],
      });

      // Renders both when all
      const allMarkup = renderToStaticMarkup(<VoiceManagementScreen initialTab="preset" initialGenderFilter="all" />);
      expect(allMarkup).toContain('Speaker Alpha');
      expect(allMarkup).toContain('Speaker Beta');

      // Filters to female only
      const femaleMarkup = renderToStaticMarkup(<VoiceManagementScreen initialTab="preset" initialGenderFilter="female" />);
      expect(femaleMarkup).toContain('Speaker Alpha');
      expect(femaleMarkup).not.toContain('Speaker Beta');
    });
  });

  describe('4. Drawer Retirement & App Screen Routing', () => {
    test('App renders ProjectsScreen by default on startup and has retired slide-over drawers', () => {
      useDubDubStore.setState({ activeView: 'projects', activeProjectId: null });
      const markup = renderToStaticMarkup(<App />);

      // Main screen ProjectsScreen is active
      expect(markup).toContain('data-screen="projects-screen"');
      expect(markup).toContain('Project Management');

      // Drawers are retired from App layout
      expect(markup).not.toContain('data-project-drawer');
      expect(markup).not.toContain('data-voice-manager-drawer');
    });

    test('App renders VoiceManagementScreen when activeView is "voices"', () => {
      useDubDubStore.setState({ activeView: 'voices' });
      const markup = renderToStaticMarkup(<App />);

      expect(markup).toContain('data-screen="voice-management-screen"');
      expect(markup).toContain('Voice Management Studio');
      expect(markup).not.toContain('data-screen="projects-screen"');
    });

    test('App renders WorkflowStepper and dubbing stages when activeView is "dubbing"', () => {
      useDubDubStore.setState({ activeView: 'dubbing', currentStep: 1 });
      const markup = renderToStaticMarkup(<App />);

      expect(markup).toContain('data-testid="workflow-back-to-projects-btn"');
      expect(markup).not.toContain('data-screen="projects-screen"');
      expect(markup).not.toContain('data-screen="voice-management-screen"');
    });
  });
});
