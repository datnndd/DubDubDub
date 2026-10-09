import React, { useEffect } from 'react';
import { useDubDubStore } from './store';
import { Header } from './components/Header';
import { WorkflowStepper } from './components/WorkflowStepper';
import { StatusFooter } from './components/StatusFooter';
import { CreateVoiceModal } from './components/CreateVoiceModal';
import { FloatingPill } from './components/FloatingPill';
import { ProjectsScreen } from './screens/ProjectsScreen';
import { VoiceManagementScreen } from './screens/VoiceManagementScreen';
import { Stage1Prepare } from './screens/Stage1Prepare';
import { Stage2ReviewTranscript } from './screens/Stage2ReviewTranscript';
import { Stage3VoiceDubbing } from './screens/Stage3VoiceDubbing';
import { Stage4EditVideo } from './screens/Stage4EditVideo';
import { SettingsModal } from './components/settings/SettingsModal';
import { CapCutExportModal } from './components/CapCutExportModal';
import { FolderScanModal } from './components/FolderScanModal';
import { SessionVideosDrawer } from './components/SessionVideosDrawer';
import { ErrorBoundary } from './components/ErrorBoundary';

export const App: React.FC = () => {
  const activeView = useDubDubStore((s) => s.activeView);
  const currentStep = useDubDubStore((s) => s.currentStep);
  const activeProjectId = useDubDubStore((s) => s.activeProjectId);
  const initializeBackend = useDubDubStore((s) => s.initializeBackend);
  const loadProjects = useDubDubStore((s) => s.loadProjects);
  const selectProject = useDubDubStore((s) => s.selectProject);
  const loadVoices = useDubDubStore((s) => s.loadVoices);

  useEffect(() => {
    initializeBackend();
    loadProjects();
    loadVoices();
    if (activeProjectId) {
      // Preload active project state without overriding the initial 'projects' screen
      selectProject(activeProjectId, false);
    }
  }, []);

  const renderStage = () => {
    switch (currentStep) {
      case 1:
        return <Stage1Prepare />;
      case 2:
        return <Stage2ReviewTranscript />;
      case 3:
        return <Stage3VoiceDubbing />;
      case 4:
        return <Stage4EditVideo />;
      default:
        return <Stage1Prepare />;
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#F9F8F5] text-stone-800 font-sans antialiased select-none text-xs">
      <Header />

      {activeView === 'projects' && (
        <main className="flex-1 min-h-0 w-full overflow-hidden flex flex-col">
          <ErrorBoundary fallbackTitle="Project Manager Error">
            <ProjectsScreen />
          </ErrorBoundary>
        </main>
      )}

      {activeView === 'voices' && (
        <main className="flex-1 min-h-0 w-full overflow-hidden flex flex-col">
          <ErrorBoundary fallbackTitle="Voice Management Error">
            <VoiceManagementScreen />
          </ErrorBoundary>
        </main>
      )}

      {activeView === 'dubbing' && (
        <>
          <WorkflowStepper />
          <main className="flex-1 min-h-0 w-full overflow-hidden flex flex-col">
            <ErrorBoundary fallbackTitle="Workflow View Error">
              {renderStage()}
            </ErrorBoundary>
          </main>
          <StatusFooter />
        </>
      )}

      <CreateVoiceModal />
      <FloatingPill />
      <SettingsModal />
      <CapCutExportModal />
      <FolderScanModal />
      <SessionVideosDrawer />
    </div>
  );
};
