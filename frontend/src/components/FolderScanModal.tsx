import React, { useState, useRef } from 'react';
import { useDubDubStore } from '../store';
import {
  X,
  FolderSearch,
  FolderOpen,
  Plus,
  Trash2,
  CheckSquare,
  Square,
  Loader2,
  Film,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  Upload,
} from 'lucide-react';
import { scanFolders, registerLocalMedia, uploadBatchMedia } from '../api/media';
import { batchCreateProjects } from '../api/projects';
import { formatBytes } from '../store/prepareSlice';

export interface ScannedVideoItem {
  path: string;
  filename: string;
  sizeBytes: number;
  parentDir: string;
  file?: File;
  selected: boolean;
}

export const FolderScanModal: React.FC = () => {
  const isOpen = useDubDubStore((s) => s.scanModalOpen);
  const setScanModalOpen = useDubDubStore((s) => s.setScanModalOpen);
  const setSessionSubprojects = useDubDubStore((s) => s.setSessionSubprojects);
  const selectProject = useDubDubStore((s) => s.selectProject);
  const setActiveView = useDubDubStore((s) => s.setActiveView);
  const setStep = useDubDubStore((s) => s.setStep);
  const setSessionDrawerOpen = useDubDubStore((s) => s.setSessionDrawerOpen);

  const [folderPaths, setFolderPaths] = useState<string[]>([]);
  const [currentPathInput, setCurrentPathInput] = useState('');
  const [isRecursive, setIsRecursive] = useState(true);
  const [isScanning, setIsScanning] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [detectedVideos, setDetectedVideos] = useState<ScannedVideoItem[]>([]);
  const [searchFilter, setSearchFilter] = useState('');

  const directoryInputRef = useRef<HTMLInputElement | null>(null);

  if (!isOpen) return null;

  const handleAddPath = () => {
    const trimmed = currentPathInput.trim().replace(/^["']|["']$/g, '');
    if (trimmed && !folderPaths.includes(trimmed)) {
      setFolderPaths([...folderPaths, trimmed]);
      setCurrentPathInput('');
      setScanError(null);
    }
  };

  const handleRemovePath = (idx: number) => {
    setFolderPaths(folderPaths.filter((_, i) => i !== idx));
  };

  const handleScan = async () => {
    const pathsToScan = [...folderPaths];
    const pendingInput = currentPathInput.trim().replace(/^["']|["']$/g, '');
    if (pendingInput && !pathsToScan.includes(pendingInput)) {
      pathsToScan.push(pendingInput);
      setFolderPaths(pathsToScan);
      setCurrentPathInput('');
    }

    if (pathsToScan.length === 0) {
      setScanError('Please enter at least one folder path or select a directory below.');
      return;
    }

    setIsScanning(true);
    setScanError(null);
    try {
      const res = await scanFolders(pathsToScan, isRecursive);
      if (res?.ok && Array.isArray(res.videos)) {
        const items: ScannedVideoItem[] = res.videos.map((v) => ({
          ...v,
          selected: true,
        }));
        setDetectedVideos(items);
        if (items.length === 0) {
          setScanError('No .mp4 video files found in the specified folder(s).');
        }
      } else {
        setScanError('Folder scan completed with no results.');
      }
    } catch (err: any) {
      setScanError(err.message || 'Failed to scan folders');
    } finally {
      setIsScanning(false);
    }
  };

  const handleBrowserDirectoryPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const mp4Files = files.filter((f) => f.name.toLowerCase().endsWith('.mp4'));
    if (mp4Files.length === 0) {
      setScanError('No .mp4 files were found in the selected folder.');
      return;
    }

    const items: ScannedVideoItem[] = mp4Files.map((f) => ({
      path: f.webkitRelativePath || f.name,
      filename: f.name,
      sizeBytes: f.size,
      parentDir: (f.webkitRelativePath || '').split('/').slice(0, -1).join('/') || 'Root Folder',
      file: f,
      selected: true,
    }));

    setDetectedVideos((prev) => {
      const existingNames = new Set(prev.map((p) => p.filename));
      const newItems = items.filter((it) => !existingNames.has(it.filename));
      return [...prev, ...newItems];
    });
    setScanError(null);
  };

  const toggleSelectVideo = (path: string) => {
    setDetectedVideos((prev) =>
      prev.map((v) => (v.path === path ? { ...v, selected: !v.selected } : v))
    );
  };

  const selectedVideos = detectedVideos.filter((v) => v.selected);
  const allSelected = detectedVideos.length > 0 && selectedVideos.length === detectedVideos.length;

  const toggleSelectAll = () => {
    const nextVal = !allSelected;
    setDetectedVideos((prev) => prev.map((v) => ({ ...v, selected: nextVal })));
  };

  const handleConfirmImport = async () => {
    if (selectedVideos.length === 0) return;

    setIsImporting(true);
    setScanError(null);

    try {
      const localItems = selectedVideos.filter((v) => !v.file);
      const browserFileItems = selectedVideos.filter((v) => v.file);

      let registeredMedia: any[] = [];

      // 1. Register local media paths without copying
      if (localItems.length > 0) {
        const localPaths = localItems.map((v) => v.path);
        const regRes = await registerLocalMedia(localPaths);
        if (regRes?.ok && Array.isArray(regRes.media)) {
          registeredMedia = [...registeredMedia, ...regRes.media];
        }
      }

      // 2. Upload browser files if any
      if (browserFileItems.length > 0) {
        const filesToUpload = browserFileItems.map((v) => v.file!).filter(Boolean);
        const upRes = await uploadBatchMedia(filesToUpload);
        if (upRes?.ok && Array.isArray(upRes.media)) {
          registeredMedia = [...registeredMedia, ...upRes.media];
        }
      }

      // Fallback: If registration was empty, create placeholder batch items with paths
      const batchItems = registeredMedia.length > 0
        ? registeredMedia.map((m) => ({
            name: m.filename,
            mediaId: m.id,
            duration: m.durationMs ? m.durationMs / 1000 : 0,
          }))
        : selectedVideos.map((v) => ({
            name: v.filename,
            mediaPath: v.path,
          }));

      // 3. Create parent project (named today's date YYYY-MM-DD) and subprojects
      const today = new Date().toISOString().split('T')[0];
      const batchRes = await batchCreateProjects({
        date: today,
        items: batchItems,
      });

      if (batchRes?.ok && Array.isArray(batchRes.projects) && batchRes.projects.length > 0) {
        setSessionSubprojects(batchRes.projects, batchRes.parent.id, batchRes.parent.name);

        // Select the first subproject to view in Stage 1
        const firstProject = batchRes.projects[0];
        await selectProject(firstProject.id, false);

        setScanModalOpen(false);
        setActiveView('dubbing');
        setStep(1);
        setSessionDrawerOpen(true);
      } else {
        throw new Error('Failed to create batch session projects');
      }
    } catch (err: any) {
      setScanError(err.message || 'Failed to import selected videos');
    } finally {
      setIsImporting(false);
    }
  };

  const filteredVideos = detectedVideos.filter((v) => {
    if (!searchFilter.trim()) return true;
    const q = searchFilter.toLowerCase();
    return v.filename.toLowerCase().includes(q) || v.path.toLowerCase().includes(q);
  });

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl border border-stone-200 shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="h-14 px-6 border-b border-stone-200 flex items-center justify-between bg-stone-50/75 shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-500/15 text-[#8D4B00] flex items-center justify-center border border-amber-500/25">
              <FolderSearch className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-stone-900">Scan &amp; Batch Import Videos</h2>
              <p className="text-[11px] text-stone-500">Recursively scan local directories for .mp4 files and group into a daily project</p>
            </div>
          </div>
          <button
            onClick={() => setScanModalOpen(false)}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-200/50 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* Section 1: Folder Selection Input */}
          <div className="space-y-3">
            <label className="block text-xs font-bold text-stone-800">
              Folder Paths to Scan (Host Machine)
            </label>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="e.g. C:\Videos\Batch or /home/user/videos"
                  value={currentPathInput}
                  onChange={(e) => setCurrentPathInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddPath();
                    }
                  }}
                  className="w-full px-3.5 py-2 text-xs rounded-xl border border-stone-200 bg-white placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-amber-500/30 focus:border-amber-500"
                />
              </div>
              <button
                type="button"
                onClick={handleAddPath}
                className="px-3 py-2 text-xs font-semibold rounded-xl border border-stone-200 bg-stone-100 hover:bg-stone-200 text-stone-700 flex items-center gap-1.5 cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" /> Add
              </button>
              <button
                type="button"
                onClick={handleScan}
                disabled={isScanning}
                className="px-4 py-2 text-xs font-bold rounded-xl bg-[#8D4B00] hover:bg-[#743D00] text-white flex items-center gap-1.5 shadow-xs cursor-pointer disabled:opacity-50"
              >
                {isScanning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FolderSearch className="w-3.5 h-3.5" />}
                <span>Scan Folders</span>
              </button>
            </div>

            {/* Folder list chips */}
            {folderPaths.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-1">
                {folderPaths.map((p, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] bg-stone-100 border border-stone-200 text-stone-800"
                  >
                    <HardDrive className="w-3 h-3 text-stone-500" />
                    <span className="font-mono truncate max-w-sm">{p}</span>
                    <button
                      type="button"
                      onClick={() => handleRemovePath(idx)}
                      className="text-stone-400 hover:text-rose-600 cursor-pointer"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}

            {/* Options & Alternative Browser Picker */}
            <div className="flex items-center justify-between pt-1 text-xs text-stone-600">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={isRecursive}
                  onChange={(e) => setIsRecursive(e.target.checked)}
                  className="rounded border-stone-300 text-amber-600 focus:ring-amber-500 cursor-pointer"
                />
                <span>Scan subfolders recursively (detect all nested .mp4 files)</span>
              </label>

              <div>
                <input
                  ref={directoryInputRef}
                  type="file"
                  // @ts-expect-error webkitdirectory attribute is non-standard
                  webkitdirectory="true"
                  directory="true"
                  multiple
                  className="hidden"
                  onChange={handleBrowserDirectoryPick}
                />
                <button
                  type="button"
                  onClick={() => directoryInputRef.current?.click()}
                  className="text-xs font-medium text-amber-700 hover:text-amber-800 flex items-center gap-1 cursor-pointer"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Or browse folder from browser…</span>
                </button>
              </div>
            </div>
          </div>

          {scanError && (
            <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{scanError}</span>
            </div>
          )}

          {/* Section 2: Detected Videos Preview Table */}
          {detectedVideos.length > 0 && (
            <div className="space-y-3 pt-2 border-t border-stone-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-bold text-stone-900">
                    Detected MP4 Videos ({detectedVideos.length})
                  </h3>
                  <span className="text-[11px] font-semibold text-amber-800 bg-amber-100 px-2 py-0.5 rounded-full">
                    {selectedVideos.length} selected
                  </span>
                </div>
                <input
                  type="text"
                  placeholder="Filter videos…"
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="px-2.5 py-1 text-xs rounded-lg border border-stone-200 bg-white placeholder-stone-400 w-44"
                />
              </div>

              {/* Table */}
              <div className="border border-stone-200 rounded-xl overflow-hidden max-h-64 overflow-y-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead className="bg-stone-50 border-b border-stone-200 sticky top-0 z-10 text-[11px] text-stone-500 font-semibold uppercase tracking-wider">
                    <tr>
                      <th className="py-2 px-3 w-10 text-center">
                        <button
                          type="button"
                          onClick={toggleSelectAll}
                          className="text-stone-600 hover:text-stone-900 cursor-pointer"
                          title={allSelected ? 'Deselect all' : 'Select all'}
                        >
                          {allSelected ? (
                            <CheckSquare className="w-4 h-4 text-[#8D4B00]" />
                          ) : (
                            <Square className="w-4 h-4" />
                          )}
                        </button>
                      </th>
                      <th className="py-2 px-3">Video File</th>
                      <th className="py-2 px-3">Folder Path</th>
                      <th className="py-2 px-3 text-right">Size</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-stone-100">
                    {filteredVideos.map((item) => (
                      <tr
                        key={item.path}
                        onClick={() => toggleSelectVideo(item.path)}
                        className={`hover:bg-amber-50/50 cursor-pointer transition-colors ${
                          item.selected ? 'bg-amber-50/30' : ''
                        }`}
                      >
                        <td className="py-2 px-3 text-center" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={item.selected}
                            onChange={() => toggleSelectVideo(item.path)}
                            className="rounded border-stone-300 text-amber-600 focus:ring-amber-500 cursor-pointer"
                          />
                        </td>
                        <td className="py-2 px-3 font-medium text-stone-900 flex items-center gap-2">
                          <Film className="w-3.5 h-3.5 text-[#8D4B00] shrink-0" />
                          <span className="truncate max-w-xs">{item.filename}</span>
                        </td>
                        <td className="py-2 px-3 text-stone-500 text-[11px] truncate max-w-sm font-mono">
                          {item.parentDir}
                        </td>
                        <td className="py-2 px-3 text-right text-stone-600 text-[11px] font-mono">
                          {formatBytes(item.sizeBytes)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="h-14 px-6 border-t border-stone-200 flex items-center justify-between bg-stone-50/75 shrink-0">
          <div className="text-xs text-stone-500 font-medium">
            {selectedVideos.length > 0 ? (
              <span>Group will be organized in parent project: <strong>{new Date().toISOString().split('T')[0]}</strong></span>
            ) : (
              <span>Select folders to scan</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setScanModalOpen(false)}
              className="px-3.5 py-1.5 rounded-xl border border-stone-200 text-stone-700 hover:bg-stone-100 text-xs font-semibold cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={selectedVideos.length === 0 || isImporting}
              onClick={handleConfirmImport}
              className="px-4 py-1.5 rounded-xl bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center gap-1.5 shadow-xs cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isImporting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
              <span>Import &amp; Proceed to Stage 1 ({selectedVideos.length})</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
