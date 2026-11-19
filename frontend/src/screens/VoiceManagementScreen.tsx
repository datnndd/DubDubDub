import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useDubDubStore } from '../store';
import {
  Mic,
  Volume2,
  Plus,
  Play,
  Pause,
  Trash2,
  CheckSquare,
  Square,
  MinusSquare,
  AlertTriangle,
  Edit2,
  Check,
  X,
  Search,
  Sparkles,
  Sliders,
  Filter,
  Loader2,
  RefreshCw,
  Layers,
  FileText,
  AlertCircle,
  CheckCircle2,
  ArrowRight,
  Headphones,
} from 'lucide-react';
import {
  previewCustomVoice,
  getCustomVoiceAudioUrl,
  getCustomVoicePreviewAudioUrl,
  previewTTS,
  type CustomVoice,
} from '../api/voices';
import {
  useVoiceAudition,
  auditionVoice,
  getStandardSamplePhrase,
} from '../services/voiceAuditionManager';
import type { VoiceOption } from '../types/dubbing';
import { ProviderDocLink } from '../components/ProviderDocLink';

export type VoiceCategoryTab = 'all' | 'custom' | 'preset';

export interface VoiceManagementScreenProps {
  initialTab?: VoiceCategoryTab;
  initialGenderFilter?: string;
}

export function matchesGenderFilter(voice: any, genderFilter: string): boolean {
  if (!genderFilter || genderFilter === 'all') return true;
  const gen = String(voice?.gender || voice?.tuning_params?.gender || '').trim().toLowerCase();
  if (gen) {
    if (genderFilter === 'female') {
      return gen === 'female' || gen === 'f' || gen === 'nữ' || gen === 'nu';
    }
    if (genderFilter === 'male') {
      return gen === 'male' || gen === 'm' || gen === 'nam';
    }
    return true;
  }

  const haystack = [
    voice?.name,
    voice?.description,
    voice?.external_voice_id,
    voice?.tuning_params?.base_voice,
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();

  if (genderFilter === 'female') {
    return (
      /(?:^|[\s(,/])(female|woman|girl|nữ|mai anh|rachel|sarah|alice|jessica|bella|lily|matilda|laura|ly|ngọc|ngoc|đoan|doan)(?:$|[\s),./])/i.test(haystack)
    );
  }
  if (genderFilter === 'male') {
    return (
      /(?:^|[\s(,/])(male|man|boy|nam|roger|george|charlie|callum|harry|liam|will|eric|chris|brian|daniel|adam|bill|bình|binh|tuyên|tuyen|vĩnh|vinh)(?:$|[\s),./])/i.test(haystack)
    );
  }
  return true;
}

export const VoiceManagementScreen: React.FC<VoiceManagementScreenProps> = ({
  initialTab = 'custom',
  initialGenderFilter = 'all',
}) => {
  const customVoices = useDubDubStore((s) => s.customVoices);
  const voices = useDubDubStore((s) => s.voices);
  const loadVoices = useDubDubStore((s) => s.loadVoices);
  const loadCustomVoices = useDubDubStore((s) => s.loadCustomVoices);
  const updateVoice = useDubDubStore((s) => s.updateVoice);
  const deleteVoice = useDubDubStore((s) => s.deleteVoice);
  const deleteVoices = useDubDubStore((s) => s.deleteVoices);
  const setCreateVoiceModalOpen = useDubDubStore((s) => s.setCreateVoiceModalOpen);
  const storeProvider = useDubDubStore((s) => s.backend?.config?.ttsType ?? 2);
  const storeLanguage = useDubDubStore((s) => s.languages?.target?.code || 'vi');

  // Filters & Tabs
  const [activeTab, setActiveTab] = useState<VoiceCategoryTab>(initialTab);
  const [search, setSearch] = useState('');
  const [languageFilter, setLanguageFilter] = useState('all');
  const [providerFilter, setProviderFilter] = useState('all');
  const [genderFilter, setGenderFilter] = useState(initialGenderFilter);
  const [kindFilter, setKindFilter] = useState<'all' | 'clone' | 'design'>('all');

  // Bulk Selection & Deletion State
  const [selectedVoiceIds, setSelectedVoiceIds] = useState<Set<string>>(new Set());
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);
  const [confirmBulkDeleteOpen, setConfirmBulkDeleteOpen] = useState(false);

  // Custom Voice In-place Editing
  const [editingVoiceId, setEditingVoiceId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [editingDesc, setEditingDesc] = useState('');

  // Delete Confirmation & Library Errors
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [libraryError, setLibraryError] = useState<string | null>(null);

  // Audition Coordinator
  const { isKeyPlaying, isKeyLoading, stop: stopAudition } = useVoiceAudition();
  const [loadingVoiceMap, setLoadingVoiceMap] = useState<Record<string, boolean>>({});

  // Test Phrase Synthesis Lab State
  const [selectedLabVoiceId, setSelectedLabVoiceId] = useState<string>('');
  const [testPhrase, setTestPhrase] = useState(getStandardSamplePhrase('vi'));
  const [testSpeed, setTestSpeed] = useState<number>(1.0);
  const [labStyle, setLabStyle] = useState<'tu_nhien' | 'tin_tuc' | 'doc_truyen'>('tu_nhien');
  const [labTemperature, setLabTemperature] = useState<number>(0.8);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [synthesisError, setSynthesisError] = useState<string | null>(null);
  const [testAudioUrl, setTestAudioUrl] = useState<string | null>(null);
  const [isTestPlaying, setIsTestPlaying] = useState(false);
  const testAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    loadVoices();
    loadCustomVoices();
  }, [loadVoices, loadCustomVoices]);

  // Clean up all playing audio on unmount or tab switch to prevent audio leakage
  useEffect(() => {
    return () => {
      stopAudition();
      if (testAudioRef.current) {
        testAudioRef.current.pause();
        testAudioRef.current.src = '';
      }
    };
  }, [stopAudition]);

  // Format preset voices vs custom voices
  const presetVoices: VoiceOption[] = useMemo(() => {
    return (voices || []).filter((v) => {
      const isClone =
        v.kind === 'clone' ||
        v.kind === 'custom' ||
        v.id.startsWith('voice_') ||
        v.id.startsWith('cv_');
      const isSystem = v.id === 'No' || v.id.toLowerCase() === 'clone';
      return !isClone && !isSystem;
    });
  }, [voices]);

  // Set default selected voice for lab when voices load
  useEffect(() => {
    if (!selectedLabVoiceId) {
      if (customVoices.length > 0) {
        setSelectedLabVoiceId(customVoices[0].id);
      } else if (presetVoices.length > 0) {
        setSelectedLabVoiceId(presetVoices[0].id);
      }
    }
  }, [customVoices, presetVoices, selectedLabVoiceId]);

  const handleStartEdit = (v: CustomVoice) => {
    setLibraryError(null);
    setEditingVoiceId(v.id);
    setEditingName(v.name);
    setEditingDesc(v.description || '');
  };

  const handleSaveEdit = async (voiceId: string) => {
    const clean = editingName.trim();
    if (!clean) {
      setLibraryError('Voice name cannot be empty.');
      return;
    }
    setLibraryError(null);
    try {
      await updateVoice(voiceId, {
        name: clean,
        description: editingDesc.trim(),
      });
      setEditingVoiceId(null);
    } catch (err: any) {
      console.error('Failed to update voice:', err);
      setLibraryError(err instanceof Error ? err.message : 'Failed to update voice.');
    }
  };

  const handleDeleteCustomVoice = async (voiceId: string) => {
    setLibraryError(null);
    try {
      await deleteVoice(voiceId, false);
      setConfirmDeleteId(null);
      setSelectedVoiceIds((prev) => {
        const next = new Set(prev);
        next.delete(voiceId);
        return next;
      });
      if (selectedLabVoiceId === voiceId) {
        setSelectedLabVoiceId('');
      }
    } catch (err: any) {
      console.error('Failed to delete voice:', err);
      setLibraryError(err instanceof Error ? err.message : 'Failed to delete voice.');
    }
  };

  const handleToggleAudition = async (voiceId: string, voiceName: string, provider: number, isCustom: boolean) => {
    const key = `voicescreen-${voiceId}`;
    const targetProvider = provider ?? storeProvider;
    const targetLanguage = storeLanguage;

    const staticSampleUrl = isCustom
      ? getCustomVoicePreviewAudioUrl(voiceId)
      : (voices.find((v) => v.id === voiceId)?.sampleUrl || undefined);

    await auditionVoice({
      key,
      voice: voiceId,
      provider: targetProvider,
      language: targetLanguage,
      staticSampleUrl,
      onLoadingChange: (loading) => {
        setLoadingVoiceMap((prev) => ({ ...prev, [voiceId]: loading }));
      },
    });
  };

  const handleSelectForLab = (voiceId: string) => {
    setSelectedLabVoiceId(voiceId);
    setSynthesisError(null);
  };

  const handleRunTestSynthesis = async () => {
    if (!selectedLabVoiceId || !testPhrase.trim()) return;
    setIsSynthesizing(true);
    setSynthesisError(null);
    stopAudition();

    try {
      // Find voice provider and details
      const customMatch = customVoices.find((cv) => cv.id === selectedLabVoiceId);
      const presetMatch = voices.find((v) => v.id === selectedLabVoiceId);
      const targetProvider = customMatch?.provider ?? presetMatch?.provider ?? storeProvider;
      const targetLanguage = customMatch?.language ?? storeLanguage;

      const targetVoice = customMatch ? customMatch.id : selectedLabVoiceId;
      const isVieNeuTarget = targetProvider === 2;
      const ttsRes = await previewTTS({
        text: testPhrase.trim(),
        voice: targetVoice,
        provider: targetProvider,
        language: targetLanguage,
        speed: testSpeed,
        ...(isVieNeuTarget ? {
          style: labStyle,
          tuningParams: {
            style: labStyle,
            temperature: labTemperature,
          },
        } : {}),
        force_refresh: true,
      });
      const generatedAudioUrl = ttsRes.preview_url || ttsRes.audio_url;
      const bustedUrl = generatedAudioUrl.includes('?')
        ? `${generatedAudioUrl}&_t=${Date.now()}`
        : `${generatedAudioUrl}?_t=${Date.now()}`;

      setTestAudioUrl(bustedUrl);

      if (testAudioRef.current) {
        testAudioRef.current.src = bustedUrl;
        testAudioRef.current.load();
        testAudioRef.current.volume = 1.0;
        testAudioRef.current
          .play()
          .then(() => setIsTestPlaying(true))
          .catch((playErr) => {
            console.warn('Autoplay prevented or deferred:', playErr);
            setIsTestPlaying(false);
          });
      }
    } catch (err: any) {
      console.error('Test phrase synthesis failed:', err);
      setSynthesisError(err.message || 'Failed to synthesize test phrase');
    } finally {
      setIsSynthesizing(false);
    }
  };

  const toggleTestPlayback = () => {
    if (!testAudioRef.current || !testAudioUrl) return;
    if (isTestPlaying) {
      testAudioRef.current.pause();
      setIsTestPlaying(false);
    } else {
      testAudioRef.current
        .play()
        .then(() => setIsTestPlaying(true))
        .catch((playErr) => {
          console.warn('Test playback prevented:', playErr);
          setIsTestPlaying(false);
        });
    }
  };

  // Filtered voice lists
  const filteredCustomVoices = useMemo(() => {
    return customVoices.filter((v) => {
      if (kindFilter === 'clone' && v.kind === 'design') return false;
      if (kindFilter === 'design' && v.kind !== 'design') return false;
      const q = search.trim().toLowerCase();
      if (q && !v.name.toLowerCase().includes(q) && !(v.description || '').toLowerCase().includes(q) && !(v.ref_text || '').toLowerCase().includes(q)) {
        return false;
      }
      if (providerFilter !== 'all' && String(v.provider) !== providerFilter) return false;
      if (languageFilter !== 'all' && v.language && !v.language.toLowerCase().includes(languageFilter.toLowerCase())) return false;
      if (!matchesGenderFilter(v, genderFilter)) return false;
      return true;
    });
  }, [customVoices, kindFilter, search, providerFilter, languageFilter, genderFilter]);

  const allFilteredCustomSelected =
    filteredCustomVoices.length > 0 && filteredCustomVoices.every((v) => selectedVoiceIds.has(v.id));
  const someFilteredCustomSelected =
    filteredCustomVoices.some((v) => selectedVoiceIds.has(v.id)) && !allFilteredCustomSelected;

  const toggleSelectVoice = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setSelectedVoiceIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAllCustomVoices = () => {
    if (allFilteredCustomSelected) {
      setSelectedVoiceIds(new Set());
    } else {
      setSelectedVoiceIds(new Set(filteredCustomVoices.map((v) => v.id)));
    }
  };

  const handleBulkDeleteCustomVoices = async () => {
    if (selectedVoiceIds.size === 0) return;
    setIsBulkDeleting(true);
    setLibraryError(null);
    try {
      const idsToDelete = Array.from(selectedVoiceIds);
      await deleteVoices(idsToDelete);
      if (idsToDelete.includes(selectedLabVoiceId)) {
        setSelectedLabVoiceId('');
      }
      setSelectedVoiceIds(new Set());
      setConfirmBulkDeleteOpen(false);
    } catch (err: any) {
      console.error('Failed to bulk delete voices:', err);
      setLibraryError(err instanceof Error ? err.message : 'Failed to bulk delete voices.');
    } finally {
      setIsBulkDeleting(false);
    }
  };

  const filteredPresetVoices = useMemo(() => {
    return presetVoices.filter((v) => {
      const q = search.trim().toLowerCase();
      if (q && !v.name.toLowerCase().includes(q) && !v.id.toLowerCase().includes(q)) {
        return false;
      }
      if (providerFilter !== 'all' && String(v.provider) !== providerFilter) return false;
      if (!matchesGenderFilter(v, genderFilter)) return false;
      return true;
    });
  }, [presetVoices, search, providerFilter, genderFilter]);

  const getProviderBadge = (provider: number) => {
    switch (provider) {
      case 2:
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-200">
            VieNeu 48k
          </span>
        );
      case 1:
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-sky-100 text-sky-900 border border-sky-200">
            OmniVoice 24k
          </span>
        );
      case 0:
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-100 text-purple-900 border border-purple-200">
            ElevenLabs
          </span>
        );
      case 3:
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-900 border border-emerald-200">
            Gemini TTS
          </span>
        );
      default:
        return (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-stone-100 text-stone-700">
            Neural Voice
          </span>
        );
    }
  };

  const selectedVoiceMeta = useMemo(() => {
    if (!selectedLabVoiceId) return null;
    const cv = customVoices.find((c) => c.id === selectedLabVoiceId);
    if (cv) return { id: cv.id, name: cv.name, isCustom: true, provider: cv.provider };
    const pv = voices.find((v) => v.id === selectedLabVoiceId);
    if (pv) return { id: pv.id, name: pv.name, isCustom: false, provider: pv.provider ?? storeProvider };
    return { id: selectedLabVoiceId, name: selectedLabVoiceId, isCustom: false, provider: storeProvider };
  }, [selectedLabVoiceId, customVoices, voices, storeProvider]);

  return (
    <div
      data-screen="voice-management-screen"
      className="flex-1 w-full h-full flex flex-col bg-[#F9F8F5] overflow-hidden"
    >
      {/* Top Banner & Control Bar */}
      <div className="bg-white border-b border-[#E7E4DC] px-6 py-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3 shrink-0 shadow-2xs">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-amber-500/15 text-[#8D4B00] flex items-center justify-center border border-amber-500/25 shadow-2xs">
            <Mic className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-sm text-stone-900 tracking-tight">Voice Management Studio</h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-[#8D4B00]">
                {customVoices.length + presetVoices.length} voices
              </span>
            </div>
            <p className="text-[10px] text-stone-500 font-medium">
              Audition samples, clone custom speakers, and test phrase synthesis across AI backbones
            </p>
          </div>
        </div>

        {/* Action: + Create Custom Voice Button */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            data-testid="create-voice-btn"
            onClick={() => setCreateVoiceModalOpen(true)}
            className="px-3.5 py-1.5 text-xs font-bold rounded-lg bg-[#8D4B00] text-white hover:bg-[#743D00] flex items-center gap-1.5 shadow-2xs cursor-pointer transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Create Custom Voice</span>
          </button>
        </div>
      </div>

      {/* Main Studio Body: Two Columns or Two Panels */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
        {/* Left / Center Panel: Voice Library & Filters */}
        <div className="flex-1 flex flex-col min-w-0 border-r border-[#E7E4DC] overflow-hidden">
          {/* Library Sub-navigation Tabs & Search Toolbar */}
          <div className="p-4 bg-white border-b border-[#E7E4DC] space-y-3 shrink-0">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              {/* Tabs */}
              <div className="flex items-center gap-1 p-0.5 bg-stone-100 rounded-lg border border-stone-200 text-xs">
                <button
                  type="button"
                  data-testid="tab-custom-voices"
                  onClick={() => setActiveTab('custom')}
                  className={`px-3 py-1 rounded-md font-semibold flex items-center gap-1.5 cursor-pointer transition-colors ${
                    activeTab === 'custom'
                      ? 'bg-white text-[#8D4B00] font-bold shadow-2xs border border-amber-200/80'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#8D4B00]" />
                  <span>Custom Cloned Voices</span>
                  <span className="px-1.5 py-0.2 bg-amber-100 text-[#8D4B00] rounded-full text-[9px] font-bold">
                    {customVoices.length}
                  </span>
                </button>

                <button
                  type="button"
                  data-testid="tab-preset-voices"
                  onClick={() => setActiveTab('preset')}
                  className={`px-3 py-1 rounded-md font-semibold flex items-center gap-1.5 cursor-pointer transition-colors ${
                    activeTab === 'preset'
                      ? 'bg-white text-[#8D4B00] font-bold shadow-2xs border border-amber-200/80'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                >
                  <Volume2 className="w-3.5 h-3.5 text-stone-500" />
                  <span>Preset Built-in Voices</span>
                  <span className="px-1.5 py-0.2 bg-stone-200 text-stone-700 rounded-full text-[9px] font-bold">
                    {presetVoices.length}
                  </span>
                </button>

                <button
                  type="button"
                  data-testid="tab-all-voices"
                  onClick={() => setActiveTab('all')}
                  className={`px-3 py-1 rounded-md font-semibold flex items-center gap-1.5 cursor-pointer transition-colors ${
                    activeTab === 'all'
                      ? 'bg-white text-[#8D4B00] font-bold shadow-2xs border border-amber-200/80'
                      : 'text-stone-600 hover:text-stone-900'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5 text-stone-400" />
                  <span>All Voices ({customVoices.length + presetVoices.length})</span>
                </button>
              </div>
            </div>

            {/* Search and Filters Bar */}
            <div className="flex items-center gap-2 flex-wrap text-xs">
              <div className="relative flex-1 min-w-[180px]">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-stone-400 pointer-events-none" />
                <input
                  type="text"
                  data-testid="voices-search-input"
                  placeholder="Filter voices by name, timbre, description…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full pl-8 pr-7 py-1 text-xs bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-[#8D4B00]"
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch('')}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-stone-400 hover:text-stone-700"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              {/* Language Filter */}
              <select
                data-testid="voice-language-filter"
                value={languageFilter}
                onChange={(e) => setLanguageFilter(e.target.value)}
                className="px-2 py-1 bg-stone-50 border border-stone-200 rounded-lg text-xs font-medium text-stone-700 focus:outline-none cursor-pointer"
              >
                <option value="all">All Languages</option>
                <option value="vi">Vietnamese (vi)</option>
                <option value="en">English (en)</option>
                <option value="zh">Chinese (zh)</option>
                <option value="ja">Japanese (ja)</option>
              </select>

              {/* Provider Filter */}
              <div className="flex items-center gap-1">
                <select
                  data-testid="voice-provider-filter"
                  value={providerFilter}
                  onChange={(e) => setProviderFilter(e.target.value)}
                  className="px-2 py-1 bg-stone-50 border border-stone-200 rounded-lg text-xs font-medium text-stone-700 focus:outline-none cursor-pointer"
                >
                  <option value="all">All Providers</option>
                  <option value="2">VieNeu-TTS (48k)</option>
                  <option value="1">OmniVoice (24k)</option>
                  <option value="0">ElevenLabs</option>
                  <option value="3">Gemini TTS</option>
                </select>
                {providerFilter !== 'all' && (
                  <ProviderDocLink providerId={Number(providerFilter)} category="tts" variant="icon" />
                )}
              </div>

              {/* Gender Filter */}
              <select
                data-testid="voice-gender-filter"
                value={genderFilter}
                onChange={(e) => setGenderFilter(e.target.value)}
                className="px-2 py-1 bg-stone-50 border border-stone-200 rounded-lg text-xs font-medium text-stone-700 focus:outline-none cursor-pointer"
              >
                <option value="all">All Genders</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>

              {/* Kind / Type Filter */}
              <select
                data-testid="voice-kind-filter"
                value={kindFilter}
                onChange={(e) => setKindFilter(e.target.value as any)}
                className="px-2 py-1 bg-stone-50 border border-stone-200 rounded-lg text-xs font-medium text-stone-700 focus:outline-none cursor-pointer"
              >
                <option value="all">All Types</option>
                <option value="clone">Cloned Only</option>
                <option value="design">Designed Only</option>
              </select>
            </div>
          </div>

          {/* Voice List Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {libraryError && (
              <div
                role="alert"
                data-testid="voice-library-error"
                className="p-2.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between gap-2"
              >
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                  <span>{libraryError}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setLibraryError(null)}
                  className="p-0.5 rounded hover:bg-rose-100 text-rose-600 cursor-pointer"
                  aria-label="Dismiss error"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
            {/* 1. Custom Voices Section */}
            {(activeTab === 'custom' || activeTab === 'all') && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-bold text-stone-700">
                  <div className="flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-[#8D4B00]" />
                    <span>Custom Cloned Library ({filteredCustomVoices.length})</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setCreateVoiceModalOpen(true)}
                    className="text-[11px] font-semibold text-[#8D4B00] hover:underline cursor-pointer flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" />
                    <span>Clone New Voice</span>
                  </button>
                </div>

                {filteredCustomVoices.length === 0 ? (
                  <div className="p-6 rounded-xl border border-dashed border-stone-200 bg-white text-center">
                    <Sparkles className="w-8 h-8 mx-auto text-amber-500/40 mb-2" />
                    <p className="font-semibold text-xs text-stone-800">No Custom Voices Cloned Yet</p>
                    <p className="text-[11px] text-stone-500 mt-1 max-w-sm mx-auto">
                      Clone a unique voice from a 3-10s audio sample to personalize dialogue, character cues, and narration.
                    </p>
                    <button
                      type="button"
                      onClick={() => setCreateVoiceModalOpen(true)}
                      className="mt-3 px-3 py-1.5 rounded-lg bg-[#8D4B00] text-white font-bold text-xs inline-flex items-center gap-1.5 shadow-2xs hover:bg-[#743D00] cursor-pointer"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      <span>Clone Voice Now</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2.5">
                    {/* Bulk Action & Selection Bar */}
                    <div className="bg-stone-50/90 border border-[#E7E4DC] rounded-lg px-3 py-1.5 flex items-center justify-between gap-3 text-xs shrink-0">
                      <div className="flex items-center gap-3">
                        <button
                          type="button"
                          data-testid="select-all-voices-btn"
                          onClick={toggleSelectAllCustomVoices}
                          className="flex items-center gap-1.5 font-semibold text-stone-700 hover:text-stone-900 cursor-pointer px-1.5 py-0.5 rounded hover:bg-stone-200/50 transition-colors"
                          title={allFilteredCustomSelected ? 'Deselect all voices' : 'Select all voices'}
                        >
                          {allFilteredCustomSelected ? (
                            <CheckSquare className="w-4 h-4 text-[#8D4B00]" />
                          ) : someFilteredCustomSelected ? (
                            <MinusSquare className="w-4 h-4 text-[#8D4B00]" />
                          ) : (
                            <Square className="w-4 h-4 text-stone-400" />
                          )}
                          <span>
                            {allFilteredCustomSelected ? 'Deselect All' : 'Select All'} ({filteredCustomVoices.length})
                          </span>
                        </button>

                        {selectedVoiceIds.size > 0 && (
                          <span
                            data-testid="selected-voices-count"
                            className="text-[11px] font-semibold text-[#8D4B00] bg-amber-100/80 px-2 py-0.5 rounded-full"
                          >
                            {selectedVoiceIds.size} selected
                          </span>
                        )}
                      </div>

                      {selectedVoiceIds.size > 0 && (
                        <button
                          type="button"
                          data-testid="bulk-delete-voices-btn"
                          onClick={() => setConfirmBulkDeleteOpen(true)}
                          className="px-2.5 py-1 rounded-md bg-rose-50 hover:bg-rose-100 text-rose-700 font-semibold text-xs border border-rose-200 flex items-center gap-1.5 cursor-pointer shadow-2xs transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5 text-rose-600" />
                          <span>Delete Selected ({selectedVoiceIds.size})</span>
                        </button>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                      {filteredCustomVoices.map((voice) => {
                        const isSelected = selectedVoiceIds.has(voice.id);
                        const isEditing = editingVoiceId === voice.id;
                        const isPlaying = isKeyPlaying(`voicescreen-${voice.id}`);
                        const isLoading = isKeyLoading(`voicescreen-${voice.id}`) || Boolean(loadingVoiceMap[voice.id]);
                        const isConfirmingDelete = confirmDeleteId === voice.id;
                        const isSelectedForLab = selectedLabVoiceId === voice.id;

                        return (
                          <div
                            key={voice.id}
                            data-voice-card={voice.id}
                            onClick={() => handleSelectForLab(voice.id)}
                            className={`p-3 rounded-xl border transition-all bg-white shadow-2xs cursor-pointer ${
                              isSelected
                                ? 'border-amber-400 ring-1 ring-amber-400/50 bg-amber-50/30'
                                : isSelectedForLab
                                ? 'border-amber-400 ring-1 ring-amber-400/50 bg-amber-50/20'
                                : 'border-stone-200 hover:border-amber-200'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex items-start gap-2 min-w-0 flex-1">
                                {/* Selection Checkbox */}
                                <button
                                  type="button"
                                  data-action="select-voice"
                                  data-testid={`select-voice-${voice.id}`}
                                  onClick={(e) => toggleSelectVoice(voice.id, e)}
                                  className="p-1 rounded hover:bg-stone-100 text-stone-400 hover:text-stone-700 transition-colors shrink-0 cursor-pointer self-center"
                                  title={isSelected ? 'Deselect voice' : 'Select voice'}
                                >
                                  {isSelected ? (
                                    <CheckSquare className="w-4 h-4 text-[#8D4B00]" />
                                  ) : (
                                    <Square className="w-4 h-4 text-stone-300" />
                                  )}
                                </button>

                                {/* Inline Audition Button */}
                                <button
                                  type="button"
                                  data-action="audition-voice"
                                  data-voice-id={voice.id}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleToggleAudition(voice.id, voice.name, voice.provider, true);
                                  }}
                                  disabled={isLoading}
                                  className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-colors cursor-pointer ${
                                    isPlaying
                                      ? 'bg-[#8D4B00] text-white shadow-2xs'
                                      : 'bg-amber-50 hover:bg-amber-100 text-[#8D4B00] border border-amber-200/80'
                                  }`}
                                  title={isPlaying ? 'Stop Audition' : 'Audition Voice Sample'}
                              >
                                {isLoading ? (
                                  <Loader2 className="w-4 h-4 animate-spin text-amber-700" />
                                ) : isPlaying ? (
                                  <Pause className="w-4 h-4" />
                                ) : (
                                  <Play className="w-4 h-4 ml-0.5" />
                                )}
                              </button>

                              <div className="min-w-0 flex-1">
                                {isEditing ? (
                                  <div className="space-y-1">
                                    <input
                                      type="text"
                                      value={editingName}
                                      onChange={(e) => setEditingName(e.target.value)}
                                      className="text-xs font-bold px-2 py-0.5 bg-stone-50 border border-stone-200 rounded w-full outline-hidden"
                                    />
                                    <input
                                      type="text"
                                      placeholder="Description..."
                                      value={editingDesc}
                                      onChange={(e) => setEditingDesc(e.target.value)}
                                      className="text-[10px] px-2 py-0.5 bg-stone-50 border border-stone-200 rounded w-full outline-hidden text-stone-600"
                                    />
                                  </div>
                                ) : (
                                  <>
                                    <div className="flex items-center gap-1.5 flex-wrap">
                                      <h3 className="font-bold text-xs text-stone-900 truncate" title={voice.name}>
                                        {voice.name}
                                      </h3>
                                      {getProviderBadge(voice.provider)}
                                      {voice.kind === 'design' ? (
                                        <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-purple-100 text-purple-800 border border-purple-200">
                                          Designed • {voice.tuning_params?.style === 'tin_tuc' ? 'Tin tức' : voice.tuning_params?.style === 'doc_truyen' ? 'Đọc truyện' : 'Tự nhiên'}
                                        </span>
                                      ) : (
                                        <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-amber-100 text-[#8D4B00]">
                                          Cloned
                                        </span>
                                      )}
                                    </div>
                                    {voice.description ? (
                                      <p className="text-[10px] text-stone-500 truncate mt-0.5">
                                        {voice.description}
                                      </p>
                                    ) : null}
                                  </>
                                )}
                              </div>
                            </div>

                            {/* Actions & Lab Selection */}
                            <div className="flex items-center gap-1 shrink-0">
                              {isEditing ? (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleSaveEdit(voice.id);
                                  }}
                                  className="p-1 rounded hover:bg-emerald-50 text-emerald-600 cursor-pointer"
                                  title="Save Changes"
                                >
                                  <Check className="w-3.5 h-3.5" />
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleStartEdit(voice);
                                  }}
                                  className="p-1 rounded hover:bg-stone-100 text-stone-400 hover:text-stone-700 cursor-pointer"
                                  title="Rename / Edit Description"
                                >
                                  <Edit2 className="w-3.5 h-3.5" />
                                </button>
                              )}

                              {!isConfirmingDelete ? (
                                <button
                                  type="button"
                                  data-testid={`delete-voice-${voice.id}`}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setConfirmDeleteId(voice.id);
                                  }}
                                  className="p-1 rounded hover:bg-rose-50 text-stone-400 hover:text-rose-600 cursor-pointer"
                                  title="Delete Custom Voice"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              ) : (
                                <div className="flex items-center gap-1 bg-rose-50 p-1 rounded-lg border border-rose-200">
                                  <span className="text-[9px] text-rose-700 font-bold px-0.5">Sure?</span>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleDeleteCustomVoice(voice.id);
                                    }}
                                    className="px-1 py-0.5 rounded text-[9px] font-bold bg-rose-600 text-white hover:bg-rose-700 cursor-pointer"
                                  >
                                    Yes
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setConfirmDeleteId(null);
                                    }}
                                    className="px-1 py-0.5 rounded text-[9px] font-bold bg-stone-200 text-stone-700 cursor-pointer"
                                  >
                                    No
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Reference Text Snippet */}
                          {voice.ref_text && (
                            <div className="mt-2 text-[10px] text-stone-500 bg-stone-50 p-1.5 rounded-lg border border-stone-100 line-clamp-1">
                              <span className="font-semibold text-stone-600">Ref: </span>
                              {voice.ref_text}
                            </div>
                          )}

                          {/* Select for Lab button */}
                          <div className="mt-2 pt-2 border-t border-stone-100 flex items-center justify-between">
                            <span className="text-[10px] text-stone-400 font-mono">ID: {voice.id}</span>
                            <button
                              type="button"
                              data-testid={`select-for-preview-${voice.id}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectForLab(voice.id);
                              }}
                              className={`px-2 py-0.5 rounded text-[10px] font-semibold cursor-pointer transition-colors ${
                                isSelectedForLab
                                  ? 'bg-[#8D4B00] text-white font-bold'
                                  : 'bg-stone-100 hover:bg-stone-200 text-stone-700'
                              }`}
                            >
                              {isSelectedForLab ? 'Selected for Lab' : 'Test in Lab'}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              </div>
            )}

            {/* 2. Preset Built-in Voices Section */}
            {(activeTab === 'preset' || activeTab === 'all') && (
              <div className="space-y-2 pt-2">
                <div className="flex items-center justify-between text-xs font-bold text-stone-700">
                  <div className="flex items-center gap-1.5">
                    <Volume2 className="w-3.5 h-3.5 text-stone-500" />
                    <span>Preset Built-in Voices ({filteredPresetVoices.length})</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {filteredPresetVoices.map((voice) => {
                    const isPlaying = isKeyPlaying(`voicescreen-${voice.id}`);
                    const isLoading = isKeyLoading(`voicescreen-${voice.id}`) || Boolean(loadingVoiceMap[voice.id]);
                    const isSelectedForLab = selectedLabVoiceId === voice.id;

                    return (
                      <div
                        key={voice.id}
                        data-voice-card={voice.id}
                        onClick={() => handleSelectForLab(voice.id)}
                        className={`p-3 rounded-xl border transition-all bg-white shadow-2xs cursor-pointer ${
                          isSelectedForLab
                            ? 'border-amber-400 ring-1 ring-amber-400/50 bg-amber-50/20'
                            : 'border-stone-200 hover:border-amber-200'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-start gap-2.5 min-w-0 flex-1">
                            <button
                              type="button"
                              data-action="audition-voice"
                              data-voice-id={voice.id}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleToggleAudition(voice.id, voice.name, voice.provider ?? storeProvider, false);
                              }}
                              disabled={isLoading}
                              className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-colors cursor-pointer ${
                                isPlaying
                                  ? 'bg-[#8D4B00] text-white shadow-2xs'
                                  : 'bg-stone-100 hover:bg-stone-200 text-stone-700'
                              }`}
                              title={isPlaying ? 'Stop Audition' : 'Audition Voice Sample'}
                            >
                              {isLoading ? (
                                <Loader2 className="w-4 h-4 animate-spin text-amber-700" />
                              ) : isPlaying ? (
                                <Pause className="w-4 h-4" />
                              ) : (
                                <Play className="w-4 h-4 ml-0.5" />
                              )}
                            </button>

                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <h3 className="font-bold text-xs text-stone-900 truncate" title={voice.name}>
                                  {voice.name}
                                </h3>
                                {voice.provider !== undefined ? getProviderBadge(voice.provider) : null}
                              </div>
                              <p className="text-[10px] text-stone-400 mt-0.5">Built-in preset model</p>
                            </div>
                          </div>

                          <button
                            type="button"
                            data-testid={`select-for-preview-${voice.id}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSelectForLab(voice.id);
                            }}
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold cursor-pointer transition-colors ${
                              isSelectedForLab
                                ? 'bg-[#8D4B00] text-white font-bold'
                                : 'bg-stone-100 hover:bg-stone-200 text-stone-700'
                            }`}
                          >
                            {isSelectedForLab ? 'Selected' : 'Test'}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: Test Phrase Synthesis & Preview Studio */}
        <div className="w-full lg:w-96 bg-white flex flex-col shrink-0 border-t lg:border-t-0 overflow-y-auto">
          <div className="p-4 border-b border-[#E7E4DC] bg-stone-50/80 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Headphones className="w-4 h-4 text-[#8D4B00]" />
              <h2 className="font-bold text-xs text-stone-900">Synthesis Lab &amp; Preview</h2>
            </div>
            {selectedVoiceMeta && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-[#8D4B00]">
                {selectedVoiceMeta.name}
              </span>
            )}
          </div>

          <div className="p-4 space-y-4 flex-1">
            {/* Selected Voice Info Box & Direct Selector */}
            <div className="p-3 rounded-xl border border-amber-200 bg-amber-50/40 space-y-2">
              <div className="text-[11px] font-bold text-stone-700 flex items-center justify-between">
                <span>Active Synthesis Voice:</span>
                {selectedVoiceMeta?.isCustom ? (
                  <span className="text-[9px] px-1.5 py-0.2 rounded bg-amber-200 text-[#8D4B00] font-bold">
                    Custom Clone
                  </span>
                ) : (
                  <span className="text-[9px] px-1.5 py-0.2 rounded bg-stone-200 text-stone-700 font-bold">
                    Preset
                  </span>
                )}
              </div>

              {/* Direct Voice Picker Dropdown */}
              <select
                data-testid="lab-voice-picker"
                value={selectedLabVoiceId}
                onChange={(e) => handleSelectForLab(e.target.value)}
                className="w-full text-xs px-2.5 py-1.5 bg-white rounded-lg border border-amber-200 text-stone-900 font-bold focus:outline-none focus:border-[#8D4B00] cursor-pointer shadow-2xs"
              >
                {!selectedLabVoiceId && <option value="">Select a voice to test...</option>}
                {(() => {
                  const labCustomVoices = genderFilter === 'all'
                    ? customVoices
                    : customVoices.filter((cv) => matchesGenderFilter(cv, genderFilter));
                  const labPresetVoices = genderFilter === 'all'
                    ? presetVoices
                    : presetVoices.filter((pv) => matchesGenderFilter(pv, genderFilter));
                  return (
                    <>
                      {labCustomVoices.length > 0 && (
                        <optgroup label="Custom Cloned Voices">
                          {labCustomVoices.map((cv) => (
                            <option key={cv.id} value={cv.id}>
                              {cv.name} ({cv.kind === 'design' ? 'Designed' : 'Cloned'})
                            </option>
                          ))}
                        </optgroup>
                      )}
                      {labPresetVoices.length > 0 && (
                        <optgroup label="Preset Voices">
                          {labPresetVoices.map((pv) => (
                            <option key={pv.id} value={pv.id}>
                              {pv.name}
                            </option>
                          ))}
                        </optgroup>
                      )}
                    </>
                  );
                })()}
              </select>

              <div className="text-[10px] text-stone-500">
                You can pick any voice above or click on a voice card in the library.
              </div>
            </div>

            {/* Test Phrase Input */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-stone-700">Test Phrase:</label>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => setTestPhrase(getStandardSamplePhrase('vi'))}
                    className="text-[10px] font-semibold text-[#8D4B00] hover:underline cursor-pointer"
                  >
                    VN Sample
                  </button>
                  <span className="text-stone-300">•</span>
                  <button
                    type="button"
                    onClick={() => setTestPhrase(getStandardSamplePhrase('en'))}
                    className="text-[10px] font-semibold text-[#8D4B00] hover:underline cursor-pointer"
                  >
                    EN Sample
                  </button>
                </div>
              </div>
              <textarea
                data-testid="test-phrase-input"
                rows={4}
                value={testPhrase}
                onChange={(e) => setTestPhrase(e.target.value)}
                placeholder="Type or paste any test sentence to evaluate voice pronunciation and tone..."
                className="w-full text-xs p-2.5 bg-stone-50 rounded-xl border border-stone-200 focus:bg-white focus:outline-none focus:border-[#8D4B00] transition-colors"
              />
              {/* Emotion Tag Shortcuts (VieNeu only) */}
              {(selectedVoiceMeta?.provider ?? storeProvider) === 2 && (
                <div className="flex items-center gap-1.5 flex-wrap pt-1" data-testid="lab-vieneu-emotion-tags">
                  <span className="text-[10px] text-stone-400 font-medium">Emotion tags:</span>
                  <button
                    type="button"
                    onClick={() => setTestPhrase((prev) => `${prev.trim()} [cười] `)}
                    className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition-colors cursor-pointer"
                  >
                    + [cười]
                  </button>
                  <button
                    type="button"
                    onClick={() => setTestPhrase((prev) => `${prev.trim()} [thở dài] `)}
                    className="px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-50 text-sky-700 border border-sky-200 hover:bg-sky-100 transition-colors cursor-pointer"
                  >
                    + [thở dài]
                  </button>
                  <button
                    type="button"
                    onClick={() => setTestPhrase((prev) => `${prev.trim()} [hắng giọng] `)}
                    className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-50 text-purple-700 border border-purple-200 hover:bg-purple-100 transition-colors cursor-pointer"
                  >
                    + [hắng giọng]
                  </button>
                </div>
              )}
            </div>

            {/* Speaking Style & Expressiveness (VieNeu only) */}
            {(selectedVoiceMeta?.provider ?? storeProvider) === 2 && (
              <>
                <div className="space-y-1 bg-stone-50 p-2.5 rounded-xl border border-stone-200" data-testid="lab-vieneu-style-control">
                  <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                    <div className="flex items-center gap-1">
                      <Sliders className="w-3.5 h-3.5 text-stone-400" />
                      <span>VieNeu Speaking Style</span>
                    </div>
                  </div>
                  <select
                    value={labStyle}
                    onChange={(e) => setLabStyle(e.target.value as any)}
                    className="w-full text-xs px-2 py-1.5 bg-white rounded-lg border border-stone-200 text-stone-800 font-medium focus:outline-none focus:border-[#8D4B00] cursor-pointer"
                  >
                    <option value="tu_nhien">Tự nhiên (Natural Dialogue)</option>
                    <option value="tin_tuc">Tin tức (News Broadcast)</option>
                    <option value="doc_truyen">Đọc truyện (Storytelling / Drama)</option>
                  </select>
                </div>

                <div className="space-y-1 bg-stone-50 p-2.5 rounded-xl border border-stone-200" data-testid="lab-vieneu-temp-control">
                  <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                    <div className="flex items-center gap-1">
                      <Sliders className="w-3.5 h-3.5 text-stone-400" />
                      <span>Expressiveness (Temp)</span>
                    </div>
                    <span className="font-mono text-[11px] text-[#8D4B00] font-bold">{labTemperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="1.2"
                    step="0.05"
                    value={labTemperature}
                    onChange={(e) => setLabTemperature(parseFloat(e.target.value))}
                    className="w-full h-1 bg-stone-200 rounded-lg appearance-none cursor-pointer accent-[#8D4B00]"
                  />
                  <div className="flex items-center justify-between text-[9px] text-stone-400">
                    <span>0.5 Calm</span>
                    <span>0.8 Normal</span>
                    <span>1.2 Expressive</span>
                  </div>
                </div>
              </>
            )}

            {/* Speed / Pace Slider */}
            <div className="space-y-1 bg-stone-50 p-2.5 rounded-xl border border-stone-200">
              <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                <div className="flex items-center gap-1">
                  <Sliders className="w-3.5 h-3.5 text-stone-400" />
                  <span>Speech Rate</span>
                </div>
                <span className="font-mono text-[11px] text-[#8D4B00] font-bold">{testSpeed.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={testSpeed}
                onChange={(e) => setTestSpeed(parseFloat(e.target.value))}
                className="w-full h-1 bg-stone-200 rounded-lg appearance-none cursor-pointer accent-[#8D4B00]"
              />
              <div className="flex items-center justify-between text-[9px] text-stone-400">
                <span>0.5x Slow</span>
                <span>1.0x Normal</span>
                <span>2.0x Fast</span>
              </div>
            </div>

            {/* Error Message */}
            {synthesisError && (
              <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{synthesisError}</span>
              </div>
            )}

            {/* Synthesize CTA Button */}
            <button
              type="button"
              data-testid="synthesize-test-phrase-btn"
              onClick={handleRunTestSynthesis}
              disabled={isSynthesizing || !selectedLabVoiceId || !testPhrase.trim()}
              className="w-full py-2.5 rounded-xl bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center justify-center gap-2 shadow-2xs transition-colors cursor-pointer disabled:opacity-50"
            >
              {isSynthesizing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Synthesizing Audio…</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Synthesize &amp; Audition</span>
                </>
              )}
            </button>

            {/* Audio Preview Player */}
            {testAudioUrl && (
              <div className="p-3 rounded-xl border border-stone-200 bg-stone-50 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={toggleTestPlayback}
                      className="w-8 h-8 rounded-full bg-[#8D4B00] text-white flex items-center justify-center cursor-pointer shadow-2xs hover:bg-[#743D00]"
                    >
                      {isTestPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                    </button>
                    <div className="text-xs font-bold text-stone-800">
                      {isTestPlaying ? 'Playing Test Audio' : 'Preview Ready'}
                    </div>
                  </div>
                  <span className="text-[10px] text-emerald-700 font-bold bg-emerald-100 px-2 py-0.5 rounded-full">
                    Synthesized
                  </span>
                </div>
              </div>
            )}

            {/* Hidden Audio Element */}
            <audio
              ref={testAudioRef}
              onEnded={() => setIsTestPlaying(false)}
              onError={() => setIsTestPlaying(false)}
              onPlay={() => setIsTestPlaying(true)}
              onPause={() => setIsTestPlaying(false)}
              className="hidden"
            />
          </div>
        </div>
      </div>

      {/* Bulk Delete Confirmation Modal */}
      {confirmBulkDeleteOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/40 backdrop-blur-xs animate-fade-in">
          <div className="bg-white rounded-2xl border border-stone-200 shadow-xl max-w-md w-full p-5 space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-xl bg-rose-50 border border-rose-200 text-rose-600 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-stone-900">
                  Delete {selectedVoiceIds.size} Custom {selectedVoiceIds.size === 1 ? 'Voice' : 'Voices'}?
                </h3>
                <p className="text-xs text-stone-600 leading-relaxed">
                  This will permanently delete the selected custom cloned voices and their reference audio files. Any speakers currently mapped to these voices will fall back to preset defaults.
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-stone-100">
              <button
                type="button"
                data-testid="cancel-bulk-delete-btn"
                onClick={() => setConfirmBulkDeleteOpen(false)}
                disabled={isBulkDeleting}
                className="px-3 py-1.5 rounded-lg border border-stone-200 text-xs font-semibold text-stone-700 hover:bg-stone-50 cursor-pointer transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                data-testid="confirm-bulk-delete-btn"
                onClick={handleBulkDeleteCustomVoices}
                disabled={isBulkDeleting}
                className="px-3.5 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold flex items-center gap-1.5 shadow-2xs cursor-pointer transition-colors disabled:opacity-50"
              >
                {isBulkDeleting ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Deleting…</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Confirm Delete ({selectedVoiceIds.size})</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
