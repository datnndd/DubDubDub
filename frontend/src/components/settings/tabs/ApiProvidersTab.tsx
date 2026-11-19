import React, { useState } from 'react';
import {
  Key,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Eye,
  EyeOff,
  Radio,
  ExternalLink,
  ShieldCheck,
  Check,
  RefreshCw,
} from 'lucide-react';
import { ProviderStatus, updateProviderSettings, testProviderConnection, fetchProviderModels, saveProviderAndLoadModels } from '../../../api/settingsApi';
import { useDubDubStore } from '../../../store';
import { ProviderDocLink } from '../../ProviderDocLink';

interface ApiProvidersTabProps {
  providers: Record<string, ProviderStatus>;
  targetProvider?: string | null;
  onRefresh: () => Promise<void>;
}

export const ApiProvidersTab: React.FC<ApiProvidersTabProps> = ({
  providers,
  targetProvider,
  onRefresh,
}) => {
  // Local edit state for forms
  const [formState, setFormState] = useState<
    Record<
      string,
      {
        apiKey?: string;
        model?: string;
        baseUrl?: string;
        mirrorUrl?: string;
        showKey?: boolean;
      }
    >
  >({});

  const [savingMap, setSavingMap] = useState<Record<string, boolean>>({});
  const [testingMap, setTestingMap] = useState<Record<string, boolean>>({});
  const [loadingModelsMap, setLoadingModelsMap] = useState<Record<string, boolean>>({});
  const [modelsMap, setModelsMap] = useState<Record<string, string[]>>({});
  const [feedbackMap, setFeedbackMap] = useState<
    Record<string, { type: 'success' | 'error'; message: string } | null>
  >({});
  const targetRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (targetProvider && targetRef.current) {
      targetRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [targetProvider]);

  const getForm = (id: string, initial: ProviderStatus) => {
    const availableModels = modelsMap[id] || initial.models || [];
    const local = formState[id] || {};
    return {
      apiKey: local.apiKey !== undefined ? local.apiKey : (initial.apiKey || ''),
      model: local.model !== undefined ? local.model : (initial.model || (availableModels && availableModels[0]) || ''),
      baseUrl: local.baseUrl !== undefined ? local.baseUrl : (initial.baseUrl || ''),
      mirrorUrl: local.mirrorUrl !== undefined ? local.mirrorUrl : (initial.mirrorUrl || ''),
      showKey: local.showKey !== undefined ? local.showKey : false,
    };
  };

  const updateForm = (id: string, updates: Partial<ReturnType<typeof getForm>>, initial: ProviderStatus) => {
    setFormState((prev) => ({
      ...prev,
      [id]: {
        ...getForm(id, initial),
        ...updates,
      },
    }));
  };

  const handleSave = async (id: string, initial: ProviderStatus) => {
    const form = getForm(id, initial);
    setSavingMap((prev) => ({ ...prev, [id]: true }));
    setFeedbackMap((prev) => ({ ...prev, [id]: null }));

    try {
      const payload: { apiKey?: string; model?: string; baseUrl?: string; mirrorUrl?: string } = {};
      if (form.apiKey !== undefined) {
        payload.apiKey = form.apiKey.trim();
      }
      if (form.model) {
        payload.model = form.model;
      }
      if (form.baseUrl !== undefined) {
        payload.baseUrl = form.baseUrl.trim();
      }
      if (form.mirrorUrl !== undefined) {
        payload.mirrorUrl = form.mirrorUrl.trim();
      }

      const category = initial.category || 'general';
      const refreshed = await saveProviderAndLoadModels(category, id, payload);
      const models = refreshed.models;
      let selectedModel = form.model;
      if (models.length > 0) {
        setModelsMap((prev) => ({ ...prev, [id]: models }));
        if (!selectedModel || !models.includes(selectedModel)) {
          selectedModel = models[0];
          await updateProviderSettings(category, id, { model: selectedModel });
        }
      } else {
        setModelsMap((prev) => {
          const next = { ...prev };
          delete next[id];
          return next;
        });
      }
      // Clear entered secret key and reset local form dirty state for this provider
      setFormState((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });

      const store = useDubDubStore.getState();
      const curRecognType = Number(store.backend?.config?.recognType);
      if (id === 'deepgram' && curRecognType === 1 && selectedModel) {
        store.updateAsrProvider(1, selectedModel);
      } else if (id === 'gemini' && curRecognType === 2 && selectedModel) {
        store.updateAsrProvider(2, selectedModel);
      } else if (id === 'elevenlabs' && curRecognType === 4 && selectedModel) {
        store.updateAsrProvider(4, selectedModel);
      }

      if (id === 'elevenlabs' || id === 'gemini') {
        await store.loadVoices?.();
      }

      await onRefresh();
      setFeedbackMap((prev) => ({
        ...prev,
        [id]: { type: 'success', message: refreshed.modelError
          ? `Settings saved. Models could not be refreshed: ${refreshed.modelError}`
          : 'Settings saved and models refreshed' },
      }));
    } catch (err: any) {
      setFeedbackMap((prev) => ({
        ...prev,
        [id]: { type: 'error', message: err.message || 'Failed to save settings' },
      }));
    } finally {
      setSavingMap((prev) => ({ ...prev, [id]: false }));
    }
  };

  const handleLoadModels = async (id: string, initial: ProviderStatus) => {
    const form = getForm(id, initial);
    setLoadingModelsMap((prev) => ({ ...prev, [id]: true }));
    setFeedbackMap((prev) => ({ ...prev, [id]: null }));

    try {
      const payload: { apiKey?: string; baseUrl?: string } = {};
      if (form.apiKey.trim()) {
        payload.apiKey = form.apiKey.trim();
      }
      if (form.baseUrl.trim()) {
        payload.baseUrl = form.baseUrl.trim();
      }

      const res = await fetchProviderModels(initial.category || 'general', id, payload);
      const models = res.models;
      if (models && models.length > 0) {
        setModelsMap((prev) => ({ ...prev, [id]: models }));
        if (!form.model || !models.includes(form.model)) {
          updateForm(id, { model: models[0] }, initial);
        }
      }
      setFeedbackMap((prev) => ({
        ...prev,
        [id]: { type: 'success', message: res.message || `Loaded ${models?.length || 0} models successfully` },
      }));
    } catch (err: any) {
      setFeedbackMap((prev) => ({
        ...prev,
        [id]: { type: 'error', message: err.message || 'Failed to load models from provider' },
      }));
    } finally {
      setLoadingModelsMap((prev) => ({ ...prev, [id]: false }));
    }
  };

  const handleTest = async (id: string, initial: ProviderStatus) => {
    const form = getForm(id, initial);
    setTestingMap((prev) => ({ ...prev, [id]: true }));
    setFeedbackMap((prev) => ({ ...prev, [id]: null }));

    try {
      const payload: { apiKey?: string; baseUrl?: string } = {};
      if (form.apiKey.trim()) {
        payload.apiKey = form.apiKey.trim();
      }
      if (form.baseUrl.trim()) {
        payload.baseUrl = form.baseUrl.trim();
      }

      const res = await testProviderConnection(initial.category || 'general', id, payload);
      const testModels = res.models;
      if (testModels && testModels.length > 0) {
        setModelsMap((prev) => ({ ...prev, [id]: testModels }));
        if (!form.model || !testModels.includes(form.model)) {
          updateForm(id, { model: testModels[0] }, initial);
        }
      }
      setFeedbackMap((prev) => ({
        ...prev,
        [id]: {
          type: 'success',
          message:
            res.models && res.models.length > 0
              ? `${res.message || 'Connection successful!'} (Loaded ${res.models.length} models)`
              : (res.message || 'Connection successful!'),
        },
      }));
    } catch (err: any) {
      setFeedbackMap((prev) => ({
        ...prev,
        [id]: { type: 'error', message: err.message || 'Connection test failed' },
      }));
    } finally {
      setTestingMap((prev) => ({ ...prev, [id]: false }));
    }
  };

  const providerList = Object.values(providers);

  return (
    <div className="space-y-4">
      <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 flex items-start gap-2.5">
        <ShieldCheck className="w-4 h-4 text-[#8D4B00] shrink-0 mt-0.5" />
        <div className="text-[11px] text-stone-700 leading-relaxed">
          <strong className="font-semibold text-stone-900">Encrypted Credentials:</strong> All API keys are
          encrypted at rest using machine-bound cryptography. Keys are masked by default to protect your privacy and
          can be revealed and edited when needed.
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {providerList.map((p) => {
          const form = getForm(p.id, p);
          const tNorm = (targetProvider || '').toLowerCase();
          const pNorm = p.id.toLowerCase();
          const isTarget = !!targetProvider && (
            tNorm === pNorm ||
            (tNorm.startsWith('gemini') && pNorm === 'gemini') ||
            (tNorm.includes('deepgram') && pNorm === 'deepgram') ||
            (tNorm.includes('eleven') && pNorm === 'elevenlabs')
          );
          const isSaving = savingMap[p.id] || false;
          const isTesting = testingMap[p.id] || false;
          const feedback = feedbackMap[p.id];

          return (
            <div
              key={p.id}
              ref={isTarget ? targetRef : undefined}
              className={`bg-white rounded-xl border p-4 shadow-2xs flex flex-col justify-between transition-all ${
                isTarget ? 'border-[#8D4B00] ring-2 ring-amber-500/20' : 'border-stone-200 hover:border-stone-300'
              }`}
            >
              <div>
                {/* Header */}
                <div className="flex items-center justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-stone-100 flex items-center justify-center text-stone-700">
                      <Key className="w-3.5 h-3.5 text-[#8D4B00]" />
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <h4 className="font-bold text-xs text-stone-900 leading-tight">{p.name}</h4>
                        <ProviderDocLink providerId={p.id} category={p.category as any} variant="badge" />
                      </div>
                      <span className="text-[10px] text-stone-400 font-medium uppercase tracking-wider">
                        {p.category}
                      </span>
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div>
                    {p.fromEnv ? (
                      <span className="px-2 py-0.5 rounded-full text-[9px] font-bold bg-purple-50 text-purple-700 border border-purple-200">
                        ENV OVERRIDE
                      </span>
                    ) : p.configured ? (
                      <span className="px-2 py-0.5 rounded-full text-[9px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
                        <CheckCircle2 className="w-2.5 h-2.5" />
                        <span>Configured</span>
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full text-[9px] font-bold bg-amber-50 text-amber-800 border border-amber-200">
                        Key Required
                      </span>
                    )}
                  </div>
                </div>

                {/* Form fields */}
                <div className="space-y-2.5 text-xs">
                  {/* API Key Input */}
                  <div>
                    <label className="text-[10px] font-semibold text-stone-600 block mb-1">
                      {p.id === 'huggingface' ? 'User Access Token' : 'API Key'}
                    </label>
                    <div className="relative flex items-center">
                      <input
                        type={form.showKey ? 'text' : 'password'}
                        disabled={isSaving}
                        placeholder={
                          p.fromEnv
                            ? 'Configured via environment variable'
                            : p.configured
                            ? '••••••••••••••••••••••••'
                            : 'Enter API key…'
                        }
                        value={form.apiKey}
                        onChange={(e) => updateForm(p.id, { apiKey: e.target.value }, p)}
                        className="w-full text-xs p-2 pr-9 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400 font-mono disabled:opacity-60 disabled:cursor-not-allowed"
                      />
                      <button
                        type="button"
                        onClick={() => updateForm(p.id, { showKey: !form.showKey }, p)}
                        className="absolute right-2 text-stone-400 hover:text-stone-700 p-1 rounded hover:bg-stone-200/50 transition-colors cursor-pointer"
                        title={form.showKey ? 'Hide key' : 'Show key'}
                        aria-label={form.showKey ? 'Hide key' : 'Show key'}
                      >
                        {form.showKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>

                  {/* Model Selector if available */}
                  {((modelsMap[p.id] && modelsMap[p.id].length > 0) || (p.models && p.models.length > 0)) && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="text-[10px] font-semibold text-stone-600">Model</label>
                        <button
                          type="button"
                          onClick={() => handleLoadModels(p.id, p)}
                          disabled={loadingModelsMap[p.id] || (!p.configured && !form.apiKey.trim())}
                          className="text-[10px] text-amber-700 hover:text-amber-800 font-medium flex items-center gap-1 disabled:opacity-50 cursor-pointer"
                          title="Fetch available models from provider"
                        >
                          <RefreshCw className={`w-2.5 h-2.5 ${loadingModelsMap[p.id] ? 'animate-spin' : ''}`} />
                          <span>{loadingModelsMap[p.id] ? 'Loading…' : 'Load Models'}</span>
                        </button>
                      </div>
                      <select
                        value={form.model}
                        onChange={(e) => updateForm(p.id, { model: e.target.value }, p)}
                        className="w-full text-xs p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400"
                      >
                        {(() => {
                          const availableModels = modelsMap[p.id] || p.models || [];
                          const displayedModels =
                            form.model && !availableModels.includes(form.model)
                              ? [form.model, ...availableModels]
                              : availableModels;
                          return displayedModels.map((m) => (
                            <option key={m} value={m}>
                              {m}
                            </option>
                          ));
                        })()}
                      </select>
                    </div>
                  )}

                  {/* Base URL if supported */}
                  {p.baseUrl !== undefined && (
                    <div>
                      <label className="text-[10px] font-semibold text-stone-600 block mb-1">Custom Base URL</label>
                      <input
                        type="text"
                        placeholder="Default API Endpoint"
                        value={form.baseUrl}
                        onChange={(e) => updateForm(p.id, { baseUrl: e.target.value }, p)}
                        className="w-full text-xs p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400 font-mono"
                      />
                    </div>
                  )}

                  {/* Mirror URL for Hugging Face */}
                  {p.mirrorUrl !== undefined && (
                    <div>
                      <label className="text-[10px] font-semibold text-stone-600 block mb-1">HF Mirror URL</label>
                      <input
                        type="text"
                        placeholder="https://hf-mirror.com"
                        value={form.mirrorUrl}
                        onChange={(e) => updateForm(p.id, { mirrorUrl: e.target.value }, p)}
                        className="w-full text-xs p-2 bg-stone-50 rounded-lg border border-stone-200 focus:bg-white focus:outline-none focus:border-amber-400 font-mono"
                      />
                    </div>
                  )}
                </div>

                {/* Feedback Alert */}
                {feedback && (
                  <div
                    className={`mt-3 p-2 rounded-lg text-[11px] flex items-start gap-1.5 ${
                      feedback.type === 'success'
                        ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                        : 'bg-rose-50 text-rose-800 border border-rose-200'
                    }`}
                  >
                    {feedback.type === 'success' ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5 text-rose-600 shrink-0 mt-0.5" />
                    )}
                    <span className="leading-tight">{feedback.message}</span>
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center justify-end gap-2 mt-4 pt-3 border-t border-stone-100">
                <button
                  type="button"
                  onClick={() => handleTest(p.id, p)}
                  disabled={isTesting || (!p.configured && !form.apiKey.trim())}
                  className="px-2.5 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-semibold flex items-center gap-1.5 disabled:opacity-50 cursor-pointer transition-colors"
                >
                  {isTesting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Radio className="w-3 h-3" />}
                  <span>Test Connection</span>
                </button>

                <button
                  type="button"
                  onClick={() => handleSave(p.id, p)}
                  disabled={isSaving}
                  className="px-3 py-1.5 rounded-lg bg-[#8D4B00] hover:bg-[#743D00] text-white text-xs font-bold flex items-center gap-1.5 disabled:opacity-50 cursor-pointer shadow-2xs transition-colors"
                >
                  {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                  <span>Save</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
