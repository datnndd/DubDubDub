import React, { useState, useRef, useEffect } from 'react';
import { useDubDubStore } from '../store';
import {
  X,
  Mic,
  Upload,
  Square,
  Play,
  Pause,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Info,
  Loader2,
  Sliders,
  Volume2,
  Wand2,
  Scissors,
  RotateCcw,
  Headphones,
} from 'lucide-react';
import { requestVoiceDesignAssist, previewTTS, trimAudio, previewCloneVoice } from '../api/voices';
import { ProviderDocLink } from './ProviderDocLink';

const VIENEU_PRESET_VOICES = [
  { id: 'Bình (nam miền Bắc)', name: 'Bình', region: 'Nam Bắc', gender: 'Nam', desc: 'Truyền cảm, ấm áp' },
  { id: 'Ly (nữ miền Bắc)', name: 'Ly', region: 'Nữ Bắc', gender: 'Nữ', desc: 'Tự nhiên, nhẹ nhàng' },
  { id: 'Ngọc (nữ miền Bắc)', name: 'Ngọc', region: 'Nữ Bắc', gender: 'Nữ', desc: 'Trong trẻo, rõ nét' },
  { id: 'Tuyên (nam miền Bắc)', name: 'Tuyên', region: 'Nam Bắc', gender: 'Nam', desc: 'Trầm ấm, đĩnh đạc' },
  { id: 'Vĩnh (nam miền Nam)', name: 'Vĩnh', region: 'Nam Nam', gender: 'Nam', desc: 'Tự nhiên, phóng khoáng' },
  { id: 'Đoan (nữ miền Nam)', name: 'Đoan', region: 'Nữ Nam', gender: 'Nữ', desc: 'Dịu dàng, ngọt ngào' },
];

export function probeAudioElementDuration(
  url: string,
  fallbackDur: number,
  onDuration: (dur: number) => void,
  AudioCtor: { new (src?: string): HTMLAudioElement } = Audio
): HTMLAudioElement {
  let settled = false;
  const finish = (rawDur: number) => {
    if (settled) return;
    settled = true;
    const dur = Number.isFinite(rawDur) && rawDur > 0 ? rawDur : fallbackDur;
    onDuration(dur);
  };

  const tempAudio = new AudioCtor(url);
  tempAudio.onloadedmetadata = () => {
    if (Number.isFinite(tempAudio.duration) && tempAudio.duration > 0) {
      finish(tempAudio.duration);
      return;
    }
    const onTimeUpdate = () => {
      tempAudio.removeEventListener?.('timeupdate', onTimeUpdate);
      const probed = tempAudio.duration;
      try {
        tempAudio.currentTime = 0;
      } catch {
        // ignore seek reset failure
      }
      finish(probed);
    };
    tempAudio.addEventListener?.('timeupdate', onTimeUpdate);
    try {
      tempAudio.currentTime = 1e101;
      setTimeout(() => {
        tempAudio.removeEventListener?.('timeupdate', onTimeUpdate);
        finish(tempAudio.duration);
      }, 150);
    } catch {
      tempAudio.removeEventListener?.('timeupdate', onTimeUpdate);
      finish(fallbackDur);
    }
  };
  tempAudio.onerror = () => {
    finish(fallbackDur);
  };
  return tempAudio;
}

export function resolveCloneAudioUploadSource(params: {
  isTrimmed: boolean;
  isTrimming: boolean;
  audioFile: File | null;
  audioBlob: Blob | null;
  rawAudioFile: File | null;
  rawAudioBlob: Blob | null;
}): { sourceAudio: File | Blob | null; useCutRange: boolean } {
  const { isTrimmed, isTrimming, audioFile, audioBlob, rawAudioFile, rawAudioBlob } = params;
  const hasRawSource = Boolean(rawAudioFile || rawAudioBlob);
  const hasDistinctTrimmedAudio =
    isTrimmed &&
    Boolean(audioFile || audioBlob) &&
    (audioBlob !== rawAudioBlob || audioFile !== rawAudioFile);
  const useCutRange = isTrimming || (isTrimmed && hasRawSource && !hasDistinctTrimmedAudio);
  const sourceAudio = hasDistinctTrimmedAudio
    ? (audioFile || audioBlob)
    : useCutRange
    ? (rawAudioFile || rawAudioBlob)
    : (audioFile || audioBlob);
  return { sourceAudio, useCutRange };
}

export const CreateVoiceModal: React.FC = () => {
  const isOpen = useDubDubStore((s) => s.isCreateVoiceModalOpen);
  const setIsOpen = useDubDubStore((s) => s.setCreateVoiceModalOpen);
  const activeTtsType = useDubDubStore((s) => s.backend?.config?.ttsType ?? 2);
  const createVoice = useDubDubStore((s) => s.createVoice);
  const customVoices = useDubDubStore((s) => s.customVoices || []);

  // Creation mode
  const [creationMode, setCreationMode] = useState<'clone' | 'design'>('clone');

  // Common voice attributes
  const [name, setName] = useState('');
  const [provider, setProvider] = useState<number>(activeTtsType);
  const [description, setDescription] = useState('');
  const [language, setLanguage] = useState('vi');
  const [refText, setRefText] = useState('');
  const [instruct, setInstruct] = useState('');

  // Cloning specific
  const [denoise, setDenoise] = useState(true);
  const [sourceMode, setSourceMode] = useState<'upload' | 'record'>('upload');
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const recordingSecondsRef = useRef<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordingTimerRef = useRef<number | null>(null);
  const isOpenRef = useRef<boolean>(isOpen);
  const isMountedRef = useRef<boolean>(true);
  isOpenRef.current = isOpen;

  // Trimming specific
  const [rawAudioFile, setRawAudioFile] = useState<File | null>(null);
  const [rawAudioBlob, setRawAudioBlob] = useState<Blob | null>(null);
  const [rawAudioUrl, setRawAudioUrl] = useState<string | null>(null);
  const [rawAudioDuration, setRawAudioDuration] = useState<number | null>(null);
  const [isTrimming, setIsTrimming] = useState(false);
  const [trimStart, setTrimStart] = useState(0.0);
  const [trimEnd, setTrimEnd] = useState(10.0);
  const [isTrimmed, setIsTrimmed] = useState(false);
  const [isAutoTrimming, setIsAutoTrimming] = useState(false);
  const [isAuditioningTrim, setIsAuditioningTrim] = useState(false);

  // Design specific
  const [baseVoice, setBaseVoice] = useState('Bình (nam miền Bắc)');
  const [style, setStyle] = useState<'tu_nhien' | 'tin_tuc' | 'doc_truyen'>('tu_nhien');
  const [temperature, setTemperature] = useState<number>(0.8);
  const [repetitionPenalty, setRepetitionPenalty] = useState<number>(1.2);
  const [silenceP, setSilenceP] = useState<number>(0.15);
  const [designPrompt, setDesignPrompt] = useState('');
  const [isAssisting, setIsAssisting] = useState(false);
  const [previewText, setPreviewText] = useState('Xin chào, đây là bản nghe thử phong cách giọng nói vừa được thiết kế.');

  // Playback & audition
  const [isPlaying, setIsPlaying] = useState(false);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const [isAuditioning, setIsAuditioning] = useState(false);
  const [auditionAudioUrl, setAuditionAudioUrl] = useState<string | null>(null);
  const [isAuditionPlaying, setIsAuditionPlaying] = useState(false);
  const auditionAudioRef = useRef<HTMLAudioElement | null>(null);

  // Clone Voice Preview before saving
  const [clonePreviewText, setClonePreviewText] = useState('Xin chào, đây là bản nghe thử giọng nói được sao chép.');
  const [isCloneAuditioning, setIsCloneAuditioning] = useState(false);
  const [cloneAuditionUrl, setCloneAuditionUrl] = useState<string | null>(null);
  const [clonePreviewFilename, setClonePreviewFilename] = useState<string | null>(null);
  const [isCloneAuditionPlaying, setIsCloneAuditionPlaying] = useState(false);
  const cloneAuditionAudioRef = useRef<HTMLAudioElement | null>(null);

  // Status
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const cleanupRecording = () => {
    if (recordingTimerRef.current !== null) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      try {
        recorder.stop();
      } catch {
        // ignore stop errors on already-closed recorder
      }
    }
    mediaRecorderRef.current = null;
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore
        }
      });
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
    recordingSecondsRef.current = 0;
    setRecordingSeconds(0);
  };

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      cleanupRecording();
    };
  }, []);

  useEffect(() => {
    if (isOpen) {
      setCreationMode('clone');
      setProvider(activeTtsType);
      setName('');
      setDescription('');
      setLanguage('vi');
      setRefText('');
      setInstruct('');
      setDenoise(true);
      setAudioBlob(null);
      setAudioFile(null);
      setAudioUrl(null);
      setAudioDuration(null);
      setRawAudioFile(null);
      setRawAudioBlob(null);
      setRawAudioUrl(null);
      setRawAudioDuration(null);
      setIsTrimming(false);
      setTrimStart(0.0);
      setTrimEnd(10.0);
      setIsTrimmed(false);
      setIsAutoTrimming(false);
      setIsAuditioningTrim(false);
      setErrorMsg(null);
      setSourceMode('upload');
      setBaseVoice('Bình (nam miền Bắc)');
      setStyle('tu_nhien');
      setTemperature(0.8);
      setRepetitionPenalty(1.2);
      setSilenceP(0.15);
      setDesignPrompt('');
      setPreviewText('Xin chào, đây là bản nghe thử phong cách giọng nói vừa được thiết kế.');
      setAuditionAudioUrl(null);
      setIsAuditionPlaying(false);
      setClonePreviewText('Xin chào, đây là bản nghe thử giọng nói được sao chép.');
      setIsCloneAuditioning(false);
      setCloneAuditionUrl(null);
      setClonePreviewFilename(null);
      setIsCloneAuditionPlaying(false);
    } else {
      cleanupRecording();
      if (audioPlayerRef.current) audioPlayerRef.current.pause();
      if (auditionAudioRef.current) auditionAudioRef.current.pause();
      if (cloneAuditionAudioRef.current) cloneAuditionAudioRef.current.pause();
    }
  }, [isOpen, activeTtsType]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (rawAudioUrl && rawAudioUrl !== audioUrl) URL.revokeObjectURL(rawAudioUrl);
      if (auditionAudioUrl && auditionAudioUrl.startsWith('blob:')) URL.revokeObjectURL(auditionAudioUrl);
      if (cloneAuditionUrl && cloneAuditionUrl.startsWith('blob:')) URL.revokeObjectURL(cloneAuditionUrl);
    };
  }, [audioUrl, rawAudioUrl, auditionAudioUrl, cloneAuditionUrl]);

  if (!isOpen) return null;

  const handleFileChange = (file: File) => {
    setErrorMsg(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    if (rawAudioUrl && rawAudioUrl !== audioUrl) URL.revokeObjectURL(rawAudioUrl);

    const url = URL.createObjectURL(file);
    setAudioFile(file);
    setAudioBlob(file);
    setAudioUrl(url);
    setRawAudioFile(file);
    setRawAudioBlob(file);
    setRawAudioUrl(url);
    setIsTrimmed(false);

    const applyProbedDuration = (dur: number) => {
      setAudioDuration(dur);
      setRawAudioDuration(dur);
      if (dur < 2.0) {
        setErrorMsg(`Audio duration (${dur.toFixed(1)}s) is too short. Minimum duration is 2.0s.`);
        setIsTrimming(false);
      } else if (dur > 30.0) {
        setIsTrimming(true);
        setTrimStart(0.0);
        setTrimEnd(Math.min(10.0, dur));
      } else {
        setIsTrimming(false);
        setTrimStart(0.0);
        setTrimEnd(Math.min(10.0, dur));
      }
    };

    probeAudioElementDuration(url, 5.0, applyProbedDuration);
  };

  const startRecording = async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!isOpenRef.current || !isMountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      mediaStreamRef.current = stream;
      const supportedTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
      const mimeType = supportedTypes.find((t) => typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported?.(t)) || '';
      const mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      const chunks: BlobPart[] = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (mediaStreamRef.current === stream) {
          mediaStreamRef.current = null;
        }
        if (!isOpenRef.current || !isMountedRef.current) {
          return;
        }
        const fallbackRecDur = recordingSecondsRef.current > 0 ? recordingSecondsRef.current : 5.0;
        const actualType = mediaRecorder.mimeType || mimeType || 'audio/webm';
        const blob = new Blob(chunks, { type: actualType });
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        if (rawAudioUrl && rawAudioUrl !== audioUrl) URL.revokeObjectURL(rawAudioUrl);
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioFile(null);
        setAudioUrl(url);
        setRawAudioBlob(blob);
        setRawAudioFile(null);
        setRawAudioUrl(url);
        setIsTrimmed(false);

        const applyRecDuration = (dur: number) => {
          setAudioDuration(dur);
          setRawAudioDuration(dur);
          if (dur < 2.0) {
            setErrorMsg(`Recording (${dur.toFixed(1)}s) is too short. Speak for at least 2.0 seconds.`);
            setIsTrimming(false);
          } else if (dur > 30.0) {
            setIsTrimming(true);
            setTrimStart(0.0);
            setTrimEnd(Math.min(10.0, dur));
          } else {
            setIsTrimming(false);
            setTrimStart(0.0);
            setTrimEnd(Math.min(10.0, dur));
          }
        };

        probeAudioElementDuration(url, fallbackRecDur, applyRecDuration);
      };

      mediaRecorder.start(250);
      setIsRecording(true);
      recordingSecondsRef.current = 0;
      setRecordingSeconds(0);

      recordingTimerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 29) {
            recordingSecondsRef.current = 30;
            stopRecording();
            return 30;
          }
          const next = prev + 1;
          recordingSecondsRef.current = next;
          return next;
        });
      }, 1000);
    } catch (err: any) {
      setErrorMsg(`Microphone access error: ${err.message || 'Permission denied'}`);
    }
  };

  const stopRecording = () => {
    if (recordingTimerRef.current !== null) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
  };

  const handleAutoDetectSpeech = async () => {
    const sourceAudio = rawAudioFile || rawAudioBlob || audioBlob;
    if (!sourceAudio) return;
    setIsAutoTrimming(true);
    setErrorMsg(null);
    try {
      const res = await trimAudio({
        audio: sourceAudio,
        auto_detect: true,
        target_duration: 10.0,
        provider,
      });
      if (res && res.ok) {
        setTrimStart(res.start_time);
        setTrimEnd(res.end_time);
      }
    } catch {
      const maxDur = rawAudioDuration || audioDuration || 30.0;
      setTrimStart(0.0);
      setTrimEnd(Math.min(10.0, maxDur));
    } finally {
      setIsAutoTrimming(false);
    }
  };

  const handleAuditionTrim = () => {
    if (!audioPlayerRef.current) return;
    const player = audioPlayerRef.current;
    if (isAuditioningTrim) {
      player.pause();
      setIsAuditioningTrim(false);
      return;
    }

    player.currentTime = trimStart;
    player.play().catch(() => {});
    setIsAuditioningTrim(true);

    const onTimeUpdate = () => {
      if (player.currentTime >= trimEnd || player.paused) {
        player.pause();
        setIsAuditioningTrim(false);
        player.removeEventListener('timeupdate', onTimeUpdate);
      }
    };
    player.addEventListener('timeupdate', onTimeUpdate);
  };

  const handleApplyTrim = async () => {
    const sourceAudio = rawAudioFile || rawAudioBlob || audioBlob;
    if (!sourceAudio) return;
    const dur = trimEnd - trimStart;
    if (dur < 2.0) {
      setErrorMsg(`Trimmed selection (${dur.toFixed(1)}s) is too short. Minimum duration is 2.0s.`);
      return;
    }
    if (dur > 30.0) {
      setErrorMsg(`Trimmed selection (${dur.toFixed(1)}s) exceeds 30.0s. Please narrow the selection.`);
      return;
    }

    setIsAutoTrimming(true);
    setErrorMsg(null);
    try {
      const res = await trimAudio({
        audio: sourceAudio,
        start_time: trimStart,
        end_time: trimEnd,
        provider,
      });
      if (res && res.ok) {
        setAudioDuration(res.duration);
        setAudioUrl(res.audio_url);
        try {
          const fetchRes = await fetch(res.audio_url);
          const blob = await fetchRes.blob();
          setAudioBlob(blob);
          setAudioFile(new File([blob], res.filename, { type: 'audio/wav' }));
        } catch {
          // Keep blob intact in test environments
        }
        setIsTrimmed(true);
        setIsTrimming(false);
      }
    } catch {
      // Offline/fallback handling
      setAudioDuration(dur);
      setIsTrimmed(true);
      setIsTrimming(false);
    } finally {
      setIsAutoTrimming(false);
    }
  };

  const handleResetTrim = () => {
    if (rawAudioUrl) {
      setAudioUrl(rawAudioUrl);
      setAudioFile(rawAudioFile);
      setAudioBlob(rawAudioBlob);
      setAudioDuration(rawAudioDuration);
    }
    setIsTrimmed(false);
    if (rawAudioDuration && rawAudioDuration > 30.0) {
      setIsTrimming(true);
      setTrimStart(0.0);
      setTrimEnd(Math.min(10.0, rawAudioDuration));
    } else {
      setIsTrimming(false);
    }
  };

  const togglePlayback = () => {
    if (!audioPlayerRef.current) return;
    if (isPlaying) {
      audioPlayerRef.current.pause();
      setIsPlaying(false);
    } else {
      audioPlayerRef.current.play();
      setIsPlaying(true);
    }
  };

  const insertEmotionTag = (tag: string) => {
    setPreviewText((prev) => `${prev.trim()} ${tag} `);
  };

  const handleApplyDesignAssist = async () => {
    if (!designPrompt.trim()) return;
    setIsAssisting(true);
    setErrorMsg(null);
    try {
      const res = await requestVoiceDesignAssist({ prompt: designPrompt });
      if (res.style === 'tin_tuc' || res.style === 'doc_truyen' || res.style === 'tu_nhien') {
        setStyle(res.style);
      }
      if (res.temperature) setTemperature(res.temperature);
      if (res.repetition_penalty) setRepetitionPenalty(res.repetition_penalty);
      if (!name.trim() && res.suggested_name) setName(res.suggested_name);
      if (res.description) setDescription(res.description);
      if (res.suggested_tags && res.suggested_tags.length > 0) {
        setPreviewText((prev) => `${prev.trim()} ${res.suggested_tags[0]}`);
      }
    } catch (err: any) {
      setErrorMsg(`AI Assistant error: ${err.message || 'Failed to analyze prompt'}`);
    } finally {
      setIsAssisting(false);
    }
  };

  const handleAuditionDesign = async () => {
    if (!previewText.trim()) return;
    setIsAuditioning(true);
    setErrorMsg(null);
    try {
      const res = await previewTTS({
        voice: baseVoice,
        provider: 2,
        language,
        text: previewText,
        style,
        tuningParams: {
          style,
          temperature,
          repetition_penalty: repetitionPenalty,
          silence_p: silenceP,
        },
      });
      if (res.audio_url || res.preview_url) {
        const rawUrl = res.audio_url || res.preview_url;
        const freshUrl = rawUrl.includes('?') ? `${rawUrl}&_t=${Date.now()}` : `${rawUrl}?_t=${Date.now()}`;
        setAuditionAudioUrl(freshUrl);
        setIsAuditionPlaying(true);
        if (auditionAudioRef.current) {
          auditionAudioRef.current.src = freshUrl;
          auditionAudioRef.current.currentTime = 0;
          auditionAudioRef.current.muted = false;
          if (auditionAudioRef.current.volume === 0) {
            auditionAudioRef.current.volume = 1.0;
          }
          auditionAudioRef.current.load();
          const p = auditionAudioRef.current.play();
          if (p !== undefined) {
            p.then(() => setIsAuditionPlaying(true)).catch((err) => {
              console.warn('Autoplay prevented:', err);
              setIsAuditionPlaying(false);
            });
          }
        }
      }
    } catch (err: any) {
      setErrorMsg(`Audition synthesis error: ${err.message || 'Failed to synthesize sample'}`);
    } finally {
      setIsAuditioning(false);
    }
  };

  const handleAuditionClone = async () => {
    const { sourceAudio, useCutRange } = resolveCloneAudioUploadSource({
      isTrimmed,
      isTrimming,
      audioFile,
      audioBlob,
      rawAudioFile,
      rawAudioBlob,
    });

    if (!sourceAudio) {
      setErrorMsg('Please upload or record reference audio first.');
      return;
    }
    if (!clonePreviewText.trim()) {
      setErrorMsg('Please enter a test phrase to preview the voice.');
      return;
    }
    if (provider === 1 && !refText.trim()) {
      setErrorMsg('OmniVoice requires reference text transcript to prevent acoustic hallucinations.');
      return;
    }
    setIsCloneAuditioning(true);
    setErrorMsg(null);

    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      setIsPlaying(false);
    }
    if (auditionAudioRef.current) {
      auditionAudioRef.current.pause();
      setIsAuditionPlaying(false);
    }

    try {
      const res = await previewCloneVoice({
        audio: sourceAudio,
        provider,
        text: clonePreviewText.trim(),
        language,
        denoise,
        cut_start: useCutRange ? trimStart : undefined,
        cut_end: useCutRange ? trimEnd : undefined,
        ref_text: refText.trim() || undefined,
      });
      if (res && (res.preview_url || res.audio_url)) {
        const rawUrl = res.preview_url || res.audio_url;
        const freshUrl = rawUrl.includes('?') ? `${rawUrl}&_t=${Date.now()}` : `${rawUrl}?_t=${Date.now()}`;
        setCloneAuditionUrl(freshUrl);
        setClonePreviewFilename(res.preview_filename);
        setIsCloneAuditionPlaying(true);
        if (cloneAuditionAudioRef.current) {
          cloneAuditionAudioRef.current.src = freshUrl;
          cloneAuditionAudioRef.current.currentTime = 0;
          cloneAuditionAudioRef.current.muted = false;
          if (cloneAuditionAudioRef.current.volume === 0) {
            cloneAuditionAudioRef.current.volume = 1.0;
          }
          cloneAuditionAudioRef.current.load();
          const p = cloneAuditionAudioRef.current.play();
          if (p !== undefined) {
            p.then(() => setIsCloneAuditionPlaying(true)).catch((err) => {
              console.warn('Autoplay prevented:', err);
              setIsCloneAuditionPlaying(false);
            });
          }
        }
      }
    } catch (err: any) {
      setErrorMsg(`Preview voice error: ${err.message || 'Failed to synthesize cloned voice preview'}`);
    } finally {
      setIsCloneAuditioning(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);

    const cleanName = name.trim();
    if (!cleanName) {
      setErrorMsg('Please enter a voice name.');
      return;
    }

    if (creationMode === 'clone') {
      if (!audioBlob) {
        setErrorMsg('Please upload or record reference audio for voice cloning.');
        return;
      }

      const effectiveDuration = isTrimmed
        ? (audioDuration ?? 10)
        : isTrimming
        ? trimEnd - trimStart
        : (audioDuration ?? 0);
      if (effectiveDuration < 2.0 || effectiveDuration > 30.0) {
        setErrorMsg(`Audio duration must be between 2.0s and 30.0s (currently ${effectiveDuration.toFixed(1)}s).`);
        return;
      }

      if (provider === 1 && !refText.trim()) {
        setErrorMsg('OmniVoice requires reference text transcript to prevent acoustic hallucinations.');
        return;
      }

      setIsSubmitting(true);
      try {
        const { sourceAudio, useCutRange } = resolveCloneAudioUploadSource({
          isTrimmed,
          isTrimming,
          audioFile,
          audioBlob,
          rawAudioFile,
          rawAudioBlob,
        });

        const payload: any = {
          name: cleanName,
          provider,
          description: description.trim(),
          language,
          kind: 'clone',
          ref_text: refText.trim(),
          instruct: instruct.trim(),
          audio: sourceAudio || undefined,
          denoise,
          tuning_params: { denoise },
          preview_filename: clonePreviewFilename || undefined,
        };
        if (useCutRange) {
          payload.cut_start = trimStart;
          payload.cut_end = trimEnd;
        }
        await createVoice(payload);
        setIsOpen(false);
      } catch (err: any) {
        setErrorMsg(err.message || 'Failed to clone voice.');
      } finally {
        setIsSubmitting(false);
      }
    } else {
      // Voice Design
      setIsSubmitting(true);
      try {
        await createVoice({
          name: cleanName,
          provider: 2, // VieNeu-TTS
          description: description.trim() || `Thiết kế từ ${baseVoice} (${style})`,
          language,
          kind: 'design',
          external_voice_id: baseVoice,
          tuning_params: {
            base_voice: baseVoice,
            style,
            temperature,
            repetition_penalty: repetitionPenalty,
            silence_p: silenceP,
          },
        });
        setIsOpen(false);
      } catch (err: any) {
        setErrorMsg(err.message || 'Failed to design voice.');
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  const trimDuration = Math.max(0, trimEnd - trimStart);
  const isTrimValid = trimDuration >= 2.0 && trimDuration <= 30.0;
  const isDurationValid = audioDuration !== null && audioDuration >= 2.0 && audioDuration <= 30.0;
  const canSubmitClone = isTrimmed ? isDurationValid : (isTrimming ? isTrimValid : isDurationValid);

  return (
    <div
      data-modal="create-voice"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-xs p-4 overflow-y-auto"
      onClick={(e) => {
        if (e.target === e.currentTarget && !isSubmitting) setIsOpen(false);
      }}
    >
      <div className="w-full max-w-lg bg-[#FAF9F6] rounded-2xl border border-[#E7E4DC] shadow-2xl overflow-hidden flex flex-col my-auto animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="h-14 px-5 border-b border-[#E7E4DC] flex items-center justify-between bg-white shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-100 flex items-center justify-center text-[#8D4B00]">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-stone-900">
                {creationMode === 'clone' ? 'Clone Custom Voice' : 'Design New Voice Persona'}
              </h3>
              <p className="text-[11px] text-stone-500">
                {creationMode === 'clone'
                  ? 'Replicate a real voice from 3-15 seconds of audio'
                  : 'Customize speaking style, emotions, and prosody with VieNeu-TTS'}
              </p>
            </div>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            disabled={isSubmitting}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Switcher: Voice Cloning vs Voice Design */}
        <div className="grid grid-cols-2 p-1.5 bg-stone-100 border-b border-[#E7E4DC] text-xs font-bold">
          <button
            type="button"
            onClick={() => setCreationMode('clone')}
            className={`py-2 px-3 rounded-xl flex items-center justify-center gap-1.5 transition-all ${
              creationMode === 'clone'
                ? 'bg-white text-stone-900 shadow-xs'
                : 'text-stone-500 hover:text-stone-800'
            }`}
          >
            <Mic className="w-3.5 h-3.5 text-[#8D4B00]" />
            Voice Cloning
          </button>
          <button
            type="button"
            onClick={() => setCreationMode('design')}
            className={`py-2 px-3 rounded-xl flex items-center justify-center gap-1.5 transition-all ${
              creationMode === 'design'
                ? 'bg-white text-stone-900 shadow-xs'
                : 'text-stone-500 hover:text-stone-800'
            }`}
          >
            <Sliders className="w-3.5 h-3.5 text-[#8D4B00]" />
            Voice Design
            <span className="px-1.5 py-0.2 rounded-full text-[9px] bg-amber-100 text-[#8D4B00]">VieNeu</span>
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4 max-h-[75vh] overflow-y-auto">
          {errorMsg && (
            <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="flex-1 font-medium">{errorMsg}</div>
            </div>
          )}

          {/* Common: Voice Name & Language Row */}
          <div className="grid grid-cols-12 gap-3">
            <div className="col-span-8 space-y-1.5">
              <label className="block text-xs font-bold text-stone-700">
                Voice Name <span className="text-rose-500">*</span>
              </label>
              <input
                type="text"
                required
                placeholder={creationMode === 'clone' ? 'e.g. Minh Thư (Studio)' : 'e.g. Hoàng Long (Storyteller)'}
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full text-xs px-3 py-2 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
              />
            </div>

            <div className="col-span-4 space-y-1.5">
              <label className="block text-xs font-bold text-stone-700">Language</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full text-xs px-3 py-2 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
              >
                <option value="vi">Vietnamese (vi)</option>
                <option value="en">English (en)</option>
                <option value="Auto">Auto Detect</option>
              </select>
            </div>
          </div>

          {/* ============================================================== */}
          {/* TAB 1: VOICE CLONING                                           */}
          {/* ============================================================== */}
          {creationMode === 'clone' && (
            <>
              {/* TTS Provider */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="block text-xs font-bold text-stone-700">Cloning Model Backbone</label>
                  <ProviderDocLink providerId={provider} category="tts" />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 2, label: 'VieNeu-TTS', desc: '48kHz Turbo (Recommended)' },
                    { id: 1, label: 'OmniVoice', desc: '24kHz Built-in' },
                    { id: 0, label: 'ElevenLabs', desc: 'Cloud Instant' },
                  ].map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      onClick={() => setProvider(item.id)}
                      className={`p-2.5 rounded-xl border text-left transition-all ${
                        provider === item.id
                          ? 'border-[#8D4B00] bg-amber-50/60 ring-1 ring-[#8D4B00]'
                          : 'border-stone-200 bg-white hover:border-stone-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="font-bold text-xs text-stone-900">{item.label}</div>
                        <ProviderDocLink providerId={item.id} category="tts" variant="icon" />
                      </div>
                      <div className="text-[10px] text-stone-500">{item.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Denoise Toggle for VieNeu */}
              {provider === 2 && (
                <div className="flex items-center justify-between p-2.5 bg-amber-50/40 rounded-xl border border-amber-200/50">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="denoise-toggle"
                      checked={denoise}
                      onChange={(e) => setDenoise(e.target.checked)}
                      className="w-4 h-4 rounded text-[#8D4B00] focus:ring-[#8D4B00] border-stone-300 cursor-pointer"
                    />
                    <label htmlFor="denoise-toggle" className="text-xs font-bold text-stone-800 cursor-pointer">
                      ONNX Auto-Denoise
                    </label>
                  </div>
                  <span className="text-[10px] text-stone-500">Cleans room reverb and background hiss</span>
                </div>
              )}

              {/* Audio Upload or Record */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="block text-xs font-bold text-stone-700">
                    Reference Audio Sample <span className="text-rose-500">*</span>
                  </label>
                  <div className="flex items-center gap-1 bg-stone-100 p-0.5 rounded-lg border border-stone-200">
                    <button
                      type="button"
                      onClick={() => setSourceMode('upload')}
                      className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all ${
                        sourceMode === 'upload' ? 'bg-white shadow-xs text-stone-900' : 'text-stone-500'
                      }`}
                    >
                      <Upload className="w-3 h-3 inline mr-1" />
                      Upload File
                    </button>
                    <button
                      type="button"
                      onClick={() => setSourceMode('record')}
                      className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all ${
                        sourceMode === 'record' ? 'bg-white shadow-xs text-stone-900' : 'text-stone-500'
                      }`}
                    >
                      <Mic className="w-3 h-3 inline mr-1" />
                      Record Mic
                    </button>
                  </div>
                </div>

                {sourceMode === 'upload' && (
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (e.dataTransfer.files?.[0]) handleFileChange(e.dataTransfer.files[0]);
                    }}
                    className={`border-2 border-dashed rounded-xl p-5 text-center transition-all ${
                      audioBlob ? 'border-emerald-300 bg-emerald-50/20' : 'border-stone-300 bg-white hover:border-amber-400'
                    }`}
                  >
                    <input
                      type="file"
                      id="voice-audio-file"
                      accept="audio/wav,audio/mp3,audio/mpeg,audio/m4a,audio/webm,audio/ogg,audio/flac,.wav,.mp3,.m4a,.webm,.ogg,.flac"
                      className="hidden"
                      onChange={(e) => {
                        if (e.target.files?.[0]) handleFileChange(e.target.files[0]);
                      }}
                    />
                    <label htmlFor="voice-audio-file" className="cursor-pointer block">
                      <div className="w-10 h-10 rounded-full bg-stone-100 flex items-center justify-center mx-auto mb-2 text-stone-500">
                        <Upload className="w-5 h-5" />
                      </div>
                      <div className="text-xs font-bold text-stone-800">
                        {audioFile ? audioFile.name : 'Click to select or drag audio here'}
                      </div>
                      <div className="text-[10px] text-stone-500 mt-1">
                        WAV, MP3, M4A, WebM (Optimal: 3-15 seconds, mono 48kHz / 24kHz)
                      </div>
                    </label>
                  </div>
                )}

                {sourceMode === 'record' && (
                  <div className="p-5 rounded-xl border border-stone-200 bg-white text-center space-y-3">
                    <div className="flex items-center justify-center gap-3">
                      {!isRecording ? (
                        <button
                          type="button"
                          onClick={startRecording}
                          className="px-4 py-2 rounded-xl bg-[#8D4B00] text-white font-bold text-xs flex items-center gap-2 hover:bg-[#733D00] shadow-sm transition-all"
                        >
                          <Mic className="w-4 h-4" />
                          Start Recording
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={stopRecording}
                          className="px-4 py-2 rounded-xl bg-rose-600 text-white font-bold text-xs flex items-center gap-2 animate-pulse shadow-sm"
                        >
                          <Square className="w-4 h-4 fill-white" />
                          Stop Recording ({recordingSeconds}s / 30s)
                        </button>
                      )}
                    </div>
                    <div className="text-[11px] text-stone-500">
                      {isRecording ? 'Speak clearly into your microphone...' : 'Record 3-10 seconds of clear speech.'}
                    </div>
                  </div>
                )}

                {audioUrl && (
                  <div className="space-y-2.5">
                    {rawAudioDuration !== null && rawAudioDuration > 30.0 && !isTrimmed && (
                      <div className="p-2.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <AlertCircle className="w-4 h-4 text-amber-700 shrink-0" />
                          <span>
                            Audio length (<strong>{rawAudioDuration.toFixed(1)}s</strong>) exceeds 30s limit. Please trim a voice segment below.
                          </span>
                        </div>
                        {!isTrimming && (
                          <button
                            type="button"
                            onClick={() => {
                              setIsTrimming(true);
                              setTrimStart(0.0);
                              setTrimEnd(Math.min(10.0, rawAudioDuration));
                            }}
                            className="px-2.5 py-1 rounded-lg bg-[#8D4B00] text-white text-[11px] font-bold shrink-0 hover:bg-[#733D00] shadow-2xs"
                          >
                            Trim Now
                          </button>
                        )}
                      </div>
                    )}

                    {isTrimming ? (
                      <div className="p-3.5 rounded-xl border border-amber-300/80 bg-amber-50/40 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5 text-xs font-bold text-stone-900">
                            <Scissors className="w-4 h-4 text-[#8D4B00]" />
                            <span>Trim Voice Selection</span>
                            {isTrimValid ? (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">
                                ✓ {trimDuration.toFixed(1)}s (Valid)
                              </span>
                            ) : trimDuration < 2.0 ? (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800">
                                Min 2.0s ({trimDuration.toFixed(1)}s)
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-800">
                                Max 30s ({trimDuration.toFixed(1)}s)
                              </span>
                            )}
                          </div>

                          <button
                            type="button"
                            onClick={handleAutoDetectSpeech}
                            disabled={isAutoTrimming}
                            className="text-[11px] font-bold text-[#8D4B00] hover:text-[#733D00] flex items-center gap-1 disabled:opacity-50"
                            title="Auto-detect optimal 10-second speech segment"
                          >
                            {isAutoTrimming ? (
                              <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Wand2 className="w-3.5 h-3.5" />
                            )}
                            Auto-Detect (10s)
                          </button>
                        </div>

                        {/* Timeline visual representation */}
                        <div className="relative h-6 bg-stone-200 rounded-lg overflow-hidden flex items-center">
                          <div
                            className="absolute h-full bg-amber-400/60 border-x-2 border-[#8D4B00]"
                            style={{
                              left: `${Math.max(0, (trimStart / (rawAudioDuration || audioDuration || 1)) * 100)}%`,
                              width: `${Math.min(100, (trimDuration / (rawAudioDuration || audioDuration || 1)) * 100)}%`,
                            }}
                          />
                          <div className="absolute inset-0 flex items-center justify-between px-2 text-[10px] text-stone-600 pointer-events-none font-mono">
                            <span>0.0s</span>
                            <span className="font-bold text-[#8D4B00] bg-white/90 px-1.5 py-0.5 rounded shadow-2xs">
                              Range: {trimStart.toFixed(1)}s - {trimEnd.toFixed(1)}s ({trimDuration.toFixed(1)}s)
                            </span>
                            <span>{(rawAudioDuration || audioDuration || 10).toFixed(1)}s</span>
                          </div>
                        </div>

                        {/* Dual range sliders */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px] text-stone-600 font-semibold">
                              <span>Start Time</span>
                              <span className="font-mono text-stone-900 font-bold">{trimStart.toFixed(1)}s</span>
                            </div>
                            <input
                              type="range"
                              min={0}
                              max={Math.max(0, (rawAudioDuration || audioDuration || 10) - 1.0)}
                              step={0.1}
                              value={trimStart}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                setTrimStart(val);
                                if (val >= trimEnd - 1.0) setTrimEnd(Math.min(rawAudioDuration || 30, val + 2.0));
                              }}
                              className="w-full accent-[#8D4B00] h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                            />
                          </div>

                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px] text-stone-600 font-semibold">
                              <span>End Time</span>
                              <span className="font-mono text-stone-900 font-bold">{trimEnd.toFixed(1)}s</span>
                            </div>
                            <input
                              type="range"
                              min={Math.min(rawAudioDuration || 30, trimStart + 1.0)}
                              max={rawAudioDuration || audioDuration || 10}
                              step={0.1}
                              value={trimEnd}
                              onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                setTrimEnd(val);
                                if (val <= trimStart + 1.0) setTrimStart(Math.max(0, val - 2.0));
                              }}
                              className="w-full accent-[#8D4B00] h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                            />
                          </div>
                        </div>

                        {/* Trimmer action row */}
                        <div className="flex items-center justify-between pt-1 border-t border-amber-200/60">
                          <button
                            type="button"
                            onClick={handleAuditionTrim}
                            className="px-3 py-1.5 rounded-lg border border-stone-300 bg-white hover:bg-stone-50 text-xs font-semibold text-stone-700 flex items-center gap-1.5 transition-colors"
                          >
                            {isAuditioningTrim ? (
                              <Pause className="w-3.5 h-3.5 text-stone-700" />
                            ) : (
                              <Play className="w-3.5 h-3.5 text-stone-700" />
                            )}
                            {isAuditioningTrim ? 'Stop Preview' : 'Audition Selection'}
                          </button>

                          <div className="flex items-center gap-2">
                            {(!rawAudioDuration || rawAudioDuration <= 30.0) && (
                              <button
                                type="button"
                                onClick={() => setIsTrimming(false)}
                                className="px-2.5 py-1.5 rounded-lg text-xs font-semibold text-stone-500 hover:text-stone-700"
                              >
                                Cancel
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={handleApplyTrim}
                              disabled={!isTrimValid || isAutoTrimming}
                              className="px-3.5 py-1.5 rounded-lg bg-[#8D4B00] hover:bg-[#733D00] text-white text-xs font-bold flex items-center gap-1.5 shadow-xs disabled:opacity-50"
                            >
                              {isAutoTrimming ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Scissors className="w-3.5 h-3.5" />
                              )}
                              Apply Trim ({trimDuration.toFixed(1)}s)
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="p-3 rounded-xl border border-stone-200 bg-white flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <button
                            type="button"
                            onClick={togglePlayback}
                            className="w-8 h-8 rounded-full bg-stone-100 hover:bg-stone-200 flex items-center justify-center text-stone-700 shrink-0"
                          >
                            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                          </button>
                          <div className="min-w-0">
                            <div className="text-xs font-bold text-stone-800 truncate">
                              {isTrimmed ? '✂️ Trimmed Voice Sample Ready' : 'Reference Sample Ready'}
                            </div>
                            <div className="text-[10px] text-stone-500">
                              {isTrimmed
                                ? `Duration: ${audioDuration !== null ? `${audioDuration.toFixed(1)}s` : '...'} (${trimStart.toFixed(1)}s - ${trimEnd.toFixed(1)}s)`
                                : `Duration: ${audioDuration !== null ? `${audioDuration.toFixed(1)}s` : 'Analyzing...'}`}
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          {isTrimmed ? (
                            <>
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 flex items-center gap-1">
                                <CheckCircle2 className="w-3 h-3" /> Ready
                              </span>
                              <button
                                type="button"
                                onClick={() => setIsTrimming(true)}
                                className="px-2 py-1 rounded-md text-[11px] font-semibold text-[#8D4B00] hover:bg-amber-50"
                              >
                                Edit Trim
                              </button>
                              <button
                                type="button"
                                onClick={handleResetTrim}
                                className="p-1 text-stone-400 hover:text-stone-600 rounded-md"
                                title="Reset to original audio"
                              >
                                <RotateCcw className="w-3.5 h-3.5" />
                              </button>
                            </>
                          ) : (
                            <>
                              {isDurationValid ? (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 flex items-center gap-1">
                                  <CheckCircle2 className="w-3 h-3" /> Valid Duration
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800 flex items-center gap-1">
                                  <AlertCircle className="w-3 h-3" /> Must be 2-30s
                                </span>
                              )}
                              <button
                                type="button"
                                onClick={() => {
                                  setIsTrimming(true);
                                  setTrimStart(0.0);
                                  setTrimEnd(Math.min(10.0, audioDuration || 10.0));
                                }}
                                className="px-2 py-1 rounded-md border border-stone-200 bg-white hover:bg-stone-50 text-[11px] font-semibold text-stone-700 flex items-center gap-1"
                              >
                                <Scissors className="w-3 h-3 text-[#8D4B00]" />
                                Trim Audio
                              </button>
                            </>
                          )}
                        </div>

                        <audio
                          ref={audioPlayerRef}
                          src={audioUrl}
                          onEnded={() => {
                            setIsPlaying(false);
                            setIsAuditioningTrim(false);
                          }}
                          className="hidden"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Reference Audio Transcript for OmniVoice */}
              {provider === 1 && (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <label className="block text-xs font-bold text-stone-700">
                      Reference Audio Transcript <span className="text-rose-500">*</span>
                    </label>
                    <span className="text-[10px] text-amber-700 font-semibold flex items-center gap-0.5">
                      <Info className="w-3 h-3" /> Prevents hallucinations
                    </span>
                  </div>
                  <textarea
                    rows={2}
                    placeholder="Exact words spoken in the reference audio sample..."
                    value={refText}
                    onChange={(e) => setRefText(e.target.value)}
                    className="w-full text-xs p-2.5 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
                  />
                </div>
              )}

              {/* Preview Cloned Voice before saving */}
              {(audioBlob || audioFile) && (
                <div className="space-y-2 p-3 bg-stone-50 rounded-xl border border-stone-200">
                  <div className="flex items-center justify-between">
                    <label className="block text-xs font-bold text-stone-700 flex items-center gap-1.5">
                      <Headphones className="w-3.5 h-3.5 text-[#8D4B00]" />
                      Preview Cloned Voice
                    </label>
                    <span className="text-[10px] text-stone-500">Listen before saving</span>
                  </div>
                  <textarea
                    rows={2}
                    placeholder="Enter phrase to preview cloned voice..."
                    value={clonePreviewText}
                    onChange={(e) => setClonePreviewText(e.target.value)}
                    className="w-full text-xs p-2 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
                  />
                  <div className="flex items-center justify-between pt-1">
                    <button
                      type="button"
                      data-action="preview-cloned-voice"
                      onClick={handleAuditionClone}
                      disabled={isCloneAuditioning || !clonePreviewText.trim() || !canSubmitClone}
                      className="px-3 py-1.5 rounded-lg bg-stone-800 hover:bg-stone-900 text-white text-xs font-bold flex items-center gap-1.5 shadow-xs disabled:opacity-50 cursor-pointer"
                      title="Preview Voice before saving"
                    >
                      {isCloneAuditioning ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          Synthesizing Cloned Voice...
                        </>
                      ) : (
                        <>
                          <Volume2 className="w-3.5 h-3.5" />
                          Preview Voice
                        </>
                      )}
                    </button>

                    <audio
                      ref={cloneAuditionAudioRef}
                      src={cloneAuditionUrl || undefined}
                      controls={Boolean(cloneAuditionUrl)}
                      className={cloneAuditionUrl ? "h-7 w-48" : "hidden"}
                      onPlay={() => setIsCloneAuditionPlaying(true)}
                      onPause={() => setIsCloneAuditionPlaying(false)}
                      onEnded={() => setIsCloneAuditionPlaying(false)}
                    />
                  </div>
                </div>
              )}
            </>
          )}

          {/* ============================================================== */}
          {/* TAB 2: VOICE DESIGN (VIENEU-TTS)                               */}
          {/* ============================================================== */}
          {creationMode === 'design' && (
            <>
              {/* AI Prompt Assistant */}
              <div className="p-3 rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200/70 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#8D4B00] flex items-center gap-1.5">
                    <Wand2 className="w-3.5 h-3.5" />
                    AI Prompt Assistant
                  </span>
                  <span className="text-[10px] text-stone-500">Auto-tunes styles & emotions</span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="e.g. Giọng kể chuyện đêm khuya truyền cảm có tiếng thở dài..."
                    value={designPrompt}
                    onChange={(e) => setDesignPrompt(e.target.value)}
                    className="flex-1 text-xs px-2.5 py-1.5 bg-white rounded-lg border border-amber-200 focus:border-[#8D4B00] outline-hidden"
                  />
                  <button
                    type="button"
                    onClick={handleApplyDesignAssist}
                    disabled={isAssisting || !designPrompt.trim()}
                    className="px-3 py-1.5 bg-[#8D4B00] hover:bg-[#733D00] text-white text-xs font-bold rounded-lg shadow-xs flex items-center gap-1 disabled:opacity-50"
                  >
                    {isAssisting ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Apply'}
                  </button>
                </div>
              </div>

              {/* Base Voice Selector */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-stone-700">Base Voice Timbre</label>
                <select
                  value={baseVoice}
                  onChange={(e) => setBaseVoice(e.target.value)}
                  className="w-full text-xs px-3 py-2 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
                >
                  <optgroup label="VieNeu Built-in Presets">
                    {VIENEU_PRESET_VOICES.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.name} — {v.region} ({v.desc})
                      </option>
                    ))}
                  </optgroup>
                  {customVoices.filter((v) => v.kind !== 'design').length > 0 && (
                    <optgroup label="Your Custom Cloned Voices">
                      {customVoices
                        .filter((v) => v.kind !== 'design')
                        .map((cv) => (
                          <option key={cv.id} value={cv.id}>
                            {cv.name} (Custom Cloned)
                          </option>
                        ))}
                    </optgroup>
                  )}
                </select>
              </div>

              {/* Speaking Style Cards */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-stone-700">Speaking Style</label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'tu_nhien', label: 'Tự nhiên', desc: 'Hội thoại đời thường, tự nhiên' },
                    { id: 'tin_tuc', label: 'Tin tức', desc: 'Trang trọng, dõng dạc, rõ ràng' },
                    { id: 'doc_truyen', label: 'Đọc truyện', desc: 'Diễn cảm, giàu cảm xúc drama' },
                  ].map((item) => (
                    <button
                      type="button"
                      key={item.id}
                      onClick={() => setStyle(item.id as any)}
                      className={`p-2.5 rounded-xl border text-left transition-all ${
                        style === item.id
                          ? 'border-[#8D4B00] bg-amber-50/70 ring-1 ring-[#8D4B00]'
                          : 'border-stone-200 bg-white hover:border-stone-300'
                      }`}
                    >
                      <div className="font-bold text-xs text-stone-900">{item.label}</div>
                      <div className="text-[10px] text-stone-500 leading-tight mt-0.5">{item.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Emotion Expression Palette */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="block text-xs font-bold text-stone-700">Non-verbal Emotion Cues</label>
                  <span className="text-[10px] text-stone-400">Click to insert into test phrase</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => insertEmotionTag('[cười]')}
                    className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition-colors"
                  >
                    + [cười] (Chuckle)
                  </button>
                  <button
                    type="button"
                    onClick={() => insertEmotionTag('[thở dài]')}
                    className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-sky-50 text-sky-700 border border-sky-200 hover:bg-sky-100 transition-colors"
                  >
                    + [thở dài] (Sigh)
                  </button>
                  <button
                    type="button"
                    onClick={() => insertEmotionTag('[hắng giọng]')}
                    className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-purple-50 text-purple-700 border border-purple-200 hover:bg-purple-100 transition-colors"
                  >
                    + [hắng giọng] (Throat clear)
                  </button>
                </div>
              </div>

              {/* Acoustic Tuning Sliders */}
              <div className="p-3 bg-white rounded-xl border border-stone-200 space-y-3">
                <div className="text-xs font-bold text-stone-800 flex items-center justify-between">
                  <span className="flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-stone-500" />
                    Acoustic Prosody Sliders
                  </span>
                </div>

                {/* Temperature */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-stone-600 font-medium">Expressiveness (Temperature)</span>
                    <span className="font-bold text-stone-900">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="1.2"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-[#8D4B00] cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-stone-400">
                    <span>Calm (0.5)</span>
                    <span>Standard (0.8)</span>
                    <span>High Dynamic (1.2)</span>
                  </div>
                </div>

                {/* Repetition penalty */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-stone-600 font-medium">Anti-Repetition Penalty</span>
                    <span className="font-bold text-stone-900">{repetitionPenalty.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="1.5"
                    step="0.05"
                    value={repetitionPenalty}
                    onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                    className="w-full accent-[#8D4B00] cursor-pointer"
                  />
                </div>
              </div>

              {/* Sample Audition Box */}
              <div className="space-y-2 p-3 bg-stone-50 rounded-xl border border-stone-200">
                <label className="block text-xs font-bold text-stone-700">Test Phrase & Live Audition</label>
                <textarea
                  rows={2}
                  value={previewText}
                  onChange={(e) => setPreviewText(e.target.value)}
                  className="w-full text-xs p-2 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
                />
                <div className="flex items-center justify-between pt-1">
                  <button
                    type="button"
                    onClick={handleAuditionDesign}
                    disabled={isAuditioning || !previewText.trim()}
                    className="px-3 py-1.5 rounded-lg bg-stone-800 hover:bg-stone-900 text-white text-xs font-bold flex items-center gap-1.5 shadow-xs disabled:opacity-50"
                  >
                    {isAuditioning ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Synthesizing...
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-3.5 h-3.5" />
                        Audition Designed Voice
                      </>
                    )}
                  </button>

                  <audio
                    ref={auditionAudioRef}
                    src={auditionAudioUrl || undefined}
                    controls={Boolean(auditionAudioUrl)}
                    className={auditionAudioUrl ? "h-7 w-48" : "hidden"}
                    onPlay={() => setIsAuditionPlaying(true)}
                    onPause={() => setIsAuditionPlaying(false)}
                    onEnded={() => setIsAuditionPlaying(false)}
                  />
                </div>
              </div>
            </>
          )}

          {/* Description */}
          <div className="space-y-1.5">
            <label className="block text-xs font-bold text-stone-700">
              Description <span className="text-stone-400 font-normal">(Optional)</span>
            </label>
            <input
              type="text"
              placeholder="e.g. Dành cho video tài liệu lịch sử hoặc kể chuyện"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full text-xs px-3 py-2 bg-white rounded-lg border border-[#E7E4DC] focus:border-[#8D4B00] outline-hidden font-medium"
            />
          </div>

          {/* Action Buttons */}
          <div className="pt-3 border-t border-[#E7E4DC] flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              disabled={isSubmitting}
              className="px-4 py-2 rounded-xl border border-stone-200 bg-white hover:bg-stone-50 text-xs font-bold text-stone-700 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={
                isSubmitting ||
                !name.trim() ||
                (creationMode === 'clone' && (!audioBlob || !canSubmitClone))
              }
              className="px-5 py-2 rounded-xl bg-[#8D4B00] hover:bg-[#733D00] text-white text-xs font-bold transition-all shadow-sm flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Saving Profile...
                </>
              ) : creationMode === 'clone' ? (
                'Save Cloned Voice'
              ) : (
                'Save Designed Voice'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
