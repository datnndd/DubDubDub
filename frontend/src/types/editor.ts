export interface AudioMixSettings {
  original: number;
  dubbed: number;
  background: number;
}

export interface SubtitleStyleSettings {
  preset: string;
  fontFamily: string;
  fontSize: number;
  color: string;
  outlineColor: string;
  outlineWidth: number;
  shadowColor: string;
  shadowSize: number;
  fontWeight?: string;
  fontStyle?: string;
  opacity?: number;
  aiLipSync?: boolean;
  deReverb?: boolean;
  faceRetouch?: boolean;
  superRes4K?: boolean;
}

export interface EditVideoState {
  audioMix: AudioMixSettings;
  backgroundAudio: { id: string; file: File | null; name: string; url: string } | null;
  thumbnail: { id: string; file: File | null; name: string; url: string } | null;
  exporting: boolean;
  error: string | null;
  activeTab: 'audio' | 'subtitles' | 'annotations' | 'assets';
  isCapCutModalOpen?: boolean;
  burnAnnotations?: boolean;
}
