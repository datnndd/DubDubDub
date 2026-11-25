import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Union
from pathlib import Path

@dataclass
class InputFile:
    name: Union[os.PathLike,str]=None
    dirname:Union[os.PathLike,str]=None
    basename:str=None
    noextname:str=None
    ext:str=None
    uuid:str=None
    target_dir:Optional[str]=None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    # Handles: dataclass_obj | dict_obj
    def __or__(self, other):
        if isinstance(other, dict):
            return asdict(self) | other
        return NotImplemented

    # Handles: dict_obj | dataclass_obj
    def __ror__(self, other):
        if isinstance(other, dict):
            return other | asdict(self)
        return NotImplemented

    def get(self, key,default=None):
        return getattr(self,key,default)


@dataclass
class SignMsg:
    type:str="logs"
    uuid:str=""
    text:str=""
    duration:Optional[float]=None

    def __init__(self, type:str="logs", uuid:str="", text:str="", duration:Optional[float]=None, **kwargs):
        self.type = type
        self.uuid = uuid
        self.text = text
        self.duration = duration
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def get(self, key,default=None):
        return getattr(self,key,default)

    # Should be in terminal state
    def is_stop(self):
        return self.type in ['end','stop','succeed','error']

    def is_error(self):
        return self.type == 'error'

@dataclass
class SrtItem:
    text: str = ""
    start_time: Union[int,float] = 0
    end_time: Union[int,float] = 0
    startraw: str = ''
    endraw: str = ''
    line: Optional[int] = 1
    time: Optional[str] = ""
    spk: Optional[str] = ""  # Speaker id
    filename: Optional[str] = ""  # Corresponding audio segment


    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def get(self, key,default=None):
        return getattr(self,key,default)

    def items(self):
        _names=("line","time","start_time","end_time","startraw","endraw","text","spk","filename")
        for k in _names:
            yield k,getattr(self,k)

    def __iter__(self):
        _names=("line","time","start_time","end_time","startraw","endraw","text","spk","filename")
        return iter(_names)


# Video translation workflow uses all attributes
@dataclass
class TaskCfgBase:
    # General section
    uuid: str = None  # Default unique task id
    project_id: Optional[str] = None  # Parent project id

    name: Union[os.PathLike,str]=None  # Normalized absolute path to original file D:/XXX/1.MP4
    dirname: Union[os.PathLike,str]=None  # Directory of original file D:/XXX
    noextname: str = None  # Original video name without extension
    basename: str = None  # noextname + ext name 1.mp4
    ext: str = None  # Extension mp4

    target_dir: str = None  # Output folder, target video output folder

    cache_folder: str = None  # Temporary folder for current file, stores intermediate files

    is_cuda: bool = False  # Whether to use CUDA acceleration

    source_language: str = None  # Original language name or code
    source_language_code: str = None  # Original language code
    source_sub: Union[os.PathLike,str]=None  # Absolute path to original subtitle file
    source_wav: Union[os.PathLike,str]=None  # Original language audio, located under temporary folder
    source_wav_output: Union[os.PathLike,str]=None  # Original language audio output, located under target folder

    target_language: str = None  # Target language name or code
    target_language_code: str = None  # Target language code
    target_sub: Union[os.PathLike,str]=None  # Absolute path to target subtitle file
    target_wav: Union[os.PathLike,str]=None  # Target language audio, located under temporary folder
    target_wav_output: Union[os.PathLike,str]=None  # Target language audio output, located under target folder


# Speech recognition
@dataclass
class TaskCfgSTT(TaskCfgBase):
    ####### Speech recognition related
    detect_language: str = None  # Subtitle detection language code
    recogn_type: int = None  # Speech recognition channel
    model_name: str = None  # Model name
    shibie_audio: Union[os.PathLike,str]=None  # Converted to pcm_s16le 16k as speech recognition audio file
    remove_noise: bool = False  # Whether to remove noise
    force_recogn: bool = False  # Re-run ASR after an explicit stage reset.
    enable_diariz: bool = False  # Whether to perform speaker diarization
    nums_diariz: int = 0  # Number of diarization speakers
    rephrase: int = 0  # 0=default segmentation, 1=LLM re-segmentation
    fix_punc: int = 0  # 0=default, 1=restore punctuation, 2=remove all punctuation
    deepgram_options: Optional[Union[dict, str]] = None
    subtitle_source: str = "audio_asr"  # subtitle source audio_asr|video_ocr
    ocr_roi: Optional[tuple] = None  # normalized (x,y,w,h)
    ocr_roi_confirmed: bool = False  # user confirmed ROI in dialog
    ocr_options: Optional[dict] = None  # coarse_interval_ms, use_angle_cls, paragraph_mode


# Dubbing
@dataclass
class TaskCfgTTS(TaskCfgBase):
    ######## Dubbing related
    tts_type: int = None  # Speech synthesis channel
    volume: str = "+0%"  # Volume
    pitch: str = "+0Hz"  # Pitch
    voice_rate: str = "+0%"  # Voice rate
    voice_role: str = None  # Voice role
    line_roles: dict[str, str] = field(default_factory=dict)  # per-subtitle voice assignments
    voice_autorate: bool = False  # Whether audio auto-accelerates
    video_autorate: bool = False  # Whether video auto-slows
    remove_silent_mid: bool = False  # Whether to remove gaps between subtitles
    align_sub_audio: bool = True  # Whether to force alignment between subtitles and voice


# Subtitle translation
@dataclass
class TaskCfgSTS(TaskCfgBase):
    ######## Subtitle translation related
    translate_type: int = None  # Subtitle translation channel
    aisendsrt: Optional[bool] = None  # None follows the global setting; bool selects the existing text/SRT prompt flow
    segments: Optional[Union[list, tuple]] = None  # Subtitle segment list to translate or recognized


# Video translation all
@dataclass
class TaskCfgVTT(TaskCfgSTT, TaskCfgTTS, TaskCfgSTS):
    ############## Video translation specific
    subtitle_language: str = None  # Soft subtitle embedded language code, 3 letters
    app_mode: str = "biaozhun"  # Work mode: standard or extract
    subtitles: str = ""  # Existing subtitle text, e.g. pre-imported
    targetdir_mp4: Union[os.PathLike,str]=None  # Final output composite mp4
    novoice_mp4: Union[os.PathLike,str]=None  # Silent video extracted from original video
    is_separate: bool = False  # Whether to perform vocal and background sound separation
    embed_bgm: bool = True  # Whether to re-embed background music
    instrument: Union[os.PathLike,str]=None  # Separated background audio
    vocal: Union[os.PathLike,str]=None  # Separated vocal audio
    clear_cache: bool = False  # Whether to clean up existing files
    background_music: Union[os.PathLike,str]=None  # Manually added background audio, normalized full path
    subtitle_type: int = 0  # Subtitle embedding type: 0=none, 1=hard, 2=soft, 3=dual hard, 4=dual soft
    only_out_mp4: bool = False  # Whether to output mp4 only, used only in video translation
    only_out_dubbed_audio: bool = False
    recogn2pass: bool = False  # Second-pass speech recognition on dubbed audio
    output_srt: int = 0  # Output subtitle format in transcribe and translate mode: 0=single, 1=bilingual target bottom, 2=bilingual target top
    copysrt_rawvideo: bool = False  # Whether to copy generated subtitles to video directory
    loop_backaudio: int = 0  # Loop background audio or stretch background audio
    backaudio_volume: float = 0.8  # Background volume
    source_audio_volume: float = 0.0  # final mix contribution from original video audio
    thumbnail: Union[os.PathLike,str]=None  # optional exported video cover image
    subtitle_style: Optional[dict] = None  # per-job hard subtitle appearance
    annotations: Optional[list] = None  # Stage 4 positioned on-screen annotations
    burn_annotations: bool = False  # Whether to burn Stage 4 annotations into rendered video
    batch:bool=False  # Batch translation mode or single video translation mode
    batch_size:int=0  # 0=concurrent batch mode, >0 n items per batch
    
    def __repr__(self):
        _msg=[]
        from videotrans.recognition import ALLOW_CHANGE_MODEL
        from videotrans.util.tools import get_recogn_type,get_tanslate_type,get_tts_type
        from videotrans.configure.config import tr,settings

        isTrue={True:"已选",False:"未选"}
        _duanjus=["默认断句","LLM重新断句"]
        _subtitles=[
                tr('nosubtitle'),
                tr('embedsubtitle'),
                tr('softsubtitle'),
                tr('embedsubtitle2'),
                tr('softsubtitle2')
            ]
        _outs=["单字幕","目标语言在下双字幕","目标语言在上双字幕"]
        _loops=["循环播放","拉长(降速播放)"]
        
        if self.app_mode=="tiqu":
            _msg.append(f"[TaskCfgVTT]当前工作模式: 转录并翻译字幕")
        else:
            _msg.append(f'[TaskCfgVTT]当前工作模式: 翻译视频或音频 {"批量翻译模式" if self.batch else "单视频模式"}' +  (f" 每批{self.batch_size}个" if self.batch and self.batch_size>0 else "") )
        
        _msg.append(f'原始输入文件名: {self.name}, \n输出结果保存到文件夹: {self.target_dir},\n临时文件夹: {self.cache_folder}')

        _msg.append(f'{isTrue[self.clear_cache]} 清理已存在')
        _msg.append(f'{"已" if self.is_cuda else "未"}启用CUDA加速')
        _msg.append(f'{isTrue[self.remove_noise]} 降噪')
        if self.enable_diariz:
            _msg.append(f'已选 识别说话人，最大说话人数量{"不限制" if self.nums_diariz<1 else self.nums_diariz+1}')
        if self.fix_punc>0:
            _msg.append(f'{"已选 恢复标点符号" if self.fix_punc==1 else "已选 删除所有标点符号"}')
        
        
        _msg.append(f"{tr('Speech Recognit')}:{get_recogn_type(self.recogn_type)}, model_name: {self.model_name if self.recogn_type in ALLOW_CHANGE_MODEL else ''}, 发音语言: {self.source_language}, 断句方式:{_duanjus[self.rephrase]}")
        
        if self.target_language in [None,'No','-'] or self.source_language==self.target_language:
            _msg.append(f'{"发音语言和目标语言相同" if self.source_language==self.target_language else "未选 目标语言"}，不翻译字幕')
        else:
            _msg.append(f"{tr('Translate channel')}:{get_tanslate_type(self.translate_type)},原始语言:{self.source_language},目标语言:{self.target_language}, {isTrue[settings['aisendsrt']]} {tr('Send SRT')}")
        
        if self.app_mode=='tiqu':
            if self.copysrt_rawvideo:
                _msg.append('已选 将生成的字幕复制到视频目录下')
            _msg.append(f"{tr('Subtitle format:')}: {_outs[self.output_srt]}")
        else:
            
            if self.voice_role in [None,'No']:
                _msg.append('未选 配音角色，不进行配音')
            else:
                _msg.append(f"{tr('Dubbing channel')}:{get_tts_type(self.tts_type)}, 角色:{self.voice_role}, 配音语言:{self.target_language}, {isTrue[self.recogn2pass]} 二次语音识别")    
                if self.recogn2pass and self.subtitle_type>2:
                    _msg.append('\t[已选中 二次语音识别，但不会生效，因已选择 嵌入双字幕，需保证原始和目标字幕时间轴一致，而二次语音识别会重新生成时间轴不同的目标字幕]')
                _msg.append(f'音量:{self.volume}, 语速:{self.voice_rate}, {isTrue[self.voice_autorate]} 音频加速, {isTrue[self.video_autorate]} 视频慢速')
                
                if not self.voice_autorate and not self.video_autorate:
                    _msg.append(f'{isTrue[self.remove_silent_mid]} 移除字幕间空隙,  {isTrue[self.align_sub_audio]} 强制对齐字幕和声音')
            _msg.append(f'字幕: {_subtitles[self.subtitle_type]} {_outs[self.output_srt] if self.subtitle_type >2 else ""}')

            _vocal_exists=self.vocal and Path(self.vocal).exists()
            _instr_exists=self.instrument and Path(self.instrument).exists()
            if self.is_separate or self.background_music or _vocal_exists or _instr_exists:
                
                _str=f"{isTrue[self.is_separate]} 分离人声与背景声"
                if self.embed_bgm:
                    _str+=f', 已选 重新嵌入背景声, 背景音量{self.backaudio_volume}, 背景声音时长 短于 视频时长时: {_loops[self.loop_backaudio]}'
                
                if self.background_music:
                    _str+=f', 手动添加了背景音频:{self.background_music}\n'
                if _vocal_exists:
                    _str+=',存在分离后的纯净人声文件'
                if _instr_exists:
                    _str+=',存在分离后的背景声音文件'
                _msg.append(_str)
            if self.only_out_mp4:
                _msg.append('已选 仅输出mp4')
        _msg.append(f'代理地址:{settings.get("proxy")}')
        return "\n".join(_msg)
