# See @docs/Synchronize.md for explanation of principles
import math
import os
import shutil
import time
from pathlib import Path

# Import soundfile and audio processing
import soundfile as sf
import numpy as np  # Added numpy for channel processing
import wave
from pydub import AudioSegment

# Try importing pyrubberband
from videotrans.configure.contants import INSTALL_RUBBERBAND_TIPS

try:
    import pyrubberband as pyrb
    HAS_RUBBERBAND = True
except ImportError:
    HAS_RUBBERBAND = False


from videotrans.configure.config import ROOT_DIR,tr, settings, logger
from videotrans.configure import config
from videotrans.util import tools
from concurrent.futures import ProcessPoolExecutor



def _cut_video_get_duration(i, task, novoice_mp4_original, preset, crf,fps_mode):
    """
    Cut video segment and apply slowdown (PTS) processing as needed.
    """
    task['actual_duration'] = 0 
    
    # Force using absolute path
    input_video_path = Path(novoice_mp4_original).resolve().as_posix()
    
    # Original clip duration
    source_duration_ms = task['end'] - task['start']
    if source_duration_ms <= 0:
        logger.error(f"[Video-Cut] 片段{i} 原始时长<=0: {task=}，跳过处理")
        return task

    # Target duration
    target_duration_ms = task.get('target_time', source_duration_ms)
    
    ss_time = tools.ms_to_time_string(ms=task['start'], sepflag='.')
    source_duration_s = source_duration_ms / 1000.0
    target_duration_s = target_duration_ms / 1000.0
    
    # PTS factor
    pts_factor = task.get('pts', 1.0)
    
    flag = f'[Video-Cut] 片段{i} [原片段时长:{source_duration_ms}ms] [目标:{target_duration_ms}ms] [PTS:{pts_factor}]'

    # Build main command
    cmd = [
        '-y',
        '-ss', ss_time,
        '-t', f'{source_duration_s:.6f}',
        '-i', input_video_path, # Use absolute path
        '-an',
        '-c:v', 'libx264', 
        '-g', '1',
        '-preset', preset, 
        '-crf', crf,
        '-pix_fmt', 'yuv420p'
    ]

    filter_complex = []
    if abs(pts_factor - 1.0) >0:
        # Cannot be accurate to milliseconds; usually tens of ms less than expected, add correction
        filter_complex.append(f"setpts={pts_factor+0.009999999}*PTS")
    else:
        filter_complex.append("setpts=1.003999000*PTS") # Inexact; actual duration short by tens of ms, add slight offset

    cmd.extend(['-vf', ",".join(filter_complex)])
    cmd.extend(fps_mode)
    
    cmd.append(os.path.basename(task['filename']))
    # Get working directory (for storing temp files)
    work_dir = Path(task['filename']).parent.as_posix()
    
    try:
        # Execute FFmpeg
        tools.runffmpeg(cmd, force_cpu=True, cmd_dir=work_dir)
        
        file_path = Path(task['filename'])
        
        # Check success; execute fallback logic on failure
        if not file_path.exists() or file_path.stat().st_size < 1024:
            logger.error(f"{flag} 变速生成失败或文件无效，尝试无变速剪切兜底:{task=}...")
            
            # [Correction] Fallback command must also include fps_mode and setpts=PTS for concatenation compatibility
            cmd_backup = [
                '-y', 
                '-ss', ss_time, 
                '-t', f'{source_duration_s:.6f}', # Fallback uses original duration
                '-i', input_video_path,
                '-an', 
                '-c:v', 'libx264', 
                '-g', '1',
                '-preset', preset, 
                '-crf', crf,
                '-pix_fmt', 'yuv420p',
                '-vf', 'setpts=PTS',  # Add explicitly
            ]+fps_mode

            cmd_backup.append(os.path.basename(task['filename']))
            tools.runffmpeg(cmd_backup, force_cpu=True, cmd_dir=work_dir)

        # Check again
        if file_path.exists() and file_path.stat().st_size >= 1024:
            try:
                real_time = tools.get_video_duration(task["filename"])
            except Exception as e:
                logger.error(f"{flag} 获取时长失败: {e}")
                real_time = 0

            task['actual_duration'] = real_time
            logger.debug(f"{flag} 完成。真实时长: {real_time}ms, 真实-应生={real_time-target_duration_ms}ms")
        else:
            task['actual_duration'] = 0
            logger.error(f"{flag} 最终生成失败。")
    except Exception as e:
        logger.error(f"{flag} 处理异常: {e}")
        try:
            if Path(task['filename']).exists():
                Path(task['filename']).unlink()
        except OSError:
            pass
            
    return task


def _change_speed_rubberband(input_path, target_duration):
    """
    Change audio speed using Rubber Band.
    """
    try:
        y, sr = sf.read(input_path)
        if len(y) == 0:
            logger.error(f"[Audio-RB] 空音频文件: {input_path=},{target_duration=}")
            return False
            
        current_duration = round((len(y) / sr) * 1000)
        
        if target_duration <= 0: target_duration = 1
        
        if target_duration > current_duration:
             # Allow slight deviation, or handle via subsequent silence padding
             logger.debug(f"[Audio-RB] 目标时长({target_duration}) > 当前时长({current_duration})，跳过变速，交由静音填充。")
             return False

        time_stretch_rate = current_duration / target_duration
        
        # Limit range
        time_stretch_rate = max(0.2, min(time_stretch_rate, 50.0))
        
        logger.debug(f"[Audio-RB] {input_path} 原长:{current_duration}ms -> 目标:{target_duration}ms 倍率:{time_stretch_rate:.2f}")

        y_stretched = pyrb.time_stretch(y, sr, time_stretch_rate)
        
        # If mono (ndim=1), duplicate to stereo
        if y_stretched.ndim == 1:
            y_stretched = np.column_stack((y_stretched, y_stretched))
        
        sf.write(input_path, y_stretched, sr)
        
    except Exception as e:
        logger.error(f"[Audio-RB] 音频处理失败 {input_path}: {e}")
        return False
    return True

def get_audio_duration_ms(audio_path) -> int:
    """Read audio duration in milliseconds from header without decoding full PCM into RAM."""
    if not audio_path:
        return 0
    path_obj = Path(audio_path)
    if not path_obj.is_file() or path_obj.stat().st_size == 0:
        return 0

    try:
        info = sf.info(str(path_obj))
        if info.samplerate > 0 and info.frames >= 0:
            return int(round(info.frames * 1000.0 / info.samplerate))
    except Exception:
        pass

    try:
        with wave.open(str(path_obj), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return int(round(frames * 1000.0 / rate))
    except Exception:
        pass

    try:
        return max(0, int(round(float(tools.get_audio_time(str(path_obj)) or 0) * 1000.0)))
    except Exception:
        return 0


def _resample_audio_array(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D or 2D float32 audio array from orig_sr to target_sr deterministically."""
    if orig_sr == target_sr or data.size == 0:
        return np.asarray(data, dtype=np.float32)
    try:
        import soxr
        resampled = soxr.resample(data, orig_sr, target_sr)
        return np.asarray(resampled, dtype=np.float32)
    except Exception:
        from scipy.signal import resample_poly
        g = math.gcd(target_sr, orig_sr)
        axis = 0
        resampled = resample_poly(data, target_sr // g, orig_sr // g, axis=axis)
        return np.asarray(resampled, dtype=np.float32)


def _precise_speed_up_audio(input_path=None, target_duration=None):
    current_duration_ms = get_audio_duration_ms(input_path)
    if current_duration_ms <= 0 or not target_duration or target_duration <= 0:
        return False

    # Construct atempo filter chain
    # atempo constraint: parameter must be in [0.5, 2.0]
    atempo_list = []
    speed_factor = current_duration_ms / target_duration

    # Handle speedup case (> 2.0)
    while speed_factor > 2.0:
        atempo_list.append("atempo=2.0")
        speed_factor /= 2.0

    while 0 < speed_factor < 0.5:
        atempo_list.append("atempo=0.5")
        speed_factor /= 0.5

    # Append remaining speed factor
    atempo_list.append(f"atempo={speed_factor}")

    # Join filters with comma to chain them, e.g. "atempo=2.0,atempo=1.5"
    filter_str = ",".join(atempo_list)
    after_path = Path(f"{input_path}-after.wav")

    cmd = [
        '-y',
        '-i',
        input_path,
        '-filter:a',
        filter_str,
        '-ar', "48000",
        '-ac', "2",
        '-c:a', 'pcm_s16le',
        str(after_path),
    ]
    try:
        tools.runffmpeg(cmd)
        if not after_path.is_file() or after_path.stat().st_size == 0:
            return False
        os.replace(after_path, input_path)
    except Exception as e:
        logger.exception(f'音频加速失败:{e}')
        return False
    finally:
        after_path.unlink(missing_ok=True)
    return True


class SpeedRate:
    MIN_CLIP_DURATION_MS = 40
    AUDIO_SAMPLE_RATE = 48000
    AUDIO_CHANNELS = 2
    # When both audio and video are enabled, if dubbing/subtitle ratio is below threshold, only speed up audio, do not slow down video
    BOTH_MODE_AUDIO_ONLY_THRESHOLD = 1.2

    def __init__(self,
                 *,
                 queue_tts=None,
                 should_videorate=False,
                 should_audiorate=False,
                 uuid=None,
                 novoice_mp4=None,
                 raw_total_time=0,
                 target_audio=None,
                 cache_folder=None,
                 remove_silent_mid=False,
                 align_sub_audio=True,
                 event_sink=None,
                 cancellation_token=None,
                 ):
        self.align_sub_audio = align_sub_audio
        self.raw_total_time = raw_total_time if raw_total_time is not None else 0
        self.remove_silent_mid = remove_silent_mid
        self.queue_tts = queue_tts
        self.len_queue = len(queue_tts)
        self.should_videorate = should_videorate
        self.should_audiorate = should_audiorate
        self.uuid = uuid
        self.event_sink = event_sink
        self.cancellation_token = cancellation_token
        self.novoice_mp4_original = novoice_mp4
        self.novoice_mp4 = novoice_mp4
        self.cache_folder = cache_folder if cache_folder else Path(
            f'{config.TEMP_DIR}/{str(uuid if uuid else time.time())}').as_posix()
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)

        self.stop_show_process = False

        self.fps_mode=["-fps_mode","vfr"]
        ## Whether to use fixed frame rate
        if settings.get('fps_mode')=='cfr':
            video_fps=tools.get_video_info(novoice_mp4,video_fps=True) if novoice_mp4 and Path(novoice_mp4).exists() else 30
            self.fps_mode=["-r",f"{video_fps}","-fps_mode","cfr"]
            
        self.target_audio = target_audio

        raw_speed_rate = float(settings.get('max_audio_speed_rate', 1.35) or 1.35)
        self.max_audio_speed_rate = min(raw_speed_rate, 1.35) if raw_speed_rate > 0 else 1.35
        self.max_video_pts_rate = float(settings.get('max_video_pts_rate', 10))

        self.audio_data = [] 
        self.video_for_clips = [] 

        self.crf = "18"
        self.preset = "veryfast"
        
        try:
            if Path(ROOT_DIR + "/crf.txt").exists():
                self.crf = str(int(Path(ROOT_DIR + "/crf.txt").read_text()))
            if Path(ROOT_DIR + "/preset.txt").exists():
                preset_tmp = str(Path(ROOT_DIR + "/preset.txt").read_text().strip())
                if preset_tmp in ['ultrafast', 'veryfast', 'medium', 'slow']:
                    self.preset = preset_tmp
        except Exception:
            pass

        self.audio_speed_rubberband = shutil.which("rubberband")
        logger.debug(f"[SpeedRate] Init. AudioRate={self.should_audiorate}, VideoRate={self.should_videorate}, Rubberband={bool(self.audio_speed_rubberband)}")
        if not HAS_RUBBERBAND or not self.audio_speed_rubberband:
            logger.warning(f"[SpeedRate] Rubberband 不可用，将使用 pydub+ffmpeg 处理音频加速(较粗糙)。\n建议安装，加速效果更精确\n{INSTALL_RUBBERBAND_TIPS}")

    def signal(self, text="", type="logs"):
        if self.event_sink:
            self.event_sink({"text": text, "type": type, "uuid": self.uuid})
        else:
            logger.info(text)

    def run(self):
        if not self.queue_tts:
            return []
        if not self.should_audiorate and not self.should_videorate:
            logger.debug("[SpeedRate] 未启用变速，进入普通拼接模式。")
            self._run_no_rate_change_mode()
            return self.queue_tts
        
        logger.debug("[SpeedRate] 启用变速，进入对齐模式。")
        
        # 1. Preprocessing
        self._prepare_data()
        
        # 2. Calculation
        self._calculate_adjustments()
        
        # 3. Audio speed adjustment
        if self.audio_data:
            self.signal(tr('Sound speed alignment stage')+'...')
            self._execute_audio_speedup_rubberband()

        # 4. Video speed adjustment
        if self.should_videorate and self.video_for_clips:
            self.signal(tr('Slow video')+'...')
            processed_video_clips = self._video_speeddown()           
            self._concat_video(processed_video_clips)
            
            # Update total duration
            if Path(self.novoice_mp4).exists():
                try:
                    self.raw_total_time = tools.get_video_duration(self.novoice_mp4)
                    logger.debug(f"[SpeedRate] 新视频生成完毕，总时长: {self.raw_total_time}ms")
                except Exception:
                    pass
            
        # 5. Audio alignment and concatenation
        self.signal(tr('Concatenating final audio'))
        self._concat_audio_aligned()

        return self.queue_tts

    def _prepare_data(self):
        """Data cleaning and pre-processing."""
        self.signal(tr("Preparing data"))
        
        if self.novoice_mp4_original and tools.vail_file(self.novoice_mp4_original):
            self.raw_total_time = tools.get_video_duration(self.novoice_mp4_original)

        if self.raw_total_time>0:
            self.queue_tts[-1]['end_time']=self.raw_total_time

        for i,current in  enumerate(self.queue_tts):
            current['source_duration'] = current['end_time'] - current['start_time']
            current['dubb_time']=0
            current.setdefault('start_time_source', current['start_time'])
            current.setdefault('end_time_source', current['end_time'])
            if current['start_time']>=current['end_time']:
                logger.error(f'第 {i} 行字幕时间轴<=0，不正确，跳过处理:{current=}\n')
                continue

            # Subtitle start point, used for splitting video and changing speed
            current['start_time_source']=current['start_time']
            current['end_time_source'] = current['end_time']
            # If first segment start time < 100ms, set to start from 0 to prevent short segment errors
            if i == 0 and current['start_time']<100:
                current['start_time_source'] = 0
            
            # Fill gaps to expand adjustment interval and reduce speed change magnitude, keeping at least 150ms natural pause
            # Except for item 0, start points remain unchanged; only move end points
            if i < len(self.queue_tts) - 1:
                next_sub = self.queue_tts[i+1]
                effective_end = max(current['end_time'], next_sub['start_time'] - 150)
                effective_end = min(effective_end, next_sub['start_time'])
                current['end_time'] = effective_end
                current['end_time_source'] = effective_end

            current['source_duration'] = current['end_time_source'] - current['start_time_source']
            
            # Check dubbing file
            if not current.get('filename') or not Path(current['filename']).exists():
                current['dubb_time'] = current['source_duration']
                logger.debug(f"[Prepare] 字幕[{current['line']}] 无配音，预留 {current['source_duration']}ms 静音占位")
            else:
                current['dubb_time'] = get_audio_duration_ms(current['filename'])

    def _calculate_adjustments(self):
        """Calculation strategy."""
        self.signal(tr("Calculating sync adjustments"))
        # Video slowdown: there might be silent video before item 0 subtitle
        if self.should_videorate and self.queue_tts[0]['start_time_source']>0:
            self.video_for_clips.append({
                    "start": 0,
                    "end": self.queue_tts[0]['start_time_source'],
                    "target_time": self.queue_tts[0]['start_time_source'],
                    "pts": 1,
                    "tts_index": -1,
                    "line": -1
            })
            
        dubbing_sec=0
        video_time=0
        if self.queue_tts[0]['start_time_source']>0:
            dubbing_sec+=self.queue_tts[0]['start_time_source']
            video_time+=self.queue_tts[0]['start_time_source']
        
        for i, it in enumerate(self.queue_tts):
            source_dur = it['source_duration']
            if source_dur<=0:
                continue

            dubb_dur = it['dubb_time']
            video_target = source_dur
            audio_target = source_dur
            
            mode_log = ""
            # Audio speedup only
            if self.should_audiorate and not self.should_videorate:
                mode_log = "Only Audio"
                # When dubbing > original subtitle duration, speed up audio; when shorter, do not process, add silence at end during merge
                if dubb_dur > source_dur:
                    ratio = dubb_dur / source_dur
                    if ratio > self.max_audio_speed_rate:
                        audio_target = int(dubb_dur / self.max_audio_speed_rate)
                    else:
                        audio_target = source_dur
            elif not self.should_audiorate and self.should_videorate:
                mode_log = "Only Video"
                # When dubbing > original subtitle duration, slow down video; when shorter, do not process, crop with setpts=pts directly
                if dubb_dur > source_dur:
                    video_target = dubb_dur
                    pts = video_target / source_dur
                    if pts > self.max_video_pts_rate:
                        video_target = int(source_dur * self.max_video_pts_rate)

            elif self.should_audiorate and self.should_videorate:
                mode_log = "Both"
                if dubb_dur > source_dur:
                    ratio = dubb_dur / source_dur
                    if ratio <= self.BOTH_MODE_AUDIO_ONLY_THRESHOLD:
                        # Ratio is small; speeding up audio alone is sufficient, no video slowdown needed
                        audio_target = source_dur
                        video_target = source_dur
                    else:
                        # Ratio is large; audio speedup and video slowdown each shoulder half the time difference
                        diff = dubb_dur - source_dur
                        joint_target = int(source_dur + (diff / 2))
                        min_audio_target = int(math.ceil(dubb_dur / self.max_audio_speed_rate))
                        audio_target = max(joint_target, min_audio_target)
                        video_target = audio_target
            
            # Logging
            flag=f"[Calc] Mode={mode_log} Line={it['line']} | 字幕可用区间={source_dur}ms, 当前实际配音时长={dubb_dur}ms -> "
            # Only register tasks that require audio speedup
            if self.should_audiorate and audio_target < dubb_dur:
                self.audio_data.append({
                    "filename": it['filename'],
                    "dubb_time": dubb_dur,
                    "target_time": audio_target
                })
                flag+=f' 配音加速目标时长={audio_target}ms'
                dubbing_sec+=audio_target
            # Register all segments; for segments without video slowdown, PTS=1.0
            if self.should_videorate:
                pts = video_target / source_dur if video_target>source_dur else 1.0
                self.video_for_clips.append({
                    "start": it['start_time_source'],
                    "end": it['end_time_source'],
                    "target_time": video_target,
                    "pts": pts,
                    "tts_index": i,
                    "line": it['line']
                })
                flag+=f' 视频慢速目标时长={video_target}ms，PTS={pts}  '
                video_time+=video_target
            
            
            logger.debug(flag)
        if self.should_videorate:
            logger.debug(f'视频应变速到时长={video_time/1000.0}s')
        if self.should_audiorate:
            logger.debug(f'配音应变速到时长={dubbing_sec/1000.0}s')


    def _execute_audio_speedup_rubberband(self):
        logger.debug(f"[Audio] 开始处理 {len(self.audio_data)} 个音频变速任务")
        if len(self.audio_data)<1:
            return
        all_task = []
        
        _wok=min(12, len(self.audio_data), max(os.cpu_count()-1,1) )
        logger.debug(f'使用{_wok}个进程处理音频加速')
        with ProcessPoolExecutor(max_workers=int(_wok)) as pool:
            for i, d in enumerate(self.audio_data):
                all_task.append(pool.submit(_change_speed_rubberband if HAS_RUBBERBAND and self.audio_speed_rubberband else _precise_speed_up_audio,d['filename'], d['target_time'] ))
        
        for i,task in enumerate(all_task):
            try:
                self.signal(f'Audio {i}/{len(all_task)}')
                res=task.result()
            except Exception:
                pass

    def _video_speeddown(self):
        data = []
        skip_i=[]
        last_index=len(self.video_for_clips)-1
        for i, clip_info in enumerate(self.video_for_clips):
            
            # Current segment pts==1.0 and next segment has pts==1.0, merge next segment to reduce cut segments, since each cut introduces tens of ms error
            if i>0 and clip_info['pts']==1.0 and len(data)>0 and data[-1]['pts']==1.0:
                data[-1]['target_time']+=clip_info['target_time']                    
                data[-1]['end']=clip_info['end']
                continue
            
            clip_info['queue_index'] = clip_info.get('tts_index',-1)
            clip_info['filename'] = Path(self.cache_folder, f"clip_{i}_{clip_info['pts']:.3f}.mp4").as_posix()
            data.append(clip_info)
       

        if len(data)<1:
            return []
        all_task = []
        logger.debug(f"[Video] 提交 {len(data)} 个视频片段处理慢速任务，原视频片段为 {len(self.video_for_clips)} 个")
        _wok=min(12, len(data), max(os.cpu_count()-1,1) )
        logger.debug(f'使用{_wok}个进程处理视频慢速')
        with ProcessPoolExecutor(max_workers=int(_wok)) as pool:
            for i, d in enumerate(data):
                all_task.append(pool.submit(_cut_video_get_duration,i, d, self.novoice_mp4_original, self.preset, self.crf,self.fps_mode  ))
          
        processed_clips = []
        for i,task in enumerate(all_task):
            try:
                self.signal(f'Video {i}/{len(all_task)}')
                res = task.result()
                if res: 
                    processed_clips.append(res)
            except Exception as e:
                logger.error(f"[Video] 任务异常: {e}")
        
        # Logging: check deviation
        pts_gt1=0
        pts_eq1=0
        real_video_time=0
        video_target_time=0
        for t in processed_clips:
            real_video_time+=t['actual_duration']
            video_target_time+=t['target_time']
            if t['pts']>1.0:
                pts_gt1+=t['actual_duration']-t['target_time']
            else:
                pts_eq1+=t['actual_duration']-t['target_time']
        logger.debug(f'真实视频时长({real_video_time})-应该生成时长({video_target_time})差值={real_video_time-video_target_time}ms,PTS>1.0的差值:{pts_gt1}ms,PTS=1.0的差值:{pts_eq1}ms')
        processed_clips.sort(key=lambda x: x.get('queue_index', -1))
        return processed_clips

    def _concat_video(self, processed_clips):
        txt_content = []
        valid_cnt = 0
        for clip in processed_clips:
            if clip.get('actual_duration', 0) > 0 and Path(clip['filename']).exists():
                path = Path(clip['filename']).as_posix()
                txt_content.append(f"file '{path}'")
                valid_cnt += 1
            else:
                logger.error(f"[Video-Concat] 忽略无效片段: {clip=}")
        
        if valid_cnt == 0: 
            logger.error("[Video-Concat] 没有有效片段，跳过拼接")
            return

        concat_list = Path(self.cache_folder, "video_concat.txt").as_posix()
        with open(concat_list, 'w', encoding='utf-8') as f:
            f.write("\n".join(txt_content))
            
        output_path = Path(self.cache_folder, "merged_video.mp4").as_posix()
        
        cmd = ['-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output_path]
        logger.debug(f"[Video-Concat] 合并 {valid_cnt} 个片段 -> {output_path}\n{cmd=}")
        self.signal(tr('Concat videos'))
        tools.runffmpeg(cmd, force_cpu=True, cmd_dir=self.cache_folder)

        if Path(output_path).exists():
            shutil.move(output_path, self.novoice_mp4)
            # Delete segments
            self._del_mp4_clip()
            
    def _del_mp4_clip(self):
        deleted_count = 0
        for f in Path(self.cache_folder).glob('clip_*.mp4'):
            try:
                f.unlink()
                deleted_count += 1
            except OSError as e:
                logger.exception(f"无法删除文件 {f.name}: {e}", exc_info=True)
        logger.debug(f"清理视频慢速中生成的视频片段，共删除了 {deleted_count} 个文件。")

    def _concat_audio_aligned(self):
        timeline = []
        dubbing_total = self.queue_tts[0]['start_time']
        if self.queue_tts[0]['start_time'] > 0:
            timeline.append(("silence", self.queue_tts[0]['start_time']))

        for i, it in enumerate(self.queue_tts):
            if it['source_duration'] <= 0:
                continue
            it['start_time'] = dubbing_total
            if not it.get('filename') or not Path(it['filename']).exists():
                it['end_time'] = it['start_time'] + it['source_duration']
                timeline.append(("silence", it['source_duration']))
                dubbing_total += it['source_duration']
                continue
            _len = get_audio_duration_ms(it['filename'])
            timeline.append(("file", it['filename']))
            it['end_time'] = it['start_time'] + _len
            dubbing_total += _len
            if _len < it['source_duration']:
                timeline.append(("silence", it['source_duration'] - _len))
                dubbing_total += it['source_duration'] - _len

        logger.debug(f"{len(timeline)=}: 配音列表累积总时长={dubbing_total}ms")
        self._exec_concat_audio(timeline)

    def _run_no_rate_change_mode(self):
        # Concatenate directly when speed is unchanged
        self.signal(tr("Merging audio (No Speed Change)..."))

        timeline = []
        total_audio_duration = 0

        for i, it in enumerate(self.queue_tts):
            prev_end = 0 if i == 0 else self.queue_tts[i - 1].get('end_pos_for_concat', 0)
            start_time = it['start_time']
            gap = start_time - prev_end

            if not self.remove_silent_mid and gap > 0:
                timeline.append(("silence", gap))
                total_audio_duration += gap

            dubb_len = 0
            if it.get('filename') and Path(it['filename']).exists():
                timeline.append(("file", it['filename']))
                dubb_len = get_audio_duration_ms(it['filename'])
            elif it.get('filename'):
                dur = max(0, it['end_time'] - it['start_time'])
                if dur > 0:
                    timeline.append(("silence", dur))
                    dubb_len = dur

            total_audio_duration += dubb_len
            it['end_pos_for_concat'] = total_audio_duration

            if self.align_sub_audio:
                it['start_time'] = total_audio_duration - dubb_len
                it['end_time'] = total_audio_duration

        if self.raw_total_time > total_audio_duration:
            timeline.append(("silence", self.raw_total_time - total_audio_duration))

        self._exec_concat_audio(timeline)

    def _create_silen_file(self, name, duration_ms):
        path = Path(self.cache_folder, f"silence_{name}.wav").as_posix()
        duration_ms = max(1, int(duration_ms))
        n_frames = max(1, int(round(duration_ms * self.AUDIO_SAMPLE_RATE / 1000.0)))
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.AUDIO_SAMPLE_RATE)
            chunk_frames = min(n_frames, self.AUDIO_SAMPLE_RATE)
            zero_chunk = b"\x00" * (chunk_frames * self.AUDIO_CHANNELS * 2)
            rem = n_frames
            while rem > 0:
                take = min(rem, chunk_frames)
                wf.writeframes(zero_chunk if take == chunk_frames else b"\x00" * (take * self.AUDIO_CHANNELS * 2))
                rem -= take
        return path

    def _write_audio_timeline(self, timeline):
        if not timeline or not self.target_audio:
            return
        target_path = Path(self.target_audio)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_wav = Path(self.cache_folder, f"final_audio_temp_{os.getpid()}.wav")
        sr_out = self.AUDIO_SAMPLE_RATE
        ch_out = self.AUDIO_CHANNELS
        silence_chunk_frames = sr_out  # 1 second chunk (~384KB float32)
        zero_block = np.zeros((silence_chunk_frames, ch_out), dtype=np.float32)

        try:
            with sf.SoundFile(
                str(temp_wav),
                mode="w",
                samplerate=sr_out,
                channels=ch_out,
                subtype="PCM_16",
                format="WAV",
            ) as sf_out:
                for item in timeline:
                    if isinstance(item, tuple) and len(item) == 2:
                        kind, val = item
                    else:
                        kind, val = "file", item

                    if kind == "silence":
                        dur_ms = int(round(float(val or 0)))
                        if dur_ms <= 0:
                            continue
                        rem_frames = max(1, int(round(dur_ms * sr_out / 1000.0)))
                        while rem_frames > 0:
                            take = min(rem_frames, silence_chunk_frames)
                            sf_out.write(zero_block[:take])
                            rem_frames -= take
                    elif kind == "file" and val and Path(val).is_file():
                        try:
                            data, sr = sf.read(str(val), dtype="float32", always_2d=False)
                        except Exception:
                            seg = AudioSegment.from_file(str(val))
                            sr = seg.frame_rate
                            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                            denom = float(1 << (8 * seg.sample_width - 1))
                            data = samples / denom
                            if seg.channels > 1:
                                data = data.reshape((-1, seg.channels))
                        if data.size == 0 or sr <= 0:
                            continue
                        if not np.all(np.isfinite(data)):
                            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                        if int(sr) != sr_out:
                            data = _resample_audio_array(data, int(sr), sr_out)
                        if data.ndim == 1:
                            data = np.column_stack((data, data))
                        elif data.shape[1] == 1:
                            data = np.repeat(data, ch_out, axis=1)
                        elif data.shape[1] > ch_out:
                            data = data[:, :ch_out]
                        np.clip(data, -1.0, 1.0, out=data)
                        sf_out.write(data)

            if temp_wav.is_file() and temp_wav.stat().st_size > 44:
                os.replace(temp_wav, target_path)
                logger.debug(f"[Audio-Concat] 最终音频已生成到: {self.target_audio}")
            else:
                logger.error("[Audio-Concat] 最终音频生成失败")
        finally:
            temp_wav.unlink(missing_ok=True)

    def _exec_concat_audio(self, file_list):
        if not file_list:
            return
        self._write_audio_timeline(file_list)


# Dedicated handling for subtitle dubbing
class TtsSpeedRate(SpeedRate):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.should_videorate=False
        self.max_audio_speed_rate = min(self.max_audio_speed_rate, 1.35) if self.max_audio_speed_rate > 0 else 1.35


    def run(self):
        if not self.should_audiorate:
            logger.debug("[SpeedRate] 未启用变速，进入普通拼接模式。")
            self._run_no_rate_change_mode()
            return self.queue_tts
        # Remove invalid timelines
        self.queue_tts=[it for it in self.queue_tts if it['end_time']-it['start_time']>0]

        logger.debug("[SpeedRate] 启用变速，进入对齐模式。")

        # 1. Preprocessing
        self._prepare_data()

        # 2. Calculation
        self._calculate_adjustments()

        # 3. Audio speed adjustment
        if self.audio_data:
            self.signal('Processing audio speed...')
            self._execute_audio_speedup_rubberband()


        self.signal('Concatenating final audio...')
        self._concat_audio_aligned()

        return self.queue_tts

    def _prepare_data(self):
        """Data cleaning and pre-processing."""
        self.signal("Preparing data...")
        
        _len = len(self.queue_tts)
        for i in range(_len):
            current = self.queue_tts[i]
            if i < _len - 1:
                next_start = self.queue_tts[i+1]['start_time']
                effective_end = max(current['end_time'], next_start - 150)
                current['end_time'] = min(effective_end, next_start)
                        
            current['source_duration'] = current['end_time'] - current['start_time']
            if current['source_duration'] <= 0:
                logger.error(f'第 {i} 行字幕时间轴<=0，不正确，跳过处理:{current=}\n')
                current['source_duration'] = 0
                continue

            # Check dubbing file
            if not current.get('filename') or not Path(current['filename']).exists():
                current['dubb_time'] = current['source_duration']
                logger.debug(f"[Prepare] 字幕[{current['line']}] 无配音，预留 {current['source_duration']}ms 静音占位")
            else:
                current['dubb_time'] = get_audio_duration_ms(current['filename'])

    def _calculate_adjustments(self):
        """Calculation strategy."""
        self.signal("Calculating sync adjustments...")

        for i, it in enumerate(self.queue_tts):
            source_dur = it['source_duration']
            dubb_dur = it['dubb_time']
            if dubb_dur<=0 or source_dur<=0:
                continue
            audio_target = dubb_dur

            mode_log = f"[为字幕配音] {i=}"
            if dubb_dur > source_dur:
                ratio = dubb_dur / source_dur
                if ratio > self.max_audio_speed_rate:
                    audio_target = int(dubb_dur / self.max_audio_speed_rate)
                else:
                    audio_target = source_dur
                self.audio_data.append({
                    "filename": it['filename'],
                    "dubb_time": dubb_dur,
                    "target_time": audio_target
                })

            logger.debug(f"[Calc] Mode={mode_log} Line={it['line']} | Source_duration={source_dur} Dubb_duration={dubb_dur} -> TargetA={audio_target}")


    def _concat_audio_aligned(self):
        logger.debug("[Audio] 开始对齐拼接...")

        timeline = []

        # Restore original timeline
        for i, it in enumerate(self.queue_tts):
            # Add leading silence
            if i == 0 and it['start_time'] > 0:
                timeline.append(("silence", it['start_time']))

            # Actual dubbing duration
            if it.get('filename') and Path(it['filename']).exists():
                timeline.append(("file", it['filename']))
                dubb_len = get_audio_duration_ms(it['filename'])
            else:
                timeline.append(("silence", it['source_duration']))
                dubb_len = it['source_duration']
            # If actual dubbing is shorter than subtitle interval, append silence at the end
            if dubb_len < it['source_duration']:
                timeline.append(("silence", it['source_duration'] - dubb_len))
        self._exec_concat_audio(timeline)

