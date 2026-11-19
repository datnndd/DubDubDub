"""
pyVideoTrans CLI — command-line interface for video translation, dubbing, and transcription.

Usage examples:
  # Speech to text
  uv run cli.py --task stt --name "D:/videos/demo.mp4" --recogn_type 1 --model_name nova-3

  # Subtitle translation
  uv run cli.py --task sts --name "D:/subs/source.srt" --target_language_code en

  # Text to speech
  uv run cli.py --task tts --name "C:/subs/movie.srt" --tts_type 0 --voice_role "zh-CN-YunyangNeural"

  # Full video translation
  uv run cli.py --task vtv --name "E:/movies/clip.mp4" --source_language_code zh-cn --target_language_code en --voice_role "en-US-GuyNeural" --cuda
"""

import asyncio
import math
import multiprocessing
import sys
import re
import argparse
from dataclasses import asdict
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------------------------------------------------------
# TEXT_DB — bilingual (zh/en) strings for all CLI output
# ---------------------------------------------------------------------------
TEXT_DB: Dict[str, Dict[str, str]] = {
    # --- Log messages ---
    "exec_stt_task": {"zh": "[执行任务] 语音转录 (STT)", "en": "[Task] Speech Transcription (STT)"},
    "exec_tts_task": {"zh": "[执行任务] 语音合成 (TTS)", "en": "[Task] Text-to-Speech (TTS)"},
    "exec_sts_task": {"zh": "[执行任务] 字幕翻译 (STS)", "en": "[Task] Subtitle Translation (STS)"},
    "exec_vtv_task": {"zh": "[执行任务] 视频翻译 (VTV)", "en": "[Task] Video Translation (VTV)"},
    "process_file":  {"zh": "[处理文件] {}", "en": "[File] {}"},
    "param_list":    {"zh": "[参数列表] {}", "en": "[Params] {}"},
    "output_dir":    {"zh": "[输出目录] {}", "en": "[Output Dir] {}"},
    "done":          {"zh": "[完成] 任务执行完毕", "en": "[Done] Task completed successfully"},
    "failed":        {"zh": "[失败] 任务执行出错: {}", "en": "[Failed] Task error: {}"},

    # --- Argparse descriptions ---
    "cli_desc": {
        "zh": "pyVideoTrans 命令行模式\n文档: https://pyvideotrans.com/cli",
        "en": "pyVideoTrans CLI Mode\nDocs: https://pyvideotrans.com/cli"
    },
    "cli_epilog": {
        "zh": "示例:\n"
              "  %(prog)s --task stt --name \"D:/demo.mp4\" --recogn_type 1 --model_name nova-3\n"
              "  %(prog)s --task tts --name \"D:/demo.srt\" --tts_type 0 --voice_role \"zh-CN-YunyangNeural\"\n"
              "  %(prog)s --task sts --name \"D:/demo.srt\" --target_language_code en\n"
              "  %(prog)s --task vtv --name \"D:/demo.mp4\" --source_language_code zh-cn --target_language_code en --voice_role \"en-US-GuyNeural\"\n"
              "  %(prog)s --list providers\n"
              "  %(prog)s --list languages",
        "en": "Examples:\n"
              "  %(prog)s --task stt --name \"D:/demo.mp4\" --recogn_type 1 --model_name nova-3\n"
              "  %(prog)s --task tts --name \"D:/demo.srt\" --tts_type 0 --voice_role \"zh-CN-YunyangNeural\"\n"
              "  %(prog)s --task sts --name \"D:/demo.srt\" --target_language_code en\n"
              "  %(prog)s --task vtv --name \"D:/demo.mp4\" --source_language_code zh-cn --target_language_code en --voice_role \"en-US-GuyNeural\"\n"
              "  %(prog)s --list providers\n"
              "  %(prog)s --list languages"
    },
    "help_task": {
        "zh": "任务类型: stt(语音转录), tts(文字配音), sts(字幕翻译), vtv(视频翻译)",
        "en": "Task type: stt(Speech to Text), tts(Text to Speech), sts(Subtitle Trans), vtv(Video Trans)"
    },
    "help_name": {
        "zh": "待处理文件的绝对路径 (请使用双引号包裹含空格的路径)",
        "en": "Absolute path of the file to process (wrap in quotes if path contains spaces)"
    },
    "help_list": {
        "zh": "列出可用选项: providers(渠道), languages(语言), models(模型)",
        "en": "List available options: providers, languages, models"
    },
    "help_output_dir": {
        "zh": "输出目录 (默认: 与输入文件同级的 _video_out 目录)",
        "en": "Output directory (default: _video_out alongside input file)"
    },
    "help_log_level": {
        "zh": "日志级别: DEBUG, INFO, WARNING, ERROR (默认: WARNING)",
        "en": "Log level: DEBUG, INFO, WARNING, ERROR (default: WARNING)"
    },
    "help_verbose": {
        "zh": "显示详细输出 (等同于 --log-level INFO)",
        "en": "Show verbose output (equivalent to --log-level INFO)"
    },
    "help_quiet": {
        "zh": "静默模式,仅输出错误",
        "en": "Quiet mode, only output errors"
    },

    # --- STT params ---
    "group_stt":       {"zh": "STT (语音转录) 参数", "en": "STT (Speech Transcription) Parameters"},
    "help_recogn_type": {"zh": "语音识别渠道编号", "en": "Speech recognition provider index"},
    "help_detect_lang":  {"zh": "音频视频发音语言", "en": "Source language of audio/video"},
    "help_model_name": {
        "zh": "语音识别模型名称\nWhisper Large-v3 模型: {}\n其他渠道请在软件界面中查看",
        "en": "ASR model name\nWhisper Large-v3 model: {}\nSee the WebUI provider settings for other models"
    },
    "help_cuda":           {"zh": "启用CUDA加速", "en": "Enable CUDA acceleration"},
    "help_remove_noise":   {"zh": "启用降噪", "en": "Enable noise reduction"},
    "help_enable_diariz":  {"zh": "启用说话人识别", "en": "Enable speaker diarization"},
    "help_nums_diariz":    {"zh": "指定说话人数量", "en": "Number of speakers"},
    "help_rephrase":       {"zh": "重新断句 (0=默认, 1=LLM断句)", "en": "Rephrase (0=default, 1=LLM)"},
    "help_fix_punc":       {"zh": "恢复标点符号", "en": "Restore punctuation"},
    "help_subtitle_source": {"zh": "字幕来源: audio_asr(语音识别) 或 video_ocr(画面硬字幕OCR)", "en": "Subtitle source: audio_asr or video_ocr"},
    "help_ocr_roi":        {"zh": "视频OCR归一化区域 x,y,w,h (例如 0.1,0.75,0.8,0.2)", "en": "Normalized video OCR ROI x,y,w,h (e.g. 0.1,0.75,0.8,0.2)"},
    "help_ocr_interval_ms": {"zh": "视频OCR粗采样间隔毫秒数 (默认: 500)", "en": "Video OCR coarse sampling interval in ms (default: 500)"},

    # --- TTS params ---
    "group_tts":         {"zh": "TTS (文字配音) 参数", "en": "TTS (Text-to-Speech) Parameters"},
    "help_tts_type":     {"zh": "配音渠道编号", "en": "TTS provider index"},
    "help_voice_role":   {"zh": "音色名称 (TTS模式必选)", "en": "Voice role name (required for TTS)"},
    "help_voice_rate":   {"zh": "语速 (如 +20%%, -10%%)", "en": "Speech rate (e.g. +20%%, -10%%)"},
    "help_volume":       {"zh": "音量 (如 +50%%, -30%%)", "en": "Volume (e.g. +50%%, -30%%)"},
    "help_pitch":        {"zh": "音调 (如 +10Hz, -5Hz)", "en": "Pitch (e.g. +10Hz, -5Hz)"},
    "help_voice_autorate": {"zh": "自动加速音频以对齐字幕", "en": "Auto-speed audio to match subtitles"},
    "help_align_sub_audio": {"zh": "强制修改字幕以对齐音频", "en": "Force subtitle adjustment to align with audio"},

    # --- Translation params ---
    "group_trans":        {"zh": "Translation (翻译) 参数", "en": "Translation Parameters"},
    "help_translate_type": {"zh": "翻译渠道编号", "en": "Translation provider index"},
    "help_source_lang":   {"zh": "源语言代码 (STS默认auto, VTV必选)", "en": "Source language (auto for STS, required for VTV)"},
    "help_target_lang":   {"zh": "目标语言代码 (必选)", "en": "Target language (required)"},

    # --- VTV extra params ---
    "group_vtv":           {"zh": "VTV (视频翻译) 额外参数", "en": "VTV Extra Parameters"},
    "help_video_autorate": {"zh": "自动慢速视频以对齐字幕", "en": "Auto-slow video to match subtitles"},
    "help_is_separate":    {"zh": "分离人声背景声", "en": "Separate vocals and background"},
    "help_recogn2pass":    {"zh": "二次语音识别", "en": "Enable 2-pass recognition"},
    "help_subtitle_type":  {"zh": "字幕类型 (0=无, 1=硬, 2=软, 3=硬双, 4=软双)", "en": "Subtitle type (0=None, 1=Hard, 2=Soft, 3=Hard Dual, 4=Soft Dual)"},
    "help_clear_cache":    {"zh": "完成后清理缓存 (默认)", "en": "Clear cache after finish (default)"},
    "help_no_clear_cache": {"zh": "不清理缓存", "en": "Do not clear cache"},

    # --- Error messages ---
    "err_missing_task": {
        "zh": "缺少 --task 参数,可选值: stt, tts, sts, vtv\n使用 --help 查看详细帮助",
        "en": "Missing --task parameter. Choose: stt, tts, sts, vtv\nUse --help for details"
    },
    "err_file_not_found": {
        "zh": "文件不存在: {}\n请检查路径是否正确,含空格的路径请用双引号包裹",
        "en": "File not found: {}\nCheck path, wrap space-containing paths in quotes"
    },
    "err_tts_role_required": {
        "zh": "TTS 模式下 --voice_role 是必选参数\n使用 --list providers 查看可用渠道和音色",
        "en": "--voice_role is required for TTS mode\nUse --list providers to see available options"
    },
    "err_sts_target_required": {
        "zh": "--target_language_code 是必选参数\n使用 --list languages 查看可用语言",
        "en": "--target_language_code is required\nUse --list languages to see available options"
    },
    "err_vtv_missing": {
        "zh": "VTV 模式缺少必选参数: {}",
        "en": "VTV mode missing required params: {}"
    },
    "err_ocr_roi_required": {
        "zh": "使用 --subtitle_source video_ocr 时必须提供 --ocr_roi (格式: x,y,w,h)",
        "en": "--ocr_roi is required when --subtitle_source is video_ocr (format: x,y,w,h)"
    },
    "err_ocr_roi_invalid": {
        "zh": "--ocr_roi 格式无效, 需为 [0,1] 范围内的4个数值 x,y,w,h 且 w,h > 0",
        "en": "Invalid --ocr_roi format; expected 4 normalized floats x,y,w,h in [0,1] with w,h > 0"
    },
    "miss_source_lang": {"zh": "--source_language_code", "en": "--source_language_code"},
    "miss_target_lang": {"zh": "--target_language_code", "en": "--target_language_code"},

    # --- List output ---
    "list_providers_header": {
        "zh": "\n=== 可用渠道 ===\n\n--- 语音识别 (STT) ---",
        "en": "\n=== Available Providers ===\n\n--- Speech Recognition (STT) ---"
    },
    "list_trans_header":     {"zh": "\n--- 翻译 (Translation) ---", "en": "\n--- Translation ---"},
    "list_tts_header":       {"zh": "\n--- 配音 (TTS) ---", "en": "\n--- Text-to-Speech (TTS) ---"},
    "list_languages_header": {"zh": "\n=== 可用语言代码 ===", "en": "\n=== Available Language Codes ==="},
    "list_models_header":    {"zh": "\n=== faster-whisper 可用模型 ===", "en": "\n=== faster-whisper Models ==="},
}


# ---------------------------------------------------------------------------
# tr() — translation helper
# ---------------------------------------------------------------------------
_lang: str = "en"


def set_lang(lang: str) -> None:
    """Set the global language for CLI output."""
    global _lang
    _lang = lang


def tr(key: str, *args) -> str:
    """Translate a TEXT_DB key to the current language."""
    lang_dict = TEXT_DB.get(key, {})
    text = lang_dict.get(_lang, lang_dict.get("en", key))
    if args:
        return text.format(*args)
    return text


# ---------------------------------------------------------------------------
# Task execution functions
# ---------------------------------------------------------------------------
def stt_fun(params: dict) -> None:
    """Execute speech-to-text task."""
    from videotrans.configure.config import app_cfg
    from videotrans.task.speech2text import SpeechToText
    from videotrans.task.taskcfg import TaskCfgSTT

    print(f"\n{tr('exec_stt_task')}")
    print(tr('process_file', params.get('name')))
    try:
        trk = SpeechToText(cfg=TaskCfgSTT(**params), out_format='srt')
        trk.prepare()
        trk.recogn()
        trk.diariz()
        trk.task_done()
        print(tr('done'))
    except Exception as e:
        print(tr('failed', str(e)), file=sys.stderr)
        raise


def tts_fun(params: dict) -> None:
    """Execute text-to-speech task."""
    from videotrans.task.dubbing import DubbingSrt
    from videotrans.task.taskcfg import TaskCfgTTS

    print(f"\n{tr('exec_tts_task')}")
    print(tr('process_file', params.get('name')))
    try:
        trk = DubbingSrt(cfg=TaskCfgTTS(**params), out_ext='wav')
        trk.prepare()
        trk.dubbing()
        trk.align()
        trk.task_done()
        print(tr('done'))
    except Exception as e:
        print(tr('failed', str(e)), file=sys.stderr)
        raise


def sts_fun(params: dict) -> None:
    """Execute subtitle translation task."""
    from videotrans.task.translate_srt import TranslateSrt
    from videotrans.task.taskcfg import TaskCfgSTS

    print(f"\n{tr('exec_sts_task')}")
    print(tr('process_file', params.get('name')))
    try:
        trk = TranslateSrt(cfg=TaskCfgSTS(**params), out_format=0)
        trk.prepare()
        trk.trans()
        trk.task_done()
        print(tr('done'))
    except Exception as e:
        print(tr('failed', str(e)), file=sys.stderr)
        raise


def vtv_fun(params: dict) -> None:
    """Execute full video translation task."""
    from videotrans.task.orchestrator import EventKind, TaskRequest, TaskStatus, run

    print(f"\n{tr('exec_vtv_task')}")
    print(tr('process_file', params.get('name')))

    def print_event(event):
        if event.kind == EventKind.STAGE_STARTED:
            print(f"[{event.stage}]")
        elif event.kind == EventKind.LOG and event.message:
            print(event.message)

    result = run(TaskRequest(params), print_event)
    if result.status == TaskStatus.FAILED:
        raise RuntimeError(result.failure.message) from None
    if result.status == TaskStatus.CANCELLED:
        raise KeyboardInterrupt
    print(tr('done'))


# ---------------------------------------------------------------------------
# List functions
# ---------------------------------------------------------------------------
def list_providers() -> None:
    """Print available providers for all categories."""
    from videotrans import recognition, translator, tts

    print(tr('list_providers_header'))
    for i, name in enumerate(recognition.RECOGN_NAME_LIST):
        print(f"  {i:2d} = {name}")

    print(tr('list_trans_header'))
    for i, name in enumerate(translator.TRANSLASTE_NAME_LIST):
        print(f"  {i:2d} = {name}")

    print(tr('list_tts_header'))
    for i, name in enumerate(tts.TTS_NAME_LIST):
        print(f"  {i:2d} = {name}")


def list_languages() -> None:
    """Print available language codes."""
    from videotrans import translator

    print(tr('list_languages_header'))
    for code, name in translator.LANGNAME_DICT.items():
        print(f"  {code:10s}  {name}")


def list_models() -> None:
    """Print available faster-whisper models."""
    from videotrans.configure.contants import FASTER_MODELS_DICT

    print(tr('list_models_header'))
    for name, repo in FASTER_MODELS_DICT.items():
        print(f"  {name:25s}  {repo}")


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=tr("cli_desc"),
        epilog=tr("cli_epilog"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--version', action='version', version='%(prog)s 4.03')

    parser.add_argument('--task', type=str, choices=['stt', 'tts', 'sts', 'vtv'],
                        help=tr("help_task"))
    parser.add_argument('--name', type=str, help=tr("help_name"))

    parser.add_argument('--list', type=str, choices=['providers', 'languages', 'models'],
                        help=tr("help_list"))
    parser.add_argument('--output-dir', type=str, default=None,
                        help=tr("help_output_dir"))
    parser.add_argument('--log-level', type=str, default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help=tr("help_log_level"))
    parser.add_argument('--verbose', '-v', action='store_true', help=tr("help_verbose"))
    parser.add_argument('--quiet', '-q', action='store_true', help=tr("help_quiet"))

    # --- STT ---
    stt_group = parser.add_argument_group(tr("group_stt"))
    stt_group.add_argument('--recogn_type', type=int, default=1, help=tr("help_recogn_type"))
    stt_group.add_argument('--detect_language', type=str, default='auto', help=tr("help_detect_lang"))
    stt_group.add_argument('--model_name', type=str, default='nova-3', help=tr("help_model_name", 'large-v3'))
    stt_group.add_argument('--cuda', action='store_true', help=tr("help_cuda"))
    stt_group.add_argument('--remove_noise', action='store_true', help=tr("help_remove_noise"))
    stt_group.add_argument('--enable_diariz', action='store_true', help=tr("help_enable_diariz"))
    stt_group.add_argument('--nums_diariz', type=int, default=-1, help=tr("help_nums_diariz"))
    stt_group.add_argument('--rephrase', type=int, default=0, help=tr("help_rephrase"))
    stt_group.add_argument('--fix_punc', action='store_true', help=tr("help_fix_punc"))
    stt_group.add_argument('--subtitle_source', type=str, default='audio_asr', choices=['audio_asr', 'video_ocr'], help=tr("help_subtitle_source"))
    stt_group.add_argument('--ocr_roi', type=str, default=None, help=tr("help_ocr_roi"))
    stt_group.add_argument('--ocr_interval_ms', type=int, default=500, help=tr("help_ocr_interval_ms"))

    # --- TTS ---
    tts_group = parser.add_argument_group(tr("group_tts"))
    from videotrans import tts
    tts_group.add_argument('--tts_type', type=int, default=tts.DEFAULT_TTS, help=tr("help_tts_type"))
    tts_group.add_argument('--voice_role', type=str, default=None, help=tr("help_voice_role"))
    tts_group.add_argument('--voice_rate', type=str, default='+0%', help=tr("help_voice_rate"))
    tts_group.add_argument('--volume', type=str, default='+0%', help=tr("help_volume"))
    tts_group.add_argument('--pitch', type=str, default='+0Hz', help=tr("help_pitch"))
    tts_group.add_argument('--voice_autorate', action='store_true', help=tr("help_voice_autorate"))
    tts_group.add_argument('--align_sub_audio', action='store_true', help=tr("help_align_sub_audio"))

    # --- Translation ---
    trans_group = parser.add_argument_group(tr("group_trans"))
    trans_group.add_argument('--translate_type', type=int, default=0, help=tr("help_translate_type"))
    trans_group.add_argument('--source_language_code', type=str, default=None, help=tr("help_source_lang"))
    trans_group.add_argument('--target_language_code', type=str, default=None, help=tr("help_target_lang"))

    # --- VTV extra ---
    vtv_group = parser.add_argument_group(tr("group_vtv"))
    vtv_group.add_argument('--video_autorate', action='store_true', help=tr("help_video_autorate"))
    vtv_group.add_argument('--is_separate', action='store_true', help=tr("help_is_separate"))
    vtv_group.add_argument('--recogn2pass', action='store_true', help=tr("help_recogn2pass"))
    vtv_group.add_argument('--subtitle_type', type=int, default=1, help=tr("help_subtitle_type"))
    vtv_group.add_argument('--clear_cache', action='store_true', default=True, help=tr("help_clear_cache"))
    vtv_group.add_argument('--no-clear-cache', dest='clear_cache', action='store_false', help=tr("help_no_clear_cache"))

    return parser


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
def _parse_ocr_roi(raw: Any) -> Tuple[float, float, float, float]:
    """Parse and validate normalized ROI x,y,w,h from string or sequence."""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        raise ValueError("Invalid ROI")
    if len(parts) != 4:
        raise ValueError("ROI must have 4 elements")
    vals: List[float] = []
    for p in parts:
        if isinstance(p, bool):
            raise ValueError("Invalid ROI element")
        f = float(p)
        if not math.isfinite(f):
            raise ValueError("Non-finite ROI element")
        vals.append(f)
    x, y, w, h = vals
    if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0 or x + w > 1.001 or y + h > 1.001:
        raise ValueError("ROI out of bounds")
    if x + w > 1.0:
        w = 1.0 - x
    if y + h > 1.0:
        h = 1.0 - y
    if w <= 0.0 or h <= 0.0:
        raise ValueError("ROI out of bounds")
    return (x, y, w, h)


def validate_task_params(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate required parameters for the given task type."""
    if not args.name:
        parser.error("--name is required")

    if not Path(args.name).exists():
        parser.error(tr("err_file_not_found", args.name))

    if args.task == 'tts' and not args.voice_role:
        parser.error(tr("err_tts_role_required"))

    if args.task == 'sts' and not args.target_language_code:
        parser.error(tr("err_sts_target_required"))

    if args.task == 'vtv':
        missing = []
        if not args.source_language_code:
            missing.append(tr("miss_source_lang"))
        if not args.target_language_code:
            missing.append(tr("miss_target_lang"))
        if missing:
            parser.error(tr("err_vtv_missing", ', '.join(missing)))

    sub_src = getattr(args, "subtitle_source", "audio_asr")
    raw_roi = getattr(args, "ocr_roi", None)
    if args.task in {"stt", "vtv"} and sub_src == "video_ocr":
        if not raw_roi:
            parser.error(tr("err_ocr_roi_required"))
        try:
            _parse_ocr_roi(raw_roi)
        except (ValueError, TypeError):
            parser.error(tr("err_ocr_roi_invalid"))
    elif raw_roi is not None:
        try:
            _parse_ocr_roi(raw_roi)
        except (ValueError, TypeError):
            parser.error(tr("err_ocr_roi_invalid"))


# ---------------------------------------------------------------------------
# Common parameter building
# ---------------------------------------------------------------------------
def build_common_params(args: argparse.Namespace, output_dir: Optional[str] = None) -> dict:
    """Build common parameters dict from parsed args."""
    from videotrans.configure.config import ROOT_DIR, TEMP_DIR
    from videotrans.util import tools
    from videotrans.util.gpus import getset_gpu

    _file_obj = tools.format_video(Path(args.name).absolute().as_posix())
    _nospacebasename = re.sub(r'[\s. #*?!:"]', '-', _file_obj["basename"])
    _cache_folder = f'{TEMP_DIR}/{_file_obj["uuid"]}'

    if output_dir:
        _target_dir = str(Path(output_dir).absolute())
    else:
        _target_dir = f'{ROOT_DIR}/output/{_nospacebasename}'

    _file_obj['target_dir'] = _target_dir

    common_params = {'name': args.name, "cache_folder": _cache_folder}
    common_params.update(asdict(_file_obj))

    return common_params


def build_stt_params(args: argparse.Namespace) -> dict:
    """Build STT-specific parameters."""
    params: Dict[str, Any] = {
        "recogn_type": args.recogn_type,
        "detect_language": args.detect_language,
        "model_name": args.model_name,
        "is_cuda": args.cuda,
        "remove_noise": args.remove_noise,
        "enable_diariz": args.enable_diariz,
        "nums_diariz": args.nums_diariz,
        "rephrase": args.rephrase,
        "fix_punc": args.fix_punc,
    }
    raw_sub_src = getattr(args, "subtitle_source", None)
    if isinstance(raw_sub_src, str):
        params["subtitle_source"] = raw_sub_src
        raw_roi = getattr(args, "ocr_roi", None)
        roi_tuple = _parse_ocr_roi(raw_roi) if isinstance(raw_roi, (str, list, tuple)) else None
        if roi_tuple is not None:
            params["ocr_roi"] = roi_tuple
            params["ocr_roi_confirmed"] = True
        elif raw_sub_src == "video_ocr":
            params["ocr_roi"] = None
            params["ocr_roi_confirmed"] = False
        raw_interval = getattr(args, "ocr_interval_ms", None)
        if isinstance(raw_interval, (int, float)) and not isinstance(raw_interval, bool):
            params["ocr_options"] = {"coarse_interval_ms": max(100, min(5000, int(raw_interval)))}
    return params


def build_tts_params(args: argparse.Namespace) -> dict:
    """Build TTS-specific parameters."""
    return {
        "tts_type": args.tts_type,
        "voice_role": args.voice_role,
        "voice_rate": args.voice_rate,
        "volume": args.volume,
        "pitch": args.pitch,
        "is_cuda": args.cuda,
        "voice_autorate": args.voice_autorate,
        "align_sub_audio": args.align_sub_audio,
        "target_language_code": args.target_language_code,
    }


def build_sts_params(args: argparse.Namespace) -> dict:
    """Build STS-specific parameters."""
    return {
        "translate_type": args.translate_type,
        "source_language_code": args.source_language_code or "auto",
        "target_language_code": args.target_language_code,
    }


def build_vtv_params(args: argparse.Namespace) -> dict:
    """Build VTV-specific parameters."""
    return {
        "source_language_code": args.source_language_code,
        "target_language_code": args.target_language_code,
        **build_stt_params(args),
        **{k: v for k, v in build_tts_params(args).items()
           if k not in ('target_language_code', 'is_cuda')},
        "is_cuda": args.cuda,
        "translate_type": args.translate_type,
        "is_separate": args.is_separate,
        "recogn2pass": args.recogn2pass,
        "subtitle_type": args.subtitle_type,
        "clear_cache": args.clear_cache,
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_level: str, verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level for the application."""
    import logging

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.INFO
    else:
        level = getattr(logging, log_level.upper(), logging.WARNING)

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main() -> int:
    """Main CLI entry point. Returns exit code (0=success, 1=error)."""
    # Parse language from system before anything else
    from videotrans.configure import config
    config.init_run()
    from videotrans.configure.config import defaulelang, app_cfg

    # Set language for CLI output
    set_lang(defaulelang if defaulelang in ('zh', 'en') else 'en')

    # Build parser and parse args
    parser = build_parser()
    args = parser.parse_args()

    # Handle --list before other validation
    if args.list:
        if args.list == 'providers':
            list_providers()
        elif args.list == 'languages':
            list_languages()
        elif args.list == 'models':
            list_models()
        return 0

    # Require --task when not using --list
    if not args.task:
        parser.error(tr("err_missing_task"))

    # Setup logging
    setup_logging(args.log_level, verbose=args.verbose, quiet=args.quiet)

    # Validate parameters
    validate_task_params(args, parser)

    # Set runtime flags
    app_cfg.exit_soft = False
    app_cfg.exec_mode = 'cli'

    # Get GPU info
    from videotrans.util.gpus import getset_gpu
    getset_gpu()

    # Build common params
    common_params = build_common_params(args, output_dir=args.output_dir)

    # Dispatch to task function
    task_map = {
        'stt': lambda: stt_fun({**common_params, **build_stt_params(args)}),
        'tts': lambda: tts_fun({**common_params, **build_tts_params(args)}),
        'sts': lambda: sts_fun({**common_params, **build_sts_params(args)}),
        'vtv': lambda: vtv_fun({**common_params, **build_vtv_params(args)}),
    }

    try:
        task_map[args.task]()
        print(tr('output_dir', common_params.get('target_dir', '')))
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(tr('failed', str(e)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    sys.exit(main())
