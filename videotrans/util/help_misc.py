# -*- coding: utf-8 -*-
import hashlib
import json
import os
import re
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import is_dataclass, asdict
from functools import lru_cache
from pathlib import Path

from videotrans import VERSION
from videotrans.configure.config import tr, app_cfg, logger, ROOT_DIR, defaulelang
from videotrans.util.network import set_proxy, process_openai_api, is_connect_hf
from videotrans.util.prompts import get_prompt, qwenmt_glossary, get_prompt_file


def open_url(url: str = None):
    import webbrowser
    title_url_dict = {
        'bbs': "https://bbs.pyvideotrans.com",
        'ffmpeg': "https://www.ffmpeg.org/download.html",
        'git': "https://github.com/jianchang512/pyvideotrans",
        'issue': "https://github.com/jianchang512/pyvideotrans/issues",
        'hfmirrorcom': "https://pyvideotrans.com/819",
        'models': "https://github.com/jianchang512/stt/releases/tag/0.0",
        'stt': "https://github.com/jianchang512/stt/",
        'gtrans': "https://pyvideotrans.com/ocrsp",
        'cuda': "https://pyvideotrans.com/gpu",
        'website': "https://pyvideotrans.com",
        'help': "https://pyvideotrans.com",
        'xinshou': "https://pyvideotrans.com/getstart",
        "about": "https://pyvideotrans.com/about",
        'download': "https://github.com/jianchang512/pyvideotrans/releases",
    }
    if url and url.startswith("http"):
        return webbrowser.open_new_tab(url)
    if url and url in title_url_dict:
        return webbrowser.open_new_tab(title_url_dict[url])
    return


def vail_file(file=None):
    if not file:
        return False
    p = Path(file)
    if not p.exists() or not p.is_file():
        return False
    if p.stat().st_size == 0:
        return False
    return True


def shutdown_system():
    # Get current operating system type
    system = platform.system()

    if system == "Windows":
        # Windows shutdown command
        subprocess.call("shutdown /s /t 0", shell=True)
    elif system == "Linux":
        # Linux shutdown command
        subprocess.call("poweroff")
    elif system == "Darwin":
        # macOS shutdown command
        subprocess.call("sudo shutdown -h now", shell=True)
    else:
        logger.error(f"Unsupported system: {system}")


# Check whether novoice.mp4 is ready
def is_novoice_mp4(novoice_mp4, noextname, uuid=None, *, event_sink=None, cancellation_token=None):
    # Pre-created
    # Check if novoice_mp4 is completed
    job_uuid = uuid or noextname

    def _is_aborted() -> bool:
        if app_cfg.exit_soft:
            return True
        if cancellation_token is not None:
            return bool(cancellation_token.is_cancelled())
        return bool(noextname and noextname in app_cfg.stoped_uuid_set)

    if _is_aborted():
        return False

    status = app_cfg.queue_novice.get(noextname)
    if status is None and vail_file(novoice_mp4):
        return True
    if status == 'end':
        return True
    if status == 'cancelled':
        return False
    if isinstance(status, str) and status.startswith('error:'):
        raise RuntimeError(f"{noextname} split no voice {status}")

    start_t = time.monotonic()
    last_emit_t = None
    last_size = 0
    while True:
        if _is_aborted():
            return False
        if noextname not in app_cfg.queue_novice:
            raise RuntimeError(f"{noextname} split no voice videoerror-1")
        status = app_cfg.queue_novice[noextname]
        if status == 'end':
            return True
        if status == 'cancelled':
            return False
        if isinstance(status, str) and status.startswith('error:'):
            raise RuntimeError(f"{noextname} split no voice {status}")

        if vail_file(novoice_mp4):
            current_size = os.path.getsize(novoice_mp4)
            if 0 < last_size == current_size and (time.monotonic() - start_t) > 1200:
                return True
            last_size = current_size

        if status == 'ing':
            now = time.monotonic()
            if event_sink and (last_emit_t is None or (now - last_emit_t) >= 1.0):
                size = f'{round(last_size / 1024 / 1024, 2)}MB' if last_size > 0 else ""
                msg = f"{noextname} {tr('spilt audio and video')} {size}"
                event_sink({"uuid": job_uuid, "text": msg, "action": "novoice_wait"})
                last_emit_t = now
            time.sleep(0.2)
            continue
        return False


# Process string with MD5 hashing
@lru_cache
def get_md5(input_string: str):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()


# Play audio
def pygameaudio(filepath):
    import os
    if not os.path.exists(filepath):
        return
    import pygame
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception as e:
        logger.exception(f'pygame play error:{e}', exc_info=True)


def read_last_n_lines(filename, n=100):
    if not Path(filename).exists():
        return []
    from collections import deque
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # Use deque to retain only the last n lines
            last_lines = deque(file, maxlen=n)
        return list(last_lines)  # Return as a list
    except FileNotFoundError:
        return []
    except Exception:
        return []


# Serialization
def serial(data: object) -> str:
    if not isinstance(data, list):
        return json.dumps(asdict(data) if is_dataclass(data) else data)
    _newlist = []
    for it in data:
        _newlist.append(asdict(it) if is_dataclass(it) else it)
    return json.dumps(_newlist)


def check_new_version():
    # Check latest version info
    try:
        import requests
        # Pure static file, returns only version info string
        # Only gets current software version number and operating system type (win32/macos/linux)
        url = f"https://pyvideotrans.com/version.json?version={VERSION}&os={sys.platform}"
        res = requests.get(url)
        res.raise_for_status()
        d = res.json()
        app_cfg.new_version_pvt = d['version']
    except Exception:
        pass


def _get_type_name(type_index, name_list):
    if type_index is None or type_index >= len(name_list):
        return '-'
    return name_list[type_index]


@lru_cache
def get_recogn_type(type_index=None):
    from videotrans.recognition import RECOGN_NAME_LIST
    return _get_type_name(type_index, RECOGN_NAME_LIST)


@lru_cache
def get_tanslate_type(type_index=None):
    from videotrans.translator import TRANSLASTE_NAME_LIST
    return _get_type_name(type_index, TRANSLASTE_NAME_LIST)


@lru_cache
def get_tts_type(type_index=None):
    from videotrans.tts import TTS_NAME_LIST
    return _get_type_name(type_index, TTS_NAME_LIST)


def atomic_write_json(data, target_path):
    target = Path(target_path)

    # Create temporary file
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix='.tmp', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    # Atomic replace (even if process crashes, original file remains intact)
    os.replace(tmp_name, target_path)
