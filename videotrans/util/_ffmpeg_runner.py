# -*- coding: utf-8 -*-
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from videotrans.configure.config import ROOT_DIR, tr, app_cfg, settings, logger


def extract_concise_error(stderr_text: str) -> str:
    if not stderr_text:
        return "Unknown error (empty stderr)"

    lines = stderr_text.strip().splitlines()
    if not lines:
        return "Unknown error (empty stderr lines)"

    result = re.findall(r'Error\s(.*?)\n', stderr_text)
    if not result:
        return " ".join(lines[-10:])
    return " ".join(result)


def runffmpeg(arg, *, noextname=None, force_cpu=True, cmd_dir=None, job_id=None, cancellation_token=None):
    if settings.get('force_lib'):
        force_cpu = True

    final_args = arg

    cmd = ['ffmpeg', "-hide_banner", "-nostdin", "-ignore_unknown", '-threads', '0']
    if "-y" not in final_args:
        cmd.append("-y")
    cmd.extend(final_args)

    if cmd and Path(cmd[-1]).suffix:
        cmd[-1] = Path(cmd[-1]).as_posix()

    if settings.get('ffmpeg_cmd'):
        custom_params = [p for p in settings.get('ffmpeg_cmd', '').split(' ') if p]
        cmd = cmd[:-1] + custom_params + cmd[-1:]

    from videotrans.core.proc_registry import (
        get_current_job_id,
        register_proc,
        unregister_proc,
        kill_process_tree,
    )

    active_job_id = job_id or get_current_job_id()
    proc = None
    try:
        if app_cfg.exit_soft or (cancellation_token and cancellation_token.is_cancelled()):
            return False

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors='replace',
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            cwd=cmd_dir,
        )
        if active_job_id:
            register_proc(active_job_id, proc)

        stdout = ""
        stderr = ""
        while True:
            if app_cfg.exit_soft or (cancellation_token and cancellation_token.is_cancelled()):
                kill_process_tree(proc)
                try:
                    out, err = proc.communicate(timeout=1.0)
                    stdout += out or ""
                    stderr += err or ""
                except Exception:
                    pass
                from videotrans.configure.excepts import FFmpegError
                raise FFmpegError("FFmpeg execution was cancelled")

            try:
                out, err = proc.communicate(timeout=0.2)
                stdout += out or ""
                stderr += err or ""
                break
            except subprocess.TimeoutExpired:
                continue

        if app_cfg.exit_soft or (cancellation_token and cancellation_token.is_cancelled()):
            from videotrans.configure.excepts import FFmpegError
            raise FFmpegError("FFmpeg execution was cancelled")

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)

        if noextname:
            app_cfg.queue_novice[noextname] = "end"
        return True
    except FileNotFoundError as e:
        logger.error(f"命令未找到: {cmd[0]}。请确保 ffmpeg 已安装并在系统 PATH 中。")
        if noextname:
            app_cfg.queue_novice[noextname] = f"error:{e}"
        raise
    except subprocess.CalledProcessError as e:
        error_message = e.stderr or ""
        logger.error(f"FFmpeg 命令执行失败 (force_cpu={force_cpu})。\n命令: {' '.join(cmd)}\n错误: {error_message} {e.stdout}")
        err = extract_concise_error(e.stderr)
        if noextname:
            app_cfg.queue_novice[noextname] = f"error:{err}"
        if sys.platform == 'win32' and 'No such file or directory' in str(e):
            _err = get_filepath_from_cmd(cmd)
            err = _err or err
        from videotrans.configure.excepts import FFmpegError
        raise FFmpegError(err) from e
    except Exception as e:
        if app_cfg.exit_soft or (cancellation_token and cancellation_token.is_cancelled()):
            if noextname:
                app_cfg.queue_novice[noextname] = "cancelled"
            raise
        if noextname:
            app_cfg.queue_novice[noextname] = f"error:{e}"
        logger.error(f"执行 ffmpeg 时发生未知错误,{cmd=}:\n{e}")
        raise
    finally:
        if proc and active_job_id:
            unregister_proc(active_job_id, proc)


def get_filepath_from_cmd(cmd: list):
    file_list = [cmd[i + 1] for i, param in enumerate(cmd) if param == '-i']
    file_list.append(cmd[-1])
    special = ['"', "'", "`", "*", "?", ":", ">", "<", "|", "\n", "\r"]
    for file in file_list:
        if len(file) >= 255:
            return tr(
                'The file path and file name may be too long. Please move the file to a flat and short directory and rename the file to a shorter name, ensuring that the length from the drive letter to the end of the file name does not exceed 255 characters: {},For example D:/videotrans/1.mp4 D:/videotrans/2.wav',
                file)
        for flag in special:
            if flag in file:
                return tr(
                    'There may be special characters in the file name or path. Please move the file to a simple directory consisting of English and numerical characters, rename the file to a simple name, and try again to avoid errors: {},For example D:/videotrans/1.mp4 D:/videotrans/2.wav',
                    file)
    return None
