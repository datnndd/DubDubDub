# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from videotrans.configure.config import ROOT_DIR

DEFAULT_SYSTEM_PROMPT = "You are a top-tier Subtitle Translation Engine."


@lru_cache(maxsize=64)
def _read_text_cached(path_str: str, mtime_ns: int, size: int) -> str:
    return Path(path_str).read_text(encoding="utf-8-sig", errors="ignore")


def _stat_read_text(path: Path) -> str:
    st = path.stat()
    return _read_text_cached(str(path), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=16)
def _parse_glossary_cached(
    path_str: str, mtime_ns: int, size: int
) -> tuple[tuple[str, str], ...]:
    raw = _read_text_cached(path_str, mtime_ns, size).strip()
    if not raw:
        return ()
    entries: list[tuple[str, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        src, dst = stripped.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if src and dst:
            entries.append((src, dst))
    return tuple(entries)


@lru_cache(maxsize=64)
def _get_prompt_file_cached(root_dir: str, ainame: str, aisendsrt: bool = True) -> str:
    folder = "prompts/srt/" if aisendsrt else "prompts/text/"
    return f"{root_dir}/videotrans/{folder}{ainame}.txt"


# Get the prompt txt file needed for current operation
def get_prompt_file(ainame, aisendsrt=True):
    return _get_prompt_file_cached(ROOT_DIR, ainame, bool(aisendsrt))


def load_prompt_template(ainame: str, aisendsrt: bool = True) -> str:
    name = (ainame or "chatgpt").strip() or "chatgpt"
    prompt_path = Path(get_prompt_file(ainame=name, aisendsrt=aisendsrt))
    if not prompt_path.is_file():
        prompt_path = Path(get_prompt_file(ainame="chatgpt", aisendsrt=aisendsrt))
    if not prompt_path.is_file():
        return ""
    return _stat_read_text(prompt_path)


def load_resegment_template(step: str = "") -> str:
    path = Path(ROOT_DIR) / "videotrans" / "prompts" / "resegment" / f"llm{step}.txt"
    if not path.is_file():
        return ""
    return _stat_read_text(path)


def load_glossary_entries() -> tuple[tuple[str, str], ...]:
    glossary_path = Path(ROOT_DIR) / "videotrans" / "glossary.txt"
    if not glossary_path.is_file():
        return ()
    st = glossary_path.stat()
    return _parse_glossary_cached(str(glossary_path), st.st_mtime_ns, st.st_size)


def filter_glossary_for_text(
    entries: Sequence[tuple[str, str]], batch_text: str
) -> list[tuple[str, str]]:
    if not entries or not batch_text:
        return []
    batch_cf = batch_text.casefold()
    matched: list[tuple[str, str]] = []
    for src, dst in entries:
        if not src:
            continue
        if len(src) <= 2 and src.isascii() and src.isalnum():
            pattern = rf"(?<![A-Za-z0-9]){re.escape(src)}(?![A-Za-z0-9])"
            if re.search(pattern, batch_text, re.IGNORECASE):
                matched.append((src, dst))
        elif src.casefold() in batch_cf:
            matched.append((src, dst))
    return matched


def format_glossary_block(matched: Sequence[tuple[str, str]]) -> str:
    if not matched:
        return ""
    rows = "\n".join(f"| {src} | {dst} |" for src, dst in matched)
    return (
        "\n\n# Glossary\n"
        "Use these exact term translations:\n"
        "| Term | Translation |\n"
        "| --- | --- |\n"
        f"{rows}\n\n"
    )


def split_prompt_template(template: str) -> tuple[str, str]:
    text = template or ""
    split_idx = -1
    for marker in ("{GLOSSARY_DICT}", "# ACTUAL TASK", "# TASK"):
        idx = text.find(marker)
        if idx != -1:
            split_idx = idx
            break
    if split_idx == -1:
        idx = text.rfind("<INPUT>")
        if idx != -1:
            split_idx = idx
    if split_idx == -1:
        idx = text.rfind("{batch_input}")
        if idx != -1:
            split_idx = idx

    if split_idx != -1:
        system_part = text[:split_idx]
        user_part = text[split_idx:]
    else:
        system_part = ""
        user_part = text

    # Trim trailing separator lines from static system instructions
    system_part = re.sub(r"(?:\r?\n\s*---\s*)+\s*$", "", system_part).strip()
    # Guarantee dynamic placeholders never leak into static system prompt
    for forbidden in ("{GLOSSARY_DICT}", "{batch_input}", "{context_block}"):
        if forbidden in system_part:
            system_part = system_part.replace(forbidden, "").strip()

    if not system_part:
        system_part = DEFAULT_SYSTEM_PROMPT

    user_part = user_part.strip()
    if "{batch_input}" not in user_part:
        user_part = f"{user_part}\n\n<INPUT>\n{{batch_input}}\n</INPUT>".strip()

    return system_part, user_part


def build_prompt_parts(
    ainame: str, aisendsrt: bool = True, *, lang: str = ""
) -> tuple[str, str]:
    template = load_prompt_template(ainame=ainame, aisendsrt=aisendsrt)
    if lang:
        template = template.replace("{lang}", lang)
    return split_prompt_template(template)


def render_user_prompt(
    user_template: str,
    *,
    batch_input: str,
    glossary_block: str = "",
    context_text: str = "",
) -> str:
    rendered = user_template or "<INPUT>\n{batch_input}\n</INPUT>"
    if "{GLOSSARY_DICT}" in rendered:
        rendered = rendered.replace("{GLOSSARY_DICT}", glossary_block or "")
        if not glossary_block:
            rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    elif glossary_block:
        rendered = f"{glossary_block.strip()}\n\n{rendered}"

    if context_text and context_text.strip():
        ctx_block = (
            '<CONTEXT_PREVIOUS_TRANSLATIONS note="Read-only prior translations for '
            'terminology and pronoun continuity. Do NOT repeat or translate these '
            'lines; output ONLY the items inside <INPUT>.">\n'
            f"{context_text.strip()}\n"
            "</CONTEXT_PREVIOUS_TRANSLATIONS>\n\n"
        )
        m_input = re.search(r"(?m)^<INPUT>\s*$", rendered)
        if m_input:
            pos = m_input.start()
            rendered = rendered[:pos] + ctx_block + rendered[pos:]
        elif "<INPUT>" in rendered:
            pos = rendered.rfind("<INPUT>")
            rendered = rendered[:pos] + ctx_block + rendered[pos:]
        else:
            rendered = f"{ctx_block}{rendered}"

    if "{batch_input}" in rendered:
        rendered = rendered.replace("{batch_input}", batch_input)
    else:
        rendered = f"{rendered}\n{batch_input}"

    return rendered.strip()


def _strip_outer_fence(text: str) -> str:
    s = text.strip()
    m = re.match(r"^\s*```[a-zA-Z0-9_-]*\s*\r?\n?(.*?)\r?\n?```\s*$", s, flags=re.S)
    return m.group(1).strip() if m else s


def clean_ai_response(raw: str, tag: str = "TRANSLATE_TEXT") -> str:
    cleaned = (raw or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(
        rf"<think>.*?(?=<{re.escape(tag)}\b|$)", "", cleaned, flags=re.I | re.S
    )
    if "</think>" in cleaned.lower():
        cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.I | re.S)
    cleaned = _strip_outer_fence(cleaned)
    m = re.search(
        rf"<{re.escape(tag)}\b[^>]*>(.*?)(?:</{re.escape(tag)}>|$)",
        cleaned,
        flags=re.I | re.S,
    )
    if m:
        cleaned = m.group(1).strip()
    cleaned = _strip_outer_fence(cleaned)
    return cleaned.strip()


# Get prompt instructions (legacy compatibility)
def get_prompt(ainame, aisendsrt=True):
    content = load_prompt_template(ainame=ainame, aisendsrt=aisendsrt)
    entries = load_glossary_entries()
    glossary = ""
    if entries:
        rows = "\n".join(f"|{src}|{dst}|" for src, dst in entries)
        glossary = (
            "\n\n# Glossary of terms\n"
            "Translations are made strictly according to the following glossary. "
            "If a term appears in a sentence, the corresponding translation must be used, "
            "not a free translation:\n"
            "| Glossary | Translation |\n"
            "| --------- | ----- |\n"
            f"{rows}\n\n"
        )
    return content.replace("{GLOSSARY_DICT}", glossary)


def qwenmt_glossary():
    entries = load_glossary_entries()
    if not entries:
        return None
    return [{"source": src, "target": dst} for src, dst in entries]
