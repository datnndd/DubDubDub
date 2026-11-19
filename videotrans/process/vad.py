import time
import traceback

import numpy as np
import scipy.io.wavfile as Wavfile
from ten_vad import TenVad

from videotrans.configure.config import logger


def get_speech_timestamp_silero(input_wav,
                                threshold=None,
                                min_speech_duration_ms=0,
                                max_speech_duration_ms=None,
                                min_silent_duration_ms=None):
    # Guard against invalid input
    min_speech_duration_ms = 0  # int(max(min_speech_duration_ms,0))
    min_silent_duration_ms = int(max(min_silent_duration_ms, 50))
    max_speech_duration_ms = int(min(max(max_speech_duration_ms, min_speech_duration_ms + 1000), 30000))
    logger.debug(
        f'[silero-VAD]: Segmentation params: {threshold=},{min_speech_duration_ms=}ms,{max_speech_duration_ms=}ms,{min_silent_duration_ms=}ms')

    sampling_rate = 16000
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import (
        VadOptions,
        get_speech_timestamps
    )
    vad_p = {
        "threshold": threshold,
        "min_speech_duration_ms": min_speech_duration_ms,
        "max_speech_duration_s": float(max_speech_duration_ms / 1000.0),
        "min_silence_duration_ms": min_silent_duration_ms,
    }

    def convert_to_milliseconds(timestamps):
        milliseconds_timestamps = []
        for timestamp in timestamps:
            milliseconds_timestamps.append(
                [
                    int(round(timestamp["start"] / sampling_rate * 1000)),
                    int(round(timestamp["end"] / sampling_rate * 1000)),
                ]
            )

        return milliseconds_timestamps

    speech_chunks = get_speech_timestamps(decode_audio(input_wav,
                                                       sampling_rate=sampling_rate),
                                          vad_options=VadOptions(**vad_p)
                                          )
    
    merged_segments=convert_to_milliseconds(speech_chunks)
    if not merged_segments:
        return [],None
    if merged_segments[0][0]<0:
        merged_segments[0][0]=0
    _vail_segments=[]
    for it in merged_segments:
        if it[1]>it[0] and it[0]>=0:
            _vail_segments.append(it)
    return _vail_segments,None




def _prepare_pcm_frames(data, hop_size=256):
    """Convert audio samples to mono int16 and reshape into contiguous (num_frames, hop_size) frames."""
    arr = np.asarray(data)
    orig_dtype = arr.dtype
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if np.issubdtype(orig_dtype, np.floating):
        arr = np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    elif orig_dtype == np.int32:
        arr = np.clip(arr / 65536.0, -32768.0, 32767.0).astype(np.int16)
    elif orig_dtype == np.uint8:
        arr = np.clip((arr - 128.0) * 256.0, -32768.0, 32767.0).astype(np.int16)
    elif orig_dtype == np.int8:
        arr = np.clip(arr * 256.0, -32768.0, 32767.0).astype(np.int16)
    elif orig_dtype == np.uint16:
        arr = np.clip(arr - 32768.0, -32768.0, 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = np.clip(arr, -32768, 32767).astype(np.int16)
    num_frames = arr.shape[0] // hop_size
    if num_frames <= 0:
        return np.empty((0, hop_size), dtype=np.int16), arr
    frames = np.ascontiguousarray(arr[:num_frames * hop_size], dtype=np.int16).reshape(num_frames, hop_size)
    return frames, arr


def _segments_from_flags(flags, min_silent_frames, max_speech_frames=None):
    """Run speech/silence state machine over a sequence of 0/1 speech flags."""
    segments = []
    triggered = False
    speech_start_frame = 0
    silence_frame_count = 0

    for i, is_speech in enumerate(flags):
        if triggered:
            if is_speech == 1:
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            is_silence_timeout = silence_frame_count >= min_silent_frames
            is_max_timeout = (
                max_speech_frames is not None
                and (i - speech_start_frame) >= max_speech_frames
            )

            if is_silence_timeout or is_max_timeout:
                end_frame = i if is_max_timeout else i - silence_frame_count
                segments.append([speech_start_frame, end_frame])
                triggered = False
                silence_frame_count = 0
        else:
            if is_speech == 1:
                triggered = True
                speech_start_frame = i
                silence_frame_count = 0

    if triggered:
        end_frame = len(flags) - silence_frame_count
        segments.append([speech_start_frame, end_frame])

    return segments


def get_speech_timestamp(input_wav=None,
                         threshold=None,
                         min_speech_duration_ms=None,
                         max_speech_duration_ms=None,
                         min_silent_duration_ms=None):
    st_ = time.time()
    
    try:
        sr, data = Wavfile.read(input_wav)
    except Exception as e:
        msg = traceback.format_exc()
        return False, f'{e} {msg}'

    # Dynamically calculate duration per frame
    hop_size = 256
    frame_duration_ms = (hop_size / sr) * 1000.0

    # Normalize parameters
    if threshold is None:
        threshold = 0.5
    else:
        threshold = float(threshold)
    min_speech_duration_ms = int(max(500, min_speech_duration_ms if min_speech_duration_ms else 1000))
    min_silent_duration_ms = int(max(50, min_silent_duration_ms if min_silent_duration_ms else 200))
    if max_speech_duration_ms is None:
        max_speech_duration_ms = 30000

    logger.debug(
        f'[Ten-VAD]: Segmentation params: {threshold=},{min_speech_duration_ms=}ms,{max_speech_duration_ms=}ms,{min_silent_duration_ms=}ms')

    frames, mono_pcm = _prepare_pcm_frames(data, hop_size=hop_size)
    num_frames = frames.shape[0]
    if num_frames <= 0:
        return False, None

    # Energy-adaptive threshold (cast to int32 to avoid int16 -32768 overflow)
    audio_energy = float(np.mean(np.abs(mono_pcm.astype(np.int32)))) if mono_pcm.size > 0 else 0.0
    adjusted_threshold = threshold
    if audio_energy > 10000:
        adjusted_threshold = max(threshold * 1.2, 0.3)
    elif audio_energy < 1000:
        adjusted_threshold = min(threshold * 0.8, 0.2)

    logger.debug(f'[Ten-VAD] Audio energy: {audio_energy}, adjusted threshold: {adjusted_threshold}')

    # --- Single-pass TenVad inference over contiguous frames ---
    min_sil_frames = max(1, int(min_silent_duration_ms / frame_duration_ms))
    ten_vad_instance = TenVad(hop_size, adjusted_threshold)
    probs = np.empty(num_frames, dtype=np.float64)
    flags = np.empty(num_frames, dtype=np.uint8)
    for i in range(num_frames):
        prob, is_speech = ten_vad_instance.process(frames[i])
        probs[i] = float(prob)
        flags[i] = 1 if is_speech == 1 else 0

    initial_segments = _segments_from_flags(flags.tolist(), min_sil_frames, max_speech_frames=None)

    if not initial_segments:
        # Completely no speech detected
        return False, None

    # Verify prob/flag consistency so mocked TenVad instances with dummy probs fall back safely
    can_reuse_probs = bool(
        np.all((probs >= float(np.float32(adjusted_threshold))) == (flags == 1))
    ) and (float(probs.max()) > 0.0 or not np.any(flags))

    # --- Process overlong segments ---
    # Use recursion queue to find micro-pauses; fallback to minimum energy cut
    max_speech_frames = max(1, int(max_speech_duration_ms / frame_duration_ms))
    min_speech_frames = max(1, int(min_speech_duration_ms / frame_duration_ms))
    segments_ms = []
    chunk_queue = [list(seg) for seg in initial_segments]

    while chunk_queue:
        s_frame, e_frame = chunk_queue.pop(0)
        dur_frames = e_frame - s_frame

        if dur_frames <= max_speech_frames:
            segments_ms.append([s_frame * frame_duration_ms, e_frame * frame_duration_ms])
            continue

        # Exceeds max duration, attempt to find breathing points / micro-pauses
        sub_segs = []
        # Try 3 tiers: standard silence -> half silence -> ultra-short silence (~30ms)
        test_conditions = [
            (1.0, 1.0),                     # Original parameters
            (0.5, 1.2),                     # Half silence duration, slightly stricter threshold
            (max(30 / min_silent_duration_ms, 0.2), 1.5)  # ~30ms, stricter threshold
        ]

        for sil_ratio, thresh_mult in test_conditions:
            test_sil_frames = max(1, int((min_silent_duration_ms * sil_ratio) / frame_duration_ms))
            test_thresh = min(adjusted_threshold * thresh_mult, 0.9)

            if can_reuse_probs:
                sub_flags = (
                    flags[s_frame:e_frame]
                    if test_thresh == adjusted_threshold
                    else (probs[s_frame:e_frame] >= float(np.float32(test_thresh))).astype(np.uint8)
                )
                temp_segs = _segments_from_flags(
                    sub_flags.tolist(), test_sil_frames, max_speech_frames=max_speech_frames
                )
            else:
                sub_data = frames[s_frame:e_frame].ravel()
                temp_segs = _detect_raw_segments(
                    sub_data, test_thresh, test_sil_frames,
                    max_speech_frames=max_speech_frames
                )
            if len(temp_segs) > 1:
                max_sub_dur = max((se - ss) for ss, se in temp_segs)
                if max_sub_dur < dur_frames:  # Successfully shortened
                    sub_segs = temp_segs
                    break

        if sub_segs:
            new_chunks = [[s_frame + ss, s_frame + se] for ss, se in sub_segs]
            chunk_queue = new_chunks + chunk_queue
        else:
            # Ultimate fallback: find minimum energy point in safety zone to cut
            # Keep both halves >= min_speech_frames when dur_frames >= 2 * min_speech_frames
            half_max = max(1, int(max_speech_frames * 0.5))
            if dur_frames >= 2 * min_speech_frames and max_speech_frames >= min_speech_frames:
                search_start = max(half_max, min_speech_frames)
                search_end = min(max_speech_frames, dur_frames - min_speech_frames)
                if search_end > search_start:
                    window = frames[s_frame + search_start : s_frame + search_end]
                    energies = np.abs(window.astype(np.int32)).sum(axis=1)
                    best_cut_idx = search_start + int(np.argmin(energies))
                else:
                    best_cut_idx = min(max_speech_frames, max(min_speech_frames, dur_frames // 2))
            else:
                search_start = min(half_max, max(1, dur_frames - 1))
                search_end = min(max_speech_frames, dur_frames)
                if search_end > search_start:
                    window = frames[s_frame + search_start : s_frame + search_end]
                    energies = np.abs(window.astype(np.int32)).sum(axis=1)
                    best_cut_idx = search_start + int(np.argmin(energies))
                else:
                    best_cut_idx = min(max_speech_frames, max(1, dur_frames // 2))

            best_cut_idx = max(1, min(best_cut_idx, dur_frames - 1))
            cut_point = s_frame + best_cut_idx
            # Place latter half first, then former half, to ensure chronological processing
            chunk_queue.insert(0, [cut_point, e_frame])
            chunk_queue.insert(0, [s_frame, cut_point])

    logger.debug(f'[Ten-VAD] Initial segmentation and overlong processing took {int(time.time() - st_)}s')

    # --- Short segment merging ---
    # Merge short segments into neighbors ONLY when combined span <= max_speech_duration_ms
    segs = [seg.copy() for seg in segments_ms]
    # Filter out invalid segments (start >= end)
    segs = [[int(max(0, s)), int(max(0, e))] for s, e in segs if e > s]
    max_merged_span_ms = int(max_speech_duration_ms)

    # Merging algorithm: using stack, dynamically check if stack top segment is too short
    merged = []
    for seg in segs:
        if not merged:
            merged.append(seg)
            continue
        # Check if top segment of stack is too short
        while merged and (merged[-1][1] - merged[-1][0]) < min_speech_duration_ms:
            if (seg[1] - merged[-1][0]) <= max_merged_span_ms:
                prev = merged.pop()
                seg[0] = prev[0]
            else:
                break
        # Now check if current seg itself is too short
        if (seg[1] - seg[0]) < min_speech_duration_ms:
            if merged and (seg[1] - merged[-1][0]) <= max_merged_span_ms:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        else:
            merged.append(seg)

    _vail_segments = []
    for s, e in merged:
        if e > s and s >= 0:
            _vail_segments.append([max(0, s), max(0, e)])

    if not _vail_segments:
        return False, None

    logger.debug(f'[Ten-VAD] Segmentation and merging total time: {int(time.time() - st_)}s')
    return _vail_segments, None


def _detect_raw_segments(data, threshold, min_silent_frames, max_speech_frames=None):
    """
    Internal VAD detection.
    """
    hop_size = 256
    if threshold is None:
        threshold = 0.5
    frames, _ = _prepare_pcm_frames(data, hop_size=hop_size)
    num_frames = frames.shape[0]
    if num_frames <= 0:
        return []

    ten_vad_instance = TenVad(hop_size, threshold)
    flags = [0] * num_frames
    for i in range(num_frames):
        _, is_speech = ten_vad_instance.process(frames[i])
        flags[i] = 1 if is_speech == 1 else 0

    return _segments_from_flags(flags, min_silent_frames, max_speech_frames=max_speech_frames)

def get_speech_timestamp0(input_wav=None,
                         threshold=None,
                         min_speech_duration_ms=None,
                         max_speech_duration_ms=None,
                         min_silent_duration_ms=None):
    # Limit range
    # Min speech duration must not be less than 250ms
    min_speech_duration_ms = int(max(250, min_speech_duration_ms))
    # Silence threshold for cutting must not be less than 50ms
    min_silent_duration_ms = int(max(50, min_silent_duration_ms))

    logger.debug(
        f'[Ten-VAD]: Segmentation params: {threshold=},{min_speech_duration_ms=}ms,{max_speech_duration_ms=}ms,{min_silent_duration_ms=}ms')
    frame_duration_ms = 16
    hop_size = 256
    st_ = time.time()
    try:
        sr, data = Wavfile.read(input_wav)
    except Exception as e:
        msg = traceback.format_exc()
        return False,f'{e} {msg}'

    # Calculate audio energy for adaptive threshold adjustment
    audio_energy = np.mean(np.abs(data)) if len(data) > 0 else 0
    # Adjust threshold based on audio energy to handle high-noise situations
    adjusted_threshold = threshold
    if audio_energy > 10000:  # High-energy audio (possibly noisy)
        adjusted_threshold = max(threshold * 1.2, 0.3)  # Increase threshold
    elif audio_energy < 1000:  # Low-energy audio
        adjusted_threshold = min(threshold * 0.8, 0.2)  # Lower threshold

    logger.debug(f'[Ten-VAD] Audio energy: {audio_energy}, adjusted threshold: {adjusted_threshold}')

    min_sil_frames = min_silent_duration_ms / frame_duration_ms
    initial_segments = _detect_raw_segments(data, adjusted_threshold, min_sil_frames, max_speech_frames=None)

    # --- Phase 2: Refine overlong segments exceeding 2s ---
    refined_segments = []
    max_frames_limit = max_speech_duration_ms / frame_duration_ms
    tighter_min_sil_frames = (min_silent_duration_ms / 2) / frame_duration_ms
    _n = 0
    _len = len(initial_segments)
    for s, e in initial_segments:
        duration = e - s
        _n += 1
        # Re-split only needed if greater than 2000ms
        if duration > (max_frames_limit + 125):
            # Extract audio segment data
            sub_data = data[s * hop_size: e * hop_size]
            # Re-detect with halved silence threshold with max duration constraint
            sub_segs = _detect_raw_segments(sub_data, adjusted_threshold, tighter_min_sil_frames,
                                            max_speech_frames=max_frames_limit)

            for ss, se in sub_segs:
                refined_segments.append([s + ss, s + se])
        else:
            refined_segments.append([s, e])

    if not refined_segments:
        return False

    # --- Phase 3: Millisecond conversion & mandatory hard cut protection ---
    # Even with secondary subdivision, if someone speaks without pausing for 30s, force hard cut
    segments_ms = []
    for s, e in refined_segments:
        start_ms = int(s * frame_duration_ms)
        end_ms = int(e * frame_duration_ms)

        # Loop to ensure not exceeding max_speech_duration_ms
        curr_s = start_ms
        while (end_ms - curr_s) > max_speech_duration_ms:
            # Attempt to truncate at silence instead of abrupt hard cut
            # Calculate middle silence region of current block
            block_data = data[int(curr_s / 1000 * sr):int((curr_s + max_speech_duration_ms) / 1000 * sr)]
            # Search for last silence region
            block_segments = _detect_raw_segments(block_data, adjusted_threshold, min_sil_frames / 2,
                                                  max_speech_frames=None)
            if block_segments and len(block_segments) > 1:
                # If multiple segments exist, use start of last segment as truncation point
                last_segment_start = block_segments[-2][1] * hop_size / sr * 1000
                truncate_point = int(curr_s + last_segment_start)
                if truncate_point > curr_s + max_speech_duration_ms * 0.8:
                    segments_ms.append([curr_s, truncate_point])
                    curr_s = truncate_point
                    continue
            # If no suitable truncation point found, use hard truncation
            segments_ms.append([curr_s, curr_s + int(max_speech_duration_ms)])
            curr_s += int(max_speech_duration_ms)

        if end_ms - curr_s > 0:
            segments_ms.append([curr_s, end_ms])

    logger.debug(f'[Ten-VAD] Segmentation took {int(time.time() - st_)}s')

    speech_len = len(segments_ms)
    if speech_len <= 1:
        return segments_ms,None

    # --- Optimized segment merging strategy ---
    merged_segments = []
    # Minimum speech segment must not be below 500ms to prevent recognition errors
    min_speech_duration_ms = max(min_speech_duration_ms or 1000, 500)

    # Pass 1: Merge consecutive short segments
    temp_segments = []
    current_merge = None
    current_duration = 0

    for i, segment in enumerate(segments_ms):
        duration = segment[1] - segment[0]

        if duration < min_speech_duration_ms:
            # Short segment, needs merging
            if current_merge is None:
                current_merge = segment.copy()
                current_duration = duration
            else:
                # Calculate gap to currently merged segment
                gap = segment[0] - current_merge[1]
                # If gap is small, merge into current segment
                if gap < min_silent_duration_ms:
                    current_merge[1] = segment[1]
                    current_duration += duration + gap
                else:
                    # Gap is large, finalize current merge and start new one
                    temp_segments.append(current_merge)
                    current_merge = segment.copy()
                    current_duration = duration
        else:
            # Long segment, check for pending merge
            if current_merge is not None:
                # Calculate gap to previous merged segment
                gap = segment[0] - current_merge[1]
                # If gap is small, merge into current long segment
                if gap < min_silent_duration_ms * 1.5:
                    segment[0] = current_merge[0]
                else:
                    # Otherwise, append merged segment
                    temp_segments.append(current_merge)
                current_merge = None
                current_duration = 0
            temp_segments.append(segment)

    # Process final merged segment
    if current_merge is not None:
        temp_segments.append(current_merge)

    # Pass 2: Inspect merged segments to ensure no overshort segments remain
    seg_copy = [s[:] for s in temp_segments]
    for i, segment in enumerate(seg_copy):
        duration = segment[1] - segment[0]

        if duration >= min_speech_duration_ms:
            merged_segments.append(segment)
        else:
            # Still too short, try merging into adjacent segment
            if i == 0 and len(seg_copy) > 1:
                # First segment, merge into next
                seg_copy[i+1][0] = segment[0]
            elif i == len(seg_copy) - 1 and len(merged_segments) > 0:
                merged_segments[-1][1] = segment[1]
            elif len(merged_segments) > 0 and i < len(seg_copy) - 1:
                # Middle segment, merge into closer neighbor
                prev_gap = segment[0] - merged_segments[-1][1]
                next_gap = seg_copy[i+1][0] - segment[1]

                if prev_gap <= next_gap:
                    merged_segments[-1][1] = segment[1]
                else:
                    seg_copy[i+1][0] = segment[0]
            else:
                # Cannot merge, keep as standalone segment
                merged_segments.append(segment)

    if not merged_segments:
        return [],None
    if merged_segments[0][0]<0:
        merged_segments[0][0]=0
    _vail_segments=[]
    for it in merged_segments:
        if it[1]>it[0]:
            _vail_segments.append(it)
    
    logger.debug(f'[Ten-VAD] Segmentation and merging total time: {int(time.time() - st_)}s')
    return _vail_segments,None


def _detect_raw_segments0(data, threshold, min_silent_frames, max_speech_frames=None):
    """
    Internal helper function: detect speech segments according to silence threshold and max length.
    """
    hop_size = 256

    ten_vad_instance = TenVad(hop_size, threshold)

    # Ensure data is 1D array
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Reduce to mono

    # Calculate valid frame count, ensuring each frame length equals hop_size
    num_frames = (data.shape[0] - hop_size) // hop_size + 1

    segments = []
    triggered = False
    speech_start_frame = 0
    silence_frame_count = 0

    for i in range(num_frames):
        # Ensure each frame length equals hop_size
        audio_frame = data[i * hop_size: (i + 1) * hop_size]

        # Ensure audio frame length is correct
        if len(audio_frame) != hop_size:
            continue

        # Ensure data type is correct
        if audio_frame.dtype != np.int16:
            audio_frame = audio_frame.astype(np.int16)

        _, is_speech = ten_vad_instance.process(audio_frame)

        if triggered:
            current_speech_len = i - speech_start_frame
            if is_speech == 1:
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            # Termination condition: 1. Silence duration met, 2. (Optional) Reached max length hard cut
            is_silence_timeout = silence_frame_count >= min_silent_frames
            is_max_timeout = max_speech_frames is not None and current_speech_len >= max_speech_frames

            if is_silence_timeout or is_max_timeout:
                if is_max_timeout:
                    end_frame = i
                else:
                    end_frame = i - silence_frame_count

                segments.append([speech_start_frame, end_frame])
                triggered = False
                silence_frame_count = 0
        else:
            if is_speech == 1:
                triggered = True
                speech_start_frame = i
                silence_frame_count = 0

    if triggered:
        end_frame = num_frames - silence_frame_count
        segments.append([speech_start_frame, end_frame])

    return segments
