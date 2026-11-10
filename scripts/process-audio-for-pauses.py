#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy>=1.26",
#   "soundfile>=0.13",
#   "lameenc>=1.8.1",
# ]
# ///

from __future__ import annotations

import argparse
from dataclasses import dataclass

import lameenc
import numpy as np
import soundfile as sf


@dataclass
class Config:
    silence_threshold_db: float = -45.0
    min_long_silence_s: float = 2.0
    keep_silence_s: float = 1.5
    window_ms: float = 30.0
    hop_ms: float = 10.0
    min_speech_ms: float = 80.0
    min_silence_ms: float = 120.0
    mp3_bitrate_kbps: int = 192


def db_to_amplitude(db: float) -> float:
    return 10 ** (db / 20.0)


def to_detection_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def frame_rms_signal(
    x: np.ndarray, window: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    starts = np.arange(0, max(1, len(x) - window + 1), hop, dtype=np.int64)
    rms = np.empty(len(starts), dtype=np.float32)

    for i, start in enumerate(starts):
        chunk = x[start : start + window]
        if len(chunk) == 0:
            rms[i] = 0.0
        else:
            rms[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    return rms, starts


def fill_short_runs(mask: np.ndarray, target_value: bool, max_len: int) -> np.ndarray:
    """
    Fill short runs of the opposite value that are surrounded by target_value.
    Useful to smooth tiny false speech islands inside silence, and vice versa.
    """
    out = mask.copy()
    n = len(out)
    i = 0

    while i < n:
        if out[i] == target_value:
            i += 1
            continue

        j = i
        while j < n and out[j] != target_value:
            j += 1

        run_len = j - i
        left_ok = i > 0 and out[i - 1] == target_value
        right_ok = j < n and out[j] == target_value

        if left_ok and right_ok and run_len <= max_len:
            out[i:j] = target_value

        i = j

    return out


def detect_silence_mask(mono: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    window = max(1, int(sr * cfg.window_ms / 1000.0))
    hop = max(1, int(sr * cfg.hop_ms / 1000.0))

    rms, starts = frame_rms_signal(mono, window, hop)
    if len(rms) == 0:
        return np.zeros(len(mono), dtype=bool)

    silence_frames = rms < db_to_amplitude(cfg.silence_threshold_db)

    max_short_speech = max(1, int(cfg.min_speech_ms / cfg.hop_ms))
    max_short_silence = max(1, int(cfg.min_silence_ms / cfg.hop_ms))

    silence_frames = fill_short_runs(
        silence_frames, target_value=True, max_len=max_short_speech
    )
    speech_frames = ~silence_frames
    speech_frames = fill_short_runs(
        speech_frames, target_value=True, max_len=max_short_silence
    )
    silence_frames = ~speech_frames

    sample_mask = np.zeros(len(mono), dtype=bool)
    for is_silence, start in zip(silence_frames, starts):
        end = min(len(sample_mask), start + hop)
        sample_mask[start:end] = is_silence

    if starts[-1] + hop < len(sample_mask):
        sample_mask[starts[-1] + hop :] = silence_frames[-1]

    return sample_mask


def iter_runs(mask: np.ndarray):
    if len(mask) == 0:
        return
    start = 0
    current = bool(mask[0])

    for i in range(1, len(mask)):
        if bool(mask[i]) != current:
            yield current, start, i
            start = i
            current = bool(mask[i])

    yield current, start, len(mask)


def cap_long_silences(
    audio: np.ndarray, silence_mask: np.ndarray, sr: int, cfg: Config
) -> np.ndarray:
    min_long = int(round(cfg.min_long_silence_s * sr))
    keep = int(round(cfg.keep_silence_s * sr))

    parts = []

    for is_silence, start, end in iter_runs(silence_mask):
        seg = audio[start:end]

        if not is_silence:
            parts.append(seg)
            continue

        seg_len = end - start
        if seg_len > min_long:
            parts.append(seg[:keep])
        else:
            parts.append(seg)

    if not parts:
        return audio[:0]

    return np.concatenate(parts, axis=0)


def float_to_pcm16_bytes(audio: np.ndarray) -> tuple[bytes, int]:
    """
    Convert float audio in [-1, 1] to interleaved PCM16 bytes.
    Returns (pcm_bytes, num_channels).
    """
    clipped = np.clip(audio, -1.0, 1.0)

    if clipped.ndim == 1:
        pcm = (clipped * 32767.0).astype(np.int16)
        return pcm.tobytes(), 1

    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.reshape(-1, pcm.shape[1]).tobytes(), pcm.shape[1]


def write_mp3(path: str, audio: np.ndarray, sr: int, bitrate_kbps: int) -> None:
    pcm_bytes, num_channels = float_to_pcm16_bytes(audio)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(num_channels)
    encoder.set_quality(2)

    mp3_data = encoder.encode(pcm_bytes)
    mp3_data += encoder.flush()

    with open(path, "wb") as f:
        f.write(mp3_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shorten silences longer than 2.0s to at most 1.5s, then write MP3."
    )
    parser.add_argument("input", help="Input audio file, ideally WAV")
    parser.add_argument("output", help="Output MP3 file")
    parser.add_argument("--threshold-db", type=float, default=-45.0)
    parser.add_argument("--min-long-silence", type=float, default=2.0)
    parser.add_argument("--keep-silence", type=float, default=1.5)
    parser.add_argument("--window-ms", type=float, default=30.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--bitrate", type=int, default=192, help="MP3 bitrate in kbps")
    args = parser.parse_args()

    cfg = Config(
        silence_threshold_db=args.threshold_db,
        min_long_silence_s=args.min_long_silence,
        keep_silence_s=args.keep_silence,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        mp3_bitrate_kbps=args.bitrate,
    )

    audio, sr = sf.read(args.input, always_2d=False, dtype="float32")
    mono = to_detection_mono(audio)
    silence_mask = detect_silence_mask(mono, sr, cfg)
    out = cap_long_silences(audio, silence_mask, sr, cfg)

    write_mp3(args.output, out, sr, bitrate_kbps=cfg.mp3_bitrate_kbps)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
