"""Speech recognition with forced word alignment.

Engine preference (quality first):
  1. WhisperX  - large-v3 + wav2vec2 forced alignment. By far the best word
                 timings, and its VAD segmentation stops the long hallucination
                 runs that plague plain Whisper on music-heavy material.
  2. faster-whisper - word timestamps from the model's cross-attention.
  3. openai-whisper - last resort, matches the original scripts' behaviour.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .matcher import norm_token

CACHE_VERSION = 2
_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORIES_ADDED: set[str] = set()


class ASRError(RuntimeError):
    """Raised when no usable ASR backend is available."""


@dataclass
class Word:
    """One recognized word on the source timeline."""

    text: str
    start: float
    end: float
    score: float = 1.0

    @property
    def norm(self) -> str:
        return norm_token(self.text)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class Transcript:
    words: list[Word]
    engine: str
    model: str
    language: str
    duration: float = 0.0

    def __len__(self) -> int:
        return len(self.words)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def _cache_key(audio: Path, model: str, language: str) -> dict:
    stat = audio.stat()
    return {
        "version": CACHE_VERSION,
        "size": stat.st_size,
        "model": model,
        "language": language,
    }


def load_cache(path: Path, audio: Path, model: str, language: str) -> Transcript | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("key") != _cache_key(audio, model, language):
        return None
    words = [Word(**w) for w in data.get("words", [])]
    if not words:
        return None
    return Transcript(
        words=words,
        engine=data.get("engine", "cache"),
        model=data.get("model", model),
        language=data.get("language", language),
        duration=float(data.get("duration") or 0.0),
    )


def save_cache(path: Path, audio: Path, transcript: Transcript) -> None:
    payload = {
        "key": _cache_key(audio, transcript.model, transcript.language),
        "engine": transcript.engine,
        "model": transcript.model,
        "language": transcript.language,
        "duration": transcript.duration,
        "words": [asdict(w) for w in transcript.words],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #


def transcribe(
    audio: Path,
    *,
    model: str = "large-v3",
    language: str = "en",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    batch_size: int = 8,
    engine: str = "auto",
    log=print,
) -> Transcript:
    """Transcribe `audio` and return word-level timings."""
    started = time.monotonic()
    order = (
        ["whisperx", "faster-whisper", "openai-whisper"]
        if engine == "auto"
        else [engine]
    )
    errors: list[str] = []
    for name in order:
        runner = _ENGINES[name]
        try:
            transcript = runner(
                audio,
                model=model,
                language=language,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                batch_size=batch_size,
                log=log,
            )
        except ImportError as exc:
            errors.append(f"{name}: not installed ({exc})")
            continue
        transcript.duration = time.monotonic() - started
        log(f"[asr] {name} produced {len(transcript.words)} words in {transcript.duration:.1f}s")
        return transcript
    raise ASRError(
        "No usable ASR backend. Tried:\n  "
        + "\n  ".join(errors)
        + "\n\nInstall the recommended stack with:\n"
        "  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
        "--index-url https://download.pytorch.org/whl/cu128\n"
        "  pip install whisperx"
    )


def _run_whisperx(audio: Path, *, model, language, device, compute_type, beam_size, batch_size, log):
    _enable_windows_ffmpeg_dlls()
    import whisperx  # noqa: PLC0415

    if device == "cpu" and compute_type == "float16":
        compute_type = "float32"

    log(f"[asr] loading whisperx {model} on {device} ({compute_type})")
    asr_model = whisperx.load_model(
        model,
        device,
        compute_type=compute_type,
        language=language,
        asr_options={"beam_size": beam_size},
    )
    samples = whisperx.load_audio(str(audio))
    result = asr_model.transcribe(samples, batch_size=batch_size, language=language)

    log("[asr] loading alignment model")
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", language), device=device
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        samples,
        device,
        return_char_alignments=False,
    )

    words: list[Word] = []
    for segment in aligned.get("segments", []):
        for entry in segment.get("words", []) or []:
            word = _word_from_dict(entry, segment)
            if word is not None:
                words.append(word)
    _fill_missing_times(words)
    return Transcript(words=words, engine="whisperx", model=model, language=language)


def _ffmpeg_dll_directories(path_value: str) -> list[Path]:
    """Return PATH entries containing shared FFmpeg libraries."""
    found: list[Path] = []
    for raw_entry in path_value.split(os.pathsep):
        raw_entry = raw_entry.strip().strip('"')
        if not raw_entry:
            continue
        directory = Path(raw_entry)
        try:
            if any(directory.glob("avcodec-*.dll")):
                found.append(directory)
        except OSError:
            continue
    return found


def _winget_ffmpeg_dll_directories(local_app_data: str | None) -> list[Path]:
    """Find Gyan's full-shared FFmpeg even before a terminal PATH refresh."""
    if not local_app_data:
        return []
    packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    found: list[Path] = []
    try:
        package_dirs = packages.glob("Gyan.FFmpeg.Shared_*")
        for package_dir in package_dirs:
            for dll in package_dir.glob(
                "ffmpeg-*-full_build-shared/bin/avcodec-*.dll"
            ):
                if dll.parent not in found:
                    found.append(dll.parent)
    except OSError:
        return []
    return found


def _enable_windows_ffmpeg_dlls() -> None:
    """Make Windows shared FFmpeg builds visible to TorchCodec.

    Python 3.8+ uses a restricted DLL search path on Windows, so placing the
    FFmpeg shared-build directory on PATH is not always enough. Keep the
    handles alive for the process lifetime so WhisperX/TorchCodec can load.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    directories = _ffmpeg_dll_directories(os.environ.get("PATH", ""))
    directories.extend(
        directory
        for directory in _winget_ffmpeg_dll_directories(
            os.environ.get("LOCALAPPDATA")
        )
        if directory not in directories
    )
    for directory in directories:
        key = str(directory).casefold()
        if key in _DLL_DIRECTORIES_ADDED:
            continue
        try:
            handle = os.add_dll_directory(str(directory))
        except OSError:
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        _DLL_DIRECTORIES_ADDED.add(key)


def _run_faster_whisper(
    audio: Path, *, model, language, device, compute_type, beam_size, batch_size, log
):
    from faster_whisper import WhisperModel  # noqa: PLC0415

    if device == "cpu" and compute_type == "float16":
        compute_type = "float32"
    log(f"[asr] loading faster-whisper {model} on {device} ({compute_type})")
    engine = WhisperModel(model, device=device, compute_type=compute_type)
    segments, _ = engine.transcribe(
        str(audio),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for segment in segments:
        for entry in segment.words or []:
            words.append(
                Word(
                    text=str(entry.word).strip(),
                    start=float(entry.start),
                    end=float(entry.end),
                    score=float(getattr(entry, "probability", 1.0) or 1.0),
                )
            )
    return Transcript(words=words, engine="faster-whisper", model=model, language=language)


def _run_openai_whisper(
    audio: Path, *, model, language, device, compute_type, beam_size, batch_size, log
):
    import whisper  # noqa: PLC0415

    log(f"[asr] loading openai-whisper {model} on {device}")
    engine = whisper.load_model(model, device=None if device == "auto" else device)
    result = engine.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        condition_on_previous_text=False,
        verbose=False,
    )
    words: list[Word] = []
    for segment in result.get("segments") or []:
        for entry in segment.get("words") or []:
            word = _word_from_dict(entry, segment)
            if word is not None:
                words.append(word)
    _fill_missing_times(words)
    return Transcript(words=words, engine="openai-whisper", model=model, language=language)


_ENGINES = {
    "whisperx": _run_whisperx,
    "faster-whisper": _run_faster_whisper,
    "openai-whisper": _run_openai_whisper,
}


def _word_from_dict(entry: dict, segment: dict) -> Word | None:
    text = str(entry.get("word") or entry.get("text") or "").strip()
    if not text:
        return None
    start = entry.get("start")
    end = entry.get("end")
    return Word(
        text=text,
        start=float(start) if start is not None else float("nan"),
        end=float(end) if end is not None else float("nan"),
        score=float(entry.get("score") or entry.get("probability") or 1.0),
    )


def _fill_missing_times(words: list[Word]) -> None:
    """WhisperX leaves numerals and symbols unaligned; interpolate their times.

    An unaligned word with NaN times would otherwise be silently unusable, and
    those gaps are exactly where a missed swear hides.
    """
    import math

    n = len(words)
    for i, word in enumerate(words):
        if not (math.isnan(word.start) or math.isnan(word.end)):
            continue
        prev_end = next(
            (words[j].end for j in range(i - 1, -1, -1) if not math.isnan(words[j].end)),
            None,
        )
        next_start = next(
            (words[j].start for j in range(i + 1, n) if not math.isnan(words[j].start)),
            None,
        )
        if prev_end is None and next_start is None:
            word.start, word.end = 0.0, 0.0
        elif prev_end is None:
            word.start, word.end = max(0.0, next_start - 0.3), next_start
        elif next_start is None:
            word.start, word.end = prev_end, prev_end + 0.3
        else:
            word.start, word.end = prev_end, max(prev_end, next_start)
        word.score = min(word.score, 0.3)
