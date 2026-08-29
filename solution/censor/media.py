"""ffmpeg / ffprobe wrappers: probing, stream selection, extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUB_CODECS = {
    "subrip",
    "srt",
    "ass",
    "ssa",
    "mov_text",
    "webvtt",
    "text",
    "eia_608",
    "subviewer",
    "microdvd",
}

IMAGE_SUB_CODECS = {
    "hdmv_pgs_subtitle",
    "dvd_subtitle",
    "dvb_subtitle",
    "dvb_teletext",
    "xsub",
}

SIDECAR_SUB_EXTS = (".srt", ".ass", ".ssa", ".vtt", ".sub")


class MediaError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or fails."""


def ensure_tools(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
    missing = [tool for tool in (ffmpeg, ffprobe) if shutil.which(tool) is None]
    if missing:
        raise MediaError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (https://ffmpeg.org/download.html) and retry."
        )


def run(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, raising MediaError with stderr context on failure."""
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-25:]
        raise MediaError(
            f"Command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n"
            + "\n".join(tail)
        )
    return proc


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #


@dataclass
class MediaInfo:
    path: Path
    streams: list[dict] = field(default_factory=list)
    format: dict = field(default_factory=dict)
    chapters: list[dict] = field(default_factory=list)

    def of_type(self, kind: str) -> list[dict]:
        """Streams of a codec_type, in ffmpeg's per-type ordinal order."""
        out = [s for s in self.streams if s.get("codec_type") == kind]
        for ordinal, stream in enumerate(out):
            stream["_ordinal"] = ordinal
        return out

    @property
    def duration(self) -> float:
        for source in (self.format.get("duration"), *(s.get("duration") for s in self.streams)):
            try:
                value = float(source)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 0.0

    def frame_rate(self, video_ordinal: int = 0) -> float:
        videos = self.of_type("video")
        if video_ordinal >= len(videos):
            return 0.0
        raw = videos[video_ordinal].get("avg_frame_rate") or "0/0"
        try:
            num, _, den = raw.partition("/")
            return float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0


def probe(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-show_chapters",
        str(path),
    ]
    proc = run(cmd, capture=True)
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"Could not parse ffprobe output for {path}: {exc}") from exc
    return MediaInfo(
        path=path,
        streams=data.get("streams") or [],
        format=data.get("format") or {},
        chapters=data.get("chapters") or [],
    )


def stream_lang(stream: dict) -> str:
    return str((stream.get("tags") or {}).get("language") or "").lower()


def stream_title(stream: dict) -> str:
    return str((stream.get("tags") or {}).get("title") or "")


def describe_streams(info: MediaInfo) -> str:
    lines = []
    for kind in ("video", "audio", "subtitle"):
        for stream in info.of_type(kind):
            bits = [
                f"{kind[0]}:{stream['_ordinal']}",
                stream.get("codec_name", "?"),
                stream_lang(stream) or "und",
            ]
            if stream.get("channels"):
                bits.append(f"{stream['channels']}ch")
            title = stream_title(stream)
            if title:
                bits.append(f'"{title}"')
            if (stream.get("disposition") or {}).get("forced"):
                bits.append("[forced]")
            if (stream.get("disposition") or {}).get("hearing_impaired"):
                bits.append("[sdh]")
            lines.append("  " + " ".join(bits))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Subtitle selection
# --------------------------------------------------------------------------- #


@dataclass
class SubtitleChoice:
    ordinal: int
    stream: dict
    codec: str
    is_text: bool
    reason: str


def pick_subtitle_stream(
    info: MediaInfo,
    lang: str = "eng",
    explicit: int | None = None,
) -> SubtitleChoice | None:
    """Pick the best text subtitle stream, preferring full (non-forced) tracks.

    `explicit` is an ffmpeg subtitle ordinal (the N in `0:s:N`), not a global
    stream index.
    """
    subs = info.of_type("subtitle")
    if not subs:
        return None

    if explicit is not None:
        if explicit >= len(subs):
            raise MediaError(
                f"--sub-stream {explicit} out of range; file has {len(subs)} subtitle stream(s)"
            )
        stream = subs[explicit]
        codec = str(stream.get("codec_name") or "")
        return SubtitleChoice(
            ordinal=explicit,
            stream=stream,
            codec=codec,
            is_text=codec in TEXT_SUB_CODECS,
            reason="explicitly requested",
        )

    wanted = lang.lower()
    aliases = {"eng", "en", "english"} if wanted in {"eng", "en", "english"} else {wanted}

    def score(stream: dict) -> tuple:
        disp = stream.get("disposition") or {}
        codec = str(stream.get("codec_name") or "")
        return (
            codec in TEXT_SUB_CODECS,
            stream_lang(stream) in aliases,
            not disp.get("forced"),
            not disp.get("hearing_impaired"),
            "sign" not in stream_title(stream).lower(),
            -stream["_ordinal"],
        )

    best = max(subs, key=score)
    codec = str(best.get("codec_name") or "")
    if stream_lang(best) not in aliases and codec in TEXT_SUB_CODECS:
        reason = f"no '{lang}' track; using best available ({stream_lang(best) or 'und'})"
    elif codec in IMAGE_SUB_CODECS:
        reason = f"only image-based subtitles found ({codec}); text extraction impossible"
    elif codec not in TEXT_SUB_CODECS:
        reason = f"unrecognised subtitle codec '{codec}'"
    else:
        reason = f"best '{lang}' text track"

    return SubtitleChoice(
        ordinal=best["_ordinal"],
        stream=best,
        codec=codec,
        is_text=codec in TEXT_SUB_CODECS,
        reason=reason,
    )


def find_sidecar_subtitle(video: Path, lang: str = "eng") -> Path | None:
    """Look for a sidecar subtitle file beside the video, preferring English."""
    stem = video.stem.lower()
    candidates: list[Path] = []
    for entry in video.parent.glob(f"{glob_escape(video.stem)}*"):
        if entry.suffix.lower() in SIDECAR_SUB_EXTS and entry.stem.lower().startswith(stem):
            candidates.append(entry)
    if not candidates:
        return None

    aliases = ("en", "eng", "english")

    def score(path: Path) -> tuple:
        tail = path.stem[len(video.stem) :].lower()
        return (
            any(alias in tail for alias in aliases),
            "forced" not in tail,
            "sdh" not in tail,
            -len(tail),
        )

    return max(candidates, key=score)


def glob_escape(text: str) -> str:
    for char in "[]?*":
        text = text.replace(char, f"[{char}]")
    return text


def extract_subtitles(
    info: MediaInfo,
    ordinal: int,
    dest: Path,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Extract an embedded text subtitle stream to `dest` (SRT or ASS by suffix)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(info.path),
        "-map",
        f"0:s:{ordinal}",
        "-c:s",
        "ass" if dest.suffix.lower() in {".ass", ".ssa"} else "srt",
        str(dest),
    ]
    run(cmd, capture=True)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise MediaError(f"Subtitle extraction produced no data for stream 0:s:{ordinal}")
    return dest


# --------------------------------------------------------------------------- #
# Audio extraction
# --------------------------------------------------------------------------- #


def extract_wav(
    info: MediaInfo,
    audio_ordinal: int,
    dest: Path,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = 16000,
) -> Path:
    """Extract one audio stream to 16 kHz mono PCM WAV for ASR.

    Always transcribing the exact stream that will be muted removes a whole
    class of "the timings were right but the wrong track was silenced" bugs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(info.path),
        "-map",
        f"0:a:{audio_ordinal}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    run(cmd, capture=True)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise MediaError(f"Audio extraction produced no data for stream 0:a:{audio_ordinal}")
    return dest


def has_encoder(name: str, ffmpeg: str = "ffmpeg") -> bool:
    try:
        proc = run([ffmpeg, "-hide_banner", "-encoders"], capture=True, check=False)
    except MediaError:
        return False
    return f" {name} " in (proc.stdout or "")
