"""Subtitle parsing, cleaning, censoring and writing."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from .matcher import Matcher, tokenize

# ASS/SSA override blocks, HTML-ish tags, and music glyphs.
_ASS_TAG_RE = re.compile(r"\{[^{}]*\}")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MUSIC_RE = re.compile(r"[\u266a\u266b\u266c\u2669\u25ba]")
# Non-speech annotations: [DOOR SLAMS], (SIGHS)
_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
# Speaker labels at the start of a line: "MAN:", "- JACK REACHER:"
_SPEAKER_RE = re.compile(r"^\s*[-\u2013]?\s*[A-Z][A-Z0-9 .'\u2019#-]{1,24}:\s*", re.MULTILINE)
_LEADING_DASH_RE = re.compile(r"^\s*[-\u2013]\s*", re.MULTILINE)
_WS_RE = re.compile(r"\s+")

_SRT_TIME_RE = re.compile(
    r"(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2})[,.](?P<ms>\d{1,3})"
)


class SubtitleError(RuntimeError):
    """Raised when a subtitle file cannot be read."""


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def sec_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms_total = int(round(seconds * 1000.0))
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, ms = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def srt_time_to_sec(text: str) -> float:
    match = _SRT_TIME_RE.search(text)
    if not match:
        raise SubtitleError(f"Not an SRT timestamp: {text!r}")
    ms = match.group("ms").ljust(3, "0")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(ms) / 1000.0
    )


# --------------------------------------------------------------------------- #
# Cue model
# --------------------------------------------------------------------------- #


@dataclass
class Cue:
    """One subtitle event on the *source* timeline (seconds)."""

    index: int
    start: float
    end: float
    text: str  # display text, tags preserved

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def clean(self) -> str:
        """Spoken words only: no tags, no music glyphs, no [SDH], no speaker labels."""
        return clean_cue_text(self.text)

    def shifted(self, offset: float) -> "Cue":
        return replace(self, start=self.start + offset, end=self.end + offset)


def clean_cue_text(text: str) -> str:
    out = _ASS_TAG_RE.sub(" ", text)
    out = _HTML_TAG_RE.sub(" ", out)
    out = _MUSIC_RE.sub(" ", out)
    out = _BRACKET_RE.sub(" ", out)
    out = _SPEAKER_RE.sub(" ", out)
    out = _LEADING_DASH_RE.sub(" ", out)
    out = out.replace("\\N", " ").replace("\\n", " ")
    return _WS_RE.sub(" ", out).strip()


def display_text(text: str) -> str:
    """Text with formatting removed but non-speech annotations kept."""
    out = _ASS_TAG_RE.sub("", text)
    out = out.replace("\\N", "\n").replace("\\n", "\n")
    return out.strip()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_subtitle_file(path: Path) -> list[Cue]:
    """Parse SRT/ASS/SSA/VTT into cues. Uses pysubs2 when available."""
    if not path.is_file():
        raise SubtitleError(f"Subtitle file not found: {path}")
    try:
        import pysubs2  # noqa: PLC0415
    except ImportError:
        return _parse_srt_text(_read_text(path))

    try:
        subs = pysubs2.load(str(path), encoding="utf-8")
    except UnicodeDecodeError:
        subs = pysubs2.load(str(path), encoding="latin-1")
    except Exception as exc:  # pysubs2 raises a variety of format errors
        if path.suffix.lower() == ".srt":
            return _parse_srt_text(_read_text(path))
        raise SubtitleError(f"Could not parse {path}: {exc}") from exc

    cues: list[Cue] = []
    for event in subs:
        if getattr(event, "is_comment", False) or getattr(event, "is_drawing", False):
            continue
        text = event.text or ""
        if not text.strip():
            continue
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=event.start / 1000.0,
                end=event.end / 1000.0,
                text=text,
            )
        )
    cues.sort(key=lambda c: (c.start, c.end))
    for position, cue in enumerate(cues, start=1):
        cue.index = position
    return cues


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise SubtitleError(f"Could not decode {path} with any known encoding")


def _parse_srt_text(raw: str) -> list[Cue]:
    """Tolerant SRT parser used when pysubs2 is unavailable."""
    cues: list[Cue] = []
    blocks = re.split(r"\r?\n\s*\r?\n", raw.replace("\ufeff", "").strip())
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip() != ""]
        if not lines:
            continue
        time_line_at = next(
            (i for i, line in enumerate(lines) if "-->" in line),
            None,
        )
        if time_line_at is None:
            continue
        left, _, right = lines[time_line_at].partition("-->")
        try:
            start = srt_time_to_sec(left)
            end = srt_time_to_sec(right)
        except SubtitleError:
            continue
        text = "\n".join(lines[time_line_at + 1 :]).strip()
        if not text:
            continue
        cues.append(Cue(index=len(cues) + 1, start=start, end=end, text=text))
    cues.sort(key=lambda c: (c.start, c.end))
    for position, cue in enumerate(cues, start=1):
        cue.index = position
    return cues


# --------------------------------------------------------------------------- #
# Censoring
# --------------------------------------------------------------------------- #


def censor_text(text: str, matcher: Matcher, mask: str = "___") -> tuple[str, int]:
    """Replace every target in `text` with `mask`. Returns (text, hit count)."""
    tokens = tokenize(text)
    matches = matcher.find_in_tokens(tokens)
    if not matches:
        return text, 0
    out: list[str] = []
    cursor = 0
    for match in matches:
        out.append(text[cursor : match.start_char])
        out.append(_mask_for(match.text, mask))
        cursor = match.end_char
    out.append(text[cursor:])
    return "".join(out), len(matches)


def _mask_for(original: str, mask: str) -> str:
    """Keep a trailing possessive/plural so '___'s reads naturally."""
    if mask == "___" and original.lower().endswith("'s"):
        return "___'s"
    return mask


def censor_cues(cues: list[Cue], matcher: Matcher, mask: str = "___") -> tuple[list[Cue], int]:
    out: list[Cue] = []
    total = 0
    for cue in cues:
        text, hits = censor_text(cue.text, matcher, mask)
        total += hits
        out.append(replace(cue, text=text))
    return out, total


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_srt(cues: list[Cue], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for position, cue in enumerate(cues, start=1):
        end = max(cue.end, cue.start + 0.001)
        chunks.append(
            f"{position}\n"
            f"{sec_to_srt_time(cue.start)} --> {sec_to_srt_time(end)}\n"
            f"{display_text(cue.text)}\n"
        )
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path
