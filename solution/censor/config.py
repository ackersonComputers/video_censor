"""Configuration: run settings and profanity word list loading."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

DEFAULT_WORDLIST = Path(__file__).with_name("wordlists") / "default.yaml"


class ConfigError(RuntimeError):
    """Raised when a config or word list file is malformed."""


# --------------------------------------------------------------------------- #
# Word list
# --------------------------------------------------------------------------- #


@dataclass
class WordList:
    """A resolved, tier-filtered set of match rules."""

    exact: dict[str, str] = field(default_factory=dict)
    substring: dict[str, str] = field(default_factory=dict)
    phrases: dict[str, str] = field(default_factory=dict)
    exclusions: set[str] = field(default_factory=set)
    tiers: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.exact) + len(self.substring) + len(self.phrases)

    def summary(self) -> str:
        return (
            f"{len(self.exact)} exact, {len(self.substring)} substring, "
            f"{len(self.phrases)} phrase rules across tiers "
            f"[{', '.join(self.tiers)}]"
        )


def _normalize_phrase_key(phrase: str) -> str:
    """Collapse a phrase to space-separated normalized tokens."""
    from .matcher import norm_token  # local import: avoids a config<->matcher cycle

    parts = [norm_token(p) for p in phrase.replace("-", " ").split()]
    return " ".join(p for p in parts if p)


def load_wordlist(
    path: Path | str | None = None,
    tiers: list[str] | None = None,
) -> WordList:
    """Load a word list YAML, keeping only the requested tiers.

    `tiers` of None means "use the file's default_tiers"; a list containing
    "all" means every tier defined in the file.
    """
    from .matcher import norm_token

    src = Path(path) if path else DEFAULT_WORDLIST
    if not src.is_file():
        raise ConfigError(f"Word list not found: {src}")

    try:
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse word list {src}: {exc}") from exc

    all_tiers = data.get("tiers") or {}
    if not isinstance(all_tiers, dict):
        raise ConfigError(f"{src}: 'tiers' must be a mapping of tier name -> rules")

    if tiers is None:
        wanted = list(data.get("default_tiers") or all_tiers.keys())
    elif any(t.lower() == "all" for t in tiers):
        wanted = list(all_tiers.keys())
    else:
        wanted = list(tiers)

    unknown = [t for t in wanted if t not in all_tiers]
    if unknown:
        raise ConfigError(
            f"{src}: unknown tier(s) {unknown}. Available: {sorted(all_tiers)}"
        )

    wl = WordList(tiers=wanted)
    for raw in data.get("exclusions") or []:
        token = norm_token(str(raw))
        if token:
            wl.exclusions.add(token)

    for tier in wanted:
        rules = all_tiers[tier] or {}
        for raw in rules.get("exact") or []:
            token = norm_token(str(raw))
            if token:
                wl.exact.setdefault(token, tier)
        for raw in rules.get("substring") or []:
            token = norm_token(str(raw))
            if token:
                wl.substring.setdefault(token, tier)
        for raw in rules.get("phrases") or []:
            key = _normalize_phrase_key(str(raw))
            if key and " " in key:
                wl.phrases.setdefault(key, tier)
            elif key:
                # A single-word "phrase" is just an exact rule.
                wl.exact.setdefault(key, tier)

    if not len(wl):
        raise ConfigError(f"{src}: selected tiers produced no rules")
    return wl


# --------------------------------------------------------------------------- #
# Run settings
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    """Everything that controls a single censor run."""

    # --- input / output -------------------------------------------------- #
    input: Path = Path()
    output: Path | None = None
    report: Path | None = None
    sidecar_srt: bool = True
    dry_run: bool = False
    report_only: bool = False
    overwrite: bool = False
    assume_yes: bool = False

    # --- word list ------------------------------------------------------- #
    wordlist: Path | None = None
    tiers: list[str] | None = None
    extra_words: list[str] = field(default_factory=list)
    allow_words: list[str] = field(default_factory=list)

    # --- stream selection ------------------------------------------------ #
    audio_index: int = 0
    video_index: int = 0
    sub_lang: str = "eng"
    sub_stream: int | None = None
    external_subs: Path | None = None
    mute_all_audio: bool = True

    # --- ASR ------------------------------------------------------------- #
    model: str = "large-v3"
    language: str = "en"
    device: str = "auto"
    engine: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    batch_size: int = 8
    asr_cache: Path | None = None
    no_asr_cache: bool = False
    allow_asr_fallback: bool = False
    skip_asr: bool = False
    no_detect: bool = False

    # --- fusion ---------------------------------------------------------- #
    sub_offset: float | None = None
    no_auto_offset: bool = False
    window_slack: float = 2.5
    fuzzy_threshold: float = 0.80
    pad_exact: tuple[float, float] = (0.06, 0.12)
    pad_fuzzy: tuple[float, float] = (0.10, 0.20)
    pad_anchor: tuple[float, float] = (0.25, 0.30)
    blind_pad: float = 0.45
    paranoid: bool = False
    extras_policy: str = "mute"  # mute | report | ignore
    extras_min_score: float = 0.50
    merge_gap: float = 0.08
    min_duration: float = 0.10
    max_interval: float = 6.0

    # --- subtitles out --------------------------------------------------- #
    mask_style: str = "___"
    keep_original_subs: bool = False
    no_censored_subs: bool = False

    # --- scene edit list ------------------------------------------------- #
    edl: Path | None = None
    edl_out: Path | None = None
    edl_preview: bool = False
    cut_mode: str = "reencode"  # reencode | copy
    blackout_fade: float = 0.0
    blur_strength: float = 30.0
    blur_mode: str = "gblur"  # gblur | pixelate

    # --- encoding -------------------------------------------------------- #
    video_encoder: str = "hevc_nvenc"
    video_quality: int = 19
    nvenc_preset: str = "p7"
    audio_codec: str = "flac"
    audio_bitrate: str = "448k"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    # --- housekeeping ---------------------------------------------------- #
    work_dir: Path | None = None
    keep_temp: bool = False

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def default_output(self) -> Path:
        return self.input.with_name(f"{self.input.stem}_censored.mkv")

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, set):
                out[key] = sorted(value)
        return out


def confirm(prompt: str, *, assume_yes: bool) -> bool:
    """Interactive yes/no prompt. Non-interactive stdin defaults to 'no'."""
    if assume_yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}
