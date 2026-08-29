"""ffmpeg filtergraph construction and muxing.

Two paths exist:

* **Mute only** - the video stream is copied bit for bit and only audio is
  touched. Fast, and lossless for the picture.
* **Scene edits present** - any ``cut``/``blackout``/``blur`` forces a video
  re-encode, done in a single ffmpeg invocation using ``trim``/``concat`` so the
  seams cannot drift out of sync the way segment files plus the concat demuxer
  can.

Mute envelopes are sample-accurate: each interval is fully silent with a short
linear fade at both edges, avoiding clicks caused by instantaneous waveform
cuts. Intervals are arranged as balanced time decision trees and split into
bounded chunks so long files remain fast and stay below parser limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import edl as edl_mod
from .align import Interval
from .config import Settings
from .edl import Edit
from .media import MediaInfo, run
from .timeline import TimeMap

# Intervals per sample-accurate gain filter. Keeping each expression bounded
# avoids command-line/parser limits on films with hundreds of hits.
ENABLE_CHUNK = 60
MUTE_FADE_SECONDS = 0.010


@dataclass
class RenderPlan:
    command: list[str]
    reencodes_video: bool
    audio_targets: list[int]
    segments: int
    intervals: int

    def describe(self) -> str:
        mode = "re-encode" if self.reencodes_video else "stream copy"
        return (
            f"video: {mode}; audio tracks muted: {len(self.audio_targets)}; "
            f"segments: {self.segments}; mute intervals: {self.intervals}"
        )


def _fmt(value: float) -> str:
    return f"{max(0.0, value):.3f}"


def between(intervals: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{_fmt(s)},{_fmt(e)})" for s, e in intervals)


def _merge_spans(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _interval_gain(start: float, end: float, fade: float) -> str:
    """Gain envelope: unity, fade to zero, silence, then fade to unity."""
    return (
        f"min(1,max(0,max(({_fmt(start)}-t)/{_fmt(fade)},"
        f"(t-{_fmt(end)})/{_fmt(fade)})))"
    )


def _gain_tree(intervals: list[tuple[float, float]], fade: float) -> str:
    """Balanced time decision tree so per-sample work grows logarithmically."""
    if len(intervals) == 1:
        return _interval_gain(*intervals[0], fade)
    middle = len(intervals) // 2
    threshold = (intervals[middle - 1][1] + intervals[middle][0]) / 2.0
    left = _gain_tree(intervals[:middle], fade)
    right = _gain_tree(intervals[middle:], fade)
    return f"if(lt(t,{_fmt(threshold)}),{left},{right})"


def volume_chain(
    intervals: list[tuple[float, float]],
    chunk: int = ENABLE_CHUNK,
    fade: float = MUTE_FADE_SECONDS,
    channel_layout: str | None = None,
    channels: int = 1,
) -> list[str]:
    """Build sample-accurate mute envelopes with click-free edge fades."""
    intervals = _merge_spans(intervals)
    filters: list[str] = []
    for start in range(0, len(intervals), chunk):
        group = intervals[start : start + chunk]
        gain = _gain_tree(group, fade)
        layout = channel_layout or f"{channels}c"
        filters.append(
            f"aeval=exprs='val(ch)*{gain}':channel_layout={layout}"
        )
    return filters


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


# --------------------------------------------------------------------------- #
# Video effect filters (applied on the SOURCE timeline, before any trim)
# --------------------------------------------------------------------------- #


def blackout_filters(edits: list[Edit]) -> list[str]:
    filters: list[str] = []
    for edit in edits:
        fade = edit.param_float("fade", 0.0)
        if fade <= 0.01:
            filters.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@1.0:t=fill:"
                f"enable='between(t,{_fmt(edit.start)},{_fmt(edit.end)})'"
            )
            continue
        # drawbox cannot ramp its own alpha, so approximate the fade with a few
        # stacked constant-alpha boxes at the head and tail.
        steps = 4
        step = fade / steps
        for index in range(1, steps + 1):
            alpha = index / steps
            filters.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@{alpha:.2f}:t=fill:"
                f"enable='between(t,{_fmt(edit.start + (index - 1) * step)},"
                f"{_fmt(edit.end - (index - 1) * step)})'"
            )
    return filters


def blur_filters(edits: list[Edit], settings: Settings) -> list[str]:
    filters: list[str] = []
    for edit in edits:
        strength = edit.param_float("strength", settings.blur_strength)
        mode = edit.params.get("mode", settings.blur_mode).lower()
        window = f"enable='between(t,{_fmt(edit.start)},{_fmt(edit.end)})'"
        if mode == "pixelate":
            block = max(4, int(strength))
            filters.append(f"pixelize=w={block}:h={block}:{window}")
        else:
            filters.append(f"gblur=sigma={strength:g}:steps=3:{window}")
    return filters


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #


def build_plan(
    *,
    settings: Settings,
    info: MediaInfo,
    intervals: list[Interval],
    edits: list[Edit],
    timemap: TimeMap,
    output: Path,
    censored_srt: Path | None = None,
    chapters_file: Path | None = None,
) -> RenderPlan:
    audio_streams = info.of_type("audio")
    if not audio_streams:
        raise ValueError(f"{info.path} has no audio streams")
    if settings.audio_index >= len(audio_streams):
        raise ValueError(
            f"--audio-index {settings.audio_index} out of range; "
            f"file has {len(audio_streams)} audio stream(s)"
        )

    targets = (
        list(range(len(audio_streams)))
        if settings.mute_all_audio
        else [settings.audio_index]
    )

    manual_mutes = [(e.start, e.end) for e in edl_mod.of_action(edits, "mute")]
    source_spans = sorted(
        [(i.start, i.end) for i in intervals] + manual_mutes
    )

    reencode = edl_mod.requires_video_encode(edits)
    segments = timemap.segments() if timemap else []
    if not segments:
        segments = [(0.0, info.duration)]

    inputs: list[str] = ["-i", str(info.path)]
    input_index = 1
    srt_input = None
    if censored_srt is not None:
        srt_input = input_index
        inputs += ["-i", str(censored_srt)]
        input_index += 1
    meta_input = None
    if chapters_file is not None:
        meta_input = input_index
        inputs += ["-f", "ffmetadata", "-i", str(chapters_file)]
        input_index += 1

    filters: list[str] = []
    maps: list[str] = []

    # --- video ----------------------------------------------------------- #
    if reencode:
        chain = blur_filters(edl_mod.of_action(edits, "blur"), settings)
        chain += blackout_filters(edl_mod.of_action(edits, "blackout"))
        head = "[0:v:%d]" % settings.video_index
        if chain:
            filters.append(f"{head}{','.join(chain)}[vfx]")
            head = "[vfx]"
        if len(segments) > 1:
            filters.append(f"{head}split={len(segments)}" + "".join(f"[vs{i}]" for i in range(len(segments))))
            for i, (start, end) in enumerate(segments):
                filters.append(
                    f"[vs{i}]trim=start={_fmt(start)}:end={_fmt(end)},setpts=PTS-STARTPTS[vt{i}]"
                )
            filters.append(
                "".join(f"[vt{i}]" for i in range(len(segments)))
                + f"concat=n={len(segments)}:v=1:a=0[vout]"
            )
        elif head == "[vfx]":
            filters.append("[vfx]null[vout]")
        else:
            filters.append(f"{head}null[vout]")
        maps += ["-map", "[vout]"]
    else:
        maps += ["-map", f"0:v:{settings.video_index}"]

    # --- audio ------------------------------------------------------------ #
    audio_filtered: list[bool] = []
    for ordinal in range(len(audio_streams)):
        label = f"a{ordinal}"
        mutes = source_spans if ordinal in targets else []
        needs_filter = bool(mutes) or len(segments) > 1
        audio_filtered.append(needs_filter)
        if not needs_filter:
            maps += ["-map", f"0:a:{ordinal}"]
            continue

        head = f"[0:a:{ordinal}]"
        stream = audio_streams[ordinal]
        chain = (
            volume_chain(
                mutes,
                channel_layout=stream.get("channel_layout"),
                channels=int(stream.get("channels", 1)),
            )
            if mutes
            else []
        )
        if chain:
            filters.append(f"{head}{','.join(chain)}[{label}fx]")
            head = f"[{label}fx]"
        if len(segments) > 1:
            filters.append(
                f"{head}asplit={len(segments)}"
                + "".join(f"[{label}s{i}]" for i in range(len(segments)))
            )
            for i, (start, end) in enumerate(segments):
                filters.append(
                    f"[{label}s{i}]atrim=start={_fmt(start)}:end={_fmt(end)},"
                    f"asetpts=PTS-STARTPTS[{label}t{i}]"
                )
            filters.append(
                "".join(f"[{label}t{i}]" for i in range(len(segments)))
                + f"concat=n={len(segments)}:v=0:a=1[{label}out]"
            )
        else:
            filters.append(f"{head}anull[{label}out]")
        maps += ["-map", f"[{label}out]"]

    # --- subtitles / attachments ------------------------------------------ #
    if srt_input is not None:
        maps += ["-map", f"{srt_input}:0"]
    if settings.keep_original_subs and not timemap:
        maps += ["-map", "0:s?"]
    maps += ["-map", "0:t?"]
    if meta_input is not None:
        maps += ["-map_metadata", "0", "-map_chapters", str(meta_input)]
    else:
        maps += ["-map_metadata", "0", "-map_chapters", "0"]

    # --- codecs ------------------------------------------------------------ #
    codecs: list[str] = []
    if reencode:
        codecs += _video_encode_args(settings, info)
    else:
        codecs += ["-c:v", "copy"]

    filtered_audio = any(m.startswith("[a") for m in maps)
    if filtered_audio:
        codecs += _audio_encode_args(settings, audio_filtered)
    else:
        codecs += ["-c:a", "copy"]
    codecs += ["-c:s", "copy"]

    dispositions: list[str] = []
    if srt_input is not None:
        dispositions += ["-disposition:s:0", "default"]
        if settings.keep_original_subs and not timemap:
            dispositions += ["-disposition:s:1", "0"]

    command = [
        settings.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        "-y" if settings.overwrite else "-n",
        *inputs,
    ]
    if filters:
        command += ["-filter_complex", ";".join(filters)]
    command += maps + codecs + dispositions
    command += ["-max_interleave_delta", "0", str(output)]

    return RenderPlan(
        command=command,
        reencodes_video=reencode,
        audio_targets=targets,
        segments=len(segments),
        intervals=len(source_spans),
    )


def _video_encode_args(settings: Settings, info: MediaInfo) -> list[str]:
    encoder = settings.video_encoder
    args = ["-c:v", encoder]
    if encoder.endswith("_nvenc"):
        args += [
            "-preset",
            settings.nvenc_preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(settings.video_quality),
            "-b:v",
            "0",
            "-spatial_aq",
            "1",
            "-temporal_aq",
            "1",
            "-rc-lookahead",
            "32",
        ]
        if encoder.startswith("hevc"):
            args += ["-tag:v", "hvc1"]
    elif encoder in {"libx264", "libx265"}:
        args += ["-preset", "slow", "-crf", str(settings.video_quality)]
    elif encoder == "copy":
        return ["-c:v", "copy"]

    videos = info.of_type("video")
    pix_fmt = videos[settings.video_index].get("pix_fmt") if videos else None
    if pix_fmt in {"yuv420p10le", "p010le"}:
        args += ["-pix_fmt", "p010le" if encoder.endswith("_nvenc") else "yuv420p10le"]
    elif pix_fmt:
        args += ["-pix_fmt", "yuv420p" if encoder.endswith("_nvenc") else pix_fmt]
    return args


def _audio_encode_args(settings: Settings, filtered: list[bool]) -> list[str]:
    """Encode only the tracks that were actually filtered; copy the rest."""
    codec = settings.audio_codec
    args: list[str] = []
    for ordinal, was_filtered in enumerate(filtered):
        if not was_filtered or codec == "copy":
            args += [f"-c:a:{ordinal}", "copy"]
            continue
        args += [f"-c:a:{ordinal}", codec]
        if codec in {"aac", "ac3", "eac3", "libopus"}:
            args += [f"-b:a:{ordinal}", settings.audio_bitrate]
        elif codec == "flac":
            args += [f"-compression_level:a:{ordinal}", "5"]
    return args


def render(plan: RenderPlan, *, dry_run: bool = False, log=print) -> None:
    if dry_run:
        log("[render] would run:\n  " + _pretty(plan.command))
        return
    log(f"[render] {plan.describe()}")
    run(plan.command)


def _pretty(command: list[str]) -> str:
    parts = []
    for token in command:
        parts.append(f'"{token}"' if " " in token or ";" in token else token)
    return " ".join(parts)
